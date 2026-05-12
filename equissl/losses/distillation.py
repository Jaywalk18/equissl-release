"""
Self-supervised losses for EquiSSL.

The current SSL rescue recipe combines DINO/iBOT-style self-distillation,
global contrastive alignment across rotated views, SO(3) rotation prediction,
and KoLeo anti-collapse regularization. This keeps the task fully
self-supervised while putting stronger pressure on downstream-useful global
features than masked reconstruction alone.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class KoLeoLoss(nn.Module):
    """KoLeo regularizer (DINOv2 §3.4): -log(d_NN) on L2-normalised features.
    Pushes features apart in unit-sphere — anti-collapse mechanism."""

    def __init__(self, eps: float = 1e-4, max_pts: int = 4096):
        super().__init__()
        self.eps = eps
        self.max_pts = max_pts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = F.normalize(x, dim=-1, eps=self.eps)
        if x.shape[0] > self.max_pts:
            idx = torch.randperm(x.shape[0], device=x.device)[: self.max_pts]
            x = x[idx]
        with torch.no_grad():
            sim = x @ x.T
            sim.fill_diagonal_(-2.0)
            nn_idx = sim.argmax(dim=1)
        dists = (x - x[nn_idx]).norm(dim=-1)
        return -torch.log(dists + self.eps).mean()


class CLSDistillationLoss(nn.Module):
    """Cross-entropy between teacher and student CLS token distributions."""

    def __init__(self, student_temp: float = 0.1, teacher_temp: float = 0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_cls, teacher_cls, center):
        teacher_logits = (teacher_cls.detach() - center) / self.teacher_temp
        teacher_logits = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
        teacher_out = F.softmax(teacher_logits, dim=-1)
        student_out = F.log_softmax(student_cls / self.student_temp, dim=-1)
        loss = -torch.sum(teacher_out * student_out, dim=-1).mean()
        return loss


class PatchDistillationLoss(nn.Module):
    """Cross-entropy between teacher and student patch token distributions, masked."""

    def __init__(self, student_temp: float = 0.1, teacher_temp: float = 0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_patches, teacher_patches, center, mask):
        teacher_logits = (teacher_patches.detach() - center.unsqueeze(0)) / self.teacher_temp
        teacher_logits = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values
        teacher_out = F.softmax(teacher_logits, dim=-1)
        student_out = F.log_softmax(student_patches / self.student_temp, dim=-1)
        per_token_loss = -torch.sum(teacher_out * student_out, dim=-1)
        mask_float = mask.float()
        loss = (per_token_loss * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        return loss


class GlobalContrastiveLoss(nn.Module):
    """InfoNCE over student rotated-view CLS against EMA teacher CLS."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    @torch.no_grad()
    def _gather_no_grad(x: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return x
        gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x)
        return torch.cat(gathered, dim=0)

    def forward(self, student_cls: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        student = F.normalize(student_cls, dim=-1)
        teacher_bank = F.normalize(self._gather_no_grad(teacher_cls.detach()), dim=-1)

        logits = student @ teacher_bank.T / self.temperature
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        labels = torch.arange(student.shape[0], device=student.device) + rank * student.shape[0]
        return F.cross_entropy(logits, labels)


class RotationPredictionLoss(nn.Module):
    """Cross-entropy over discretized ZYX Euler angles."""

    def __init__(self, n_bins: int = 36):
        super().__init__()
        self.n_bins = n_bins

    @property
    def pi(self):
        return 3.141592653589793

    @staticmethod
    def _rotation_matrix_to_euler(R):
        """Convert (B, 3, 3) rotation matrix to (B, 3) Euler angles (ZYX)."""
        # Clamp for numerical stability
        R = R.clamp(-1, 1)
        # yaw (Z), pitch (Y), roll (X)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        pitch = torch.asin(-R[:, 2, 0].clamp(-1, 1))
        roll = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        return torch.stack([yaw, pitch, roll], dim=-1)

    def forward(self, rot_logits: torch.Tensor, rot_matrix: torch.Tensor) -> torch.Tensor:
        euler = self._rotation_matrix_to_euler(rot_matrix)
        bin_size = 2 * self.pi / self.n_bins
        targets = ((euler + self.pi) / bin_size).long().clamp(0, self.n_bins - 1)
        logits = rot_logits.view(-1, 3, self.n_bins)
        return F.cross_entropy(logits.reshape(-1, self.n_bins), targets.reshape(-1))


class EquiSSLLoss(nn.Module):
    """DINO/iBOT + contrastive + rotation prediction + KoLeo."""

    def __init__(
        self,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        cls_weight: float = 1.0,
        patch_weight: float = 1.0,
        contrast_weight: float = 1.0,
        contrast_temp: float = 0.2,
        rot_weight: float = 0.5,
        koleo_weight: float = 0.01,
        rot_bins: int = 36,
    ):
        super().__init__()
        self.cls_loss_fn = CLSDistillationLoss(student_temp, teacher_temp)
        self.patch_loss_fn = PatchDistillationLoss(student_temp, teacher_temp)
        self.contrast_loss_fn = GlobalContrastiveLoss(contrast_temp)
        self.rot_loss_fn = RotationPredictionLoss(rot_bins)
        self.koleo_loss_fn = KoLeoLoss()
        self.cls_weight = cls_weight
        self.patch_weight = patch_weight
        self.contrast_weight = contrast_weight
        self.rot_weight = rot_weight
        self.koleo_weight = koleo_weight

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cls_loss = self.cls_loss_fn(
            outputs["student_cls_proj"],
            outputs["teacher_cls_proj"],
            outputs["cls_center"],
        )
        patch_loss = self.patch_loss_fn(
            outputs["student_patch_proj"],
            outputs["teacher_patch_proj"],
            outputs["patch_center"],
            outputs["mask"],
        )

        contrast_loss = torch.tensor(0.0, device=cls_loss.device)
        if self.contrast_weight > 0:
            contrast_loss = self.contrast_loss_fn(
                outputs["student_cls_features"],
                outputs["teacher_cls_features"],
            )

        rot_loss = torch.tensor(0.0, device=cls_loss.device)
        if self.rot_weight > 0 and "rot_logits" in outputs:
            rot_loss = self.rot_loss_fn(
                outputs["rot_logits"],
                outputs["rot_matrix"],
            )

        koleo_loss = torch.tensor(0.0, device=cls_loss.device)
        if self.koleo_weight > 0 and "student_patch_features" in outputs:
            koleo_loss = self.koleo_loss_fn(outputs["student_patch_features"])

        total = (self.cls_weight * cls_loss
                 + self.patch_weight * patch_loss
                 + self.contrast_weight * contrast_loss
                 + self.rot_weight * rot_loss
                 + self.koleo_weight * koleo_loss)
        return {
            "total": total,
            "cls": cls_loss,
            "patch": patch_loss,
            "contrast": contrast_loss,
            "rot": rot_loss,
            "koleo": koleo_loss,
        }
