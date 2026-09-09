"""
EquiSSL Encoder: Equivariant Self-Supervised Learning.

Combines SphereUFormer's icosphere backbone with iBOT-style masked
self-distillation, adapted for SO(3) equivariance on the sphere.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, "${SPHERE_UFORMER_SRC}")

from network.sphere_model import SphereUFormer, SphereUFormerModule, InputProj, CenterDownsample, NearestUpsample
from trimesh_utils import IcoSphereRef

from .projection_heads import ProjectionHead, PatchProjectionHead
from .masking import IcosphereMasking, IcosphereBlockMasking
from .equivariant_rpe import GaugeEquivariantRPE
from ..utils.sphere import (
    sample_so3_rotation,
    compute_rotation_permutation,
    apply_rotation_to_features,
    compute_inverse_permutation,
    compute_node_areas,
)


class SphericalEncoder(nn.Module):
    """
    Encoder-only version of SphereUFormer for SSL pretraining.
    Extracts the encoder path from U-Net and adds a CLS token.
    """

    def __init__(
        self,
        img_rank: int = 7,
        node_type: str = "vertex",
        in_channels: int = 3,
        embed_dim: int = 32,
        num_scales: int = 4,
        in_scale_factor: int = 2,
        enc_depths: Tuple = (2, 2, 2, 2),
        bottleneck_depth: int = 2,
        d_head_coef: int = 1,
        enc_num_heads: Tuple = (2, 4, 8, 16),
        bottleneck_num_heads: Optional[int] = None,
        win_size_coef: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        abs_pos_enc_in: bool = False,
        abs_pos_enc: bool = False,
        rel_pos_bias: bool = True,
        rel_pos_bias_size: int = 7,
        equivariant_rpe: bool = False,
        n_gauges: int = 6,
        area_weighted: bool = True,
    ):
        super().__init__()

        self.icosphere_ref = IcoSphereRef(node_type)
        self._equivariant_rpe = equivariant_rpe
        self._n_gauges = n_gauges
        self._area_weighted = area_weighted
        self.img_rank = img_rank
        self.node_type = node_type

        if in_scale_factor == 2:
            proj_rank = img_rank - 1
        elif in_scale_factor == 1:
            proj_rank = img_rank
        else:
            raise ValueError(f"Unsupported in_scale_factor={in_scale_factor}")

        self.proj_rank = proj_rank
        normals = self.icosphere_ref.get_normals(proj_rank)
        self.num_tokens = len(normals)

        if bottleneck_num_heads is None:
            bottleneck_num_heads = enc_num_heads[-1] * 2

        self.input_proj = InputProj(in_channels, embed_dim, act_layer=nn.GELU)

        if in_scale_factor == 2:
            self.input_downsample = CenterDownsample(img_rank, proj_rank, self.icosphere_ref)
        else:
            self.input_downsample = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths) + bottleneck_depth)]
        dpr_idx = 0

        self.enc_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        for i in range(num_scales):
            rank_i = proj_rank - i
            dim_i = embed_dim * (2 ** i)

            self.enc_blocks.append(
                SphereUFormerModule(
                    rank=rank_i,
                    icosphere_ref=self.icosphere_ref,
                    dim=dim_i,
                    depth=enc_depths[i],
                    num_heads=enc_num_heads[i],
                    d_head_coef=d_head_coef,
                    win_size_coef=win_size_coef,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=[dpr[dpr_idx + j] for j in range(enc_depths[i])],
                    abs_pos_enc=abs_pos_enc,
                    rel_pos_bias=rel_pos_bias,
                    rel_pos_bias_size=rel_pos_bias_size,
                )
            )
            dpr_idx += enc_depths[i]

            if i < num_scales - 1:
                next_rank = rank_i - 1
                next_dim = dim_i * 2
                self.downsample_blocks.append(nn.Sequential(
                    CenterDownsample(rank_i, next_rank, self.icosphere_ref),
                    nn.LayerNorm(dim_i),
                    nn.Linear(dim_i, next_dim),
                ))
            else:
                next_rank = rank_i - 1
                next_dim = dim_i * 2
                self.downsample_blocks.append(nn.Sequential(
                    CenterDownsample(rank_i, next_rank, self.icosphere_ref),
                    nn.LayerNorm(dim_i),
                    nn.Linear(dim_i, next_dim),
                ))

        bottleneck_rank = proj_rank - num_scales
        bottleneck_dim = embed_dim * (2 ** num_scales)
        self.bottleneck = SphereUFormerModule(
            rank=bottleneck_rank,
            icosphere_ref=self.icosphere_ref,
            dim=bottleneck_dim,
            depth=bottleneck_depth,
            num_heads=bottleneck_num_heads,
            d_head_coef=d_head_coef,
            win_size_coef=win_size_coef,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=[dpr[dpr_idx + j] for j in range(bottleneck_depth)],
            abs_pos_enc=abs_pos_enc,
            rel_pos_bias=rel_pos_bias,
            rel_pos_bias_size=rel_pos_bias_size,
        )

        self.bottleneck_dim = bottleneck_dim
        self.norm = nn.LayerNorm(bottleneck_dim)

        if equivariant_rpe:
            self._replace_rpe_with_gauge_equivariant()

    def _replace_rpe_with_gauge_equivariant(self):
        """Replace all RelativePositionBias in attention blocks with GaugeEquivariantRPE."""
        for module in self.modules():
            if hasattr(module, 'rel_pos_bias') and hasattr(module, 'apply_rel_pos_bias'):
                if not isinstance(module.rel_pos_bias, GaugeEquivariantRPE):
                    old_rpb = module.rel_pos_bias
                    new_rpb = GaugeEquivariantRPE(
                        rank=old_rpb.rank,
                        icosphere_ref=self.icosphere_ref,
                        win_size_coef=getattr(module, 'win_size_coef', 2),
                        rel_pos_bias_size=old_rpb.bias_grid.shape[-1],
                        num_heads=old_rpb.bias_grid.shape[1],
                        n_gauges=self._n_gauges,
                        area_weighted=self._area_weighted,
                    )
                    module.rel_pos_bias = new_rpb

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_enc_outs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, N_img, 3) — icosphere RGB features at img_rank
            mask: (B, N_proj) bool — which tokens to mask (optional, for student)
            return_enc_outs: if True, return intermediate encoder features for decoder

        Returns:
            dict with 'cls' (B, D), 'patch' (B, N_bottleneck, D),
            and optionally 'enc_outs' list of encoder stage outputs
        """
        x = self.input_downsample(x)
        x = self.input_proj(x)

        if mask is not None:
            x = x * (~mask).unsqueeze(-1).float()

        enc_outs = []
        for i, (enc, down) in enumerate(zip(self.enc_blocks, self.downsample_blocks)):
            x = enc(x)
            enc_outs.append(x)
            x = down(x)

        x = self.bottleneck(x)
        x = self.norm(x)

        cls_token = x.mean(dim=1)
        patch_tokens = x

        result = {"cls": cls_token, "patch": patch_tokens}
        if return_enc_outs:
            result["enc_outs"] = enc_outs
        return result


class EquiSSLEncoder(nn.Module):
    """
    EquiSSL: Equivariant Self-Supervised Learning Encoder.

    Teacher-Student masked self-distillation on icosphere:
    - Teacher sees full icosphere (no mask, no rotation)
    - Student sees rotated + masked icosphere
    - CLS: rotation-invariant loss
    - Patch: rotation-equivariant loss (teacher patches permuted to match student)
    """

    def __init__(
        self,
        img_rank: int = 7,
        node_type: str = "vertex",
        embed_dim: int = 32,
        num_scales: int = 4,
        proj_dim: int = 256,
        out_dim: int = 65536,
        mask_ratio: float = 0.75,
        teacher_momentum: float = 0.996,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        abs_pos_enc_in: bool = False,
        abs_pos_enc: bool = False,
        rel_pos_bias: bool = True,
        dec_depths: Tuple = (2, 2, 2, 2),
        dec_num_heads: Tuple = (16, 16, 8, 4),
        rot_bins: int = 36,
        **backbone_kwargs,
    ):
        super().__init__()

        self.teacher_momentum = teacher_momentum
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        backbone_args = dict(
            img_rank=img_rank,
            node_type=node_type,
            embed_dim=embed_dim,
            num_scales=num_scales,
            abs_pos_enc_in=abs_pos_enc_in,
            abs_pos_enc=abs_pos_enc,
            rel_pos_bias=rel_pos_bias,
            **backbone_kwargs,
        )

        self.student = SphericalEncoder(**backbone_args)
        self.teacher = SphericalEncoder(**backbone_args)
        self._copy_student_to_teacher()

        bottleneck_dim = self.student.bottleneck_dim

        self.student_cls_head = ProjectionHead(
            in_dim=bottleneck_dim, hidden_dim=2048, bottleneck_dim=proj_dim, out_dim=out_dim
        )
        self.student_patch_head = PatchProjectionHead(
            in_dim=bottleneck_dim, hidden_dim=2048, out_dim=out_dim
        )
        self.teacher_cls_head = ProjectionHead(
            in_dim=bottleneck_dim, hidden_dim=2048, bottleneck_dim=proj_dim, out_dim=out_dim
        )
        self.teacher_patch_head = PatchProjectionHead(
            in_dim=bottleneck_dim, hidden_dim=2048, out_dim=out_dim
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 3 * rot_bins),
        )
        self._copy_head_params()

        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_cls_head.parameters():
            p.requires_grad = False
        for p in self.teacher_patch_head.parameters():
            p.requires_grad = False

        icosphere_ref = self.student.icosphere_ref
        num_tokens = self.student.num_tokens
        areas = compute_node_areas(icosphere_ref, self.student.proj_rank)
        self.masking = IcosphereMasking(
            num_nodes=num_tokens, mask_ratio=mask_ratio, area_weights=areas
        )

        normals = icosphere_ref.get_normals(self.student.proj_rank)
        self.register_buffer("normals", torch.tensor(normals, dtype=torch.float32), persistent=False)

        img_normals = icosphere_ref.get_normals(img_rank)
        self.register_buffer("img_normals", torch.tensor(img_normals, dtype=torch.float32), persistent=False)

        bottleneck_rank = self.student.proj_rank - num_scales
        bn_normals = icosphere_ref.get_normals(bottleneck_rank)
        self.register_buffer("bn_normals", torch.tensor(bn_normals, dtype=torch.float32), persistent=False)
        self.bn_num_nodes = len(bn_normals)

        bn_areas = compute_node_areas(icosphere_ref, bottleneck_rank)
        self.bn_masking = IcosphereMasking(
            num_nodes=self.bn_num_nodes, mask_ratio=mask_ratio, area_weights=bn_areas
        )

        self.register_buffer("cls_center", torch.zeros(1, out_dim), persistent=True)
        self.register_buffer("patch_center", torch.zeros(1, out_dim), persistent=True)

        dec_kwargs = {k: v for k, v in backbone_kwargs.items()
                      if k in ('d_head_coef', 'win_size_coef', 'mlp_ratio', 'qkv_bias',
                               'drop_path_rate', 'rel_pos_bias_size',
                               'equivariant_rpe', 'n_gauges', 'area_weighted')}
        self.decoder = SphericalDecoder(
            icosphere_ref=icosphere_ref,
            proj_rank=self.student.proj_rank,
            num_scales=num_scales,
            embed_dim=embed_dim,
            dec_depths=dec_depths,
            dec_num_heads=dec_num_heads,
            abs_pos_enc=abs_pos_enc,
            rel_pos_bias=rel_pos_bias,
            **dec_kwargs,
        )
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
        )

    def _copy_student_to_teacher(self):
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.copy_(sp.data)

    def _copy_head_params(self):
        for tp, sp in zip(self.teacher_cls_head.parameters(), self.student_cls_head.parameters()):
            tp.data.copy_(sp.data)
        for tp, sp in zip(self.teacher_patch_head.parameters(), self.student_patch_head.parameters()):
            tp.data.copy_(sp.data)

    @torch.no_grad()
    def update_teacher(self, momentum: Optional[float] = None):
        m = momentum if momentum is not None else self.teacher_momentum
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)
        for tp, sp in zip(self.teacher_cls_head.parameters(), self.student_cls_head.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)
        for tp, sp in zip(self.teacher_patch_head.parameters(), self.student_patch_head.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1 - m)

    @torch.no_grad()
    def update_center(self, teacher_cls_out: torch.Tensor, teacher_patch_out: torch.Tensor):
        cls_mean = teacher_cls_out.mean(dim=0, keepdim=True)
        patch_mean = teacher_patch_out.mean(dim=(0, 1), keepdim=True).squeeze(1)

        if dist.is_initialized():
            dist.all_reduce(cls_mean)
            cls_mean /= dist.get_world_size()
            dist.all_reduce(patch_mean)
            patch_mean /= dist.get_world_size()

        m = self.center_momentum
        self.cls_center = m * self.cls_center + (1 - m) * cls_mean
        self.patch_center = m * self.patch_center + (1 - m) * patch_mean

    def _generate_rotation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random SO(3) rotation and compute node permutations
        at img_rank, proj_rank, and bottleneck_rank levels.

        Returns:
            rot_matrix: (3, 3) numpy
            img_perm: (N_img,) numpy int64
            proj_perm: (N_proj,) numpy int64
            bn_perm: (N_bottleneck,) numpy int64
        """
        rot_matrix = sample_so3_rotation()
        img_perm = compute_rotation_permutation(
            self.img_normals.cpu().numpy(), rot_matrix
        )
        proj_perm = compute_rotation_permutation(
            self.normals.cpu().numpy(), rot_matrix
        )
        bn_perm = compute_rotation_permutation(
            self.bn_normals.cpu().numpy(), rot_matrix
        )
        return rot_matrix, img_perm, proj_perm, bn_perm

    def forward(self, sphere_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            sphere_rgb: (B, N_img, 3) — icosphere RGB at img_rank

        Returns:
            dict with keys for EquiSSLLoss:
                student_cls_proj, student_cls_features, student_patch_proj,
                teacher_cls_proj, teacher_cls_features, teacher_patch_proj,
                mask, cls_center, patch_center,
                rot_logits, rot_matrix
        """
        B = sphere_rgb.shape[0]
        device = sphere_rgb.device

        rot_matrix, img_perm, proj_perm, bn_perm = self._generate_rotation()
        img_perm_t = torch.tensor(img_perm, dtype=torch.long, device=device)
        proj_perm_t = torch.tensor(proj_perm, dtype=torch.long, device=device)
        bn_perm_t = torch.tensor(bn_perm, dtype=torch.long, device=device)

        input_mask = self.masking(B, device)
        bn_mask = self.bn_masking(B, device)

        sphere_rgb_rotated = apply_rotation_to_features(sphere_rgb, img_perm_t)

        with torch.no_grad():
            self.teacher.eval()
            self.teacher_cls_head.eval()
            self.teacher_patch_head.eval()
            teacher_out = self.teacher(sphere_rgb, mask=None)
            teacher_cls_features = teacher_out["cls"]
            teacher_cls_proj = self.teacher_cls_head(teacher_cls_features)
            teacher_patch_proj = self.teacher_patch_head(teacher_out["patch"])
            self.teacher.train()
            self.teacher_cls_head.train()
            self.teacher_patch_head.train()

            teacher_patch_proj_aligned = teacher_patch_proj[:, bn_perm_t]

        student_out = self.student(sphere_rgb_rotated, mask=input_mask, return_enc_outs=False)
        student_cls_features = student_out["cls"]  # Before projection
        student_cls_proj = self.student_cls_head(student_cls_features)
        student_patch_proj = self.student_patch_head(student_out["patch"])
        rot_logits = self.rotation_head(student_cls_features)

        # Convert rot_matrix to tensor, add batch dim
        rot_matrix_t = torch.tensor(rot_matrix, dtype=torch.float32, device=device).unsqueeze(0)
        rot_matrix_t = rot_matrix_t.expand(B, -1, -1)

        self.update_center(teacher_cls_proj, teacher_patch_proj)

        return {
            "student_cls_proj": student_cls_proj,
            "student_cls_features": student_cls_features,
            "student_patch_proj": student_patch_proj,
            "student_patch_features": student_out["patch"],
            "teacher_cls_proj": teacher_cls_proj,
            "teacher_cls_features": teacher_cls_features,
            "teacher_patch_proj": teacher_patch_proj_aligned,
            "mask": bn_mask,
            "cls_center": self.cls_center,
            "patch_center": self.patch_center,
            "rot_logits": rot_logits,
            "rot_matrix": rot_matrix_t,
        }


class SphericalDecoder(nn.Module):
    """
    U-Net decoder for SphericalEncoder, mirroring SphereUFormer's design.
    Takes bottleneck features + encoder skip connections, upsamples back to proj_rank.
    """

    def __init__(
        self,
        icosphere_ref: IcoSphereRef,
        proj_rank: int,
        num_scales: int = 4,
        embed_dim: int = 32,
        dec_depths: Tuple = (2, 2, 2, 2),
        dec_num_heads: Tuple = (16, 16, 8, 4),
        d_head_coef: int = 1,
        win_size_coef: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        abs_pos_enc: bool = False,
        rel_pos_bias: bool = True,
        rel_pos_bias_size: int = 7,
        equivariant_rpe: bool = False,
        n_gauges: int = 6,
        area_weighted: bool = True,
    ):
        super().__init__()

        self.num_scales = num_scales
        self._icosphere_ref = icosphere_ref
        self._equivariant_rpe = equivariant_rpe
        self._n_gauges = n_gauges
        self._area_weighted = area_weighted
        dec_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]

        self.upsample_blocks = nn.ModuleList()
        self.dec_norm1 = nn.ModuleList()
        self.dec_norm2 = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(num_scales):
            reverse_i = num_scales - i - 1
            dim_i = embed_dim * (2 ** reverse_i)
            in_rank = proj_rank - reverse_i - 1
            out_rank = proj_rank - reverse_i

            self.upsample_blocks.append(nn.Sequential(
                nn.LayerNorm(dim_i * 2),
                nn.Linear(dim_i * 2, dim_i),
                NearestUpsample(in_rank, out_rank, icosphere_ref),
            ))

            self.dec_norm1.append(nn.LayerNorm(dim_i))
            self.dec_norm2.append(nn.LayerNorm(dim_i))

            self.dec_blocks.append(nn.Sequential(
                nn.Linear(dim_i * 2, dim_i),
                SphereUFormerModule(
                    rank=out_rank,
                    icosphere_ref=icosphere_ref,
                    dim=dim_i,
                    depth=dec_depths[i],
                    num_heads=dec_num_heads[i],
                    d_head_coef=d_head_coef,
                    win_size_coef=win_size_coef,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dec_dpr[int(sum(dec_depths[:i])):int(sum(dec_depths[:i+1]))],
                    abs_pos_enc=abs_pos_enc,
                    rel_pos_bias=rel_pos_bias,
                    rel_pos_bias_size=rel_pos_bias_size,
                ),
            ))

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        if equivariant_rpe:
            for module in self.modules():
                if hasattr(module, 'rel_pos_bias') and hasattr(module, 'apply_rel_pos_bias'):
                    if not isinstance(module.rel_pos_bias, GaugeEquivariantRPE):
                        old_rpb = module.rel_pos_bias
                        module.rel_pos_bias = GaugeEquivariantRPE(
                            rank=old_rpb.rank, icosphere_ref=self._icosphere_ref,
                            win_size_coef=getattr(module, 'win_size_coef', 2),
                            rel_pos_bias_size=old_rpb.bias_grid.shape[-1],
                            num_heads=old_rpb.bias_grid.shape[1],
                            n_gauges=n_gauges, area_weighted=area_weighted,
                        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        enc_outs: list,
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: (B, N_bn, D_bn) from encoder bottleneck
            enc_outs: list of encoder stage outputs [stage0, stage1, ...], high-res first

        Returns:
            (B, N_proj, embed_dim)
        """
        y = bottleneck
        for i in range(self.num_scales):
            y = self.upsample_blocks[i](y)
            skip = enc_outs[self.num_scales - 1 - i]
            y = torch.cat([self.dec_norm1[i](y), self.dec_norm2[i](skip)], dim=-1)
            y = self.dec_blocks[i](y)

        y = self.output_proj(y)
        return y


class EquiSSLSegUNet(nn.Module):
    """
    EquiSSL U-Net for segmentation: pretrained encoder + decoder + seg head.
    Segmentation is done at proj_rank resolution (40962 nodes).
    """

    def __init__(
        self,
        encoder: SphericalEncoder,
        num_classes: int = 14,
        dec_depths: Tuple = (2, 2, 2, 2),
        dec_num_heads: Tuple = (16, 16, 8, 4),
        freeze_encoder: bool = False,
        **decoder_kwargs,
    ):
        super().__init__()
        self.encoder = encoder

        embed_dim = encoder.input_proj.proj[0].out_features

        self.decoder = SphericalDecoder(
            icosphere_ref=encoder.icosphere_ref,
            proj_rank=encoder.proj_rank,
            num_scales=len(encoder.enc_blocks),
            embed_dim=embed_dim,
            dec_depths=dec_depths,
            dec_num_heads=dec_num_heads,
            **decoder_kwargs,
        )

        out_dim = self.decoder.output_proj.out_features
        self.seg_head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_result = self.encoder(x, mask=None, return_enc_outs=True)
        dec_out = self.decoder(enc_result["patch"], enc_result["enc_outs"])
        return self.seg_head(dec_out)
