# EquiSSL — Anonymous Code Release

This is the anonymous code release accompanying a double-blind submission to
**SIGGRAPH Asia 2026 Technical Papers (Journal track / ACM TOG)**.

Author identity is withheld during the review and will be disclosed in the
camera-ready version.

## Method (one-paragraph summary)

EquiSSL combines two levels of rotation equivariance on the icosphere mesh
for 360° panoramic perception:

1. **Architectural equivariance** — Gauge-pooled Relative Position Embedding
   (GE-RPE): averages relative-position biases across C_n gauge frames so
   attention is gauge-invariant and SO(3)-equivariant on the spherical graph.
2. **Representational equivariance** — Rotation-consistent iBOT + MAE
   pretraining, where teacher tokens are permuted to align with the
   student's rotated frame, yielding features that transform covariantly
   with SO(3).

Downstream tasks: semantic segmentation, depth estimation, and rotation
robustness on Stanford2D3D.

## Layout

| directory | contents |
|---|---|
| `equissl/` | model code (encoder, decoder, GE-RPE module, dataloaders, loss) |
| `tools/` | training and evaluation entry points |
| `scripts/` | shell launchers for the experimental pipelines |
| `configs/` | YAML training configurations |
| `figures/` | paper-figure renderers and analysis scripts |
| `docs/benchmark_protocol.md` | evaluation protocol used in the paper |

## Setup

Requires PyTorch and standard deps; see `requirements.txt`. The SphereUFormer
backbone (CVPR 2025) is an external dependency — clone it from its original
public release and point the env var below to its source root.

```bash
export EQUISSL_ROOT="$(pwd)"
export SPHERE_UFORMER_SRC="/path/to/sphere_uformer/src"
export STANFORD2D3D_PATH="/path/to/Stanford2D3D/extracted"
export PYTHONPATH="${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}:${PYTHONPATH}"
```

Paths inside scripts use `${EQUISSL_ROOT}`, `${SPHERE_UFORMER_SRC}`, and
`${STANFORD2D3D_PATH}` as placeholders — replace with your local filesystem
locations.

## Reproduction (high-level)

| step | entry point |
|---|---|
| Pretrain (iBOT + MAE + KoLeo) | `torchrun --nproc_per_node=N tools/pretrain.py --config configs/pretrain_v9_repaired.yaml --output_dir outputs/<name>` |
| Finetune semantic segmentation | `python tools/finetune_seg.py --pretrained <ckpt> --output_dir <name>` |
| Finetune depth | `python tools/finetune_depth.py --pretrained <ckpt> --output_dir <name>` |
| Rotation-robustness eval (Pose35) | `python tools/eval_pose35.py --checkpoint <ckpt> --max_angle 35.0 --num_rotations 10 --num_repeats 3` |

See `docs/benchmark_protocol.md` for the formal evaluation protocol used in
the paper, including the Stanford2D3D split convention and the Pose35
rotation-robustness definition.

## License

To be specified in the camera-ready release.
