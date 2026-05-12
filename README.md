# EquiSSL — Anonymous Code Release

### *Gauge-Equivariant Attention for Rotation-Stable 360° Scene Understanding*

Anonymous code release accompanying a double-blind submission to **SIGGRAPH
Asia 2026 Technical Papers** (Journal track / ACM TOG). Author identity is
withheld during review and will be disclosed in the camera-ready release.

## Motivation

A single defect — gauge dependence in the relative-position bias of an
icosphere transformer — couples two failure modes: rotation transfer at
inference time and rotation-consistent SSL pretraining. A parameter-free
Reynolds average over a finite cyclic subgroup of gauge rotations unblocks
both.

![Motivation: rotation drop under three configurations (random RPE, SSL +
gauge-dependent RPE, SSL + GE-RPE).](docs/figs/motivation.png)

## Pipeline

EquiSSL inserts the gauge-equivariant relative-position embedding (GE-RPE)
inside the attention layer, and pretrains the backbone with a
rotation-consistent iBOT + MAE objective. Both stages share the same
Reynolds-average mechanism on a discrete cyclic subgroup
*C<sub>n</sub>* ⊂ SO(2); the inference stage uses it inside attention, the
pretraining stage uses it to align teacher and student tokens in a common
rotated frame.

![Pipeline: pretraining and deployment stages, with the
GE-RPE Reynolds-average box inside one attention layer.](docs/figs/pipeline.png)

## Method (one-paragraph summary)

EquiSSL combines two levels of rotation equivariance on the icosphere mesh
for 360° panoramic perception:

1. **Architectural equivariance** — Gauge-pooled Relative Position
   Embedding (GE-RPE): averages relative-position biases across
   *C<sub>n</sub>* gauge frames so attention is gauge-invariant and
   SO(3)-equivariant on the spherical graph.
2. **Representational equivariance** — Rotation-consistent iBOT + MAE
   pretraining, where teacher tokens are permuted to align with the
   student's rotated frame, yielding features that transform covariantly
   with SO(3).

Downstream tasks: semantic segmentation, depth estimation, and rotation
robustness on Stanford2D3D; zero-shot transfer on Structured3D and
Matterport3D.

## Layout

| directory | contents |
|---|---|
| `equissl/` | Model code — encoder, decoder, GE-RPE module, dataloaders, losses |
| `equissl/models/equivariant_rpe.py` | Reference `GaugeEquivariantRPE` implementation (Reynolds average over *C<sub>n</sub>*) |
| `tools/` | Training and evaluation entry points |
| `scripts/` | Shell launchers for the experimental pipelines |
| `configs/` | YAML training configurations |
| `figures/` | Paper-figure renderers and analysis scripts |
| `docs/benchmark_protocol.md` | Evaluation protocol used in the paper |

## Setup

Requires PyTorch and standard deps; see `requirements.txt`. The
SphereUFormer backbone (CVPR 2025) is an external dependency — clone it
from its original public release and point the env var below to its source
root.

```bash
export EQUISSL_ROOT="$(pwd)"
export SPHERE_UFORMER_SRC="/path/to/sphere_uformer/src"
export STANFORD2D3D_PATH="/path/to/Stanford2D3D/extracted"
export PYTHONPATH="${EQUISSL_ROOT}:${SPHERE_UFORMER_SRC}:${PYTHONPATH}"
```

Paths inside scripts use `${EQUISSL_ROOT}`, `${SPHERE_UFORMER_SRC}`, and
`${STANFORD2D3D_PATH}` as placeholders — replace with your local
filesystem locations.

## Reproduction (high-level)

| step | entry point |
|---|---|
| Pretrain (iBOT + MAE + KoLeo) | `torchrun --nproc_per_node=N tools/pretrain.py --config configs/pretrain_v9_repaired.yaml --output_dir outputs/<name>` |
| Finetune semantic segmentation | `python tools/finetune_seg.py --pretrained <ckpt> --output_dir <name>` |
| Finetune depth | `python tools/finetune_depth.py --pretrained <ckpt> --output_dir <name>` |
| Rotation-robustness eval (θ<sub>max</sub>=90°) | `python tools/eval_pose35.py --checkpoint <ckpt> --max_angle 90.0 --num_rotations 10 --num_repeats 3` |
| Zero-shot Structured3D transfer | `python tools/eval_s3d_zeroshot.py --checkpoint <ckpt>` |

The rotation evaluation script is parametric in `--max_angle`; the paper
headline uses θ<sub>max</sub>=90°. The script's historical name
(`eval_pose35.py`) reflects the original ±35° sweep and is retained for
backward compatibility with checkpoint metadata. See
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) for the formal
evaluation protocol including the Stanford2D3D split convention and the
SO(3) rotation-robustness definition.

## Reference module

The core of the method is a ~150-line Reynolds-average wrapper over the
existing 2-D RPE table — no extra parameters, no extra optimisation
machinery. The reference implementation lives at
[`equissl/models/equivariant_rpe.py`](equissl/models/equivariant_rpe.py)
and is a near-drop-in replacement for the Swin-style 2-D RPE bias used in
icosphere transformers.

## Anonymity

This release is scrubbed for double-blind compliance: no author names,
institutions, server hostnames, personal filesystem paths, account
identifiers, or acknowledgements appear in code, comments, or
configuration. Output directories, dataset roots, and the SphereUFormer
source root are referenced via the environment variables listed in
"Setup".

## License

To be specified in the camera-ready release.
