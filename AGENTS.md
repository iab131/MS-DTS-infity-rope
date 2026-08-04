# Infinity-RoPE Working Instructions

## Research goal

Study and improve long-duration, action-controllable video generation with
Infinity-RoPE. Use the MS-DTS-RoPE project as prior research context and a
source of lessons, not as a mandate to port its code or architecture.

Read `docs/MS_DTS_RESEARCH_HANDOFF.md` before planning related work.

## Default workflow

- Start by tracing Infinity-RoPE's current behavior and establishing a
  reproducible baseline. Do not begin by copying MS-DTS code.
- Treat the current checkout and tests as authoritative. Historical results in
  the handoff are evidence from a different checkout and date.
- Keep changes minimal, isolated, and tied to an explicit hypothesis.
- Preserve all pre-existing modified and untracked files. Never clean or reset
  the worktree without explicit permission.
- Use deterministic/unit and metadata checks before GPU generation.
- For comparisons, keep every setting identical except the named method. Log
  failures and OOMs instead of silently reducing frames, resolution, or other
  settings.
- A dry run or partial result matrix is planning evidence, not visual-quality
  evidence.

## Environment and machine safety

- Work from `/home/sigasia2026/projects/infinity-rope`.
- Reuse the Conda environment with `conda activate wan`.
- FlashAttention is already installed and GPU-verified in `wan`; do not rebuild
  it unless a demonstrated incompatibility requires it.
- Before installing anything, check whether it is already importable. Do not
  downgrade the shared Torch, Diffusers, NumPy, or Pydantic stack merely to
  match broad or old repository pins.
- If native compilation is unavoidable, set `MAX_JOBS=1`,
  `CMAKE_BUILD_PARALLEL_LEVEL=1`, `MAKEFLAGS=-j1`, `OMP_NUM_THREADS=1`, and
  `MKL_NUM_THREADS=1`.
- Do not run `setup_env.sh` blindly: it combines dependency installation with
  several model/checkpoint downloads.
- Do not download model weights or start expensive generation unless the user
  explicitly requests it. Check RAM, disk, GPU availability, and the exact
  command first.

## Snapshot from 2026-08-03

- `wan`: Python 3.10, Torch 2.11.0+cu130, CUDA 13.0, FlashAttention 2.8.3.
- A bounded FlashAttention CUDA smoke test passed on the RTX PRO 6000
  Blackwell GPU.
- WandB, ONNX, ONNX Runtime, ONNX Script, ONNX Converter Common, TensorRT, and
  PyCUDA were added to `wan` without rebuilding FlashAttention.
- `pip check` still reports unrelated pre-existing gaps for `fvcore`, `pyiqa`,
  and unsupported `decord`; do not "fix" these unless the active task uses
  them.
- OpenAI CLIP is importable even though its distribution metadata is absent.
- Recheck this snapshot before relying on it; environments drift.

## Scope boundary

Infinity-RoPE already has duration-controlled action segments, hard-cut syntax,
KV flushing, and autoregressive self-rollout. Understand those mechanisms on
their own terms. Any future relationship to MS-DTS-RoPE must be justified by a
specific experiment or failure in Infinity-RoPE, not assumed in advance.
