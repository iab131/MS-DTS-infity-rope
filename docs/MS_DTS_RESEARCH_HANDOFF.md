# MS-DTS-RoPE Research Handoff for Infinity-RoPE

Status snapshot: 2026-08-03

## Why this document exists

Future Codex sessions will start in the Infinity-RoPE repository rather than
the MS-DTS-RoPE repository. This document preserves the useful context,
evidence, mistakes, and operating constraints from the earlier work without
turning that work into a predetermined implementation plan.

## Goal in this repository

The goal is to study and improve **long-duration, action-controllable video
generation using Infinity-RoPE**.

MS-DTS-RoPE is background knowledge. There is currently no decision to port its
Shot-Time Field, prompt-routing system, temporal RoPE modes, or other controls
into Infinity-RoPE. First understand Infinity-RoPE, reproduce its baseline, and
identify a concrete limitation. Only then decide whether an earlier idea is
relevant.

## Repository locations and current state

### Infinity-RoPE

- Path: `/home/sigasia2026/projects/infinity-rope`
- Upstream: `https://github.com/yesiltepe-hidir/infinity-rope.git`
- Branch at handoff: `main`
- Main entry point: `inference.py`
- Pipeline implementations: `pipeline/`
- Wan wrappers: `utils/wan_wrapper.py`
- Core Wan modules: `wan/modules/causal_model.py`, `wan/modules/model.py`, and
  `wan/modules/attention.py`
- Prompt examples: `prompts/infinity_rope_prompts.txt`

Infinity-RoPE already represents action duration with `[Ns]`, chains actions
with `|`, requests scene cuts with `#`, and uses KV flushing/index changes for
long action-controlled rollout. Do not describe this as equivalent to
MS-DTS-RoPE until the actual execution paths have been compared.

Pre-existing untracked files observed before this handoff were:

- `algorithm_attention_memory_policy_20260803_140906.md`
- `survey_algorithms_20260803_140833.md`

They belong to the existing worktree. Preserve them.

### MS-DTS-RoPE

- Path: `/home/sigasia2026/projects/MS-DTS-RoPE-Enhe`
- Branch at handoff: `overnight/multishot-improvement`
- Base commit at handoff: `a23036f`
- The implementation is present in a dirty, uncommitted checkout. Do not treat
  it as a clean package or copy files without inspecting the diff.

The live implementation path is:

```text
generate.py
  -> wan/text2video.py
  -> wan/utils/multishot.py
  -> wan/modules/model.py
```

Focused regression coverage is in `tests/test_multishot.py`.

## What MS-DTS-RoPE explored

MS-DTS-RoPE investigated structured multi-shot control for Wan generation. Its
central design separates two notions of time:

- **Global DTS time** describes progress across the generated sequence.
- **Shot-local temporal RoPE** describes temporal position within an individual
  shot.

The main mechanisms were:

1. **Shot-Time Field**
   (`wan/utils/multishot.py::build_shot_time_field`): maps requested shot
   durations to causal-VAE latent time, allocates positive durations exactly,
   and represents hard, soft, and hybrid transitions with normalized weights.
2. **Prompt token routing**
   (`build_prompt_token_spans` and
   `build_anchor_token_spans_with_weights`): keeps global and shot-specific
   prompt regions distinct while allowing controlled access to shared context.
3. **Multi-shot temporal frequencies and attention biasing**
   (`wan/modules/model.py`): applies shot-local temporal frequencies and
   conditional control paths while keeping conditional/unconditional CFG
   temporal RoPE aligned.
4. **Temporal Dy-YaRN experiment**
   (`temporal_dy_yarn_temporal_freqs`): an isolated temporal-only,
   DyPE/YaRN-inspired mode. It is not the original spatial DyPE method and must
   never be presented as such.

Other MS-DTS controls included prompt relay, event/query steering, and memory
paths. They were deliberately kept separate from the temporal Dy-YaRN change.

## What was actually verified

The following evidence belongs to the MS-DTS checkout and prior runs; it does
not prove anything about Infinity-RoPE:

- 78 deterministic unit tests passed in the end-to-end multi-shot checkout.
- 51,735 duration-allocation cases passed.
- 600 hard/soft/hybrid transition configurations stayed finite, non-negative,
  normalized, and limited to adjacent shots.
- Repository storyboards through 241 frames passed metadata, prompt,
  Shot-Time Field, and local UMT5 tokenizer checks.
- Raw Wan and cosine DTS formulas were checked against their baselines.
- A bounded raw 81-frame generation completed in 500.57 seconds with 36,587
  MiB peak VRAM in the temporal comparison workflow.

These were implementation and bounded-execution checks, not a completed visual
quality study.

## What remains unknown or incomplete

- A continuous latent sequence cannot completely reset at a hard cut; actual
  visual transition quality remains dependent on the model and prompt.
- The raw/cosine-DTS/`temporal_dy_yarn` matrix at 81/161/241 frames was not
  completed or fully reviewed.
- The requested six-run Wan2.1 comparison at 961 and 1921 frames was never
  implemented or executed.
- There is no evidence that `temporal_dy_yarn` is visually better than raw Wan
  or cosine DTS.
- No prior MS-DTS result establishes that its architecture is appropriate for
  Infinity-RoPE's autoregressive self-rollout.

## Mistakes and lessons to preserve

### Implementation lessons

- Fix shared normalization and validation at the common boundary. Patching
  missing/zero durations in individual callers creates inconsistent behavior.
- Repeated anchor text must be resolved within the current shot's character
  range; global search can bind to the global prompt or an earlier shot.
- Validate the storyboard root and every shot as mappings before calling
  `.get`; malformed list roots previously caused an `AttributeError`.
- Build prompt labels from the final Wan-valid `4n+1` frame count, not the
  pre-adjustment request.
- Preserve the distinction between global sequence time and shot-local time.
- Keep conditional-only prompt/event/query/memory controls out of the
  unconditional CFG branch while sharing temporal RoPE between CFG branches.

### Experiment lessons

- Keep all generation settings identical across methods. Change only the
  method or duration named by the experiment.
- Never silently lower resolution, frame count, steps, or other settings to
  rescue an OOM. Record the failure instead.
- Every attempted run should record the exact command, method, frame count,
  playback duration, runtime, peak VRAM, output path, and failure status.
- `nvidia-smi` telemetry is comma-delimited (`timestamp, memory`). Parse the
  final comma-separated field and test the parser on a sample line before an
  expensive run.
- A dry-run manifest containing "not run" rows is a planning artifact, not
  comparative evidence.
- A partial matrix and one successful baseline cannot support a method-quality
  conclusion.
- Inspect and preserve dirty worktrees before editing. Separate new changes
  from existing code, reports, scripts, and generated outputs.

### Environment and machine-safety lessons

- Reuse the existing `wan` environment. FlashAttention 2.8.3 already works
  with Torch 2.11.0+cu130 and CUDA 13.0 on this machine.
- A separate environment attempt unnecessarily started rebuilding
  FlashAttention and coincided with a machine crash. Check imports first and
  do not rebuild working native extensions.
- If compilation is unavoidable, use one job and monitor memory, swap, disk,
  and active compiler processes.
- Infinity-RoPE's `setup_env.sh` mixes environment creation, dependency
  installation, FlashAttention compilation, and several large model downloads.
  Execute those concerns separately.
- The shared `wan` environment intentionally retains newer Diffusers, NumPy,
  and Pydantic versions than Infinity-RoPE's exact pins. Core imports and a
  bounded FlashAttention CUDA call passed; do not downgrade the shared
  environment without a reproduced incompatibility.
- `pip check` has unrelated pre-existing `fvcore`, `pyiqa`, and `decord`
  complaints. Do not expand scope to those packages unless the active workflow
  imports them.

## Recommended starting workflow in Infinity-RoPE

1. Activate and inspect, without installing:

   ```bash
   cd /home/sigasia2026/projects/infinity-rope
   conda activate wan
   git status --short --branch
   python -c "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__)"
   ```

2. Trace the current Infinity-RoPE path:

   ```text
   inference.py
     -> pipeline/
     -> utils/wan_wrapper.py
     -> wan/modules/causal_model.py
   ```

3. Write down one explicit question about Infinity-RoPE behavior before
   changing code. Examples include duration accuracy, action-boundary control,
   scene-cut behavior, identity consistency, or long-rollout degradation.
4. Add the smallest deterministic or metadata-level check that can answer part
   of that question.
5. Run GPU generation only after the user approves weights and the exact
   experiment. Start bounded and matched.
6. Report evidence separately from interpretation. Clearly label unrun,
   partial, OOM, and visually reviewed cases.

## Definition of a useful future result

A useful result in Infinity-RoPE should include:

- a stated hypothesis;
- the exact current baseline;
- a minimal, isolated change if a change is needed;
- deterministic regression coverage where possible;
- a matched generation command and structured run record when generation is
  required;
- explicit limits and failures; and
- no claim that MS-DTS behavior transfers unless the Infinity-RoPE evidence
  demonstrates it.

## Local references

- MS-DTS implementation:
  `/home/sigasia2026/projects/MS-DTS-RoPE-Enhe`
- MS-DTS main multi-shot helper:
  `/home/sigasia2026/projects/MS-DTS-RoPE-Enhe/wan/utils/multishot.py`
- MS-DTS temporal/model paths:
  `/home/sigasia2026/projects/MS-DTS-RoPE-Enhe/wan/modules/model.py`
- MS-DTS tests:
  `/home/sigasia2026/projects/MS-DTS-RoPE-Enhe/tests/test_multishot.py`
- MS-DTS comparison runner:
  `/home/sigasia2026/projects/MS-DTS-RoPE-Enhe/scripts/run_temporal_dy_yarn_comparison.sh`

Always verify these paths and the live Git state before relying on them.
