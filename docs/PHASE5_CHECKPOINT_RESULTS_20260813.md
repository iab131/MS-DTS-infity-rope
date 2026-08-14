# Phase 5 Checkpoint Results — 2026-08-13/14

**Scope:** preregistered 63-run checkpoint only. Mechanism frozen at `e556855`;
runner/provenance commit `eb85a7df609165ac30d8593bb78130144984d27e`.

## Mechanical completion

| Item | Result |
| --- | --- |
| Matrix | 7 categories × 1 storyboard × seeds 101/202/303 × 3 main arms = 63 |
| Completion | 63/63 completed, 63 return code 0, no reruns or omitted cells |
| Runtime | 4,209.862 s total; 61.638–77.878 s/run |
| Peak direct-process VRAM | 26,904 or 27,852 MiB |
| Provenance | Exact command, prompt, seed, arm, config, commit, hashes, and telemetry in `outputs/phase5_generalization_checkpoint_20260813/runs.json` |
| Raw / encoded artifacts | 63 / 63 SHA-256 recorded |

All seven planned categories have nine matched cells (three seeds × three
arms): human→object, animal→vehicle, indoor→outdoor, visually similar semantic
change, visually dissimilar semantic change, same-subject action, and
same-object motion.

## Boundary accounting and mechanical divergence

The actual 285-frame rollout boundaries are RGB frames **46, 94, 142, 190,
238**. The original 49-series estimate was corrected from completed raw output
and policy divergence; it did not alter any generated command or output.

| Arm | First raw RGB divergence from live | Cases |
| --- | --- | --- |
| `live_infinity_rope` | none | 21/21 |
| `always_reset` | frame 46 (first `|`) | 21/21 |
| `native_state_rebinding` | frame 94 (first `#`) | 21/21 |

This verifies activation timing only. It is not an identity, prompt-adherence,
leakage, flicker, or quality measure.

## Blinded review package — scores pending

Reviewer-visible files are in
`outputs/phase5_generalization_checkpoint_20260813/blinded_review/`:

- 21 synchronized, three-arm anonymized videos;
- 105 temporal strips (five boundaries per case), each with two pre-boundary
  frames, the complete first new latent block, and three later blocks;
- `blinded_review_form.csv`: 315 blank arm×boundary rows, 1–5 plus `uncertain`;
- `review_instructions.md`.

The real arm mapping is deliberately segregated at
`blinded_review/private/arm_mapping.json`; it is not shown in review sheets or
videos. No qualitative review has been performed by Codex and no human scores
have been entered, so the requested per-category paired outcomes, collapse
rates, hard-cut leakage, latency distribution, counterexamples, and uncertain
entries are **pending**.

## Interpretation boundary and checkpoint gate

The checkpoint must not be presented as the full 138-run benchmark and cannot
yet test the preregistered semantic success criteria. The correct next action
is blinded score entry followed by descriptive checkpoint analysis, with no
mechanism tuning. No remaining Phase 5 runs launch automatically.

`CHECKPOINT MIXED — REVIEW BEFORE MORE GPU`
