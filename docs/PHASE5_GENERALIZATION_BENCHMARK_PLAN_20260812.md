# Phase 5 Generalization Benchmark — Preregistered Plan

**Status:** design only. Do not run this plan automatically.
**Mechanism:** frozen `Native AR State Invalidation / Rebinding` at
`e556855` / `83bd320`; no new model component, classifier, or coordinate rule.

## Objective and matched sequence

Test generality of the frozen policy, rather than search another retention
variant. Each scenario is a repeated mixed-boundary storyboard:

```text
A1 | A2 # B1 | B2 # C1 | C2
```

`|` requests same entity/environment/style with a changed action or motion.
`#` requests a semantic scene discontinuity. Prompts, durations, resolution,
DMD steps, cache capacity, seed, model, and output indexing are matched across
arms. Record each explicit boundary and its first RGB frame before reviewing.

## Suite, arms, and scale

Create two storyboard scenarios per category (14 total):

1. human → object;
2. animal → vehicle;
3. indoor → outdoor;
4. visually similar scene change;
5. visually dissimilar scene change;
6. same subject / new action;
7. same object / new motion.

Use three fixed seeds per storyboard, after a CPU-only prompt/config audit.
Main matched arms are:

| Arm | `|` | `#` |
| --- | --- | --- |
| Live Infinity-RoPE | Live native retention | Live KV Flush + RoPE Cut |
| Always reset | `transition_no_sink` | `transition_no_sink` + RoPE Cut |
| Frozen policy | Exact live path | `transition_no_sink` + existing RoPE Cut |

This is **14 × 3 × 3 = 126 GPU runs**. On a two-storyboard, three-seed
diagnostic subset, add sink-only and recent-only only at hard cuts and only for
the fixed causal ablation: **2 × 3 × 2 = 12** runs. Total planned ceiling:
**138 runs**. An interim, preregistered checkpoint is 7 categories × one
storyboard × three seeds × three main arms = 63 runs; do not reinterpret it as
the completed benchmark.

## Endpoints

Automate only mechanical provenance and clearly defined pixel-time facts:
boundary frame/block, arm, prompt, seed, command/config hash, output hash,
runtime, peak VRAM, failure, and RGB divergence timing. Use no automated metric
as a proxy for identity, semantic leakage, or video quality without validation.

| Boundary | Blinded human-review fields |
| --- | --- |
| Hard `#` | previous-scene semantic leakage; new-scene prompt adherence; transition latency in RGB frames; flicker/artifact severity; later-scene stability |
| Continuity `|` | subject identity consistency; background consistency; requested action/motion adherence; temporal flicker; motion continuity; collapse/failure rate |

For each case create synchronized, arm-anonymized temporal strips/videos with
two pre-boundary frames, the first new block, and at least three following
blocks. Two independent reviewers score each endpoint on a predefined 1–5
ordinal rubric plus `uncertain`; resolve only score-entry errors, retain both
scores, and report inter-rater agreement. Randomize case and arm order; hide
method names and prior review notes.

## Preregistered success criteria

The frozen policy is promising only if all of the following hold on the full
matrix, with paired seed-level reporting and bootstrap confidence intervals:

1. At hard cuts, it lowers old-scene leakage versus live in at least 70% of
   scored boundaries, with the paired mean leakage difference favoring policy
   and its 95% bootstrap interval excluding zero.
2. At continuity boundaries, it is no worse than live by more than 0.25 mean
   ordinal points on any of identity, background, action, flicker, or motion,
   and has no more than one additional collapse across all continuity cases.
3. Versus always-reset, it reduces continuity collapse/failure by at least 50
   percentage points without a >0.25 adverse mean difference in hard-cut
   latency or later-scene stability.
4. The result is not confined to one visual-similarity regime: neither similar
   nor dissimilar hard cuts may have a policy-favoring leakage rate below 60%.
5. On the ablation subset, neither sink-only nor recent-only matches both the
   policy's hard-cut leakage result and its continuity result. Otherwise the
   full-state explanation is weakened and must be revised.

Report counterexamples, failed generations, reviewers' `uncertain` entries,
and all raw outputs. If any criterion fails, report the failed criterion rather
than selecting a new policy. No Phase 5 result authorizes a learned boundary
classifier, soft decay, retrieval, or a further mechanism search.

## Checkpoint execution record — 2026-08-14

The preregistered **63-cell checkpoint** (one storyboard per category, seeds
101/202/303, three main arms) completed without a GPU-run failure. This does
not execute the remaining 75 planned ceiling cells or the partial-retention
subset. The frozen inference mechanism was unchanged; runner commit was
`eb85a7df609165ac30d8593bb78130144984d27e` and the frozen mechanism remains
`e556855`.

Mechanical correction: the pre-run raw-frame schedule incorrectly assumed
`[49,97,145,193,241]`. The existing independent-first-frame rollout produces
285 RGB frames and recorded divergences establish the actual boundaries as
`[46,94,142,190,238]`. This corrects manifest/review metadata only; prompts,
commands, seed, duration, model, cache behavior, and generated outputs were
not changed or rerun.

Blinded human scores are required before the benchmark's semantic endpoints or
success criteria can be evaluated. See
`docs/PHASE5_CHECKPOINT_RESULTS_20260813.md` for the completed-run ledger and
review-package locations.
