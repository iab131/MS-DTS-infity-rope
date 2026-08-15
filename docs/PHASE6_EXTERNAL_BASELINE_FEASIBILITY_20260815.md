# Phase 6 — External Baseline Feasibility (2026-08-15)

Status: **no GPU work; frozen inference mechanism unchanged.** This audit uses
only original paper PDFs and official repositories.

## Decision

Echo-Forcing is the closest *candidate*: its official README names Wan2.1-
T2V-1.3B and `gdhe17/Self-Forcing`'s DMD checkpoint, while explicitly
supporting smooth `[TIME]` and hard `[TIME#]` transitions. However, the same
official repository says its code was temporarily withdrawn on 2026-07-24. No
released implementation remains to pin its cache operations, memory budget,
timing, or feature toggles. It is therefore **not fairly runnable today**.

Grounded Forcing is not a fair direct baseline: it introduces trained Dual
Memory, Dual-Reference RoPE, and APR components. Anchor Forcing is operational
in principle, but uses a separately trained checkpoint, anchor-guided recache,
and RoPE-realignment distillation. It is an external quality baseline only with
a clearly labelled training confound, not a mechanism-discrimination baseline.

## Echo-Forcing feasibility audit

| Question | Official-source finding | Consequence |
|---|---|---|
| Same/directly compatible backbone? | Official README requests Wan2.1-T2V-1.3B and the public Self-Forcing DMD checkpoint, the same model family/checkpoint used here. | Theoretically direct. |
| Smooth and hard modes? | Explicit tags are `[TIME]` smooth, `[TIME#]` hard, and `[TIME@]` recall. Appendix B defines smooth/hard/recall and hard offset 45. | Yes at syntax/method level. |
| State after hard cut? | Echo retains structured anchors, compressed history, and a recent window. Appendix C.4 says old tokens are preserved after a transition, then an initial clean new-scene block is generated and used for token-wise decay. | Old K/V are accessible during initial B generation; not zero-old-state. |
| Extra components? | Hierarchical Temporal Memory, Scene Recall Frames, Difference-aware Memory Decay, and optional routing/recall. | More transition state/processing than frozen rebinding; no reference-image input is specified. |
| Same `A1 | A2 # B1 | B2`, seed, resolution, duration? | README accepts `|` and duration tags, but public examples use 10-second segments and implementation is withdrawn. | Prompt syntax adapts; exact timing, RNG, resolution, and cache settings cannot be verified or matched. |

The smallest unavoidable mismatch is the missing official executable source.
Recreating Echo from the paper would be an unofficial new method, not a
baseline.

Sources: [Echo-Forcing paper](https://arxiv.org/abs/2605.16003) and
[official repository](https://github.com/mingqiangWu/Echo-Forcing). The
repository's withdrawn-code notice and required Wan/Self-Forcing assets are
decisive; Appendix B Eq. 17--18 and Appendix C.4 Eq. 19--27 establish the
hard-mode offset and post-clean soft-decay order.

## Grounded Forcing feasibility audit

| Question | Official-source finding | Consequence |
|---|---|---|
| Backbone/resolution | Wan2.1-T2V-1.3B, 832×480, 16 FPS. | Surface-level compatibility only. |
| Smooth/hard support | APR targets smooth prompt switching; a trigger token or drastic shift causes multi-shot LTM reset. | Relevant regimes, but not the same explicit `|`/`#` interface. |
| State after scene transition | §4.2 flushes LTM but explicitly retains GCM at temporal coordinate zero. | First new scene sees old global tokens; not zero prior state. |
| Training/components | Two-stage training: 1,200 short-clip steps, then 1,000 dynamic-switch long-sequence steps, each on 32 H20 GPUs; Dual Memory, DR-RoPE, APR. | Not training-free or matched. |
| Same storyboard/settings? | No official implementation or released checkpoint was identified in the paper's official materials; native clips are five seconds. | Conceptual comparison only. |

Source: [Grounded Forcing paper](https://arxiv.org/abs/2604.06939),
§4.1--4.3 and §5.1. Missing public code is reported as unavailability, not
proof that no private code exists.

## Ranked external candidates

1. **Echo-Forcing — strongest would-be direct candidate.** Same stated
   Wan/Self-Forcing assets and training-free status, but official source is
   withdrawn.
2. **Anchor Forcing — runnable quality baseline, not a clean mechanism
   comparator.** Its official repo releases code/weights, but documents an
   independent Torch 2.6/CUDA 12.4 environment, trained anchor recache, and
   RoPE-realignment distillation.
3. **Grounded Forcing — conceptual comparator only.** It is closest to a dual
   semantic/dynamics explanation but adds trained components and retains GCM.

Anchor sources: [paper](https://arxiv.org/abs/2603.13405),
[official repository](https://github.com/vivoCameraResearch/Anchor-Forcing).

## Conditional 18-run matrix (not executable now)

If official Echo source returns and a future audit verifies the same checkpoint,
resolution, DMD schedule, duration mapping, and explicit smooth/hard controls,
the minimal matrix is three cases (a Phase-1 stale-background case, visually
similar cut, train/boat motion case) × seeds 101/202 × live Infinity-RoPE,
frozen rebinding, Echo = **18 runs**. Review `|` continuity and `#` leakage,
B establishment, RGB-frame latency, and later stability with synchronized
three-arm videos and compact notes. Do not add always-reset or a large form.

## Novelty consequence if a direct Echo comparison becomes possible

| Result | Consequence |
|---|---|
| Ours approximately equals Echo | Likely too incremental unless state size, runtime, cache complexity, or latency is measurably better. |
| Ours is worse | Reassess the paper direction before more evaluation. |
| Ours is better | Broaden the comparison and quantify leakage, continuity failure, latency, runtime, and state complexity. |
| Similar quality, simpler state handling | Potentially useful only with quantified native-cache entries, added memory, operations, training requirement, and establishment latency. |

Current conclusion: Echo is a **strong partial novelty collision** but not an
executable fair baseline while official code is withdrawn. Grounded and Anchor
cannot replace it for mechanism discrimination without a substantial trained-
method confound.

NO FAIR DIRECT BASELINE — PAPER COMPARISON MUST BE MECHANISM-LEVEL
