# Phase 0: AR Scene Boundaries as State and Time Boundaries

**Status:** research direction and benchmark preparation only. No new scene-time
mechanism, GPU run, or novelty claim is made here.

## Origin and constrained conceptual transfer

The original MS-DTS-RoPE plan sought training-free multi-prompt video control
by separating a global sequence-time coordinate from a shot-local coordinate.
Its proposed Shot-Time Field assigned each latent frame to a shot and used
that assignment to coordinate temporal RoPE, prompt routing, steering, memory,
and soft/hard transitions. The transferable *question* is whether one rollout
needs both global continuity and a local scene coordinate. The concept does
not transfer as an implementation: MS-DTS used a different Wan checkout and
proposal stack (prompt routing, query steering, latent blending, and memory),
none of which is enabled or validated in this Infinity-RoPE checkout.

Infinity-RoPE already supplies a different basis: a globally continuous AR
rollout, a bounded self-attention cache, KV Flush, Block-Relativistic RoPE,
and RoPE Cut. Phase 0 therefore studies the live state boundary before
implementing any Shot-Time Field analogue.

## Why the direction changed

### Observed evidence

- In the executed woman-greenhouse → blue-pickup-desert retention ablation,
  recent old-scene non-sink KV (`sink+2`/`sink+1`) coincided with visible
  old-person carry-over. `sink_only` removed the persistent visible person
  after its first blended B frame but retained greenhouse-like background.
  `transition_no_sink` was the only reviewed arm that became truck/desert-only
  from decoded frame 34. This is one prompt, one seed, and qualitative review.
- All arms first diverged at the intended first B block / raw frame 33. Thus
  the intervention is target-causal, but it is not a matched-token comparison:
  the four first-B contexts have 6/5/4/3 frames.
- The raw-KV branch was strong and source-specific but scene-entangled. Layer,
  lifetime, spatial mask, erosion, alpha, denoising-time, compact pooled-token,
  and clean-pass variants did not produce clean identity recall.
- The latent-subject branch showed that a masked latent patch can carry useful
  appearance without immediate full-frame greenhouse reconstruction, but it
  produced geometric paste artifacts. Writing that patch into the AR clean
  cache persisted both A1-like subject attributes and local source context.
  Cache-write erosion reduced, but did not remove, that local contamination.

### Interpretation, not a result claim

The replicated core effect is now narrower and stronger: a retained sink is
linked to stale previous-scene/background retention (8/8 cases), while
`transition_no_sink` gives the cleanest new-scene establishment (8/8). The
matrix gives 0/8 unambiguous sustained recent-KV-only entity reductions. This
motivates treating a semantic cut as an explicit state/time boundary rather
than claiming monotonic recent-KV entity retention or trying to recover
identity from contextualized historical representations.

## Live transition audit (code authoritative on 2026-08-10)

The parser creates a scene boundary for every `|`; a boundary enters
`scene_cut_boundaries` only when the preceding segment contains `#`. The block
loop changes `conditional_dict` at every scene boundary. It performs KV Flush
or the memory-policy transition at every boundary, while the RoPE Cut flag is
true only for `#` boundaries.

| Item | Live behavior |
| --- | --- |
| No-policy `kv_flush` | Runs at every `|` boundary when `--attention-memory-policy` is absent. It resets cross-attention, retains cache slot 0, copies the last **two** cached latent frames into slots 1--2, and sets `local_end_index=3*1560=4680`. Thus this checkout's no-policy first-B context is sink + two recent frames, not the paper-level “sink + last frame” description. `global_end_index` is not reset. |
| Policy transition | With `--attention-memory-policy`, `apply_memory_transition` replaces `kv_flush`. `sink+2`, `sink+1`, and `sink_only` retain respectively two, one, or zero old non-sink frames after the physical sink; the configured decay is applied only when enabled. The executed basic effect disabled decay/retrieval/archive/consolidation. |
| `transition_no_sink` | At a cut, it sets `local_end_index=0`, so the first B attention call has no accessible old K/V (no sink and no old local frames). The subsequent DMD and clean-cache writes start at slot zero and overwrite the old physical sink. After the clean pass, metadata records `persistent_sink_frame_id=current_start_frame`; this is a label, not a separate cache owner. |
| Sink representation | The cache stores raw K/V initially, but when `local_start_index==0` it writes the first frame's **RoPE-transformed K** back to cache slot zero. Its V stays raw. On later calls, the stored K in slot zero is copied directly into attention rather than re-RoPEd; non-sink K is raw and re-RoPEd on use. |
| Standard sink at a cut | `kv_flush` preserves the existing transformed K slot zero and its V, then copies latest raw K/V into slots 1--2. Policy `sink_*` modes similarly preserve physical slot zero while compacting selected old non-sink entries. |
| Cross-attention | No-policy `kv_flush` always sets every `crossattn_cache[i]["is_init"] = False`. Policy transitions do the same when `--memory-crossattn-reset` is enabled (default and used by the prior ablation). |
| RoPE Cut | `scene_cut=True` is set only for the first block after a `#` boundary and is cleared before the next block. For a three-frame block, `rope_cut` gives current queries and current raw keys temporal coordinates `[45,46,47]`; retained raw local keys occupy their compact local prefix, while a preserved transformed sink has no newly assigned coordinate. |
| First block after a cut | For the four policy arms its logged logical first-B contexts are: `sink+2` `[sink, old7, old8, B9..11]`; `sink+1` `[sink, old8, B9..11]`; `sink_only` `[sink, B9..11]`; `transition_no_sink` `[B9..11]`. The actual current tensors undergo all four DMD calls plus the timestep-zero clean cache pass with the same cut flag. |
| After the first B block | The loop clears `scene_cut`. `global_end_index` and `current_start_frame` stay global. Under `transition_no_sink`, B frame 9 now physically occupies slot zero with cut-phase transformed K and raw V; the following block uses normal cache rotation/relative RoPE. There is no live scene-local cache object and no explicit scene-local temporal origin. |

This supersedes earlier wording that `transition_no_sink` left the old sink
unmutated. It first excludes the old state from attention, then normal cache
writes replace slot zero with the new scene. It also resolves a second conflict:
the live no-policy `kv_flush` retains two recent frames, whereas the original
Infinity-RoPE paper describes retaining one last frame.

## Scene-Epoch / Scene-Time hypothesis

**Hypothesis, not implementation:** a globally continuous Infinity-RoPE
rollout may benefit from scene-local AR epochs. At a hard semantic boundary,
the new scene would own its sink/cache state and a local temporal origin while
global rollout accounting remains continuous. The first new-scene block would
establish that state before later blocks reuse it. Phase 0 does not establish
whether this should be a new cache layout, an RoPE coordinate rule, or only an
evaluation description.

## What cannot be claimed

- No Scene-Epoch, Scene-Time Field, or scene-local cache ownership method has
  been implemented or evaluated.
- No novelty claim is justified; nearby work already changes sinks, cache
  memory, RoPE regions, temporal prompt allocation, and multi-shot boundaries.
- The existing result does not establish sink causality, general hard-cut
  quality, semantic scene establishment, long-horizon benefits, or superiority
  over Infinity-RoPE/Echo-Forcing/DySink/Anchor Forcing.
- Raw-KV and latent results are exploratory evidence against the tested direct
  identity-recall representations, not a benchmark of entity-memory methods.

## Current basic effect (consolidated)

| Category | Record |
| --- | --- |
| Observed | One matched woman→truck prompt at seed 101: clean block 3 is identical across arms; first B divergence is block 4/raw frame 33; first-B contexts are 6/5/4/3 frames for `sink+2`/`sink+1`/`sink_only`/`transition_no_sink`. |
| Human visual observation | `sink+2` and `sink+1` retained head-in-windshield plus greenhouse; `sink_only` retained greenhouse after its first blended frame; `transition_no_sink` showed a residual first blend then truck/desert without person or greenhouse in sampled frames from 34 onward. |
| Interpretation | Recent non-sink state is a strong candidate contributor to old-entity carry-over; retaining the sink is compatible with old-background persistence; a one-block no-old-context reset is the cleanest tested semantic cut. |
| Replication required | Four strongly different A→B pairs × two seeds × four live arms, fixed configuration, per-run metadata, frame-level review, and failure reporting. This is sufficient for a Phase-1 benchmark decision, not a method paper claim. |

## Phase-1 hard-cut benchmark (executed; no new method)

`docs/HARD_CUT_BENCHMARK_PHASE0_20260810.json` specified four strongly
different A→B pairs and seeds 101/202. The explicitly authorized serial
`--execute` run completed the exact matrix below with no model-code change.
Its ledger contains every command/configuration, prompt, seed, arm, transition
block/frame, runtime, PID-scoped VRAM, exit status, and output path:
`outputs/hard_cut_transition_phase1_20260810/runs.json`.

| Arm | Live path isolated |
| --- | --- |
| `live_kv_flush` | No-policy baseline: `kv_flush`, cross-attention reset, retained physical sink + two latest old frames in this checkout. |
| `sink_plus1` | Policy transition: old sink + one latest old frame. |
| `sink_only` | Policy transition: old sink only. |
| `transition_no_sink` | Policy transition: no accessible old state in first B block; B establishes slot-zero state through normal writes. |

The matrix is **4 pairs × 2 seeds × 4 arms = 32 GPU runs**. It completed
32/32 with return code zero: 42.308--48.726 s/run, 22,964--23,176 MiB direct
process peak VRAM, and no missing output folders. The visual endpoint is the
four-arm synchronized transition evidence, not an automated semantic metric.
Artifacts are eight videos, eight sheets, and an all-case summary under
`outputs/hard_cut_transition_phase1_20260810/comparison/`.

### Phase-1 result and Phase-2 gate

**Observed:** every retained-sink arm preserves old scene/background semantics
into later B blocks in all 8 pair×seed cases; `transition_no_sink` has the
cleanest B scene in all 8. **Human review:** all arms briefly dissolve at the
boundary, but only retained-sink arms stabilize into an old/new composite.
**Interpretation:** this supports stale previous-scene AR state as a
reproducible hard-cut failure mode. It does *not* establish a clean monotonic
recent-KV-only entity effect: reducing recent retention gave 0/8 unambiguous
sustained entity-leakage reductions, and sink-only still retained aquarium fish
and all four source environments.

The smallest justified next test is **not run**: `transition_no_sink` versus
that same reset plus a scene-local temporal/RoPE epoch, for greenhouse→pickup
and aquarium→locomotive at seeds 101/202 (8 runs). It requires an invariant
audit that global accounting, cache ownership, prompt conditioning, and all
other behavior are unchanged; only the post-cut temporal coordinate rule may
differ. Phase 1 does not justify calling this a novel Scene-Epoch method.

## Phase 2A: scene-local RoPE equivalence audit (CPU-only; no GPU run)

The proposed scene-local epoch is **not mathematically redundant** with live
`transition_no_sink`. This conclusion comes from the exact live cache path and
the deterministic float64 probe at
`outputs/hard_cut_transition_phase2a_20260810/scene_local_rope_probe.json`;
it is not a visual or model-quality result.

| Live no-sink stage | Current temporal phases | Hypothetical scene-local phases | Consequence |
| --- | --- | --- | --- |
| First B DMD calls | Q and current raw K use RoPE Cut `[45,46,47]` | Q and K `[0,1,2]` | A common temporal translation: standard RoPE relative logits are invariant. |
| First B clean cache pass | Same `[45,46,47]`; raw K/V are re-written, then cache slot zero retains only first-frame **transformed K** at 45, V raw | Same write but transformed sink K at 0 | The persisted sink differs even though first-block attention is invariant. |
| Second B | `scene_cut=False`; Q `[12,13,14]`; K `[45,1,2,3,4,5]` (special sink, then compact raw local K) | Q `[3,4,5]`; K `[0,1,2,3,4,5]` | This is not one common offset over the attended Q/K pairs. |
| Third B | Q `[15,16,17]`; K `[45,1,2,3,4,5,6,7,8]` | Q `[6,7,8]`; K `[0,1,2,3,4,5,6,7,8]` | The mismatch continues as B grows. |

The probe holds raw Q/K/V/cache tensors identical and changes only these
coordinates. It reports the following max/mean absolute errors (synthetic
float64 RoPE/attention; every live layer shares this same positional/cache
path, but these are not 30 learned-layer output measurements):

| Path | Logit max / mean | Attention-output max / mean |
| --- | --- | --- |
| First B DMD | 4.44e-16 / 7.63e-17 | 2.22e-16 / 3.61e-17 |
| First B clean pass | 4.44e-16 / 7.63e-17 | 2.22e-16 / 3.61e-17 |
| Clean-pass transformed sink K | 1.45866 / 0.21141 | n/a |
| Second B, full context | 2.36073 / 0.54294 | 0.94993 / 0.19853 |
| Second B, raw non-sink K only | 2.36073 / 0.58578 | 1.02461 / 0.20711 |
| Third B, full context | 2.95903 / 0.72675 | 1.21477 / 0.26518 |

Thus special transformed-sink handling breaks a common phase translation into
future blocks, and the raw non-sink-only row proves it is not the sole cause:
later live queries remain globally indexed while raw cached K is compactly
re-RoPEd. A literal common temporal translation of *every* attended Q and K
would preserve RoPE logits, but that is not the live no-sink implementation or
the proposed `[0,1,2]`, `[3,4,5]`, … scene epoch. The previously proposed
8-run comparison remains a real, smallest GPU comparison; it is not run here.

## Prior-art guard

The collision matrix is maintained in `docs/ICML_NOVELTY_MATRIX.md`. Primary
references: [Infinity-RoPE](https://arxiv.org/abs/2511.20649),
[Echo-Forcing](https://arxiv.org/abs/2605.16003),
[DySink](https://arxiv.org/abs/2605.21028),
[Anchor Forcing](https://arxiv.org/abs/2603.13405),
[EM-Vid](https://arxiv.org/abs/2605.23610),
[CineWeaver](https://arxiv.org/abs/2607.26529),
[Prompt Relay](https://arxiv.org/abs/2604.10030), and
[SwitchCraft](https://arxiv.org/abs/2602.23956).
