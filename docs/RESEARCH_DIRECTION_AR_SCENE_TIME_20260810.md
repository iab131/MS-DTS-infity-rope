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

The replicated core effect is now narrower: retained previous-scene AR state
causes hard-cut contamination in the tested matrix, while
`transition_no_sink` gives the most consistently clean new-scene
establishment. The matrix gives 0/8 unambiguous sustained recent-KV-only
*entity* reductions. This motivates testing state lifetime at semantic
boundaries rather than claiming monotonic entity retention or trying to
recover identity from contextualized historical representations. The earlier
8/8 sink-specific wording is explicitly superseded below by user manual review
and the Phase-3B factorial.

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

**Superseded Codex review:** this earlier text attributed all 8/8 retained-arm
contamination to the sink. The user’s later manual review found sink-only clean
for fish→train and astronaut→fox. Treat the retained-sink-specific statement
as obsolete; the controlled factorial below separates sink from recent state.

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

## Phase 2B: minimal scene-local RoPE epoch (executed)

This was a deliberately narrow opt-in coordinate test, not a general
Scene-Epoch framework. It activates only for an `#` hard cut with the live
`transition_no_sink` policy. Global frame/block/output bookkeeping, prompt
switching, cache capacity, cross-attention reset, DMD calls, seed, and model
weights remain unchanged. The first B block uses self-attention temporal
phases `[0,1,2]`; later B blocks continue `[3,4,5]`, `[6,7,8]`, …, including
re-RoPE of raw non-sink K and phase-zero storage of the new B transformed
sink. With the flag off, the existing transition path is unchanged.

The registered 2 pairs × 2 seeds × 2 arms matrix completed 8/8 GPU runs:
greenhouse-woman→desert-pickup and aquarium-fish→snowy-locomotive, seeds
101/202. Each run used the same no-old-state reset and differed only in this
self-attention coordinate rule. The exact commands, provenance, runtime
(44.393--48.224 s/run), and 23,176 MiB direct-process peak VRAM are in
`outputs/hard_cut_scene_local_rope_epoch_phase2b_20260810/runs.json`.

**Observed:** A outputs are bit-identical. Raw decoded tensors first differ at
RGB frame 34 in all four cases (frame 33 is equal), and the differences grow
in later B blocks; therefore this is a behaviorally exercised positional
intervention, not an inactive flag. The deterministic attention tests already
showed first-B coordinate equivalence within numerical tolerance and confirmed
the phase-zero B sink and coherent later local Q/K coordinates.

**Human visual review:** The shared first-B dissolve is visually unchanged.
In all four pair×seed cases, both arms reach the requested pickup/desert or
locomotive/snow scene with no reviewed sustained A-background leakage. Across
B2/B3 and later samples, neither arm shows a consistent advantage in prompt
adherence, entity quality, spatial coherence, temporal coherence, or obvious
artifacts. The raw divergence is visually minor and directionally inconsistent.
Synchronized two-arm videos, temporal sheets, and the four-case summary are
under `outputs/hard_cut_scene_local_rope_epoch_phase2b_20260810/comparison/`.

**Decision — NEUTRAL:** the coherent local epoch is mathematically/numerically
distinct from live `transition_no_sink`, but is not behaviorally useful in
this small controlled matrix. Do not promote local RoPE-origin reset to an
independent contribution or build Scene-Time around it. This does not rule out
all scene-local state designs; it closes this isolated coordinate-only branch.

## Phase 3A: same-scene action boundary (executed)

This is the complementary experiment to the hard-cut matrix. It does **not**
revive Scene-Time or add a new policy. The live `transition_no_sink` policy
already applies at every `|` boundary; when the boundary has no `#`, it clears
old self-attention state with `scene_cut=False`, so normal RoPE behavior is
retained. The policy log confirms B's first context is only current frames
9--11 at normal positions `[0,1,2]`; no RoPE Cut or epoch flag was active.

The matched matrix compares live `kv_flush` (sink + two recent frames) against
this normal-boundary `transition_no_sink` intervention for greenhouse woman
turn→wave and desert pickup drive→stop, at seeds 101/202. All **8/8** GPU runs
completed (42.432--48.105 s/run; 22,964 MiB live and 23,176 MiB no-sink peak
VRAM). Exact commands and outputs are in
`outputs/same_scene_action_transition_phase3a_20260811/runs.json`.

**Observed:** both requested A scenes are visibly present before the boundary
in every case (recognizable woman/greenhouse and pickup/desert). A RGB tensors
are exactly equal across arms; every pair first differs at RGB frame 34, with
frame 33 equal. The no-sink policy logs independently confirm zero retained
non-sink frames, excluded old sink, `scene_cut=false`, and no local epoch.

**Human visual review:** live retention preserves subject/entity appearance,
environment, and coherent later motion in all 4 pair×seed cases. The woman
performs the requested wave, and the pickup remains a stable same-scene B
vehicle; the short generated window makes drive→stop adherence a qualitative,
not measured, judgment. In contrast, no-sink produces a boundary-adjacent
noise/recomposition collapse from frame 34 onward in all four cases: identity,
scene continuity, usable action evidence, and later stability are lost. This
is much stronger than the brief hard-cut dissolve, and no meaningful action
comparison remains after collapse.

**Decision — POSITIVE TRADEOFF:** within this tested regime, retained AR state
is valuable for same-scene action continuity (4/4), while Phase 1 found old
  retained previous-scene state harmful for semantic hard cuts. This supports a
*boundary-conditioned AR-state lifetime effect* as empirical motivation only;
it is not a new boundary-aware policy, causal mechanism proof, or novelty
claim. The catastrophic normal-boundary no-sink result may reflect removal of
state the rollout requires, not a uniquely semantic role for the sink. Stop
  after Phase 3A: no classifier, soft decay, routing, memory, or Phase 3B.

## Phase 3B: sink × recent-local-state factorial (executed)

### User manual review supersession of Phase 1

This is a separate human review record, not a silent edit of the earlier Codex
review. The user found: fish→train has live/sink+1 composites but clean
sink-only and no-sink; girl→pickup has no-sink as the only clearly clean
desert; chef→boat has no-sink clean while retained arms preserve chef or form
composites; astronaut→fox has clean sink-only and no-sink while stronger
retention is worse. Thus the durable Phase-1 claim is retained
previous-scene **AR state** contamination and complete no-old-state reset as
the most consistently clean intervention—not universal sink causality.

### Minimal intervention and execution

`recent_only_no_sink` is the sole new opt-in mode. At the transition it copies
the two latest raw non-sink frames into the compact prefix, excludes the old
transformed sink, and uses their live-compaction temporal treatment
(`{1,2,45,46,47}` at a hard cut; `{1,2,3,4,5}` normally). At the clean pass it
replaces that temporary prefix with the first B block’s normal new sink/cache.
Cross-attention reset, prompt schedule, RoPE Cut, DMD schedule, capacity,
weights, seed, and all memory features remain unchanged.

The exact-provenance harness reused 12 Phase-1 hard-cut controls and ran only
four new recent-only cells; it reused 8 Phase-3A same-scene controls and ran
eight new sink-only/recent-only cells. All **12/12** fresh GPU runs completed
(hard 44.524--48.594 s; same-scene 44.453--45.889 s). Ledgers and synchronized
four-arm comparisons are in
`outputs/hard_cut_state_retention_factorial_phase3b_20260811/` and
`outputs/same_scene_state_retention_factorial_phase3b_20260811/`.

### Human visual review and factorial outcome

| Boundary | Sink + recent2 | Sink only | Recent2 only | Neither |
| --- | --- | --- | --- | --- |
| Hard cut | Source/B composite in the reviewed greenhouse and aquarium cases | Clean train in fish→train, but greenhouse/girl contamination remains in girl→pickup | Source/B composite in all four pair×seed cases | Most consistently clean B scene |
| Same-scene action | Usable continuity | Usable continuity in all four cases | Usable continuity in all four cases | Catastrophic colored-noise/recomposition collapse in all four cases |

**Answers in this matrix:** (1) sink is not necessary for hard-cut
contamination; (2) two recent raw frames alone are sufficient for hard-cut
contamination in these four cells; (3) sink alone is sufficient for usable
same-scene continuity here; (4) recent frames alone are also sufficient here;
(5) continuity does not require their combination. These are controlled visual
results, not semantic metrics or a causal claim beyond the tested state groups.

**Decision:** complete state removal cleanly establishes a new semantic scene,
but is unsafe at same-scene action boundaries. The data support a
boundary-conditioned state-lifetime *effect* and reject universal
sink-specific causality. Stop after Phase 3B; no automatic policy, classifier,
memory, routing, decay, or additional phase is implemented.

## Phase 3B user-review supersession and Phase 3C integrated policy (executed)

### Superseding user review of the factorial

This user review supersedes the weaker earlier wording that sink-only and
recent-only are usable continuity solutions. On hard cuts, fish→train retains
fish/aquarium-rock content for live and sink-only; recent-only reduces/loses
the fish but is blurry and flashes; no-sink is the clean stable train. For
girl→pickup, only no-sink clearly removes the greenhouse; live, sink-only, and
recent-only remain contaminated to different degrees. On same-scene actions,
live is best/stable, no-sink catastrophically rainbow/noise-recomposes, and
both partial arms reset/flash; in the woman case they also ghost/malform hands
and appearance.

The current interpretation is therefore: full retained AR state is the best
tested same-scene continuity condition; full state removal is the best tested
hard semantic cut; partial retention is an unstable compromise. It can keep
enough old semantics to contaminate a cut while losing enough state to flash,
reset, ghost, or deform. There is no evidence for a useful sink-only or
recent-only operating point, and component-retention sweeps are closed.

### Minimal opt-in policy

`--boundary-conditioned-ar-state` only connects existing transition paths. At
a normal `|`, it calls the live `kv_flush` exactly (transformed sink plus the
two latest local frames, normal RoPE, cross-attention reset). At a hard `#`, it
calls the verified `transition_no_sink` path (zero old self-attention K/V,
RoPE Cut, cross-attention reset). The first hard-cut clean pass establishes
the new scene state normally. No classifier, new RoPE coordinates, memory,
routing, decay, latent intervention, or changed generation setting is added.

CPU checks prove the `|` helper is exactly equal to live `kv_flush`, the `#`
helper is exactly equal to `transition_no_sink` with RoPE Cut, and the disabled
path remains the existing live path. The per-run policy logs confirm 6-frame /
9,360-token live contexts at blocks 4 and 10, and a current-only 3-frame /
4,680-token RoPE-Cut context at block 7.

### Mixed-boundary evaluation: A1 | A2 # B1 | B2

The preregistered 2 scenarios × 2 seeds × 3 arms matrix completed **12/12**
GPU runs: greenhouse woman→desert pickup and autumn fox→snowy locomotive;
seeds 101/202; arms live `kv_flush`, `always_reset`, and
`boundary_conditioned`. All runs exit zero (48.556--56.905 s/run; 615.081 s
total). Direct-process peak VRAM is 22,964 MiB for live/conditioned and
23,176 MiB for always-reset. Commands, prompt schedules, output paths, and
metadata are in
`outputs/mixed_boundary_state_lifetime_phase3c_20260812/runs.json`.

**Observed:** `always_reset` first diverges from live at RGB frame 34 in all
four cases, immediately after the first normal boundary. The conditioned arm
is bit-identical to live through frame 69 and first diverges at frame 70,
immediately after the `#` boundary, in all four cases. Thus the policy is
inactive through A1→A2 and active precisely at A2→B1.

**Human visual review:** at the first `|`, conditioned preserves the stable
woman/greenhouse or fox/forest action transition seen in live; always-reset
instead shows the known colored-noise/recomposition collapse. At `#`, live
shows a source/new-scene composite (woman/greenhouse with pickup or
fox/forest with locomotive), while conditioned reaches the clean desert pickup
or snowy locomotive after the shared first-block dissolve. At B1→B2,
conditioned retains a stable B subject and B scene; always-reset again flashes
or recomposes. Synchronized three-arm videos, four temporal sheets, and an
all-case sheet are under
`outputs/mixed_boundary_state_lifetime_phase3c_20260812/comparison/`.

**Interpretation — positive integrated demonstration:** in these two
mixed-boundary scenarios, the explicit boundary label selects the empirically
preferred existing state lifetime: live retention where continuity is
requested, and no-old-state reset where a semantic cut is requested. This is
not an automatic semantic-boundary method, general policy claim, causal proof
of a particular cache component, or novelty claim. No Phase 3D/classifier or
additional mechanism is authorized by this result.

## Phase 3C personal visual-review supersession (2026-08-12)

This append-only personal review supersedes any stronger Codex-only qualitative
wording above. At normal `|` boundaries, live Infinity-RoPE is clean/stable and
boundary-conditioned is clean/stable, as it should be because it preserves the
live path. `always_reset` later collapses into rainbow/noise during A2/B2.

At hard `#`, visual establishment still takes roughly five RGB frames. Do not
describe Phase 3C as an instantaneous cut, or as showing clear hard-cut visual
superiority over live in every reviewed mixed-boundary case. The strongest
hard-cut-cleanup evidence remains the dedicated Phase-1 and Phase-3B matrices.

The correct Phase-3C status is **integrated feasibility/compatibility evidence**:
one explicit boundary annotation can retain the native live path where
continuity is requested and select the separately validated no-old-state path
where a hard cut is requested. It is not a new visual hard-cut result by itself.

The mechanism is frozen at `e556855` pending the novelty gate in
`docs/BOUNDARY_CONDITIONED_AR_STATE_NOVELTY_GATE_20260812.md` and the
primary-source audit in `docs/PHASE4_FULL_PAPER_NOVELTY_AUDIT_20260812.md`.
