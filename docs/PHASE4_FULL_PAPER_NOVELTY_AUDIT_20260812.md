# Phase 4 — Full-Paper Novelty Audit: Boundary-Conditioned AR State Lifetime

**Date:** 2026-08-12
**Status:** **candidate research contribution; not yet a novelty claim**
**Scope:** literature and frozen-mechanism audit only. No inference code,
dependencies, model settings, or GPU artifacts were changed for this audit.

## 1. Question under audit

Candidate framing:

> **Boundary-Conditioned AR State Lifetime:** a training-free inference policy
> motivated by a causal finding that cached autoregressive state has opposing
> utility across semantic continuation and discontinuity boundaries.

This is not claimed as “clear the KV cache at `#`.” The frozen rule at commit
`e556855` selects between two native Infinity-RoPE paths using an explicit user
annotation:

| Boundary | Frozen behavior | Previous native self-attention state available to first new block? | What establishes future state? |
| --- | --- | --- | --- |
| Normal `|` | Exact live `kv_flush`: transformed sink + two most recent local frames; normal RoPE; existing cross-attention reset | Yes | Ordinary clean pass / rolling cache update |
| Hard `#` | Existing RoPE Cut + verified `transition_no_sink`; no old sink or old local K/V | No | The first new block’s normal clean pass |

This audit asks exactly:

1. Has earlier work distinguished continuation/action boundaries from semantic
   scene-discontinuity boundaries and applied different retention lifetimes to
   native rolling AR self-attention state?
2. Has earlier work used the stricter rule: preserve normal rolling AR state at
   a same-scene/action boundary; at a hard semantic boundary expose zero
   previous-scene self-attention state to the first new block; then establish
   fresh state from that block normally?

## 2. Evidence and interpretation guard

The mechanism is frozen at `e556855`. Phase 1 and Phase 3B are the controlled
hard-cut evidence. The Phase 3C mixed-boundary result is **integrated
feasibility/compatibility evidence**, not a new hard-cut-quality result:

- User review: live and boundary-conditioned are clean/stable at normal `|`;
  always-reset later rainbow/noise-recomposes in A2/B2.
- User review: hard `#` establishment still takes roughly five RGB frames. Do
  not call it instantaneous or claim Phase 3C beats live at every hard cut.
- Therefore, the only supported integration statement is that one explicit
  annotation can preserve the live path for continuity and select a separately
  validated no-old-state path for a hard boundary.

“Observed,” “human visual review,” and “interpretation” are not conflated in
this document. Literature descriptions below state only what primary sources
specify; an omitted implementation detail is marked **not specified**, not
inferred from an abstract.

## 3. Audit method and primary-source set

Sources were original arXiv/CVPR PDFs, official project pages, or official
repositories when cache semantics needed code inspection. Search was expanded
through cited prompt-switch/cache work and exact-mechanism queries: AR-video
prompt switching cache, hard-cut KV cache, semantic-boundary cache reset,
state invalidation, scene-transition forgetting, continuity/discontinuity cache
policy, scene-aware cache lifecycle, prompt-boundary AR state, and rolling-KV
scene reset. This is a high-coverage audit as of 2026-08-12, not a guarantee
that an unpublished or unindexed implementation does not exist.

### Primary sources

1. [Infinity-RoPE, paper](https://arxiv.org/abs/2511.20649), §§4.2–4.3.
2. [Echo-Forcing, paper](https://arxiv.org/abs/2605.16003), §§3.1–3.3 and
   Appendix B.
3. [Anchor Forcing, paper](https://arxiv.org/abs/2603.13405), §§3.2–3.3.
4. [CineWeaver, paper](https://arxiv.org/abs/2607.26529), §§3.2.1–3.2.3.
5. [ShotStream, paper](https://arxiv.org/abs/2603.25746), §3.2.
6. [DySink, paper](https://arxiv.org/abs/2605.21028), §§1–3.
7. [Grounded Forcing, paper](https://arxiv.org/abs/2604.06939), §§4.1–4.3.
8. [MultiShotMaster, paper](https://arxiv.org/abs/2512.03041), §3.2.
9. [LongLive, paper](https://arxiv.org/abs/2509.22622), §3.1.
10. [PaFu-KV, paper](https://arxiv.org/abs/2601.21896), Appendix B.
11. [Astrolabe, official repository](https://github.com/franklinz233/Astrolabe),
    `README.md` scene/prompt-switching section and
    [`pipeline/scene_causal_inference.py`](https://github.com/franklinz233/Astrolabe/blob/main/pipeline/scene_causal_inference.py).

## 4. Direct answers

### 4.1 Boundary-type question

**Yes, broadly.** Echo-Forcing explicitly supports manually tagged smooth,
hard-cut, and recall modes; its Appendix B assigns smooth an offset of zero and
hard cut an offset of 45. Grounded Forcing discusses smooth semantic inheritance
through Asymmetric Proximity Recache (APR) and multi-shot Local Temporal Memory
(LTM) reset for a scene transition. CineWeaver, ShotStream, and MultiShotMaster
also model shot boundaries, though not as native rolling-cache lifetime policies
of the frozen form.

This eliminates any claim to first recognition that continuation and
discontinuity need different treatment.

### 4.2 Strict implementation question

**No exact collision found in the audited primary sources.** The closest
methods do not specify the frozen two-case native-cache access rule:

- Infinity-RoPE’s RoPE Cut retains the global sink: its paper says the new
  segment attends to itself **and the sink** (§4.3). The live `#` path therefore
  does not make all old self-attention state inaccessible.
- Echo-Forcing’s hard mode uses a RoPE offset plus difference-aware decay while
  maintaining hierarchical anchors/scene memories; it is not a zero-old-native-
  state rule.
- Grounded Forcing flushes LTM at a scene transition but explicitly retains its
  Global Consistency Memory (GCM).
- LongLive/Anchor Forcing/PaFu-KV recache old generated visual frames under the
  new prompt to preserve continuity; that is a new-prompt re-encoding path, not
  an inaccessible-old-state hard-cut path.
- The Astrolabe official implementation exposes `|`/`#` syntax, but inspection
  of its scene pipeline shows `_kv_flush` copies the sink plus latest local
  frames even for `#`; its README’s shorthand “KV-cache is flushed” is not a
  full previous-state exclusion.
- CineWeaver’s boundary masking is a training-free full-sequence DiT operation,
  not a rolling native AR K/V reset. ShotStream keeps a global context cache at
  every shot boundary; MultiShotMaster is trained full multi-shot generation.

This is a narrow mechanistic difference. It is not evidence that the frozen rule
is non-obvious enough for a paper without direct comparative evaluation.

## 5. Mechanism-level collision table

“Inaccessible” means no previous-scene native self-attention state can be read
by the first new block. “Fresh state” means the paper specifies post-cut state
being built from the new block rather than merely continuing a retained/cache-
recached state.

| Work / backbone | AR and training | Boundary trigger / types | Normal prompt/action behavior | Hard-cut behavior | Sink / recent K/V / inaccessible? / fresh state | Position / memory | Closest source statement and intended failure | Collision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Infinity-RoPE** / Self-Forcing, Wan2.1-1.3B | Causal AR; training-free wrapper | Prompt action control; RoPE Cut for discontinuity. Paper distinguishes KV Flush and Cut, not a documented semantic classifier. | KV Flush retains global sink and last generated latent frame (live checkout uses sink + two local frames). | RoPE Cut reindexes the cut block; paper says new segment attends to itself **and sink**. | Sink retained; recent state suppressed/rolled; **not inaccessible** because sink remains; normal cache updates continue. | Block-Relativistic RoPE; no retrieval. | §§4.2–4.3: “global sink latent frame and the last generated latent frame”; Cut “only to itself and the sink.” Failure: prompt inertia / inability to make cuts. | **STRONG PARTIAL** — direct base and delimiter/RoPE-cut precedent, but not zero-old-state at `#`. |
| **Echo-Forcing** / AR streaming backbones | Causal AR; training-free | Explicit tags: smooth `[s]`, hard `[s#]`, recall `[s@]`; optional prompt-similarity routing. | Smooth: continuous offset 0; hierarchical anchors + compressed history + recent window; discrepancy decay retains compatible content. | Hard: offset 45; difference-aware decay after first clean new block; historical anchors/scene recall memory remain managed. | Anchors/recent windows persist; old tokens decay selectively; **not inaccessible**; new clean block is a decay reference, not documented fresh-only cache establishment. | Relative RoPE, scene recall frames, decay/routing. | Appendix B Eq. 17–18; §3.3 says preserve consistent regions and suppress changed regions. Failure: entangled historical KV/background contamination. | **STRONG PARTIAL** — explicit smooth/hard types and cache lifecycle, but materially richer memory/decay and no strict exclusion. |
| **Grounded Forcing** / Wan2.1-1.3B | Causal AR; trained in two stages | Prompt switches use APR; scene transition may be trigger token or drastic prompt shift. Two semantic regimes are described. | APR interpolates old/new recached K/V by temporal proximity to preserve inheritance. | LTM reset is used for multi-shot transition; GCM is explicitly retained for identity. | Local recent LTM flushed; GCM anchors persist; **not inaccessible**; new local state rolls after reset. | GCM at phase 0, LTM relative RoPE; dual memory + APR. | §4.2: “selectively flush ... LTM while retaining GCM”; §4.3 Eq. 11–12. Failure: temporal contamination versus semantic inheritance. | **STRONG PARTIAL** — closest conceptual type-aware lifetime policy, but its retained GCM directly differs from zero previous native state. |
| **Anchor Forcing** / CausVid, Wan2.1 | Causal AR; trained/distilled RoPE re-alignment | Every interactive prompt switch is treated by anchor-guided recache; no hard/normal type split specified. | Recompute local cache under new prompt while using sink + junction anchor memory. | No separate semantic hard-cut rule specified. | Fixed sink, local recent, and junction KV remain accessible; **not inaccessible**; recache builds prompt-updated local state. | Tri-region RoPE; anchor/junction memory. | §3.2 Eq. 6 and anchor-guided recache; failure: recache loses boundary cues / quality. | **STRONG PARTIAL** — native AR prompt-switch cache management, but opposite retention goal and no boundary-type rule. |
| **LongLive** / causal frame-level AR | Causal AR; trained (streaming long tuning) | Prompt switch, one KV-recache step at each boundary; no hard/normal distinction. | Re-encodes generated visual prefix with new prompt to retain motion/appearance while removing old prompt semantics. | No separate hard-cut zero-state policy specified. | First-frame sink and recent window remain semantic visual context through recache; **not inaccessible**; refreshed cache continues normally. | Frame sink + short window; no retrieval. | §3.1: recompute cache using generated frames + next prompt. Failure: retention gives prompt lag; clearing breaks continuity. | **STRONG PARTIAL** — canonical prompt-switch cache tradeoff, but recache rather than type-conditioned access lifetime. |
| **PaFu-KV** / AR diffusion backbone | Causal AR; trained salience head | Prompt switch handled by one KV recache; no type split specified. | Same recache principle: old frames as visual context with new prompt. | No hard rule specified. | Salience-selected cache / frame sink; prior visuals remain through recache; **not inaccessible**; refreshed cache continues. | Salience memory policy; no scene-type routing. | Appendix B, “KV Re-caching.” Failure: prompt inertia versus discontinuity. | **STRONG PARTIAL** — adjacent cache-policy work, but trained salience and no semantic two-case rule. |
| **ShotStream** / Wan2.1 | Causal next-shot AR; trained bidirectional teacher then DMD causal student | Every shot boundary; no continuation-vs-hard cache lifetime choice. | Global cache holds sparse conditional frames; local cache keeps current-shot generated frames. | Uses discrete RoPE jump at every shot; global cache is still queried. | Global cache retained; local cache is shot-local; **not inaccessible**; next shot uses both. | Dual cache + RoPE discontinuity; no retrieval at boundary. | §3.2: global inter-shot cache and local intra-shot cache. Failure: inter-shot consistency / AR error accumulation. | **STRONG PARTIAL** — causal multi-shot dual cache, but trained and preserves global state across all shots. |
| **CineWeaver** / pretrained DiT | Bidirectional/full-sequence diffusion; training-free | Explicit shots and transition frames; no rolling AR prompt-boundary policy. | Per-shot prompts; non-transition tokens can share global video context. | Gap-frame RoPE and mask isolate each shot’s first transition frames; VAE decoding state resets per shot. | No native rolling sink/recent cache; transition tokens block cross-shot self-attention, but this is **not** a rolling K/V exclusion rule; per-shot VAE state re-establishes. | Gap RoPE, masked self-attention, shot references. | §§3.2.1–3.2.3. Failure: pretrained temporal-continuity bias and decoder leakage. | **STRONG PARTIAL** conceptually; architecture and state object differ. |
| **MultiShotMaster** / pretrained T2V | Bidirectional/full multi-shot; trained | Shot boundaries in annotated multi-shot sequences. | Full inter-shot video attention supports global consistency. | Per-shot Narrative RoPE phase shift marks boundaries. | No rolling sink/local K/V policy; **not applicable** to native state inaccessibility; no fresh rolling state operation. | Multi-shot Narrative RoPE and reference-token injection. | §3.2 Eq. 2. Failure: ambiguity between within-shot and across-shot frames. | **WEAK** — boundary positional method, different architecture/training. |
| **DySink** / AR long-video backbones | Causal AR; trained weights used for allocations | No prompt/action or hard-scene boundary policy. | N/A. | N/A. | Replaces static early frame sinks with retrieved dynamic sinks; local window persists; no state exclusion rule. | Retrieval bank + sink anomaly gate. | §§1–3. Failure: obsolete/static sink collapse. | **WEAK** — sink treatment only, no boundary types. |
| **Astrolabe** / Self-Forcing, LongLive, Causal Forcing, Krea | Causal AR; RL-trained adapters | Official repo accepts `|` and `#`; README calls `#` a hard-cut cache flush. | `|` rolls cache forward. | Code’s `_kv_flush(scene_cut=True)` still copies sink plus last two local frames, then enables cut coordinates. | Sink + recent K/V remain; **not inaccessible**; ordinary cache update follows. | Existing scene-cut RoPE; no special memory rule. | Official README “Soft vs. hard transitions”; scene pipeline lines 59–67. Failure: not its paper’s central mechanism. | **STRONG PARTIAL implementation collision** — same UI-level distinction, but code disproves an exact zero-state collision. |

## 6. Reviewer attack and response

### Attack: Infinity-RoPE

> “Infinity-RoPE already marks hard cuts and flushes cache. This work merely
> changes the number of retained frames based on an existing delimiter.”

**Assessment:** partly undefeated. Infinity-RoPE is the direct base; KV Flush,
RoPE Cut, `#`, and cache retention are already its language. The narrow factual
response is that its Cut preserves a sink, whereas frozen hard mode prevents the
first new block from reading any previous-scene self-attention K/V, and the
paper’s contribution would be the causal opposing-utility evidence plus an
integrated policy, not the delimiter or reset concept. This only survives if
evaluation directly demonstrates that exact access distinction against the
native cut/cache baselines. Without that, the reviewer is right.

### Attack: Echo-Forcing

> “Echo-Forcing already classifies smooth versus hard transitions and manages
> old scene memory differently. Boundary-conditioned lifetime is a simplified
> special case.”

**Assessment:** strong. Echo-Forcing defeats any claim to the general idea of
type-aware smooth/hard cache behavior or scene-aware forgetting. The response is
only that Echo keeps a structured/decayed historical memory and its hard mode is
not the strict native-state blackout. A paper must compare this claim at the
mechanism level; otherwise “simplified special case” is a likely verdict.

### Attack: Anchor Forcing / LongLive

> “Prompt-switch cache state is already the problem setting. Existing recache
> methods solve prompt inertia versus continuity; this is another cache update.”

**Assessment:** strong. The distinction is not prompt-switch caching in general,
but using *explicit semantic boundary type* to choose retain versus strict
inaccessibility instead of re-encoding the old visual prefix under a new prompt.
The paper needs matched recache comparisons and a clear explanation why its
state-access intervention differs from recontextualizing old frames.

### Attack: ShotStream

> “ShotStream already has a global cache for inter-shot consistency, a local
> cache for intra-shot coherence, and a RoPE boundary signal.”

**Assessment:** partially defeated. ShotStream is causal and multi-shot but
trained, and it always retains global cross-shot cache; it does not specify the
normal-retain/hard-zero-native-state rule. Still, it makes a broad “dual-
timescale state” or “shot-aware AR cache” claim untenable.

### Attack: CineWeaver

> “CineWeaver already breaks continuity only at shot boundaries with positional
and attention isolation, then resets decoding state; this is the same cinematic
goal.”

**Assessment:** partially defeated. CineWeaver is a compelling training-free
shot-boundary baseline, but it operates in a bidirectional full-sequence DiT
with masked attention/gap frames and per-shot VAE reset, not native rolling AR
self-attention-state lifetime. Do not claim the cinematic goal is new; only the
native-AR access policy may remain distinct.

## 7. Surviving claim, exclusions, and decision

### A. Strongest surviving novelty claim

For a fixed causal video-diffusion rollout, we identify and test an opposing-
utility property of *native rolling self-attention state*: retaining the complete
live state supports same-scene action continuation, whereas making all
previous-scene native self-attention K/V inaccessible provides the cleanest
tested hard semantic cut. We operationalize that causal result as a minimal,
training-free, explicitly annotated boundary-conditioned lifetime policy that
leaves the continuation path unchanged and lets the first hard-cut block build
fresh ordinary state. This is a candidate claim contingent on direct
mechanism-level comparisons, not a first claim.

### B. What is not novel

- Hard cuts, prompt switching, or explicit `#`-style boundary notation.
- KV flush/recache, cache reset, scene forgetting, stale-sink mitigation, or
  rolling-cache management in AR video generally.
- Training-free multi-shot generation, shot-aware masking, gap/phase-shifted
  RoPE, or scene-local/shot-local positional treatment.
- Smooth-versus-hard transition types or automatic routing between them.
- Anchor/global/local memory decomposition, retrieval, entity memory, decay,
  or any automatic semantic-boundary classifier.
- A claim that sink-only or recent-only is a usable operating point.

### C. Closest prior work (ranked)

1. **Echo-Forcing — STRONG PARTIAL.** It explicitly distinguishes smooth,
   hard, and recall modes and has scene-memory/forgetting logic.
2. **Infinity-RoPE — STRONG PARTIAL / direct base.** It already supplies KV
   Flush, RoPE Cut, hard-cut syntax, and sink-retaining Cut semantics.
3. **Grounded Forcing — STRONG PARTIAL.** It combines smooth prompt
   inheritance with local-memory reset at a scene transition, but keeps global
   semantic memory.

LongLive/Anchor Forcing are immediately next for prompt-switch recache;
ShotStream is the closest trained causal multi-shot cache architecture.

### D. Novelty verdict

`PROMISING BUT NEEDS MECHANISM REFRAMING`

No audited source specifies the exact frozen strict state-access rule, but too
many close works already cover the broader story. A paper should not sell a
generic boundary-aware cache lifecycle. It should test whether strict
previous-native-state inaccessibility is a necessary and useful primitive beyond
Infinity-RoPE Cut, Echo’s hard mode, Grounded’s LTM reset, and recache. If that
comparison fails, pivot before more mechanism work.

### E. Next experiment recommendation (only if proceeding past the gate)

Run a preregistered **mechanism-discrimination evaluation**, not a new method:
use matched continuation and hard-boundary prompt suites, multiple seeds, and
the frozen policy against (i) live Infinity-RoPE Cut/flush, (ii) strict
no-old-state, and faithful feasible reproductions/configurations of the closest
type-aware baselines. Predefine human review for continuity, old-scene leakage,
new-scene adherence, first stable new-scene frame, and artifacts. The critical
ablation is the retained old sink at `#`; it directly tests the only narrow
distinction that currently survives. Do not run this automatically.

## 8. Narrow object-level reframe (supersedes only the candidate framing)

The broad verdict in §7 is preserved: a generic transition-type-aware cache
policy collides with Echo-Forcing, Grounded Forcing, and Infinity-RoPE. The
surviving candidate is instead **Native AR State Invalidation / Rebinding**:
at a continuation label preserve the live native rolling self-attention K/V;
at a semantic discontinuity label make every previous-scene native K/V entry
inaccessible to the first new causal block; let that block's normal clean pass
establish fresh native state; then resume ordinary AR.

The strict object-level audit is in
`docs/NATIVE_AR_STATE_REBINDING_MECHANISM_SPEC_20260812.md`. It is deliberately
not a claim about an auxiliary/global memory: Grounded Forcing retains GCM,
ShotStream retains global cache, Echo-Forcing retains managed anchors/history,
and Infinity-RoPE retains a sink after its native Cut. No audited source states
the strict zero-old-native-state rule, but this absence is not a claim of
firstness.

### Revised candidate verdict

`MECHANISM CLAIM SURVIVES — READY FOR GENERALIZATION BENCHMARK`

This is a conditional research decision, not an award of novelty. The sole
recommended next step is the preregistered frozen-mechanism generalization
matrix in `docs/PHASE5_GENERALIZATION_BENCHMARK_PLAN_20260812.md`; it must
demonstrate the opposing-utility result beyond the existing cases or the claim
reverts to `PROMISING BUT NEEDS MECHANISM REFRAMING`.
