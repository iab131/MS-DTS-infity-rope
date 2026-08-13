# Boundary-Conditioned AR State Novelty Gate — 2026-08-12

## Status

**Candidate research contribution; not yet a novelty claim.**

Candidate framing (now narrowed): **Native AR State Invalidation / Rebinding**
is a training-free inference policy motivated by a causal finding that the
*native rolling self-attention state* has opposing utility across explicitly
labeled semantic continuation and discontinuity boundaries.

This is not reducible to “clear the KV cache at `#`.” The candidate is the
controlled selection between two native rolling-AR-state lifetimes at an
explicit boundary type: retain the live state where continuity is requested;
make previous-scene self-attention state inaccessible only where a hard
semantic discontinuity is requested; then let the new block establish ordinary
new state. Whether that distinction is sufficiently non-obvious and not already
claimed by prior work is the purpose of this gate.

## Frozen mechanism and evidence boundary

The mechanism is frozen at `e556855`. Normal `|` calls the unmodified live
Infinity-RoPE `kv_flush` path (sink plus two local frames); hard `#` calls the
already-verified `transition_no_sink` path with the existing RoPE Cut. Nothing
here authorizes a classifier, memory, routing, new RoPE, decay, or another
retention variant.

The dedicated Phase-1/3B hard-cut matrices are the strongest cleanup evidence.
The mixed Phase-3C matrix is only integrated feasibility/compatibility evidence:
it shows that the explicit labels can retain the live path at `|` while choosing
the previously tested no-old-state path at `#`. It does not establish an
instantaneous cut or hard-cut superiority over live in every mixed case.

## Gate questions

1. **Boundary-type question.** Has prior work distinguished continuation-style
   prompt/action boundaries from semantic scene-discontinuity boundaries and
   assigned different lifetimes to native rolling AR self-attention state?
2. **Exact-rule question.** Has prior work used this rule: preserve normal
   rolling AR state at a same-scene/action boundary; at a hard semantic boundary
   expose zero previous-scene self-attention state to the first new block; then
   establish fresh state from that block normally?

The full primary-source audit is
`docs/PHASE4_FULL_PAPER_NOVELTY_AUDIT_20260812.md`.

## Initial broad gate outcome (preserved)

The first question is **yes in a broad sense**: Echo-Forcing explicitly labels
smooth versus hard transitions, and Grounded Forcing separately treats smooth
prompt inheritance and multi-shot local reset. The exact-rule question is **no
exact collision found in the audited primary sources**, with an important
qualification: the closest methods retain an anchor/global memory, use
recache/interpolation, or manipulate full-sequence attention rather than remove
all prior native self-attention state. This leaves a narrow implementation-level
distinction, not a “first boundary-aware cache policy” claim.

**Verdict: `PROMISING BUT NEEDS MECHANISM REFRAMING`.** Any paper must lead
with the causal opposing-utility result and the strict native-state access rule,
then directly compare against the closest type-aware alternatives. It must not
claim first scene-aware cache, first hard cut, first forgetting, first
training-free multi-shot generation, first stale-sink solution, first scene-local
RoPE, or automatic semantic-boundary detection.

## Object-level reframing and current gate outcome (2026-08-12)

The prior verdict is preserved above. The reframe does not claim that
transition-type-aware cache handling is new: Echo-Forcing and Grounded Forcing
already make transition-type/memory decisions, and Infinity-RoPE already has
KV Flush plus RoPE Cut. The candidate object is narrower: **native rolling
self-attention K/V access at the first new causal block**.

At `|`, the frozen policy preserves the backbone's live native state unchanged.
At `#`, every previous-scene native self-attention K/V entry is inaccessible to
the first new block; that block runs normally and its ordinary clean pass binds
fresh native state. This is not decay, retrieval, an auxiliary/global-memory
reset, scene-local RoPE, or automatic boundary classification.

The object-level primary-source table in
`docs/NATIVE_AR_STATE_REBINDING_MECHANISM_SPEC_20260812.md` finds no audited
method that specifies this exact zero-previous-native-state first-block rule:
Infinity-RoPE retains its sink; Echo-Forcing manages anchors/history; Grounded
Forcing retains GCM; Anchor Forcing retains anchor/junction state; and
ShotStream retains global cache. This is not proof of firstness, only the
absence of a direct collision in the audited sources.

**Current verdict: `MECHANISM CLAIM SURVIVES — READY FOR GENERALIZATION
BENCHMARK`.** It remains a candidate paper contribution, not a novelty claim.
The preregistered benchmark in
`docs/PHASE5_GENERALIZATION_BENCHMARK_PLAN_20260812.md` must test it before
stronger novelty language is used.
