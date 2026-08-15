# Paper Outline — Native AR State Rebinding

**Working status:** candidate research contribution; not a novelty claim. The mechanism is frozen at `e556855`.

## 1. Problem

Autoregressive video rollouts must cross opposite storyboard events: a continuation wants state to remain usable, while a semantic scene discontinuity can make prior native state visually inappropriate. The paper asks whether this opposing utility can be demonstrated for the *native rolling self-attention state* of a causal video backbone, without introducing an auxiliary memory system.

## 2. Empirical discovery

Controlled hard-cut studies found stale visual source-scene retention with full state, while complete no-old-native-state reset was the cleanest tested hard-cut intervention. Same-scene studies found the opposite failure: unconditional removal caused rainbow/noise recomposition; partial sink/recent-state conditions were unstable compromises. These are observations in one Infinity-RoPE/Self-Forcing setting, not universal backbone claims.

## 3. Native AR state validity hypothesis

Native rolling self-attention K/V has a validity domain defined here by the explicit storyboard boundary label. At `|`, the live state remains valid; at `#`, all prior-scene native K/V is inaccessible to the first new causal block. That block runs normally, its clean pass establishes fresh native state, and ordinary rollout resumes. No learned boundary function, retrieval, semantic-memory decomposition, scene-local RoPE, or cache blending is claimed.

## 4. Method

```text
if boundary == "|":
    preserve the unmodified live Infinity-RoPE transition path
if boundary == "#":
    apply verified transition_no_sink with existing RoPE Cut
    expose zero prior-scene native self-attention K/V to the first new block
    let the normal clean pass establish new native state
    resume ordinary rolling autoregression
```

Global output indexing, prompt schedule, cross-attention behavior, DMD calls, weights, cache capacity, and normal RoPE behavior are unchanged.

## 5. Controlled mechanism ablations

| Study | Result / role |
| --- | --- |
| Hard-cut state-retention factorial | Full retained state best preserves same-scene continuity; complete removal is the cleanest tested hard-cut arm; partial sink-only/recent-only retention is unstable. |
| Phase 2B local RoPE epoch | Numerically active but visually neutral; positional re-origin is excluded as the explanation. |
| Raw-KV / latent identity work | Exploratory representation work; closed, not part of the method. |
| Fresh-scene prime / offset control | Prime did not clearly improve motion; offset-only was unstable; closed. |
| Early native handoff | Infeasible under the no-custom-cache constraint; no GPU result. |
| Standalone-B motion control | Mixed/inconclusive; do not attribute train/boat/smoke oddness specifically to rebinding. |

## 6. Generalization evaluation

The Phase-5 checkpoint contains seven categories, one `A1 | A2 # B1 | B2 # C1 | C2` storyboard/category, three seeds, and live / always-reset / rebinding arms (63 completed videos). Its primary endpoint is blinded human review; only 60 explicit user scores are currently available, with later notes carried qualitatively. Phase 7 adds objective *failure-mode* proxies, not generic quality.

| Boundary event | Live | Always reset | Rebinding |
| --- | --- | --- | --- |
| Continuation `|` | user-reviewed usable/stable; collapse proxy 0.778 | collapse proxy 2.340; 63/63 high-discontinuity flags | first `|` bit-identical to live; proxy 0.743 |
| Hard cut `#` | source-reference AUC 0.523; stale-context counterexamples exist | AUC 0.078 but interpretation is confounded by earlier `|` collapse | AUC 0.159; lower than live in 40/42 paired cuts, with two reversals |

The automatic late-B1 target-stability latency proxy was non-discriminative. Human review remains necessary for B-prompt adherence, transition latency, motion, and artifacts.

## 7. Closest prior work / mechanism comparison

Compare state objects, not broad labels: immediately after a hard boundary, which native self-attention K/V, sink, recent local state, and auxiliary/global memory remain visible to the first new causal block? The existing novelty audit identifies conceptual overlap with Infinity-RoPE, Echo-Forcing, Grounded Forcing, Anchor Forcing, and ShotStream. The external-baseline feasibility audit concludes that no fair directly runnable baseline is available without mismatched trained components/frameworks; comparison is mechanism-level unless that changes.

## 8. Limitations

- One backbone/checkpoint and one storyboard/category; 63 runs are a checkpoint, not the full preregistered benchmark.
- Boundary labels are authored, not automatically inferred.
- DINO source-reference similarity is a validated-in-subset visual proxy, not a semantic, identity, or text-adherence metric.
- No locally cached image/text evaluator was available without downloading weights; prompt adherence remains visual review.
- Hard scene establishment is not instantaneous; mixed-boundary review observed roughly five RGB frames.
- Visually similar cuts contain explicit counterexamples, and motion effects are unresolved/backbone/prompt dependent.
- This does not establish novelty over all transition-aware or memory-based video systems.

## 9. Claims we explicitly do NOT make

- first scene-aware cache, cache reset, semantic forgetting, hard-cut method, training-free multi-shot method, prompt-switch policy, stale-sink method, or automatic semantic-boundary detector;
- generic video-quality, identity, motion, or prompt-adherence improvement;
- instantaneous hard cuts, universal hard-cut success, or a useful partial sink-only/recent-only operating point;
- a Scene-Time Field, scene-local RoPE advantage, learned validity function, retrieval/memory method, or auxiliary/global-memory policy.
