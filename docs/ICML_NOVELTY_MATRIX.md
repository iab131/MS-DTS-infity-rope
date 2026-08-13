# ICML Novelty Collision Matrix

**Status (2026-08-12):** a prior-art guard, not a novelty assertion. “Scene
Epoch” and “Scene-Time Field” are internal hypothesis labels only. No claim is
permitted until a live Infinity-RoPE intervention, multi-pair/seed evidence,
and a full paper-level literature review establish a distinct contribution.

| Work | Hard-cut / boundary mechanism | Sink and cache lifetime / ownership | Global vs local temporal coordinates | Prompt switching / forgetting / establishment | Historical or entity memory | Training requirement | Phase-0 collision risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Infinity-RoPE](https://arxiv.org/abs/2511.20649) | KV Flush plus RoPE Cut in one AR rollout | Persistent global sink plus recent cache; live checkout currently keeps sink + two recent frames in `kv_flush` | Block-Relativistic RoPE and RoPE Cut | Prompt changes trigger flush; `#` enables a coordinate discontinuity | No entity-memory method in the base mechanism | Training-free inference framework | Direct baseline and strongest collision. A boundary-state policy is not novel merely because it is expressed as an epoch. |
| [Echo-Forcing](https://arxiv.org/abs/2605.16003) | Supports hard cuts and transitions through structured scene memory | Hierarchical anchors, compressed history, recent window, adaptive decay | Relative RoPE under bounded memory | Explicit old/new scene discrepancy and forgetting | Spatially structured scene recall frames | Reported training-free | Strong collision on scene memory lifecycle, forgetting, and interactive prompt-switch state. |
| [DySink](https://arxiv.org/abs/2605.21028) | Addresses obsolete context/sink collapse rather than the current fixed hard-cut ablation | Retrieval-selected dynamic frame sinks plus anomaly gate | Discusses RoPE phase re-alignment; no verified scene-local epoch claim here | Adapts old context to current visual state | Compact retrieval bank used as dynamic sinks | Training requirement not asserted here without full-paper verification | Strong collision on replacing static sink ownership; Phase 0 must distinguish a semantic-boundary reset from dynamic sink selection. |
| [Anchor Forcing](https://arxiv.org/abs/2603.13405) | Re-cache at prompt switches with anchor warm start | Anchor caches retain semantic state and recent cues | Tri-region RoPE with region-specific origins | Interactive prompt switching and post-switch stabilization | Anchor memory | Includes RoPE re-alignment distillation | Strong collision on prompt-switch cache ownership and local reference origins. An inference-only scene-local time claim requires clear separation. |
| [EM-Vid](https://arxiv.org/abs/2605.23610) | Multi-shot script, not the present AR cache-cut operator | Entity-indexed latent-patch bank, sparse conditioning, budgeted updates | Stores patches at original positions; no claim of AR scene epochs | Shot-specific text with leakage mitigation | Explicit entity latent memory | Method reported training-free; base uses StoryMem LoRA | Directly rules out claiming the exploratory raw-KV/latent branch as entity-memory novelty. |
| [CineWeaver](https://arxiv.org/abs/2607.26529) | Gap frames, masked transition frames, shot-wise attention/FFN, independent shot VAE decoding | Anchor memory for global appearance; not one rolling AR cache owner | Shot-aware positional gaps; shot-local attention partitions | Shot-wise cross-attention and explicit transition isolation | Reference tokens and anchor memory | Training-free | Strong collision on breaking temporal continuity and shot-local isolation, though its bidirectional multi-shot construction differs from live AR state. |
| [Prompt Relay](https://arxiv.org/abs/2604.10030) | Soft temporal prompt-boundary decay; no AR cache reset | Does not own a self-attention sink/cache policy | Frame-time windows tied to prompt spans | Cross-attention temporal routing; adjacent prompts overlap at boundaries | None | Training-free | Collision on temporal prompt assignment only; it does not establish scene-local AR state semantics. |
| [SwitchCraft](https://arxiv.org/abs/2602.23956) | Event windows with early cross-attention control; no cache hard cut | No sink/cache ownership mechanism | Event-to-frame time allocation | Event-aligned query steering and strength balancing | None | Training-free | Collision on event grounding and prompt switching, not AR cache epochs. Do not add query steering to this line of work. |

## Strict comparison conclusions

1. **Hard cuts:** Infinity-RoPE and CineWeaver already make hard boundaries an
   explicit inference-time concern. The current `transition_no_sink` result is
   a live mechanism audit/ablation, not a new hard-cut concept.
2. **Sinks and cache ownership:** Echo-Forcing, DySink, and Anchor Forcing make
   cache/sink policy a primary contribution. “Scene-local sink ownership” is
   therefore high-risk until its behavior and distinction are demonstrated.
3. **Time coordinates:** MS-DTS proposed global plus shot-local time;
   Infinity-RoPE has block-relative coordinates plus RoPE Cut; Anchor Forcing
   uses region-specific origins; CineWeaver changes positional/attention
   structure. A “Scene-Time Field” cannot be called novel from the present
   evidence.
4. **Prompt switching and forgetting:** Prompt Relay/SwitchCraft control prompt
   alignment, while Echo-Forcing explicitly handles forgetting. Phase 0 must
   isolate self-attention AR state, not import routing or steering.
5. **Entity memory:** EM-Vid and related work cover entity-centric memory. The
   closed raw-KV/latent explorations are negative representation evidence and
   do not support an entity-memory contribution.
6. **Training:** most listed systems report inference-time components, but
   Anchor Forcing includes a distillation component. Any future claim needs an
   exact training/inference comparison and matched backbone constraints.

**Current evidence boundary:** user manual review supersedes the weaker
Phase-3B claim that either partial factor is a usable continuity solution.
Full retained state is best in the reviewed same-scene cases; full removal is
best at hard cuts; partial retention flashes/resets or remains contaminated.
The Phase-3C annotated `|`/`#` switch is a positive four-case integrated
demonstration of choosing existing live paths, not a new cache-ownership or
boundary-classification method. The listed work still overlaps substantially
with cache ownership, forgetting, prompt switching, and boundaries. This is
insufficient for an ICML novelty claim or any automatic-policy claim.

## 2026-08-12 novelty-gate supersession

The Phase-3C mixed-boundary review is **integrated feasibility/compatibility
evidence**, not a universal hard-cut visual win: hard `#` establishment still
takes roughly five RGB frames, and the strong cleanup evidence remains the
dedicated Phase-1/3B matrices. At normal `|`, live and boundary-conditioned are
clean/stable; always-reset later rainbow/noise-recomposes.

The full primary-source audit finds that Echo-Forcing already exposes explicit
smooth/hard transition types, while Grounded Forcing has smooth prompt
inheritance and a scene-transition local-memory reset. Neither audited source
was found to specify the exact frozen rule of preserving native rolling state at
normal action boundaries while making **all** previous native self-attention
state inaccessible at hard boundaries and then establishing it from the first
new block. This is a narrow distinction, not a first-claim basis. Current
status: **candidate research contribution; not yet a novelty claim**;
`PROMISING BUT NEEDS MECHANISM REFRAMING`. See
`docs/PHASE4_FULL_PAPER_NOVELTY_AUDIT_20260812.md`.
