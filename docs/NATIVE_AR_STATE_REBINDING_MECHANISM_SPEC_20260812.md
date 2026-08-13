# Native AR State Invalidation / Rebinding — Frozen Mechanism Spec

**Status:** candidate research contribution; not yet a novelty claim.
**Frozen implementation:** `e556855` / `83bd320`. This is a specification of
the existing opt-in policy, not authorization for a new inference mechanism.

## State object and explicit labels

`native_state` means the backbone's native rolling *self-attention* K/V at
every layer: its transformed sink and its recent/local entries. It does not
include text conditioning, cross-attention, model weights, VAE latents, or an
auxiliary/global/retrieval memory. Boundary labels are storyboard inputs; there
is no learned validity predictor.

| Label | Validity decision | First new block | State afterward |
| --- | --- | --- | --- |
| Continuity `|` | Previous native state remains valid. | Preserve the exact live Infinity-RoPE `kv_flush` path (sink + two recent local frames). | Ordinary clean-cache update and rolling AR continue. |
| Semantic discontinuity `#` | Previous-scene native state is invalid. | Make **every** old native K/V inaccessible, run the block normally with existing RoPE Cut. | Its normal timestep-0 clean pass establishes fresh native state; ordinary rolling AR resumes. |

```text
native_state_valid(boundary_label):
    if boundary_label == CONTINUITY:
        preserve_live_native_state()          # exactly live kv_flush
        generate_next_block_normally()
        establish_or_update_native_state_normally()
    elif boundary_label == SEMANTIC_DISCONTINUITY:
        invalidate_previous_native_state_for_first_block()
        generate_next_block_normally()        # existing RoPE Cut remains active
        establish_fresh_native_state_normally()# normal clean cache pass
        resume_normal_rolling_autoregression()
```

The invalidation is neither gradual decay nor retrieval. It does not retain an
auxiliary/global scene memory, rebase RoPE coordinates, or change the denoising
schedule. The visible first hard-cut block is generated normally; its clean
pass, rather than a custom writer, is the rebinding event.

## Object-level hard-boundary comparison

“Zero” below has a strict meaning: no previous-scene **native rolling
self-attention** entry is readable by the first new causal block. A missing
implementation detail is not converted into a claim of zero.

| Work | Native K/V at hard boundary | Sink / local entries | Auxiliary or historical state | Zero old native K/V to first new block? | Fresh state establishment | Training | Primary source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Infinity-RoPE | Cut segment attends to itself and global sink. | Sink retained; paper flush retains sink + last latent (live path has sink + two local). | None specified. | **No.** Sink remains readable. | Ordinary subsequent cache update. | Training-free wrapper. | [§§4.2–4.3](https://arxiv.org/abs/2511.20649) / live `kv_flush`. |
| Echo-Forcing | Hard mode changes temporal offset and applies difference-aware decay. | Recent window/anchors are managed, not specified as fully inaccessible. | Hierarchical anchors, compressed historical scene memory, recall. | **No evidence of zero; treat as no.** | First clean block is a decay reference, not stated as a fresh-only native cache. | Training-free. | [§3.3, App. B Eq. 17–18](https://arxiv.org/abs/2605.16003). |
| Grounded Forcing | Scene transition flushes Local Temporal Memory. | Local LTM removed. | Global Consistency Memory explicitly retained. | **No.** Old GCM remains visible. | New LTM rolls after reset. | Two-stage trained. | [§4.2; §4.3 Eq. 11–12](https://arxiv.org/abs/2604.06939). |
| Anchor Forcing | No distinct hard-boundary type is specified; recache uses anchor/junction/local state. | Sink, local, and junction cache remain accessible. | Anchor/junction memory. | **No.** | Prompt-updated recache. | Trained/distilled. | [§3.2 Eq. 4–6](https://arxiv.org/abs/2603.13405). |
| ShotStream | Every shot uses global context plus shot-local cache. | Local is shot-local; global cache remains queried. | Sparse global conditional-frame cache. | **No.** Global state remains readable. | Next-shot local cache forms during rollout. | Bidirectional teacher + DMD causal student. | [§3.2](https://arxiv.org/abs/2603.25746). |

**Collision decision:** no audited primary source executes the exact strict
operation “first new causal block sees zero accessible previous native K/V,
then its ordinary clean pass establishes a new native rolling state.” This is
not a first claim: Echo-Forcing and Grounded Forcing directly collide with the
broader idea of type-aware transition memory, while Infinity-RoPE is the direct
base and already supplies `#`, KV Flush, and RoPE Cut.

## Frozen causal evidence

### Observed

- **Hard cuts (Phase 1/3B):** retained previous AR state produced old-scene
  contamination in the reviewed matrix; complete no-old-state reset was the
  most consistently clean tested hard-cut arm. User review explicitly found
  sink-only/recent-only contamination or instability in key cases.
- **Continuity (Phase 3A/3B):** full live state was the strongest tested
  same-scene condition. Complete removal caused repeated rainbow/noise
  recomposition; partial state caused resets, flashes, ghosting, or deformation.
- **RoPE (Phase 2B):** a coherent local temporal epoch was numerically active
  but visually neutral. Positional re-origin is not the supported explanation.
- **Phase 3C:** compatibility only. `|` preserved live behavior, and `#`
  selected the no-old-state path in the same rollout. New-scene establishment
  still took roughly five RGB frames; it was not instantaneous.

### Interpretation

The tested native state has opposite utility across the two explicitly labeled
boundary regimes. This does **not** assign independent semantic roles to sink
or local entries, prove universal generality, establish an optimal transition,
or show an automatic boundary detector.

### Candidate claim

For the native rolling self-attention state of a causal video model, explicit
semantic discontinuities may require complete state invalidation followed by
ordinary state rebinding, whereas explicit continuations retain the unchanged
live state. The claim is conditional on the frozen model and awaits the
preregistered generalization benchmark.

## Strongest reviewer attack and bounded response

> “This merely changes Infinity-RoPE's cache flush at an already explicit `#`
> delimiter. Echo-Forcing and Grounded Forcing already use different
> transition-memory behavior; calling the old cache ‘invalid’ adds no method.”

That objection defeats a broad boundary-policy claim. The bounded scientific
response is empirical, not terminological: the factorial showed full retained
native state as the strongest tested continuity condition, full removal as the
most consistently clean hard-cut condition, and partial sink/recent retention
as an unstable compromise. Unconditional reset fails continuity. The frozen
hard branch makes zero prior **native** K/V readable to the first causal block,
then uses its normal clean pass to establish a replacement state; this differs
from retaining an auxiliary/global memory while resetting a local store. The
response remains incremental unless the preregistered benchmark replicates both
sides across categories and directly rules out a useful partial-retention arm.
