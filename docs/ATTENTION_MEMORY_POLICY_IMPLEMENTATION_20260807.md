# Attention-as-Memory-Policy Implementation Scaffold

Status: implemented for deterministic/unit validation only; **no new GPU
generation has been run**. This is an experimental ablation scaffold, not
evidence that descriptor routing improves identity consistency.

## Opt-in configuration

`--attention-memory-policy` enables the system. It is mutually exclusive with
the existing `--noncontiguous-kv` Phase 1 oracle path so oracle results remain
reproducible. Component switches are:

| Component | Switches | Default under policy |
| --- | --- | --- |
| Retrieval | `--[no-]memory-retrieval`, `--memory-k`, `--memory-manual-frame-ids` | on, k=5 |
| Routing descriptors | `--memory-descriptor-layers` | `0,1,5,14,16` |
| Injection layers | `--memory-injection-layers` | `0,1,5,14,16` |
| Context | `--memory-context-mode {replace_recent,prepend}` | `prepend` |
| Local transition | `--memory-local-retention {sink_only,sink+1,sink+2}`, `--[no-]memory-crossattn-reset` | `sink_only`, reset |
| Fixed decay | `--[no-]memory-decay`, `--memory-decay-beta` | on, 0.3 |
| Archive | `--[no-]memory-archive`, `--memory-archive-top-m`, archive retention flags | on, top-M=3 |
| Consolidation | `--[no-]memory-consolidation`, `--memory-consolidate-n-max`, `--memory-target-budget`, `--memory-diversity-threshold` | on, 200 -> 150, 0.9 |
| Logging | `--memory-policy-log` | `<output_folder>/memory_policy.jsonl` |

`--[no-]memory-transition-auto-retrieval` defaults to disabled. Automatic
visual routing at a first-new-scene block is experimental; manual IDs remain
available there and can be limited by `--memory-manual-target-blocks`.

The manual list is ordered and overrides similarity routing; automatic routing
uses mean-pooled raw-K cosine similarity over the selected descriptor layers.
Similarity (clamped at zero) updates selected-frame utility. This deliberately
does not request FlashAttention weights.

## Data flow

For a normal block, the last raw non-sink local K yields a query descriptor.
The router excludes the sink and active local frame IDs, selects MemoryStore
entries, adds their similarity to utility, packs only selected CPU K/V, and
passes those transiently to each attention layer. The clean pass then clones
new raw K/V to CPU, derives one descriptor per configured layer and writes a
per-frame `{scene_id, frame_id, descriptors, utility, layers}` record. `layers`
contains only configured injection layers. No
MemoryStore entry is written during denoising passes.

At a prompt boundary, SceneArchive optionally utility-compresses top-M frames
from the completed scene, then the policy resets cross attention (unless
disabled), applies the configured local retention and optional fixed decay,
and retains the existing `scene_cut` RoPE signal. Consolidation runs after a
write only when `N_max` is exceeded: high-utility entries are retained first,
then descriptor-diverse candidates fill the target budget.

`replace_recent` keeps the six-frame Phase 1 budget and can replace only the
two live non-sink local slots. `prepend` makes `[sink, retrieved, local,
current]` and expands its attention span. Both preserve the transformed sink
in slot zero. Prepend assigns retrieved positions after the sink and shifts
raw local/current RoPE positions; replacement retains the existing Phase 1
slot positions.

## JSONL event contract

The policy log writes `config`, `retrieval`, `context`, `write`, and
`transition` records. These carry exact component settings, query source,
retrieved IDs/scores/scenes, excluded IDs, context ordering/positions/token
count, memory/archive sizes, archive source IDs, retention/decay, and
consolidation action. It is intended to be copied into the paper ledger after
each GPU run; it is not a result by itself.

## Live-code deviations and assumptions

- The global sink key is already RoPE-transformed in the persistent cache.
  All other cached and captured historical K/V is raw. The sink is copied
  directly and never re-rotated.
- Descriptor and injection layers are independent. Only configured injection
  layers retain CPU historical K/V and receive transient K/V; all other live
  Wan layers use their normal local attention. Descriptor layers still derive
  their mean-pooled raw K during the clean capture without retaining their K/V.
- After `sink_only`, no non-sink local raw K remains to form a first-new-scene
  query. The scaffold carries the pre-transition final raw descriptor as a
  logged fallback (`query_source=pre_transition_raw`). Whether it retrieves
  useful memory is an untested assumption.
- The fixed beta decay operates only on retained non-sink local slots. Prompt
  conflict decay, attention-weight utility, descriptor semantic accuracy,
  archive retrieval, and identity steering are not implemented or supported.

## Deterministic verification

```bash
conda run -n wan python -m unittest \
  tests.test_inference_cli tests.test_attention_memory_policy tests.test_noncontiguous_kv -v
```

This covers CPU K/V ownership, routing/manual order, utility/archive/
consolidation behavior, context size/layout, sink retention/decay, event logs,
CLI exposure, and the prior Phase 1 regressions.
