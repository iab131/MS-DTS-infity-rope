# Fixed-grid selective recall oracle — executed record

Experiment `E20260809-FIXED-GRID-SELECTIVE-RECALL-ORACLE`, 2026-08-09.
Status: implementation, preflight, and the two authorized model arms complete.

## Scope and attribution

This is a bounded oracle separability experiment, not a novel masking method.
It is inspired by EM-Vid's entity-centric sparse memory motivation, DiTCtrl's
foreground/background mask-guided K/V sharing, and BachVid's foreground/
background attention separation and K re-positioning. It does not reproduce or
claim the method of any of those works. Fixed masks are supplied manually to
ask one narrow question: can historical subject content affect only target
subject queries, while historical background content affects only target
background queries, without replacing the normal local/current context?

## Immutable masks and positions

The committed source/target masks are
`docs/attention_memory_policy_fixed_grid_masks_20260809.json` (SHA-256
`743a7c6e2a4d6c41c01da9b77e553d59e3cf3ccb989d6a96160c3a209c4fd5cf`).
The JSON records every manually verified inclusive zero-indexed row span and
its generated binary 30x52 grid.

- Source ID 6 uses decoded reset-only frame 26, 541 subject tokens, and
  temporary history slot 1.
- Source ID 7 uses decoded reset-only frame 30, 542 subject tokens, and
  temporary history slot 2.
- The target conservative union contains 467 subject tokens per frame and is
  repeated, without spatial reindexing, over target latent IDs 21/22/23
  (decoded frames 81/85/89) at A2 block 8.
- The background arm uses the complement after an eight-connected one-token
  dilation independently on both source masks and the target mask. It retains
  922/921 historical tokens from IDs 6/7 and 1,011 target queries per frame.

Historical K stays raw in CPU memory until its executing layer. Only then is K
RoPE-encoded with each token's original row-major H/W coordinate and source
slot 1 or 2; paired V is unchanged. The subject arm packs only subject history
and targets only subject queries. The background arm packs only each source's
dilated complement and targets only the target dilated complement. The other
historical token group is absent from that arm.

At block 8, full retrieved-K/V injection and `replace_recent` assembly are
bypassed. Base attention keeps the verified reset-only order
`[sink:18,local:19,local:20,current:21,current:22,current:23]` (6 frames /
9,360 tokens), including its existing sink/local/current RoPE behavior. The
masked historical result is a separate addition only at selected current-query
indices. Fixed-grid CLI validation accepts only this matched reset protocol:
`--memory-local-retention transition_no_sink` and
`--memory-context-mode replace_recent`. The transition clean pass records A2
frame 18 as the new persistent sink used by later ordinary context logging.

## CPU-only overlay and audit preflight

Executed without a model or GPU:

```bash
conda run -n wan python scripts/prepare_fixed_grid_memory_oracle.py \
  --mask-path docs/attention_memory_policy_fixed_grid_masks_20260809.json \
  --video-path outputs/attention_memory_policy_reset_then_recall_verified/reset_only_context_logged/0-0_ema.mp4 \
  --output-dir outputs/attention_memory_policy_fixed_grid_selective_recall/preflight
```

The command wrote source overlays for IDs 6/7, target overlays for all three
block-8 query frames, and
`outputs/attention_memory_policy_fixed_grid_selective_recall/preflight/mask_audit.json`.
The audit records source indices/counts/row-column coordinates, temporal slots,
all expanded target subject/background query indices, target block/frame IDs,
input and overlay hashes, and the unchanged base-context order. Audit SHA-256:
`a6d7b90a0835ed39eedfab0d93d3733e048f26a0c0f635d6906189fd6fbafaf8`.

## Matched executed arms

The commands below were run once each. They differ only in `<mode>` and output
folder; reset-only and verified full-A controls were reused unchanged.

```bash
conda run -n wan python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt \
  --output_folder outputs/attention_memory_policy_fixed_grid_selective_recall/<mode> \
  --seed 101 --num_samples 1 --save_with_index --output_index 0 \
  --attention-memory-policy --memory-context-mode replace_recent --memory-k 2 \
  --memory-descriptor-layers 0,1,5,14,16 \
  --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --memory-local-retention transition_no_sink \
  --no-memory-decay --no-memory-archive --no-memory-consolidation \
  --no-memory-transition-auto-retrieval \
  --memory-manual-frame-ids 6,7 --memory-manual-target-blocks 8 \
  --memory-fixed-grid-mask-path docs/attention_memory_policy_fixed_grid_masks_20260809.json \
  --memory-fixed-grid-mode <subject_to_subject|background_to_background> \
  --memory-policy-log outputs/attention_memory_policy_fixed_grid_selective_recall/<mode>/memory_policy.jsonl \
  --save-clean-latent-blocks 7,8,9 --save-raw-decoded
```

## Execution and invariant check

| Arm | Exit / runtime | Peak VRAM | Historical source K | Target queries |
| --- | --- | --- | --- | --- |
| `subject_to_subject_A_memory` | 0 / 53 s | 23,243 MiB | subject: 541 (ID 6) + 542 (ID 7) | 1,401 (467 × 3) |
| `background_to_background_A_memory` | 0 / 54 s | 23,243 MiB | dilated-complement: 922 (ID 6) + 921 (ID 7) | 3,033 (1,011 × 3) |

Both JSONLs log the same positional facts: source IDs `[6,7]`, original
row/column source coordinates, source temporal slots `{6:1,7:2}`, target
frames `[21,22,23]`, all 30 injection layers, and
`base_context_unchanged=true`. The normal block-8 base order is exactly
`[sink:18,local:19,local:20,current:21,current:22,current:23]`; the
selective branch is the only extra computation. All non-manual archive,
consolidation, decay, and automatic transition retrieval remain disabled.
Each new result is exactly equal to reset-only at saved clean block 7; visual
and raw differences begin at the retrieval block rather than before it.

## Four-arm oracle readout

Review used
`outputs/attention_memory_policy_fixed_grid_selective_recall/comparison/four_arm_recall_sheet.png`
(rows `reset_only | full_A | subject_to_subject | background_to_background`,
decoded frames 77/81/85/89/93) and the two new MP4s. The companion
`four_arm_metrics.json` is a pixel-proxy record, not a semantic score.

- `reset_only` retains the snowy observatory and its prompted woman.
- Reused `full_A` immediately replaces the observatory with greenhouse arches
  and orchids, confirming the known scene-entangled control effect.
- `subject_to_subject` changes face/hair/clothing-region appearance at the
  target, while much of the snowy scene remains visible. It nevertheless adds
  a bright green greenhouse-like structure behind the woman and creates a
  sharp retrieval-boundary discontinuity. Relative to reset over RGB 81--92,
  its MAE is 0.0905393; the frame-80→81 adjacent RGB change is 0.0678933
  (reset: 0.0258359).
- `background_to_background` preferentially transfers greenhouse
  arches/orchids into the scene while leaving the woman more reset-like than
  full-A at early recall frames. It strongly damages snowy-observatory
  preservation and remains discontinuous after the one-block pulse (RGB
  92→93 adjacent change 0.2033033; reset: 0.0245210).

This single-prompt, one-seed oracle is evidence that raw historical K/V has a
substantial spatially separable component: background-only routing strongly
recalls the old background, while subject-only routing concentrates a smaller
effect at the woman. It does **not** establish clean identity-only recall:
subject routing still leaks old-scene structure and both new arms show
retrieval-boundary artifacts. It remains an oracle separability result, not a
novel masking claim or a general policy result; no automated masking,
tracking, SAM, descriptors, or additional memory-policy machinery was added.

## Subject-core / boundary ablation (one seed)

The same immutable source masks and A2 target union were used to derive three
additional subject-only arms with 8-connected binary erosion. No input mask was
redrawn and sparse source coordinates remained the original row-major 30x52
positions; temporal source slots remain 1/2. The CPU preflight wrote full,
erode1, erode2, and removed-boundary-ring overlays for both A sources and all
three A2 target frames under
`outputs/attention_memory_policy_fixed_grid_selective_recall/subject_core_boundary_ablation/preflight/`.
Its audit SHA-256 is
`7da283110bb0849f9b5b576e796d31c5ba8e9d63c6cf5c91e6ea9f36776419b0`.

| Arm | Source ID 6 / 7 tokens | Target tokens/frame (total) | Exit / runtime / peak VRAM |
| --- | ---: | ---: | --- |
| reused `subject_full` | 541 / 542 | 467 (1,401) | reused prior control |
| `subject_erode1` | 426 / 427 | 366 (1,098) | 0 / 51 s / 23,243 MiB |
| `subject_erode2` | 322 / 323 | 276 (828) | 0 / 52 s / 23,243 MiB |
| `subject_boundary_only` | 115 / 115 | 101 (303) | 0 / 51 s / 23,247 MiB |

All three new JSONLs confirm IDs 6/7, slots `{6:1,7:2}`, all 30 injection
layers, block 8 target frames `[21,22,23]`, and the unchanged six-frame base
`[sink:18,local:19,local:20,current:21,current:22,current:23]`. Each is
exactly equal to reset-only at saved clean block 7, so the effects begin at
retrieval rather than before it.

Five-arm visual review used
`comparison/five_arm_subject_core_boundary_sheet.png` (reset / full / erode1 /
erode2 / boundary ring; decoded frames 77/81/85/89/93). Both eroded cores still
restore the A1 woman appearance and the bright local greenhouse-like structure
behind her. The snow observatory remains outside the subject much better than
full-A, but erosion does not remove the local A1-background halo. The boundary
ring alone leaves the A2 woman largely reset-like while producing a weaker,
localized green/orange edge halo.

The pixel proxy is consistent with that visual readout: total RGB MAE vs reset
over frames 81--92 decreases full/e1/e2/ring =
0.09054/0.07708/0.06836/0.03211, but erode2 still changes the original
subject-core region by 0.22161 MAE and retains a visible local scene recall.
The target-boundary adjacent-frame change similarly falls from full 0.06789 to
erode1 0.05775 and erode2 0.04893, but remains above reset 0.02584; ring is
0.03181. These are pixel/discontinuity proxies, not identity or halo scores.

**Conclusion:** mixed boundary tokens contribute to the leakage magnitude, but
they are not its main cause in this oracle. Even the two-token eroded core
reproduces A1 surroundings with the woman, so raw subject-core K/V is already
context-entangled. Simple spatial erosion is insufficient for clean
subject-only recall. The subsequent alpha-only experiment is reported below;
no automatic masks, tracking, alpha blending, finer layers, or new
memory-policy mechanism was run.

## Erode2 historical-output strength (one seed)

The alpha experiment changes only the existing erode2 historical branch at
block 8. Let `O_base` be normal reset-only attention and `O_mem` be the prior
erode2 selective result. The implementation uses
`O_base + alpha * (O_mem - O_base)` at the selected erode2 queries. Alpha 0
hard-bypasses that branch and therefore reuses reset-only exactly; alpha 1 has
the previous erode2 addition path without a multiply and reuses its existing
artifact. Background queries always take `O_base`.

All settings are otherwise identical: seed 101, prompt, IDs 6/7, slots 1/2,
block 8, pulse-1 retrieval, all 30 layers, erode2 counts 322/323 source and
276 target queries/frame (828 total), no automatic routing/archive/
consolidation/decay, and base
`[sink:18,local:19,local:20,current:21,current:22,current:23]` (6 frames /
9,360 tokens). The three new intermediate arms exit successfully:

| Alpha | Artifact | Runtime / peak VRAM |
| ---: | --- | --- |
| 0.00 | reused reset-only | exact hard bypass |
| 0.10 | `alpha_0_10` | 49 s / 23,243 MiB |
| 0.25 | `alpha_0_25` | 49 s / 23,243 MiB |
| 0.50 | `alpha_0_50` | 51 s / 23,243 MiB |
| 1.00 | reused existing erode2 | unchanged full-strength path |

Each new JSONL logs alpha, the same IDs/counts/coordinates/slots/query set,
and `base_context_unchanged=true`; saved block 7 is numerically equal to
reset-only. The five-strength sheet is
`subject_core_boundary_ablation/alpha_strength/comparison/five_alpha_temporal_sheet.png`
(pre-recall 77; recall 81/85/89; post-recall 93/105/113), SHA-256
`c02acedcbff8e81d9d13bb594ff429d90752ba28c299e130891312dc96258ca7`.

Visual temporal readout:

- **Alpha 0.00:** stable snow woman and observatory.
- **Alpha 0.10:** no clear A1 appearance recovery in this one-seed review;
  snow scene and A2 woman remain effectively reset-like.
- **Alpha 0.25:** a small face/pose appearance perturbation appears during the
  recall block, without a clear A1 greenhouse flash in the reviewed stills;
  post-recall A2 scene is preserved. This is a weak correction, not verified
  A1 identity recovery.
- **Alpha 0.50:** clear A1-like hair/face/clothing perturbation and local green
  greenhouse flash at recall; subsequent A2 blocks restore most snow scenery
  while leaving an A1/A2-looking hybrid woman.
- **Alpha 1.00:** strongest A1 woman and local surroundings flash, followed by
  the previously observed scene reconciliation and hybrid post-recall woman.

The pixel proxy tracks this threshold-like tradeoff, but is not a semantic
score. Core-region recall-block MAE vs reset is 0.07135/0.09958/0.18980/0.22161
at alpha 0.10/0.25/0.50/1.00; one-token exterior-halo MAE is
0.01928/0.03604/0.06378/0.07508. The 80→81 discontinuity is
0.02686/0.03072/0.04214/0.04893 (reset 0.02584). Thus lower alpha can preserve
the established A2 scene and reduce the hard overwrite, but no tested alpha
delivers a clearly verified A1 appearance correction without some tradeoff:
0.10 is near-inert, 0.25 is subtle, and 0.50+ visibly leak/overwrite locally.
No temporal alpha schedule was added or run.
