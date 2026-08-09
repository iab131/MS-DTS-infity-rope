# Fixed-grid selective recall oracle — preflight record

Experiment `E20260809-FIXED-GRID-SELECTIVE-RECALL-ORACLE`, 2026-08-09.
Status: implementation and CPU preflight complete; model runs and visual results
are pending.

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
indices.

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
`af29b82d475cd4b23f0cdf0244e789955113a62eaaa9b585f34160dc689a84bb`.

## Pending matched arms

Both commands below are specifications only; neither has been run. They differ
only in `<mode>` and output folder.

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

No claim about identity retention, background retention, leakage reduction,
quality, runtime, VRAM, or paper-level generality is supported until both arms
run and receive matched artifact and human review.
