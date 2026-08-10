# Coarse layer-selective delayed-recall ablation

Experiment: `E20260809-AMP-LAYER-SELECTIVE-P1`  
Date: 2026-08-09  
Base: `0efd5ad17e57f19ab2029ffc6da44be46e4a415d` plus existing uncommitted
reset/logging fixes. This is a manual A-history oracle, not descriptor routing.

## Question and fixed method

Can a coarse transformer-layer range retain Amara-like subject information
without restoring the A greenhouse scene? The exact prompt is unchanged from
`NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt` (Amara greenhouse → yellow
car desert → Amara snowy observatory), seed 101, EMA checkpoint,
`configs/self_forcing_dmd.yaml`, 480x832, four steps, `[3,3,4]` blocks, and
saved A2 RGB frames 69--116.

Every arm uses `transition_no_sink` at B block 4 and A2 block 7, so block 7
is current-only `[current:18,current:19,current:20]` at cut positions
`[45,46,47]`. At delayed A2 block 8/current IDs 21--23, manual source A IDs
6,7 use the same matched physical context
`[sink:18,history:6,history:7,current:21,current:22,current:23]`: six latent
frames / 9,360 tokens. Logical slots are 0--5; logged RoPE values are
preserved sink, history K `[1,2]`, current K `[3,4,5]`, query `[21,22,23]`.
The sink is not re-rotated. `replace_recent` replaces local 19/20 to preserve
the six-frame budget.

Automatic routing, archive, consolidation, and decay are off. Retrieval is
manual only at block 8; all other blocks are identical policy paths. The sole
variable is `--memory-injection-layers`:

| Arm | Injection layers | Status |
| --- | --- | --- |
| reset_only | none | Reused verified `reset_only_context_logged` output |
| L0--9 | 0--9 | Newly run |
| L10--19 | 10--19 | Newly run |
| L20--29 | 20--29 | Newly run |
| L0--29 | 0--29 | Reused verified `delayed_correct_memory` output; settings identical |

## Exact new-arm command

`<layers>` is `0,1,2,3,4,5,6,7,8,9`, `10,11,12,13,14,15,16,17,18,19`, or
`20,21,22,23,24,25,26,27,28,29`.

```bash
conda run -n wan python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt \
  --output_folder outputs/attention_memory_policy_layer_selective/<arm> \
  --seed 101 --num_samples 1 --save_with_index --output_index 0 \
  --attention-memory-policy --memory-context-mode replace_recent --memory-k 2 \
  --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers <layers> \
  --memory-manual-frame-ids 6,7 --memory-manual-target-blocks 8 \
  --memory-local-retention transition_no_sink \
  --no-memory-decay --no-memory-archive --no-memory-consolidation \
  --no-memory-transition-auto-retrieval \
  --memory-policy-log outputs/attention_memory_policy_layer_selective/<arm>/memory_policy.jsonl \
  --save-clean-latent-blocks 7,8,9 --save-raw-decoded
```

## Raw causality record

Every range has manual override IDs 6/7 at block 8, source scene 0, the exact
six-frame context above, exit 0, and sampled peak device VRAM 23,243 MiB.
Each equals reset-only through A2 block 7 / RGB frame 80 (maximum absolute
difference 0.0) and first differs at A2 block 8 / RGB frame 81.

| Arm | Runtime (s) | Raw RGB SHA-256 | RGB MAE vs reset | PSNR vs reset (dB) | Block-8 latent max abs |
| --- | ---: | --- | ---: | ---: | ---: |
| L0--9 | 48.92 | `47f887e4779956c2fc49aff577ae2faae87be7143f961134812b05b6dd645d11` | 0.0257268 | 20.6858 | 4.5546875 |
| L10--19 | 49.88 | `b104f239a5489014f52dae941e51b03a4fc358c1488efb97a8e34e14044717e4` | 0.0300226 | 19.9450 | 4.6328125 |
| L20--29 | 49.21 | `ac6beb446c90241203f416c85535dcae06577736a33e2f714620acb9193725df` | 0.0113323 | 26.9958 | 2.2265625 |
| L0--29 (reused) | 50.80 | `fbe2c5fb69dd3e41c9c9bbb1a3b42a4135ae51e47ecc454ccd31305977999adc` | 0.1138135 | 11.6895 | 5.4765625 |

MAE/PSNR are raw RGB perturbation measures, not subject/scene quality metrics.

## Human review (one seed; not a metric)

Still-sheet columns are A2 frames 69/81/93 (blocks 7/8/9). Row order is
reset, L0--9, L10--19, L20--29, L0--29.

| Layers | Original-A woman/appearance recall | Greenhouse/orchid recall | Snowy-observatory retention | Obvious artifacts |
| --- | --- | --- | --- | --- |
| reset | Woman/blue suit appear, but this is prompt-conditioned baseline | None after blend | Yes | Initial cut blend only |
| 0--9 | No verified incremental A-appearance recall beyond reset | None | Yes | No obvious new artifact in reviewed frames |
| 10--19 | No verified incremental A-appearance recall beyond reset | None | Yes | No obvious new artifact in reviewed frames |
| 20--29 | No verified incremental A-appearance recall beyond reset | None | Yes | No obvious new artifact in reviewed frames |
| 0--29 | Woman/blue suit present, but no valid advantage over prompt baseline | Strong greenhouse and orange-orchid return | No | Scene replacement rather than localized artifact |

A2 repeats the woman’s major visual attributes, so this experiment cannot
attribute woman appearance to memory unless it visibly exceeds reset-only.
None of the coarse ten-layer ranges does so. The all-layer A-memory result
does visibly restore the A scene; the previous verified B-memory result
restored B car/desert content. Together they demonstrate source-selective but
scene-entangled raw-K/V recall.

## Conclusion

No coarse layer range is a promising identity-only range in this single-seed
oracle. Each ten-layer range preserves the desired snowy scene, while the
all-layer intervention restores the unwanted A scene. This suggests that the
strong scene effect requires a distributed combination across ranges, but does
not establish layer-wise subject/scene separability. Per the stop rule, no
finer sweep was launched.

Artifacts:

- `outputs/attention_memory_policy_layer_selective/comparison/A2_blocks_7_8_9_five_way_reset_L0_9_L10_19_L20_29_L0_29.mp4`
- `outputs/attention_memory_policy_layer_selective/comparison/A2_blocks_7_8_9_five_way_still_sheet_reset_L0_9_L10_19_L20_29_L0_29.png`
