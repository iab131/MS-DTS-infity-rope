# L0--9 source-specificity oracle — raw experiment record

Experiment `E20260809-AMP-L0-9-SOURCE-SPECIFICITY-P1`, 2026-08-09. Base commit
`0efd5ad17e57f19ab2029ffc6da44be46e4a415d` plus existing uncommitted policy
work. No routing, archive, decay, consolidation, lifetime tuning, or layer sweep
was added.

## Question and fixed setup

Does the small L0--9 pulse-1 perturbation reflect source-specific historical
recall, or a generic effect of transient K/V injection? The exact prompt is
unchanged in `docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`: Amara/
orchids A, yellow-car B, and snowy-observatory Amara A2. Requested A/B/A2
2.25/2.25/3.0 s schedules `[3,3,4]`: A IDs 0--8, B 9--17, A2 18--29. Saved
decoded RGB is 117 frames/7.3125 s at 16 FPS, not requested 7.5 s.

Fixed: Self-Forcing DMD EMA checkpoint, `configs/self_forcing_dmd.yaml`, seed
101, 480x832, four steps, cache six, descriptor layers `0,1,5,14,16`,
injection layers 0--9, `transition_no_sink` at B block 4/A2 block 7,
`replace_recent`, k=2, cross-attention reset on; automatic routing, archive,
decay, and consolidation off. A2 block 7 is current-only IDs 18--20 at
positions `[45,46,47]`, then frame 18 becomes the preserved transformed sink.

Sources were previously human-gated: A IDs 6/7 contain Amara; B IDs 16/17
contain yellow car/no woman or humanoid. B retains greenhouse-background
leakage, making it an entity-negative but not isolated-environment negative.

| Arm | Source IDs at block 8 | Exact context |
| --- | --- | --- |
| reset_only (reused) | none | `[sink:18,local:19,local:20,current:21,22,23]` |
| L0--9 correct-A pulse_1 (reused) | 6,7 / scene 0 | `[sink:18,history:6,history:7,current:21,22,23]` |
| L0--9 wrong-car pulse_1 (new) | 16,17 / scene 1 | `[sink:18,history:16,history:17,current:21,22,23]` |

Every target is six frames / 9,360 tokens, slots 0--5. Sink K is preserved,
never re-rotated; retrieved K is at `[1,2]`, current K `[3,4,5]`, query
positions `[21,22,23]`. At blocks 9--10 both pulses resume sink + two local +
current×3 at the same budget.

Reuse caveat: the old reset config lists all injection layers, but retrieval is
disabled so it injects nothing. Correct-A predates the explicit lifetime field,
but its target-8 JSONL proves behavior identical to `pulse_1`.

## Exact new-arm command

```bash
conda run -n wan python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt \
  --output_folder outputs/attention_memory_policy_source_specificity/l0_9_pulse_1_wrong_car \
  --seed 101 --num_samples 1 --save_with_index --output_index 0 \
  --attention-memory-policy --memory-context-mode replace_recent --memory-k 2 \
  --memory-descriptor-layers 0,1,5,14,16 \
  --memory-injection-layers 0,1,2,3,4,5,6,7,8,9 \
  --memory-manual-frame-ids 16,17 --memory-manual-target-blocks 8 \
  --memory-retrieval-lifetime pulse_1 \
  --memory-local-retention transition_no_sink \
  --no-memory-decay --no-memory-archive --no-memory-consolidation \
  --no-memory-transition-auto-retrieval \
  --memory-policy-log outputs/attention_memory_policy_source_specificity/l0_9_pulse_1_wrong_car/memory_policy.jsonl \
  --save-clean-latent-blocks 7,8,9,10 --save-raw-decoded
```

## Raw results

Wrong-car exits 0 in 52.15 s, peak sampled VRAM 23,243 MiB. Hashes are SHA-256
of contiguous raw decoded RGB tensor bytes; MAE/PSNR are raw RGB `[0,1]`
perturbation measures, not semantic-quality metrics.

| Arm | Tensor SHA-256 | First diff vs reset | MAE vs reset | PSNR vs reset (dB) |
| --- | --- | ---: | ---: | ---: |
| reset_only | `45bdd1aa4e3dce0ecffc5c925a41b3da42cd5a55e62609918742f5a29259e5b7` | none | 0 | inf |
| correct-A | `47f887e4779956c2fc49aff577ae2faae87be7143f961134812b05b6dd645d11` | RGB 81 / A2 b8 | 0.0257268 | 20.6858 |
| wrong-car | `89bbcb38d63cf4a80041b21ef08422d357571deab05ec6d63f75bb4c86336506` | RGB 81 / A2 b8 | 0.0188217 | 22.7431 |

Both pulses equal reset through RGB 80/A2 block 7. Wrong-car first differs from
correct-A at RGB 81 (MAE 0.0241277, max abs 1.0). A2 block-8/9/10 MAE vs reset
is correct-A `0.0712880/0.0913844/0.0881634` and wrong-car
`0.0429851/0.0652959/0.0752305`. Wrong-car clean-latent max difference vs
reset is 0.0/4.1171875/4.1484375 at blocks 7/8/9; correct-A is
0.0/4.5546875/4.4375.

## Human review and limited conclusion

The sheet/video review RGB 69/81/93/105 (A2 blocks 7--10). The reset blend is
identical. Correct-A and wrong-car both show small face/pose/detail perturbation
relative to reset and differ at raw pixels, but no specific original-A facial or
appearance feature is visibly recovered beyond the explicit A2 prompt. Wrong-car
does not visibly introduce car, desert, or other identifiable B content; snowy
observatory remains. This one-seed review does not show semantic source
specificity for L0--9. The visible effect is consistent with generic transient
facial perturbation, though source-dependent raw differences do not prove source
independence. No lifetime tuning or finer layer sweep is motivated.

Artifacts:

- `outputs/attention_memory_policy_source_specificity/l0_9_pulse_1_wrong_car/`
- `outputs/attention_memory_policy_source_specificity/comparison/A2_blocks_7_10_three_way_reset_correctA_wrongcar.mp4`
- `outputs/attention_memory_policy_source_specificity/comparison/A2_blocks_7_10_three_way_still_sheet_reset_correctA_wrongcar.png`

