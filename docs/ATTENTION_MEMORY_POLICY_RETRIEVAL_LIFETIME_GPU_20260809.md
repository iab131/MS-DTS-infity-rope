# Early-layer manual-retrieval lifetime oracle — raw experiment record

Experiment `E20260809-AMP-LIFETIME-L0-9-P1`, 2026-08-09. Base commit
`0efd5ad17e57f19ab2029ffc6da44be46e4a415d` plus the uncommitted
`--memory-retrieval-lifetime` control and prior transition/logging fixes. This
is a manual A-history oracle, not automatic routing or a new memory mechanism.

## Policy-log audit and fixed method

Before this change, the verified all-layer and L0--9 JSONLs both recorded:

| A2 block | Existing manual KV event | Meaning |
| --- | --- | --- |
| 7 | `[]`, `manual_target_not_selected` | reset block; no retrieval |
| 8 | `[6,7]`, `manual_override` | only manual injection block |
| 9 | `[]`, `manual_target_not_selected` | no injection |
| 10 | `[]`, `manual_target_not_selected` | no injection |

Thus explicit `--memory-manual-target-blocks 8` was already a one-block pulse,
not persistent. Old manual IDs *without* a target restriction persisted. The
new option starts at the first target: `pulse_1` (default), `pulse_2`, or
`persistent`; JSONL records `retrieval_lifetime` and `manual_lifetime_expired`.

The prompt remains exactly `docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`:

```text
A distinctive Ethiopian woman, Amara, with deep brown skin, a small crescent-shaped scar through her left eyebrow, two gold beauty marks below her right eye, tightly coiled black hair in a high braided crown, and a cobalt-blue tailored jumpsuit with a silver sunburst brooch on her left lapel, carefully tends luminous orange orchids inside a sunlit art-deco greenhouse with brass arches and jade tiles, cinematic medium shot.[2.25s#] | A bright yellow low-slung sports car drives alone on an empty sunlit desert highway between red sandstone mesas, no people, no driver visible, no pedestrians, no humanoids, no faces, and no human figures, cinematic wide shot.[2.25s#] | The same distinctive Ethiopian woman Amara, with deep brown skin, the crescent scar through her left eyebrow, two gold beauty marks below her right eye, the high braided crown, cobalt-blue tailored jumpsuit, and silver sunburst brooch on her left lapel, walks through a snowy midnight mountain observatory lit by telescope lamps and aurora light, cinematic medium shot.[3s]
```

Requested A/B/A2 durations are 2.25/2.25/3.0 s. Live schedule is `[3,3,4]`:
A latent IDs 0--8, B 9--17, A2 18--29. Decoded RGB is 117 frames at 16 FPS /
7.3125 s, rather than requested 7.5 s. Fixed settings: Self-Forcing DMD EMA
checkpoint, `configs/self_forcing_dmd.yaml`, seed 101, 480x832, four steps,
cache six, descriptor layers `0,1,5,14,16`, injection layers 0--9, verified A
source IDs 6/7, `transition_no_sink`, `replace_recent`, k=2, and cross-attention
reset. Automatic routing, decay, archive, and consolidation are off.

At A2 block 7 all arms are current-only `[current:18,current:19,current:20]`,
cut positions `[45,46,47]`, 3 frames / 4,680 tokens, then frame 18 becomes the
preserved transformed sink. Later contexts are exactly six frames / 9,360 tokens:

| Arm | Manual A IDs 6/7 | A2 contexts at blocks 8--10 |
| --- | --- | --- |
| reset_only | none | sink + two local + current×3 |
| L0--9 pulse_1 | 8 | history at 8, local at 9--10 |
| L0--9 pulse_2 | 8--9 | history at 8--9, local at 10 |
| L0--9 persistent | 8--10 | history at 8--10 |

History context is `[sink:18,history:6,history:7,current×3]`; absent-history
context is `[sink:18,local,local,current×3]`. Both have slots 0--5. The sink
is preserved, never re-rotated; history/local K positions are `[1,2]`, current
K positions `[3,4,5]`, and query positions are global block IDs.

## Exact commands

`<lifetime>` is `pulse_2` or `persistent`; output folder uses `l0_9_<lifetime>`.

```bash
conda run -n wan python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt \
  --output_folder outputs/attention_memory_policy_retrieval_lifetime/l0_9_<lifetime> \
  --seed 101 --num_samples 1 --save_with_index --output_index 0 \
  --attention-memory-policy --memory-context-mode replace_recent --memory-k 2 \
  --memory-descriptor-layers 0,1,5,14,16 \
  --memory-injection-layers 0,1,2,3,4,5,6,7,8,9 \
  --memory-manual-frame-ids 6,7 --memory-manual-target-blocks 8 \
  --memory-retrieval-lifetime <lifetime> \
  --memory-local-retention transition_no_sink \
  --no-memory-decay --no-memory-archive --no-memory-consolidation \
  --no-memory-transition-auto-retrieval \
  --memory-policy-log outputs/attention_memory_policy_retrieval_lifetime/l0_9_<lifetime>/memory_policy.jsonl \
  --save-clean-latent-blocks 7,8,9,10 --save-raw-decoded
```

Reused control paths: reset is `...reset_then_recall_verified/reset_only_context_logged`;
L0--9 pulse-1 is `...layer_selective/layers_0_9`, verified above as block-8 only.

## Raw results

All runs exit 0 and sampled peak VRAM is 23,243 MiB. Hashes are SHA-256 of
contiguous raw decoded RGB tensor bytes; `.pt` serialization file hashes differ
from tensor hashes and are retained through the output paths. MAE/PSNR compare
raw RGB `[0,1]` with reset-only and quantify perturbation, not visual quality.

| Arm | Runtime (s) | Tensor SHA-256 | MAE | PSNR (dB) | First diff vs reset |
| --- | ---: | --- | ---: | ---: | ---: |
| reset_only (reused) | 51.26 | `45bdd1aa4e3dce0ecffc5c925a41b3da42cd5a55e62609918742f5a29259e5b7` | 0 | inf | none |
| L0--9 pulse_1 (reused) | 48.92 | `47f887e4779956c2fc49aff577ae2faae87be7143f961134812b05b6dd645d11` | 0.0257268 | 20.6858 | RGB 81 / A2 b8 |
| L0--9 pulse_2 | 50.05 | `fa140fe0dd00c7c7490b8f14b56b80df21daef180dd15e43d2fd114ab41fc9bd` | 0.0303887 | 19.7575 | RGB 81 / A2 b8 |
| L0--9 persistent | 48.81 | `520b3f44f44f43352665c4cb0e9ff2c3aa3444f28267f1ba4d0f2dbb2571f523` | 0.0312111 | 19.5709 | RGB 81 / A2 b8 |

All retrieval arms are exactly equal to reset through RGB 80/A2 block 7. Pulse-2
first differs from pulse-1 at RGB 93/A2 block 9; persistent first differs from
pulse-2 at RGB 105/A2 block 10, matching the extra injection blocks. Per-block
MAE vs reset for A2 blocks 7/8/9/10 is pulse-1
`0/0.0712880/0.0913844/0.0881634`, pulse-2
`0/0.0712880/0.1069306/0.1180708`, and persistent
`0/0.0712880/0.1069306/0.1260897`.

Mean adjacent-frame RGB absolute change for blocks 7/8/9/10 is reset
`0.03457/0.02534/0.02151/0.01907`, pulse-1
`0.03457/0.02170/0.01706/0.01613`, pulse-2
`0.03457/0.02170/0.01236/0.01123`, persistent
`0.03457/0.02170/0.01236/0.01736`. This is a motion quantity, not a flicker
detector.

## Human review, confounds, and conclusion

The synchronized video and sheet inspect RGB 69/81/93/105 (A2 blocks 7--10).
The reset blend is shared. In this one-seed human review, all L0--9 lifetimes
retain snowy aurora-observatory content after reset; no greenhouse/orchid return
or incremental Amara face/appearance correction beyond the explicit A2 prompt
is visibly verified. No obvious added abrupt reset/flicker is visible, and the
following block remains coherent. The motion proxy is not evidence of temporal
stability. This establishes only causal lifetime control at matched budget, not
identity benefit, no-flicker behavior, or a reason for a finer sweep.

Artifacts:

- `outputs/attention_memory_policy_retrieval_lifetime/l0_9_pulse_2/`
- `outputs/attention_memory_policy_retrieval_lifetime/l0_9_persistent/`
- `outputs/attention_memory_policy_retrieval_lifetime/comparison/A2_blocks_7_10_four_way_reset_pulse1_pulse2_persistent.mp4`
- `outputs/attention_memory_policy_retrieval_lifetime/comparison/A2_blocks_7_10_four_way_still_sheet_reset_pulse1_pulse2_persistent.png`
