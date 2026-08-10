# Reset-then-recall oracle — raw experiment record

Experiment `E20260808-AMP-RESET-DELAYED-RECALL-P1`, 2026-08-08. Base commit
`0efd5ad17e57f19ab2029ffc6da44be46e4a415d` plus uncommitted focused
transition/logging fixes. This is a manual all-layer oracle, not automatic
routing or a new memory architecture.

## Question and exact setup

Can a no-old-context scene reset establish A2, then a two-frame delayed K/V
recall preserve useful old content without restoring its old scene?

Prompt: `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt` (verbatim):

```text
A distinctive Ethiopian woman, Amara, with deep brown skin, a small crescent-shaped scar through her left eyebrow, two gold beauty marks below her right eye, tightly coiled black hair in a high braided crown, and a cobalt-blue tailored jumpsuit with a silver sunburst brooch on her left lapel, carefully tends luminous orange orchids inside a sunlit art-deco greenhouse with brass arches and jade tiles, cinematic medium shot.[2.25s#] | A bright yellow low-slung sports car drives alone on an empty sunlit desert highway between red sandstone mesas, no people, no driver visible, no pedestrians, no humanoids, no faces, and no human figures, cinematic wide shot.[2.25s#] | The same distinctive Ethiopian woman Amara, with deep brown skin, the crescent scar through her left eyebrow, two gold beauty marks below her right eye, the high braided crown, cobalt-blue tailored jumpsuit, and silver sunburst brooch on her left lapel, walks through a snowy midnight mountain observatory lit by telescope lamps and aurora light, cinematic medium shot.[3s]
```

Requested/scheduled A/B/A2 is 2.25/2.25/3.0 s; live blocks are `[3,3,4]`:
A IDs 0--8, B 9--17, A2 18--29. Saved RGB is 117 frames/7.3125 s: A 0--32,
B 33--68, A2 69--116; this differs from requested/scheduled 7.5 s.

Fixed: Self-Forcing DMD EMA checkpoint,
`configs/self_forcing_dmd.yaml`, seed 101, 480x832, four steps, cache six,
cross-attention reset, all injection layers 0--29, descriptor layers
0/1/5/14/16. Automatic routing, archive, consolidation, and decay are off.

At hard cuts B block 4 and A2 block 7, every arm uses `transition_no_sink`:
`[current:<start>,current:<start+1>,current:<start+2>]`, cut positions
`[45,46,47]`, 3 frames/4,680 tokens. After A2 block 7, its first clean frame
18 becomes the preserved new-scene sink.

| Arm | Manual IDs at block 8 | Exact block-8 context | Frames/tokens |
| --- | --- | --- | --- |
| reset_only | none | `[sink:18,local:19,local:20,current:21,current:22,current:23]` | 6 / 9,360 |
| delayed_correct_memory | 6,7 (A) | `[sink:18,history:6,history:7,current:21,current:22,current:23]` | 6 / 9,360 |
| delayed_wrong_memory | 16,17 (B car) | `[sink:18,history:16,history:17,current:21,current:22,current:23]` | 6 / 9,360 |

Logical slots are `[0,1,2,3,4,5]`. Reset-only actual RoPE record is preserved
sink, local K `[1,2]`, current K `[3,4,5]`, query `[21,22,23]`; manual arms
are preserved sink, history K `[1,2]`, no local K, current K `[3,4,5]`, query
`[21,22,23]`. The request's physical post-sink “prepend” is achieved with
the existing `replace_recent` mode: it replaces A2 locals 19/20, since literal
`prepend` would expand to eight frames. Sink 18 is preserved, not re-rotated.

## Exact commands

```bash
conda run -n wan python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt \
  --output_folder outputs/attention_memory_policy_reset_then_recall_verified/<arm> \
  --seed 101 --num_samples 1 --save_with_index --output_index 0 \
  --attention-memory-policy --memory-context-mode replace_recent --memory-k 2 \
  --memory-descriptor-layers 0,1,5,14,16 \
  --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --memory-local-retention transition_no_sink \
  --no-memory-decay --no-memory-archive --no-memory-consolidation \
  --no-memory-transition-auto-retrieval \
  --memory-policy-log outputs/attention_memory_policy_reset_then_recall_verified/<arm>/memory_policy.jsonl \
  --save-clean-latent-blocks 6,7,8,9 --save-raw-decoded
```

`reset_only_context_logged` additionally has `--no-memory-retrieval`.
Correct/wrong respectively add `--memory-manual-frame-ids 6,7` or `16,17`
and `--memory-manual-target-blocks 8`.

## Raw results

| Arm | Exit | Runtime (s) | Peak VRAM (MiB) | Raw RGB SHA-256 |
| --- | ---: | ---: | ---: | --- |
| reset_only | 0 | 51.26 | 23,243 | `45bdd1aa4e3dce0ecffc5c925a41b3da42cd5a55e62609918742f5a29259e5b7` |
| delayed_correct_memory | 0 | 50.80 | 23,243 | `fbe2c5fb69dd3e41c9c9bbb1a3b42a4135ae51e47ecc454ccd31305977999adc` |
| delayed_wrong_memory | 0 | 53.65 | 23,243 | `fa715926bd1342e4e12a5391a3712ea99d222a789cba4a772504b42174f928b9` |

Official raw outputs/logs/snapshots are under
`outputs/attention_memory_policy_reset_then_recall_verified/`. The reset-only
official raw tensor equals the earlier reset render byte-for-byte.

Both manual arms are exactly equal to reset through clean blocks 6/7 and RGB
frame 80 (max abs 0.0); first divergence is block 8/frame 81. Metrics are
perturbation-only (RGB in `[0,1]`): correct block-8/9 max/MAE
5.4765625/1.1690958 and 5.6250000/1.1349733; raw MAE 0.1138135, PSNR
11.6895 dB. Wrong is 4.9765625/1.0477773 and 5.1406250/1.0127320; raw MAE
0.0815432, PSNR 14.9524 dB. Raw maximum is 1.0 in both comparisons.

## Human review and limit

The sheet uses A2 frames 69/81/93 (first frames of blocks 7/8/9), row order
reset/correct/wrong. Block 7 is visually identical and blended. Reset-only
then reaches the snowy aurora observatory woman. A-history restores the
greenhouse and orchids; B-car history restores the yellow desert car. A2
repeats Amara's attributes, so correct-memory's woman is not an identity
advantage over reset-only. This is strong one-seed source-selective steering,
but it reintroduces full old scenes rather than demonstrated identity-only
recall.

Two prior complete trios are preserved: the first mislabeled the new sink as
ID 0; the second used Python `[-0:]`, falsely logging replaced locals and
8/12,480. They are logging-only confounds, excluded here. Focused tests now
cover new-sink labels, actual RoPE records, and zero-local context logging.

Comparison artifacts:

- `outputs/attention_memory_policy_reset_then_recall_verified/comparison/A2_blocks_7_8_9_official_reset_correct_wrong.mp4`
- `outputs/attention_memory_policy_reset_then_recall_verified/comparison/A2_blocks_7_8_9_official_still_sheet_reset_correct_wrong.png`
