# Long-Gap Oracle A-B-A: Raw Experiment Record

Experiment ID: `E20260807-LG-P1`
Local experiment date: 2026-08-07
Commit: `c0d3f9e8c81ae072000f432f7fd48f9035bc2698`

## Research question and hypothesis

After a 15-second visually unrelated B shot, can manual same-entity A history
or wrong-entity B history affect only the first A2 target block at a matched
6-frame / 9,360-token context budget? The preregistered operational hypothesis
was exact baseline equality through block 33 / decoded frame 392 and first
history divergence at block 34 / decoded frame 393.

## Prompt and effective boundaries

The verbatim prompt is `NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt`.
Requested A/B/A2 durations are 10/15/10 seconds. The live model used 141
latents / 561 decoded RGB frames because scene allocation is block-quantized:

| Shot | Blocks | Global latent IDs | Decoded RGB frames | Effective duration |
| --- | --- | --- | --- | ---: |
| A | 1--13 | 0--38 | 0--152 | 9.5625 s |
| B | 14--33 | 39--98 | 153--392 | 15.0 s |
| A2 | 34--47 | 99--140 | 393--560 | 10.5 s |

## Method and fixed settings

- EMA checkpoint: `/home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt`
- Config: `configs/self_forcing_dmd.yaml`; 480x832; four denoising steps;
  BF16; `local_attn_size=6`; `sink_size=1`; seed 101.
- Source blocks 13 and 15; target block 34; retrieval count 1.
- Same-entity history: clean A/global latent ID 38. Wrong-entity history:
  clean B/global latent ID 44.
- Baseline context `[sink:0,recent:97,recent:98,current:99,current:100,current:101]`;
  same `[sink:0,history:38,recent:98,current:99,current:100,current:101]`;
  wrong `[sink:0,history:44,recent:98,current:99,current:100,current:101]`.
  All use RoPE positions `[0,1,2,3,4,5]` and total 9,360 tokens.
- Saved diagnostics: only clean blocks 33--35 and one raw decoded tensor before
  MP4 conversion. No descriptor routing, MemoryStore, or retrieval count 2.

## Exact commands and raw outcomes

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt --output_folder outputs/noncontiguous_phase1_long_gap/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 13,15 --noncontiguous-target-block 34 --noncontiguous-kv-mode baseline --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 33,34,35 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt --output_folder outputs/noncontiguous_phase1_long_gap/same_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 13,15 --noncontiguous-target-block 34 --noncontiguous-kv-mode same_entity_history --noncontiguous-history-frame-id 38 --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 33,34,35 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt --output_folder outputs/noncontiguous_phase1_long_gap/wrong_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 13,15 --noncontiguous-target-block 34 --noncontiguous-kv-mode wrong_entity_history --noncontiguous-history-frame-id 44 --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 33,34,35 --save-raw-decoded
```

| Mode | Exit | Runtime | Sampled peak VRAM | Raw decoded SHA-256 | MP4 SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 0 | 107.595 s | 69,936 MiB | `f73b3016657c04699b2f951d264465c14e36af89a53268a7764e53827c6b16cc` | `a4b5d9c2795dcfc57c2adaf1585fc5b4907bce0259130e5376fb5eebc0f9afa7` |
| same entity, ID 38 | 0 | 91.100 s | 49,671 MiB | `e8d993d0d0719f0b012ea3c3acedb796591a929a9439446928022db7368810cd` | `08b1c3e51d840da0f7a83697563e15d5fe155da2b0ed7d6959969556ce6a44bc` |
| wrong entity, ID 44 | 0 | 90.984 s | 49,671 MiB | `a37186ee076306eedd4cfb19b3794af0b3cac4ad5ef0492f773261fa5f61efb8` | `2dbe9e7cd2d56314a3b4542838eb8b66060cc0f1784b4336ffcc9d67beb6f4be` |

Raw paths, clean snapshot hashes, per-run commands, logs, and sampled VRAM are
in `outputs/noncontiguous_phase1_long_gap/{mode}/`; consolidated metrics are
in `outputs/noncontiguous_phase1_long_gap/metrics.json`.

## Quantitative results

| Comparison vs baseline | Block 33 max abs | First clean-latent divergence | Block 34 max abs | Block 35 max abs | First raw decoded frame | Raw frames 0--392 max abs | All-frame RGB MAE (unit interval) | All-frame max abs |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same entity | 0.0 | block 34, local 0, global 99 | 3.703125 | 4.203125 | 393 | 0.0 | 0.075158 | 1.0 |
| wrong entity | 0.0 | block 34, local 0, global 99 | 3.726563 | 4.238281 | 393 | 0.0 | 0.080417 | 1.0 |

## Visual observations (human review)

No human visual review has been recorded. The generated comparison is an
artifact for later review, not evidence of identity preservation or degradation.

- `outputs/noncontiguous_phase1_long_gap/block33_to35_baseline_same_wrong.mp4`
- 2496x480, 36 frames, 16 FPS, SHA-256
  `ec484b2f7a48789886793fa5a2def1b7f862b586c64a12f843d1c4647305554f`

## Failures, confounds, and interpretation limit

- No inference command failed.
- Baseline sampled peak device VRAM is higher than either later run. The
  sampler reports device-wide usage, and baseline was first in process/cache
  warm-up order. This single ordering is not evidence that a method changes
  VRAM or runtime; it must not be used as a performance claim.
- This is a single seed and has no human or automated identity score. The
  result cannot establish that same history improves identity, or that wrong
  history harms it.

## Limited conclusion and next experiment

This experiment demonstrates, at one seed, that both manually selected
long-gap histories leave target-preceding raw tensors unchanged and first alter
the output at the configured first A2 block. It does not demonstrate a visual
identity benefit. Next: randomized multi-seed repetitions with balanced
warm-up/order and a preregistered identity/consistency evaluation protocol.
