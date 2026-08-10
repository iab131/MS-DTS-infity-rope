# True Wrong-Memory Oracle Preparation

Experiment ID: `E20260807-TRUE-WRONG-P1`
Status: prepared; no run has started at the time of this entry.

## Prompt and exact boundaries

Prompt file: `docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`

| Shot | Requested / effective duration | Blocks | Global latent IDs | Decoded frames |
| --- | --- | --- | --- | --- |
| A: distinctive woman / greenhouse | 2.25 s / 2.25 s | 1--3 | 0--8 | 0--32 |
| B: yellow car only / desert highway | 2.25 s / 2.25 s | 4--6 | 9--17 | 33--68 |
| A2: same woman / observatory | 3.0 s / 3.0 s | 7--10 | 18--29 | 69--116 |

Target is first A2 block 8, current IDs `[21,22,23]`. The manual same source
is A/block 3/ID 8; the manual wrong source is B/block 6/ID 17. B source ID 17
maps to decoded frames 65--68 for the causal VAE contact-sheet verification.

## Matched contexts and artifact limit

| Mode | Context ordering | Total |
| --- | --- | --- |
| baseline | `[sink:0,recent:19,recent:20,current:21,current:22,current:23]` | 6 frames / 9,360 tokens |
| same entity | `[sink:0,history:8,recent:20,current:21,current:22,current:23]` | 6 frames / 9,360 tokens |
| wrong entity | `[sink:0,history:17,recent:20,current:21,current:22,current:23]` | 6 frames / 9,360 tokens |

All use temporary RoPE positions `[0,1,2,3,4,5]`, source blocks `3,6`, target
block `8`, and retrieval count `1`. Save only clean blocks `7,8,9` and raw
decoded output before MP4 conversion.

## Commands (not run)

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/noncontiguous_phase1_true_wrong/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 3,6 --noncontiguous-target-block 8 --noncontiguous-kv-mode baseline --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 7,8,9 --save-raw-decoded
```

Before either history command, inspect the baseline raw decoded contact sheet
for source ID 17 (frames 65--68). Continue only if the rendered frames clearly
show the yellow sports car and no woman/person/humanoid. Record the human
verification in the ledger. If it fails, preserve the baseline and stop this
wrong-memory study rather than treating it as a semantic-negative control.

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/noncontiguous_phase1_true_wrong/same_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 3,6 --noncontiguous-target-block 8 --noncontiguous-kv-mode same_entity_history --noncontiguous-history-frame-id 8 --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 7,8,9 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/noncontiguous_phase1_true_wrong/wrong_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 3,6 --noncontiguous-target-block 8 --noncontiguous-kv-mode wrong_entity_history --noncontiguous-history-frame-id 17 --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 7,8,9 --save-raw-decoded
```

No descriptor routing, MemoryStore, or retrieval-count-2 run belongs to this
study.

## Execution addendum (2026-08-07 local; append-only)

- Commit `5d0eda3a6a0efc7e1520a151d9c0ec28d47cb4a1`; all three runs exit 0.
  Per-run commands, runtime, sampled VRAM, hashes, and logs are in `run.json`
  and `outputs/noncontiguous_phase1_followups_metrics.json`.
- Before wrong-history execution, the required source gate was completed:
  `outputs/noncontiguous_phase1_true_wrong/baseline/b_source_id17_frames65_68.png`
  visibly shows the yellow car and no woman/humanoid. Greenhouse-like
  background leakage means B is car/no-person valid but not a fully isolated
  desert environment.
- Raw outputs equal baseline through clean block 7 / frame 80; both histories
  first differ at block 8, ID 21, frame 81. This is target causality only.
