# Phase 1 Long-Gap Oracle Preparation

Prepared on 2026-08-07. No GPU run has been launched.

## Exact prompt and live block accounting

Prompt file: `docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt`

The requested labels are A = 10 seconds, B = 15 seconds, A2 = 10 seconds.
The live parser uses 0.75-second three-latent blocks and independently floors
each scene duration, while total-duration allocation rounds the overall latent
count up to a multiple of three. This produces the following actual schedule:

| Shot | Requested label | One-based blocks | Global latent IDs | Decoded RGB frames | Effective span |
| --- | ---: | --- | --- | --- | ---: |
| A: greenhouse woman | 10 s | 1--13 | 0--38 | 0--152 | 9.5625 s |
| B: underwater robot | 15 s | 14--33 | 39--98 | 153--392 | 15.0 s |
| A2: observatory woman | 10 s | 34--47 | 99--140 | 393--560 | 10.5 s |

The final A2 block is the live total-duration rounding remainder; no prompt
text is altered to hide that fact. The first A2 block, and therefore the
injection target, is **block 34** with current global latent IDs `[99,100,101]`.

## Manual oracle selections and matched context

- Source blocks: `13,15`; target block: `34`; retrieval count: `1`.
- Same entity: A/block 13/final global latent ID **38**.
- Wrong entity: B/block 15/final global latent ID **44**.
- Target's persistent recent frames are `[97,98]` from B/block 33; neither
  manual selection can be sink ID 0, retained recent ID 98, or current IDs
  99--101.

| Mode | Target ordering | RoPE positions | Total |
| --- | --- | --- | --- |
| baseline | `[sink:0, recent:97, recent:98, current:99, current:100, current:101]` | `[0,1,2,3,4,5]` | 6 frames / 9,360 tokens |
| same_entity_history | `[sink:0, history:38, recent:98, current:99, current:100, current:101]` | `[0,1,2,3,4,5]` | 6 frames / 9,360 tokens |
| wrong_entity_history | `[sink:0, history:44, recent:98, current:99, current:100, current:101]` | `[0,1,2,3,4,5]` | 6 frames / 9,360 tokens |

## Saved artifacts: target-adjacent only

Each command saves only clean-latent snapshots for blocks `33,34,35` (last B,
first A2 target, and next A2) plus one complete raw decoded tensor before MP4
conversion. With `--output_index 0`, per-mode output paths are:

- `0_clean_latents_block_33.pt`, `0_clean_latents_block_34.pt`,
  `0_clean_latents_block_35.pt`
- `0_raw_decoded_before_mp4.pt`
- `0-0_ema.mp4`

No full per-block latent archive is requested or saved.

## Commands (not run)

From `/home/sigasia2026/projects/infinity-rope`, create the temporary model
link only if it is absent, and remove it after the matrix:

```bash
test ! -e wan_models && ln -s /home/sigasia2026/models wan_models
```

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt --output_folder outputs/noncontiguous_phase1_long_gap/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 13,15 --noncontiguous-target-block 34 --noncontiguous-kv-mode baseline --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 33,34,35 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt --output_folder outputs/noncontiguous_phase1_long_gap/same_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 13,15 --noncontiguous-target-block 34 --noncontiguous-kv-mode same_entity_history --noncontiguous-history-frame-id 38 --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 33,34,35 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt --output_folder outputs/noncontiguous_phase1_long_gap/wrong_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 13,15 --noncontiguous-target-block 34 --noncontiguous-kv-mode wrong_entity_history --noncontiguous-history-frame-id 44 --noncontiguous-retrieval-count 1 --save-clean-latent-blocks 33,34,35 --save-raw-decoded
```

```bash
unlink wan_models
```

No descriptor routing, MemoryStore, or retrieval-count-2 path is involved.
