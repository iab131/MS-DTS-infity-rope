# Matched Non-Contiguous KV Phase 1 GPU Experiment

Run date: 2026-08-04

Checkout: `44fa22c` (code under test: `3f004b4` and `46880a1`)

## Fixed settings

- Prompt: `A girl in a red dress dances gracefully in a warmly lit kitchen, keeping the same appearance, outfit, and cinematic lighting throughout the action.[7.5s]`
- Seed: `101`
- Checkpoint: `/home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt` with `--use_ema`
- Config: `configs/self_forcing_dmd.yaml` (480x832, four denoising steps, three latent frames per block, `local_attn_size=6`, `sink_size=1`)
- Target block: 8; source blocks: 2, 3, 4; retrieval count: 1
- Output: 30 latent frames, 117 decoded frames at 16 FPS (7.3125 seconds)
- Source-frame pool: global latent IDs 3--11. Coherent history selected ID 11; seeded random history selected ID 6. Neither selection is the sink (0), retained recent frame (20), or current frames (21--23).

## Exact commands and results

All commands ran from `/home/sigasia2026/projects/infinity-rope`. GPU telemetry used:

```bash
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 1
```

### Baseline

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path outputs/noncontiguous_phase1/prompt.txt --output_folder outputs/noncontiguous_phase1/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 2,3,4 --noncontiguous-target-block 8 --noncontiguous-kv-mode baseline --noncontiguous-retrieval-count 1
```

- Output: `outputs/noncontiguous_phase1/baseline/0-0_ema.mp4`
- Runtime: 49.81 seconds; peak VRAM: 23,031 MiB; exit status: 0.
- Context: global IDs `[0, 19, 20, 21, 22, 23]`; ordering `[sink:0, recent:19, recent:20, current:21, current:22, current:23]`; RoPE positions `[0, 1, 2, 3, 4, 5]`; 9,360 tokens.

### Coherent history

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path outputs/noncontiguous_phase1/prompt.txt --output_folder outputs/noncontiguous_phase1/coherent_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 2,3,4 --noncontiguous-target-block 8 --noncontiguous-kv-mode coherent_history --noncontiguous-retrieval-count 1
```

- Output: `outputs/noncontiguous_phase1/coherent_history/0-0_ema.mp4`
- Runtime: 47.39 seconds; peak VRAM: 25,575 MiB; exit status: 0.
- Selected history ID: 11 (source block 4).
- Context: global IDs `[0, 11, 20, 21, 22, 23]`; ordering `[sink:0, history:11, recent:20, current:21, current:22, current:23]`; RoPE positions `[0, 1, 2, 3, 4, 5]`; 9,360 tokens.

### Random history

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path outputs/noncontiguous_phase1/prompt.txt --output_folder outputs/noncontiguous_phase1/random_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 2,3,4 --noncontiguous-target-block 8 --noncontiguous-kv-mode random_history --noncontiguous-retrieval-count 1
```

- Output: `outputs/noncontiguous_phase1/random_history/0-0_ema.mp4`
- Runtime: 46.52 seconds; peak VRAM: 25,575 MiB; exit status: 0.
- Selected history ID: 6 (source block 3), deterministically selected with seed 101.
- Context: global IDs `[0, 6, 20, 21, 22, 23]`; ordering `[sink:0, history:6, recent:20, current:21, current:22, current:23]`; RoPE positions `[0, 1, 2, 3, 4, 5]`; 9,360 tokens.

## Target-block comparison

The three-column video is baseline | coherent history | random history. It contains physical decoded frames 69--116, which correspond to latent blocks 7--10 inclusive under the 4x temporal VAE mapping.

- Output: `outputs/noncontiguous_phase1/block7_to10_baseline_coherent_random.mp4`
- Shape: 2496x480, 48 frames, 16 FPS, 3.0 seconds.
- Construction command:

```bash
ffmpeg -y -i outputs/noncontiguous_phase1/baseline/0-0_ema.mp4 -i outputs/noncontiguous_phase1/coherent_history/0-0_ema.mp4 -i outputs/noncontiguous_phase1/random_history/0-0_ema.mp4 -filter_complex "[0:v]trim=start_frame=69:end_frame=117,setpts=PTS-STARTPTS[b];[1:v]trim=start_frame=69:end_frame=117,setpts=PTS-STARTPTS[c];[2:v]trim=start_frame=69:end_frame=117,setpts=PTS-STARTPTS[r];[b][c][r]hstack=inputs=3[out]" -map "[out]" -an -c:v libx264 -crf 18 -pix_fmt yuv420p outputs/noncontiguous_phase1/block7_to10_baseline_coherent_random.mp4
```

Raw logs, runtime data, and one-second VRAM telemetry are retained beside every output. No settings were changed after a run began, no run failed, no retrieval-count-2 run was attempted, and Phase 2 was not implemented.

## Artifact hashes

```text
3bff83288824ed30914da557e0f125fd12c3c880b6ba8fea8d0420ce3ff883e0  baseline/0-0_ema.mp4
55a120d66fbb244d61dadc4ceae64d8c7521524b5763d52a561581295b9ef04d  coherent_history/0-0_ema.mp4
b9539da7aa63edfafaf7c6493711fa84eb7dc8ddf85d0bba8b274a16d46017eb  random_history/0-0_ema.mp4
ecc209925f8dd00fb6021300b569876ba73ea7dcdf2192caef409ca76b570d30  block7_to10_baseline_coherent_random.mp4
```
