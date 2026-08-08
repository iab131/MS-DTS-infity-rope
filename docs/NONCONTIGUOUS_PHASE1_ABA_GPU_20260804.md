# Matched Phase 1 Oracle A-B-A GPU Matrix

Run date: 2026-08-04

Commit under test: `361ed7e` (`feat: add oracle noncontiguous KV experiment`)

## Fixed settings

- Prompt: `A woman with long black hair wearing a red dress dances in a warmly lit kitchen, cinematic medium shot.[2.25s#] | A blue robot in a rain-soaked neon alley turns toward camera, cinematic medium shot.[2.25s#] | The same woman with long black hair wearing the same red dress returns to the warmly lit kitchen and continues dancing, cinematic medium shot.[3s]`
- Config: `configs/self_forcing_dmd.yaml`; 480x832; four denoising steps;
  `local_attn_size=6`; `sink_size=1`.
- Checkpoint: `/home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt`
  with EMA weights.
- Seed: 101. Target block: 8. Source blocks: 3 and 6. Retrieval count: 1.
- Every target context is six latent frames / 9,360 tokens. Baseline is
  `[sink:0, recent:19, recent:20, current:21, current:22, current:23]`;
  same entity is `[sink:0, history:8, recent:20, current:21, current:22, current:23]`;
  wrong entity is `[sink:0, history:17, recent:20, current:21, current:22, current:23]`.
- All target contexts use temporary RoPE positions `[0, 1, 2, 3, 4, 5]`.

## Commands and outcomes

The temporary harness performed the equivalent prepared inference arguments,
saved raw decoded tensors before MP4 encoding, and emitted per-block clean
latent snapshots. The two successful history-run commands were:

```bash
conda run -n wan python /tmp/run_infinity_rope_aba_matrix.py same_entity_history 2>&1 | tee outputs/noncontiguous_phase1_aba/matrix_same_entity_history.log
conda run -n wan python /tmp/run_infinity_rope_aba_matrix.py wrong_entity_history 2>&1 | tee outputs/noncontiguous_phase1_aba/matrix_wrong_entity_history.log
```

The first all-mode command was:

```bash
set -o pipefail; conda run -n wan python /tmp/run_infinity_rope_aba_matrix.py 2>&1 | tee outputs/noncontiguous_phase1_aba/matrix.log
```

It attempted to log before `outputs/noncontiguous_phase1_aba` existed. `tee`
failed, the outer command was interrupted (exit 130), and this is preserved as
the only harness failure. Its baseline inference had already completed and
saved a valid `run.json`, raw tensor, per-block snapshots, and MP4. No
generation setting was changed; the two history runs were then run individually
as above.

| Mode | Selected global latent ID | Runtime* | Peak VRAM (`nvidia-smi`) | Exit | Raw decoded SHA-256 | MP4 SHA-256 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| baseline | none | 11.581 s | 23,025 MiB | inference completed; outer combined harness exit 130 | `cd334157d2f34a5a833b68ab5624d5d9819df3586e7f6b4767b09f039624e0c4` | `339249f17b99dd1e9c62abb0696bd0da8c14b35cc6c2b1f666a4b1b3cb15b392` |
| same_entity_history | 8 (A/block 3) | 11.567 s | 23,025 MiB | 0 | `28ca0132bb61c3c083b4367fd718116676b1c4a1a100264f5e0b10e325091d78` | `123dec46044ee8ae455a14e56cfd157e168678c9bcdc083e37d4da31a6a32a2b` |
| wrong_entity_history | 17 (B/block 6) | 11.611 s | 23,025 MiB | 0 | `b8d166803229291a0efb9cc3d0a45cadd2d933d703e4075659a84cf1f256da43` | `e2bdc5623d54a5b7b3e99582b6252e50788b20f76e24f265325c86db11a1521b` |

\*Runtime is harness inference through raw-tensor save and MP4 write; model
load time is excluded. Torch peak allocated memory was 19,890.19 MiB in all
three cases.

Outputs:

- `outputs/noncontiguous_phase1_aba/baseline/0-0_ema.mp4`
- `outputs/noncontiguous_phase1_aba/same_entity_history/0-0_ema.mp4`
- `outputs/noncontiguous_phase1_aba/wrong_entity_history/0-0_ema.mp4`
- `outputs/noncontiguous_phase1_aba/metrics.json` (raw hashes, full per-block
  latent hashes, and exact differences)

## Causality and divergence

Saved clean latents are exactly equal through blocks 1--7 in both history
cases. Raw decoded RGB is exactly equal through decoded frame 80, before
target block 8. The first divergence is therefore causally aligned with the
target context change:

| Comparison | First divergent clean latent | First divergent decoded frame | Pre-block-8 max abs | Overall raw decoded max abs |
| --- | --- | ---: | ---: | ---: |
| baseline vs same entity | block 8, local index 0, global ID 21; clean-latent max abs 4.078125 | 81 | 0.0 | 1.0 |
| baseline vs wrong entity | block 8, local index 0, global ID 21; clean-latent max abs 3.617188 | 81 | 0.0 | 1.0 |

## Blocks 7--10 side-by-side

The comparison is baseline | same entity | wrong entity, decoded frames
69--116 (48 frames, 16 FPS), 2496x480:

```bash
ffmpeg -y -i outputs/noncontiguous_phase1_aba/baseline/0-0_ema.mp4 -i outputs/noncontiguous_phase1_aba/same_entity_history/0-0_ema.mp4 -i outputs/noncontiguous_phase1_aba/wrong_entity_history/0-0_ema.mp4 -filter_complex "[0:v]trim=start_frame=69:end_frame=117,setpts=PTS-STARTPTS[b];[1:v]trim=start_frame=69:end_frame=117,setpts=PTS-STARTPTS[s];[2:v]trim=start_frame=69:end_frame=117,setpts=PTS-STARTPTS[w];[b][s][w]hstack=inputs=3[out]" -map "[out]" -an -c:v libx264 -crf 18 -pix_fmt yuv420p outputs/noncontiguous_phase1_aba/block7_to10_baseline_same_wrong.mp4
```

- Output: `outputs/noncontiguous_phase1_aba/block7_to10_baseline_same_wrong.mp4`
- SHA-256: `abb8b07f777f8d6dd3c43ddf731b45a6a17ed7f8ac94fc1dec418e7989982eb5`

No descriptor routing, MemoryStore, or retrieval-count-2 experiment was run.
