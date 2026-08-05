# Matched Phase 1 Oracle A-B-A Preparation

Prepared on 2026-08-04. This document defines commands only. They have not
been launched.

## Fixed design

- Prompt file: `docs/NONCONTIGUOUS_PHASE1_ABA_PROMPT.txt`
- Exact prompt: `A woman with long black hair wearing a red dress dances in a warmly lit kitchen, cinematic medium shot.[2.25s#] | A blue robot in a rain-soaked neon alley turns toward camera, cinematic medium shot.[2.25s#] | The same woman with long black hair wearing the same red dress returns to the warmly lit kitchen and continues dancing, cinematic medium shot.[3s]`
- The live duration parser yields A/B/A2 block counts `[3, 3, 4]`; total
  duration is 7.5 seconds and the current T2V configuration yields 30 latent
  frames / 117 decoded frames.
- Fixed settings: seed `101`; `configs/self_forcing_dmd.yaml`; EMA checkpoint
  `/home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt`;
  480x832; four denoising steps; target block 8; source blocks 3 and 6;
  retrieval count 1; local attention 6; sink size 1.
- Target A2 block 8: current global latent IDs `[21, 22, 23]`; retained local
  recent ID `[20]`; sink ID `0`. All contexts remain 6 latent frames / 9,360
  tokens: `[sink, history-or-recent, recent, current, current, current]`.

## Oracle frame selections

| Run | Manual source | Global latent ID | Target ordering | Temporary RoPE positions |
| --- | --- | ---: | --- | --- |
| baseline | none | none | `[sink:0, recent:19, recent:20, current:21, current:22, current:23]` | `[0, 1, 2, 3, 4, 5]` |
| same_entity_history | A, block 3 final latent | 8 | `[sink:0, history:8, recent:20, current:21, current:22, current:23]` | `[0, 1, 2, 3, 4, 5]` |
| wrong_entity_history | B, block 6 final latent | 17 | `[sink:0, history:17, recent:20, current:21, current:22, current:23]` | `[0, 1, 2, 3, 4, 5]` |

IDs 8 and 17 are manually specified rather than routed by a descriptor. They
cannot select the sink, target-current frames, or retained recent frame.

## Exact commands (not run)

Run each from `/home/sigasia2026/projects/infinity-rope`.

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_ABA_PROMPT.txt --output_folder outputs/noncontiguous_phase1_aba/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 3,6 --noncontiguous-target-block 8 --noncontiguous-kv-mode baseline --noncontiguous-retrieval-count 1

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_ABA_PROMPT.txt --output_folder outputs/noncontiguous_phase1_aba/same_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 3,6 --noncontiguous-target-block 8 --noncontiguous-kv-mode same_entity_history --noncontiguous-history-frame-id 8 --noncontiguous-retrieval-count 1

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_ABA_PROMPT.txt --output_folder outputs/noncontiguous_phase1_aba/wrong_entity_history --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 3,6 --noncontiguous-target-block 8 --noncontiguous-kv-mode wrong_entity_history --noncontiguous-history-frame-id 17 --noncontiguous-retrieval-count 1
```

## Expected VRAM impact

Historical clean KV is now moved to CPU immediately after every selected
source clean pass. At target time, only the manually selected 1,560-token
frame for one transformer layer's K/V is copied back to GPU; it is transient
and never written to the persistent cache. At BF16, that K/V pair is about
9.14 MiB per executing layer. The prior GPU-side three-source experiment
peaked 2,544 MiB above its baseline (25,575 versus 23,031 MiB); this A-B-A
implementation should remove the multi-block source bank from peak VRAM.
This is an estimate, not a measured result.

The transformed sink remains in its original persistent-cache slot and is not
re-rotated. Persistent-cache indices, writes, and eviction logic are unchanged.
