# Combined-Policy Short A/B/A2 Oracle: Raw Experiment Record

Experiment: `E20260807-AMP-SHORT-ABA-P1`  
Commit under test: `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`  
Status: completed; visual conclusions pending human review.

## Fixed method

The verbatim prompt is `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`.
All successful arms used EMA, `configs/self_forcing_dmd.yaml`, seed 101,
480x832, four denoising steps, 16 FPS, `local_attn_size=6`, and `sink_size=1`.
The requested/scheduled block durations are A/B/A2 = 2.25/2.25/3.0 seconds
(blocks `[3,3,4]`). The saved raw tensor has 117 RGB frames: A is 0--32
(2.0625 s), B 33--68 (2.25 s), and A2 69--116 (3.0 s), for 7.3125 seconds.
The initial temporal VAE boundary accounts for the requested/scheduled versus
saved-RGB total difference.

| Arm | Policy at cuts | Retrieval at block 7 | Selected IDs | Target context |
| --- | --- | --- | --- | --- |
| baseline | existing inference | disabled | none | normal local path |
| hard_flush | `sink_only`, cross-attention reset | disabled | none | no injected history |
| correct_memory | `sink_only`, cross-attention reset | manual | 6,7 (A) | `[sink:0,history:6,history:7,current:18,current:19,current:20]` |
| wrong_memory | `sink_only`, cross-attention reset | manual | 16,17 (B car) | `[sink:0,history:16,history:17,current:18,current:19,current:20]` |

For all policy arms, automatic transition routing, decay, archive, and
consolidation were disabled. C/D used all transformer layers `0..29` as
injection layers, while descriptor layers remained `0,1,5,14,16`.

## Exact successful commands

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --save-clean-latent-blocks 6,7,8 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/hard_flush --seed 101 --num_samples 1 --save_with_index --output_index 0 --attention-memory-policy --no-memory-retrieval --memory-context-mode prepend --memory-k 2 --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --memory-local-retention sink_only --no-memory-decay --no-memory-archive --no-memory-consolidation --no-memory-transition-auto-retrieval --memory-policy-log outputs/attention_memory_policy_short_aba/hard_flush/memory_policy.jsonl --save-clean-latent-blocks 6,7,8 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/correct_memory --seed 101 --num_samples 1 --save_with_index --output_index 0 --attention-memory-policy --memory-context-mode prepend --memory-k 2 --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --memory-local-retention sink_only --no-memory-decay --no-memory-archive --no-memory-consolidation --no-memory-transition-auto-retrieval --memory-manual-frame-ids 6,7 --memory-manual-target-blocks 7 --memory-policy-log outputs/attention_memory_policy_short_aba/correct_memory/memory_policy.jsonl --save-clean-latent-blocks 6,7,8 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/wrong_memory --seed 101 --num_samples 1 --save_with_index --output_index 0 --attention-memory-policy --memory-context-mode prepend --memory-k 2 --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --memory-local-retention sink_only --no-memory-decay --no-memory-archive --no-memory-consolidation --no-memory-transition-auto-retrieval --memory-manual-frame-ids 16,17 --memory-manual-target-blocks 7 --memory-policy-log outputs/attention_memory_policy_short_aba/wrong_memory/memory_policy.jsonl --save-clean-latent-blocks 6,7,8 --save-raw-decoded
```

## Raw outcomes

| Arm | Exit | Runtime (s) | Peak device VRAM (MiB) | Raw decoded SHA-256 | MP4 SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 0 | 47.56 | 23,031 | `3996069c408677c0e96eb89573b5b57bd544ca549b0b4b9177aaeebbf0b224db` | `c5b4f2e7ed7c40f9f68ce8c61e2d6bda172c63225a56ee2203f21a54a6fd09a5` |
| hard_flush | 0 | 51.14 | 23,243 | `d2918757bfc746234ec09d04e51873057db42a16392f2ca1b9727f529d2ac198` | `d35ad00d9c599f5e3c7c1aba8e9d6943816b5b4e32e29149893022e4bd121ebe` |
| correct_memory | 0 | 52.51 | 23,243 | `1a67005d1355e67a27033f2cebf9d82c1ea672ff62e0fd5230f3938d287e9a2b` | `8e82507c06a8c30279ba86ccac5124b04163d2652ee95780f075a8692470e70c` |
| wrong_memory | 0 | 53.34 | 23,243 | `aa024fe1214d11f80fe9efdc969e31b610660f2ef572a3588d32683639e7bd3c` | `fa46c1ba94f02105f6f6e4d0892306b3ab2bcf9fd6039a122f5c622f2182a91d` |

Peak VRAM is sampled device-wide telemetry; run order/warm-up prevents a
performance claim.

## Causality and logged context

`hard_flush` versus both manual-memory arms is exactly equal through raw RGB
frame 68 (`max abs = 0.0`) and through saved clean block 6 (`max abs = 0.0`).
Both first diverge at raw frame 69 and clean block 7/local latent 0. The
saved block-7/block-8 clean-latent maxima are 5.125/5.625 for correct memory
and 4.9375/5.171875 for wrong memory.

At target block 7, correct/wrong policy JSONL records respectively show
manual override, query source `pre_transition_raw`, source scenes 0/1, ordered
IDs `[6,7]`/`[16,17]`, positions `[0,1,2,3,4,5]`, zero local frames, three
current frames, six total frames, and 9,360 total tokens. The sink remains
slot zero. Automatic routing was not used.

| Raw RGB comparison | First differing frame | Pre-first max abs | All-frame MAE | PSNR (dB) | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline vs hard_flush | 33 | 0.0 | 0.181675345 | 10.593921 | 1.0 |
| hard_flush vs correct_memory | 69 | 0.0 | 0.101799488 | 12.677845 | 1.0 |
| hard_flush vs wrong_memory | 69 | 0.0 | 0.099336967 | 13.134525 | 1.0 |
| correct_memory vs wrong_memory | 69 | 0.0 | 0.094375961 | 13.703700 | 1.0 |

The raw tensors are RGB on `[0,1]`; MAE and PSNR therefore use peak value 1.

## Comparisons and failures

- A→B comparison (raw RGB frames 21--56, blocks 3--5):
  `outputs/attention_memory_policy_short_aba/transition_A_to_B_baseline_hard_flush_correct_wrong.mp4`,
  SHA-256 `64fcd67f4ace14b49e566ac33d6ef4ad53047bd95ea61d09d041a33eef8fe750`.
- B→A2 comparison (raw RGB frames 57--92, blocks 6--8):
  `outputs/attention_memory_policy_short_aba/transition_B_to_A2_baseline_hard_flush_correct_wrong.mp4`,
  SHA-256 `2f514cc955d59e372b737534524db2342aa21d3b0e561d906d4a4e312d2c8d52`.
- Preserved failed attempts: baseline attempt 1 exited 1 because the expected
  relative `wan_models` link was absent; hard_flush attempt 1 exited 1 from an
  undefined scene-ID logging variable; correct_memory attempt 1 exited 1 from
  JSON serialization of a target-block set. The local model link was restored
  without changing settings, and the two scaffold bugs were regression-tested
  and amended into the commit before successful reruns.

## Interpretation limit

This one-seed experiment establishes target-causal perturbation under the
combined hard-flush plus two-frame manual-recall policy. It does not establish
identity recovery, semantic selectivity, quality improvement, or a correct-
versus-wrong visual difference. Human visual review is pending; no additional
memory mechanism was added or tuned after these videos.
