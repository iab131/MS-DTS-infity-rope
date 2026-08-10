# Identity-Selectivity Oracle Source-Gate Record

Experiment ID: `E20260808-AMP-IDENTITY-SELECTIVITY-P1`  
Commit: `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`  
Status: **source-gated incomplete; manual-memory arms deliberately not run**.

## Research question and preregistered method

Can two manually selected clean A K/V frames restore three distinctive woman
attributes at first A2 under `sink_only` transition forgetting, while truck K/V
does not? The exact prompt is in
`ATTENTION_MEMORY_POLICY_IDENTITY_SELECTIVITY_PROMPT_20260808.txt`; A2 repeats
none of the three A attributes.

The planned four arms were normal baseline; hard_flush; hard_flush plus manual
A IDs `[6,7]`; and hard_flush plus manual truck IDs `[16,17]`. Manual arms
would use target block 7/current IDs `[18,19,20]`, all 30 injection layers,
`k=2`, `prepend`, and
`[sink,memory,memory,current×3]` = six frames / 9,360 tokens. Automatic
routing, decay, archive, and consolidation were to remain disabled.

## Executed commands and raw outcomes

Only the source-gating baseline and hard-flush commands executed:

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/ATTENTION_MEMORY_POLICY_IDENTITY_SELECTIVITY_PROMPT_20260808.txt --output_folder outputs/attention_memory_policy_identity_selectivity/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --save-clean-latent-blocks 6,7,8 --save-raw-decoded

conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/ATTENTION_MEMORY_POLICY_IDENTITY_SELECTIVITY_PROMPT_20260808.txt --output_folder outputs/attention_memory_policy_identity_selectivity/hard_flush --seed 101 --num_samples 1 --save_with_index --output_index 0 --attention-memory-policy --no-memory-retrieval --memory-context-mode prepend --memory-k 2 --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --memory-local-retention sink_only --no-memory-decay --no-memory-archive --no-memory-consolidation --no-memory-transition-auto-retrieval --memory-policy-log outputs/attention_memory_policy_identity_selectivity/hard_flush/memory_policy.jsonl --save-clean-latent-blocks 6,7,8 --save-raw-decoded
```

| Arm | Exit | Runtime | Peak sampled VRAM | Raw decoded SHA-256 | MP4 SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 0 | 48.14 s | 23,031 MiB | `83182cf20a135467cbc80d1f5c49f9efdaecb9bd1aac50b3305fd960a0595abd` | `c33487ee8c532ad70d34e1f715b5f1f266126ea0559ca66369123e26428450d3` |
| hard_flush | 0 | 51.90 s | 23,243 MiB | `e4430d8f5d5929fbb5fbd93ec0c074101fa0c4decd3133230e0867093de3de47` | `c4a3f06b7d12873d254de8941cfd400f396ce4483e627fe7757b02895da6ad5c` |

The saved raw tensors are 117 RGB frames at 16 FPS. Hard_flush first differs
from baseline at frame 33 (A→B); pre-frame-33 max absolute RGB difference is
0.0. Overall raw RGB MAE is 0.094046980, PSNR 15.160388 dB, and maximum
absolute difference 1.0. This is a transition-policy difference, not a memory
retrieval result.

## Source gate (human visual review of decoded hard-flush frames)

The gate required every selected A source frame to visibly contain: (1) a
bright white left hair streak, (2) a small red star below the left eye, and
(3) the large yellow chest patch. It required B source frames to contain only
the truck, with no people or humanoids.

| Candidate | Review result | Gate |
| --- | --- | --- |
| A IDs 0--8 | Black bob and yellow chest patch are visible. No white hair streak or small red under-eye star is visible; the model consistently substitutes a prominent red horizontal/diagonal eye-level beam. | **Fail** |
| B IDs 16,17 | Bright blue/turquoise vintage pickup only; no person/humanoid is visible. The greenhouse persists rather than the requested desert. | Entity/no-person pass; environmental-control caveat |

Artifacts: `hard_flush/a_source_ids0_5_frames1_24.png` SHA-256
`dfe2d15f9066e6a4faebc76680205bff400c332463f671b7d2c240c429c5bbb0`,
`a_source_ids6_7_frames25_32.png` SHA-256
`8fd082cc28a1f94fbeb789860076696fead45d1625b0d5bd8b218befe6efd08b`,
and `a_source_id8_frame33.png` SHA-256
`6ac5f6d58f7e21e41d728862716f991cfe87644c36eda8e72fa9466287af495c`.
Truck contact sheet `b_source_ids16_17_frames61_68.png` SHA-256
`41fa90b09c33b518d1224b7c171af1166dca3916fbf80887bddf046f0dc69c7a`.

## A2 visual screen (human review; no retrieval arm)

The two-row still sheet has baseline then hard_flush rows; each row contains
original A frame 28, early A2 frame 69, and mid A2 frame 93:
`outputs/attention_memory_policy_identity_selectivity/source_gate_baseline_hard_flush_A_early_mid_A2.png`
(SHA-256 `a02dcef38d1d6bae4962fb9fd93313a0d3eb2ed73e6c1cb8cd007f086b87dddf`).

- White hair streak: absent in the source and not observed in screened A2 frames.
- Red face mark: the requested small star is absent in the source and not observed in screened A2 frames; the source instead has the red eye beam.
- Yellow chest patch: visible in source; persists visibly in the screened hard-flush mid-A2 frame, but this is not a memory effect because no retrieval arm ran.
- Unwanted greenhouse recall: strong in B and A2; the greenhouse remains visible behind truck/woman content.
- Snowy-mountain compliance: absent in screened early/mid A2 frames.

The two-arm B→A2 source-gate comparison is
`outputs/attention_memory_policy_identity_selectivity/transition_B_to_A2_baseline_hard_flush_source_gate.mp4`
(SHA-256 `a228390cf434f8e71278a3f3f2fbfc8ab86cf5d3dd119c6ff9b0ff126386352c`).
No four-arm still sheet or comparison video was created because that would
mislabel invalid A K/V as “correct memory.”

## Conclusion and next decision

This run does not test identity selectivity: the requested A identity anchors
were not present in any admissible source frame. It does demonstrate a prompt-
adherence/source-validity failure and greenhouse leakage before retrieval.
Choose a new independently source-gated prompt/seed before launching manual
retrieval; do not infer memory selectivity from this incomplete matrix.
