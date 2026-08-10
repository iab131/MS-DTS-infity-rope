# Combined-Policy Short A/B/A2 Oracle Preparation

Experiment ID: `E20260807-AMP-SHORT-ABA-P1`  
Status: prepared; **not run**.

This is an interaction test of transition forgetting and historical recall,
not a semantic-retrieval validation. It reuses the exact prompt from
`NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`.

## Fixed schedule and source admission

| Shot | Requested / scheduled-block nominal / saved raw RGB | Blocks | Global latent IDs / raw RGB frame indices |
| --- | --- | --- | --- |
| A, Amara / greenhouse | 2.25 s / 2.25 s / 2.0625 s | 1--3 | 0--8 / 0--32 |
| B, yellow car / desert prompt | 2.25 s / 2.25 s / 2.25 s | 4--6 | 9--17 / 33--68 |
| A2, Amara / observatory | 3.0 s / 3.0 s / 3.0 s | 7--10 | 18--29 / 69--116 |

Target is first A2 block 7, current latent IDs `[18,19,20]`. Selected A
memory IDs are `[6,7]`; selected B memory IDs are `[16,17]`.

Human source review from the completed prior baseline checked both pairs.
`a_source_ids6_7_frames25_32.png` visibly contains Amara only;
`b_source_ids16_17_frames61_68.png` visibly contains the yellow sports car
and no woman/humanoid. Both retain greenhouse-background leakage, so only the
entity/no-person source condition is admitted. Contact-sheet SHA-256 values:
`ed53fe64de5c740137a875ed7a6c22375d15741a0ab5a078e77b367009d1dd8c` and
`a4b32b54ec6c00991ffe44cc77529138a0006a4cc1a485b8445ec4a2b6ccdb61`.

## Policy contract

Descriptor layers are `0,1,5,14,16`; injection layers are all 30 layers
`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29`
so C/D have the specified context at every layer. Automatic transition-block
routing is explicitly disabled. Archive, consolidation, and decay are also
disabled; manual retrieval remains enabled only at block 7.

At block 7, after `sink_only`, `prepend` produces exactly:

```text
[sink:0, memory:<id1>, memory:<id2>, current:18, current:19, current:20]
positions [0,1,2,3,4,5]
6 latent frames / 9,360 tokens
```

No retained non-sink local frames exist at that boundary. This is matched only
because `sink_only` leaves `[sink,current×3]` before the two-frame prepend;
later blocks may have a different prepend span and must be read from JSONL.

## Exact commands (not run)

All arms use EMA, the same config, seed 101, 480x832, four denoising steps,
one sample, and only blocks 6--8/raw decoded diagnostics.

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/baseline --seed 101 --num_samples 1 --save_with_index --output_index 0 --save-clean-latent-blocks 6,7,8 --save-raw-decoded
```

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/hard_flush --seed 101 --num_samples 1 --save_with_index --output_index 0 --attention-memory-policy --no-memory-retrieval --memory-context-mode prepend --memory-k 2 --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --memory-local-retention sink_only --no-memory-decay --no-memory-archive --no-memory-consolidation --no-memory-transition-auto-retrieval --memory-policy-log outputs/attention_memory_policy_short_aba/hard_flush/memory_policy.jsonl --save-clean-latent-blocks 6,7,8 --save-raw-decoded
```

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/correct_memory --seed 101 --num_samples 1 --save_with_index --output_index 0 --attention-memory-policy --memory-context-mode prepend --memory-k 2 --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --memory-local-retention sink_only --no-memory-decay --no-memory-archive --no-memory-consolidation --no-memory-transition-auto-retrieval --memory-manual-frame-ids 6,7 --memory-manual-target-blocks 7 --memory-policy-log outputs/attention_memory_policy_short_aba/correct_memory/memory_policy.jsonl --save-clean-latent-blocks 6,7,8 --save-raw-decoded
```

```bash
conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt --output_folder outputs/attention_memory_policy_short_aba/wrong_memory --seed 101 --num_samples 1 --save_with_index --output_index 0 --attention-memory-policy --memory-context-mode prepend --memory-k 2 --memory-descriptor-layers 0,1,5,14,16 --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --memory-local-retention sink_only --no-memory-decay --no-memory-archive --no-memory-consolidation --no-memory-transition-auto-retrieval --memory-manual-frame-ids 16,17 --memory-manual-target-blocks 7 --memory-policy-log outputs/attention_memory_policy_short_aba/wrong_memory/memory_policy.jsonl --save-clean-latent-blocks 6,7,8 --save-raw-decoded
```

## Planned interpretations and stop conditions

- A vs B isolates `sink_only` transition forgetting with no retrieval.
- B vs C/D isolates manual historical K/V at first A2 with the transition
  policy fixed.
- Automatic transition routing remains disabled and cannot be credited or
  blamed for any observed outcome.
- Preserve any failure unchanged. Before interpretation, verify raw equality
  for B/C/D through clean blocks 1--6, then derive the decoded pre-target
  boundary from the saved raw tensor rather than reusing the prior block-8
  oracle's frame-80 boundary. Parse JSONL context records and separate RGB
  from human visual observations.
