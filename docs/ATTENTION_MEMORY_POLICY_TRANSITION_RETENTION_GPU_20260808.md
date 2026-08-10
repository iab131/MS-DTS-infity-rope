# Transition-retention ablation — woman to truck (2026-08-08)

## Scope

This is a single-seed, opt-in transition-attention ablation. It tests local
retention at the first woman-to-truck cut; it does not test descriptor routing,
historical retrieval, a MemoryStore, or identity recovery. Retrieval, archive,
consolidation, and decay were disabled in every arm.

Commit under test: `0efd5ad17e57f19ab2029ffc6da44be46e4a415d` plus the
uncommitted, focused `transition_no_sink` option in this worktree.

## Exact input and effective schedule

Prompt (verbatim, also in
`ATTENTION_MEMORY_POLICY_TRANSITION_RETENTION_PROMPT_20260808.txt`):

```text
A woman with a short black bob featuring one bright white streak on the left, a small red star mark under her left eye, and a green jacket with a large yellow circular chest patch, standing in a glass greenhouse, cinematic medium shot.[2.25s#] | A bright blue vintage pickup truck alone on a dry desert road, no people, no driver visible, no pedestrians, no humanoids, no faces, and no human figures, cinematic wide shot.[2.25s]
```

The live parser scheduled `[3,3]` latent blocks. A is blocks 1--3 / global
latent IDs 0--8; B is blocks 4--6 / IDs 9--17. The intervention is B block 4
(IDs 9--11); saved decoded B starts at raw RGB frame 33. The request is
2.25 s + 2.25 s; saved raw RGB is 69 frames at 16 FPS (4.3125 s), with A
frames 0--32 and B frames 33--68. This saved duration is not the requested
4.5 s total.

Fixed generation settings: Self-Forcing DMD EMA checkpoint
`/home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt`,
`configs/self_forcing_dmd.yaml`, seed 101, 480x832, four denoising steps,
six-frame local cache, and all 30 injection layers. Cross-attention reset
remained enabled at the cut. The policy was enabled solely to exercise the
retention switch; no historical K/V was injected.

## Arms and exact first-B-block attention contexts

| Arm | Retention setting | Context ordering | RoPE temporal positions | Frames / tokens |
| --- | --- | --- | --- | --- |
| Baseline | `sink+2` | `[sink:0, local:7, local:8, current:9, current:10, current:11]` | `[0,1,2,45,46,47]` | 6 / 9,360 |
| Sink+1 | `sink+1` | `[sink:0, local:8, current:9, current:10, current:11]` | `[0,1,45,46,47]` | 5 / 7,800 |
| Sink-only | `sink_only` | `[sink:0, current:9, current:10, current:11]` | `[0,45,46,47]` | 4 / 6,240 |
| Transition-no-sink (experimental) | `transition_no_sink` | `[current:9, current:10, current:11]` | `[45,46,47]` | 3 / 4,680 |

`transition_no_sink` sets the first new-scene block's usable local-cache end
to zero: it excludes the transformed sink and all previous-scene local K/V
only for that block. The ordinary persistent cache write then stores the new
B block starting at slot zero, so following B blocks resume normal cache
behavior. The transformed old sink is not rotated or mutated; it is simply
not offered to that one attention call.

## Exact commands

Each `<arm>` was one of `baseline_sink_plus2`, `sink_plus1`, `sink_only`, or
`transition_no_sink`, with `<retention>` respectively `sink+2`, `sink+1`,
`sink_only`, or `transition_no_sink`.

```bash
conda run -n wan python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --use_ema \
  --data_path docs/ATTENTION_MEMORY_POLICY_TRANSITION_RETENTION_PROMPT_20260808.txt \
  --output_folder outputs/attention_memory_policy_transition_retention/<arm> \
  --seed 101 --num_samples 1 --save_with_index --output_index 0 \
  --attention-memory-policy --no-memory-retrieval \
  --memory-context-mode prepend --memory-k 2 \
  --memory-descriptor-layers 0,1,5,14,16 \
  --memory-injection-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --memory-local-retention <retention> \
  --no-memory-decay --no-memory-archive --no-memory-consolidation \
  --no-memory-transition-auto-retrieval \
  --memory-policy-log outputs/attention_memory_policy_transition_retention/<arm>/memory_policy.jsonl \
  --save-clean-latent-blocks 3,4,5 --save-raw-decoded
```

## Raw run record

All four commands exited 0. The device-wide sampled peak was 23,243 MiB in
each arm; it is not a per-process performance measurement.

| Arm | Wall runtime (s) | Raw RGB SHA-256 | Output |
| --- | ---: | --- | --- |
| Baseline | 50.90 | `bcb44e987c0c01eb6a8468e9f1335aac60212759bfdaa7001db66b2d5037164e` | `outputs/attention_memory_policy_transition_retention/baseline_sink_plus2/0-0_ema.mp4` |
| Sink+1 | 47.89 | `fbb8c47df98a06c91cf76709df61bb0dc5126ae60b5002f423457523c037e0a3` | `outputs/attention_memory_policy_transition_retention/sink_plus1/0-0_ema.mp4` |
| Sink-only | 47.08 | `ee3795bdfe280249c0c6eaa4e2352fdfe23c62ca6dc8cc09162c043b5aa3cbc4` | `outputs/attention_memory_policy_transition_retention/sink_only/0-0_ema.mp4` |
| Transition-no-sink | 45.31 | `5897dfd78bd7cd39862629c79b7a90e06d3ffb447e47736f93079c379bf03e50` | `outputs/attention_memory_policy_transition_retention/transition_no_sink/0-0_ema.mp4` |

Clean block 3 is exactly equal in all arms (SHA-256
`83fc30c10cec125c28605c7aba7b7df4db77c6307726a896f930ec20de9d97d0`).
Blocks 4 and 5, and raw RGB, differ in every non-baseline arm; the first
causal divergence is therefore the intended first B block / raw frame 33.
Per-arm clean snapshots, raw decoded tensors, JSONL context logs, `run.log`,
and `vram.csv` are colocated with each output.

## Human visual review (not a metric)

Reviewed synchronized frames 33--44 and frames 33--66 sampled every three
frames. In the table, “settled” means a truck-only desert scene without a
woman or greenhouse, not merely a recognizable truck.

| Arm | Woman/person leakage into B | Old greenhouse/background | Truck completeness / wheel geometry | Motion / stabilization after cut |
| --- | --- | --- | --- | --- |
| Baseline sink+2 | Clear woman's head remains in the truck windshield in early B frames. | Persists through all sampled B frames. | Frontal truck resolves, but its cabin is contaminated; wheels are mostly outside the frontal crop. | No requested vehicle motion exists to score; it never reaches a settled requested desert scene in sampled B. |
| Sink+1 | The same visible head-in-windshield leakage remains early. | Persists through all sampled B frames. | Frontal truck is recognizable but contaminated; wheel geometry is not cleanly assessable in the crop. | No settled requested desert scene in sampled B. |
| Sink-only | First transition frame blends the woman; no clearly visible woman remains in the following sampled truck frames. | Persists through all sampled B frames. | A clean frontal pickup is visible after the blend, but wheels are mostly off-frame. | Stable truck framing after the early blend, yet not a settled requested desert scene because the greenhouse remains. |
| Transition-no-sink | Frame 33 is a residual cut blend; from frame 34 onward sampled frames show no person. | Gone from frame 34 onward sampled frames. | Complete side-view blue pickup with two round wheels in the desert. | Static wide-shot compliance is visually stable from frame 34 onward; settled one decoded frame after the cut. |

The intended four-way transition video is
`outputs/attention_memory_policy_transition_retention/comparison/A_to_B_four_way_frames_21_56_baseline_sinkplus2_sinkplus1_sinkonly_no_sink.mp4`.
The requested synchronized first-12-B-frame sheet is
`outputs/attention_memory_policy_transition_retention/comparison/first12_B_frames_33_44_baseline_sinkplus2_sinkplus1_sinkonly_no_sink.png`;
row order is baseline, sink+1, sink-only, transition-no-sink. Additional
early and temporal review sheets are in the same directory.

## Limited conclusion

At this prompt and seed, retaining the two or one recent non-sink frames
produces early human leakage inside the truck and persistent greenhouse
background. Sink-only removes the obvious persistent human after the first
blended frame but not the greenhouse. The experimental first-block
no-sink policy is the only arm that reaches a truck-only desert scene in the
reviewed B frames. This is one-seed visual evidence that recent non-sink local
K/V is a strong contributor to this entity carry-over; it does not establish
that the sink is the only source of background persistence, a general
transition policy, or a memory-retrieval result.
