# Non-Contiguous KV Paper Ledger

This append-only ledger is the paper-facing source of prompts, exact settings,
and observed results. “Not run” is never evidence.

## 2026-08-04: Short single-scene tolerance matrix

- Prompt: `A girl in a red dress dances gracefully in a warmly lit kitchen, keeping the same appearance, outfit, and cinematic lighting throughout the action.[7.5s]`
- Settings: seed 101; target block 8; sources 2,3,4; retrieval count 1;
  480x832; 30 latent frames / 117 decoded frames.
- Result: matched 9,360-token runs completed. The original decoded MP4 matrix
  differed before target block 8; it is not target-causal evidence. See
  `NONCONTIGUOUS_PHASE1_GPU_20260804.md`.

## 2026-08-04: Short oracle A-B-A matrix

- Prompt: `A woman with long black hair wearing a red dress dances in a warmly lit kitchen, cinematic medium shot.[2.25s#] | A blue robot in a rain-soaked neon alley turns toward camera, cinematic medium shot.[2.25s#] | The same woman with long black hair wearing the same red dress returns to the warmly lit kitchen and continues dancing, cinematic medium shot.[3s]`
- Settings: seed 101; target block 8; sources 3,6; same entity ID 8; wrong
  entity ID 17; retrieval count 1; 6 frames / 9,360 tokens.
- Result: raw outputs are identical through block 7 / decoded frame 80. Both
  history modes first diverge at global latent 21 / decoded frame 81. See
  `NONCONTIGUOUS_PHASE1_ABA_GPU_20260804.md`.

## 2026-08-07: Long-gap oracle A-B-A matrix

- Prompt: recorded verbatim in
  `NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt`.
- Planned settings: seed 101; target block 34; source blocks 13,15; same
  entity ID 38; wrong entity ID 44; retrieval count 1; 6 frames / 9,360
  tokens; snapshots only for blocks 33--35 plus raw decoded output.
- Status: **not run**. Exact effective block schedule and commands are in
  `NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PREPARATION_20260807.md`.

## Ledger protocol (added 2026-08-07)

Each experiment record below is append-only. Raw artifacts, measurements, and
exit statuses are recorded before interpretation. Requested prompt durations
and effective model-quantized durations are separate fields. “Human review” is
never inferred from a metric and is left explicitly unreviewed until a person
records an observation.

## Running Methods details

- Model path: `/home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt`,
  EMA weights, `configs/self_forcing_dmd.yaml`, 480x832, four denoising steps,
  BF16, seed 101 unless a record says otherwise.
- The causal configuration uses three latent frames per block, 1,560 tokens
  per latent frame, `local_attn_size=6`, and `sink_size=1`; matched target
  context is always six frames / 9,360 tokens.
- Source clean-pass K/V is raw, cloned before CPU offload, and transient. At
  target time only selected frame K/V is materialized on GPU per transformer
  layer. Persistent cache writes, indices, eviction, and the transformed sink
  slot are unchanged; the sink is never re-RoPEd.
- Matched retrieval replaces recent non-sink frames: baseline
  `[sink,recent,recent,current×3]`; one-frame history
  `[sink,history,recent,current×3]`. History selection uses a private Python
  RNG; it does not advance Torch's generation RNG.
- `--save-clean-latent-blocks` saves only specified clean-pass blocks.
  `--save-raw-decoded` saves the decoded tensor before MP4 conversion. These
  exports are diagnostics, not model inputs.

## Claim → evidence status

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| Matched historical K/V injection changes generation only when the target is reached. | Supported, single seed | Short A-B-A raw clean latents equal baseline through block 7; first divergence is target block 8/global latent 21. |
| The historical slot does not enlarge attention context. | Supported, implementation and logs | All reported target contexts are six frames / 9,360 tokens; focused tests cover assembly. |
| CPU source-K/V offload preserves baseline inference. | Supported, single seed | Normal, opt-in baseline, and repeated opt-in baseline had exact clean-latent, persistent-K/V, and pre-MP4 RGB equality in the causality gate. |
| Same-entity history improves identity persistence. | Unsupported | No human review, identity metric, multi-seed result, or long-gap result yet. |
| Wrong-entity history damages identity persistence. | Unsupported | Output divergence alone is not an identity-quality measure. |
| The original short tolerance MP4 matrix is target-causal. | Contradicted | Its decoded RGB first differed before target block 8; it is retained as a confounded negative result. |
| Long-gap identity recovery works. | Unsupported | E20260807-LG-P1 is planned, not observed. |

## Figure / table index

| ID | Artifact | Intended later use | Status |
| --- | --- | --- | --- |
| F1 | `outputs/noncontiguous_phase1/block7_to10_baseline_coherent_random.mp4` | Original tolerance comparison | Confounded before target; limitations only |
| T1 | `docs/NONCONTIGUOUS_PHASE1_GPU_20260804.md` | Original matched settings / telemetry table | Recorded |
| F2 | `outputs/noncontiguous_phase1_aba/block7_to10_baseline_same_wrong.mp4` | Short oracle qualitative triptych | Raw causal gate passed; no human review logged |
| T2 | `outputs/noncontiguous_phase1_aba/metrics.json` | Short oracle raw tensor hashes and divergence table | Recorded |
| T3 | `outputs/noncontiguous_phase1_causality_gate/metrics.json` | Capture/offload causality gate table | Recorded |
| F3/T4 | `outputs/noncontiguous_phase1_long_gap/` | Long-gap target-adjacent comparison and metrics | Planned, not run |

## E20260804-SHORT-TOLERANCE (backfilled raw record)

- Date / commit: 2026-08-04; checkout `44fa22c` (experiment code commits
  `3f004b4`, `46880a1`).
- Research question / hypothesis: can a selected historical clean-K/V frame
  replace a recent local slot at block 8 while retaining the baseline token
  budget? Hypothesis: target output changes under historical context.
- Exact method: baseline vs `coherent_history` vs seeded `random_history`;
  source blocks 2,3,4; target 8; retrieval count 1. Coherent selected global
  ID 11; random selected ID 6.
- Prompt and effective schedule: `A girl in a red dress dances gracefully in a
  warmly lit kitchen, keeping the same appearance, outfit, and cinematic
  lighting throughout the action.[7.5s]`; 30 latent / 117 decoded frames.
- Settings: EMA checkpoint/config/seed/resolution/steps/cache as Running
  Methods. Contexts were 6 frames / 9,360 tokens. Exact commands, output
  paths, hashes, runtimes, and exit statuses are retained verbatim in
  `NONCONTIGUOUS_PHASE1_GPU_20260804.md`.
- Quantitative raw result: baseline/coherent and baseline/random decoded RGB
  first differed at frame 45; coherent/random at frame 49, before target
  decoded frame 81. See `INFERENCE_FINDINGS.md` for MAE/PSNR.
- Human review: not recorded.
- Confound / interpretation: pre-target divergence prevents a strict
  target-only visual comparison. Conclusion: this run demonstrates only that
  outputs differ; it cannot support a target-causal or quality claim.
- Next experiment motivated: short oracle A-B-A causality gate.

## E20260804-SHORT-ABA (backfilled raw record)

- Date / commit: 2026-08-04; commit `361ed7e`.
- Research question / hypothesis: does manually selected same/wrong entity
  history change only the returning-shot target under a fixed 9,360-token
  budget? Hypothesis: pre-target tensors remain identical and target tensors
  diverge.
- Exact method: baseline, `same_entity_history` ID 8 from A/block 3, and
  `wrong_entity_history` ID 17 from B/block 6; target block 8; source blocks
  3,6; retrieval count 1. Target orderings are `[0,19,20,21,22,23]`,
  `[0,8,20,21,22,23]`, and `[0,17,20,21,22,23]` respectively.
- Prompt / effective boundaries: A 2.25 s blocks 1--3, B 2.25 s blocks 4--6,
  A2 3.0 s blocks 7--10; 30 latent / 117 decoded frames. Exact prompt,
  commands, checkpoint/config/seed/resolution/steps, runtime, VRAM, output
  paths, hashes, and the exit-130 logging-pipe confound are retained verbatim
  in `NONCONTIGUOUS_PHASE1_ABA_GPU_20260804.md`.
- Quantitative raw result: same and wrong histories are exactly equal to
  baseline through blocks 1--7 and decoded frame 80. Both first diverge at
  block 8, global latent 21, decoded frame 81; raw pre-target maximum absolute
  difference is 0.0.
- Human review: not recorded. Conclusion: single-seed evidence supports
  target-causal intervention, not identity improvement.
- Next experiment motivated: a longer A→B→A2 gap with target-adjacent-only
  diagnostics.

## E20260807-LG-P1 (planned long-gap oracle record)

- Date / commit: 2026-08-07; commit will be recorded immediately before the
  first invocation. Status: **not run**.
- Research question / hypothesis: after a 15-second unrelated B shot, does a
  manually selected A identity frame alter only the first A2 block under the
  fixed six-frame context? Hypothesis: baseline/same/wrong are identical
  through block 33; history variants first diverge at block 34.
- Exact prompt: `NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt` (verbatim).
  Requested A/B/A2 durations are 10/15/10 s; effective blocks are A 1--13
  (IDs 0--38, frames 0--152, 9.5625 s), B 14--33 (39--98, 153--392, 15.0 s),
  A2 34--47 (99--140, 393--560, 10.5 s).
- Planned method/settings: source blocks 13,15; target 34; retrieval count 1;
  same ID 38; wrong ID 44; target orders baseline `[0,97,98,99,100,101]`,
  same `[0,38,98,99,100,101]`, wrong `[0,44,98,99,100,101]`; RoPE positions
  `[0,1,2,3,4,5]`; 6 frames / 9,360 tokens. Checkpoint/config/seed/resolution/
  steps/cache are Running Methods.
- Exact commands: `NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PREPARATION_20260807.md`.
  Planned raw artifacts: MP4, raw decoded tensor, and only clean blocks 33--35.
- Quantitative result / human review / failures: **not observed**.
- Planned conclusion limit: no conclusion before all three exit statuses,
  hashes, raw comparisons, and any separately labeled human review are added.

## E20260807-LG-P1 (observed raw run addendum; append-only)

- Date / commit: 2026-08-07 local; `c0d3f9e8c81ae072000f432f7fd48f9035bc2698`.
  Research question and preregistered hypothesis are in the planned record
  above. Exact prompt, requested versus effective boundaries, method, context
  orders, commands, logs, and all artifact paths are in
  `NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_GPU_20260807.md`.
- Raw run outcomes: baseline (exit 0, 107.595 s, 69,936 MiB sampled VRAM),
  same entity ID 38 (exit 0, 91.100 s, 49,671 MiB), wrong entity ID 44
  (exit 0, 90.984 s, 49,671 MiB). Raw decoded / MP4 SHA-256 values and the
  saved clean-block hashes are retained in
  `outputs/noncontiguous_phase1_long_gap/metrics.json`.
- Quantitative result with units: baseline vs both histories has block-33
  clean-latent max absolute difference 0.0 and raw RGB frames 0--392 max
  absolute difference 0.0. Both first differ at block 34/local latent 0/global
  ID 99 and raw decoded frame 393. Same history all-frame unit-interval RGB
  MAE is 0.075158; wrong history is 0.080417. These are differences, not
  identity-quality scores.
- Human review: not recorded. Failure/confounds: no failed inference; the
  baseline-first device-wide VRAM sample is higher, so no runtime or memory
  claim is supported by this ordering. One seed and no identity metric prevent
  any identity-benefit conclusion.
- Limited conclusion: one-seed long-gap manual histories are target-causal at
  the raw tensor boundary. Next experiment: multi-seed, balanced-order runs
  with preregistered human or automated identity evaluation.

## Claim → evidence update (E20260807-LG-P1)

| Candidate paper claim | Status after long-gap run | Evidence / limitation |
| --- | --- | --- |
| Manual long-gap historical K/V affects only the configured first A2 target. | Supported, single seed | Exact through block 33/frame 392; first divergence at block 34/global 99/frame 393 for both interventions. |
| Same-entity history is visually better than wrong-entity history. | Unsupported | No human review or identity metric; MAE is not a quality score. |
| Historical K/V reduces runtime or VRAM. | Unsupported | Baseline-first telemetry is warm-up/order-confounded and device-wide. |

## Figure / table index update (E20260807-LG-P1)

| ID | Artifact | Intended later use | Status |
| --- | --- | --- | --- |
| F3 | `outputs/noncontiguous_phase1_long_gap/block33_to35_baseline_same_wrong.mp4` | Long-gap target-adjacent triptych | Generated; human review not recorded |
| T4 | `outputs/noncontiguous_phase1_long_gap/metrics.json` | Long-gap raw hashes and target-adjacent difference table | Generated |
| T5 | `outputs/noncontiguous_phase1_long_gap/{mode}/run.log`, `run.json`, `vram.json` | Reproducibility, command, exit, and telemetry appendix | Generated |

## Negative-control validity addendum (2026-08-07)

- Important negative observation: in E20260807-LG-P1, the requested unrelated
  robot B shot retained Amara's identity and showed robotic-arm leakage. Its
  selected “wrong entity” history is therefore **not a valid semantic-negative
  control**. It remains a target-causal tensor intervention, but it cannot
  support a claim about semantically wrong identity memory.
- Consequence: all identity-quality interpretation for that wrong-memory arm is
  unsupported. E20260807-TRUE-WRONG-P1 substitutes an explicitly car-only B
  shot and requires human contact-sheet verification of source ID 17 before
  its wrong-history inference is allowed to run.

## E20260807-LG-MEMSTRENGTH (planned record)

- Research question / hypothesis: does one versus two manually selected clean
  A frames produce a detectable target-only change under the same six-frame
  context? The baseline and r=1 evidence are reused from E20260807-LG-P1;
  r=2 uses ordered distinct A IDs `[37,38]`, source blocks 13,15, target 34,
  and context `[sink:0,history:37,history:38,current:99,current:100,current:101]`.
- Requested/effective duration, prompt, checkpoint/config/seed/resolution/
  steps/cache, and target-adjacent artifacts are exactly E20260807-LG-P1.
  The r=2 command will save only blocks 33--35 plus raw decoded output.
- Status: **not run**. Cross-commit reuse of baseline/r=1 will be disclosed as
  a limitation; no memory-strength claim is allowed without observed r=2 data.

## E20260807-TRUE-WRONG-P1 (planned record)

- Research question / hypothesis: with a visually verified car-only B source,
  do same/wrong manual histories remain target-causal at short gap? A/B/A2
  requested and effective durations are 2.25/2.25/3.0 seconds; blocks are
  A 1--3, B 4--6, A2 7--10. Same ID 8, wrong ID 17, target 8, sources 3,6,
  retrieval count 1, six frames / 9,360 tokens, raw output plus clean 7--9.
- Exact prompt and commands are in
  `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PREPARATION_20260807.md`.
- Status: **not run**. Human B-source verification is a required gate, not an
  expected result.
