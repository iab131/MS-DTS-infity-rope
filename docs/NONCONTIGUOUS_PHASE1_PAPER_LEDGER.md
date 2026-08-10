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

## E20260807-LG-MEMSTRENGTH (observed raw-run addendum; append-only)

- Date / commits: 2026-08-07 local. Baseline and r=1 reuse completed
  `c0d3f9e8c81ae072000f432f7fd48f9035bc2698` E20260807-LG-P1 artifacts; r=2
  ran at `5d0eda3a6a0efc7e1520a151d9c0ec28d47cb4a1`.
- Research question: does replacing two recent non-sink local frames with two
  distinct A frames create a larger target-only perturbation than replacing
  one? This is not an identity-quality test.
- Prompt / boundaries: verbatim prompt
  `NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt`; requested A/B/A2
  10/15/10 s; effective A blocks 1--13 (IDs 0--38, 9.5625 s), B 14--33
  (39--98, 15.0 s), A2 34--47 (99--140, 10.5 s).
- Method: source blocks 13,15; target 34; seed 101; EMA checkpoint/config/
  resolution/four steps/cache as Running Methods. Context orders are baseline
  `[0,97,98,99,100,101]`, r=1 `[0,38,98,99,100,101]`, and r=2
  `[0,37,38,99,100,101]`; RoPE `[0,1,2,3,4,5]`; six frames / 9,360 tokens.
- Exact r=2 command: `conda run -n wan python inference.py --config_path configs/self_forcing_dmd.yaml --checkpoint_path /home/sigasia2026/models/baselines/Self-Forcing/checkpoints/self_forcing_dmd.pt --use_ema --data_path docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_PROMPT.txt --output_folder outputs/noncontiguous_phase1_long_gap/same_entity_history_r2 --seed 101 --num_samples 1 --save_with_index --output_index 0 --noncontiguous-kv --noncontiguous-source-blocks 13,15 --noncontiguous-target-block 34 --noncontiguous-kv-mode same_entity_history --noncontiguous-retrieval-count 2 --save-clean-latent-blocks 33,34,35 --save-raw-decoded --noncontiguous-history-frame-ids 37,38`. Earlier baseline/r=1 commands remain in their `run.json`.
- Raw outcomes: r=2 exit 0; 90.129 s; 49,671 MiB sampled VRAM; raw SHA-256
  `3ca0d9724940e102ed2256d5a875242029c2f64d9ab182ec018c7a386c814a21`; output
  `outputs/noncontiguous_phase1_long_gap/same_entity_history_r2/`. Preserved
  failed first invocation: parser exit 2 (incorrectly supplied r=1), no model
  output, `same_entity_history_r2/attempt1_run.{json,log}`.
- Quantitative result (max absolute clean-latent difference; RGB MAE unit
  interval): all arms equal baseline at block 33 and raw frames 0--392 (0.0).
  r=1 first differs block 34/ID 99/frame 393; block 34/35 maxima
  3.703125/4.203125; MAE 0.075158. r=2 first differs at the same boundary;
  maxima 4.1640625/5.046875; MAE 0.065835. Raw metrics/hashes/logs:
  `outputs/noncontiguous_phase1_followups_metrics.json`.
- Human review: no output-quality review. Triptych:
  `outputs/noncontiguous_phase1_long_gap/block33_to35_baseline_r1_r2.mp4`.
- Confounds: baseline/r=1 are prior-commit artifacts. r=2's greater target
  latent maximum but lower full-video RGB MAE is non-monotonic; RGB difference
  is not quality. Runtime/VRAM are order-confounded.
- Limited conclusion: r=1 and r=2 create target-causal perturbations at the
  matched budget; this one seed does not establish stronger visible or
  identity steering for r=2. Next: same-commit balanced multi-seed visual/
  identity evaluation if capacity permits.

## E20260807-TRUE-WRONG-P1 (observed raw-run addendum; append-only)

- Date / commit: 2026-08-07 local / `5d0eda3a6a0efc7e1520a151d9c0ec28d47cb4a1`.
  Question: do verified car-only and A histories remain target-causal in short
  A/B/A2?
- Exact prompt/boundaries/commands: prompt
  `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`; requested/effective A/B/A2
  2.25/2.25/3.0 s; A blocks 1--3 (0--8), B 4--6 (9--17), A2 7--10 (18--29).
  Commands remain verbatim in
  `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PREPARATION_20260807.md` and run JSON.
- Method: source blocks 3,6; target 8; seed 101; EMA checkpoint/config/
  resolution/four steps/cache as Running Methods. Contexts: baseline
  `[0,19,20,21,22,23]`, A ID 8 `[0,8,20,21,22,23]`, car ID 17
  `[0,17,20,21,22,23]`; RoPE `[0,1,2,3,4,5]`; six frames / 9,360 tokens;
  only blocks 7--9 and raw decoded tensors saved.
- Human source gate: contact sheet
  `outputs/noncontiguous_phase1_true_wrong/baseline/b_source_id17_frames65_68.png`
  visibly contains a bright yellow sports car and no woman/humanoid in frames
  65--68 (latent 17). Greenhouse-like background leakage remained, so B is not
  a fully clean desert-only environmental control.
- Raw outcomes: all exits 0. Baseline: 48.065 s, 23,025 MiB, raw SHA
  `3996069c408677c0e96eb89573b5b57bd544ca549b0b4b9177aaeebbf0b224db`.
  Same A: 49.270 s, 23,025 MiB, raw SHA
  `3cbe62ddf327479e1704b58d26b0e733e1f8d6039ce0c60d2fac5acd2310724f`.
  Wrong car: 48.521 s, 23,025 MiB, raw SHA
  `9f370f399a75e2e222caf5c4313a422c0c04cfb1b4575dded670623b53568cb3`.
  Paths: `outputs/noncontiguous_phase1_true_wrong/{baseline,same_entity_history,wrong_entity_history}/`.
- Quantitative result (max absolute clean-latent difference; RGB MAE unit
  interval): interventions equal baseline through block 7/frame 80 (0.0) and
  first differ at block 8/ID 21/frame 81. Same: block 8/9 maxima
  2.64453125/3.6328125; MAE 0.021560. Wrong car: 2.482421875/3.5; MAE
  0.042433. Full raw metrics/hashes/logs:
  `outputs/noncontiguous_phase1_followups_metrics.json`.
- Human review: source-only above; no output identity/quality review.
  Triptych: `outputs/noncontiguous_phase1_true_wrong/block7_to9_baseline_same_car_wrong.mp4`.
- Failure/confounds: no inference failure. Car/no-person source condition
  passed unlike the robot control, but background leakage, one seed, and no
  output identity metric/review preclude semantic steering or quality claims.
- Limited conclusion: A and car K/V each induce target-causal tensor changes
  under the matched budget; no returning-woman identity effect is shown. Next:
  blinded human review or a validated identity metric over balanced multi-seed
  runs.

## Claim → evidence update (follow-up studies)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| One/two selected historical frames perturb long-gap target without enlarging context. | Supported, single seed | Both first diverge at block 34/ID 99; six frames / 9,360 tokens. |
| Two frames monotonically increase visible steering. | Unsupported | Latent maxima rise but RGB MAE falls; no output review/identity metric. |
| Car wrong-memory source excludes woman/humanoid. | Supported by source-only human review | Contact sheet frames 65--68; greenhouse leakage remains. |
| Semantically wrong car K/V changes returning identity. | Unsupported | Causality/RGB difference are not identity measures. |

## Figure / table index update (follow-up studies)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F4 | `outputs/noncontiguous_phase1_long_gap/block33_to35_baseline_r1_r2.mp4` | r=1/r=2 target triptych | Generated; no human output review |
| F5 | `outputs/noncontiguous_phase1_true_wrong/block7_to9_baseline_same_car_wrong.mp4` | True-wrong target triptych | Generated; no human output review |
| F6 | `outputs/noncontiguous_phase1_true_wrong/baseline/b_source_id17_frames65_68.png` | Source validity contact sheet | Car/no-person source verified; environmental caveat |
| T6 | `outputs/noncontiguous_phase1_followups_metrics.json` | Raw metrics/hashes/timing/commands | Generated |
| T7 | `outputs/noncontiguous_phase1_{long_gap,true_wrong}/**/{run.log,run.json,vram.json}` | Reproducibility appendix | Generated; r=2 parser failure retained |

## Methods details update (follow-up studies)

- Ordered multi-frame history uses `--noncontiguous-history-frame-ids`; r=2
  IDs `37,38` replace both non-sink recent slots. CPU-offloaded K/V stays
  transient and transfers only for target attention.
- Context is six frames / 9,360 tokens for r=0/1/2. The transformed persistent
  sink remains slot zero and is never re-rotated; temporary RoPE is
  `[0,1,2,3,4,5]`.

## Attention-memory policy logging implementation (not an experiment)

- The opt-in attention-memory scaffold writes a JSONL record per `config`,
  `retrieval`, `context`, `write`, and `transition` event. It records component
  switches, descriptor layers, manual override, retrieved frame/scene IDs and
  scores, query source, context ordering/positions/token count, memory/archive
  size, archive selection, local retention/decay, and consolidation actions.
- This is implementation provenance only. No full-pipeline GPU run, descriptor
  semantic-accuracy measurement, visual review, or paper claim is recorded by
  this note. See `ATTENTION_MEMORY_POLICY_IMPLEMENTATION_20260807.md` for the
  live-code deviations and exact CLI contract.

## E20260807-AMP-SHORT-ABA-P1 (planned combined-policy oracle)

- Research question / hypothesis: when a scene transition removes competing
  local context, can two manually selected historical K/V frames cause a
  target-only perturbation at returning A2? This tests transition forgetting
  plus historical recall, not identity retrieval quality.
- Prompt / schedule: the recorded woman → car → woman prompt in
  `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`; requested/effective
  durations 2.25/2.25/3.0 s; A blocks 1--3/IDs 0--8, B 4--6/9--17, A2
  7--10/18--29. Target block 8/current IDs 21--23.
- Conditions: A normal baseline; B `sink_only` with retrieval off; C the same
  transition policy plus manual woman IDs `[6,7]`; D the same plus manual car
  IDs `[16,17]`. Descriptor layers are `[0,1,5,14,16]`; all 30 layers inject
  in C/D, giving `[sink:0,memory:<id>,memory:<id>,current:21,current:22,current:23]`,
  positions `[0,1,2,3,4,5]`, six frames / 9,360 tokens.
- Source gate: human review of prior decoded contact sheets confirms A
  IDs 6,7 contain Amara and B IDs 16,17 contain yellow car/no woman or
  humanoid. Greenhouse-background leakage remains a stated confound.
- Automatic first-transition routing is explicitly disabled; manual target
  block 8 remains enabled. Exact commands, paths, diagnostics, and stop
  conditions are in `ATTENTION_MEMORY_POLICY_SHORT_ABA_ORACLE_PREPARATION_20260807.md`.
- Status: **not run**. No expected visual/identity result is an observation.

## E20260807-AMP-SHORT-ABA-P1 schedule correction (planned; supersedes target details above)

- Live parser evidence: `CausalInferencePipeline._parse_scene_durations` on
  the exact recorded prompt returned effective blocks `[3,3,4]`. Thus A is
  blocks 1--3/IDs 0--8, B is 4--6/IDs 9--17, and the **first A2 block is 7**
  with current IDs `[18,19,20]`.
- Correction: the original planned entry accidentally inherited the older
  block-8 oracle target. C/D now use `--memory-manual-target-blocks 7`, save
  target-adjacent clean blocks 6--8, and require
  `[sink:0,memory:<id>,memory:<id>,current:18,current:19,current:20]` with
  positions `[0,1,2,3,4,5]` (six frames / 9,360 tokens). The exact corrected
  commands are in `ATTENTION_MEMORY_POLICY_SHORT_ABA_ORACLE_PREPARATION_20260807.md`.
- Status: **not run**. This correction changes no observed result.

## E20260807-AMP-SHORT-ABA-P1 execution update

- Date / commit: 2026-08-07; `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`.
- Research question / hypothesis: does `sink_only` transition forgetting make
  a two-frame manual historical-K/V intervention at first A2 target-causal?
  This is an interaction test, not a semantic-routing or identity hypothesis.
- Prompt / schedule: exact prompt remains
  `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`; parser blocks `[3,3,4]`.
  Requested/scheduled nominal A/B/A2 are 2.25/2.25/3.0 s. Saved raw RGB has
  117 frames at 16 FPS: A 0--32 (2.0625 s), B 33--68 (2.25 s), A2 69--116
  (3.0 s). The `4n-3` decoded boundary means saved duration is 7.3125 s, not
  the 7.5-s requested/scheduled total.
- Exact method: baseline; `hard_flush` (`sink_only`, retrieval off); manual
  A IDs `[6,7]`; manual car IDs `[16,17]`. C/D inject at all 30 layers,
  `k=2`, `prepend`, target block 7/current IDs `[18,19,20]`. Automatic
  routing, archive, consolidation, and decay were disabled; cross-attention
  reset remained enabled. Exact commands, all settings, logs, and hashes:
  `ATTENTION_MEMORY_POLICY_SHORT_ABA_GPU_20260807.md`.
- Logged target contexts: correct `[sink:0,history:6,history:7,current:18,
  current:19,current:20]`; wrong `[sink:0,history:16,history:17,current:18,
  current:19,current:20]`; positions `[0,1,2,3,4,5]`; zero local frames;
  six frames / 9,360 tokens. Query source was `pre_transition_raw`, but both
  routes were explicit manual overrides, not automatic descriptor routing.
- Raw outcomes (exit/runtime/peak sampled VRAM MiB): baseline 0/47.56/23,031;
  hard_flush 0/51.14/23,243; correct 0/52.51/23,243; wrong 0/53.34/23,243.
  Raw tensor and MP4 hashes are preserved in the GPU record. Device-wide
  telemetry and run order preclude a performance conclusion.
- Quantitative causality: hard_flush vs correct and wrong are raw-RGB exactly
  equal through frame 68 (max abs 0.0) and saved clean block 6 (max abs 0.0).
  Both first diverge at raw frame 69 / clean block 7 local latent 0. Block
  7/8 clean-latent maxima are correct 5.125/5.625 and wrong 4.9375/5.171875.
  Raw RGB MAE/PSNR: hard vs correct 0.101799488/12.677845 dB; hard vs wrong
  0.099336967/13.134525 dB. Baseline vs hard first diverges at frame 33,
  consistent with the A→B hard-flush transition.
- Human review: **pending**. No visual or identity conclusion is recorded.
- Failures/confounds: preserved first attempts include a missing local
  `wan_models` link, then an undefined active scene-ID clean-write variable,
  then JSON logging of a set. The link was restored without changing settings;
  both code defects were regression-tested and amended before successful
  reruns. One seed, source greenhouse leakage, and no visual/identity metric
  limit interpretation.
- Limited conclusion: manual A and car K/V each cause a first-A2-block
  perturbation under the matched six-frame hard-flush context. This does not
  show identity recovery, semantic selectivity, or correct-memory superiority.
  Next: blinded human review of the two transition comparisons before any
  policy tuning or new memory mechanism.

## Claim → evidence update (combined-policy oracle)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| Hard-flush plus manual K/V can change only first A2 output. | Supported, one seed | Hard/C/D raw equality through frame 68; first change frame 69/block 7. |
| Manual recall uses matched six-frame context. | Supported, implementation/log | Target JSONL has sink + 2 history + 3 current, positions 0--5, 9,360 tokens. |
| Correct woman memory improves identity over car memory. | Unsupported | No human review or identity metric; RGB distance is not identity quality. |
| Sink-only forgetting is behaviorally inert. | Contradicted | Baseline vs hard_flush first differs at A→B raw frame 33. |

## Figure / table index update (combined-policy oracle)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F7 | `outputs/attention_memory_policy_short_aba/transition_A_to_B_baseline_hard_flush_correct_wrong.mp4` | Four-way hard-flush transition | Generated; human review pending |
| F8 | `outputs/attention_memory_policy_short_aba/transition_B_to_A2_baseline_hard_flush_correct_wrong.mp4` | Four-way recall transition | Generated; human review pending |
| T8 | `outputs/attention_memory_policy_short_aba/{baseline,hard_flush,correct_memory,wrong_memory}/` | Raw tensors, snapshots, JSONL, logs, VRAM | Generated; failed attempts retained |

## Methods details update (combined-policy oracle)

- The manual all-layer runs retain historical CPU K/V for layers 0--29 and
  transfer the two selected frames transiently only for target attention.
  The persistent sink stays special in slot zero; no retrieved K is written
  back to the persistent cache.
- The first `sink_only` A2 block has no retained non-sink local frames. Thus
  `prepend` reaches the six-frame target budget exactly; later blocks are not
  assumed to share that composition.

## E20260808-AMP-IDENTITY-SELECTIVITY-P1 (source-gated incomplete)

- Date / commit: 2026-08-08; `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`.
- Research question / hypothesis: two valid, distinctive woman A memories
  should be compared with two truck memories at first A2 under the matched
  all-layer hard-flush context. This is an identity-selectivity test only if
  the required source attributes are visibly present before injection.
- Exact prompt: recorded verbatim in
  `ATTENTION_MEMORY_POLICY_IDENTITY_SELECTIVITY_PROMPT_20260808.txt`. A contains
  white left hair streak, red under-left-eye star, and yellow chest patch; A2
  says only “The same woman from the first shot standing on a snowy mountain”.
  Requested/scheduled blocks are A/B/A2 2.25/2.25/3.0 s and `[3,3,4]`.
- Executed arms / settings: baseline and `sink_only` hard_flush only; both
  seed 101, EMA checkpoint, config, 480x832, four steps, raw/snapshot blocks
  6--8. The planned C/D commands were not launched. If admitted, they would
  have used IDs `[6,7]` and `[16,17]`, target 7, k=2, all 30 injection layers,
  six frames / 9,360 tokens, automatic routing/archive/consolidation/decay
  disabled. Exact executed commands and intended manual commands are in
  `ATTENTION_MEMORY_POLICY_IDENTITY_SELECTIVITY_20260808.md`.
- Source gate (human review): all inspected A IDs 0--8 visibly have black bob
  and yellow patch, but none visibly has the required white streak or small red
  star; all substitute a red eye-level beam. Candidate truck IDs 16,17 show
  truck only/no people or humanoids, but retain greenhouse instead of desert.
  Therefore the correct-memory source condition failed and manual C/D were
  deliberately not run.
- Raw outcomes: baseline exit 0/runtime 48.14 s/peak 23,031 MiB; hard_flush
  0/51.90 s/23,243 MiB. Baseline vs hard first differs at raw frame 33, with
  preceding maximum absolute RGB difference 0.0; raw MAE 0.094046980 and PSNR
  15.160388 dB. This is hard-flush transition behavior, not memory evidence.
- Human A2 screen: source, early A2, and mid A2 decoded stills show no white
  streak or small red star; greenhouse recall is strong and snowy-mountain
  compliance absent. The yellow patch remains visible in the hard-flush mid-A2
  screen, but no retrieval ran, so it cannot be attributed to memory.
- Failures/confounds: this is a negative source-validity result, not an OOM or
  implementation failure. No four-arm comparison was fabricated; existing
  greenhouse leakage invalidates the requested desert environmental contrast.
- Limited conclusion: no identity-selectivity claim is supported. Next:
  generate or choose a new independently source-gated prompt/seed before any
  manual historical-K/V condition.

## Claim → evidence update (identity-selectivity source gate)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| Prompted A attributes are available as a valid manual-memory source. | Contradicted for this seed/prompt | All A source frames lack the white streak and red star. |
| Truck provides a clean desert wrong-memory environmental control. | Contradicted for this seed/prompt | Truck is person-free but greenhouse background persists. |
| Historical K/V selectively restores identity attributes. | Unsupported | Correct/wrong manual arms were not run by source gate. |

## Figure / table index update (identity-selectivity source gate)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F9 | `outputs/attention_memory_policy_identity_selectivity/hard_flush/a_source_ids0_5_frames1_24.png` and related A source sheets | Source-admission negative evidence | Generated; human review recorded |
| F10 | `outputs/attention_memory_policy_identity_selectivity/source_gate_baseline_hard_flush_A_early_mid_A2.png` | Source/early/mid A2 screen | Generated; two arms only, not a memory comparison |
| F11 | `outputs/attention_memory_policy_identity_selectivity/transition_B_to_A2_baseline_hard_flush_source_gate.mp4` | Baseline/hard source-gate transition | Generated; not four-arm |

## E20260808-AMP-TRANSITION-RETENTION-P1 (executed)

- Experiment ID/date/commit: E20260808-AMP-TRANSITION-RETENTION-P1,
  2026-08-08; base commit `0efd5ad17e57f19ab2029ffc6da44be46e4a415d` plus
  the uncommitted, focused `transition_no_sink` experiment option.
- Research question / hypothesis: does retaining recent non-sink local K/V at
  a woman→truck scene cut cause entity carry-over, and does retaining only the
  sink leave background carry-over unresolved? The experimental hypothesis for
  `transition_no_sink` is that removing both old-scene sink and local K/V for
  one transition block will make the new scene visible sooner.
- Exact prompt / schedule: verbatim prompt is in
  `ATTENTION_MEMORY_POLICY_TRANSITION_RETENTION_PROMPT_20260808.txt` (woman in
  glass greenhouse → bright-blue pickup alone on desert road). Requested
  durations are 2.25 s / 2.25 s; live schedule is `[3,3]` latent blocks, A
  blocks 1--3 / IDs 0--8 and B 4--6 / IDs 9--17. Target B block 4 is IDs
  9--11 and saved raw B starts RGB frame 33. Saved raw output is 69 frames at
  16 FPS (4.3125 s), not the requested 4.5 s.
- Exact method: baseline `sink+2`, `sink+1`, `sink_only`, and opt-in
  `transition_no_sink`. Every arm used seed 101, Self-Forcing DMD EMA,
  `configs/self_forcing_dmd.yaml`, 480x832, four steps, six-frame cache,
  `prepend`, all 30 injection layers, retrieval/decay/archive/consolidation
  and transition automatic retrieval off, and cross-attention reset on. No
  historical frame IDs were selected, no retrieval scores exist, and no
  transient history was injected.
- Logged B-block contexts / positions / count: sink+2
  `[sink:0,local:7,local:8,current:9,current:10,current:11]` /
  `[0,1,2,45,46,47]` / 6 frames, 9,360 tokens; sink+1
  `[sink:0,local:8,current:9,current:10,current:11]` / `[0,1,45,46,47]` /
  5, 7,800; sink-only `[sink:0,current:9,current:10,current:11]` /
  `[0,45,46,47]` / 4, 6,240; transition-no-sink
  `[current:9,current:10,current:11]` / `[45,46,47]` / 3, 4,680. The latter
  sets usable cache end to zero only for block 4, then normal B cache writes
  resume. The old transformed sink is excluded, not re-rotated or altered.
- Exact commands, per-arm paths, runtimes, peak sampled VRAM, tensor hashes,
  and exit status are preserved verbatim in
  `ATTENTION_MEMORY_POLICY_TRANSITION_RETENTION_GPU_20260808.md`. All arms
  exited 0; wall runtimes were 50.90/47.89/47.08/45.31 s and sampled peak VRAM
  was 23,243 MiB for baseline/sink+1/sink-only/no-sink. Raw output paths are
  under `outputs/attention_memory_policy_transition_retention/<arm>/`; all
  include raw decoded RGB, clean blocks 3--5, JSONL, log, and VRAM CSV.
- Quantitative causality: all arms share clean-block-3 SHA-256
  `83fc30c10cec125c28605c7aba7b7df4db77c6307726a896f930ec20de9d97d0`.
  Each differs from baseline at clean block 4 and raw RGB frame 33, exactly
  the first B block. This is a timing/hashing result, not an image-quality
  metric.
- Human review (clearly qualitative): baseline and sink+1 retain a woman's
  head in the early truck windshield and greenhouse throughout sampled B;
  sink-only has a first-frame blend then no clearly visible woman but still
  greenhouse; transition-no-sink has a first-frame blend then a complete
  truck-only desert scene from frame 34 in sampled output. No explicit vehicle
  motion was requested, so there is no motion-quality claim. The requested
  first-12 frame sheet and four-way video are in the figure index below.
- Failure/confounds: no GPU failure. This is one prompt and one seed; retention
  settings intentionally change context token count, so it is a retention
  ablation, not a matched-token memory comparison. The first no-sink RGB frame
  still blends the preceding scene. Human review samples B at frames 33--44
  plus every third frame through 66 rather than an independently blinded study.
- Limited conclusion: recent non-sink local K/V is a strong contributor to
  early woman leakage at this cut. Sink-only is insufficient for greenhouse
  background persistence. One-block no-sink is promising for this failure but
  is experimental; it does not establish a general transition rule, sink-only
  causality for background, or any retrieval/identity benefit. Next:
  reproducibility over prompts/seeds before enabling it as a policy default.

## Claim → evidence update (transition retention)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| Recent non-sink local K/V contributes to woman carry-over at woman→truck transitions. | Supported, one prompt/seed, qualitative | Baseline/sink+1 retain head-in-windshield; sink-only does not after its blend. |
| Sink-only removes old-scene background. | Contradicted, one prompt/seed | Greenhouse remains through sampled B under sink-only. |
| One transition block with no old-scene attention produces a clean new scene sooner. | Supported, one prompt/seed, qualitative | No-sink is desert/truck-only from sampled frame 34; first frame still blends. |
| This transition policy improves historical identity retrieval. | Unsupported | Retrieval was disabled in every arm. |

## Figure / table index update (transition retention)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F12 | `outputs/attention_memory_policy_transition_retention/comparison/first12_B_frames_33_44_baseline_sinkplus2_sinkplus1_sinkonly_no_sink.png` | Requested synchronized first 12 B frames; row order baseline, sink+1, sink-only, no-sink | Generated; human review recorded |
| F13 | `outputs/attention_memory_policy_transition_retention/comparison/A_to_B_four_way_frames_21_56_baseline_sinkplus2_sinkplus1_sinkonly_no_sink.mp4` | Four-way A→B transition video | Generated; human review recorded |
| F14 | `outputs/attention_memory_policy_transition_retention/comparison/{early_B_frames_33_36_fullres_grid,B_temporal_samples_frames_33_66_every3}.png` | High-resolution leakage and stabilization review | Generated; qualitative only |
| T9 | `docs/ATTENTION_MEMORY_POLICY_TRANSITION_RETENTION_GPU_20260808.md` and `outputs/attention_memory_policy_transition_retention/<arm>/{memory_policy.jsonl,run.log,vram.csv}` | Commands, contexts, timing, hashes, raw diagnostics | Generated |

## Methods details update (transition retention)

- `--memory-local-retention transition_no_sink` is an opt-in experimental
  local-retention setting. On only the first block after a detected cut it
  excludes both the transformed persistent sink and preceding local frames by
  setting the usable local cache end to zero. It neither mutates nor re-rotates
  the sink. Standard write/rolling behavior then writes the first new-scene
  block at the first cache slot, allowing following blocks to use the new scene.
- Transition JSONL `attention_context` records every ordering, global frame ID,
  position, frame count, and token count. Because this ablation removes local
  frames, frame/token counts intentionally vary 6/5/4/3 and should never be
  described as a matched 6-frame memory experiment.

## E20260808-AMP-RESET-DELAYED-RECALL-P1 (executed)

- Date/commit: 2026-08-08; base `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`
  plus focused uncommitted transition/logging fixes. Research question: can a
  no-old-context reset establish A2, then delayed two-frame history preserve
  useful content without restoring its old scene?
- Prompt/boundaries: exact Amara greenhouse → yellow-car desert → snowy
  observatory text is in `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`.
  Requested/scheduled A/B/A2 2.25/2.25/3.0 s maps to `[3,3,4]`; A IDs 0--8,
  B 9--17, A2 18--29. Saved RGB is 117 frames/7.3125 s (A 0--32, B 33--68,
  A2 69--116), not 7.5 s.
- Method/settings: every hard cut used `transition_no_sink` current-only
  `[45,46,47]` / 3 frames/4,680 tokens. At delayed A2 block 8, reset has
  `[sink:18,local:19,local:20,current:21,current:22,current:23]`; correct A
  IDs 6,7 and wrong car IDs 16,17 respectively use
  `[sink:18,history,history,current×3]`. All target contexts are six frames/
  9,360 tokens. Existing `replace_recent` gives this physical post-sink
  placement by replacing local 19/20; literal prepend would make eight frames.
  Sink 18 is preserved, never re-rotated. All 30 injection layers; EMA/config,
  seed 101, 480x832, four steps, cache six; auto routing/archive/consolidation/
  decay off. Exact commands are in the raw record.
- Context/positions: logical slots 0--5; correct/wrong actual RoPE record is
  preserved sink, history K 1/2, no local K, current K 3/4/5, query 21/22/23.
  Source IDs 6/7 were prior-reviewed A frames; 16/17 person-free car frames,
  with source-background leakage retained as a confound.
- Runs/results: official reset/correct/wrong exit 0, runtime 51.26/50.80/53.65
  s, sampled peak VRAM 23,243 MiB. They equal exactly through clean blocks 6/7
  and raw frame 80; first divergence is target block 8/frame 81. Correct/wrong
  RGB MAE 0.1138135/0.0815432 and PSNR 11.6895/14.9524 dB are perturbation
  measures only. Paths/hashes/logs are in
  `ATTENTION_MEMORY_POLICY_RESET_THEN_RECALL_GPU_20260808.md`.
- Human review: first A2 block is equal and blended. Reset-only reaches snowy
  observatory woman; A history restores greenhouse/orchids; wrong car history
  restores yellow desert car. A2 repeats identity attributes, so this is not
  an identity-improvement result.
- Failures/confounds: two earlier full trios remain preserved: sink label ID 0,
  then `[-0:]` falsely logged replaced locals / 8 frames. Logging-only fixes
  produced a final reset raw tensor identical to the prior render. One seed,
  source background leakage, and non-blinded review limit interpretation.
- Limited conclusion: delayed raw K/V has visible source-selective steering,
  but reintroduces whole old scenes; it does not preserve old identity without
  background leakage. Next: do not claim a policy solution from this oracle.

## Claim → evidence update (reset-then-recall)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| One-block no-sink reset establishes a distinct A2 before recall. | Supported, one prompt/seed, human review | Reset resolves to snowy observatory after equal blend. |
| Delayed two-frame history is target-causal at matched budget. | Supported, one seed | Equality through frame 80; first change block 8/frame 81; six/9,360. |
| A history preserves identity without old-scene leakage. | Contradicted for this oracle | Greenhouse/orchids return; A2 already repeats identity attributes. |
| Wrong car history reintroduces wrong semantic content. | Supported, one prompt/seed, human review | Yellow car/desert replaces A2 woman/observatory. |

## Figure / table index update (reset-then-recall)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F15 | `outputs/attention_memory_policy_reset_then_recall_verified/comparison/A2_blocks_7_8_9_official_reset_correct_wrong.mp4` | Three-way A2 blocks 7--9 | Generated; one-seed human review |
| F16 | `outputs/attention_memory_policy_reset_then_recall_verified/comparison/A2_blocks_7_8_9_official_still_sheet_reset_correct_wrong.png` | A2 frames 69/81/93; rows reset/correct/wrong | Generated; one-seed human review |
| T10 | `docs/ATTENTION_MEMORY_POLICY_RESET_THEN_RECALL_GPU_20260808.md` | Commands, raw hashes, contexts, metrics, limitations | Generated |

## Methods details update (reset-then-recall)

- After `transition_no_sink`, cache slot zero is the first clean new-scene
  frame. Logs must label it by global ID (A2 ID 18 here), while preserving its
  transformed K. With k=2 and two local non-sink frames, `replace_recent`
  yields the matched six-frame physical post-sink history context; `prepend`
  would expand to eight. `items[-0:]` must never be used for zero retained
  local IDs.

## E20260809-AMP-LAYER-SELECTIVE-P1 (executed)

- Date/commit: 2026-08-09; base `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`
  plus existing uncommitted transition/logging fixes. Research question: can a
  coarse injection-layer range preserve A woman content without A greenhouse
  scene recall? This is a manual oracle, not descriptor routing.
- Prompt/boundaries/settings: exact Amara greenhouse → yellow-car desert →
  snowy observatory prompt and seed 101 are unchanged from the verified
  reset-then-recall experiment. Blocks `[3,3,4]`; target A2 block 8/IDs 21--23;
  A source IDs 6,7; `transition_no_sink` both cuts; EMA/config, 480x832,
  four steps, cache six, cross reset on; routing/archive/consolidation/decay
  off. Every target has `[sink:18,history:6,history:7,current×3]`, logical
  slots 0--5, six frames/9,360 tokens, and preserved sink/history K 1/2/current
  K 3/4/5/query 21/22/23. Only injection layers differ.
- Arms: reset-only reused; 0--9, 10--19, 20--29 newly ran; verified all-layer
  0--29 A-memory result reused because settings are identical. Every manual
  arm exits 0, peaks at sampled 23,243 MiB, equals reset through block 7/raw
  frame 80, and first differs at target block 8/frame 81. New runtimes are
  48.92/49.88/49.21 s; raw MAE versus reset 0.0257268/0.0300226/0.0113323
  and PSNR 20.6858/19.9450/26.9958 dB. Full hashes/commands/logs in
  `ATTENTION_MEMORY_POLICY_LAYER_SELECTIVE_GPU_20260809.md`.
- Human review: reset and every coarse ten-layer range retain snowy observatory
  in reviewed blocks 8--9, show no greenhouse/orchids, and show no verified
  incremental original-A woman appearance beyond prompt baseline. All-layer A
  history strongly restores greenhouse/orchids; the previous verified B-memory
  result restores B car/desert. A2 repeats woman attributes, so this does not
  test identity benefit cleanly.
- Limited conclusion: no promising coarse identity-only range was found. The
  contrast is consistent with strong scene recall requiring a distributed
  all-layer combination, but does not demonstrate subject/scene separability.
  Per stop rule, no finer layer sweep was launched.

## Claim → evidence update (coarse layer selectivity)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| A coarse layer group isolates historical subject recall from scene recall. | Unsupported | No range exceeds prompt baseline for A appearance; all retain snowy scene. |
| All-layer A and B histories are source-selective. | Supported, one prompt/seed, human review | A recalls greenhouse/orchids; B recalls car/desert. |
| Source-selective raw K/V recall is scene-entangled. | Supported, one prompt/seed, human review | Recalled A/B content includes their source scenes. |
| Strong all-layer scene recall is attributable to one 10-layer group. | Unsupported | No individual coarse group visibly recreates greenhouse. |

## Figure / table index update (coarse layer selectivity)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F17 | `outputs/attention_memory_policy_layer_selective/comparison/A2_blocks_7_8_9_five_way_reset_L0_9_L10_19_L20_29_L0_29.mp4` | Five-way synchronized A2 layer ablation | Generated; one-seed human review |
| F18 | `outputs/attention_memory_policy_layer_selective/comparison/A2_blocks_7_8_9_five_way_still_sheet_reset_L0_9_L10_19_L20_29_L0_29.png` | Frames 69/81/93; rows reset, 0--9, 10--19, 20--29, all | Generated; one-seed human review |
| T11 | `docs/ATTENTION_MEMORY_POLICY_LAYER_SELECTIVE_GPU_20260809.md` | Commands, hashes, contexts, human-review fields | Generated |

## E20260809-AMP-LIFETIME-L0-9-P1 (executed)

- Date/commit: 2026-08-09; base `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`
  plus uncommitted retrieval-lifetime and existing transition/logging fixes.
  Question: can L0--9 A-history duration yield a small appearance correction
  without restoring A's greenhouse scene or causing a temporal reset?
- Policy audit: the verified A2 JSONLs actually inject manual A IDs 6/7 only
  at block 8; blocks 7, 9, and 10 are `manual_target_not_selected`. Thus that
  explicit target-restricted setup was already pulse-1, while unrestricted
  manual IDs were the generic persistent path. New lifetime settings from
  target 8 are pulse-1 (8), pulse-2 (8--9), and persistent (8--10).
- Prompt/boundaries/settings: verbatim prompt and exact commands are in
  `ATTENTION_MEMORY_POLICY_RETRIEVAL_LIFETIME_GPU_20260809.md`. The unchanged
  requested A/B/A2 2.25/2.25/3.0 s prompt schedules `[3,3,4]`: A IDs 0--8,
  B 9--17, A2 18--29; saved RGB is 117 frames/7.3125 s, not requested 7.5 s.
  Fixed: EMA/config, seed 101, 480x832, four steps, cache six, A sources 6/7,
  descriptor 0/1/5/14/16, injection 0--9, `transition_no_sink`,
  `replace_recent`, k=2, automatic routing/archive/consolidation/decay off.
- Context/positions: A2 reset block 7 is current IDs 18--20, RoPE `[45,46,47]`,
  3/4,680; it makes 18 the preserved transformed sink. Every recall block is
  `[sink:18,history:6,history:7,current×3]`, 6/9,360, with preserved sink,
  history K 1/2, current K 3/4/5, and global queries. Non-retrieval blocks
  retain exactly sink + two local + current×3, also 6/9,360.
- Runs/results: reset-only and pulse-1 reused settings-equivalent artifacts;
  pulse-2/persistent newly exit 0 in 50.05/48.81 s, each sampled at 23,243 MiB.
  All retrieval arms equal reset through A2 block 7/RGB 80 and first differ at
  block 8/RGB 81. Pulse-2 first separates from pulse-1 at block 9/RGB 93;
  persistent from pulse-2 at block 10/RGB 105. RGB MAE vs reset is
  0.0257268/0.0303887/0.0312111 and PSNR 20.6858/19.7575/19.5709 dB for
  pulse-1/2/persistent; perturbation metrics only. Tensor hashes, raw paths,
  and policy events are in the raw report.
- Human review: all arms share the A2 reset blend. Through blocks 8--10, no
  incremental A-appearance correction beyond the explicit A2 prompt, no
  greenhouse/orchid return, and no obvious extra flicker/reset is verified in
  this one-seed review; snowy observatory remains and following blocks look
  coherent. Adjacent-frame RGB differences are not flicker metrics.
- Confounds/conclusion: no failures; one prompt/seed, A2 repeats identity
  details, and review is non-blinded. This demonstrates causal lifetime control
  at matched budget only—not identity benefit, no-flicker behavior, or a
  rationale for a finer layer sweep. No finer sweep was launched.

## Claim → evidence update (manual retrieval lifetime)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| Manual lifetime controls intended recall blocks. | Supported, one run each | JSONL IDs 6/7 at 8 only, 8--9, or 8--10; added divergence follows at 8, 9, and 10. |
| L0--9 history yields visible identity correction without scene recall. | Unsupported | No incremental A appearance over the explicit A2 prompt. |
| Longer L0--9 recall causes obvious flicker/reset. | Unsupported | No obvious effect in one-seed review; no validated flicker measure. |
| Longer L0--9 recall restores greenhouse scene. | Contradicted for this oracle, one seed | No greenhouse/orchids visibly verified through A2 block 10. |

## Figure / table index update (manual retrieval lifetime)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F19 | `outputs/attention_memory_policy_retrieval_lifetime/comparison/A2_blocks_7_10_four_way_reset_pulse1_pulse2_persistent.mp4` | Synchronized reset/pulse-1/pulse-2/persistent A2 blocks 7--10 | Generated; one-seed review |
| F20 | `outputs/attention_memory_policy_retrieval_lifetime/comparison/A2_blocks_7_10_four_way_still_sheet_reset_pulse1_pulse2_persistent.png` | Rows reset/pulse-1/pulse-2/persistent; RGB 69/81/93/105 | Generated; qualitative only |
| T12 | `docs/ATTENTION_MEMORY_POLICY_RETRIEVAL_LIFETIME_GPU_20260809.md` and `outputs/attention_memory_policy_retrieval_lifetime/l0_9_{pulse_2,persistent}/` | Commands, logs, hashes, timing, VRAM, raw metrics | Generated |

## Methods details update (manual retrieval lifetime)

- `--memory-retrieval-lifetime {pulse_1,pulse_2,persistent}` starts at the
  minimum explicit manual target block. Default pulse-1 preserves the verified
  single-target oracle. Target-restricted calls log `manual_target_not_selected`
  before target and `manual_lifetime_expired` after a pulse; unrestricted manual
  IDs retain their old always-allowed path.
- Paper hashes are SHA-256 over contiguous raw decoded RGB tensor bytes. The
  serialized `.pt` file hash includes packaging and is different; paths retain
  file-level provenance.


## E20260809-AMP-L0-9-SOURCE-SPECIFICITY-P1 (executed)

- Date/commit: 2026-08-09; base `0efd5ad17e57f19ab2029ffc6da44be46e4a415d`
  plus existing uncommitted policy work. Question: does L0--9 pulse-1 encode
  source-specific recall or generic transient K/V perturbation?
- Prompt/boundaries/settings: exact unchanged Amara greenhouse → yellow-car →
  snowy-observatory prompt is in `NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PROMPT.txt`.
  Requested 2.25/2.25/3.0 s schedules `[3,3,4]`, IDs A 0--8/B 9--17/A2 18--29;
  saved RGB is 117 frames/7.3125 s. Seed 101, EMA/config, 480x832, four steps,
  cache six, descriptor 0/1/5/14/16, injection 0--9, k=2, `replace_recent`,
  `transition_no_sink`, and all non-manual mechanisms off are fixed.
- Sources/context: reused correct A IDs 6/7 (scene 0) and newly run B-car IDs
  16/17 (scene 1); B is car/no-person but has known greenhouse leakage. A2
  reset block 7 is current-only 3/4,680 at `[45,46,47]`; pulse block 8 is sink
  18 + two history + current×3, exactly 6/9,360, preserved sink/history K 1/2/
  current K 3/4/5/query 21--23. Both pulses stop after block 8.
- Reuse boundary: reset retrieval is disabled, so its all-layer injection capture
  list is inactive; correct-A predates explicit lifetime logging but its target-8
  JSONL proves pulse-1 behavior. No output was silently regenerated.
- Raw results: new wrong-car exits 0 in 52.15 s/23,243 MiB. Correct-A/wrong-car
  first diverge from reset at RGB 81/A2 b8 after equality through RGB 80; they
  also differ from one another at RGB 81. Raw MAE vs reset is 0.0257268/0.0188217
  and PSNR 20.6858/22.7431 dB for A/car; perturbation metrics only. Commands,
  hashes, policy log, snapshots, and raw output are in the raw report.
- Human review: both histories make small face/pose/detail changes relative to
  reset, but neither recovers a specific original-A feature beyond the A2 prompt.
  Wrong-car shows no car/desert or other B-specific property; snowy observatory
  remains. This is not visible semantic source-specificity.
- Confounds/conclusion: one prompt/seed, A2 repeats Amara attributes, B scene
  leaks greenhouse, and review is non-blinded. Source-dependent raw differences
  do not prove semantic recall. Visible evidence is consistent with generic
  transient facial perturbation; do not tune lifetime or run a finer sweep.

## Claim → evidence update (L0--9 source specificity)

| Candidate paper claim | Status | Evidence / limitation |
| --- | --- | --- |
| L0--9 correct-A visibly restores a specific A feature. | Unsupported | No feature exceeds explicit A2 prompt baseline. |
| L0--9 wrong-car visibly restores car/desert content. | Contradicted for this oracle, one seed | No car/desert feature visible through A2 b10. |
| L0--9 source choice changes raw generation causally. | Supported, one run each | A/car tensors differ from reset and each other beginning at target b8/RGB 81. |
| L0--9 face change is semantic source-specific recall. | Unsupported | Small changes lack identifiable A/B semantic correspondence. |

## Figure / table index update (L0--9 source specificity)

| ID | Artifact | Intended use | Status |
| --- | --- | --- | --- |
| F21 | `outputs/attention_memory_policy_source_specificity/comparison/A2_blocks_7_10_three_way_reset_correctA_wrongcar.mp4` | Synchronized A2 b7--10 reset/correct-A/wrong-car review | Generated; one-seed qualitative review |
| F22 | `outputs/attention_memory_policy_source_specificity/comparison/A2_blocks_7_10_three_way_still_sheet_reset_correctA_wrongcar.png` | RGB 69/81/93/105 rows reset/correct-A/wrong-car | Generated; qualitative only |
| T13 | `docs/ATTENTION_MEMORY_POLICY_L0_9_SOURCE_SPECIFICITY_GPU_20260809.md` and `outputs/attention_memory_policy_source_specificity/l0_9_pulse_1_wrong_car/` | Command, policy log, metrics, raw output, limitations | Generated |

## E20260809-AMP-FIXED-GRID-SELECTIVE-RECALL-ORACLE (executed; one seed)

- Scope: manual oracle separability protocol inspired by EM-Vid, DiTCtrl, and
  BachVid; not a novel masking method.
- Immutable inputs: manually checked 30x52 source masks for A IDs 6/7 and one
  three-frame target conservative union are stored in
  `docs/attention_memory_policy_fixed_grid_masks_20260809.json` (SHA-256
  `743a7c6e2a4d6c41c01da9b77e553d59e3cf3ccb989d6a96160c3a209c4fd5cf`).
  Source K uses original row-major H/W positions and temporary slots 1/2;
  background is the eight-connected one-token dilated complement on each
  source and on the target.
- Fixed protocol: seed 101, manual IDs 6/7, target block 8, `replace_recent`,
  and `transition_no_sink`. The reset removes old context for A2 block 7 and
  records its first clean frame 18 as the new sink. Block 8 retains the normal
  base order `[sink:18,local:19,local:20,current:21,current:22,current:23]`;
  masked history is a separate per-layer branch and never replaces locals.
- CPU preflight: five source/target overlays were checked and
  `outputs/attention_memory_policy_fixed_grid_selective_recall/preflight/mask_audit.json`
  records exact source/query indices, row/column coordinates, slots, expanded
  three-frame query sets, hashes, and the derived base order. Audit SHA-256 is
  `a6d7b90a0835ed39eedfab0d93d3733e048f26a0c0f635d6906189fd6fbafaf8`.
- Executed arms: `subject_to_subject_A_memory` exits 0 in 53 s at 23,243 MiB;
  `background_to_background_A_memory` exits 0 in 54 s at 23,243 MiB. Their
  JSONLs retain the exact base order and report slots 1/2, source counts
  541/542 and 922/921, and target query counts 1,401 and 3,033 respectively.
  Both new arms are exactly equal to reset through saved clean block 7.
- Human result: full A restores greenhouse/orchids. Subject-only routing
  substantially affects woman-region appearance but leaves a visible green
  greenhouse-like leak and a retrieval-boundary discontinuity; background-only
  routing strongly restores greenhouse arches/orchids while preferentially
  retaining the woman at early target frames. This supports spatial
  separability of the raw-KV effect, but contradicts clean identity-only
  recall in this one-seed oracle.
- Evidence: `docs/ATTENTION_MEMORY_POLICY_FIXED_GRID_SELECTIVE_RECALL_20260809.md`,
  the exact-mask preflight overlays/audit, policy JSONLs, raw tensors, MP4s,
  and `comparison/four_arm_recall_sheet.png`. This is not a novel masking
  method claim and is not paper-level generalization.

## E20260809-AMP-SUBJECT-CORE-BOUNDARY-ABLATION (executed; one seed)

- Fixed protocol and controls: exact reset→establish→delayed block-8 recall,
  seed 101, A IDs 6/7, all 30 layers, `transition_no_sink`, `replace_recent`,
  and the existing six-frame local/current base. Reused reset-only and
  subject-full; ran only erode1, erode2, and full-minus-erode1 boundary ring.
- Deterministic 8-connected erosion counts are source 426/427 and target 366
  per frame for erode1; 322/323 and 276 for erode2; 115/115 and 101 for ring.
  Source H/W coordinates and slots 1/2 are preserved; every new arm is equal
  to reset through saved block 7.
- Human result: both eroded cores still restore the A1 woman together with a
  bright local greenhouse-like structure. The ring alone mainly leaves A2's
  woman intact and yields a weaker local edge halo. Boundary filtering reduces
  magnitude but does not eliminate local scene recall from erode2.
- Claim status: the boundary is a contributor, not the primary explanation;
  raw subject-core K/V is context-entangled in this one-seed oracle. Do not
  claim clean spatial recovery or generalize beyond this manual experiment.
- Evidence: `subject_core_boundary_ablation/preflight/mask_audit.json` SHA-256
  `7da283110bb0849f9b5b576e796d31c5ba8e9d63c6cf5c91e6ea9f36776419b0`,
  policy logs, raw outputs, and
  `comparison/five_arm_subject_core_boundary_sheet.png`.

## E20260809-AMP-ERODE2-ALPHA-STRENGTH (executed; one seed)

- Method: at only selected erode2 historical-query outputs, interpolate
  `O_base + alpha * (O_mem - O_base)`. Alpha 0 hard-bypasses history and reuses
  reset-only; alpha 1 uses the untouched prior erode2 path. Background/local/
  current attention remains baseline.
- Matched settings: seed 101, A IDs 6/7, target block 8, pulse-1, all 30
  layers, 322/323 source tokens, 828 target queries, slots 1/2, exact six-frame
  context, and all automatic policy machinery disabled. Only alpha changed.
- Results: new 0.10/0.25/0.50 arms exit 0 in 49/49/51 s at 23,243 MiB. Their
  JSONLs confirm the normal base and numerical reset equality through block 7.
- Human conclusion: 0.10 is near-inert; 0.25 produces subtle appearance
  perturbation while preserving snow but no verified A1 identity recovery;
  0.50 and 1.00 show increasingly strong A1 woman/local-scene flash, followed
  by A2-scene reconciliation and hybrid appearance. Lower strength reduces,
  rather than solves, the scene/identity tradeoff.
- Evidence: five-alpha pre/recall/post sheet SHA-256
  `c02acedcbff8e81d9d13bb594ff429d90752ba28c299e130891312dc96258ca7`
  and companion metrics under `subject_core_boundary_ablation/alpha_strength/`.
  No temporal alpha scheduling or additional sweeps were run.

## E20260809-AMP-ERODE2-DMD-TIMESTEP-SELECTIVITY (executed; one seed)

- Observed, not assumed, DMD execution schedule is high→low
  `[1000.0, 937.5, 833.3333129882812, 625.0]`; clean cache pass is timestep
  0. New JSONLs record schedule/gate vectors at target block 8.
- Reused reset and alpha-0.50 all-step-plus-clean controls. New late-1/no-clean,
  late-2/no-clean, and late-2/clean arms keep every other erode2 oracle setting
  fixed and pass numerical block-7 reset equality plus unchanged-base checks.
- Human result: latest-one nearly removes greenhouse leakage but produces ugly
  face perturbation; latest-two gives stronger A1-like appearance and small
  local leakage while preserving most snow; all-step remains the strongest
  local overwrite. Clean-pass history does not increase later propagation: it
  makes blocks 9--10 more reset/A2-like than late-2/no-clean in this oracle.
- Claim status: late timesteps spatially reduce scene rewrite but do not yet
  deliver artifact-free appearance recovery. Evidence is one prompt/seed only;
  do not generalize or tune further from this result.
- Evidence: five-arm sheet SHA-256
  `fd3d5fff233972c6c7e2a663effabfbe433ef7c1371cc00ec7a05c4de36070e0`
  and metrics/policy logs under `subject_core_boundary_ablation/dmd_timestep_selectivity/`.

## E20260809-AMP-ERODE2-CLEAN-PASS-ONLY (executed; one seed)

- Method: add `clean_only` to the existing observed-order gate. It disables
  history at all DMD calls `[1000.0, 937.5, 833.3333129882812, 625.0]` and
  enables it only at timestep-zero clean cache update. The historical branch,
  erode2 masks, raw H/W positions, slots 1/2, all 30 layers, alpha, and normal
  six-frame base remain unchanged.
- Controls/results: reset-only and prior latest-two-plus-clean alpha-0.50 are
  reused. New clean-only alpha-0.50/1.00 runs exit 0 in 56/53 s at
  23,798/23,243 MiB. For both, generated block-8 latent and decoded frames
  81--92 are exactly reset-equal before/through the clean pass; block-9 latent
  diverges, proving a future-cache-only intervention.
- Human conclusion: clean-only prevents the A1/local-greenhouse recall-block
  flash and retains snow, but later causes a modest (alpha 0.50) to obvious
  (alpha 1.00) face/cheek/brooch hybrid/deformation. A1 braided-crown hair and
  credible A1 facial recovery are not established. The 8→9 discontinuity
  remains and grows at full strength.
- Claim status: clean cache writes can causally influence later rollout without
  directly modifying the retrieval block, but this one-seed oracle is not a
  clean identity correction or a paper-level generalization. Stop here; no
  extra time ranges, alpha, masks, layers, routing, segmentation, or schedule.
- Evidence: four-arm sheet SHA-256
  `0e39cc4ee70bc8741578d5b8251519f9f158d7990bbba6b09ad076bc02e2e7b4`, companion
  metrics, policy logs, raw tensors, and MP4s under
  `subject_core_boundary_ablation/clean_pass_only/`.

## E20260809-AMP-COMPACT-ENTITY-REPRESENTATION (executed; one seed)

- Method: a representation oracle, not a masking-method or identity-memory
  claim. Per layer/head, mean-pool only A1 full-subject raw K and V from each
  source frame into one token per frame (two tokens total), then expose them
  only to the same A2 full-subject queries. All matched reset→establish,
  block-8, pulse-1, all-30-layer, and disabled-automatic-policy settings stay
  fixed; reset and full spatial subject-KV are reused controls.
- Position: the key receives temporal RoPE at slots 1/2 only. H/W rotary
  components use identity and no H/W coordinate exists for pooled tokens. This
  is an explicit neutral-spatial treatment enabled by factorized Infinity-RoPE,
  not a fake source grid location. Audit confirms 541/542 source tokens,
  1/1 pooled tokens, 1,401 target queries, and the unchanged 9,360-token base.
- Result: new compact arm exits 0 in 51 s at 23,243 MiB. It suppresses the
  recognizable local greenhouse reconstruction relative to full spatial KV,
  but produces a dark/severely distorted subject at recall and later a
  perturbed A2-like woman. A1 hair and credible facial identity are absent.
- Claim status: mean-pooled raw K/V removes useful spatial layout and does not
  create a viable semantic/entity memory. Stop raw-KV spatial injection tuning;
  this is one prompt/seed representation evidence only.
- Evidence: compact audit SHA-256
  `7ab974612986d998f10be0dafbfd92f7095703f8fbe3b440a39054571ea29698` and
  three-arm sheet SHA-256
  `90768fe28e7e72dbe9a13b8a48d9477722157f52c5dbc520beed62e22feaf7cb` under
  `compact_entity_memory/`.

## E20260809-AMP-RAW-KV-IDENTITY-STOP (decision)

- Across the verified reset→establish delayed-recall controls, raw historical
  self-attention KV is strong and source-specific as a perturbation mechanism:
  A-memory changes differ from B/wrong-source memory, and full subject KV can
  restore A1-like local appearance. It is nevertheless unsuitable for clean
  identity recall under the tested oracle variants. Spatial KV is scene-
  entangled; erosion, alpha, layer/lifetime, and DMD-time selectivity trade
  useful recall against local A1 scene leakage or artifacts; clean-cache-only
  changes later state without credible identity recovery; temporal-only mean
  pooling removes scene layout but destroys usable subject content.
- Decision: stop raw self-attention-KV identity injection. Do not add more
  alpha, layer, timestep, mask, pooling, routing, tracking, segmentation, or
  automatic-memory-policy sweeps to this branch.
- Next hypothesis is prepared but unexecuted: a manual entity-latent oracle.
  The live pipeline already holds generated VAE latents as
  `[B,F,16,60,104]`; its 30x52 token masks lift to 60x104 by 2x2 replication.
  The clean isolation point is after unchanged final block-8 DMD prediction
  and before writing `output`: source A1 subject patches can be retained from
  latent frames 6/7, while only A2 subject latent cells are replaced/conditioned
  and the baseline latent is retained for the timestep-zero cache pass. This
  keeps both the DMD schedule and cache/background path unchanged. This is a
  proposed oracle, not an implemented or evaluated method.

## E20260809-AMP-SUBJECT-LATENT-PATCH (executed; one seed)

- Method: a direct VAE-latent representation oracle, not a learned
  identity-memory method. The verified reset→establish schedule is unchanged:
  seed 101, A source IDs 6/7, reset with `transition_no_sink`, target block 8,
  all four DMD calls unchanged, six-frame / 9,360-token base, and automatic
  routing/archive/consolidation/decay disabled. Reset-only and the existing
  full subject-KV arm are reused controls.
- Intervention: after final block-8 DMD prediction, before its output-latent
  write only, masked A1 VAE latents `[B,16,60,104]` are copied from source
  frames 6/7. The 30x52 masks lift by exact 2x2 replication. Target frames map
  to source 6, mean(6,7), and source 7. To avoid reading unmasked A1
  background, only source∩target support is copied: 386, 386, and 387 target
  tokens, respectively; target-only cells retain the baseline prediction.
- Isolation checks: the patch audit records `outside_target_equal=true` and
  `outside_target_max_abs=0.0` at the latent intervention. Its clean cache
  input is the original baseline `denoised_pred`; saved clean tensors for
  blocks 8 and 9 are exactly reset-equal (max absolute difference 0.0). Thus
  the patch cannot alter the AR cache write. RGB outside-mask differences are
  not expected to be bit-exact because decoding has a spatial/temporal
  receptive field; this does not alter the latent/cache isolation check.
- Human result: within block 8, the A1 woman is visibly recognizable (face,
  high bun/hair, blue clothing) while the snowy observatory remains intact
  outside the subject region and no greenhouse/orchid reconstruction appears.
  The three target frames are consistent enough to show the fixed mapping, but
  the hard mask makes a pasted-looking subject with visible boundary/seam
  behavior. The immediately following decoded frame is a ghostly A1/A2 blend;
  it largely resolves by later A2 frames because the generation/cache path was
  baseline. This is direct latent-patch behavior, not credible clean identity
  transfer.
- Interpretation: compared with raw spatial subject-KV, latent masking removes
  the old-scene rewrite but exposes an alignment/blending problem. It is
  evidence that VAE latent content can be spatially separated from the A1
  greenhouse in this oracle, not evidence that it provides usable long-term
  identity memory. Stop after this single oracle; no alpha, feathering,
  tracking, warping, persistence, or further representation sweep is run.
- Evidence: patch event in
  `outputs/attention_memory_policy_fixed_grid_selective_recall/latent_subject_patch/memory_policy.jsonl`,
  raw tensor SHA-256 `8dd389e847d36c227e560c0780761a0ee0765ea8bf2541216c1e839655db6967`,
  and three-arm review sheet
  `latent_subject_patch/comparison/three_arm_latent_subject_patch_temporal_sheet.png`
  SHA-256 `dbeb89a20f474582708aa4b30261e934655ad97c60ffd39df100d990c2776188`.
