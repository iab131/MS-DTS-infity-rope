# Inference Findings

Append only verified runtime, cache, experiment, or environment findings here.
Separate observed behavior from proposals, and record commands/tests for every
claim.

## 2026-08-04: Non-Contiguous Context Tolerance Phase 1

### Observed cache behavior

- A causal block runs several denoising forwards against the same cache slots;
  the final clean-context forward overwrites those slots and supplies the KV
  used by later blocks.
- Values and non-sink keys are stored raw. The first sink key is stored after
  RoPE and is explicitly copied back into attention without re-rotation.
- With `local_attn_size: 6` and `sink_size: 1`, the active DMD configuration
  has a six-latent-frame cache. Scene flush resets cross-attention and keeps
  the sink plus the final two latent frames.

### Phase 1 result

- Commit `5ea0ab4` adds the opt-in `--noncontiguous-kv` experiment. It captures
  raw clean-pass KV from selected one-based source blocks and prepends it at a
  selected target block using temporary contiguous RoPE positions.
- The target-block log reports exact retrieved, local, and current frame/token
  counts. No MemoryStore, descriptors, routing, decay, compression, or
  consolidation was added.
- Focused verification:

  ```bash
  conda run -n wan python -m unittest tests.test_noncontiguous_kv -v
  ```

  Five tests passed, including the compact FlashAttention check. No full GPU
  generation was run.

### Mismatches and limits

- The attention-memory policy says all cached self-attention keys are raw;
  the live sink-key exception above makes that inaccurate.
- The document's blocks 2--4 correspond to nine latent frames in this
  checkout (`num_frame_per_block: 3`), not three frames.
- Phase 1 currently prepends retrieved KV to the live local context. It does
  not implement the document's matched-total-span condition that would reduce
  the local window to compensate.
- `conda run -n wan python inference.py --help` fails before argument parsing:
  installed torchvision `0.26.0+cu130` has no `torchvision.io.write_video`.
  `HEAD` already imported that symbol; this was left unchanged because it is
  outside the experiment and dependency changes were forbidden.

## 2026-08-04: Matched Non-Contiguous Context Refinement

### Observed implementation correction

- The first Phase 1 implementation prepended full source blocks, increasing
  attention length. That did not satisfy the research experiment's matched
  context condition.
- Commit `3f004b4` replaces only the two non-sink local slots. The target
  context remains exactly six latent frames (9,360 tokens):
  `[sink, recent, recent, current x3]`, `[sink, history, recent, current x3]`,
  or `[sink, history, history, current x3]`.
- The transformed sink key is read from its original cache slot and copied
  directly into the temporary context. It is never re-RoPEd. Historical KV is
  raw clean-pass data, sliced per latent frame, RoPEd only for transient slots
  1--2, and never written to the persistent cache.
- `coherent_history` selects final latent frames from the newest configured
  source blocks; `random_history` uses a private seeded Python RNG, so it does
  not change the Torch RNG stream used by denoising.

### Verification and export fix

- `conda run -n wan python -m unittest tests.test_noncontiguous_kv -v` passed
  eight focused tests after the refinement. No full GPU generation was run.
- Commit `46880a1` moves `torchvision.io.write_video` behind a lazy import and
  falls back to the already-installed `imageio` MP4 writer. The prior
  `conda run -n wan python inference.py --help` import failure is resolved.

## 2026-08-04: Matched Phase 1 GPU Evidence

- The matched baseline/coherent/random matrix completed at 480x832 with the
  same 7.5-second prompt, seed 101, EMA checkpoint, source blocks 2--4, and
  target block 8. All three contexts were exactly six frames / 9,360 tokens.
- Coherent history selected source-block-4 global frame 11. Seeded random
  history selected source-block-3 global frame 6. Neither could select sink,
  current, or retained-recent context frames because the candidate pool was
  captured source-block frames only.
- See `docs/NONCONTIGUOUS_PHASE1_GPU_20260804.md` for commands, telemetry,
  output paths, target-block comparison, and checksums. This is execution
  evidence only; no visual-quality conclusion has been made.

## 2026-08-04: Decoded RGB comparison and A-B-A preparation

### Observed decoded-frame comparison

- RGB frames were decoded with the installed OpenCV and compared as uint8 RGB.
  RGB MAE is the mean absolute component error; PSNR uses peak 255. Exact
  matches have infinite PSNR and are omitted from finite-PSNR averages.
- The causality control is decoded frames 0--80, before target block 8 (the
  target starts at decoded frame 81). It is not exact: baseline/coherent and
  baseline/random first diverge at frame 45, while coherent/random first
  diverge at frame 49. Thus the original three-run output is confounded for a
  strict target-only causal comparison.

| Pair | First differing frame | Control max MAE / min PSNR | All-frame mean MAE / finite mean PSNR / min PSNR |
| --- | ---: | --- | --- |
| baseline vs coherent | 45 | 1.064231 / 43.955 dB | 2.510562 / 34.230 dB / 20.331 dB |
| baseline vs random | 45 | 1.060765 / 43.943 dB | 3.516948 / 33.492 dB / 17.685 dB |
| coherent vs random | 49 | 1.059219 / 43.956 dB | 3.564348 / 32.325 dB / 17.833 dB |

- Frames 0--44 are exact for all pairs. The first differing decoded frame is
  immediately after source block 4 in the prior layout, which is consistent
  with an opt-in clean-KV capture scheduling/allocation effect, but the exact
  lower-level CUDA mechanism has not been isolated without a new GPU run.

### Corrective preparation, not a new experiment result

- Opt-in baseline now follows the same source clean-KV capture and CPU-offload
  schedule as the history variants. It still injects no history. The fully
  disabled path remains unchanged.
- Captured clean K/V is popped to CPU after every source clean pass. At the
  target, only manually selected frame K/V is packed on CPU and materialized
  on GPU one transformer layer at a time.
- See `docs/NONCONTIGUOUS_PHASE1_ABA_PREPARATION_20260804.md` for the
  unlaunched matched oracle A/B/A2 prompt, frame IDs, commands, and VRAM
  estimate. No MemoryStore, descriptors, routing, compression, or matrix run
  was added.

## 2026-08-04: A-B-A causality gate

- Normal inference, opt-in capture/offload baseline, and a repeated opt-in
  baseline were exactly equal at every observed boundary: all ten clean latent
  hashes, all ten complete persistent K/V hashes, cache indices, and the raw
  decoded RGB tensor before MP4 encoding. Every direct max absolute difference
  was `0.0`.
- The gate used the A/B/A2 prompt but did not run either history-injection
  mode. The prior three-video divergence therefore does not reproduce in the
  current CPU-offloaded capture path with the A-B-A source blocks 3 and 6.
- Full commands, per-block SHA-256 evidence, runtimes, and raw metrics are in
  `docs/NONCONTIGUOUS_PHASE1_CAUSALITY_GATE_20260804.md` and
  `outputs/noncontiguous_phase1_causality_gate/metrics.json`.
- The standalone gate runner must explicitly disable gradients, as
  `inference.py` does. Its first attempt omitted this and OOMed at block 2 from
  retained autograd activations; no configuration changed for the recorded
  three-run gate after that correction.

## 2026-08-04: Matched oracle A-B-A GPU result

- Commit `361ed7e` completed the baseline, same-entity (manual global latent
  ID 8), and wrong-entity (ID 17) matrix at target block 8 with matched
  six-frame / 9,360-token contexts. Both history runs are exactly equal to
  baseline through clean block 7 and raw decoded frame 80.
- Both first diverge at target global latent ID 21 and decoded frame 81. This
  passes the target-causality condition for the A-B-A comparison. See
  `docs/NONCONTIGUOUS_PHASE1_ABA_GPU_20260804.md` for raw hashes, commands,
  telemetry, output paths, and the blocks 7--10 side-by-side artifact.
- One outer harness command exited 130 because `tee` was given a not-yet-made
  output directory. Its baseline inference artifact was already complete;
  individual same/wrong commands then completed with unchanged settings. The
  failure and distinction are recorded in the GPU report.

## 2026-08-07: Long-gap oracle preparation

- The requested 10 s / 15 s / 10 s prompt resolves in the live three-frame
  block implementation to A blocks 1--13, B 14--33, and A2 34--47. Per-scene
  flooring gives A 9.5625 s, while global latent rounding appends a final A2
  block for 10.5 s. This is documented rather than normalized away.
- Target block 34 will receive manual A ID 38 or B ID 44 while preserving the
  six-frame / 9,360-token matched context. The planned output stores only
  blocks 33--35 clean latents and raw decoded video before MP4 encoding.
- The central paper-facing prompt/result ledger is
  `docs/NONCONTIGUOUS_PHASE1_PAPER_LEDGER.md`. No long-gap GPU generation has
  been launched.

## 2026-08-07: Long-gap oracle result

- E20260807-LG-P1 completed all three runs at commit `c0d3f9e`. Same-entity
  ID 38 and wrong-entity ID 44 are exactly equal to baseline through clean
  block 33 and raw decoded frame 392, then first diverge at target block 34,
  global latent 99, decoded frame 393.
- This is one-seed target-causality evidence only. No human visual review or
  identity metric was run; raw RGB MAE does not demonstrate identity quality.
- Baseline-first sampled device VRAM is higher than later runs and is treated
  as warm-up/order-confounded, not a performance effect. Full raw data and
  interpretation limits are in
  `docs/NONCONTIGUOUS_PHASE1_LONG_GAP_ORACLE_GPU_20260807.md`.

## 2026-08-07: Wrong-memory negative-control invalidation

- Human review found that the intended robot B shot in the long-gap sequence
  retained Amara's identity and leaked robotic arms. Its wrong-history arm is
  invalid as a semantic-negative control and must not be used to claim that
  wrong-entity memory has an identity effect.
- The follow-up short true-wrong oracle instead uses an explicitly car-only B
  shot and requires a human contact-sheet check of source ID 17 before any
  wrong-history run. See
  `docs/NONCONTIGUOUS_PHASE1_TRUE_WRONG_MEMORY_PREPARATION_20260807.md`.

## 2026-08-07: Follow-up oracle results

- Long-gap r=2 (A IDs 37 then 38) kept six frames / 9,360 tokens and first
  diverged at target block 34/ID 99/raw frame 393. Its block-34/35 maxima
  (4.1640625/5.046875) exceeded r=1 (3.703125/4.203125), but RGB MAE was lower
  (0.065835 vs 0.075158): non-monotonic single-seed perturbation evidence, not
  visible-memory-strength or identity evidence. Baseline/r=1 use `c0d3f9e`;
  r=2 uses `5d0eda3`.
- The first r=2 launcher attempt supplied retrieval count 1 and exited 2 at
  parsing, without a model output; it is preserved in
  `outputs/noncontiguous_phase1_long_gap/same_entity_history_r2/attempt1_run.json`.
- True-wrong source ID 17 passed the car/no-person condition: decoded frames
  65--68 visibly show yellow car/no woman or humanoid. Greenhouse-like
  background leakage remains. Both interventions equal baseline through block
  7/frame 80 and first differ at target block 8/ID 21/frame 81; no output
  identity/quality review was performed.
- Paper-ready results and limits: `NONCONTIGUOUS_PHASE1_PAPER_LEDGER.md`.

## 2026-08-07: Attention-as-Memory-Policy implementation facts

- The opt-in scaffold captures only clean-pass raw K/V, clones every selected
  frame to CPU, and moves only routed frame K/V back to GPU as transient
  per-layer attention input. Persistent cache writes, rolling indices, and
  eviction are not used for MemoryStore entries.
- The live 30-layer model has independent per-layer attention calls. Historical
  CPU K/V is now retained only for configured injection layers; other layers
  receive no transient K/V and retain normal local attention. Representative
  descriptor layers remain independent mean-pooled raw-K readers.
- The six-frame local cache plus three-frame block means matched
  `replace_recent` has only two non-sink local slots available after rolling
  eviction. The full policy defaults to `prepend`; `replace_recent` skips and
  logs requests that cannot fit rather than enlarging the matched context.
- `prepend` keeps the already-RoPEd sink in slot zero, places retrieved raw K
  after it, and shifts only raw local/current RoPE coordinates. Replacement
  preserves the existing Phase 1 slot coordinates. Neither behavior has been
  GPU-tested as part of the full pipeline.
- A `sink_only` transition has no local raw K for the first new-scene query.
  The implementation explicitly logs and uses the final pre-transition raw-K
  descriptor as a fallback. Its semantic suitability is an assumption, not a
  validated retrieval signal.
- No prompt-conflict decay, FlashAttention-weight utility, descriptor
  semantic-accuracy result, archive retrieval, or identity-quality claim was
  added. CPU validation command: `conda run -n wan python -m unittest
  tests.test_inference_cli tests.test_attention_memory_policy
  tests.test_noncontiguous_kv -v`.

## 2026-08-07: Attention-memory policy layer and transition refinements

- Descriptor layers and injection layers are independent. The MemoryStore now
  preserves historical CPU K/V only for configured injection layers; every
  other transformer layer retains normal local attention. Descriptor layers
  still read mean-pooled raw clean K, so they do not imply K/V injection.
- Automatic descriptor routing is disabled by default only on the first block
  after a scene transition. JSONL logs identify `local_raw`,
  `pre_transition_raw`, or unavailable query source and distinguish an
  `automatic_transition_disabled` skip from a manual override. Manual memory
  can be constrained to explicit target blocks.
- Existing decoded source frames verified global A IDs 6,7 as Amara-only and
  B IDs 16,17 as yellow-car-only. All retain greenhouse-background leakage;
  this validates source entity contrast, not isolated-scene semantics.
- Mistake corrected in the unrun combined-policy oracle: live duration parsing
  of the exact woman → car → woman prompt yields `[3,3,4]` blocks, so A2 starts
  at block 7/current IDs 18--20, not the inherited block-8 target. The planned
  manual-retrieval commands and ledger correction now target block 7.

## 2026-08-07: Combined-policy short A/B/A2 oracle

- The normal baseline and the `sink_only` hard-flush arm first differ at saved
  raw RGB frame 33 (the A→B boundary). This means hard flush is a real policy
  intervention and manual-memory comparisons must use hard_flush, not normal
  baseline, as their causal control.
- Hard_flush versus either manual all-layer two-frame history is exactly equal
  through raw frame 68 and saved clean block 6; both first differ at raw frame
  69 / clean block 7 local latent 0. Target JSONL verifies `[sink,history,
  history,current×3]`, positions 0--5, and 9,360 tokens for A IDs 6,7 and car
  IDs 16,17.
- A requested/scheduled 7.5-second `[3,3,4]` prompt saved 117 RGB frames at
  16 FPS (7.3125 seconds): the first shot has raw frames 0--32, unlike the
  nominal 2.25-second block duration. Paper records must distinguish requested,
  scheduled-block, and saved-RGB durations.
- GPU execution surfaced two scaffold defects before successful manual runs:
  clean capture referenced undefined `scene_id`, and JSONL config could not
  serialize manual target-block sets. Both have focused regression tests in
  commit `0efd5ad`; the first attempts remain in output logs. A missing
  workspace `wan_models` link also prevented the initial baseline load and was
  restored as a local symlink to the existing `/home/sigasia2026/models` tree;
  no dependency or generation setting changed.
- Raw RGB MAE/PSNR shows perturbation (hard vs A memory 0.101799488 / 12.677845
  dB; hard vs car 0.099336967 / 13.134525 dB) but does not measure identity or
  visual quality. Human visual conclusions remain pending.

## 2026-08-08: Identity-selectivity source gate

- The requested white hair streak plus small red under-left-eye star did not
  survive prompt rendering at seed 101. Across inspected A source IDs 0--8,
  the woman has a black bob and yellow patch but a red eye-level beam instead
  of both required identity anchors. This disqualifies the source for a
  “correct-memory” identity-selectivity condition.
- The B source has only a bright blue/turquoise pickup truck and no
  people/humanoids, but it visibly remains in a greenhouse. It is a person-free
  object source, not the requested desert environmental negative.
- Baseline/hard A2 screening also has strong greenhouse recall and no snowy
  mountain; white streak and small star are absent. The yellow patch persists
  in one hard-flush screen without retrieval, so it cannot be called a memory
  effect. The manual C/D arms were correctly withheld rather than using invalid
  sources. Full artifacts, human-review statements, and commands are in
  `ATTENTION_MEMORY_POLICY_IDENTITY_SELECTIVITY_20260808.md`.

## 2026-08-08: Transition-retention ablation (woman → truck)

- Human review from the preceding source-gate experiment is now recorded as a
  separate transition finding: the ordinary baseline produced a truck while
  retaining greenhouse/farm context, then a catastrophic car→woman carry-over;
  `sink_only` removed that catastrophic entity carry-over at A2 but did not
  remove greenhouse persistence at A→B. This is evidence about transition
  retention, not identity retrieval.
- A matched seed-101 woman→truck retention ablation independently reproduces
  the A→B half: `sink+2` and `sink+1` show a woman's head in the early truck
  windshield and retain the greenhouse across sampled B frames. `sink_only`
  removes the obvious post-blend person but retains the greenhouse. The
  opt-in, experimental `transition_no_sink` excludes both sink and prior local
  K/V only for B block 4; frame 33 is still a cut blend, but frame 34 onward
  sampled frames show an intact blue pickup, desert, and no person.
- Exact first-B contexts are: `sink+2` six frames/9,360 tokens;
  `sink+1` five/7,800; `sink_only` four/6,240; and `transition_no_sink`
  three/4,680. Their current positions are respectively preceded by
  `[0,1,2]`, `[0,1]`, `[0]`, and no retained positions, with current positions
  `[45,46,47]` in all cases. In the experimental arm, old state is excluded
  from the first B attention call, then normal DMD/clean writes begin at slot
  zero and replace the old physical sink with B state.
- All four outputs are bit-identical through clean block 3 and first diverge at
  B block 4/raw RGB frame 33. This supports a causal local-attention effect at
  the intended cut but remains one prompt/seed, qualitative evidence. Full
  commands, raw hashes, review sheets, video, and limits are in
  `docs/ATTENTION_MEMORY_POLICY_TRANSITION_RETENTION_GPU_20260808.md`.

## 2026-08-08: Reset-then-delayed-recall oracle

- A `transition_no_sink` reset writes the first clean new-scene frame into
  cache slot zero. At delayed A2 block 8 that physical preserved sink is global
  frame 18, not original frame 0; JSONL now records it accurately.
- Matched two-frame delayed history uses live `replace_recent`, physically
  `[sink:18,history,history,current:21..23]`: six frames/9,360 tokens. The
  new logger records preserved sink, history K `[1,2]`, current K `[3,4,5]`,
  and query `[21,22,23]`; logical slots 0--5 are not query positions.
- Two logging defects were caught and preserved: a new sink was labeled ID 0,
  then Python `[-0:]` incorrectly listed replaced local IDs and 8/12,480.
  Both are logging-only output attempts, excluded from evidence; regression
  tests cover the sink label, actual coordinates, and zero-local slice. The
  reset raw tensor is byte-identical across the logging-only rerun.
- Official reset/correct/wrong outputs equal through A2 block 7/raw frame 80
  and first diverge at delayed target block 8/frame 81. Human review: reset
  reaches the snowy observatory; A history recalls greenhouse/orchids; B-car
  history recalls yellow car/desert. This demonstrates one-seed source-selective
  steering, but not identity-only recall without old-scene leakage. See
  `ATTENTION_MEMORY_POLICY_RESET_THEN_RECALL_GPU_20260808.md`.

## 2026-08-09: Coarse layer-selective A-history ablation

- The verified reset-then-recall setup was held fixed at A IDs 6/7, block 8,
  `transition_no_sink` reset, matched six-frame / 9,360-token context, and
  all manual-routing safeguards. Only injection layers varied: 0--9, 10--19,
  and 20--29; the verified 0--29 result was reused unchanged.
- All layer ranges are exactly equal to reset through A2 block 7/raw frame 80
  and first diverge at block 8/frame 81. Each JSONL records
  `[sink:18,history:6,history:7,current:21..23]`, logical slots 0--5, and
  preserved-sink/history-K/current-K/query coordinates.
- Human review finds no ten-layer range with verified A-woman recall beyond
  the A2 prompt baseline, no greenhouse/orchid recall, and retained snowy
  observatory across reviewed blocks 8--9. In contrast, all-layer A history
  strongly restores greenhouse/orchids; prior verified B history restores
  car/desert. This is source-selective but scene-entangled historical K/V
  recall, not evidence of subject/scene layer separation.
- No coarse range meets the stated identity-only promise, so no finer sweep was
  launched. The paper-ready raw record and five-way comparison are in
  `docs/ATTENTION_MEMORY_POLICY_LAYER_SELECTIVE_GPU_20260809.md`.

## 2026-08-09: Manual retrieval lifetime audit and L0--9 oracle

- Inspection of both verified policy JSONLs corrected an assumption: with
  `--memory-manual-target-blocks 8`, IDs 6/7 were injected at A2 block 8 only;
  blocks 7, 9, and 10 logged `manual_target_not_selected`. Manual IDs with no
  target restriction were the generic persistent path. Explicit lifetime is
  required for an unambiguous pulse experiment, not a baseline change.
- L0--9 pulse-1, pulse-2, and persistent retain matched six-frame/9,360-token
  recall contexts. Logs show IDs 6/7 at block 8 only, 8--9, and 8--10; added
  raw divergence begins exactly at RGB 81, 93, and 105 respectively.
- One-seed human review finds no incremental Amara appearance correction beyond
  the explicit A2 prompt, no visible greenhouse/orchid recall, and no obvious
  extra flicker. Adjacent-frame RGB change is not a flicker metric. Full raw
  record is `docs/ATTENTION_MEMORY_POLICY_RETRIEVAL_LIFETIME_GPU_20260809.md`.


## 2026-08-09: L0--9 manual source-specificity oracle

- The missing wrong-memory arm used prior-gated B IDs 16/17: yellow-car-only,
  no woman/humanoid, with greenhouse-background leakage retained as source
  confound. It injected only at A2 block 8 with the exact six-frame/9,360-token
  context and same preserved-sink RoPE coordinates as A IDs 6/7.
- Correct-A and wrong-car both equal reset through RGB 80 and diverge at target
  RGB 81; they differ from each other at the same target. This is source-
  dependent raw perturbation, not source-dependent semantic recall.
- Human review through A2 block 10 finds small face/pose/detail changes in both
  arms but no verified original-A feature beyond the A2 prompt and no car/desert
  trait in wrong-car. Visible evidence is consistent with generic transient
  facial perturbation. Do not infer source independence from raw differences
  alone. Full record: `docs/ATTENTION_MEMORY_POLICY_L0_9_SOURCE_SPECIFICITY_GPU_20260809.md`.

## 2026-08-09: Fixed-grid selective-recall oracle preflight (superseded below)

- The manually supplied inclusive spans reconstruct exactly into distinct
  30x52 source masks for A IDs 6/7 and one target conservative-union mask.
  CPU-only overlays at reset frames 26/30 and target frames 81/85/89 were
  checked; `mask_audit.json` records the exact source/query indices,
  row/column coordinates, slots 1/2, dilated complements, and input hashes.
- The protocol requires manual IDs 6/7, target block 8, `replace_recent`, and
  `transition_no_sink`. After the no-old-context A2 block 7, the first clean
  frame is recorded as sink 18. The block-8 base remains
  `[sink:18,local:19,local:20,current:21,current:22,current:23]`; selective
  subject or background history is separate from that base.
- At this preflight point the model arms had not run; the executed one-seed
  result immediately below supersedes that status.

## 2026-08-09: Fixed-grid selective-recall oracle (one-seed result)

- Both authorized arms completed with all 30 layers, source IDs 6/7, target
  block 8, `transition_no_sink`, `replace_recent`, and no automatic policy
  mechanisms. JSONLs verify the unchanged six-frame base
  `[sink:18,local:19,local:20,current:21,current:22,current:23]`; only the
  historical branch is selectively gated.
- Subject-to-subject uses 541/542 historical subject tokens and 1,401 target
  queries. It changes face/hair/clothing-region appearance and preserves much
  of the snow scene, but leaks a bright greenhouse-like structure and creates
  a pronounced target-boundary discontinuity. It is not clean identity-only
  recall.
- Background-to-background uses 922/921 dilated-complement source tokens and
  3,033 target background queries. It strongly restores greenhouse
  arches/orchids and disrupts snowy-observatory preservation, while the woman
  remains comparatively reset-like at early target frames. This is the
  expected direction for a spatially separable background effect.
- The four-arm sheet is
  `outputs/attention_memory_policy_fixed_grid_selective_recall/comparison/four_arm_recall_sheet.png`.
  The result is an oracle separability observation only; no automated masks,
  tracking, SAM, descriptors, or further memory-policy work was added.

## 2026-08-09: Subject-core / boundary ablation (one-seed result)

- The original manual 30x52 subject masks were eroded with an 8-connected
  one- and two-token binary erosion; the full-minus-erode1 set is the boundary
  ring. Preflight overlays cover all four variants at both A source frames and
  all three A2 target frames, and the audit preserves source coordinates,
  slots 1/2, query indices, and the six-frame base order.
- Erode1/erode2/ring source counts are 426/427, 322/323, and 115/115; A2
  per-frame target counts are 366, 276, and 101. New arms are numerically
  equal to reset at saved clean block 7 and log only the selective historical
  branch at block 8.
- Both eroded cores still bring back the A1 woman *and* a bright local
  greenhouse-like background structure, despite preserved snowy background
  outside the subject. The ring alone gives a weaker localized halo while the
  A2 woman remains largely reset-like. Therefore the boundary contributes but
  cannot explain the leakage: the raw subject core is context-entangled in this
  oracle.
- Pixel MAE/discontinuity values are supporting proxies only. The five-arm
  sheet and metrics are under
  `outputs/attention_memory_policy_fixed_grid_selective_recall/subject_core_boundary_ablation/comparison/`.
  The subsequent alpha-only result is recorded below; no automatic masks,
  tracking, alpha blending, finer layer sweep, or new policy mechanism was run.

## 2026-08-09: Erode2 fixed-strength interpolation (one-seed result)

- The historical-only erode2 attention addition now supports alpha
  interpolation: alpha 0 bypasses the branch exactly, alpha 1 preserves the
  existing result, and intermediate strengths scale only the selected
  historical delta. Background queries and normal local/current attention are
  untouched.
- Reused alpha 0 reset-only and alpha 1 erode2 controls; new alpha 0.10/0.25/
  0.50 arms keep all other matched settings fixed and pass the same block-8
  provenance/base-context checks.
- Visual result: 0.10 is effectively reset-like; 0.25 gives subtle
  woman-region perturbation with retained snow scene but no verified A1
  appearance correction; 0.50+ increasingly produce the A1 woman plus a local
  green greenhouse flash. After the one-block pulse, the AR rollout largely
  restores the A2 scene while retaining a hybrid A1/A2-looking woman.
- The result shows a continuous reduction in raw perturbation, not a clean
  identity-correction operating point in this one seed. No temporal alpha
  schedule, automatic mask, tracking, alpha blending, or further sweep was
  added. Sheet and proxy metrics are under
  `outputs/attention_memory_policy_fixed_grid_selective_recall/subject_core_boundary_ablation/alpha_strength/comparison/`.

## 2026-08-09: Erode2 DMD-timestep selectivity (one-seed result)

- The actual DMD calls are `1000.0 → 937.5 → 833.3333129882812 → 625.0`
  (high→low noise); clean cache update is a separate timestep 0. The fixed-grid
  gate logs this observed schedule and activates either all, final one, or
  final two calls, independently of the clean pass.
- Final-step-only keeps the snow scene and nearly removes local greenhouse
  leakage, but causes an ugly/deformed face rather than clean A1 recovery.
  Final-two strengthens the A1-like woman with a far smaller local flash than
  all-step recall, but still yields face artifacts and a later hybrid.
- Clean-pass history in the final-two arm does not cause stronger propagation
  into blocks 9--10. Relative to no-clean, it leaves post-recall frames closer
  to reset/A2 while making the block-8→9 handoff somewhat larger.
- This is evidence for denoising-time selectivity as a scene-preservation
  lever, not a finished identity-recall solution. No extra timestep values,
  alpha values, masks, layers, routing, segmentation, or temporal schedule was
  added. Five-arm artifacts are under
  `outputs/attention_memory_policy_fixed_grid_selective_recall/subject_core_boundary_ablation/dmd_timestep_selectivity/comparison/`.

## 2026-08-09: Erode2 clean-pass-only recall (one-seed result)

- `clean_only` activates the same alpha-scaled, subject-to-subject erode2
  historical branch only for timestep-zero clean-cache capture. All four
  observed DMD calls remain baseline. The new alpha-0.50 and alpha-1.00 runs
  retain the source IDs 6/7, all 30 layers, target block 8, six-frame context,
  and disabled automatic policy machinery.
- Both strengths are exactly equal to reset-only for the generated block-8
  latent and decoded frames 81--92. Thus no mid-block greenhouse/scene flash
  can be attributed to this cache-only intervention. Block 9 diverges after
  the cache update, establishing cache-mediated causal influence on future
  generation.
- Visually, alpha 0.50 causes a modest later woman perturbation and alpha 1.00
  causes a clearer face/cheek/brooch deformation; neither recovers the A1
  braided-crown hair or a credible A1 face. Snow remains largely preserved and
  no greenhouse flash appears, but the 8→9 handoff is still less smooth than
  reset, especially at alpha 1.00.
- The result separates visible recall-block overwrite from future cache effects
  but does not solve identity recovery. Artifacts are under
  `outputs/attention_memory_policy_fixed_grid_selective_recall/subject_core_boundary_ablation/clean_pass_only/comparison/`.

## 2026-08-09: Compact entity-memory representation oracle (one-seed result)

- From each full A1 subject mask, raw K and V are mean-pooled independently per
  layer/head into one token per frame. The two compact memory tokens use source
  temporal slots 1/2 but no H/W coordinate: temporal RoPE is applied while the
  height/width rotary components are identity. Current/local/sink RoPE and the
  9,360-token base remain unchanged; only 1,401 A2 subject queries receive the
  compact branch.
- The arm executes successfully and confirms the intended non-spatial path in
  its JSONL. It avoids the full subject-KV arm's recognizable local greenhouse
  reconstruction, but makes the woman a dark, severely distorted silhouette
  during recall and leaves a perturbed A2-like woman later. No credible A1
  braided-crown hair or face recovery is visible.
- Therefore mean-pooled raw KV is not a usable entity-memory representation in
  this one-seed oracle. This is representation evidence, not a validated
  identity-memory method; raw spatial-KV injection tuning stops here. Artifacts
  are under
  `outputs/attention_memory_policy_fixed_grid_selective_recall/compact_entity_memory/`.

## 2026-08-09: Raw-KV identity branch closed; latent oracle prepared

- The tested raw-KV variants establish a strong, source-specific perturbation
  effect but not clean identity recall: spatial KV is scene-entangled, while
  temporal-only mean pooling removes recognizable scene layout and also destroys
  usable subject content. No more raw-KV alpha/layer/timestep/mask/pooling
  sweeps are warranted under this oracle.
- The next unexecuted hypothesis is subject latent memory. The live pipeline
  keeps generated VAE latents directly as `[B,F,16,60,104]` and decodes only at
  the end; masks map from 30x52 video tokens to those latent cells by 2x2
  replication. The clean planned insertion point is after the final unchanged
  DMD prediction for A2 block 8, before the output write. A later implementation
  must keep the baseline `denoised_pred` in the timestep-zero cache pass so
  subject-only output editing cannot contaminate the future cache/background.
- This is preparation only: no latent-memory code, GPU run, automatic mask,
  routing, or new method claim has been added.

## 2026-08-09: Subject latent-patch representation oracle (one seed)

- The only new arm keeps every DMD call and the clean cache input baseline,
  then directly writes masked source-latent content to block-8 output cells.
  Source masks for IDs 6/7 and the A2 target mask are lifted from 30x52 to
  60x104 by exact 2x2 replication. The fixed temporal map is source 6,
  half source 6 plus half source 7, then source 7. To avoid accidental
  greenhouse copying, only cells supported by both source and target masks are
  copied (386/386/387 token cells); unsupported target cells stay baseline.
- Numerical audit: outside-target output latents are exactly unchanged
  (`max_abs=0.0`), block-8 clean-cache input is exactly its baseline prediction,
  and saved clean latents for blocks 8 and 9 exactly equal reset-only. Hence
  the visible edit is not written into the autoregressive cache. Decoder RGB
  can nevertheless differ just outside the mask due to the decoder receptive
  field, so only the latent claim is bit-exact.
- Visual review: the patch makes the A1 woman's face, high bun/hair, and blue
  outfit recognizable during the three target frames without a greenhouse or
  orchid reconstruction; the snow scene stays substantially intact. It looks
  like a hard, spatially misaligned paste rather than identity transfer, with
  a visible mask-edge seam and a ghostly blend just after the patched block.
  By later A2 frames the baseline scene largely resumes. This is promising
  spatial separation but not a validated identity-memory representation.
- The three-arm temporal sheet is
  `outputs/attention_memory_policy_fixed_grid_selective_recall/latent_subject_patch/comparison/three_arm_latent_subject_patch_temporal_sheet.png`.
  No blending, warping, tracking, automatic mask, or additional sweep was
  added.

## 2026-08-09: Affine-aligned subject latent-patch oracle (one seed)

- The registration arm keeps the same unchanged DMD and clean-cache path, but
  uses a per-source 2D bbox affine warp at the existing output-latent write.
  Both A1 masks map from latent bbox `[14,2]–[75,59]` to target
  `[40,0]–[79,59]` with x/y scale `0.639344/1.035088`; source/target centroids
  and transforms are logged. The 16 channels use one FP32 sampling grid and
  the source mask uses the matching nearest warp. Target 2 averages separately
  warped sources only where both warped masks are valid.
- Exact checks pass: 1,301 valid latent cells per target frame, no output
  latent change outside target (`max_abs=0.0`), baseline clean-cache input,
  and exact reset equality for saved clean blocks 7--9. The snow outside the
  target latent mask is therefore preserved by construction.
- Visual result is negative for simple registration. The A1 face/bun/outfit
  remains visible, but affine alignment creates a split A1/A2 face-and-crown
  composite, hard seams, and a post-block ghost. Unlike the unaligned patch,
  it also makes local green/orange A1 greenhouse content visible within the
  admitted target patch. The three target frames repeat the same artifact; no
  smooth geometry/pose transfer is established.
- This stops after one registered arm. The comparison sheet and crops are in
  `outputs/attention_memory_policy_fixed_grid_selective_recall/affine_aligned_latent_subject_patch/comparison/`;
  no blending, feathering, tracking, optical flow, or further registration
  variant was added.

## 2026-08-09: Unaligned subject-latent cache-persistence oracle (one seed)

- `latent_subject_patch_persistent` reuses the exact unaligned block-8 patch,
  but feeds that modified latent to the timestep-zero clean cache call. It has
  no raw KV or later patch insertion. The policy log proves this distinct
  intervention (`clean_cache_input=patched_subject_latent`, not baseline),
  while block-8 output latents stay exactly unchanged outside the target mask.
- The cache has a causal future effect: clean block 7 remains reset-equal,
  block 8's exported pre-clean prediction remains reset-equal, and block 9
  diverges (max latent difference 4.6875). The source retrieval branch is
  expired at block 9; the later result is normal AR rollout from modified cache
  state rather than further memory injection.
- Visual review: the isolated patch ghosts then returns to the A2 crown/long
  hair woman. The persistent arm instead evolves into a clearer A1-like
  high-bun/downward-gaze woman with cobalt outfit/brooch through blocks 9--10.
  The initial 8→9 transition remains abrupt. It also preserves a local
  greenhouse-like green structure behind her, while the broader snow scene
  remains recognizable. Hair/bun is the strongest qualitative A1 attribute;
  face similarity is not established as an identity metric and cobalt clothing
  is prompt-shared.
- The fixed 7.5-second schedule ends at block 10, so no block 11 exists to
  inspect without violating the matched setup. Artifacts are under
  `outputs/attention_memory_policy_fixed_grid_selective_recall/latent_subject_patch_persistent/comparison/`.
  Stop here; no further persistence, mask, blend, warp, or raw-KV experiment
  was added.

## 2026-08-10: Hard-cut AR-state basic effect (consolidated)

- **Observed:** the executed woman-greenhouse → blue-pickup-desert ablation at
  seed 101 is equal through clean block 3 and first diverges at B block 4/raw
  frame 33. Its first-B contexts are 6/5/4/3 frames for `sink+2`, `sink+1`,
  `sink_only`, and `transition_no_sink`; this intentionally changes attention
  context size.
- **Human observation:** `sink+2`/`sink+1` retain woman and greenhouse cues;
  `sink_only` removes persistent woman after its blended first B frame but
  retains greenhouse; `transition_no_sink` is truck/desert-only in reviewed
  samples from frame 34. This is non-blinded one-prompt/one-seed review.
- **Interpretation:** retained recent non-sink AR state is a strong candidate
  contributor to old-entity carry-over; a retained sink is compatible with
  old-background persistence; the one-block no-old-context arm is the cleanest
  tested semantic cut. This does not establish a general policy or sink-only
  causality.
- **Code correction:** `transition_no_sink` makes old K/V inaccessible for the
  first B attention call, then DMD and clean writes replace physical slot zero
  with B state. The no-policy `kv_flush` path in this checkout keeps slot zero
  plus two latest frames, not the paper-level one-last-frame description.
- **Replication status:** the CPU-validated 4-pair × 2-seed × 4-arm matrix is
  now executed and recorded immediately below. No Scene-Epoch/Scene-Time method
  exists yet.

## 2026-08-10: Hard-cut AR-state Phase-1 replication (32 GPU runs)

### OBSERVED

- Ran the CPU-validated manifest unchanged: four A→B prompt pairs, seeds
  101/202, and `live_kv_flush` / `sink_plus1` / `sink_only` /
  `transition_no_sink` for **32/32 completed** runs. Every run exited zero,
  wrote its raw decoded tensor, block-3/4/5 latents, and MP4. The execution
  ledger records the exact prompt, seed, arm, command, first B block/frame
  (4/33), runtime, PID-scoped peak VRAM, and output path:
  `outputs/hard_cut_transition_phase1_20260810/runs.json`.
- Runtime was 42.308--48.726 s/run (1,446.497 s total); direct-process peak
  VRAM was 22,964 MiB for `live_kv_flush` and 23,176 MiB for every policy arm.
  These are execution metadata, not quality metrics.

### HUMAN VISUAL REVIEW

All review is non-blinded visual inspection of the synchronized four-arm
videos and temporal sheets. In every case, B's requested entity appears in all
arms, but only `transition_no_sink` reaches a clean B scene after the brief
first-B dissolve. The three retained-sink arms remain source/B composites.

| A→B cases (both seeds) | `live_kv_flush`, `sink_plus1`, `sink_only` | `transition_no_sink` | Artifact / review fields |
| --- | --- | --- | --- |
| greenhouse woman → desert pickup | Pickup is recognizable, but greenhouse structure persists through f44--f68; the woman is a cut-boundary ghost rather than a sustained later subject. | Desert/pickup is clean by f44 and remains so. | New-prompt adherence: mixed vs clean; old subject: transition-only; old background: retained vs absent; artifact: f33--f36 dissolve. |
| aquarium fish → snowy locomotive | Locomotive is recognizable but coral, water, and foreground fish remain visible later. | Snowy viaduct/locomotive is clean by f44. | New-prompt adherence: mixed vs clean; old entity/background: sustained fish/aquarium vs absent; artifact: first-B dissolve. |
| kitchen chef → storm sailboat | Boat is recognizable but counter, lights, and food remain around it later; chef is not a sustained later subject. | Ocean/boat is clean by f44. | New-prompt adherence: mixed vs clean; old subject: transition-only; old background: sustained vs absent; artifact: first-B dissolve. |
| moon observatory → autumn fox | Fox is recognizable but circular windows/telescope observatory structure remains around it later; astronaut is not sustained later. | Forest/fox is clean by f44. | New-prompt adherence: mixed vs clean; old subject: transition-only; old background: sustained vs absent; artifact: first-B dissolve. |

The eight synchronized videos, eight temporal sheets, and one all-case summary
are in `outputs/hard_cut_transition_phase1_20260810/comparison/`.

### INTERPRETATION

- **Recent-KV effect:** reducing old non-sink retention from the live
  sink+two path to sink+one or sink-only produced **0/8 unambiguous sustained
  source-entity reductions**. This matrix therefore does *not* replicate the
  earlier woman→car claim as a clean monotonic recent-KV/entity effect; most
  source subjects vanish after the boundary blend in all arms, while aquarium
  fish persist even with sink-only.
- **Superseded by user manual review / Phase 3B:** do not treat this earlier
  8/8 sink-specific assertion as current. Sink-only is clean for fish→train
  and astronaut→fox under the user review; the later factorial isolates recent
  state as another sufficient hard-cut contamination source.
- **Clean cut:** `transition_no_sink` is the cleanest hard semantic cut in
  **8/8 cases**, with no reviewed later-block counterexample where it harms B
  quality relative to the retained composite. All arms have a brief f33--f36
  dissolve; the reset arm intentionally sacrifices source-scene continuity and
  does not remove that one-block transition artifact.
- The original woman→car result is representative for source-scene retention
  and no-sink cleanup, but exceptional/insufficient as evidence for a general
  recent-KV-only entity-leakage separation. These are controlled visual
  observations, not automated semantic scores or a Scene-Epoch result.

**Smallest proposed Phase 2 (not implemented or run):** compare the existing
`transition_no_sink` arm against the same previous-scene reset plus an explicit
scene-local temporal/RoPE epoch, first on greenhouse→pickup and
aquarium→locomotive at seeds 101/202 (**2 pairs × 2 seeds × 2 arms = 8 runs**).
It needs preflight invariants proving that only post-cut temporal coordinates
change; no cache, routing, memory, or prompt-control mechanism should change.

## 2026-08-10: Phase 2A scene-local RoPE equivalence audit (CPU only)

- **OBSERVED (live trace):** `transition_no_sink` sets local cache end to zero
  at B block 4. During all four DMD calls and timestep-zero clean pass, RoPE
  Cut gives current B Q/K `[45,46,47]`; the clean pass writes B raw K/V then
  preserves only B frame 9's K already transformed at 45 in cache slot zero
  (V is raw). At B block 5 `scene_cut` is false: Q is `[12,13,14]`, while K is
  `[45,1,2,3,4,5]`; at B block 6 Q is `[15,16,17]`, K is
  `[45,1,2,3,4,5,6,7,8]`.
- **Deterministic probe:** identical synthetic float64 raw Q/K/V/cache tensors
  show first-B current `[45,46,47]` versus local `[0,1,2]` is numerically
  invariant (logit max 4.44e-16; attention-output max 2.22e-16). The clean
  sink K differs max/mean 1.45866/0.21141. At B block 5, current versus local
  `[3,4,5]` gives full-context logit/output max 2.36073/0.94993 and raw
  non-sink-only logit/output max 2.36073/1.02461; B block 6 full-context
  maxes are 2.95903/1.21477. Evidence:
  `outputs/hard_cut_transition_phase2a_20260810/scene_local_rope_probe.json`.
- **INTERPRETATION:** standard RoPE translation invariance applies only if the
  same phase offset reaches every attended Q and K. The proposed scene-local
  epoch is genuinely distinct from live no-sink because the clean pass leaves
  a special phase-45 transformed sink while later raw K is compactly re-RoPEd
  and later Q remains global. The raw-non-sink-only probe isolates a second
  non-invariance, independent of the special sink. This establishes neither
  visual benefit nor a contribution claim; it only keeps the 8-run comparison
  eligible. No GPU generation or Phase-2 implementation was performed.

## 2026-08-10: Phase 2B coherent scene-local RoPE epoch (8 GPU runs)

### OBSERVED

- Added one opt-in self-attention coordinate flag, restricted to an `#`
  hard-cut `transition_no_sink` transition. It gives B temporal phases
  `[0,1,2]`, then `[3,4,5]`, `[6,7,8]`, …; B raw non-sink K uses that same
  epoch and the clean pass stores its transformed B sink at phase 0. Global
  frame/output accounting and all non-self-attention behavior remain live.
  CPU checks cover first-B equivalence, phase-zero sink storage, later coherent
  Q/K, hard-cut-only activation, global accounting, and flag-off baseline
  preservation.
- Completed the preregistered greenhouse→pickup and aquarium→locomotive
  matrix at seeds 101/202: **8/8** completed, zero failures, 44.393--48.224
  s/run, 23,176 MiB direct-process peak VRAM. The run ledger is
  `outputs/hard_cut_scene_local_rope_epoch_phase2b_20260810/runs.json`.
- A RGB frames are exactly equal across arms. In every pair×seed, raw decoded
  divergence first occurs at frame 34 (the second RGB frame of B); frame 33
  remains equal. Later B max/mean RGB differences increase, so the experiment
  is numerically active rather than a no-op.

### HUMAN VISUAL REVIEW

- The first B dissolve is visually the same in both arms. Both variants reach
  the requested pickup/desert or locomotive/snow scene after it, without a
  reviewed sustained greenhouse/aquarium carry-over.
- In all four pair×seed comparisons, later B2/B3+ samples show no consistent
  improvement or degradation in B-prompt adherence, subject/entity quality,
  spatial coherence, temporal coherence, or obvious artifacts. Small raw
  differences manifest as non-systematic local appearance/motion variation,
  not a stable quality change.
- Evidence: four synchronized two-arm videos and sheets plus one all-case
  summary in `outputs/hard_cut_scene_local_rope_epoch_phase2b_20260810/comparison/`.

### INTERPRETATION

- This is a **neutral** result. The scene-local temporal epoch is distinct
  from current `transition_no_sink` and reaches later B generation, but it
  does not produce a useful visible gain in this controlled 4-case review.
  Do not build a Scene-Time contribution around this coordinate-only rule or
  infer that scene-local time is beneficial. No Phase 3 is authorized here.

## 2026-08-11: Phase 3A same-scene action boundary (8 GPU runs)

### OBSERVED

- Audited the live policy path before execution: `transition_no_sink` already
  applies at a normal `|` boundary. With no `#`, it sets `local_end_index=0`
  and excludes old self-attention state while leaving `scene_cut=false`; no
  RoPE Cut or scene-local epoch is activated. The four no-sink policy logs
  record exactly that state and B's `[0,1,2]` normal positions.
- Completed greenhouse woman turn→wave and desert pickup drive→stop at seeds
  101/202, live `kv_flush` versus normal-boundary `transition_no_sink`: **8/8
  completed**, 42.432--48.105 s/run. Peak direct-process VRAM is 22,964 MiB
  live and 23,176 MiB policy. Ledger:
  `outputs/same_scene_action_transition_phase3a_20260811/runs.json`.
- Both requested A scenes are present in each case before B, making continuity
  review valid. A is bit-identical across arms; raw decoded tensors first
  diverge at RGB frame 34 in all four cases (frame 33 exact). Divergence data:
  `outputs/same_scene_action_transition_phase3a_20260811/comparison/raw_divergence.json`.

### HUMAN VISUAL REVIEW

- Live retention preserves woman appearance and greenhouse continuity in both
  seeds, with a readable B wave. It also preserves pickup/desert appearance
  and scene composition in both seeds. The short B window makes the requested
  drive→stop action only qualitatively reviewable, but the vehicle is stable.
- Normal-boundary no-sink fails in **4/4**: immediately after the equal first
  B RGB frame, severe colored-noise/recomposition overtakes the woman and
  pickup scenes. It loses identity, background continuity, useful action
  evidence, motion continuity, and later-block stability. This is not a
  hard-cut style clean reset or a mild dissolve.
- Synchronized two-arm videos, sheets, and a four-case summary are under
  `outputs/same_scene_action_transition_phase3a_20260811/comparison/`.

### INTERPRETATION

- **Superseded by Phase 3B:** this initial normal no-sink comparison shows
  complete state removal is unsafe for same-scene continuity, but does not say
  that the sink itself is necessary. The later factorial finds sink-only and
  recent-only each usable in the tested window.

## 2026-08-11: Phase 3B sink × recent-local-state factorial (12 new GPU runs)

### OBSERVED

- **User manual-review supersession:** Phase-1’s earlier Codex review is not
  discarded, but its claim that retained sink causes stale scene semantics in
  8/8 is superseded. The user finds sink-only clean for fish→train and
  astronaut→fox, while no-sink remains the most consistently clean hard-cut
  arm. The defensible claim is retained previous-scene AR-state contamination,
  not universal sink causality.
- Added only `recent_only_no_sink`: it excludes the old transformed sink,
  retains two latest old raw local frames, uses compact live-compatible
  positions `[1,2,45,46,47]` at hard cuts or `[1,2,3,4,5]` normally, then
  writes the B clean-pass state as the normal new sink/cache. Policy logs
  verify these exact contexts, two retained frames, excluded sink, and matched
  cross-attention reset.
- Hard cuts: 12 exact-provenance controls reused and 4 fresh recent-only runs
  completed (44.524--48.594 s). Same-scene: 8 controls reused and 8 fresh
  sink-only/recent-only runs completed (44.453--45.889 s). Ledgers:
  `outputs/hard_cut_state_retention_factorial_phase3b_20260811/runs.json` and
  `outputs/same_scene_state_retention_factorial_phase3b_20260811/runs.json`.

### HUMAN VISUAL REVIEW

- **Hard cuts:** recent-only yields source/B composites in all four reviewed
  greenhouse→pickup and fish→train pair×seed cells, despite no old transformed
  sink. Sink-only is clean for fish→train but remains contaminated for
  greenhouse→pickup. Neither is the most consistently clean B scene.
- **Same-scene:** live, sink-only, and recent-only all retain usable woman /
  greenhouse and pickup / desert continuity in 4/4 cases. Neither repeats the
  severe colored-noise/recomposition collapse observed in Phase 3A’s no-sink
  arm. Action assessment remains qualitative in the short pickup window.
- Four-arm sheets/videos are under each Phase-3B `comparison/` directory.

### INTERPRETATION

- Sink is **not necessary** for hard-cut contamination; recent frames alone
  are sufficient in these cells. Sink alone is not universally sufficient for
  contamination. For same-scene continuity, either sink alone or recent state
  alone is sufficient here; their combination is not required. Complete state
  removal is the only consistently clean hard-cut intervention and the only
  catastrophic same-scene intervention.
- This is evidence for a boundary-conditioned AR-state lifetime effect, not an
  automatic policy, sink-only mechanism, novelty claim, or general causal law.
  Stop after Phase 3B.

## 2026-08-12: Phase 3B manual-review correction and Phase 3C mixed-boundary policy (12 GPU runs)

### OBSERVED

- **User review supersession:** partial-state conclusions above are superseded.
  On hard cuts, fish→train has old fish/aquarium rocks in live and sink-only;
  recent-only reduces old fish but is blurry/flashing; no-sink is the cleanest
  stable train. Girl→pickup is clearly greenhouse-free only with no-sink. On
  same-scene actions, live is best/stable; no-sink catastrophically
  rainbow/noise-recomposes; sink-only partially recomposes and recent-only
  flashes, with ghosted/malformed woman hands/appearance. Partial retention is
  not a usable operating point in this review.
- Added only `--boundary-conditioned-ar-state`: `|` invokes the existing live
  sink+recent2 `kv_flush`; `#` invokes existing `transition_no_sink` plus the
  existing RoPE Cut. Cross-attention reset remains live in both cases. No
  classifier, memory, routing, soft decay, RoPE method, or latent mechanism
  was added.
- Deterministic CPU tests prove normal `|` cache state equals live `kv_flush`
  exactly and hard `#` equals `transition_no_sink` with RoPE Cut. With the flag
  off, inference keeps the old live path.
- Completed the registered mixed A1 | A2 # B1 | B2 matrix: two scenarios
  (woman/greenhouse→pickup/desert; fox/autumn→locomotive/snow), seeds 101/202,
  live vs always-reset vs boundary-conditioned: **12/12 exit-zero**,
  48.556--56.905 s/run (615.081 s total). Peak direct-process VRAM is 22,964
  MiB live/conditioned and 23,176 MiB always-reset. Ledger:
  `outputs/mixed_boundary_state_lifetime_phase3c_20260812/runs.json`.
- Per-frame decoded tensors show always-reset first differs from live at RGB
  f34 in every case; boundary-conditioned first differs at f70 in every case.
  This matches the first normal and hard boundaries, respectively; the
  conditioned arm is bit-identical to live through f69.

### HUMAN VISUAL REVIEW

- At A1→A2 `|`, live and boundary-conditioned preserve the same stable
  woman/greenhouse and fox/forest transitions. Always-reset exhibits the known
  noisy recomposition instead.
- At A2→B1 `#`, live carries old visual content into a pickup/desert or
  locomotive/snow composite. Boundary-conditioned removes that carry-over and
  reaches the requested B scene after the common first-block dissolve.
- At B1→B2 `|`, boundary-conditioned keeps the B vehicle/locomotive and B
  scene stable, while always-reset again flashes/recomposes. Evidence: three
  arm synchronized videos, temporal sheets, and summary in
  `outputs/mixed_boundary_state_lifetime_phase3c_20260812/comparison/`.

### INTERPRETATION

- Full live state is the best tested same-scene continuity condition; complete
  state removal is the best tested hard semantic-cut condition. Partial
  retention is an unstable compromise, not a candidate policy.
- The integrated result is a **positive controlled demonstration** that an
  explicit `|`/`#` label can choose between two already-validated live paths.
  It is not an automatic classifier, a general semantic-boundary policy, a
  sink-causality proof, or a novelty claim. Stop here: no further component
  sweeps or additional mechanism is supported.

## 2026-08-12: Phase 3C personal visual-review supersession

### HUMAN VISUAL REVIEW — USER SUPERSESSION

- This append-only user review supersedes any stronger Codex-only Phase-3C
  qualitative wording. At normal `|` boundaries, live Infinity-RoPE is
  clean/stable; boundary-conditioned is clean/stable and should preserve live
  behavior; always-reset later collapses into rainbow/noise during A2/B2.
- At hard `#`, the new scene still takes roughly five RGB frames to establish.
  Phase 3C does **not** demonstrate an instantaneous cut, nor clear hard-cut
  visual superiority over live in every mixed case. The dedicated Phase-1/3B
  matrices remain the primary hard-cut-cleanup evidence.

### INTERPRETATION

- Phase 3C is integrated feasibility/compatibility evidence only: the explicit
  label can preserve live behavior at a continuity boundary while selecting the
  previously validated no-old-state path at a hard boundary. It is not an
  additional standalone hard-cut-quality claim.
- Mechanism is frozen at `e556855`. The next action is a strict primary-source
  novelty gate, recorded in
  `docs/BOUNDARY_CONDITIONED_AR_STATE_NOVELTY_GATE_20260812.md` and
  `docs/PHASE4_FULL_PAPER_NOVELTY_AUDIT_20260812.md`; no new method or GPU work
  is authorized.

## 2026-08-09: Subject-latent cache-write-mask ablation (one seed)

- This is the requested cache-state-only ablation. `persistent_cache_erode1`
  and `persistent_cache_erode2` retain the exact unaligned full subject patch
  in visible block 8, but write only the target erode1 or erode2 core into the
  block-8 timestep-zero clean-cache input. There is no raw KV, no later patch,
  and no alignment, blend, tracking, or changed denoising step.
- Both new RGB tensors are bit-identical to `persistent_full` throughout
  decoded block-8 frames 81--92 (max difference 0); the pre-clean block-8
  latent remains reset-equal for every arm. Policy logs independently verify
  that output latents outside the full target mask are exactly baseline
  (`outside_target_equal=true`, max abs 0). Thus the visible transplant is
  controlled; only the clean-cache write differs.
- Exact source-supported visible-mask counts are 386/386/387 30x52 tokens
  (1,544/1,544/1,548 latent cells) for target frames 21/22/23. Cache erode1
  writes 343/343/344 tokens (1,372/1,372/1,376 cells); erode2 writes
  276/276/276 (1,104 cells). The preflight overlays and audit retain the
  unchanged six-frame / 9,360-token base context.
- Visual review of blocks 9--10: erode1 remains close to full persistence,
  retaining the A1-like high bun/downward face and cobalt outfit; erode2 still
  carries a bun/outfit influence but is weaker and less stable. The broad snowy
  observatory remains recognizable in all persistent variants. Greenhouse-like
  local structure is attenuated, not removed: at decoded frame 113, outside-
  subject mean absolute RGB change versus reset is 0.03732 full, 0.02865
  erode1, and 0.02802 erode2. These are perturbation proxies, not identity or
  leakage metrics. The abrupt 8→9 transition remains.
- Conclusion: shrinking only the cache write gives a modest local-context
  reduction while retaining much of the qualitative subject persistence, but
  neither core establishes clean A1 identity persistence nor removes the local
  source-scene trace. Stop after this ablation.
- Evidence: [mask audit](../outputs/attention_memory_policy_fixed_grid_selective_recall/latent_subject_patch_persistent/cache_write_mask_ablation/preflight/mask_audit.json),
  [four-arm temporal sheet](../outputs/attention_memory_policy_fixed_grid_selective_recall/latent_subject_patch_persistent/cache_write_mask_ablation/comparison/four_arm_cache_write_temporal_sheet.png),
  [subject crops](../outputs/attention_memory_policy_fixed_grid_selective_recall/latent_subject_patch_persistent/cache_write_mask_ablation/comparison/four_arm_cache_write_subject_crops.png),
  and `four_arm_cache_write_metrics.json`. SHA-256 audit
  `c8de4ddb0fbbe029cbea3cb9e8daf97e565d86f31e59bfb313f81cafec1ad18b`,
  erode1 policy `df79e14978f5de9bed7ad78daf7fdba7166bd556c03c398c1735c2e667f5aa5d`,
  erode2 policy `a964aa012b0e9d7f7c2c63860dfd23918499500fbd3abcae56ba056159e15fc1`.

## Phase 5 checkpoint — mechanical completion only (2026-08-14)

- The frozen Native AR State Invalidation/Rebinding 63-cell checkpoint completed
  **63/63** exit-zero GPU runs: seven fixed categories, one six-segment
  `A1 | A2 # B1 | B2 # C1 | C2` storyboard each, seeds 101/202/303, and arms
  live, always-reset, and frozen policy. No inference behavior changed.
- Mechanical provenance: total generation runtime 4,209.862 s; per-run range
  61.638–77.878 s; direct-process peak VRAM 26,904 or 27,852 MiB; all raw and
  MP4 SHA-256 hashes and commands are in
  `outputs/phase5_generalization_checkpoint_20260813/runs.json`.
- Actual RGB boundary frames are 46/94/142/190/238 (285 frames), not the
  pre-run naïve 49-series; this arises from the existing independent-first-frame
  rollout. The corrected schedule is metadata-only and no output was rerun.
- No semantic conclusion is recorded: a private mapping, 21 anonymized videos,
  105 strips, and a blank 315-row 1–5/uncertain review form were created. Human
  review is pending; RGB divergence is provenance, not a quality metric.

## Phase 5 checkpoint — decoded user review (2026-08-14)

- With user authorization, the anonymous labels were mapped through the private
  arm file. The explicit 60 ordinal rows give continuity means live 2.75,
  always-reset 1.00, and rebinding 3.67; hard-cut means live 1.25,
  always-reset 3.00, and rebinding 3.88. This is a small scored subset, not the
  full checkpoint endpoint.
- User notes repeat the expected failures—always-reset rainbow/noise at `|` and
  live old-scene retention at hard cuts—but also identify visually similar-cut
  weakness, slow/weird rebinding motion, and seeds where live hard cuts are
  usable. Treat these as category-dependent counterexamples.
- Conclusion: **CHECKPOINT MIXED — REVIEW BEFORE MORE GPU.** The frozen policy
  remains worth a preregistered full evaluation only after the scoring protocol
  is completed/normalized; no mechanism tuning or additional GPU run is
  authorized by this record.

## Fresh-scene-prime offset control — CPU feasibility (2026-08-14)

### OBSERVED (code audit and CPU tests only; no GPU result)

- A normal hidden B block necessarily advances the native cache cursor by one
  three-latent-frame block. Reusing the visible B cursor would overwrite the
  hidden cache rather than leave it as causal history; `cache_start` is not
  used by the self-attention cache path.
- The existing `transition_no_sink` operation can be applied *after* that
  hidden normal-path block with `cross_attention_reset=False`. It sets the
  readable self-attention `local_end_index` to zero and clears `scene_cut`,
  while retaining the advanced `global_end_index` and the B cross-attention
  cache. Thus it supplies the required offset-only control without a custom
  cache writer or a new RoPE rule.
- The CUDA default-generator state is captured before the hidden block and
  restored before visible B. Visible B retains its original preallocated noise
  slice; offset-only and fresh-prime consequently share visible initial/DMD
  noise and the same advanced native cursor. The hidden block is never written
  to output, so visible output indexing and frame count remain unchanged.
- The registered CPU-only matrix is two known motion-failure cases
  (aquarium→locomotive and kitchen→sailboat), seeds 101/202, and frozen
  rebinding / offset-only / fresh-prime: 12 runs. See
  `docs/FRESH_SCENE_PRIME_OFFSET_CONTROL_20260814.json`.

### INTERPRETATION

- Offset-only versus fresh-prime isolates readable fresh-B native
  self-attention state from the unavoidable ordinary timeline advance. It does
  not separate the prime from B cross-attention cache, which is deliberately
  matched between those two arms.
- No GPU has been launched. The pending review asks only whether fresh native
  state improves motion without returning A semantics or creating new latency
  artifacts; it is not a claim of a new mechanism or novelty.

## Fresh-scene-prime offset control — 12-run result (2026-08-14)

### OBSERVED

- The matched aquarium→locomotive and kitchen→sailboat matrix completed all
  12 requested output cells (seeds 101/202; frozen rebinding, offset-only,
  fresh-prime). Every cell has a raw decoded tensor, MP4, policy log, and
  SHA-256 record. The serial runner failed only in its final reference-arm
  annotation because it did not recognize `frozen_rebinding`; the recovered
  ledger explicitly marks return codes, exact runtime, and per-run peak VRAM
  as unavailable rather than fabricating them.
- Output frame count is 69 in every arm/case, and RGB frames 1--33 are
  bit-identical across all three arms. Both offset arms first differ from
  frozen rebinding at RGB frame 34. Their policy logs record the same hidden
  state start (9), visible state start (12), and restored CUDA generator;
  offset-only alone records `prime_native_state_inaccessible=true` with no
  cross-attention reset.
- In all four reviewed temporal sheets, offset-only collapses into persistent
  multicolored noise immediately after the hard boundary. Fresh-prime produces
  recognizable later locomotives/sailboats without an obvious sampled
  fish/aquarium or chef/kitchen reconstruction. Frozen rebinding also reaches
  recognizable B scenes in these sparse samples.

### INTERPRETATION

- The offset-only control rules out interpreting any stable fresh-prime output
  as a harmless consequence of advancing the native timeline alone: the same
  cursor with inaccessible hidden native K/V is catastrophically unstable.
- The current synchronized sheets do **not** establish a clear improvement in
  train smoke/motion or boat dynamics over frozen rebinding. Direct-video human
  review is still required for that narrow endpoint; no claim of dynamics
  improvement, novelty, or a new method is supported.
- Per the preregistered stop condition, stop this prime branch here. Do not
  launch additional prime runs or a novelty audit automatically.

## Early-native-handoff — feasibility stop (2026-08-14)

### OBSERVED (code audit and deterministic CPU probe only; no GPU result)

- The live four-call DMD execution order is `1000 -> 937.5 -> 833.3333 ->
  625` (high noise to low noise). Every call for one visible latent block uses
  the same `current_start`.
- After the first B DMD call, each native self-attention cache has advanced
  `global_end_index` and `local_end_index` by the three B latent frames. The
  existing `transition_no_sink` operation intentionally clears only
  `local_end_index`; it retains `global_end_index`.
- A deterministic CPU probe applied that existing operation between DMD calls
  while preserving the same B `current_start`. The next native attention call
  computed a zero-length destination slice and failed with `RuntimeError` when
  writing the three current B K/V tokens. This follows directly from
  `local_end + current_end - global_end == 0` after the first call.
- No existing operation can make A K/V inaccessible at that point while
  retaining a valid same-block insertion cursor: resetting the global cursor
  would be a new cursor-rewrite rule, while retaining/reinserting B K/V would
  require a custom cache copy/writer. Both violate the preregistered
  accessibility-only constraint. RoPE Cut, cross-attention behavior, output
  indexing, and RNG were therefore not changed.

### INTERPRETATION

- Early native handoff is infeasible as a clean isolated intervention in the
  current backbone. Native-state accessibility and same-block cache cursor
  validity cannot be separated between existing DMD calls using only the live
  reset operations.
- Stop this branch before implementation and GPU generation. This is a
  feasibility negative, not evidence about whether early old-state access
  would improve motion.

## Standalone-B dynamics control — eight-run result (2026-08-15)

### OBSERVED

- The requested B1-only control completed for the exact Phase-5 bright-blue
  freight-train and white-fishing-boat B1 prompts, seeds 101/202, with no
  preceding scene, boundary, reset, or experimental inference flag. Each fresh
  3.0-second rollout has 12 latent frames and 45 decoded RGB frames.
- The B1-only prompts do not request the reported steam-rise or boat
  direction-change behavior. Per the preregistered conditional instruction,
  four exact-text B2-alone controls were therefore also run, using the original
  B2 wording (including `same`) without merging prompts. All eight standalone
  cells have raw tensors, MP4s, four clean-latent snapshots, hashes, and GPU
  telemetry; see
  `outputs/standalone_b_dynamics_control_20260814/runs.json`.
- Each standalone output is compared with the first 45 RGB frames of the
  matching Phase-5 `native_state_rebinding` segment: B1 frames 94--138 and B2
  frames 142--186. The fresh one-scene decoder has the normal initial
  three-RGB-frame offset, so this equal-length crop preserves the requested
  3-second latent duration without inventing a warm-up block.
- The synchronized sheets show the known B1 transition dissolve in the rebinding
  reference at its first sampled frame, while the standalone output begins in
  B directly. Beyond that transition frame, both arms visibly contain
  recognizable train/boat scenes. The static sheets do not show a clear,
  uniform standalone advantage in train, smoke, boat, or wave dynamics across
  the two seeds.

### INTERPRETATION

- This is **mixed/inconclusive**, not evidence that Native-State Rebinding is
  the cause of the reported dynamics abnormality. At minimum, the prompt/model
  can produce nontrivial train, smoke, boat, and wave dynamics from fresh B-only
  rollouts too; the current control does not isolate a uniform transition-only
  degradation.
- The remaining comparison endpoint is visual motion inspection of the eight
  synchronized videos, not a numerical quality score. Do not develop a motion
  mechanism or launch a follow-up automatically. If direct review finds a
  decisive per-case difference, report that category/seed dependence before
  proposing any isolated cause test.

## Motion-diagnosis branch — closed (2026-08-15)

### OBSERVED

- Fresh-scene prime did not show a clear motion improvement over frozen
  rebinding; early-native handoff was infeasible without a prohibited cache
  cursor rewrite or K/V repacking; and standalone exact-B controls did not show
  a clear, uniform train/boat/smoke motion advantage over rebinding.

### INTERPRETATION

- Current evidence does **not** attribute the observed train/boat/smoke motion
  weirdness specifically to Native AR State Rebinding. Treat it as
  backbone/prompt/seed-dependent under the tested controls. The motion branch
  is closed: do not add another motion mechanism.
