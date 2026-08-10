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
  `[45,46,47]` in all cases. The old transformed sink was excluded, never
  re-rotated or mutated, in the experimental arm; ordinary cache writes resume
  after its one transition block.
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
