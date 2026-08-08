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
