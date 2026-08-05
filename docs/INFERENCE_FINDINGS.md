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
