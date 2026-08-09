# Fixed-Grid Selective Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether A-frame historical K/V can be spatially gated to the returning woman or to background without changing Infinity-RoPE's normal current/local attention.

**Architecture:** Store fixed 30x52 source and target masks as reproducible JSON, slice only historical A K/V by original token indices, and split target queries into subject/background FlashAttention calls before scattering outputs back to their original positions. The normal six-frame `replace_recent` context, sink handling, current RoPE, and automatic memory controls remain unchanged.

**Tech Stack:** Python, PyTorch, existing FlashAttention wrapper, unittest, Pillow already used by the environment.

## Global Constraints

- This is an oracle separability experiment, not a masking method.
- Source masks are distinct fixed masks for A IDs 6 and 7; the target is one conservative block-8 union across its three latent frames.
- Historical sparse tokens retain original `(frame, h, w)` spatial RoPE coordinates and use only temporary history temporal slots 1 and 2.
- Subject/background masking applies only to historical-memory attention; normal local/current attention is unchanged.
- Background excludes a one-token dilated subject boundary.
- Run only subject-to-subject and background-to-background; reuse reset-only and full-A controls.

---

### Task 1: Define and verify fixed-grid mask mechanics

**Files:**
- Create: `pipeline/fixed_grid_memory_masks.py`
- Create: `tests/test_fixed_grid_memory_masks.py`

**Interfaces:**
- Produces `FixedGridMemoryMasks.from_json(path)`, `subject_query_indices()`, `background_query_indices()`, and `history_token_indices(frame_id)`.

- [ ] Write failing tests for 30x52 validation, one-token dilation exclusion, and per-frame flattened indices.
- [ ] Run the focused test and confirm it fails because the module is absent.
- [ ] Implement the minimal JSON parser/index helpers.
- [ ] Run the focused test and confirm it passes.

### Task 2: Add grouped historical-memory attention

**Files:**
- Modify: `wan/modules/causal_model.py`
- Modify: `tests/test_fixed_grid_memory_masks.py`

**Interfaces:**
- Consumes optional per-layer `selective_memory` metadata with history token indices and target query indices.
- Produces an attention tensor in the original query order.

- [ ] Write a failing CPU test proving subject/background historical calls are isolated and scattered into their original query positions.
- [ ] Run the focused test and confirm it fails because selective attention is absent.
- [ ] Implement one grouped-attention helper; retain full normal attention for all query tokens and add only the selected historical result to matching query outputs.
- [ ] Run the focused test and confirm it passes.

### Task 3: Wire the manual oracle and evidence artifacts

**Files:**
- Modify: `inference.py`
- Modify: `pipeline/causal_inference.py`
- Create: `scripts/prepare_fixed_grid_memory_oracle.py`
- Create: `docs/ATTENTION_MEMORY_POLICY_FIXED_GRID_SELECTIVE_RECALL_20260809.md`
- Modify: `docs/NONCONTIGUOUS_PHASE1_PAPER_LEDGER.md`
- Modify: `docs/INFERENCE_FINDINGS.md`

**Interfaces:**
- Consumes `--memory-fixed-grid-mask-path` and `--memory-fixed-grid-mode {subject_to_subject,background_to_background}`.
- Produces validated mask overlays, a JSON audit with exact source/query counts and coordinates, and policy-log context fields.

- [ ] Write failing CLI/config tests for both required selective flags.
- [ ] Run the focused test and confirm it fails because flags are absent.
- [ ] Wire the immutable grid-mask metadata only when manual memory is active at the requested block; log masks/counts/temporal positions.
- [ ] Generate source/A2 overlay evidence and audit it before GPU inference.
- [ ] Run focused and existing memory-policy tests, then perform the two matched GPU arms only after positional checks pass.

### Task 4: Record and publish the bounded result

**Files:**
- Modify: `docs/ATTENTION_MEMORY_POLICY_FIXED_GRID_SELECTIVE_RECALL_20260809.md`
- Modify: `docs/NONCONTIGUOUS_PHASE1_PAPER_LEDGER.md`
- Modify: `docs/INFERENCE_FINDINGS.md`

- [ ] Create the four-arm comparison from two reused controls and the two new outputs.
- [ ] Record commands, checks, exact hashes/counts, run outcomes, visual review, and limits.
- [ ] Run final focused verification and `git diff --check`.
- [ ] Commit only oracle changes and push to `origin/main`.
