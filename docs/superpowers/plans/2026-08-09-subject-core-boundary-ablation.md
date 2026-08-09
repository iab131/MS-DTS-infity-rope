# Subject-core / boundary oracle implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether A1-background leakage is concentrated in the one-token subject boundary or persists in eroded subject-core historical K/V.

**Architecture:** Keep the committed 30x52 manual source/target masks immutable. Derive 8-connected one- and two-token erosions plus the full-minus-erode1 ring in the existing mask helper, then route those sparse token indices through the existing selective historical branch. Existing source H/W indices, temporal slots 1/2, and normal local/current attention remain unchanged.

**Tech Stack:** Existing Python, PyTorch/FlashAttention path, OpenCV preflight, unittest, existing `wan` environment.

## Global Constraints

- Exact prompt/seed 101, history IDs 6/7, target block 8, all 30 layers, `transition_no_sink`, and `replace_recent`.
- Run only `subject_erode1`, `subject_erode2`, and `subject_boundary_only`; reuse reset-only and subject-full controls.
- No automatic masks, tracking, SAM, alpha blending, layer sweeps, or policy changes. Preserve existing dirty worktree files.

---

### Task 1: Deterministic derived subject masks

**Files:**
- Modify: `pipeline/fixed_grid_memory_masks.py`
- Modify: `tests/test_fixed_grid_memory_masks.py`

**Interfaces:**
- Produces `history_token_indices_for_mode(mode, frame_id)` and `target_query_indices_for_mode(mode)` for `subject_to_subject`, `subject_erode1`, `subject_erode2`, and `subject_boundary_only`.

- [ ] **Step 1: Write failing mask-count and containment tests.**

```python
for mode in ("subject_erode1", "subject_erode2", "subject_boundary_only"):
    self.assertTrue(masks.target_query_indices_for_mode(mode))
self.assertEqual(set(full), set(core1) | set(ring))
self.assertFalse(set(core1) & set(ring))
self.assertTrue(set(core2) < set(core1) < set(full))
```

- [ ] **Step 2: Run the focused test and verify it fails because the helper is absent.**
- [ ] **Step 3: Implement 8-connected binary erosion and the four explicit modes.**
- [ ] **Step 4: Re-run focused tests and commit the helper/test change.**

### Task 2: Route and audit the derived modes

**Files:**
- Modify: `inference.py`
- Modify: `pipeline/causal_inference.py`
- Modify: `scripts/prepare_fixed_grid_memory_oracle.py`
- Modify: `tests/test_fixed_grid_memory_masks.py`
- Modify: `tests/test_prepare_fixed_grid_memory_oracle.py`

**Interfaces:**
- Consumes the Task 1 mode-index methods.
- Produces exact mode-specific source/query counts, original coordinates, slots, and full/core/ring source and target overlays in a CPU-only audit.

- [ ] **Step 1: Write failing CLI/mode-routing and audit tests.**
- [ ] **Step 2: Run focused tests and verify expected failure.**
- [ ] **Step 3: Permit only the three new explicit modes and use the shared mode-index methods for packing/logging; extend the existing overlay script.**
- [ ] **Step 4: Re-run focused tests, inspect generated overlays/audit, and commit.**

### Task 3: Matched inference and bounded report

**Files:**
- Modify: `docs/ATTENTION_MEMORY_POLICY_FIXED_GRID_SELECTIVE_RECALL_20260809.md`
- Modify: `docs/NONCONTIGUOUS_PHASE1_PAPER_LEDGER.md`
- Modify: `docs/INFERENCE_FINDINGS.md`
- Create: output artifacts under `outputs/attention_memory_policy_fixed_grid_selective_recall/subject_core_boundary_ablation/`

- [ ] **Step 1: Run CPU preflight, verify overlays/counts/coordinates/slots/base order, and run the focused unit suite.**
- [ ] **Step 2: Check idle GPU/disk and run exactly the three new arms with telemetry.**
- [ ] **Step 3: Compare reset/full/erode1/erode2/ring visually and with clearly labelled pixel discontinuity proxies.**
- [ ] **Step 4: Append evidence and limitations to the ledger/findings, run final checks, commit, and push.**
