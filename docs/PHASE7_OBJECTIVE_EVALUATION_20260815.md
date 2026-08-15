# Phase 7 — Objective Failure-Mode Evaluation (2026-08-15)

## Scope and provenance

This is a read-only analysis of the completed Phase-5 checkpoint: seven fixed storyboards, three seeds (101/202/303), three arms, and 63 existing raw decoded videos. It changes neither inference nor output video. Runner provenance is `eb85a7df609165ac30d8593bb78130144984d27e`; frozen mechanism provenance remains `e556855`.

Raw records, per-frame curves, validation, and the plotting artifact are under `outputs/phase7_objective_evaluation_20260815/`. Analysis code: `scripts/phase7_objective_evaluation.py`; focused tests: `tests/test_phase7_objective_evaluation.py`.

## Evaluation-tool audit

| Need | Available without environment change | Decision |
| --- | --- | --- |
| Image/image representation | Locally cached `dino_vitbase16_pretrain.pth`; `timm` ViT-B/16 DINO strict-loads with zero missing/unexpected keys | Use DINO cosine similarity as a **visual-reference retention proxy**. |
| Image/text representation | `clip`, `open_clip`, and `transformers` import, but no local CLIP/OpenCLIP weight is present | Do not download weights or report automatic prompt adherence. |
| Low-level temporal/chromatic analysis | NumPy, Torch, OpenCV, SciPy, scikit-image, imageio, and decord are installed | Use RGB step, colorfulness, and high-frequency change for the known rainbow/noise failure. |
| Optical flow/tracking | OpenCV is available | Not used: it measures motion, not old-scene retention or rainbow collapse, and the motion branch is closed. |

This is not a generic video-quality evaluation. DINO similarity does not prove textual semantics, identity, or causal cache content.

## Locked metrics

At each `#`, the eight preceding RGB frames define a source visual-context reference. DINO cosine similarity is measured across the next 48 frames; its mean is **source-reference AUC**. Lower means less visual similarity to the immediate source context. It is not called semantic leakage without validation.

At each `|`, a fixed eight-frame pre-window and 15-frame post-window measure local DINO feature distance, RGB frame step, colorfulness, and high-frequency energy. Each is normalized only by the preceding local segment; the mean `log(1 + ratio)` is the collapse signal. It targets abrupt chromatic/temporal recomposition, not normal action motion or aesthetic quality.

A late-B1 target-stability latency proxy was also attempted. It is non-discriminative: 31.24 frames for live and 31.07 for rebinding (median 32 and 31). It is not used as evidence of transition latency or prompt adherence.

For a diagnostic high-discontinuity count, the threshold was fixed once as the 95th percentile (0.9622) of all live and rebinding `|` values. It is an outlier flag, not a semantic or human-quality label.

## Validation against pre-existing user review

No score or threshold was changed to fit the review. The 60 explicitly scored copied-form rows were unblinded only with the already-existing private map. The user’s ordinal score is broad, rather than a single leakage/collapse label, so this is directional validation only.

| Proxy vs. existing user score | Rows | Spearman rho | Expected direction | Result |
| --- | ---: | ---: | --- | --- |
| Hard-cut source-reference AUC | 24 | -0.580 | cleaner/higher-rated rows retain less source visual context | matches |
| Continuity collapse signal | 36 | -0.701 | better/higher-rated rows show less abnormal recomposition | matches |

This clears the minimum for **descriptive** checkpoint use. It does not validate DINO as a semantic-leakage detector outside these videos, replace user review, or make the 63-run checkpoint a full paper benchmark.

## Results

### Hard-cut visual-source retention

| Arm | Boundaries | Source-reference AUC, mean (median) |
| --- | ---: | ---: |
| Live Infinity-RoPE | 42 | 0.523 (0.431) |
| Always reset | 42 | 0.078 (0.077) |
| Native-state rebinding | 42 | **0.159 (0.157)** |

Always-reset’s hard-cut number is descriptive only: an earlier `|` reset can already produce rainbow/noise before a later `#`, so a low source match does not establish a clean hard-cut solution.

The paired primary comparison, rebinding minus live, is -0.364 AUC (paired bootstrap 95% CI [-0.449, -0.281], n=42). Rebinding is lower in 40/42 matched hard boundaries. The two reversals are both the *second* cut of the visually-similar semantic-change storyboard (seeds 202 and 303): live 0.232/0.286 versus rebinding 0.251/0.305.

| Category | Live AUC | Rebinding AUC |
| --- | ---: | ---: |
| human → object | 0.682 | 0.119 |
| animal → vehicle | 0.523 | 0.142 |
| indoor → outdoor | 0.519 | 0.123 |
| visually similar semantic change | 0.429 | 0.310 |
| visually dissimilar semantic change | 0.325 | 0.168 |
| same-subject action storyboard | 0.272 | 0.081 |
| same-object motion storyboard | 0.908 | 0.171 |

The effect appears in each seed: live/rebinding means are 0.532/0.154 (101), 0.488/0.165 (202), and 0.548/0.159 (303). The per-frame curve drops sharply for rebinding after the cut and remains low; live remains near 0.51 from RGB frame 4 through frame 47. This is evidence of less **visual source-context retention**, consistent with reviewed stale-scene examples—not a direct measure of prompt adherence or a claim that every live cut fails.

### Normal-boundary collapse

| Arm | `|` boundaries | Collapse signal, mean (median) | High-discontinuity count |
| --- | ---: | ---: | ---: |
| Live Infinity-RoPE | 63 | 0.778 (0.782) | 4/63 |
| Always reset | 63 | **2.340 (2.386)** | **63/63** |
| Native-state rebinding | 63 | 0.743 (0.733) | 3/63 |

Rebinding minus live is -0.035 (paired bootstrap 95% CI [-0.062, -0.007], n=63); this small proxy difference is not a visual-quality gain. At the first `|` of all 21 storyboard×seed cases, rebinding’s raw decoded frames and this metric are exactly equal to live because the frozen policy preserves the live path until the first `#`. Later `|` windows are not expected to be bit-equal: the prior hard boundary has legitimately selected a different scene state.

All categories show the same large reset separation. Live/rebinding category means range 0.671–0.882 / 0.678–0.819, whereas always-reset ranges 2.000–2.525. Rebinding minus always-reset is -1.598 (paired bootstrap 95% CI [-1.676, -1.522], n=63).

## OBSERVED

- DINO supplies a local, strictly loaded visual-reference feature. No cached image/text evaluator exists, so automatic B-prompt adherence is intentionally absent.
- The locked source-reference and collapse proxies track the pre-existing scored review subset in the expected direction.
- Rebinding reduces post-`#` source visual-context similarity versus live in 40/42 paired boundaries, with two visually-similar-cut reversals.
- Always reset is a chromatic/temporal discontinuity outlier at every normal boundary under the fixed diagnostic threshold. Rebinding’s first normal boundary is mechanically identical to live.
- The automatic late-B1 target-stability latency proxy is non-discriminative.

## INTERPRETATION

The objective evidence strengthens a narrow causal story: carrying the live native rollout through a hard boundary is associated with persistent visual source-context similarity, while unconditional removal is associated with the known reset-collapse signature at continuity boundaries. It does **not** prove that DINO is old-scene semantics, generic quality improvement, universal hard-cut success, or instantaneous establishment. The visually-similar counterexamples and non-discriminative latency proxy require a narrow paper claim.

## Decision

`PAPER SIGNAL MIXED — CLAIM MUST NARROW`

If more work is approved, it should be a broader fixed evaluation with blinded review and independently specified semantic measurement—not a new mechanism or a post-hoc metric sweep.
