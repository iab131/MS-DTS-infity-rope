#!/usr/bin/env python3
"""Objective, failure-mode-specific measurements for completed Phase-5 videos.

This intentionally measures visual-reference retention and abrupt chromatic/
temporal disruption.  It does not claim prompt adherence, identity, or generic
video quality.
"""

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
HARD_BOUNDARIES = (94, 190)  # one-indexed RGB frames in completed Phase-5 output
CONTINUITY_BOUNDARIES = (46, 142, 238)
ARMS = ("live_infinity_rope", "always_reset", "native_state_rebinding")


def _unit(features):
    features = np.asarray(features, dtype=np.float64)
    return features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)


def hard_cut_metrics(features, cut_frame, source_count=8, stable_start=30,
                     stable_stop=48, required_stable_frames=4):
    """Reference similarities and a fixed, self-referenced establishment time.

    ``cut_frame`` is a zero-indexed first-new-scene frame.  The target reference
    is late B1 only; it never uses a later prompt segment.
    """
    features = _unit(features)
    source = features[cut_frame - source_count:cut_frame].mean(axis=0, keepdims=True)
    source = _unit(source)[0]
    post = features[cut_frame:cut_frame + stable_stop]
    target_frames = post[stable_start:stable_stop]
    target = _unit(target_frames.mean(axis=0, keepdims=True))[0]
    source_similarity = post @ source
    target_similarity = post @ target
    stable_self_similarity = target_frames @ target
    threshold = float(np.quantile(stable_self_similarity, 0.10))
    latency = None
    for offset in range(0, len(target_similarity) - required_stable_frames + 1):
        if np.all(target_similarity[offset:offset + required_stable_frames] >= threshold):
            latency = offset
            break
    return {
        "source_similarity": source_similarity,
        "target_similarity": target_similarity,
        "source_similarity_auc": float(source_similarity.mean()),
        "target_similarity_auc": float(target_similarity.mean()),
        "target_stability_threshold": threshold,
        "transition_latency_frames": latency,
    }


def _colorfulness(frames):
    rg = frames[..., 0] - frames[..., 1]
    yb = 0.5 * (frames[..., 0] + frames[..., 1]) - frames[..., 2]
    return np.sqrt(rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2) + 0.3 * np.sqrt(
        rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2)


def _high_frequency(frames):
    gray = frames.mean(axis=-1)
    return (np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2)) +
            np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2)))


def _safe_ratio(post, pre):
    return float(post / max(float(np.median(pre)), 0.01))


def continuity_metrics(features, frames, boundary_frame, pre_count=8, post_count=15):
    """Fixed local abnormal-discontinuity signal for visible reset collapse.

    The score combines feature, RGB-step, colorfulness, and high-frequency
    changes normalized only by the immediately preceding segment.
    """
    features = _unit(features)
    first = boundary_frame - pre_count
    last = min(len(frames), boundary_frame + post_count)
    local_frames = np.asarray(frames[first:last], dtype=np.float32)
    local_features = features[first:last]
    split = boundary_frame - first
    rgb_steps = np.abs(np.diff(local_frames, axis=0)).mean(axis=(1, 2, 3))
    feature_steps = 1.0 - np.sum(local_features[1:] * local_features[:-1], axis=1)
    color = _colorfulness(local_frames)
    high_frequency = _high_frequency(local_frames)
    pre_slice = slice(0, max(split - 1, 1))
    post_slice = slice(max(split - 1, 0), None)
    rgb_ratio = _safe_ratio(np.max(rgb_steps[post_slice]), rgb_steps[pre_slice])
    feature_ratio = _safe_ratio(np.max(feature_steps[post_slice]), feature_steps[pre_slice])
    color_ratio = _safe_ratio(np.max(color[split:]), color[:split])
    high_frequency_ratio = _safe_ratio(np.max(high_frequency[split:]), high_frequency[:split])
    return {
        "rgb_step_ratio": rgb_ratio,
        "feature_step_ratio": feature_ratio,
        "colorfulness_ratio": color_ratio,
        "high_frequency_ratio": high_frequency_ratio,
        "pre_colorfulness": float(np.median(color[:split])),
        "post_colorfulness": float(np.max(color[split:])),
        "collapse_signal": float(np.mean(np.log1p([
            rgb_ratio, feature_ratio, color_ratio, high_frequency_ratio]))),
    }


def _needed_indices():
    indices = set()
    for frame in HARD_BOUNDARIES:
        start = frame - 1
        indices.update(range(start - 8, start + 48))
    for frame in CONTINUITY_BOUNDARIES:
        start = frame - 1
        indices.update(range(start - 8, start + 15))
    return sorted(indices)


def _load_dino():
    import timm
    import torch

    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"missing required local DINO checkpoint: {checkpoint}")
    model = timm.create_model("vit_base_patch16_224.dino", pretrained=False, num_classes=0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DINO checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval(), checkpoint


def _dino_features(model, video, indices, batch_size=24):
    import torch
    import torch.nn.functional as F

    selected = video[0, indices].float()
    features = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            frames = F.interpolate(selected[start:start + batch_size], size=(224, 224), mode="bicubic", align_corners=False)
            features.append(model((frames - mean) / std).cpu().numpy())
    return np.concatenate(features), selected.permute(0, 2, 3, 1).numpy()


def _frame_maps(indices, features, frames):
    feature_map = {index: features[position] for position, index in enumerate(indices)}
    frame_map = {index: frames[position] for position, index in enumerate(indices)}
    return feature_map, frame_map


def _stack_from_map(frame_map, start, stop):
    return np.stack([frame_map[index] for index in range(start, stop)])


def _features_from_map(feature_map, start, stop):
    return np.stack([feature_map[index] for index in range(start, stop)])


def _tool_audit():
    packages = ("torch", "torchvision", "transformers", "open_clip", "clip", "timm", "cv2", "scipy", "skimage", "decord")
    return {
        "packages": {name: bool(importlib.util.find_spec(name)) for name in packages},
        "dino_checkpoint": str(Path.home() / ".cache/torch/hub/checkpoints/dino_vitbase16_pretrain.pth"),
        "clip_checkpoint_found": any((Path.home() / ".cache/clip").glob("*.pt")),
    }


def _read_csv_rows(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        headers = [header.strip() for header in next(reader)]
        return [dict(zip(headers, (value.strip() for value in row))) for row in reader]


def _review_rows():
    root = ROOT / "outputs/phase5_generalization_checkpoint_20260813/blinded_review"
    mapping = json.loads((root / "private/arm_mapping.json").read_text())
    rows = _read_csv_rows(root / "blinded_review_form copy.csv")
    result = []
    for row in rows:
        score = row.get("score_1_to_5", "").strip()
        if not score:
            continue
        case_id = row["case_id"].strip()
        result.append({
            "pair_id": case_id.split("__seed")[0],
            "seed": int(case_id.rsplit("seed", 1)[1]),
            "boundary_index": int(row["boundary_index"]),
            "boundary_type": row["boundary_type"].strip(),
            "arm_id": mapping[case_id][row["anonymous_arm"].strip()],
            "score": int(score),
            "notes": row.get("notes", "").strip(),
        })
    return result


def _bootstrap_mean_difference(values, paired_keys, draws=10000, seed=20260815):
    """Paired bootstrap CI for a mapping key -> (treatment, control)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray([values[key][0] - values[key][1] for key in paired_keys], dtype=float)
    if not len(diffs):
        return None
    samples = rng.choice(diffs, size=(draws, len(diffs)), replace=True).mean(axis=1)
    return {"mean_difference": float(diffs.mean()), "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], "n": int(len(diffs))}


def run(output_dir):
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = _tool_audit()
    model, checkpoint = _load_dino()
    audit["selected_metric"] = "DINO ViT-B/16 image-image cosine similarity"
    audit["selected_checkpoint"] = str(checkpoint)
    indices = _needed_indices()
    runs = json.loads((ROOT / "outputs/phase5_generalization_checkpoint_20260813/runs.json").read_text())["runs"]
    records, curves = [], []
    for number, run_record in enumerate(runs, 1):
        raw_path = ROOT / run_record["output_folder"] / "0_raw_decoded_before_mp4.pt"
        video = torch.load(raw_path, map_location="cpu", weights_only=True)
        features, frames = _dino_features(model, video, indices)
        feature_map, frame_map = _frame_maps(indices, features, frames)
        del video
        base = {key: run_record[key] for key in ("run_id", "pair_id", "seed", "arm_id")}
        for boundary_index, frame in enumerate((46, 94, 142, 190, 238), 1):
            cut = frame - 1
            if frame in HARD_BOUNDARIES:
                metric = hard_cut_metrics(_features_from_map(feature_map, cut - 8, cut + 48), 8)
                record = {**base, "boundary_index": boundary_index, "boundary_type": "#", "frame": frame,
                          **{k: v for k, v in metric.items() if not isinstance(v, np.ndarray)}}
                records.append(record)
                for offset, (source, target) in enumerate(zip(metric["source_similarity"], metric["target_similarity"])):
                    curves.append({**base, "boundary_index": boundary_index, "frame_after_cut": offset,
                                   "source_reference_similarity": float(source), "late_b1_reference_similarity": float(target)})
            else:
                metric = continuity_metrics(
                    _features_from_map(feature_map, cut - 8, cut + 15),
                    _stack_from_map(frame_map, cut - 8, cut + 15), 8,
                )
                records.append({**base, "boundary_index": boundary_index, "boundary_type": "|", "frame": frame, **metric})
        print(f"[{number:02d}/{len(runs)}] {run_record['run_id']}", flush=True)
    (output_dir / "metric_records.json").write_text(json.dumps(records, indent=2) + "\n")
    with (output_dir / "hard_cut_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    audit["run_count"] = len(runs)
    (output_dir / "tool_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    validation = validate(records, output_dir)
    summarize(records, validation, output_dir)


def validate(records, output_dir):
    """One locked validation against pre-existing scored human rows; no tuning."""
    from scipy.stats import spearmanr

    lookup = {(r["pair_id"], r["seed"], r["arm_id"], r["boundary_index"]): r for r in records}
    rows = _review_rows()
    hard_scores, hard_metric, continuity_scores, continuity_metric = [], [], [], []
    for row in rows:
        record = lookup.get((row["pair_id"], row["seed"], row["arm_id"], row["boundary_index"]))
        if record is None:
            continue
        if row["boundary_type"] == "#":
            hard_scores.append(row["score"])
            hard_metric.append(record["source_similarity_auc"])
        else:
            continuity_scores.append(row["score"])
            continuity_metric.append(record["collapse_signal"])
    result = {
        "scored_rows": len(rows),
        "hard_cut": {"n": len(hard_scores), "spearman_quality_vs_source_similarity_auc": float(spearmanr(hard_scores, hard_metric).statistic)},
        "continuity": {"n": len(continuity_scores), "spearman_quality_vs_collapse_signal": float(spearmanr(continuity_scores, continuity_metric).statistic)},
        "criterion": "Both pre-specified correlations must be negative; this only validates direction against the scored subset, not semantic ground truth.",
    }
    result["credible_for_descriptive_full_checkpoint"] = bool(
        result["hard_cut"]["spearman_quality_vs_source_similarity_auc"] < 0 and
        result["continuity"]["spearman_quality_vs_collapse_signal"] < 0
    )
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _group_summary(records, metric, fields):
    groups = defaultdict(list)
    for record in records:
        value = record.get(metric)
        if value is not None:
            groups[tuple(record[field] for field in fields)].append(float(value))
    return [{**dict(zip(fields, key)), "n": len(values), "mean": float(np.mean(values)),
             "median": float(np.median(values))} for key, values in sorted(groups.items())]


def _paired_metric(records, arm_a, arm_b, metric, boundary_type):
    table = {}
    for record in records:
        if record["boundary_type"] != boundary_type or record["arm_id"] not in (arm_a, arm_b):
            continue
        value = record.get(metric)
        if value is not None:
            table.setdefault((record["pair_id"], record["seed"], record["boundary_index"]), {})[record["arm_id"]] = value
    values = {key: (float(value[arm_a]), float(value[arm_b])) for key, value in table.items()
              if arm_a in value and arm_b in value}
    return _bootstrap_mean_difference(values, sorted(values))


def summarize(records, validation, output_dir):
    hard = [record for record in records if record["boundary_type"] == "#"]
    continuity = [record for record in records if record["boundary_type"] == "|"]
    reference = [record["collapse_signal"] for record in continuity
                 if record["arm_id"] in ("live_infinity_rope", "native_state_rebinding")]
    collapse_threshold = float(np.quantile(reference, 0.95))
    flagged = [record for record in continuity if record["collapse_signal"] > collapse_threshold]
    summary = {
        "scope": "Completed Phase-5 63-run checkpoint only; image-reference and collapse proxies, not semantic ground truth or generic video quality.",
        "validation": validation,
        "hard_cut": {
            "source_similarity_auc_by_arm": _group_summary(hard, "source_similarity_auc", ("arm_id",)),
            "transition_latency_by_arm": _group_summary(hard, "transition_latency_frames", ("arm_id",)),
            "source_similarity_auc_by_category_and_arm": _group_summary(hard, "source_similarity_auc", ("pair_id", "arm_id")),
            "source_similarity_auc_by_seed_and_arm": _group_summary(hard, "source_similarity_auc", ("seed", "arm_id")),
            "rebinding_minus_live_source_similarity_auc": _paired_metric(hard, "native_state_rebinding", "live_infinity_rope", "source_similarity_auc", "#"),
            "rebinding_minus_live_transition_latency": _paired_metric(hard, "native_state_rebinding", "live_infinity_rope", "transition_latency_frames", "#"),
            "note": "Always-reset hard-cut values are reported descriptively only: its earlier normal-boundary collapse can contaminate its later hard-cut input.",
        },
        "continuity": {
            "collapse_signal_by_arm": _group_summary(continuity, "collapse_signal", ("arm_id",)),
            "collapse_signal_by_category_and_arm": _group_summary(continuity, "collapse_signal", ("pair_id", "arm_id")),
            "collapse_signal_by_seed_and_arm": _group_summary(continuity, "collapse_signal", ("seed", "arm_id")),
            "objective_high_discontinuity_threshold": collapse_threshold,
            "objective_high_discontinuity_counts": _group_summary(flagged, "collapse_signal", ("arm_id",)),
            "rebinding_minus_live_collapse_signal": _paired_metric(continuity, "native_state_rebinding", "live_infinity_rope", "collapse_signal", "|"),
            "rebinding_minus_always_reset_collapse_signal": _paired_metric(continuity, "native_state_rebinding", "always_reset", "collapse_signal", "|"),
            "threshold_note": "Flag means above the fixed 95th percentile of live+rebinding continuity values; it is a diagnostic high-discontinuity flag, not a human semantic verdict.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def plot_existing(output_dir):
    import matplotlib.pyplot as plt

    with (output_dir / "hard_cut_curves.csv").open(newline="") as handle:
        curves = list(csv.DictReader(handle))
    records = json.loads((output_dir / "metric_records.json").read_text())
    colors = {"live_infinity_rope": "#bb5a4a", "always_reset": "#777777", "native_state_rebinding": "#2e7d65"}
    labels = {"live_infinity_rope": "live", "always_reset": "always reset", "native_state_rebinding": "rebinding"}
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for arm in ARMS:
        points = defaultdict(list)
        for row in curves:
            if row["arm_id"] == arm:
                points[int(row["frame_after_cut"])].append(float(row["source_reference_similarity"]))
        x = sorted(points)
        mean = np.asarray([np.mean(points[index]) for index in x])
        lo = np.asarray([np.quantile(points[index], .025) for index in x])
        hi = np.asarray([np.quantile(points[index], .975) for index in x])
        axes[0].plot(x, mean, color=colors[arm], label=labels[arm])
        axes[0].fill_between(x, lo, hi, color=colors[arm], alpha=.12)
    axes[0].set(xlabel="RGB frames after #", ylabel="DINO source-reference similarity",
                title="Hard-cut visual-source retention")
    axes[0].legend(frameon=False)
    values = [[record["collapse_signal"] for record in records if record["boundary_type"] == "|" and record["arm_id"] == arm]
              for arm in ARMS]
    axes[1].boxplot(values, tick_labels=[labels[arm] for arm in ARMS], showfliers=True)
    axes[1].set(ylabel="local chromatic/temporal disruption", title="Normal-boundary collapse proxy")
    figure.savefig(output_dir / "phase7_failure_mode_proxies.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase7_objective_evaluation_20260815")
    parser.add_argument("--audit-tools", action="store_true")
    parser.add_argument("--plot-existing", action="store_true")
    args = parser.parse_args()
    if args.audit_tools:
        print(json.dumps(_tool_audit(), indent=2))
        return
    if args.plot_existing:
        plot_existing(args.output_dir)
        return
    run(args.output_dir)


if __name__ == "__main__":
    main()
