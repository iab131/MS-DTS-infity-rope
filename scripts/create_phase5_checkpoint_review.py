#!/usr/bin/env python3
"""Create anonymized Phase-5 checkpoint review artifacts without scoring them."""

import argparse
import csv
import json
from pathlib import Path
from random import Random

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

try:
    from scripts.create_hard_cut_comparisons import _hstack, _label, _rgb_frame, _vstack
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from create_hard_cut_comparisons import _hstack, _label, _rgb_frame, _vstack


ROOT = Path(__file__).resolve().parents[1]
ARM_LABELS = ("Arm A", "Arm B", "Arm C")


def phase5_boundary_windows(frame_count, transition_raw_frames):
    """Return [start, stop) RGB windows: two pre, full block, three later blocks."""
    windows = []
    for first_frame in transition_raw_frames:
        start, stop = first_frame - 3, first_frame + 47
        if start < 0 or stop > frame_count:
            raise ValueError(f"need RGB frames through {stop}, received {frame_count}")
        windows.append((start, stop))
    return windows


def blinded_arm_order(pair_id, seed, arm_ids):
    """Deterministic per-case randomization; mapping is stored privately by caller."""
    order = list(arm_ids)
    Random(f"phase5-checkpoint:{pair_id}:{seed}").shuffle(order)
    return tuple(order)


def build_review_rows(manifest):
    """Create blank anonymous scoring rows and the separate private arm mapping."""
    rows, mappings = [], {}
    arm_ids = tuple(arm["id"] for arm in manifest["arms"])
    for pair in manifest["pairs"]:
        for seed in manifest["seeds"]:
            case_id = f'{pair["id"]}__seed{seed}'
            order = blinded_arm_order(pair["id"], seed, arm_ids)
            mappings[case_id] = dict(zip(ARM_LABELS, order))
            for boundary_index, boundary_type in enumerate(pair["boundary_after"], start=1):
                field_set = ("identity_entity_consistency; background_consistency; action_motion_adherence; "
                             "temporal_flicker; motion_continuity; collapse_failure"
                             if boundary_type == "|" else
                             "previous_scene_semantic_leakage; new_scene_prompt_adherence; "
                             "transition_latency_rgb_frames; flicker_artifact_severity; later_scene_stability")
                for label in ARM_LABELS:
                    rows.append({
                        "case_id": case_id, "boundary_index": boundary_index,
                        "boundary_type": boundary_type, "anonymous_arm": label,
                        "fields": field_set, "reviewer_id": "", "score_1_to_5": "",
                        "uncertain": "", "notes": "",
                    })
    return rows, mappings


def _load_video(root, pair_id, seed, arm):
    path = root / pair_id / f"seed_{seed}" / arm / "0_raw_decoded_before_mp4.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _sheet(videos, order, window, title):
    start, stop = window
    first = start + 2
    indices = [start, start + 1, first, first + 11, first + 23, first + 35, stop - 1]
    rows = []
    for label, arm in zip(ARM_LABELS, order):
        frames = [_label(_rgb_frame(videos[arm], index, 0.24), f"RGB f{index + 1}") for index in indices]
        rows.append(_label(_hstack(frames), label))
    return _label(_vstack(rows), title)


def _video(videos, order, start, stop, output, title):
    with imageio.get_writer(output, fps=16, codec="libx264", quality=8) as writer:
        for index in range(start, stop):
            rows = [_label(_rgb_frame(videos[arm], index, 0.42), f"{label} | RGB f{index + 1}")
                    for label, arm in zip(ARM_LABELS, order)]
            writer.append_data(np.asarray(_label(_vstack(rows), title)))


def create_review(manifest_path, root, output):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows, mapping = build_review_rows(manifest)
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private"
    private.mkdir(exist_ok=True)
    (private / "arm_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    columns = list(rows[0])
    with (output / "blinded_review_form.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "review_instructions.md").write_text(
        "# Phase 5 blinded review\n\nScore each listed field from 1 (worst) to 5 (best); "
        "mark `uncertain` rather than guessing. Review arm labels are anonymous. "
        "Do not infer expected outcomes from the ordering.\n", encoding="utf-8")
    for case_number, pair in enumerate(manifest["pairs"], start=1):
        for seed in manifest["seeds"]:
            order = blinded_arm_order(pair["id"], seed, tuple(arm["id"] for arm in manifest["arms"]))
            videos = {arm: _load_video(root, pair["id"], seed, arm) for arm in order}
            frame_count = min(video.shape[1] for video in videos.values())
            windows = phase5_boundary_windows(frame_count, pair["transition_raw_frames"])
            case_name = f"case_{case_number:02d}_seed_{seed}"
            for boundary_number, window in enumerate(windows, start=1):
                _sheet(videos, order, window, f"{case_name} | boundary {boundary_number}").save(
                    output / f"{case_name}_boundary_{boundary_number}_temporal_sheet.png")
            _video(videos, order, windows[0][0], windows[-1][1],
                   output / f"{case_name}_synchronized.mp4", f"{case_name} | anonymized arms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "docs/PHASE5_GENERALIZATION_CHECKPOINT_20260813.json")
    parser.add_argument("--root", type=Path,
                        default=ROOT / "outputs/phase5_generalization_checkpoint_20260813")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    create_review(args.manifest, args.root, args.output or args.root / "blinded_review")


if __name__ == "__main__":
    main()
