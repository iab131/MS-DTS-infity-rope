#!/usr/bin/env python3
"""Create synchronized four-arm videos and review sheets from hard-cut outputs."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("live_kv_flush", "sink_plus1", "sink_only", "transition_no_sink")


def arms_for_manifest(manifest):
    """Use the checked-in arm order for a matched comparison manifest."""
    arms = tuple(arm["id"] for arm in manifest["arms"])
    if not arms:
        raise ValueError("comparison manifest has no arms")
    return arms


def transition_frame_indices(frame_count):
    """Zero-based RGB samples: two pre-cut, B block 4, then B blocks 5--6."""
    if frame_count < 69:
        raise ValueError("hard-cut review expects at least 69 RGB frames")
    return [24, 31, 32, 35, 43, 47, 55, 59, 67]


def _rgb_frame(video, index, scale):
    frame = video[0, index].permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
    image = Image.fromarray(frame).resize((round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
                                          Image.Resampling.LANCZOS)
    return image


def _label(image, text):
    band = Image.new("RGB", (image.width, image.height + 20), "black")
    band.paste(image, (0, 20))
    ImageDraw.Draw(band).text((4, 3), text, fill="white")
    return band


def _hstack(images):
    canvas = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _vstack(images):
    canvas = Image.new("RGB", (max(image.width for image in images), sum(image.height for image in images)))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def _load_case(root, pair_id, seed, arms=ARMS, arm_roots=None):
    videos = {}
    for arm in arms:
        arm_root = root if arm_roots is None else arm_roots.get(arm, root)
        path = arm_root / pair_id / f"seed_{seed}" / arm / "0_raw_decoded_before_mp4.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        video = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(video.shape[0:3]) != (1, video.shape[1], 3):
            raise ValueError(f"unexpected decoded tensor shape: {path}: {tuple(video.shape)}")
        videos[arm] = video
    return videos


def _case_sheet(videos, pair_id, seed, arms=ARMS):
    indices = transition_frame_indices(min(video.shape[1] for video in videos.values()))
    rows = []
    for arm in arms:
        frames = [_label(_rgb_frame(videos[arm], index, 0.28), f"f{index + 1}") for index in indices]
        rows.append(_label(_hstack(frames), arm))
    return _label(_vstack(rows), f"{pair_id}  seed {seed}  |  pre-cut: f25,f32  |  B: f33--f68")


def _case_video(videos, pair_id, seed, output, arms=ARMS):
    with imageio.get_writer(output, fps=16, codec="libx264", quality=8) as writer:
        for index in range(24, 69):
            rows = [_label(_rgb_frame(videos[arm], index, 0.45), f"{arm} | RGB frame {index + 1}")
                    for arm in arms]
            writer.append_data(np.asarray(_label(_vstack(rows), f"{pair_id}  seed {seed}")))


def _summary_case(videos, pair_id, seed, arms=ARMS):
    rows = []
    for arm in arms:
        frames = [_label(_rgb_frame(videos[arm], index, 0.20), f"f{index + 1}")
                  for index in (31, 35, 55, 67)]
        rows.append(_label(_hstack(frames), arm))
    return _label(_vstack(rows), f"{pair_id} / seed {seed}")


def create_comparisons(root, manifest_path, output):
    """Generate eight case videos/sheets plus a single all-case summary sheet."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    arms = arms_for_manifest(manifest)
    arm_roots = {arm["id"]: ROOT / arm["reuse_output_root"]
                 for arm in manifest["arms"] if "reuse_output_root" in arm}
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for pair in manifest["pairs"]:
        for seed in manifest["seeds"]:
            videos = _load_case(root, pair["id"], seed, arms, arm_roots)
            stem = f'{pair["id"]}_seed_{seed}_{len(arms)}_arm'
            _case_sheet(videos, pair["id"], seed, arms).save(output / f"{stem}_temporal_sheet.png")
            _case_video(videos, pair["id"], seed, output / f"{stem}_transition.mp4", arms)
            summaries.append(_summary_case(videos, pair["id"], seed, arms))
    summary_rows = [_hstack(summaries[index:index + 2]) for index in range(0, len(summaries), 2)]
    _label(_vstack(summary_rows), "Hard-cut comparison | f32 pre-cut, f36 first B block, f56/f68 later B") \
        .save(output / "hard_cut_all_cases_summary.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=ROOT / "outputs/hard_cut_transition_phase0_20260810")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "docs/HARD_CUT_BENCHMARK_PHASE0_20260810.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    create_comparisons(args.root, args.manifest, args.output or args.root / "comparison")


if __name__ == "__main__":
    main()
