#!/usr/bin/env python3
"""Create CPU-only fixed-grid source/target overlays and a provenance audit."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks  # noqa: E402


FRAME_TOKENS = 30 * 52
SOURCE_DECODED_FRAMES = {6: 26, 7: 30}
TARGET_LATENT_FRAMES = [21, 22, 23]
TARGET_DECODED_FRAMES = [81, 85, 89]
SUBJECT_ABLATION_MODES = (
    ("subject_to_subject", "full subject"),
    ("subject_erode1", "erode1 core"),
    ("subject_erode2", "erode2 core"),
    ("subject_boundary_only", "boundary ring"),
)


def _coordinates(indices):
    return [list(divmod(index, 52)) for index in indices]


def _expanded(indices):
    return [frame * FRAME_TOKENS + index for frame in range(3) for index in indices]


def build_mask_audit(masks):
    """Return exact token provenance independent of video decoding."""
    current_start_frame, context_non_sink_frames, current_num_frames = 21, 2, 3
    sink_frame = current_start_frame - context_non_sink_frames - 1
    base_ordering = (
        [f"sink:{sink_frame}"] +
        [f"local:{frame_id}" for frame_id in range(
            current_start_frame - context_non_sink_frames, current_start_frame)] +
        [f"current:{frame_id}" for frame_id in range(
            current_start_frame, current_start_frame + current_num_frames)]
    )
    source_history = {}
    for frame_id, slot in ((6, 1), (7, 2)):
        subject = masks.history_token_indices(frame_id)
        background = masks.history_background_token_indices(frame_id)
        source_history[str(frame_id)] = {
            "decoded_frame": SOURCE_DECODED_FRAMES[frame_id],
            "temporal_slot": slot,
            "subject_token_count": len(subject),
            "subject_token_indices": subject,
            "subject_row_col_coordinates": _coordinates(subject),
            "background_token_count": len(background),
            "background_token_indices": background,
            "background_row_col_coordinates": _coordinates(background),
        }
    subject_queries = masks.subject_query_indices()
    background_queries = masks.background_query_indices()
    subject_ablation = {}
    for mode in ("subject_to_subject", "subject_erode1", "subject_erode2", "subject_boundary_only"):
        target = masks.target_query_indices_for_mode(mode)
        subject_ablation[mode] = {
            "source": {
                str(frame_id): {
                    "token_count": len(masks.history_token_indices_for_mode(mode, frame_id)),
                    "token_indices": masks.history_token_indices_for_mode(mode, frame_id),
                    "row_col_coordinates": _coordinates(
                        masks.history_token_indices_for_mode(mode, frame_id)),
                }
                for frame_id in (6, 7)
            },
            "target_per_frame_token_count": len(target),
            "target_per_frame_indices": target,
            "target_per_frame_row_col_coordinates": _coordinates(target),
            "target_query_count": 3 * len(target),
            "target_query_indices": _expanded(target),
        }
    return {
        "grid": {"height": 30, "width": 52, "tokens_per_frame": FRAME_TOKENS,
                 "index_order": "frame_then_row_major"},
        "source_history": source_history,
        "source_totals": {
            "subject_token_count": sum(item["subject_token_count"] for item in source_history.values()),
            "background_token_count": sum(item["background_token_count"] for item in source_history.values()),
        },
        "target": {
            "block": 8,
            "latent_frame_ids": TARGET_LATENT_FRAMES,
            "decoded_query_frames": TARGET_DECODED_FRAMES,
            "subject_per_frame_indices": subject_queries,
            "subject_per_frame_row_col_coordinates": _coordinates(subject_queries),
            "subject_query_indices": _expanded(subject_queries),
            "subject_query_count": 3 * len(subject_queries),
            "background_per_frame_indices": background_queries,
            "background_per_frame_row_col_coordinates": _coordinates(background_queries),
            "background_query_indices": _expanded(background_queries),
            "background_query_count": 3 * len(background_queries),
        },
        "subject_ablation": subject_ablation,
        "base_context": {
            "ordering": base_ordering,
            "derived_from": {
                "current_start_frame": current_start_frame,
                "context_non_sink_frames": context_non_sink_frames,
                "current_num_frames": current_num_frames,
            },
            "frames": len(base_ordering),
            "tokens": len(base_ordering) * FRAME_TOKENS,
            "local_current_order_unchanged": True,
            "historical_tokens_are_separate_from_base_context": True,
        },
        "arm_isolation": {
            "subject_to_subject": "source subject tokens only -> target subject queries only",
            "background_to_background":
                "source dilated-complement tokens only -> target dilated-complement queries only",
        },
    }


def _read_frames(path, requested):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open reset-only MP4: {path}")
    frames = {}
    index = 0
    wanted = set(requested)
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = frame
            wanted.remove(index)
        index += 1
    capture.release()
    if wanted:
        raise ValueError(f"reset-only MP4 is missing decoded frames: {sorted(wanted)}")
    if any(frame.shape[:2] != (480, 832) for frame in frames.values()):
        raise ValueError("fixed-grid overlays require the verified 480x832 reset-only MP4")
    return frames


def _write_overlay(frame, mask, path, label):
    mask_pixels = np.repeat(np.repeat(
        np.asarray(mask, dtype=bool).reshape(30, 52), 16, axis=0), 16, axis=1)
    output = frame.copy()
    tint = np.zeros_like(output)
    tint[:, :, 1] = 255
    output[mask_pixels] = cv2.addWeighted(
        output[mask_pixels], 0.45, tint[mask_pixels], 0.55, 0)
    for row in range(31):
        y = min(row * 16, 479)
        cv2.line(output, (0, y), (831, y), (255, 128, 0), 1)
    for column in range(53):
        x = min(column * 16, 831)
        cv2.line(output, (x, 0), (x, 479), (255, 128, 0), 1)
    cv2.putText(output, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), output):
        raise OSError(f"failed to write overlay: {path}")


def _latent_patch_support_masks(masks, erode_steps):
    """Return 30x52 source-supported target masks for the 6 -> 6/7 -> 7 patch map."""
    target = (masks.target_subject_mask if erode_steps == 0 else
              masks.target_mask_for_mode(f"subject_erode{erode_steps}"))
    source6 = tuple(a and b for a, b in zip(masks.source_masks[6], target))
    source7 = tuple(a and b for a, b in zip(masks.source_masks[7], target))
    return (source6, tuple(a and b for a, b in zip(source6, source7)), source7)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask-path", type=Path,
        default=ROOT / "docs/attention_memory_policy_fixed_grid_masks_20260809.json")
    parser.add_argument(
        "--video-path", type=Path,
        default=ROOT / "outputs/attention_memory_policy_reset_then_recall_verified/"
                       "reset_only_context_logged/0-0_ema.mp4")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/attention_memory_policy_fixed_grid_selective_recall/preflight")
    parser.add_argument("--subject-ablation-overlays", action="store_true",
                        help="Write full/core/ring source and target overlays for the fixed oracle.")
    parser.add_argument("--latent-cache-write-overlays", action="store_true",
                        help="Write full visible-patch and eroded cache-write overlays for block 8.")
    args = parser.parse_args()

    masks = FixedGridMemoryMasks.from_json(args.mask_path)
    requested = list(SOURCE_DECODED_FRAMES.values()) + TARGET_DECODED_FRAMES
    frames = _read_frames(args.video_path, requested)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for frame_id, decoded_frame in SOURCE_DECODED_FRAMES.items():
        path = args.output_dir / f"source-{frame_id}-frame-{decoded_frame}-overlay.png"
        _write_overlay(
            frames[decoded_frame], masks.source_masks[frame_id], path,
            f"source ID {frame_id} / decoded frame {decoded_frame}")
        artifacts.append(path)
    for latent_frame, decoded_frame in zip(TARGET_LATENT_FRAMES, TARGET_DECODED_FRAMES):
        path = args.output_dir / f"target-block-8-frame-{decoded_frame}-overlay.png"
        _write_overlay(
            frames[decoded_frame], masks.target_subject_mask, path,
            f"target block 8 / latent ID {latent_frame} / decoded frame {decoded_frame}")
        artifacts.append(path)

    if args.subject_ablation_overlays:
        for mode, label in SUBJECT_ABLATION_MODES:
            for frame_id, decoded_frame in SOURCE_DECODED_FRAMES.items():
                path = args.output_dir / (
                    f"{mode}-source-{frame_id}-frame-{decoded_frame}-overlay.png")
                _write_overlay(frames[decoded_frame], masks.source_mask_for_mode(mode, frame_id), path,
                               f"{label} / source ID {frame_id} / decoded frame {decoded_frame}")
                artifacts.append(path)
            for latent_frame, decoded_frame in zip(TARGET_LATENT_FRAMES, TARGET_DECODED_FRAMES):
                path = args.output_dir / (
                    f"{mode}-target-block-8-frame-{decoded_frame}-overlay.png")
                _write_overlay(frames[decoded_frame], masks.target_mask_for_mode(mode), path,
                               f"{label} / target block 8 / latent ID {latent_frame} / decoded frame {decoded_frame}")
                artifacts.append(path)

    cache_write_masks = {}
    if args.latent_cache_write_overlays:
        for erode_steps, label in ((0, "full visible patch"), (1, "cache erode1 write"),
                                   (2, "cache erode2 write")):
            supports = _latent_patch_support_masks(masks, erode_steps)
            cache_write_masks[str(erode_steps)] = {
                "per_target_token_counts": [int(sum(mask)) for mask in supports],
                "per_target_latent_cell_counts": [int(sum(mask) * 4) for mask in supports],
            }
            stem = "full_visible_patch" if erode_steps == 0 else f"cache_erode{erode_steps}"
            for latent_frame, decoded_frame, support in zip(
                    TARGET_LATENT_FRAMES, TARGET_DECODED_FRAMES, supports):
                path = args.output_dir / f"{stem}-target-block-8-frame-{decoded_frame}-overlay.png"
                _write_overlay(
                    frames[decoded_frame], support, path,
                    f"{label} / target block 8 / latent ID {latent_frame}")
                artifacts.append(path)

    audit = build_mask_audit(masks)
    audit["inputs"] = {
        "mask_path": str(args.mask_path),
        "mask_sha256": _sha256(args.mask_path),
        "reset_only_video_path": str(args.video_path),
        "reset_only_video_sha256": _sha256(args.video_path),
    }
    if cache_write_masks:
        audit["latent_cache_write_masks"] = cache_write_masks
    audit["overlays"] = [
        {"path": str(path), "sha256": _sha256(path)} for path in artifacts
    ]
    audit_path = args.output_dir / "mask_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(audit_path)


if __name__ == "__main__":
    main()
