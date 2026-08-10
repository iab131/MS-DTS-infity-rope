#!/usr/bin/env python3
"""Render the fixed four-arm cache-write ablation and exact RGB/latent checks."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks  # noqa: E402

BASE = ROOT / "outputs/attention_memory_policy_fixed_grid_selective_recall"
OUT = BASE / "latent_subject_patch_persistent/cache_write_mask_ablation/comparison"
ARMS = {
    "reset_only": ROOT / "outputs/attention_memory_policy_reset_then_recall_verified/reset_only_context_logged",
    "persistent_full": BASE / "latent_subject_patch_persistent",
    "persistent_cache_erode1": BASE / "latent_subject_patch_persistent/cache_write_mask_ablation/persistent_cache_erode1",
    "persistent_cache_erode2": BASE / "latent_subject_patch_persistent/cache_write_mask_ablation/persistent_cache_erode2",
}
FRAMES = [77, 81, 85, 89, 93, 97, 101, 105, 109, 113]


def _rgb(frame):
    return np.clip(frame.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)[:, :, ::-1]


def _sheet(raw, crop=False):
    tiles = []
    for name in ARMS:
        row = []
        for frame_id in FRAMES:
            image = _rgb(raw[name][0, frame_id])
            if crop:
                image = image[0:480, 288:608]
            image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
            cv2.putText(image, str(frame_id), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            row.append(image)
        row = cv2.hconcat(row)
        cv2.putText(row, name, (4, 116), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        tiles.append(row)
    return cv2.vconcat(tiles)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {name: torch.load(path / "0_raw_decoded_before_mp4.pt", map_location="cpu", weights_only=True)
           for name, path in ARMS.items()}
    reset = raw["reset_only"]
    masks = FixedGridMemoryMasks.from_json(ROOT / "docs/attention_memory_policy_fixed_grid_masks_20260809.json")
    subject = np.repeat(np.repeat(np.array(masks.target_subject_mask, dtype=bool).reshape(30, 52), 16, 0), 16, 1)
    subject = torch.from_numpy(subject)[None, None, None].expand(1, 1, 3, -1, -1)
    metrics = {"frames": FRAMES, "visible_block8_exact_vs_persistent_full": {}}
    full = raw["persistent_full"]
    for name, tensor in raw.items():
        per_frame = {}
        for frame_id in FRAMES:
            delta = (tensor[:, frame_id:frame_id + 1] - reset[:, frame_id:frame_id + 1]).abs()
            per_frame[str(frame_id)] = {
                "subject_mean_abs_rgb_delta_vs_reset": float(delta[subject].mean()),
                "outside_subject_mean_abs_rgb_delta_vs_reset": float(delta[~subject].mean()),
            }
        metrics[name] = per_frame
        if name.startswith("persistent_cache_"):
            metrics["visible_block8_exact_vs_persistent_full"][name] = bool(torch.equal(
                tensor[:, 81:93], full[:, 81:93]))
    for name, path in ARMS.items():
        latent = torch.load(path / "0_clean_latents_block_08.pt", map_location="cpu", weights_only=True)
        reset_latent = torch.load(ARMS["reset_only"] / "0_clean_latents_block_08.pt",
                                  map_location="cpu", weights_only=True)
        metrics.setdefault("preclean_block8_equals_reset", {})[name] = bool(torch.equal(latent, reset_latent))
    (OUT / "four_arm_cache_write_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cv2.imwrite(str(OUT / "four_arm_cache_write_temporal_sheet.png"), _sheet(raw))
    cv2.imwrite(str(OUT / "four_arm_cache_write_subject_crops.png"), _sheet(raw, crop=True))


if __name__ == "__main__":
    main()
