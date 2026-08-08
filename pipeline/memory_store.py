"""CPU-resident clean-pass memory for the opt-in attention-memory policy."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    scene_id: int
    frame_id: int
    descriptors: Dict[int, torch.Tensor]
    layers: Dict[int, Dict[str, torch.Tensor]]
    utility: float = 0.0


@dataclass
class SceneArchiveEntry:
    scene_id: int
    frame_ids: List[int]
    descriptor: Dict[int, torch.Tensor]
    layers: Dict[int, Dict[str, torch.Tensor]]
    utility: float


class MemoryStore:
    """Frame-addressable raw clean K/V. K/V and descriptors stay on CPU."""

    def __init__(self, frame_tokens: int, descriptor_layers: Iterable[int], memory_budget: int,
                 injection_layers: Optional[Iterable[int]] = None):
        if frame_tokens <= 0 or memory_budget <= 0:
            raise ValueError("frame_tokens and memory_budget must be positive")
        self.frame_tokens = frame_tokens
        self.descriptor_layers = tuple(sorted(set(descriptor_layers)))
        if not self.descriptor_layers:
            raise ValueError("at least one descriptor layer is required")
        self.memory_budget = memory_budget
        self.injection_layers = tuple(sorted(set(
            self.descriptor_layers if injection_layers is None else injection_layers)))
        if not self.injection_layers:
            raise ValueError("at least one injection layer is required")
        self.entries: List[MemoryEntry] = []
        self.archive: List[SceneArchiveEntry] = []

    def __len__(self):
        return len(self.entries)

    @property
    def frame_ids(self):
        return [entry.frame_id for entry in self.entries]

    def add_clean_block(self, scene_id: int, frame_ids: List[int], layers: List[Dict[str, torch.Tensor]]):
        if not frame_ids or not layers:
            return []
        block_tokens = len(frame_ids) * self.frame_tokens
        if any(layer["k"].shape[0] != 1 or layer["k"].shape[1] != block_tokens for layer in layers):
            raise ValueError("clean K/V must be one batch with one complete span per frame")
        if any(layer["k"].shape != layer["v"].shape for layer in layers):
            raise ValueError("clean K and V must have matching shapes")

        added = []
        for index, frame_id in enumerate(frame_ids):
            start, end = index * self.frame_tokens, (index + 1) * self.frame_tokens
            missing = [layer for layer in set(self.descriptor_layers) | set(self.injection_layers)
                       if layer >= len(layers)]
            if missing:
                raise ValueError(f"descriptor or injection layers are not captured: {missing}")
            frame_layers = {
                layer_index: {
                    "k": layer["k"][:, start:end].detach().to(device="cpu", copy=True),
                    "v": layer["v"][:, start:end].detach().to(device="cpu", copy=True),
                }
                for layer_index, layer in enumerate(layers) if layer_index in self.injection_layers
            }
            descriptors = {
                layer: layers[layer]["k"][0, start:end].detach().float().mean(dim=0).cpu()
                for layer in self.descriptor_layers
            }
            entry = MemoryEntry(scene_id, frame_id, descriptors, frame_layers)
            self.entries.append(entry)
            added.append(entry)
        return added

    def get_entries(self, frame_ids: Iterable[int]):
        by_id = {entry.frame_id: entry for entry in self.entries}
        missing = [frame_id for frame_id in frame_ids if frame_id not in by_id]
        if missing:
            raise ValueError(f"manual memory frames are unavailable: {missing}")
        return [by_id[frame_id] for frame_id in frame_ids]

    def pack_kv(self, entries: List[MemoryEntry], num_layers: Optional[int] = None):
        if not entries:
            return None
        layer_count = max(entries[0].layers) + 1 if num_layers is None else num_layers
        packed = [None] * layer_count
        for layer in entries[0].layers:
            packed[layer] = {
                "k": torch.cat([entry.layers[layer]["k"] for entry in entries], dim=1),
                "v": torch.cat([entry.layers[layer]["v"] for entry in entries], dim=1),
            }
        return packed

    def archive_scene(self, scene_id: int, top_m: int):
        candidates = [entry for entry in self.entries if entry.scene_id == scene_id]
        if not candidates or top_m <= 0:
            return None
        selected = sorted(candidates, key=lambda entry: (-entry.utility, entry.frame_id))[:top_m]
        weights = torch.tensor([max(0.0, entry.utility) for entry in selected], dtype=torch.float32)
        if weights.sum() == 0:
            weights.fill_(1.0 / len(selected))
        else:
            weights /= weights.sum()
        layer_ids = selected[0].layers.keys()
        layers = {
            layer: {
                name: sum(weight * entry.layers[layer][name].float() for weight, entry in zip(weights, selected)).to(
                    selected[0].layers[layer][name].dtype)
                for name in ("k", "v")
            }
            for layer in layer_ids
        }
        descriptor = {
            layer: sum(weight * entry.descriptors[layer].float() for weight, entry in zip(weights, selected)).cpu()
            for layer in self.descriptor_layers
        }
        archive = SceneArchiveEntry(
            scene_id=scene_id,
            frame_ids=[entry.frame_id for entry in selected],
            descriptor=descriptor,
            layers=layers,
            utility=float(sum(entry.utility for entry in selected)),
        )
        self.archive.append(archive)
        return archive

    def consolidate(self, target_budget: Optional[int] = None, diversity_threshold: float = 0.9):
        target_budget = self.memory_budget if target_budget is None else target_budget
        if target_budget <= 0:
            raise ValueError("target_budget must be positive")
        if len(self.entries) <= target_budget:
            return {"performed": False, "before": len(self.entries), "after": len(self.entries), "removed": []}
        ranked = sorted(self.entries, key=lambda entry: (-entry.utility, entry.frame_id))
        keep = ranked[:max(1, target_budget // 2)]
        for candidate in ranked[len(keep):]:
            if all(self._descriptor_similarity(candidate, kept) < diversity_threshold for kept in keep):
                keep.append(candidate)
            if len(keep) == target_budget:
                break
        removed = sorted(set(self.frame_ids) - {entry.frame_id for entry in keep})
        self.entries = sorted(keep, key=lambda entry: entry.frame_id)
        return {"performed": True, "before": len(ranked), "after": len(self.entries), "removed": removed}

    def trim_archive(self, recent_scenes: int, high_utility: int):
        if len(self.archive) <= recent_scenes + high_utility:
            return []
        recent = self.archive[-recent_scenes:] if recent_scenes else []
        high = sorted(self.archive, key=lambda entry: (-entry.utility, -entry.scene_id))[:high_utility]
        keep = {id(entry) for entry in recent + high}
        removed = [entry.scene_id for entry in self.archive if id(entry) not in keep]
        self.archive = [entry for entry in self.archive if id(entry) in keep]
        return removed

    def _descriptor_similarity(self, left: MemoryEntry, right: MemoryEntry):
        scores = []
        for layer in self.descriptor_layers:
            a, b = left.descriptors[layer].flatten(), right.descriptors[layer].flatten()
            scores.append(float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()))
        return sum(scores) / len(scores)
