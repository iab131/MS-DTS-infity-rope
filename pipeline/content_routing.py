"""Descriptor routing and JSONL instrumentation for attention memory."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclass
class RoutedMemory:
    entries: list
    scores: List[Optional[float]]


def retrieval_allowed(retrieval_enabled, is_transition_block, transition_auto_retrieval,
                      manual_frame_ids, manual_target_blocks, block_number,
                      manual_retrieval_lifetime="pulse_1"):
    """Keep first-new-scene visual routing experimental without blocking manual oracle use."""
    if not retrieval_enabled:
        return False, "retrieval_disabled"
    if manual_frame_ids is not None:
        if manual_target_blocks is not None:
            first_block = min(manual_target_blocks)
            if manual_retrieval_lifetime == "pulse_1":
                allowed = block_number == first_block
            elif manual_retrieval_lifetime == "pulse_2":
                allowed = first_block <= block_number < first_block + 2
            elif manual_retrieval_lifetime == "persistent":
                allowed = block_number >= first_block
            else:
                raise ValueError(f"unknown manual retrieval lifetime: {manual_retrieval_lifetime}")
            if not allowed:
                return False, "manual_target_not_selected" if block_number < first_block else "manual_lifetime_expired"
        return True, "manual_override"
    if is_transition_block and not transition_auto_retrieval:
        return False, "automatic_transition_disabled"
    return True, "automatic_routing"


def route_memory(store, query_descriptors: Dict[int, torch.Tensor], k: int,
                 exclude_frame_ids: Iterable[int] = (), manual_frame_ids: Optional[List[int]] = None):
    if k < 0:
        raise ValueError("retrieval k must be non-negative")
    if manual_frame_ids is not None:
        if len(manual_frame_ids) != k or len(set(manual_frame_ids)) != len(manual_frame_ids):
            raise ValueError("manual retrieval requires k distinct frame IDs")
        return RoutedMemory(store.get_entries(manual_frame_ids), [None] * k)
    excluded = set(exclude_frame_ids)
    candidates = [entry for entry in store.entries if entry.frame_id not in excluded]
    scored = []
    for entry in candidates:
        similarities = []
        for layer, query in query_descriptors.items():
            if layer not in entry.descriptors:
                continue
            query = query.detach().float().cpu().flatten()
            descriptor = entry.descriptors[layer].float().flatten()
            similarities.append(float(F.cosine_similarity(query.unsqueeze(0), descriptor.unsqueeze(0)).item()))
        if similarities:
            scored.append((sum(similarities) / len(similarities), entry))
    selected = sorted(scored, key=lambda item: (-item[0], item[1].frame_id))[:k]
    for score, entry in selected:
        entry.utility += max(0.0, score)
    return RoutedMemory([entry for _, entry in selected], [score for score, _ in selected])


class MemoryPolicyEventLogger:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None

    def write(self, event: str, payload: dict):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **payload}, sort_keys=True, default=_json_default) + "\n")

    def read_rows(self):
        if self.path is None or not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
