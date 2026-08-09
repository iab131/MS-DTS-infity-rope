"""Validation and flattened-index helpers for the fixed-grid recall oracle."""

import json
from dataclasses import dataclass
from pathlib import Path


GRID_HEIGHT = 30
GRID_WIDTH = 52


@dataclass(frozen=True)
class FixedGridMemoryMasks:
    """Fixed 30x52 source masks and one target subject mask.

    Indices are row-major within one latent frame.  The background query set
    removes the subject plus its eight-connected one-token boundary.
    """

    source_masks: dict[int, tuple[int, ...]]
    target_subject_mask: tuple[int, ...]
    height: int = GRID_HEIGHT
    width: int = GRID_WIDTH

    @classmethod
    def from_json(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        height = payload.get("height")
        width = payload.get("width")
        if isinstance(payload.get("grid"), dict):
            height = payload["grid"].get("height", height)
            width = payload["grid"].get("width", width)
        if (height, width) != (GRID_HEIGHT, GRID_WIDTH):
            raise ValueError("fixed-grid masks must use a 30x52 grid")

        source_payload = payload.get("source_masks")
        if not isinstance(source_payload, dict):
            raise ValueError("source_masks must be an object")
        source_masks = {
            int(frame_id): cls._flatten_mask(mask, "source mask")
            for frame_id, mask in source_payload.items()
        }
        if not {6, 7}.issubset(source_masks):
            raise ValueError("source_masks must include frame IDs 6 and 7")
        target = cls._flatten_mask(payload.get("target_subject_mask"), "target subject mask")
        return cls(source_masks, target)

    @staticmethod
    def _flatten_mask(mask, name):
        if not isinstance(mask, list):
            raise ValueError(f"{name} must be a list")
        if len(mask) == GRID_HEIGHT and all(isinstance(row, list) for row in mask):
            if any(len(row) != GRID_WIDTH for row in mask):
                raise ValueError(f"{name} must be 30x52")
            values = [value for row in mask for value in row]
        elif len(mask) == GRID_HEIGHT * GRID_WIDTH:
            values = mask
        else:
            raise ValueError(f"{name} must be 30x52")
        if any(value not in (0, 1, False, True) for value in values):
            raise ValueError(f"{name} must contain only 0/1 values")
        return tuple(int(value) for value in values)

    def subject_query_indices(self):
        return [index for index, value in enumerate(self.target_subject_mask) if value]

    def background_query_indices(self):
        excluded = set(self.subject_query_indices())
        for index, value in enumerate(self.target_subject_mask):
            if not value:
                continue
            row, column = divmod(index, self.width)
            for neighbor_row in range(max(0, row - 1), min(self.height, row + 2)):
                for neighbor_column in range(max(0, column - 1), min(self.width, column + 2)):
                    excluded.add(neighbor_row * self.width + neighbor_column)
        return [index for index in range(self.height * self.width) if index not in excluded]

    def history_token_indices(self, frame_id):
        return [index for index, value in enumerate(self.source_masks[int(frame_id)]) if value]
