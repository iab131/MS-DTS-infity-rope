"""Validation and flattened-index helpers for the fixed-grid recall oracle."""

import json
from dataclasses import dataclass
from pathlib import Path


GRID_HEIGHT = 30
GRID_WIDTH = 52
SUBJECT_MODES = {
    "subject_to_subject": 0,
    "subject_erode1": 1,
    "subject_erode2": 2,
    "subject_boundary_only": None,
}


def validate_fixed_grid_options(mask_path, mode, attention_memory_policy,
                                manual_frame_ids, manual_target_blocks,
                                local_retention=None, context_mode=None):
    """Validate the exact opt-in manual oracle configuration."""
    if mask_path is None and mode is None:
        return None
    if mask_path is None or mode is None:
        raise ValueError(
            "--memory-fixed-grid-mask-path and --memory-fixed-grid-mode must be provided together")
    if not attention_memory_policy:
        raise ValueError("fixed-grid recall requires --attention-memory-policy")
    if mode not in {*SUBJECT_MODES, "background_to_background", "compact_entity_memory", "latent_subject_patch",
                    "affine_aligned_latent_subject_patch", "latent_subject_patch_persistent"}:
        raise ValueError("unsupported fixed-grid recall mode")
    if manual_frame_ids != [6, 7]:
        raise ValueError("fixed-grid recall requires frame IDs 6,7 via --memory-manual-frame-ids")
    if manual_target_blocks != {8}:
        raise ValueError("fixed-grid recall requires --memory-manual-target-blocks target block 8")
    if local_retention != "transition_no_sink":
        raise ValueError("fixed-grid recall requires --memory-local-retention transition_no_sink")
    if context_mode != "replace_recent":
        raise ValueError("fixed-grid recall requires --memory-context-mode replace_recent")
    return {"mask_path": str(mask_path), "mode": mode}


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
        if set(source_masks) != {6, 7}:
            raise ValueError("source_masks must contain exactly frame IDs 6 and 7")
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
        return self.target_query_indices_for_mode("subject_to_subject")

    def background_query_indices(self):
        return self._dilated_complement(self.target_subject_mask)

    def _dilated_complement(self, mask):
        excluded = set()
        for index, value in enumerate(mask):
            if not value:
                continue
            row, column = divmod(index, self.width)
            for neighbor_row in range(max(0, row - 1), min(self.height, row + 2)):
                for neighbor_column in range(max(0, column - 1), min(self.width, column + 2)):
                    excluded.add(neighbor_row * self.width + neighbor_column)
        return [index for index in range(self.height * self.width) if index not in excluded]

    def history_token_indices(self, frame_id):
        return self.history_token_indices_for_mode("subject_to_subject", frame_id)

    def _erode(self, mask):
        """Return the 8-connected one-token interior of a binary grid mask."""
        eroded = []
        for index, value in enumerate(mask):
            row, column = divmod(index, self.width)
            eroded.append(int(value and all(
                0 <= row + dr < self.height and 0 <= column + dc < self.width and
                mask[(row + dr) * self.width + column + dc]
                for dr in (-1, 0, 1) for dc in (-1, 0, 1))))
        return tuple(eroded)

    def _subject_mask_for_mode(self, mask, mode):
        if mode not in SUBJECT_MODES:
            raise ValueError(f"{mode} is not a subject mask mode")
        erode_steps = SUBJECT_MODES[mode]
        if erode_steps is None:
            eroded = self._erode(mask)
            return tuple(int(value and not eroded[index]) for index, value in enumerate(mask))
        for _ in range(erode_steps):
            mask = self._erode(mask)
        return mask

    @staticmethod
    def _mask_indices(mask):
        return [index for index, value in enumerate(mask) if value]

    def source_mask_for_mode(self, mode, frame_id):
        return self._subject_mask_for_mode(self.source_masks[int(frame_id)], mode)

    def target_mask_for_mode(self, mode):
        return self._subject_mask_for_mode(self.target_subject_mask, mode)

    def history_token_indices_for_mode(self, mode, frame_id):
        return self._mask_indices(self.source_mask_for_mode(mode, frame_id))

    def target_query_indices_for_mode(self, mode):
        return self._mask_indices(self.target_mask_for_mode(mode))

    def history_background_token_indices(self, frame_id):
        """Return source background outside its eight-connected one-token dilation."""
        return self._dilated_complement(self.source_masks[int(frame_id)])
