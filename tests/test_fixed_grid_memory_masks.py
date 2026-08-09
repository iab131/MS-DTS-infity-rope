#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks


class FixedGridMemoryMasksTest(unittest.TestCase):
    def write_mask_file(self, payload):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with handle:
            json.dump(payload, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_requires_the_fixed_30_by_52_grid(self):
        path = self.write_mask_file({
            "height": 29,
            "width": 52,
            "source_masks": {"6": [0], "7": [1]},
            "target_subject_mask": [0],
        })
        with self.assertRaisesRegex(ValueError, "30x52"):
            FixedGridMemoryMasks.from_json(path)

    def test_background_excludes_one_token_dilated_subject_boundary(self):
        subject = [[0] * 52 for _ in range(30)]
        subject[10][20] = 1
        masks = FixedGridMemoryMasks.from_json(self.write_mask_file({
            "height": 30,
            "width": 52,
            "source_masks": {"6": subject, "7": subject},
            "target_subject_mask": subject,
        }))

        self.assertEqual(masks.subject_query_indices(), [10 * 52 + 20])
        background = set(masks.background_query_indices())
        self.assertNotIn(10 * 52 + 20, background)
        for row in range(9, 12):
            for column in range(19, 22):
                self.assertNotIn(row * 52 + column, background)
        self.assertIn(0, background)

    def test_history_indices_are_flattened_per_source_frame(self):
        source = [[0] * 52 for _ in range(30)]
        source[2][3] = 1
        source[29][51] = 1
        masks = FixedGridMemoryMasks.from_json(self.write_mask_file({
            "height": 30,
            "width": 52,
            "source_masks": {"6": source, "7": source},
            "target_subject_mask": source,
        }))

        self.assertEqual(masks.history_token_indices(6), [2 * 52 + 3, 29 * 52 + 51])
        with self.assertRaises(KeyError):
            masks.history_token_indices(8)


if __name__ == "__main__":
    unittest.main()
