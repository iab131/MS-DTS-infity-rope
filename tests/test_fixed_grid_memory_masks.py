#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks
from wan.modules.causal_model import (
    grouped_selective_attention,
    sparse_historical_rope,
)


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

    def test_rejects_source_frames_other_than_six_and_seven(self):
        mask = [0] * (30 * 52)
        path = self.write_mask_file({
            "height": 30,
            "width": 52,
            "source_masks": {"6": mask, "7": mask, "8": mask},
            "target_subject_mask": mask,
        })
        with self.assertRaisesRegex(ValueError, "exactly frame IDs 6 and 7"):
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
        source_6 = [[0] * 52 for _ in range(30)]
        source_6[2][3] = 1
        source_7 = [[0] * 52 for _ in range(30)]
        source_7[29][51] = 1
        masks = FixedGridMemoryMasks.from_json(self.write_mask_file({
            "height": 30,
            "width": 52,
            "source_masks": {"6": source_6, "7": source_7},
            "target_subject_mask": source_6,
        }))

        self.assertEqual(masks.history_token_indices(6), [2 * 52 + 3])
        self.assertEqual(masks.history_token_indices(7), [29 * 52 + 51])

    def test_grouped_selective_attention_adds_isolated_history_groups_in_query_order(self):
        calls = []

        def cpu_attention(query, key, value):
            calls.append((
                query[:, :, 0, 0].tolist(),
                key[:, :, 0, 0].tolist(),
                value[:, :, 0, 0].tolist(),
            ))
            return query + value.sum(dim=1, keepdim=True)

        query = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
        base_key = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1)
        base_value = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1)
        selective_memory = [
            {
                "query_indices": torch.tensor([3, 1]),
                "historical_key": torch.tensor([1000.0]).view(1, 1, 1, 1),
                "historical_value": torch.tensor([100.0]).view(1, 1, 1, 1),
            },
            {
                "query_indices": torch.tensor([4]),
                "historical_key": torch.tensor([2000.0]).view(1, 1, 1, 1),
                "historical_value": torch.tensor([200.0]).view(1, 1, 1, 1),
            },
        ]

        output = grouped_selective_attention(
            query, base_key, base_value, selective_memory, attention_fn=cpu_attention)

        self.assertEqual(calls, [
            ([[0.0, 1.0, 2.0, 3.0, 4.0]], [[10.0, 20.0]], [[10.0, 20.0]]),
            ([[3.0, 1.0]], [[1000.0]], [[100.0]]),
            ([[4.0]], [[2000.0]], [[200.0]]),
        ])
        self.assertEqual(output[:, :, 0, 0].tolist(), [[30.0, 132.0, 32.0, 136.0, 238.0]])

    def test_sparse_historical_rope_keeps_original_spatial_coordinates(self):
        angles = torch.tensor([
            [0, 1, 0, 1, 0, 1],
            [10, 11, 20, 21, 30, 31],
            [40, 41, 50, 51, 60, 61],
        ], dtype=torch.float64)
        freqs = torch.polar(torch.ones_like(angles), angles)
        key = torch.tensor([1.0, 0.0] * 6, dtype=torch.float32).view(1, 1, 1, 12).repeat(1, 2, 1, 1)

        output = sparse_historical_rope(
            key, torch.tensor([[2, 2, 2]]), freqs,
            torch.tensor([0, 3]), torch.tensor([1, 2]))
        complex_output = torch.view_as_complex(output.to(torch.float64).reshape(1, 2, 1, 6, 2))

        self.assertTrue(torch.allclose(
            complex_output[0, 0, 0], torch.polar(torch.ones(6, dtype=torch.float64), torch.tensor([10, 11, 0, 1, 0, 1], dtype=torch.float64))))
        self.assertTrue(torch.allclose(
            complex_output[0, 1, 0], torch.polar(torch.ones(6, dtype=torch.float64), torch.tensor([40, 41, 20, 21, 30, 31], dtype=torch.float64))))


if __name__ == "__main__":
    unittest.main()
