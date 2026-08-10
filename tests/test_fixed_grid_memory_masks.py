#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.fixed_grid_memory_masks import (
    FixedGridMemoryMasks,
    validate_fixed_grid_options,
)
from wan.modules.causal_model import (
    grouped_selective_attention,
    sparse_historical_rope,
    temporal_only_historical_rope,
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

    def test_subject_core_and_boundary_modes_partition_the_full_mask(self):
        root = Path(__file__).resolve().parents[1]
        masks = FixedGridMemoryMasks.from_json(
            root / "docs/attention_memory_policy_fixed_grid_masks_20260809.json")

        for frame_id in (6, 7):
            full = set(masks.history_token_indices_for_mode("subject_to_subject", frame_id))
            erode1 = set(masks.history_token_indices_for_mode("subject_erode1", frame_id))
            erode2 = set(masks.history_token_indices_for_mode("subject_erode2", frame_id))
            boundary = set(masks.history_token_indices_for_mode("subject_boundary_only", frame_id))
            self.assertEqual(full, erode1 | boundary)
            self.assertFalse(erode1 & boundary)
            self.assertTrue(erode2 < erode1 < full)

        full = set(masks.target_query_indices_for_mode("subject_to_subject"))
        erode1 = set(masks.target_query_indices_for_mode("subject_erode1"))
        erode2 = set(masks.target_query_indices_for_mode("subject_erode2"))
        boundary = set(masks.target_query_indices_for_mode("subject_boundary_only"))
        self.assertEqual(full, erode1 | boundary)
        self.assertFalse(erode1 & boundary)
        self.assertTrue(erode2 < erode1 < full)

    def test_history_background_excludes_each_sources_dilated_subject(self):
        source_6 = [[0] * 52 for _ in range(30)]
        source_6[10][20] = 1
        source_7 = [[0] * 52 for _ in range(30)]
        source_7[0][0] = 1
        masks = FixedGridMemoryMasks.from_json(self.write_mask_file({
            "height": 30,
            "width": 52,
            "source_masks": {"6": source_6, "7": source_7},
            "target_subject_mask": source_6,
        }))

        background_6 = set(masks.history_background_token_indices(6))
        for row in range(9, 12):
            for column in range(19, 22):
                self.assertNotIn(row * 52 + column, background_6)
        self.assertIn(0, background_6)
        self.assertNotIn(0, masks.history_background_token_indices(7))
        self.assertNotIn(1, masks.history_background_token_indices(7))
        self.assertNotIn(52, masks.history_background_token_indices(7))
        self.assertNotIn(53, masks.history_background_token_indices(7))

    def test_fixed_grid_options_are_jointly_opt_in_and_exact(self):
        self.assertIsNone(validate_fixed_grid_options(None, None, False, None, None))
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            validate_fixed_grid_options("masks.json", None, True, [6, 7], {8})
        with self.assertRaisesRegex(ValueError, "--attention-memory-policy"):
            validate_fixed_grid_options("masks.json", "subject_to_subject", False, [6, 7], {8})
        with self.assertRaisesRegex(ValueError, "frame IDs 6,7"):
            validate_fixed_grid_options("masks.json", "subject_to_subject", True, [7, 6], {8})
        with self.assertRaisesRegex(ValueError, "target block 8"):
            validate_fixed_grid_options("masks.json", "subject_to_subject", True, [6, 7], {7})
        with self.assertRaisesRegex(ValueError, "transition_no_sink"):
            validate_fixed_grid_options(
                "masks.json", "subject_to_subject", True, [6, 7], {8},
                local_retention="sink_only", context_mode="replace_recent")
        with self.assertRaisesRegex(ValueError, "replace_recent"):
            validate_fixed_grid_options(
                "masks.json", "subject_to_subject", True, [6, 7], {8},
                local_retention="transition_no_sink", context_mode="prepend")
        self.assertEqual(
            validate_fixed_grid_options(
                "masks.json", "background_to_background", True, [6, 7], {8},
                local_retention="transition_no_sink", context_mode="replace_recent"),
            {"mask_path": "masks.json", "mode": "background_to_background"},
        )
        self.assertEqual(
            validate_fixed_grid_options(
                "masks.json", "subject_erode2", True, [6, 7], {8},
                local_retention="transition_no_sink", context_mode="replace_recent"),
            {"mask_path": "masks.json", "mode": "subject_erode2"},
        )
        self.assertEqual(
            validate_fixed_grid_options(
                "masks.json", "compact_entity_memory", True, [6, 7], {8},
                local_retention="transition_no_sink", context_mode="replace_recent"),
            {"mask_path": "masks.json", "mode": "compact_entity_memory"},
        )
        self.assertEqual(
            validate_fixed_grid_options(
                "masks.json", "latent_subject_patch", True, [6, 7], {8},
                local_retention="transition_no_sink", context_mode="replace_recent"),
            {"mask_path": "masks.json", "mode": "latent_subject_patch"},
        )

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

    def test_grouped_selective_attention_interpolates_only_historical_branch(self):
        calls = []

        def cpu_attention(query, key, value):
            calls.append(key.shape[1])
            return query + value.sum(dim=1, keepdim=True)

        query = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1)
        base_key = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1)
        base_value = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1)
        group = {
            "query_indices": torch.tensor([1]),
            "historical_key": torch.tensor([1000.0]).view(1, 1, 1, 1),
            "historical_value": torch.tensor([100.0]).view(1, 1, 1, 1),
        }

        baseline = grouped_selective_attention(
            query, base_key, base_value, [{**group, "alpha": 0.0}], attention_fn=cpu_attention)
        alpha_one = grouped_selective_attention(
            query, base_key, base_value, [{**group, "alpha": 1.0}], attention_fn=cpu_attention)
        alpha_quarter = grouped_selective_attention(
            query, base_key, base_value, [{**group, "alpha": 0.25}], attention_fn=cpu_attention)

        self.assertEqual(baseline[:, :, 0, 0].tolist(), [[30.0, 31.0, 32.0]])
        self.assertEqual(alpha_one[:, :, 0, 0].tolist(), [[30.0, 132.0, 32.0]])
        self.assertEqual(alpha_quarter[:, :, 0, 0].tolist(), [[30.0, 56.25, 32.0]])
        self.assertEqual(calls, [2, 2, 1, 2, 1])

    def test_grouped_selective_attention_rejects_duplicate_indices_within_a_group(self):
        calls = []

        def cpu_attention(query, key, value):
            calls.append(key.shape[1])
            return query

        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            grouped_selective_attention(
                torch.zeros(1, 2, 1, 1),
                torch.zeros(1, 1, 1, 1),
                torch.zeros(1, 1, 1, 1),
                [{
                    "query_indices": torch.tensor([1, 1]),
                    "historical_key": torch.zeros(1, 1, 1, 1),
                    "historical_value": torch.zeros(1, 1, 1, 1),
                }],
                attention_fn=cpu_attention,
            )
        self.assertEqual(calls, [1])

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

    def test_temporal_only_historical_rope_has_no_spatial_coordinate(self):
        angles = torch.tensor([
            [0, 1, 0, 1, 0, 1],
            [10, 11, 20, 21, 30, 31],
            [40, 41, 50, 51, 60, 61],
        ], dtype=torch.float64)
        freqs = torch.polar(torch.ones_like(angles), angles)
        key = torch.tensor([1.0, 0.0] * 6, dtype=torch.float32).view(1, 1, 1, 12).repeat(1, 2, 1, 1)

        output = temporal_only_historical_rope(key, freqs, torch.tensor([1, 2]))
        complex_output = torch.view_as_complex(output.to(torch.float64).reshape(1, 2, 1, 6, 2))

        self.assertTrue(torch.allclose(
            complex_output[0, 0, 0], torch.polar(torch.ones(6, dtype=torch.float64), torch.tensor([10, 11, 0, 0, 0, 0], dtype=torch.float64))))
        self.assertTrue(torch.allclose(
            complex_output[0, 1, 0], torch.polar(torch.ones(6, dtype=torch.float64), torch.tensor([40, 41, 0, 0, 0, 0], dtype=torch.float64))))


if __name__ == "__main__":
    unittest.main()
