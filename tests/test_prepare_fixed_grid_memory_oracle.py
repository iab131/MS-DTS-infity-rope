#!/usr/bin/env python3
import unittest
from pathlib import Path

from pipeline.fixed_grid_memory_masks import FixedGridMemoryMasks
from scripts.prepare_fixed_grid_memory_oracle import build_mask_audit


class PrepareFixedGridMemoryOracleTest(unittest.TestCase):
    def test_audit_records_exact_source_and_expanded_target_provenance(self):
        root = Path(__file__).resolve().parents[1]
        masks = FixedGridMemoryMasks.from_json(
            root / "docs/attention_memory_policy_fixed_grid_masks_20260809.json")

        audit = build_mask_audit(masks)

        self.assertEqual(audit["source_history"]["6"]["temporal_slot"], 1)
        self.assertEqual(audit["source_history"]["7"]["temporal_slot"], 2)
        self.assertEqual(
            audit["source_history"]["6"]["subject_token_count"],
            len(audit["source_history"]["6"]["subject_row_col_coordinates"]),
        )
        subject = masks.subject_query_indices()
        background = masks.background_query_indices()
        self.assertEqual(audit["target"]["subject_query_count"], 3 * len(subject))
        self.assertEqual(audit["target"]["background_query_count"], 3 * len(background))
        self.assertEqual(
            audit["target"]["subject_query_indices"][-len(subject):],
            [2 * 1560 + index for index in subject],
        )
        self.assertEqual(
            audit["base_context"]["ordering"],
            ["sink:18", "local:19", "local:20", "current:21", "current:22", "current:23"],
        )
        self.assertEqual(audit["base_context"]["derived_from"], {
            "current_start_frame": 21,
            "context_non_sink_frames": 2,
            "current_num_frames": 3,
        })
        self.assertTrue(audit["base_context"]["local_current_order_unchanged"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
