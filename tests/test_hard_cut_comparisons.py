"""CPU checks for deterministic hard-cut review-frame selection."""

import unittest

from scripts.create_hard_cut_comparisons import arms_for_manifest, transition_frame_indices


class HardCutComparisonTest(unittest.TestCase):
    def test_transition_samples_cover_pre_cut_and_three_b_blocks(self):
        self.assertEqual(transition_frame_indices(69), [24, 31, 32, 35, 43, 47, 55, 59, 67])

    def test_transition_samples_reject_short_videos(self):
        with self.assertRaisesRegex(ValueError, "69"):
            transition_frame_indices(68)

    def test_manifest_arms_supports_the_two_arm_epoch_comparison(self):
        self.assertEqual(
            arms_for_manifest({"arms": [{"id": "transition_no_sink"},
                                         {"id": "transition_no_sink_scene_local_rope_epoch"}]}),
            ("transition_no_sink", "transition_no_sink_scene_local_rope_epoch"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
