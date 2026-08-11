"""CPU checks for deterministic hard-cut review-frame selection."""

import unittest

from scripts.create_hard_cut_comparisons import transition_frame_indices


class HardCutComparisonTest(unittest.TestCase):
    def test_transition_samples_cover_pre_cut_and_three_b_blocks(self):
        self.assertEqual(transition_frame_indices(69), [24, 31, 32, 35, 43, 47, 55, 59, 67])

    def test_transition_samples_reject_short_videos(self):
        with self.assertRaisesRegex(ValueError, "69"):
            transition_frame_indices(68)


if __name__ == "__main__":
    unittest.main(verbosity=2)
