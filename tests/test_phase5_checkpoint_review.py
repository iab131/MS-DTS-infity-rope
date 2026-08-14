import unittest
from pathlib import Path
import subprocess
import sys


from scripts.create_phase5_checkpoint_review import (
    blinded_arm_order, build_review_rows, phase5_boundary_windows,
)


class Phase5CheckpointReviewTest(unittest.TestCase):
    def test_review_script_runs_directly(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "scripts/create_phase5_checkpoint_review.py", "--help"],
                                cwd=root, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Create anonymized", result.stdout)

    def test_each_boundary_window_has_two_pre_and_three_later_blocks(self):
        windows = phase5_boundary_windows(285, [46, 94, 142, 190, 238])

        self.assertEqual(windows[0], (43, 93))
        self.assertEqual(windows[-1], (235, 285))

    def test_blinded_order_is_deterministic_but_hides_real_arm_names(self):
        order = blinded_arm_order("human_to_object", 101,
                                  ("live_infinity_rope", "always_reset", "native_state_rebinding"))

        self.assertEqual(set(order), {"live_infinity_rope", "always_reset", "native_state_rebinding"})
        self.assertEqual(order, blinded_arm_order("human_to_object", 101,
                                                  ("live_infinity_rope", "always_reset", "native_state_rebinding")))

    def test_review_rows_are_blank_and_anonymous(self):
        manifest = {
            "pairs": [{"id": "case", "transition_raw_frames": [46, 94],
                       "boundary_after": ["|", "#"]}],
            "seeds": [101],
            "arms": [{"id": "live_infinity_rope"}, {"id": "always_reset"},
                     {"id": "native_state_rebinding"}],
        }
        rows, mapping = build_review_rows(manifest)

        self.assertEqual(len(rows), 6)
        self.assertEqual(set(mapping["case__seed101"].values()), {
            "live_infinity_rope", "always_reset", "native_state_rebinding"})
        self.assertNotIn("arm_id", rows[0])
        self.assertEqual(rows[0]["score_1_to_5"], "")
        self.assertNotIn("live_infinity_rope", str(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
