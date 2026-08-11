"""CPU proof checks for the Phase-2A scene-local RoPE question."""

import unittest

from scripts.scene_local_rope_equivalence import coordinate_trace, run_equivalence_probe


class SceneLocalRopeEquivalenceTest(unittest.TestCase):
    def test_trace_matches_live_no_sink_cache_and_cut_behavior(self):
        trace = coordinate_trace()

        self.assertEqual(trace["first_B"]["current_qk_positions"], [45, 46, 47])
        self.assertEqual(trace["first_B"]["scene_local_qk_positions"], [0, 1, 2])
        self.assertEqual(trace["second_B"]["current_key_positions"], [45, 1, 2, 3, 4, 5])
        self.assertEqual(trace["second_B"]["scene_local_key_positions"], [0, 1, 2, 3, 4, 5])
        self.assertFalse(trace["second_B"]["scene_cut"])

    def test_probe_keeps_first_block_attention_but_changes_future_attention(self):
        report = run_equivalence_probe(seed=7)

        self.assertLess(report["first_B_denoise"]["attention_output_max_abs"], 1e-10)
        self.assertLess(report["first_B_clean_pass"]["attention_logits_max_abs"], 1e-10)
        self.assertGreater(report["clean_pass_sink_key"]["max_abs"], 1e-4)
        self.assertGreater(report["second_B_full_context"]["attention_logits_max_abs"], 1e-4)
        self.assertGreater(report["second_B_raw_non_sink_only"]["attention_logits_max_abs"], 1e-4)
        self.assertGreater(report["third_B_full_context"]["attention_output_max_abs"], 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
