#!/usr/bin/env python3
"""CPU checks for the declarative Phase-0 hard-cut benchmark."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pipeline.causal_inference import CausalInferencePipeline
from scripts.hard_cut_transition_benchmark import (
    _gpu_memory_for_pid, build_run_rows, execute_rows, load_manifest,
)


class HardCutTransitionBenchmarkTest(unittest.TestCase):
    def test_manifest_expands_four_pairs_two_seeds_and_four_live_arms(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] / "docs/HARD_CUT_BENCHMARK_PHASE0_20260810.json")
        rows = build_run_rows(manifest)

        self.assertEqual(len(rows), 32)
        self.assertEqual({row["arm_id"] for row in rows}, {
            "live_kv_flush", "sink_plus1", "sink_only", "transition_no_sink"})
        self.assertEqual({row["seed"] for row in rows}, {101, 202})
        self.assertTrue(all(row["first_b_block"] == 4 for row in rows))
        self.assertTrue(all(row["first_b_raw_frame"] == 33 for row in rows))
        self.assertTrue(all("new_prompt_adherence" in row["review_fields"] for row in rows))

    def test_live_baseline_uses_kv_flush_and_policy_arms_only_change_retention(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] / "docs/HARD_CUT_BENCHMARK_PHASE0_20260810.json")
        rows = {row["arm_id"]: row for row in build_run_rows(manifest) if row["pair_id"] == "greenhouse_to_pickup" and row["seed"] == 101}

        self.assertNotIn("--attention-memory-policy", rows["live_kv_flush"]["command"])
        self.assertIn("--data_path", rows["live_kv_flush"]["command"])
        for arm_id, retention in (("sink_plus1", "sink+1"), ("sink_only", "sink_only"),
                                  ("transition_no_sink", "transition_no_sink")):
            command = rows[arm_id]["command"]
            self.assertEqual(command[command.index("--memory-local-retention") + 1], retention)

    def test_commands_take_matched_settings_from_the_manifest(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] / "docs/HARD_CUT_BENCHMARK_PHASE0_20260810.json")
        manifest["matched_settings"]["config_path"] = "configs/test.yaml"
        manifest["matched_settings"]["num_samples"] = 3
        manifest["matched_settings"]["memory_crossattn_reset"] = False
        policy_command = next(row["command"] for row in build_run_rows(manifest)
                              if row["arm_id"] == "sink_only")

        self.assertIn("configs/test.yaml", policy_command)
        self.assertEqual(policy_command[policy_command.index("--num_samples") + 1], "3")
        self.assertIn("--no-memory-crossattn-reset", policy_command)

    def test_missing_gpu_pid_is_unavailable_not_zero_memory(self):
        with patch("scripts.hard_cut_transition_benchmark.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout="999, 123\n")):
            self.assertIsNone(_gpu_memory_for_pid(1234))

    def test_phase2b_manifest_expands_the_eight_matched_epoch_runs(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] /
            "docs/HARD_CUT_SCENE_LOCAL_ROPE_EPOCH_PHASE2B_20260810.json")
        rows = build_run_rows(manifest)

        self.assertEqual(len(rows), 8)
        epoch = next(row for row in rows if row["arm_id"] == "transition_no_sink_scene_local_rope_epoch")
        control = next(row for row in rows if row["arm_id"] == "transition_no_sink")
        self.assertIn("--scene-local-rope-epoch", epoch["command"])
        self.assertNotIn("--scene-local-rope-epoch", control["command"])

    def test_phase3a_normal_boundary_manifest_uses_no_rope_cut_marker(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] /
            "docs/SAME_SCENE_ACTION_TRANSITION_PHASE3A_20260811.json")
        rows = build_run_rows(manifest)

        self.assertEqual(len(rows), 8)
        self.assertEqual({row["arm_id"] for row in rows},
                         {"live_kv_flush", "transition_no_sink"})
        self.assertTrue(all("[2.25s#]" not in row["prompt"] for row in rows))
        self.assertTrue(all("[2.25s] |" in row["prompt"] for row in rows))
        live = next(row for row in rows if row["arm_id"] == "live_kv_flush")
        reset = next(row for row in rows if row["arm_id"] == "transition_no_sink")
        self.assertNotIn("--attention-memory-policy", live["command"])
        self.assertEqual(reset["command"][reset["command"].index("--memory-local-retention") + 1],
                         "transition_no_sink")

    def test_phase3b_factorials_expand_four_arms_and_mark_exact_reuse(self):
        root = Path(__file__).resolve().parents[1]
        hard = load_manifest(root / "docs/HARD_CUT_STATE_RETENTION_FACTORIAL_PHASE3B_20260811.json")
        normal = load_manifest(root / "docs/SAME_SCENE_STATE_RETENTION_FACTORIAL_PHASE3B_20260811.json")
        hard_rows, normal_rows = build_run_rows(hard), build_run_rows(normal)

        self.assertEqual(len(hard_rows), 16)
        self.assertEqual(len(normal_rows), 16)
        for rows in (hard_rows, normal_rows):
            self.assertEqual({row["arm_id"] for row in rows}, {
                "live_kv_flush", "sink_only", "recent_only_no_sink", "transition_no_sink"})
            self.assertEqual(sum("reuse_ledger" in row for row in rows),
                             12 if rows is hard_rows else 8)
        recent = next(row for row in hard_rows if row["arm_id"] == "recent_only_no_sink")
        self.assertEqual(recent["command"][recent["command"].index("--memory-local-retention") + 1],
                         "recent_only_no_sink")

    def test_phase3c_mixed_boundaries_use_only_the_opt_in_policy_arm(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] /
            "docs/MIXED_BOUNDARY_STATE_LIFETIME_PHASE3C_20260812.json")
        rows = build_run_rows(manifest)

        self.assertEqual(len(rows), 12)
        self.assertEqual({row["arm_id"] for row in rows}, {
            "live_kv_flush", "always_reset", "boundary_conditioned"})
        conditioned = next(row for row in rows if row["arm_id"] == "boundary_conditioned")
        self.assertIn("--boundary-conditioned-ar-state", conditioned["command"])
        self.assertNotIn("--attention-memory-policy", conditioned["command"])
        self.assertIn("[2.25s] |", conditioned["prompt"])
        self.assertIn("[2.25s#] |", conditioned["prompt"])
        self.assertEqual(conditioned["transition_blocks"], [4, 7, 10])

    def test_phase5_checkpoint_is_a_fixed_63_cell_six_segment_matrix(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] /
            "docs/PHASE5_GENERALIZATION_CHECKPOINT_20260813.json")
        rows = build_run_rows(manifest)

        self.assertEqual(len(manifest["categories"]), 7)
        self.assertEqual(len(manifest["pairs"]), 7)
        self.assertEqual(manifest["seeds"], [101, 202, 303])
        self.assertEqual(len(rows), 63)
        self.assertEqual({row["arm_id"] for row in rows}, {
            "live_infinity_rope", "always_reset", "native_state_rebinding"})
        self.assertTrue(all(row["transition_blocks"] == [5, 9, 13, 17, 21]
                            for row in rows))
        self.assertTrue(all(row["transition_raw_frames"] == [49, 97, 145, 193, 241]
                            for row in rows))

    def test_phase5_arm_commands_preserve_or_invalidate_native_state_only_at_expected_boundaries(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] /
            "docs/PHASE5_GENERALIZATION_CHECKPOINT_20260813.json")
        rows = {row["arm_id"]: row for row in build_run_rows(manifest)
                if row["pair_id"] == "human_to_object" and row["seed"] == 101}

        self.assertNotIn("--attention-memory-policy", rows["live_infinity_rope"]["command"])
        reset = rows["always_reset"]["command"]
        self.assertEqual(reset[reset.index("--memory-local-retention") + 1], "transition_no_sink")
        self.assertIn("--boundary-conditioned-ar-state", rows["native_state_rebinding"]["command"])
        self.assertNotIn("--attention-memory-policy", rows["native_state_rebinding"]["command"])
        for row in rows.values():
            self.assertIn("[3.0s] |", row["prompt"])
            self.assertEqual(row["prompt"].count("[3.0s#]"), 2)

    def test_phase5_prompts_parse_to_six_four_block_scenes_with_only_hard_cuts_after_a2_b2(self):
        manifest = load_manifest(
            Path(__file__).resolve().parents[1] /
            "docs/PHASE5_GENERALIZATION_CHECKPOINT_20260813.json")
        parser_owner = SimpleNamespace(default_fps=16, num_frame_per_block=3,
                                       default_blocks_per_scene=14)

        for pair in manifest["pairs"]:
            prompt = (Path(__file__).resolve().parents[1] / pair["prompt_path"]).read_text(encoding="utf-8")
            scenes, blocks, cuts = CausalInferencePipeline._parse_scene_durations(parser_owner, prompt)
            self.assertEqual(len(scenes), 6)
            self.assertEqual(blocks, [4] * 6)
            self.assertEqual(cuts, [False, True, False, True, False, False])

    def test_execution_keeps_a_sampled_peak_after_the_process_exits(self):
        row = build_run_rows(load_manifest(
            Path(__file__).resolve().parents[1] / "docs/HARD_CUT_BENCHMARK_PHASE0_20260810.json"))[0]
        process = Mock(pid=4321, returncode=0)
        process.poll.side_effect = [None, 0]
        with patch("scripts.hard_cut_transition_benchmark.subprocess.Popen", return_value=process), \
             patch("scripts.hard_cut_transition_benchmark._gpu_memory_for_pid", side_effect=[123, None]), \
             patch("scripts.hard_cut_transition_benchmark.time.sleep"), \
             patch("scripts.hard_cut_transition_benchmark.time.monotonic", side_effect=[0, 1]):
            result = execute_rows([row])

        self.assertEqual(result[0]["peak_vram_mib"], 123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
