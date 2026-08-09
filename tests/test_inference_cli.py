import subprocess
import sys
import unittest
from pathlib import Path


class InferenceCliTest(unittest.TestCase):
    def test_snapshot_flags_are_exposed(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "inference.py", "--help"], cwd=root,
            text=True, capture_output=True, check=True)
        self.assertIn("--save-clean-latent-blocks", result.stdout)
        self.assertIn("--save-raw-decoded", result.stdout)
        self.assertIn("--attention-memory-policy", result.stdout)
        self.assertIn("--memory-retrieval-lifetime", result.stdout)
        self.assertIn("--memory-context-mode", result.stdout)
        self.assertIn("--memory-descriptor-layers", result.stdout)
        self.assertIn("--memory-injection-layers", result.stdout)
        self.assertIn("--memory-local-retention", result.stdout)
        self.assertIn("transition_no_sink", result.stdout)
        self.assertIn("--memory-transition-auto-retrieval", result.stdout)
        self.assertIn("--memory-fixed-grid-mask-path", result.stdout)
        self.assertIn("--memory-fixed-grid-mode", result.stdout)
        self.assertIn("subject_erode1", result.stdout)
        self.assertIn("subject_erode2", result.stdout)
        self.assertIn("subject_boundary_only", result.stdout)

    def test_fixed_grid_cli_flags_must_be_paired(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "inference.py", "--memory-fixed-grid-mask-path", "masks.json"],
            cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be provided together", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
