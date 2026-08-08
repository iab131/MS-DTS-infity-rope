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
        self.assertIn("--memory-context-mode", result.stdout)
        self.assertIn("--memory-descriptor-layers", result.stdout)
        self.assertIn("--memory-injection-layers", result.stdout)
        self.assertIn("--memory-local-retention", result.stdout)
        self.assertIn("--memory-transition-auto-retrieval", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
