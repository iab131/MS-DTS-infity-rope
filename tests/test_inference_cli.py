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


if __name__ == "__main__":
    unittest.main(verbosity=2)
