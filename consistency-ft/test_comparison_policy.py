"""Comparison policy is export metadata, independent of frozen question scopes."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_benchmark import refresh


class ComparisonPolicyRefreshTests(unittest.TestCase):
    def test_refresh_preserves_policy_until_an_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tuples = root / "data/tuples"
            tuples.mkdir(parents=True)
            (tuples / "Negation.jsonl").write_text(json.dumps({
                "id": "fixture", "check_type": "Negation",
                "questions": {"P": {"title": "A spacecraft lands."}, "not_P": {"title": "No spacecraft lands."}},
            }) + "\n")
            snapshots = root / "snapshots"
            initial = refresh(root, snapshots)
            self.assertNotIn("comparison_policy", initial)
            selected = refresh(root, snapshots, comparison_policy="complete-historical-plus-native")
            self.assertEqual(selected["comparison_policy"], "complete-historical-plus-native")
            preserved = refresh(root, snapshots)
            self.assertEqual(preserved["comparison_policy"], "complete-historical-plus-native")
            overridden = refresh(root, snapshots, comparison_policy="all-settings")
            self.assertEqual(overridden["comparison_policy"], "all-settings")
            unchanged = (snapshots / "manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "Unknown comparison policy"):
                refresh(root, snapshots, comparison_policy="typo")
            self.assertEqual((snapshots / "manifest.json").read_bytes(), unchanged)


if __name__ == "__main__":
    unittest.main()
