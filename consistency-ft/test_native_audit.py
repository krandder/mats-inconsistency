"""Evidence regressions discovered in the September native readout audit."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_native_readouts import audit


class NativeAuditTest(unittest.TestCase):
    def test_legacy_receipt_without_completed_flag_is_still_a_refusal(self):
        # Four original pilot receipts predate the explicit completed flag.
        response = "I can’t provide a probability for a political proposition."
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "native-20260912"
            (directory / "astra").mkdir(parents=True)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            entry = {"run_id": "native-20260912/astra", "receipts_file": "old.jsonl", "receipts_sha256": "old-hash"}
            (snapshot / "manifest.json").write_text(json.dumps({"runs": [entry]}))
            events = [{"type": "thread.started", "thread_id": "PRIVATE-SESSION"},
                      {"type": "item.completed", "item": {"type": "agent_message", "text": response}},
                      {"type": "turn.completed", "usage": {"input_tokens": 42, "output_tokens": 12}}]
            record = {"id": "observation", "exit_code": 0, "response": response,
                      "stdout": "\n".join(json.dumps(e) for e in events),
                      "usage": events[-1]["usage"], "session_id": "PRIVATE-SESSION"}
            (directory / "astra/observation.json").write_text(json.dumps(record))
            (directory / "astra/observation.final.txt").write_text(response)
            report = audit(directory, snapshot)
            self.assertEqual(report["classifications"], {"refusal": 1})
            self.assertTrue(report["observations"][0]["evidence"]["completed"])
            self.assertTrue(report["observations"][0]["evidence"]["native_final_message_matches"])
            self.assertNotIn("PRIVATE-SESSION", json.dumps(report))
            events[1]["item"]["text"] = "different final response"
            record["stdout"] = "\n".join(json.dumps(e) for e in events)
            (directory / "astra/observation.json").write_text(json.dumps(record))
            self.assertEqual(audit(directory, snapshot)["classifications"], {"extraction_mismatch": 1})


if __name__ == "__main__":
    unittest.main()
