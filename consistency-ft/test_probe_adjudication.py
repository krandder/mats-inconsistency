"""Offline adjudication integrity tests using invented coin/label fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import adjudicate_native_probes as review


class AdjudicationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.audit = {"schema_version": 1, "catalog_sha256": "c" * 64, "probes": [], "source_captures": []}
        self.study = {"schema_version": 1, "publication_status": "COMPLETE", "catalog_count": 129, "configuration_count": 9,
                      "completed_observations": 1161, "catalog_sha256": "c" * 64, "configurations": []}
        for n in range(129):
            p = {"question": f"Which label appears on synthetic card {n}?", "answer": "Blue", "accept": ["blue"]}
            source = {"probe_id": review.sha(str(n).encode()), "catalog_index": n + 1, "original_probe": p,
                      "answer_evidence_level": "synthetic-fixture", "core_answer_status": "synthetic-fixture",
                      "evidence": [{"capture_id": f"capture-{n}", "url": f"https://example.test/card/{n}",
                                    "excerpt": "Synthetic card's printed label: Blue."}],
                      "date_flags": [], "prompt_flags": [], "grading_flags": [], "proposed_corrections": []}
            self.audit["probes"].append(source)
        for i, (provider, effort) in enumerate(review.SETTINGS):
            rows = []
            for n, source in enumerate(self.audit["probes"]):
                response = ("Blue label" if i == 8 else "Blue") if n == 0 else "Green" if n == 1 else "UNKNOWN"
                grade = "exact_alias_match" if response == "Blue" else "explicit_unknown" if response == "UNKNOWN" else "manual_review"
                rows.append({"probe_id": source["probe_id"], "catalog_index": n + 1,
                             "original_probe": source["original_probe"], "response": response,
                             "original_key_grade": {"status": grade}})
            self.study["configurations"].append({"provider": provider, "reasoning": effort,
                "configuration": provider + "/" + effort, "model": review.MODEL_IDS[provider],
                "completed_observations": 129, "full_catalog_completed": True,
                "unreconciled_launches": 0, "probes": rows})
        self.write("audit.json", self.audit)
        self.study["source_audit_sha256"] = self.hash("audit.json")
        self.write("study.json", self.study)
        self.rubric = {"schema_version": 1, "study_sha256": self.hash("study.json"), "source_audit_sha256": self.hash("audit.json"),
                       "catalog_sha256": "c" * 64, "allowed_verdicts": sorted(review.VERDICTS)}
        self.write("rubric.json", self.rubric)
        items = {}
        for c in self.study["configurations"]:
            for row in c["probes"]:
                if row["response"] == "UNKNOWN":
                    continue
                source = self.audit["probes"][row["catalog_index"] - 1]
                iid = review.sha(review.canonical([row["probe_id"], row["response"]]))
                items[iid] = {"item_id": iid, "probe_id": row["probe_id"], "catalog_index": row["catalog_index"],
                              "question": row["original_probe"]["question"], "response": row["response"],
                              "reference_answer": row["original_probe"]["answer"],
                              "source_review": {k: source[k] for k in ("answer_evidence_level", "core_answer_status",
                                  "evidence", "date_flags", "prompt_flags", "grading_flags", "proposed_corrections")}}
        self.packet = {"schema_version": 1, "rubric_sha256": self.hash("rubric.json"),
                       "study_sha256": self.hash("study.json"), "source_audit_sha256": self.hash("audit.json"),
                       "items": [v for _, v in sorted(items.items())]}
        self.write("packet.json", self.packet)
        self.decisions = [{"item_id": item["item_id"],
                           "verdict": "contradicted_core_answer" if item["response"] == "Green" else "supported_core_answer",
                           "evidence_capture_ids": [item["source_review"]["evidence"][0]["capture_id"]],
                           "rationale": "Synthetic fixture source specifies the card's printed label.", "qualifications": []}
                          for item in self.packet["items"]]
        for label in ("a", "b"):
            self.write_review(label, self.decisions)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def hash(self, name):
        return review.sha((self.root / name).read_bytes())

    def inputs(self):
        return review.load_inputs(*(self.root / name for name in ("study.json", "audit.json", "rubric.json", "packet.json")))

    def write_review(self, label, decisions):
        self.write(label + ".json", {"schema_version": 1, "reviewer": "review-" + label,
            "rubric_sha256": self.hash("rubric.json"), "packet_sha256": self.hash("packet.json"),
            "reviews": decisions})

    def build(self, resolution=None):
        return review.build(self.inputs(), self.root / "a.json", self.root / "b.json", resolution)

    def resolution(self, decisions, **updates):
        doc = {"schema_version": 1, "rubric_sha256": self.hash("rubric.json"),
               "review_a_sha256": self.hash("a.json"), "review_b_sha256": self.hash("b.json"),
               "resolutions": decisions, "supplemental_evidence_sha256": None, **updates}
        self.write("resolution.json", doc)
        return self.root / "resolution.json"

    def test_complete_unique_pairs_shared_across_models_preserve_raw_grades(self):
        result = self.build()
        self.assertEqual((result["observations"], result["unique_pairs"], result["manual_pairs"]), (1161, 130, 3))
        self.assertEqual(result["mechanical_unknown_pairs"], 127)
        self.assertTrue(all(c["verdict_counts"] == {"supported_core_answer": 1, "contradicted_core_answer": 1, "abstained": 127}
                            for c in result["configurations"]))
        self.assertEqual(result["configurations"][-1]["original_unmatched_now_supported"], 1)
        self.assertEqual(result["configurations"][-1]["probes"][0]["original_key_status"], "manual_review")
        self.assertEqual(result["configurations"][0]["probes"][0]["item_id"], result["configurations"][1]["probes"][0]["item_id"])
        self.assertIsNone(result["verified_cutoff"])
        self.assertFalse(result["original_key_changed"])

    def test_mutated_study_source_or_rubric_bindings_fail(self):
        for name, data in (("study.json", self.study), ("audit.json", self.audit)):
            with self.subTest(name=name):
                self.write(name, {**data, "extra": "changed bytes"})
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    self.inputs()
                self.write(name, data)

    def test_packet_cannot_add_model_labels_or_alter_response(self):
        for field, value in (("model", "secret-label"), ("response", "Red")):
            packet = copy.deepcopy(self.packet)
            packet["items"][0][field] = value
            self.write("packet.json", packet)
            with self.assertRaisesRegex(ValueError, "packet"):
                self.inputs()

    def test_missing_or_duplicate_manual_decisions_rejected(self):
        for rows in (self.decisions[:-1], self.decisions + self.decisions[:1]):
            self.write_review("b", rows)
            with self.assertRaises(ValueError):
                self.build()

    def test_unknown_verdict_wrong_source_and_missing_source_rejected(self):
        for update in ({"verdict": "correct"}, {"evidence_capture_ids": ["capture-128"]}, {"evidence_capture_ids": []}):
            decisions = copy.deepcopy(self.decisions)
            decisions[0].update(update)
            self.write_review("b", decisions)
            with self.assertRaises(ValueError):
                self.build()

    def test_independent_disagreement_cannot_be_silently_published(self):
        other = copy.deepcopy(self.decisions)
        other[0]["verdict"] = "unresolved"
        self.write_review("b", other)
        with self.assertRaisesRegex(ValueError, "disagreements lack"):
            self.build()
        result = self.build(self.resolution([other[0]]))
        self.assertEqual(result["independent_verdict_disagreements"], 1)
        unit = next(u for u in result["units"] if u["item_id"] == other[0]["item_id"])
        self.assertEqual(unit["origin"], "resolved_disagreement")
        self.assertEqual(unit["decision"]["verdict"], "unresolved")
        self.assertEqual([r["verdict"] for r in unit["independent_reviews"]], [self.decisions[0]["verdict"], "unresolved"])

    def test_resolution_must_bind_to_both_review_files(self):
        with self.assertRaisesRegex(ValueError, "Resolution does not bind"):
            self.build(self.resolution([], review_a_sha256="0" * 64))

    def test_agreement_override_is_visible_and_original_reviews_retained(self):
        changed = {**self.decisions[0], "verdict": "incomplete_core_answer", "rationale": "Documented synthetic correction."}
        result = self.build(self.resolution([changed]))
        self.assertEqual(result["documented_overrides_of_agreements"], 1)
        unit = next(u for u in result["units"] if u["item_id"] == changed["item_id"])
        self.assertEqual(unit["origin"], "documented_review_override")
        self.assertEqual(len(unit["independent_reviews"]), 2)

    def test_review_identity_and_packet_digest_are_required(self):
        doc = json.loads((self.root / "b.json").read_text())
        doc["packet_sha256"] = "0" * 64
        self.write("b.json", doc)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.build()

    def test_private_paths_empty_rationale_and_malformed_qualifications_rejected(self):
        for update in ({"rationale": ""}, {"rationale": "See /home/agent/private"}, {"qualifications": [{}]},
                       {"rationale": "session_id=private-value"}, {"rationale": "account_name=private-value"},
                       {"rationale": "Read /tmp/private-result"}):
            rows = copy.deepcopy(self.decisions)
            rows[0].update(update)
            self.write_review("b", rows)
            with self.assertRaises(ValueError):
                self.build()

    def test_duplicate_configuration_exact_model_and_recorded_audit_hash_rejected(self):
        for field, value in (("configuration", "astra/low"), ("model", "unrelated-model")):
            study = copy.deepcopy(self.study)
            study["configurations"][1][field] = value
            self.write("study.json", study)
            rubric = {**self.rubric, "study_sha256": self.hash("study.json")}
            self.write("rubric.json", rubric)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.inputs()
        study = {**self.study, "source_audit_sha256": "0" * 64}
        self.write("study.json", study)
        self.write("rubric.json", {**self.rubric, "study_sha256": self.hash("study.json")})
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            self.inputs()

    def test_duplicate_keys_and_nonfinite_json_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}'):
            (self.root / "bad.json").write_text(text)
            with self.assertRaises(ValueError):
                review.read(self.root / "bad.json")

    def test_markdown_escapes_response_and_preserves_full_denominator(self):
        report = self.build()
        report["units"][0]["response"] = "<script>alert(1)</script> | injected\nrow"
        text = review.markdown(report)
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("\\| injected<br>row", text)
        self.assertIn("1,161 observations", text)
        self.assertIn("not human adjudication", text)

    def test_supplemental_evidence_requires_retained_hashes_and_supporting_excerpt(self):
        item = self.packet["items"][0]
        source = b"Synthetic official card record: the full label is Blue."
        (self.root / "extra.body").write_bytes(source)
        (self.root / "extra.txt").write_bytes(source)
        entry = {"probe_id": item["probe_id"], "capture_id": "extra", "url": "https://example.test/official-card",
                 "captured_at": "2026-09-13T00:00:00+00:00", "body_sha256": review.sha(source), "text_sha256": review.sha(source),
                 "excerpt": "the full label is Blue", "excerpt_text_normalization": "consecutive whitespace collapsed to one space",
                 "claim_supported": "Synthetic fixture only.", "publication_date": "2026-01-01", "limitations": []}
        self.write("supplement.json", {"schema_version": 1, "evidence": [entry]})
        inputs = self.inputs()
        with self.assertRaisesRegex(ValueError, "raw captures"):
            review.load_supplemental(self.root / "supplement.json", inputs, None)
        evidence, digest = review.load_supplemental(self.root / "supplement.json", inputs, self.root)
        self.assertEqual(evidence[item["probe_id"]][0]["capture_id"], "extra")
        decision = {**self.decisions[0], "evidence_capture_ids": ["extra"], "rationale": "Extra synthetic record resolves the label."}
        path = self.resolution([decision], supplemental_evidence_sha256=digest)
        result = review.build(inputs, self.root / "a.json", self.root / "b.json", path,
                              supplemental_path=self.root / "supplement.json", supplemental_capture_dir=self.root)
        self.assertEqual(result["supplemental_evidence_sha256"], digest)
        self.assertEqual(result["documented_overrides_of_agreements"], 1)
        (self.root / "extra.txt").write_text("Tampered source text")
        with self.assertRaisesRegex(ValueError, "capture hash mismatch"):
            review.load_supplemental(self.root / "supplement.json", inputs, self.root)

    def test_supplemental_excerpt_and_unrelated_probe_cannot_be_fabricated(self):
        source = b"Blue label."
        (self.root / "extra.body").write_bytes(source)
        (self.root / "extra.txt").write_bytes(source)
        entry = {"probe_id": self.packet["items"][0]["probe_id"], "capture_id": "extra", "url": "https://example.test/record",
                 "captured_at": "2026-09-13T00:00:00+00:00", "body_sha256": review.sha(source), "text_sha256": review.sha(source),
                 "excerpt": "Red label", "excerpt_text_normalization": "consecutive whitespace collapsed to one space",
                 "claim_supported": "Fixture.", "publication_date": "2026-01-01", "limitations": []}
        self.write("supplement.json", {"schema_version": 1, "evidence": [entry]})
        with self.assertRaisesRegex(ValueError, "excerpt absent"):
            review.load_supplemental(self.root / "supplement.json", self.inputs(), self.root)
        entry.update(probe_id="0" * 64, excerpt="Blue label")
        self.write("supplement.json", {"schema_version": 1, "evidence": [entry]})
        with self.assertRaisesRegex(ValueError, "probe binding mismatch"):
            review.load_supplemental(self.root / "supplement.json", self.inputs(), self.root)


if __name__ == "__main__":
    unittest.main()
