"""Synthetic-only reviewed-label joins, original-byte preservation and HTML safety."""
import copy
from collections import Counter
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import adjudicate_native_probes as adjudication
from scripts import export_native_probes as exporter
from scripts import render_probe_adjudication as renderer
import test_native_probe_export as native_fixture


class ProbeAdjudicationRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_tmp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.fixture_tmp.name)
        native_fixture.NativeProbeExportTests.make_fixture(cls.fixture)
        audit = adjudication.read(cls.fixture / "audit.json")[0]
        audit["source_captures"] = [{"capture_id": "synthetic-capture"}]
        cls.write_json(cls.fixture / "audit.json", audit)
        study = exporter.build_export(cls.fixture / "runs", cls.fixture / "audit.json", catalog_path=cls.fixture / "catalog.jsonl")
        cls.write_json(cls.fixture / "study.json", study)
        audit, audit_hash = adjudication.read(cls.fixture / "audit.json")
        study_hash = adjudication.read(cls.fixture / "study.json")[1]
        rubric = {"schema_version": 1, "study_sha256": study_hash, "source_audit_sha256": audit_hash,
                  "catalog_sha256": study["catalog_sha256"], "allowed_verdicts": sorted(adjudication.VERDICTS)}
        cls.write_json(cls.fixture / "rubric.json", rubric)
        rubric_hash = adjudication.read(cls.fixture / "rubric.json")[1]
        sources = {r["probe_id"]: r for r in audit["probes"]}
        items = {}
        for c in study["configurations"]:
            for row in c["probes"]:
                if row["response"] == "UNKNOWN":
                    continue
                iid = adjudication.sha(adjudication.canonical([row["probe_id"], row["response"]]))
                source = sources[row["probe_id"]]
                items[iid] = {"item_id": iid, "probe_id": row["probe_id"], "catalog_index": row["catalog_index"],
                    "question": row["original_probe"]["question"], "response": row["response"], "reference_answer": row["original_probe"]["answer"],
                    "source_review": {k: source[k] for k in renderer.SOURCE_FIELDS}}
        packet = {"schema_version": 1, "rubric_sha256": rubric_hash, "study_sha256": study_hash,
                  "source_audit_sha256": audit_hash, "items": [v for _, v in sorted(items.items())]}
        cls.write_json(cls.fixture / "packet.json", packet)
        packet_hash = adjudication.read(cls.fixture / "packet.json")[1]
        for reviewer in ("review-a", "review-b"):
            rows = [{"item_id": item["item_id"],
                     "verdict": "supported_core_answer" if item["response"] == "North Lab" else "contradicted_core_answer",
                     "evidence_capture_ids": [item["source_review"]["evidence"][0]["capture_id"]],
                     "rationale": reviewer + " inspected the invented widget label.", "qualifications": [reviewer + " fixture-only qualification."]}
                    for item in packet["items"]]
            cls.write_json(cls.fixture / (reviewer + ".json"), {"schema_version": 1, "reviewer": reviewer,
                "rubric_sha256": rubric_hash, "packet_sha256": packet_hash, "reviews": rows})
        inputs = adjudication.load_inputs(*(cls.fixture / p for p in ("study.json", "audit.json", "rubric.json", "packet.json")))
        review = adjudication.build(inputs, cls.fixture / "review-a.json", cls.fixture / "review-b.json")
        cls.write_json(cls.fixture / "review.json", review)
        shutil.rmtree(cls.fixture / "runs")

    @classmethod
    def tearDownClass(cls):
        cls.fixture_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(self.fixture, self.root, dirs_exist_ok=True)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def read(self, name):
        return adjudication.read(self.root / name)[0]

    def write(self, name, value):
        self.write_json(self.root / name, value)

    def render(self, **kwargs):
        kwargs.setdefault("review_a_path", self.root / "review-a.json")
        kwargs.setdefault("review_b_path", self.root / "review-b.json")
        if (self.root / "resolutions.json").exists():
            kwargs.setdefault("resolutions_path", self.root / "resolutions.json")
        return renderer.render(*(self.root / p for p in ("study.json", "review.json", "audit.json", "rubric.json")), **kwargs)

    def add_supplement(self):
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["response"] == "North Lab")
        iid = unit["item_id"]
        for name in ("review-a.json", "review-b.json"):
            doc = self.read(name)
            row = next(r for r in doc["reviews"] if r["item_id"] == iid)
            row.update(verdict="unresolved", rationale="Initial synthetic review left the widget label uncertain.",
                       qualifications=["Initial source packet uncertainty is retained."])
            self.write(name, doc)
        text = b"The new source identifies North Lab as the widget maker."
        cid = "synthetic-supplement"
        capture_dir = self.root / "captures"
        capture_dir.mkdir()
        for suffix in ("body", "txt"):
            (capture_dir / (cid + "." + suffix)).write_bytes(text)
        source = {"probe_id": unit["probe_id"], "capture_id": cid, "url": "https://example.org/new-evidence",
            "captured_at": "2026-09-13T08:00:00Z", "body_sha256": adjudication.sha(text), "text_sha256": adjudication.sha(text),
            "excerpt": text.decode(), "excerpt_text_normalization": "consecutive whitespace collapsed to one space",
            "claim_supported": "The complete invented widget maker identity is North Lab.", "publication_date": "2026-09-13",
            "limitations": ["Supplementary evidence does not rewrite the initial independent reviews."]}
        self.write("supplement.json", {"schema_version": 1, "evidence": [source]})
        hashes = {name: adjudication.read(self.root / name)[1] for name in ("rubric.json", "review-a.json", "review-b.json", "supplement.json")}
        decision = {"item_id": iid, "verdict": "supported_core_answer", "evidence_capture_ids": [cid],
            "rationale": "Documented final review uses the supplementary widget source.", "qualifications": ["Core identity only."]}
        self.write("resolutions.json", {"schema_version": 1, "rubric_sha256": hashes["rubric.json"],
            "review_a_sha256": hashes["review-a.json"], "review_b_sha256": hashes["review-b.json"],
            "supplemental_evidence_sha256": hashes["supplement.json"], "resolutions": [decision]})
        inputs = adjudication.load_inputs(*(self.root / p for p in ("study.json", "audit.json", "rubric.json", "packet.json")))
        result = adjudication.build(inputs, self.root / "review-a.json", self.root / "review-b.json", self.root / "resolutions.json",
            supplemental_path=self.root / "supplement.json", supplemental_capture_dir=capture_dir)
        self.write("review.json", result)
        return iid

    def test_complete_html_keeps_original_grades_and_all_rows(self):
        paths = [self.root / p for p in ("study.json", "review.json", "audit.json", "rubric.json", "review-a.json", "review-b.json")]
        before = [p.read_bytes() for p in paths]
        with patch.object(exporter, "build_export", side_effect=AssertionError("original study must not be regenerated")):
            html = self.render()
        self.assertEqual([p.read_bytes() for p in paths], before)
        self.assertEqual(html.count("class='full'"), 9)
        self.assertEqual(html.count("data-reviewed-verdict="), 1161)
        self.assertEqual(html.count(">ORIGINAL-KEY match</span>"), 387)
        self.assertIn("Agent-reviewed core answers", html)
        self.assertIn("native-probe-adjudication.json", html)
        self.assertIn("probe-adjudication-rubric.json", html)
        self.assertIn("native-probe-adjudication-20260913.md", html)
        self.assertIn("review-a inspected", html)
        self.assertIn("review-b inspected", html)
        self.assertIn("not human adjudication", html)

    def test_byte_changes_in_each_frozen_input_rejected(self):
        for name in ("study.json", "audit.json", "rubric.json", "review-a.json", "review-b.json"):
            with self.subTest(name=name):
                path = self.root / name
                old = path.read_bytes()
                path.write_bytes(old + b"\n")
                with self.assertRaisesRegex(ValueError, "hash"):
                    self.render()
                path.write_bytes(old)

    def test_changed_response_or_unit_id_rejected(self):
        original = self.read("review.json")
        for field, value in (("response", "Fabricated response"), ("item_id", "f" * 64), ("question", "Another question")):
            with self.subTest(field=field):
                review = copy.deepcopy(original)
                review["units"][0][field] = value
                self.write("review.json", review)
                with self.assertRaises(ValueError):
                    self.render()

    def test_missing_duplicate_and_unrelated_units_rejected(self):
        original = self.read("review.json")
        for mode in ("missing", "duplicate", "unrelated"):
            review = copy.deepcopy(original)
            if mode == "missing":
                review["units"].pop()
            elif mode == "duplicate":
                review["units"].append(copy.deepcopy(review["units"][0]))
            else:
                review["units"][0]["probe_id"] = "f" * 64
            self.write("review.json", review)
            with self.assertRaises(ValueError):
                self.render()

    def test_setting_coverage_and_original_grade_tampering_rejected(self):
        original = self.read("review.json")
        for mode in ("missing_setting", "duplicate_row", "changed_raw_grade", "forged_count"):
            review = copy.deepcopy(original)
            if mode == "missing_setting":
                review["configurations"].pop()
            elif mode == "duplicate_row":
                review["configurations"][0]["probes"][1] = review["configurations"][0]["probes"][0]
            elif mode == "changed_raw_grade":
                review["configurations"][0]["probes"][0]["original_key_status"] = "manual_review"
            else:
                review["configurations"][0]["verdict_counts"]["supported_core_answer"] = 129
            self.write("review.json", review)
            with self.assertRaises(ValueError):
                self.render()

    def test_false_agreement_and_missing_independent_review_rejected(self):
        original = self.read("review.json")
        for mode in ("disagree", "missing", "lost_qualification"):
            review = copy.deepcopy(original)
            unit = next(u for u in review["units"] if u["independent_reviews"])
            if mode == "disagree":
                unit["independent_reviews"][1]["verdict"] = "unresolved"
            elif mode == "missing":
                unit["independent_reviews"].pop()
            else:
                unit["decision"]["qualifications"] = []
            self.write("review.json", review)
            with self.assertRaises(ValueError):
                self.render()

    def test_mechanical_unknown_cannot_be_relabeled(self):
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["response"] == "UNKNOWN")
        unit["decision"]["verdict"] = "supported_core_answer"
        unit["decision"]["evidence_capture_ids"] = [unit["source_review"]["evidence"][0]["capture_id"]]
        self.write("review.json", review)
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            self.render()

    def test_response_source_and_review_content_escaped(self):
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["independent_reviews"])
        text = "<img src=x onerror=alert(1)> & <script>bad()</script>"
        for name in ("review-a.json", "review-b.json"):
            document = self.read(name)
            row = next(r for r in document["reviews"] if r["item_id"] == unit["item_id"])
            row.update(rationale=text, qualifications=[text])
            self.write(name, document)
        inputs = adjudication.load_inputs(*(self.root / p for p in ("study.json", "audit.json", "rubric.json", "packet.json")))
        review = adjudication.build(inputs, self.root / "review-a.json", self.root / "review-b.json")
        self.write("review.json", review)
        html = self.render()
        self.assertNotIn("<img src=x", html)
        self.assertNotIn("<script>bad()", html)
        self.assertIn("&lt;script&gt;bad()", html)
        public = copy.deepcopy(next(u for u in review["units"] if u["item_id"] == unit["item_id"]))
        public["source_review"]["evidence"][0]["url"] = "javascript:alert(1)"
        rendered = renderer.row_html(public)
        self.assertNotIn("href='javascript", rendered)

    def test_private_rationale_rejected_before_display(self):
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["independent_reviews"])
        unit["decision"]["rationale"] = "session_id=PRIVATE-CANARY"
        self.write("review.json", review)
        with self.assertRaisesRegex(ValueError, "Private"):
            self.render()

    def test_cli_writes_only_html_and_protects_input_paths(self):
        args = ["--study", str(self.root / "study.json"), "--review", str(self.root / "review.json"),
                "--source-audit", str(self.root / "audit.json"), "--rubric", str(self.root / "rubric.json"),
                "--review-a", str(self.root / "review-a.json"), "--review-b", str(self.root / "review-b.json")]
        before = (self.root / "study.json").read_bytes()
        with patch("sys.stdout", new_callable=io.StringIO):
            renderer.main(args + ["--output", str(self.root / "preview.html")])
        self.assertEqual((self.root / "study.json").read_bytes(), before)
        self.assertIn("Agent-reviewed core answers", (self.root / "preview.html").read_text())
        for protected in (self.root / "study.json", self.root / "review-a.json", self.root / "review-b.json",
                          renderer.ROOT / "dash/index.html", renderer.ROOT / "dash/corrected.html",
                          renderer.ROOT / "dash/corrected-details.html", renderer.ROOT / "dash/open-agency.html"):
            with self.assertRaises(SystemExit), patch("sys.stderr", new_callable=io.StringIO):
                renderer.main(args + ["--output", str(protected)])

    def test_supplemental_override_preserves_initial_uncertainty_and_original_bytes(self):
        original_bytes = (self.root / "study.json").read_bytes()
        self.add_supplement()
        html = self.render(supplemental_path=self.root / "supplement.json")
        self.assertEqual((self.root / "study.json").read_bytes(), original_bytes)
        self.assertIn("Supplementary evidence added after independent review", html)
        self.assertIn("review-a · Unresolved", html)
        self.assertIn("review-b · Unresolved", html)
        self.assertIn("Initial source packet uncertainty is retained.", html)
        self.assertIn("Documented final review uses the supplementary widget source.", html)
        self.assertIn("probe-adjudication-supplement.json", html)
        self.assertIn("Documented overrides of initial agreements: 1", html)

    def test_supplemental_hash_missing_file_and_unbound_unit_evidence_rejected(self):
        iid = self.add_supplement()
        with self.assertRaisesRegex(ValueError, "file is required"):
            self.render()
        path = self.root / "supplement.json"
        old = path.read_bytes()
        path.write_bytes(old + b"\n")
        with self.assertRaisesRegex(ValueError, "Supplemental evidence byte hash"):
            self.render(supplemental_path=path)
        path.write_bytes(old)
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["item_id"] == iid)
        unit["supplemental_evidence"][0]["excerpt"] = "Fabricated source claim"
        self.write("review.json", review)
        with self.assertRaisesRegex(ValueError, "supplemental evidence differs"):
            self.render(supplemental_path=path)

    def test_initial_review_cannot_be_rewritten_to_cite_later_supplement(self):
        iid = self.add_supplement()
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["item_id"] == iid)
        unit["independent_reviews"][0]["evidence_capture_ids"] = ["synthetic-supplement"]
        self.write("review.json", review)
        with self.assertRaisesRegex(ValueError, "unrelated evidence"):
            self.render(supplemental_path=self.root / "supplement.json")

    def test_embedded_rationale_cannot_change_while_frozen_review_hash_is_retained(self):
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["independent_reviews"])
        unit["independent_reviews"][0]["rationale"] = "A substituted rationale absent from the frozen review."
        self.write("review.json", review)
        with self.assertRaisesRegex(ValueError, "frozen independent reviews"):
            self.render()

    def test_substituted_override_rejected_even_when_all_derived_counts_are_updated(self):
        iid = self.add_supplement()
        review = self.read("review.json")
        unit = next(u for u in review["units"] if u["item_id"] == iid)
        unit["decision"].update(verdict="unresolved", rationale="A replacement absent from the frozen resolution.")
        for configuration in review["configurations"]:
            for row in configuration["probes"]:
                if row["item_id"] == iid:
                    row["reviewed_verdict"] = "unresolved"
            rows = configuration["probes"]
            configuration["verdict_counts"] = dict(Counter(r["reviewed_verdict"] for r in rows))
            configuration["original_matches_not_supported_as_complete"] = sum(
                r["original_key_status"] == "exact_alias_match" and r["reviewed_verdict"] != "supported_core_answer" for r in rows)
            configuration["original_unmatched_now_supported"] = sum(
                r["original_key_status"] == "manual_review" and r["reviewed_verdict"] == "supported_core_answer" for r in rows)
        self.write("review.json", review)
        with self.assertRaisesRegex(ValueError, "frozen resolution"):
            self.render(supplemental_path=self.root / "supplement.json")

    def test_resolution_bytes_missing_file_and_rehashed_wrong_bindings_rejected(self):
        self.add_supplement()
        path = self.root / "resolutions.json"
        old = path.read_bytes()
        path.write_bytes(old + b"\n")
        with self.assertRaisesRegex(ValueError, "Resolution evidence byte hash"):
            self.render(supplemental_path=self.root / "supplement.json")
        path.write_bytes(old)
        with self.assertRaisesRegex(ValueError, "resolution file is required"):
            self.render(supplemental_path=self.root / "supplement.json", resolutions_path=self.root / "missing.json")
        document = self.read("resolutions.json")
        document["review_a_sha256"] = "0" * 64
        self.write("resolutions.json", document)
        review = self.read("review.json")
        review["resolutions_sha256"] = adjudication.read(path)[1]
        self.write("review.json", review)
        with self.assertRaisesRegex(ValueError, "Resolution does not bind"):
            self.render(supplemental_path=self.root / "supplement.json")

    def test_public_file_defaults_bind_reviews_and_resolution(self):
        self.add_supplement()
        with patch.object(renderer, "DEFAULT_REVIEW_A", self.root / "review-a.json"), \
             patch.object(renderer, "DEFAULT_REVIEW_B", self.root / "review-b.json"), \
             patch.object(renderer, "DEFAULT_RESOLUTIONS", self.root / "resolutions.json"):
            html = renderer.render(*(self.root / p for p in ("study.json", "review.json", "audit.json", "rubric.json")),
                                   supplemental_path=self.root / "supplement.json")
        self.assertIn("Documented final review uses the supplementary widget source.", html)


if __name__ == "__main__":
    unittest.main()
