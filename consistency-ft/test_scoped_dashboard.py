import contextlib
import io
import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import CHECK_FIELDS
from scripts import build_dash
from scripts.export_benchmark import NATIVE_CONFIGS, comparison_context, rebuild_database


class ScopedDashboardTests(unittest.TestCase):
    """Rendering must not reintroduce excluded questions or unequal denominators."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.scope = {
            "scope_id": "reviewed-fixture",
            "included_tuple_ids": ["shared", "reviewed-only"],
            "included_proposition_ids": ["allowed-proposition"],
            "check_type_counts": {check: int(check in ("Negation", "ExpEvidence")) for check in CHECK_FIELDS},
            "exclusions": [],
        }
        (self.directory / "scope.json").write_text(json.dumps(self.scope))
        self.manifest = {"schema_version": 2, "comparison_scope_id": self.scope["scope_id"],
                         "scopes": {self.scope["scope_id"]: {"file": "scope.json", "sha256": "fixture"}}}
        self.native = "native/gpt-5.6-luna@none"
        self.archive = "accounts/fireworks/models/muse-glimmer-30b"
        self.propositions = [{"id": "allowed-proposition", "text": "A spacecraft lands.\n\nResolution criteria: Confirmed landing."},
                             {"id": "excluded-proposition", "text": "EXCLUDED QUESTION\n\nResolution criteria: Excluded."}]
        self.questions = {
            tuple_id: {role: {"title": title} for role in CHECK_FIELDS[check]}
            for tuple_id, check, title in [("shared", "Negation", "A spacecraft lands."),
                                          ("reviewed-only", "ExpEvidence", "REVIEWED BUT NOT SHARED"),
                                          ("excluded", "Negation", "EXCLUDED QUESTION")]
        }

    def row(self, tuple_id, violation, check="Negation"):
        return {"id": tuple_id, "check_type": check, "violation": violation,
                "probabilities": {role: .4 for role in CHECK_FIELDS[check]}}

    def fixture_run(self, model, rows, date="2026-09-13", kind="checks", suffix=""):
        native = model.startswith("native/")
        key = "violation" if kind == "checks" else "abs_diff"
        filename = f'{model.rsplit("/", 1)[-1]}-{kind}{suffix}.jsonl'
        return {"model": model, "kind": kind, "rows": rows, "n": len(rows),
                "stat": sum(row[key] for row in rows) / len(rows) if rows else None,
                "run_id": filename, "file": filename, "source_file": filename,
                "protocol_id": "descriptive-v2" if native else "legacy-unspecified",
                "scope_id": self.scope["scope_id"] if native else None,
                "reasoning": "none" if native else "unknown", "provider": "openai" if native else "fireworks",
                "requested_model": model.removeprefix("native/").partition("@")[0], "sha256": "fixture", "receipts": [],
                "transport": "subscription-native-cli" if native else "fireworks-api",
                "recorded_at": date, "timestamp_source": "fixture", "status": "complete",
                "coverage": {"valid_propositions": 1, "completed_propositions": 1},
                "target_tuples": 300, "target_propositions": 591}

    def render_pages(self, runs, tuples=None):
        if tuples is None:
            tuples = [{"id": ident, "check_type": check} for ident, check in
                      [("shared", "Negation"), ("reviewed-only", "ExpEvidence"), ("excluded", "Negation")]]
        with patch.object(build_dash, "SNAPSHOTS", self.directory), \
                patch.object(build_dash, "OUTPUT", self.directory / "index.html"), \
                patch.object(build_dash, "load_snapshot", return_value=(self.manifest, self.propositions, tuples, runs)), \
                patch.object(build_dash, "load_questions", return_value=self.questions), \
                patch.object(build_dash, "rebuild_database"), \
                patch.object(build_dash.shutil, "copyfile") as copyfile, \
                patch.object(build_dash, "usage_table", return_value="USAGE DETAILS ONLY"), \
                contextlib.redirect_stdout(io.StringIO()):
            build_dash.main()
            copyfile.assert_any_call(self.directory / "scope.json", self.directory / "reviewed-scope.json")
        return {name: (self.directory / name).read_text() for name in
                ("index.html", "details.html", "corrected.html", "corrected-details.html")}

    def render(self, runs):
        pages = self.render_pages(runs)
        return pages["corrected.html"], pages["corrected-details.html"]

    def test_latest_settings_share_denominator_and_excluded_examples_never_render(self):
        runs = [self.fixture_run(self.native, [self.row("shared", .1), self.row("reviewed-only", .3, "ExpEvidence"), self.row("excluded", .99)]),
                self.fixture_run(self.archive, [self.row("shared", .2)], date="2026-08-13"),
                self.fixture_run(self.native, [self.row("excluded", .001)], date="2026-09-12", suffix="-old")]
        home, details = self.render(runs)
        embedded = json.loads(re.search(r'id="dashboard-data">(.*?)</script>', home, re.S).group(1))
        self.assertEqual({row[0]: row[1:] for row in embedded["ranking"]},
                         {self.native: [.1, 1], self.archive: [.2, 1]})
        self.assertEqual(embedded["comparison"]["shared_tuple_ids"], ["shared"])
        self.assertEqual({row["shared_n"] for row in embedded["comparison"]["models"]}, {1})
        self.assertEqual({row["reviewed_n"] for row in embedded["comparison"]["models"]}, {1, 2})
        self.assertIn("1 shared tuples across all 2 settings", home)
        self.assertIn("from 2 reviewed tuples", home)
        self.assertIn("Archived coverage for muse-glimmer-30b is 1/2", home)
        self.assertIn("Selection by shared availability", home)
        self.assertIn("highest shared-set mean", home)
        self.assertNotIn("PARTIAL RUN", home)
        self.assertIn("A spacecraft lands.", home)
        self.assertNotIn("EXCLUDED QUESTION", home)
        self.assertNotIn("REVIEWED BUT NOT SHARED", home)
        self.assertIn("Cond Cond unavailable: no eligible reviewed tuples.", home)
        self.assertIn("Exp Evidence unavailable: no tuples scored by every participating setting.", home)
        self.assertIn("Unavailable: no shared scored tuples", home)
        self.assertNotIn("USAGE DETAILS ONLY", home)
        self.assertIn("USAGE DETAILS ONLY", details)
        self.assertIn("Reviewed-scope mean", details)
        self.assertIn('href="/reviewed-scope.json"', details)
        self.assertIn('>2/2</td>', details)
        self.assertIn('>1/2</td>', details)
        self.assertIn("shared ranking tuples", details)
        self.assertIn("Run summaries below retain each run’s own coverage", home)
        self.assertIn('href="/corrected-details.html#coverage"', home)
        self.assertIn('href="/corrected.html#check-results"', details)
        self.assertEqual(len(embedded["runs"]), 3)

    def test_elicitation_uses_allowed_ids_or_exact_titles_and_preserves_missing_native_logprobs(self):
        pair = {"verbalized_probability": .4, "logprob_probability": .3, "abs_diff": .1}
        elicit = self.fixture_run(self.archive, [
            {**pair, "id": "legacy-uuid", "title": "A spacecraft lands."},
            {**pair, "id": "allowed-proposition", "title": "Alias retained by proposition ID"},
            {**pair, "id": "excluded-proposition", "title": "EXCLUDED QUESTION"},
            {**pair, "id": "unrelated", "title": "A spacecraft lands. Extra text"},
        ], kind="elicit")
        filtered = build_dash.scoped_elicitation_runs({self.archive: elicit}, self.propositions, {"allowed-proposition"})
        self.assertEqual([row["id"] for row in filtered[self.archive]["rows"]], ["legacy-uuid", "allowed-proposition"])
        table = build_dash.elicitation_table(filtered, [self.native, self.archive])
        self.assertIn("Logprobs unavailable from native client", table)
        home, _ = self.render([self.fixture_run(self.native, [self.row("shared", .1)]),
                               self.fixture_run(self.archive, [self.row("shared", .2)]), elicit])
        self.assertNotIn("EXCLUDED QUESTION", home)
        self.assertNotIn("A spacecraft lands. Extra text", home)
        self.assertIn("Historical pairs are restricted to the 1 reviewed propositions", home)
        empty = build_dash.scoped_elicitation_runs({self.archive: elicit}, self.propositions, set())
        self.assertIn("No eligible paired elicitation data", build_dash.elicitation_table(empty, [self.archive]))

    def test_scoped_headlines_honor_stated_target_without_changing_legacy_defaults(self):
        ranking = [("model-a", .1, 20), ("model-b", .2, 19)]
        self.assertEqual(build_dash.headline_runs(ranking), [])
        self.assertEqual(build_dash.headline_runs(ranking, target=20), [ranking[0]])
        self.assertEqual(build_dash.headline_runs(ranking, target=0), [])
        self.assertIn("model-a (20/300)", build_dash.ranking_chart(ranking))
        scoped = build_dash.ranking_chart(ranking, target=20)
        self.assertNotIn("model-a (20/", scoped)
        self.assertIn("model-b (19/20)", scoped)

    def test_empty_shared_intersection_retains_settings_without_zero_scores(self):
        home, details = self.render([self.fixture_run(self.native, [self.row("shared", .1)]),
                                    self.fixture_run(self.archive, [self.row("reviewed-only", .2, "ExpEvidence")])])
        embedded = json.loads(re.search(r'id="dashboard-data">(.*?)</script>', home, re.S).group(1))
        self.assertEqual(embedded["ranking"], [])
        self.assertEqual(len(embedded["comparison"]["models"]), 2)
        self.assertIn("0 shared tuples across all 2 settings", home)
        self.assertIn("no complete comparison", home)
        self.assertNotIn('class="numeric score">0.0000', details)
        self.assertEqual(details.count('data-model="'), 2)

    def participation_fixture(self):
        native = self.fixture_run(self.native, [self.row("shared", .1), self.row("reviewed-only", .3, "ExpEvidence")])
        native["target_tuples"] = 2
        complete = self.fixture_run("accounts/fireworks/models/complete-archive",
                                    [self.row("shared", .2), self.row("reviewed-only", .6, "ExpEvidence"), self.row("excluded", .9)])
        complete["target_tuples"] = 3
        incomplete = self.fixture_run(self.archive, [self.row("shared", .001)])
        incomplete["target_tuples"] = 3
        return [native, complete, incomplete]

    def test_incomplete_archive_is_visible_unranked_without_shrinking_ranked_denominators(self):
        self.manifest["comparison_policy"] = "complete-historical-plus-native"
        runs = self.participation_fixture()
        home, details = self.render(runs)
        embedded = json.loads(re.search(r'id="dashboard-data">(.*?)</script>', home, re.S).group(1))
        ranking = {row[0]: row[1:] for row in embedded["ranking"]}
        self.assertEqual(set(ranking), {self.native, runs[1]["model"]})
        self.assertEqual({row[1] for row in ranking.values()}, {2})
        self.assertAlmostEqual(ranking[self.native][0], .2)
        self.assertAlmostEqual(ranking[runs[1]["model"]][0], .4)
        archive = next(row for row in embedded["comparison"]["models"] if row["model"] == self.archive)
        self.assertIsNone(archive["shared_n"])
        self.assertEqual(archive["reviewed_n"], 1)
        self.assertFalse(archive["comparison_eligible"])
        chart = re.search(r'id="model-ranking">(.*?)</article>', home, re.S).group(1)
        self.assertIn("muse-glimmer-30b (1/2 reviewed)", chart)
        self.assertIn("unranked: incomplete historical coverage", chart)
        self.assertEqual(chart.count('class="bar mark"'), 2)
        self.assertNotIn("0.001", chart)
        heatmap = re.search(r'id="check-results">(.*?)</article>', home, re.S).group(1)
        self.assertIn("muse-glimmer-30b · unranked", heatmap)
        self.assertIn("unranked: incomplete historical coverage", heatmap)
        self.assertIn("2 shared tuples across 2 participating settings", home)
        self.assertNotIn("nonpolitical", home)
        self.assertNotIn("review excluded political topics", details)
        row = re.search(r'<tr data-model="' + re.escape(self.archive) + r'".*?</tr>', details, re.S).group(0)
        self.assertIn("unranked: incomplete historical coverage", row)
        self.assertIn('>1/2</td>', row)
        self.assertIn('>0.0010</td>', row)  # Explicitly labeled reviewed-scope mean, using its own count.
        self.assertNotIn('class="numeric score">0.0010', row)
        headings = ["Model ranking", "Mean violation by model", "Elicitation readouts", "Worst violations by model", "Runs"]
        positions = [home.index('<h2>' + heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))

    def test_declared_policy_retains_incomplete_native_and_checks_original_historical_target(self):
        self.manifest["comparison_policy"] = "complete-historical-plus-native"
        runs = self.participation_fixture()
        runs[0]["rows"] = runs[0]["rows"][:1]  # Native remains a participant even while partial.
        runs[0]["n"] = 1
        runs[2]["rows"].append(self.row("reviewed-only", .01, "ExpEvidence"))
        runs[2]["n"] = 2  # All reviewed rows, but not its original target of three.
        context = comparison_context(self.manifest, runs, self.directory)
        self.assertEqual(context["common_tuple_ids"], {"shared"})
        self.assertEqual(set(context["participating_checks"]), {self.native, runs[1]["model"]})
        self.assertEqual(context["unranked_models"], {self.archive: "incomplete historical coverage"})
        self.assertEqual(context["scoped_checks"][self.archive]["n"], 2)

    def test_default_and_explicit_legacy_policy_keep_all_settings_intersection(self):
        runs = self.participation_fixture()
        implicit = comparison_context(self.manifest, runs, self.directory)
        explicit = comparison_context({**self.manifest, "comparison_policy": "all-settings"}, runs, self.directory)
        self.assertEqual(implicit, explicit)
        self.assertEqual(implicit["common_tuple_ids"], {"shared"})
        self.assertEqual(len(implicit["participating_checks"]), 3)
        self.assertEqual(implicit["unranked_models"], {})
        with self.assertRaisesRegex(ValueError, "Unknown comparison policy"):
            comparison_context({**self.manifest, "comparison_policy": "typo"}, runs, self.directory)

    def test_sqlite_and_dashboard_agree_on_participation_counts_and_null_unranked_scores(self):
        self.manifest["comparison_policy"] = "complete-historical-plus-native"
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))
        runs = self.participation_fixture()
        tuples = [{"id": ident, "check_type": check,
                   "propositions": {role: "allowed-proposition" for role in CHECK_FIELDS[check]}}
                  for ident, check in [("shared", "Negation"), ("reviewed-only", "ExpEvidence"), ("excluded", "Negation")]]
        with patch("scripts.export_benchmark.load_snapshot", return_value=(self.manifest, self.propositions, tuples, runs)):
            database = rebuild_database(self.directory, self.directory / "fixture.sqlite")
        home, _ = self.render(runs)
        embedded = json.loads(re.search(r'id="dashboard-data">(.*?)</script>', home, re.S).group(1))
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
            rows = db.execute("SELECT model, mean_violation, scored_count FROM comparison_ranking WHERE comparison_eligible=1 ORDER BY mean_violation").fetchall()
            self.assertEqual([list(row) for row in rows], embedded["ranking"])
            archive = db.execute("SELECT comparison_eligible, unranked_reason, scored_count, mean_violation, reviewed_scored_count, original_scored_count, original_target_tuples FROM comparison_ranking WHERE model=?", (self.archive,)).fetchone()
            self.assertEqual(archive, (0, "incomplete historical coverage", None, None, 1, 1, 3))
            original = db.execute("SELECT scored_count, mean_violation FROM ranking WHERE model=?", (self.archive,)).fetchone()
            self.assertEqual(original, (1, .001))
            metadata = json.loads(db.execute("SELECT value FROM metadata WHERE key='comparison_scope'").fetchone()[0])
            self.assertEqual(metadata["comparison_policy"], "complete-historical-plus-native")
            self.assertEqual(metadata["participating_model_settings"], 2)
            self.assertEqual(metadata["model_settings"], 3)

    def test_usage_labels_current_quota_interval_and_cumulative_unknowns_separately(self):
        report = {"campaign": "scoped", "as_of": "later", "providers": [{"provider": "Example", "window": "weekly",
                  "before": 2, "after": 3, "used_delta": 1, "before_at": "start", "after_at": "end"}],
                  "cumulative_recorded_native_tokens_including_setup": {"total_tokens": 123456},
                  "cumulative_recorded_cli_launches": 1000, "cumulative_interrupted_launches_without_receipts": 40,
                  "cumulative_native_usage_unknown_count": 42}
        (self.directory / "subscription-usage.json").write_text(json.dumps(report))
        with patch.object(build_dash, "SNAPSHOTS", self.directory):
            rendered = build_dash.usage_table()
        for phrase in ("123,456 recorded native tokens", "1,000 CLI launches", "40 interrupted launches",
                       "unknown for 42 launches", "scoped: start to end", "percentages are not summed"):
            self.assertIn(phrase, rendered)

    def test_original_and_corrected_pages_keep_campaigns_cohorts_and_readouts_separate(self):
        self.manifest["comparison_policy"] = "complete-historical-plus-native"
        ids = ["shared", "reviewed-only", "original-only"] + [f"original-{i:03d}" for i in range(3, 300)]
        tuples = [{"id": ident, "check_type": "Negation"} for ident in ids]
        self.questions = {ident: {role: {"title": "ORIGINAL SOURCE " + ident} for role in CHECK_FIELDS["Negation"]}
                          for ident in ids}
        self.scope["check_type_counts"] = {check: 2 if check == "Negation" else 0 for check in CHECK_FIELDS}
        (self.directory / "scope.json").write_text(json.dumps(self.scope))
        frozen_scope = (self.directory / "scope.json").read_bytes()
        runs = []
        for run_key, provider, model, effort in NATIVE_CONFIGS:
            identity = f"native/{model}@{effort}"
            original_ids = ids[:-1] if run_key == "luna-none" else ids
            original = self.fixture_run(identity, [self.row(ident, .2) for ident in original_ids], date="2026-09-12T17:00:00Z")
            original.update(run_id=f"native-20260912/{run_key}", file=f"original-{run_key}.jsonl",
                            protocol_id="numeric-v1", scope_id=None, reasoning=effort, provider=provider,
                            target_tuples=300, status="completed-with-missing" if len(original_ids) < 300 else "complete")
            final = self.fixture_run(identity, [self.row("shared", .8), self.row("reviewed-only", .6)], date="2026-09-13T18:00:00Z")
            final.update(run_id=f"native-20260913-scoped-final/{run_key}", file=f"final-{run_key}.jsonl",
                         reasoning=effort, provider=provider, target_tuples=2)
            runs += [original, final]
        for index in range(14):
            runs.append(self.fixture_run(f"accounts/fireworks/models/archive-{index}",
                                        [self.row(ident, .3) for ident in ids], date="2026-08-13"))
        runs.append(self.fixture_run(self.archive, [self.row("original-only", .001)], date="2026-08-13"))
        pair = {"verbalized_probability": .4, "logprob_probability": .3, "abs_diff": .1}
        runs.append(self.fixture_run("accounts/fireworks/models/archive-0", [
            {**pair, "id": "allowed-proposition", "title": "A spacecraft lands."},
            {**pair, "id": "excluded-proposition", "title": "EXCLUDED QUESTION"},
        ], kind="elicit", date="2026-08-13"))

        pages = self.render_pages(runs, tuples)
        embedded = {name: json.loads(re.search(r'id="dashboard-data">(.*?)</script>', page, re.S).group(1))
                    for name, page in pages.items()}
        original, corrected = embedded["index.html"], embedded["corrected.html"]
        self.assertEqual(original["view"], "original")
        self.assertEqual(corrected["view"], "corrected")
        self.assertEqual(original["native_campaigns"], ["native-20260912"])
        self.assertEqual(corrected["native_campaigns"], ["native-20260913-scoped-final"])
        self.assertEqual(original["comparison"]["original_target"], 300)
        self.assertEqual(original["comparison"]["shared_target"], 299)
        self.assertEqual(corrected["comparison"]["reviewed_target"], 2)
        self.assertEqual(corrected["comparison"]["shared_target"], 2)
        self.assertEqual(set(original["comparison"]["shared_tuple_ids"]), set(ids[:-1]))
        self.assertEqual(set(corrected["comparison"]["shared_tuple_ids"]), {"shared", "reviewed-only"})
        for view, expected_n in ((original, 299), (corrected, 2)):
            self.assertEqual(len(view["comparison"]["models"]), 24)
            self.assertEqual(len(view["ranking"]), 23)
            self.assertEqual({row[2] for row in view["ranking"]}, {expected_n})
            muse = next(row for row in view["comparison"]["models"] if row["model"] == self.archive)
            self.assertFalse(muse["comparison_eligible"])
            self.assertIsNone(muse["shared_n"])
        for run_key, _, model, effort in NATIVE_CONFIGS:
            identity = f"native/{model}@{effort}"
            original_row = next(row for row in original["latest"] if row["model"] == identity)
            final_row = next(row for row in corrected["latest"] if row["model"] == identity)
            self.assertEqual(original_row["run_id"], f"native-20260912/{run_key}")
            self.assertEqual(original_row["protocol_id"], "numeric-v1")
            self.assertIsNone(original_row["scope_id"])
            self.assertEqual(original_row["recorded_at"], "2026-09-12T17:00:00Z")
            self.assertAlmostEqual(original_row["stat"], .2)
            self.assertAlmostEqual(final_row["stat"], .7)
            self.assertIn(build_dash.short_model(identity), pages["index.html"])
            self.assertIn(build_dash.short_model(identity), pages["corrected.html"])
        self.assertNotIn("2026-09-13T18:00:00Z", pages["index.html"])
        self.assertNotIn("native-20260913-scoped-final", pages["index.html"])
        self.assertIn("EXCLUDED QUESTION", pages["index.html"])
        self.assertNotIn("EXCLUDED QUESTION", pages["corrected.html"])
        self.assertIn("No topic or semantic subset is applied here", pages["index.html"])
        self.assertIn("from the original 300-tuple benchmark", pages["index.html"])
        self.assertIn("299/300 original", pages["index.html"])
        self.assertIn("muse-glimmer-30b (1/300 original)", pages["index.html"])
        self.assertIn("muse-glimmer-30b (0/2 reviewed)", pages["corrected.html"])
        self.assertNotIn("highest full-run mean", pages["index.html"])
        self.assertIn("highest shared-set mean", pages["index.html"])
        self.assertIn("Original-cohort mean", pages["details.html"])
        self.assertIn(">299/300</td>", pages["details.html"])
        self.assertIn("Reviewed-scope mean", pages["corrected-details.html"])
        self.assertIn('href="/#check-results"', pages["details.html"])
        self.assertIn('href="/corrected.html#check-results"', pages["corrected-details.html"])
        self.assertIn('href="/details.html#coverage"', pages["index.html"])
        self.assertIn('href="/corrected-details.html#coverage"', pages["corrected.html"])
        self.assertNotIn("USAGE DETAILS ONLY", pages["index.html"])
        self.assertNotIn("USAGE DETAILS ONLY", pages["corrected.html"])
        self.assertEqual(embedded["details.html"], original)
        self.assertEqual(embedded["corrected-details.html"], corrected)
        self.assertEqual((self.directory / "scope.json").read_bytes(), frozen_scope)

    def test_original_campaign_selector_rejects_protocol_or_scope_relabeling(self):
        run = self.fixture_run(self.native, [self.row("shared", .1)])
        run["run_id"] = "native-20260912/luna-none"
        with self.assertRaisesRegex(ValueError, "original native campaign must use numeric-v1"):
            build_dash.original_benchmark_runs([run])
        run["protocol_id"] = "numeric-v1"
        with self.assertRaisesRegex(ValueError, "full source cohort"):
            build_dash.original_benchmark_runs([run])
        run["scope_id"] = None
        self.assertEqual(build_dash.original_benchmark_runs([run]), [run])
        self.assertEqual(build_dash.original_benchmark_runs([{**run, "run_id": "native-new/luna-none"}]), [])


if __name__ == "__main__":
    unittest.main()
