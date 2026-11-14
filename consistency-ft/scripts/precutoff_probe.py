"""Contamination probe: short factual questions about dated public events, 2024-09 .. 2025-12.

Each question is asked cold (no date or as-of hint), one call per question,
and graded by alias match (accent- and case-insensitive).  Accuracy by event
month locates the model's effective knowledge cutoff.  A model is "clean" for
the JEV test set (questions close May-Dec 2025, earliest as-of 2025-04-28) only
if it fails on events after about 2025-03.

Usage: precutoff_probe.py --model o3 --out DIR [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precutoff_client import MODELS, chat  # noqa: E402

# (event month, question, accepted aliases)
QUESTIONS = [
    ("2024-09", "Who won the men's singles title at the 2024 US Open tennis championships?", ["sinner"]),
    ("2024-09", "Who won Sri Lanka's 2024 presidential election?", ["dissanayake", "dissanayaka", "anura"]),
    ("2024-09", "Who won the Liberal Democratic Party leadership election in Japan in September 2024?", ["ishiba"]),
    ("2024-10", "Who won the 2024 men's Ballon d'Or?", ["rodri"]),
    ("2024-10", "Which team won the 2024 World Series?", ["dodgers"]),
    ("2024-10", "Who won the 2024 Nobel Prize in Literature?", ["han kang"]),
    ("2024-11", "Which candidate finished first in the first round of Romania's presidential election in November 2024?", ["georgescu"]),
    ("2024-11", "Who won Uruguay's 2024 presidential runoff?", ["orsi"]),
    ("2024-11", "Who won the 2024 Las Vegas Grand Prix in Formula 1?", ["russell"]),
    ("2024-12", "To which country did Bashar al-Assad flee in December 2024?", ["russia", "moscow"]),
    ("2024-12", "What did South Korea's president declare on the night of December 3, 2024?", ["martial law"]),
    ("2024-12", "Who was appointed Prime Minister of France in December 2024?", ["bayrou"]),
    ("2025-01", "Who won the women's singles title at the 2025 Australian Open?", ["keys"]),
    ("2025-01", "Which Los Angeles neighborhood was largely destroyed by a wildfire in January 2025?", ["palisades"]),
    ("2025-01", "Who won the men's singles title at the 2025 Australian Open?", ["sinner"]),
    ("2025-02", "Which team won Super Bowl LIX?", ["eagles", "philadelphia"]),
    ("2025-02", "Which album won Album of the Year at the 2025 Grammy Awards?", ["cowboy carter"]),
    ("2025-02", "What was the final score of Super Bowl LIX?", ["40-22", "40 - 22", "40–22", "40 to 22"]),
    ("2025-03", "Which film won Best Picture at the 2025 Academy Awards?", ["anora"]),
    ("2025-03", "Who won Best Actress at the 2025 Academy Awards?", ["mikey madison"]),
    ("2025-03", "Which country was hit by a magnitude 7.7 earthquake on March 28, 2025?", ["myanmar", "burma"]),
    ("2025-04", "Who won the 2025 Masters golf tournament?", ["mcilroy"]),
    ("2025-04", "On what date did Pope Francis die? Give the full date.", ["april 21, 2025", "21 april 2025", "2025-04-21", "april 21 2025", "21st april 2025", "april 21st, 2025"]),
    ("2025-04", "Which party won the most seats in Canada's April 2025 federal election?", ["liberal"]),
    ("2025-05", "What papal name did the pope elected in May 2025 take?", ["leo xiv", "leo 14"]),
    ("2025-05", "Which club won the 2025 UEFA Champions League final?", ["paris saint-germain", "paris saint germain", "psg"]),
    ("2025-05", "Which country won the Eurovision Song Contest 2025?", ["austria"]),
    ("2025-05", "Who won Romania's presidential election runoff in May 2025?", ["nicusor"]),
    ("2025-06", "Who won South Korea's presidential election held on June 3, 2025?", ["lee jae-myung", "lee jae myung", "lee jaemyung"]),
    ("2025-06", "Which team won the 2025 NBA Finals?", ["thunder", "oklahoma"]),
    ("2025-06", "Who won the women's singles title at the 2025 French Open (Roland Garros)?", ["gauff"]),
    ("2025-07", "Who won the women's singles title at Wimbledon 2025?", ["swiatek", "świątek"]),
    ("2025-07", "Which club won the 2025 FIFA Club World Cup final?", ["chelsea"]),
    ("2025-07", "Which national team won the UEFA Women's Euro 2025?", ["england"]),
    ("2025-08", "Which candidate finished first in the first round of Bolivia's August 2025 presidential election?", ["rodrigo paz", "paz pereira"]),
    ("2025-08", "What is the name of the flagship model OpenAI released on August 7, 2025?", ["gpt-5", "gpt 5", "gpt5"]),
    ("2025-09", "Who won the men's singles title at the 2025 US Open tennis championships?", ["alcaraz"]),
    ("2025-09", "Who won the 2025 men's Ballon d'Or?", ["dembele", "dembélé"]),
    ("2025-09", "Who became interim Prime Minister of Nepal in September 2025?", ["karki"]),
    ("2025-09", "Who was appointed Prime Minister of France in September 2025?", ["lecornu"]),
    ("2025-10", "Who won the 2025 Nobel Peace Prize?", ["machado"]),
    ("2025-10", "Who won the 2025 Nobel Prize in Literature?", ["krasznahorkai"]),
    ("2025-10", "Who became Japan's prime minister in October 2025?", ["takaichi"]),
    ("2025-11", "Who won the November 2025 New York City mayoral election?", ["mamdani"]),
    ("2025-11", "Which team won the 2025 World Series?", ["dodgers"]),
    ("2025-11", "Who won the 2025 Virginia gubernatorial election?", ["spanberger"]),
    ("2025-12", "Who won Chile's presidential runoff in December 2025?", ["kast"]),
    ("2025-12", "Who was declared the winner of Honduras's November 2025 presidential election?", ["asfura"]),
    ("2025-12", "Who or what did Time magazine name its 2025 Person of the Year?", ["architects of ai", "architects of artificial intelligence"]),
]
SYSTEM = "Answer factual questions from your own knowledge."
INSTR = "\n\nAnswer with only the short answer (a name, team, date or phrase), no explanation. If you are not sure, give your single best guess."


def norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def grade(answer: str | None, aliases: list[str]) -> bool:
    if not answer:
        return False
    a = norm(answer)
    return any(norm(x) in a for x in aliases)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def one(q):
        month, text, aliases = q
        r = chat(args.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text + INSTR}], max_tokens=8000)
        return {"month": month, "question": text, "aliases": aliases, "answer": r["text"], "correct": grade(r["text"], aliases),
                "usage": r["usage"], "cost": r["cost"], "served_model": r["served_model"], "provider": r["provider"],
                "finish_reason": r["finish_reason"], "errors": r["errors"]}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, QUESTIONS))
    with (out / f"probe-{args.model}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = defaultdict(list)
    for r in rows:
        by[r["month"]].append(r["correct"])
    summary = {"model": args.model, "provider_model": MODELS[args.model][1],
               "served_models": sorted({r["served_model"] or "?" for r in rows}),
               "providers": sorted({str(r["provider"]) for r in rows}),
               "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
               "no_answer": sum(r["answer"] is None for r in rows),
               "by_month": {m: {"n": len(v), "correct": sum(v)} for m, v in sorted(by.items())},
               "acc_2024-09_to_2025-03": None, "acc_2025-04_to_2025-12": None,
               "cost_usd": sum(r["cost"] or 0 for r in rows)}
    early = [r["correct"] for r in rows if r["month"] <= "2025-03"]
    late = [r["correct"] for r in rows if r["month"] >= "2025-04"]
    summary["acc_2024-09_to_2025-03"] = sum(early) / len(early)
    summary["acc_2025-04_to_2025-12"] = sum(late) / len(late)
    (out / f"probe-{args.model}-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("model", "served_models", "accuracy", "acc_2024-09_to_2025-03",
                                              "acc_2025-04_to_2025-12", "no_answer", "cost_usd")}))
    print(" ".join(f"{m}:{v['correct']}/{v['n']}" for m, v in summary["by_month"].items()))


if __name__ == "__main__":
    main()
