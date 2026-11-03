"""RQ4 (quick version): how does prompt design change hallucination?

Runs every answer strategy in prompts.ANSWER_STRATEGIES on the same questions
with the SAME retrieved evidence, then verifies each answer's claims with NLI.

Usage:  python -m eval.prompt_compare [--collection default]
"""

import argparse
import json

from veritas import prompts
from veritas.claims import extract_claims
from veritas.config import settings
from veritas.generate import generate_answer
from veritas.pipeline import Veritas
from veritas.report import Report
from veritas.retrieve import retrieve
from veritas.verify import verify_claims

QUESTIONS = [
    "How can hallucinations in large language models be detected, and how effective are these methods?",
    "Does retrieval-augmented generation reduce hallucination compared with a model answering from memory?",
    "Does fine-tuning a language model on new factual knowledge increase hallucination?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="default")
    args = ap.parse_args()

    v = Veritas(args.collection)
    per_strategy = {s: [] for s in prompts.ANSWER_STRATEGIES}
    answers = []
    for q in QUESTIONS:
        evidence = retrieve(v.index, q)
        for s in prompts.ANSWER_STRATEGIES:
            ans = generate_answer(v.llm, q, evidence, s)
            res = verify_claims(v.nli, extract_claims(v.llm, ans), evidence)
            per_strategy[s] += res
            answers.append({"question": q, "strategy": s, "answer": ans, "stats": Report.stats(res)})
            print(f"{s:14s} {Report.stats(res)['hallucination_rate']:.0%}  {q[:60]}")

    summary = {s: {"type": prompts.ANSWER_STRATEGIES[s]["type"], **Report.stats(r)} for s, r in per_strategy.items()}
    summary["_meta"] = {"questions": QUESTIONS, "llm": v.llm.model, "collection": args.collection}
    settings.ensure_dirs()
    (settings.output_dir / "prompt_compare.json").write_text(json.dumps(summary, indent=2))
    (settings.output_dir / "prompt_compare_answers.json").write_text(json.dumps(answers, indent=2))
    print(f"\n{'strategy':14s} {'claims':>6s} {'halluc':>7s} {'cite-acc':>8s}")
    for s, r in summary.items():
        if not s.startswith("_"):
            ca = f"{r['citation_accuracy']:.0%}" if r["citation_accuracy"] is not None else "–"
            print(f"{s:14s} {r['n_claims']:6d} {r['hallucination_rate']:7.0%} {ca:>8s}")


if __name__ == "__main__":
    main()
