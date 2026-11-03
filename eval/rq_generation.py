"""RQ1 / RQ3 / RQ4: generation experiments on SciFact.

For N SciFact claims phrased as research questions, every condition answers
using the SAME retrieved evidence (top-5 from the full SciFact corpus):

  closed_book             plain LLM, no evidence             (RQ1 baseline)
  bad_vague ... cot       RAG with different prompt designs  (RQ4)
  veritas_corrected       role_instruct + one correction pass (RQ1, full system)

Each answer is decomposed into claims and verified by NLI against the
evidence. Per-claim rows are saved so analysis (RQ3 ablation, calibration,
error analysis) can be done later without new API calls.

Usage:  python -m eval.rq_generation --n 20 [--samples 4] [--judge]
"""

import argparse
import csv
import json
import random

from eval.scifact import build_index, load_claims
from veritas import prompts
from veritas.claims import extract_claims
from veritas.config import settings
from veritas.generate import closed_book_answer, generate_answer, sample_answers
from veritas.pipeline import Veritas
from veritas.report import Report
from veritas.retrieve import format_evidence, retrieve
from veritas.verify import verify_claims

STRATEGIES = list(prompts.ANSWER_STRATEGIES)


def question_from_claim(claim: str) -> str:
    return f"What does the research evidence say about the following statement: \"{claim.rstrip('.')}\"?"


def judge(llm, claim: str, evidence) -> str:
    msg = prompts.CLAIM_JUDGE.format(claim=claim, evidence=format_evidence(evidence))
    return str(llm.chat_json([{"role": "user", "content": msg}]).get("label", "")).upper()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of questions")
    ap.add_argument("--samples", type=int, default=4, help="self-consistency samples for role_instruct (RQ3)")
    ap.add_argument("--judge", action="store_true", help="LLM-judge silver labels per claim (RQ3 ground truth)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    v = Veritas("scifact")
    print(f"SciFact index: {build_index(v)} chunks")

    rows = [r for r in load_claims("dev") if r["label"] != "NOT_ENOUGH_INFO"]
    seen, picked = set(), []
    for r in random.Random(args.seed).sample(rows, len(rows)):
        if r["claim_id"] not in seen:
            seen.add(r["claim_id"])
            picked.append(r)
    picked = picked[:args.n]

    claim_rows, answers = [], []
    for qi, r in enumerate(picked, 1):
        q = question_from_claim(r["claim"])
        evidence = retrieve(v.index, q)
        gold_doc_retrieved = any(e.chunk.doc_name == f"scifact-{r['doc_id']}" for e in evidence)
        print(f"[{qi}/{len(picked)}] {r['claim'][:70]}...  (gold abstract retrieved: {gold_doc_retrieved})")

        conds = {"closed_book": closed_book_answer(v.llm, q)}
        conds.update({s: generate_answer(v.llm, q, evidence, s) for s in STRATEGIES})
        samples = sample_answers(v.llm, q, evidence, args.samples) if args.samples else []

        for cond, ans in conds.items():
            res = verify_claims(v.nli, extract_claims(v.llm, ans), evidence, samples if cond == "role_instruct" else None)
            if cond == "role_instruct":
                weak = [c for c in res if c.label in ("unsupported", "contradicted")]
                fixed = v._correct_once(q, ans, weak, evidence) if weak else ans
                fixed_res = verify_claims(v.nli, extract_claims(v.llm, fixed), evidence) if weak else res
                answers.append({"q": qi, "condition": "veritas_corrected", "answer": fixed})
                _collect(claim_rows, qi, r, "veritas_corrected", fixed_res, gold_doc_retrieved)
            answers.append({"q": qi, "condition": cond, "answer": ans})
            _collect(claim_rows, qi, r, cond, res, gold_doc_retrieved)
            if args.judge and cond == "role_instruct":
                for row in claim_rows[-len(res):]:
                    row["judge"] = judge(v.llm, row["claim"], evidence)

    settings.ensure_dirs()
    out_csv = settings.output_dir / "rq_generation_claims.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for row in claim_rows for k in row}))
        w.writeheader()
        w.writerows(claim_rows)
    (settings.output_dir / "rq_generation_answers.json").write_text(json.dumps(answers, indent=2))

    summary = {}
    for cond in ["closed_book", *STRATEGIES, "veritas_corrected"]:
        sub = [row for row in claim_rows if row["condition"] == cond]
        n_q = len({row["q"] for row in sub}) or 1
        s = Report.stats([_Fake(row) for row in sub])
        summary[cond] = {**s, "claims_per_answer": len(sub) / n_q}
    summary["_meta"] = {"n_questions": len(picked), "llm": v.llm.model, "api_calls": v.llm.calls,
                        "samples": args.samples, "judge": args.judge}
    (settings.output_dir / "rq_generation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'condition':20s} {'claims':>6s} {'halluc.':>8s} {'conf':>6s} {'cite-acc':>9s}")
    for cond, s in summary.items():
        if cond.startswith("_"):
            continue
        ca = f"{s['citation_accuracy']:.2f}" if s["citation_accuracy"] is not None else "–"
        print(f"{cond:20s} {s['n_claims']:6d} {s['hallucination_rate']:8.1%} {s['mean_confidence']:6.2f} {ca:>9s}")
    print(f"Saved {out_csv}")


class _Fake:
    """Adapter so Report.stats can summarise CSV rows."""
    def __init__(self, row):
        self.label, self.confidence = row["label"], row["confidence"]
        self.citation_correct = row["citation_correct"]


def _collect(rows, qi, gold, cond, results, gold_retrieved) -> None:
    for c in results:
        rows.append({"q": qi, "gold_label": gold["label"], "gold_doc_retrieved": gold_retrieved,
                     "condition": cond, "claim": c.text, "citations": " ".join(c.citations),
                     "entail": round(c.entail, 4), "contradict": round(c.contradict, 4),
                     "consistency": None if c.consistency is None else round(c.consistency, 4),
                     "confidence": round(c.confidence, 4), "label": c.label,
                     "citation_correct": c.citation_correct})


if __name__ == "__main__":
    main()
