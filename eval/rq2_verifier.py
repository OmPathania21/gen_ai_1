"""RQ2: Can a transformer NLI model reliably verify scientific claims?

Each SciFact dev claim is checked against the abstract it cites. The gold label
is SUPPORT / CONTRADICT / NOT_ENOUGH_INFO. We compare:
  - NLI (DeBERTa-v3, VERITAS verifier)       -> argmax of entail / contradict / neutral
  - LLM-as-judge (classification prompt)     -> optional baseline, uses API calls

Usage:  python -m eval.rq2_verifier [--judge] [--limit N]
"""

import argparse
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from eval.scifact import doc_text, load_claims, load_corpus
from veritas import prompts
from veritas.config import settings
from veritas.verify import get_nli

LABELS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
NLI_TO_LABEL = {"entail": "SUPPORT", "contradict": "CONTRADICT", "neutral": "NOT_ENOUGH_INFO"}
JUDGE_TO_LABEL = {"SUPPORTED": "SUPPORT", "CONTRADICTED": "CONTRADICT", "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO"}


def metrics(gold: list[str], pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro"),
        "per_class": classification_report(gold, pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(gold, pred, labels=LABELS).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="also run the LLM-as-judge baseline")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    corpus, rows = load_corpus(), load_claims("dev")
    rows = rows[:args.limit] if args.limit else rows
    gold = [r["label"] for r in rows]

    nli = get_nli()
    probs = nli.predict([(doc_text(corpus[r["doc_id"]]), r["claim"]) for r in rows])
    nli_pred = [NLI_TO_LABEL[max(p, key=p.get)] for p in probs]

    results = {"n": len(rows), "label_distribution": {l: gold.count(l) for l in LABELS},
               "nli": metrics(gold, nli_pred),
               # binary view: can P(entail) separate supported claims from everything else?
               "nli_auroc_supported": roc_auc_score([g == "SUPPORT" for g in gold], [p["entail"] for p in probs])}

    if args.judge:
        from veritas.llm import LLM

        llm = LLM()
        judge_pred = []
        for r in rows:
            msg = prompts.CLAIM_JUDGE.format(claim=r["claim"], evidence=doc_text(corpus[r["doc_id"]]))
            out = llm.chat_json([{"role": "user", "content": msg}])
            judge_pred.append(JUDGE_TO_LABEL.get(str(out.get("label", "")).upper(), "NOT_ENOUGH_INFO"))
        results["llm_judge"] = metrics(gold, judge_pred)
        results["llm_judge_model"] = llm.model

    settings.ensure_dirs()
    out = settings.output_dir / "rq2_verifier.json"
    out.write_text(json.dumps(results, indent=2))
    for name in ("nli", "llm_judge"):
        if name in results:
            print(f"{name:10s} accuracy={results[name]['accuracy']:.3f}  macro-F1={results[name]['macro_f1']:.3f}")
    print(f"NLI AUROC (supported vs not) = {results['nli_auroc_supported']:.3f}\nSaved {out}")


if __name__ == "__main__":
    main()
