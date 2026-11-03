"""Command line:

python -m veritas ingest data/sample_docs [--collection NAME]
python -m veritas ask "your question" [--collection NAME] [--strategy role_instruct] [--samples 4]
python -m veritas docs [--collection NAME]
"""

import argparse
import json
from datetime import datetime

from .config import settings
from .pipeline import Veritas
from .prompts import ANSWER_STRATEGIES


def main() -> None:
    ap = argparse.ArgumentParser(prog="veritas")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("paths", nargs="+")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")
    p_ask.add_argument("--strategy", default="role_instruct", choices=list(ANSWER_STRATEGIES))
    p_ask.add_argument("--samples", type=int, default=None)
    p_ask.add_argument("--no-rerank", action="store_true")
    p_ask.add_argument("--no-correct", action="store_true")
    p_ask.add_argument("--lay", action="store_true", help="add a plain-language summary")

    sub.add_parser("docs")
    for p in (p_ing, p_ask, sub.choices["docs"]):
        p.add_argument("--collection", default="default")
    args = ap.parse_args()

    v = Veritas(args.collection)
    if args.cmd == "ingest":
        for name, n in v.ingest(args.paths).items():
            print(f"  {name}: {n} new chunks")
        print(f"Collection '{args.collection}' now holds {v.index.count()} chunks.")
    elif args.cmd == "docs":
        for name, n in v.index.documents().items():
            print(f"  {name}: {n} chunks")
    else:
        rep = v.run(args.question, strategy=args.strategy, n_samples=args.samples,
                    use_reranker=not args.no_rerank, correct=not args.no_correct, lay_summary=args.lay)
        settings.ensure_dirs()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        md = settings.output_dir / f"report-{stamp}.md"
        md.write_text(rep.to_markdown())
        (settings.output_dir / f"report-{stamp}.json").write_text(json.dumps(rep.to_dict(), indent=2, default=str))
        print(rep.to_markdown())
        print(f"\nTimings (s): {rep.timings}\nSaved: {md}")


if __name__ == "__main__":
    main()
