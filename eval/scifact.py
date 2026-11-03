"""SciFact benchmark helpers (Wadden et al., 2020).

5,183 scientific abstracts + expert-written claims labelled SUPPORT / CONTRADICT
(or no evidence = NOT ENOUGH INFO) against specific abstracts.
"""

import json
from pathlib import Path

from veritas.config import ROOT

DATA = ROOT / "data" / "scifact" / "data"
COLLECTION = "scifact"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_corpus() -> dict[int, dict]:
    return {d["doc_id"]: d for d in _jsonl(DATA / "corpus.jsonl")}


def doc_text(doc: dict) -> str:
    return f"{doc['title']}. " + " ".join(doc["abstract"])


def doc_name(doc_id: int) -> str:
    return f"scifact-{doc_id}"


def load_claims(split: str = "dev") -> list[dict]:
    """Flatten to one row per (claim, cited abstract) with a gold label."""
    rows = []
    for c in _jsonl(DATA / f"claims_{split}.jsonl"):
        for did in c["cited_doc_ids"]:
            ev = c["evidence"].get(str(did), [])
            label = ev[0]["label"] if ev else "NOT_ENOUGH_INFO"
            rationale = sorted({s for e in ev for s in e["sentences"]})
            rows.append({"claim_id": c["id"], "claim": c["claim"], "doc_id": did, "label": label, "rationale": rationale})
    return rows


def build_index(veritas) -> int:
    """Index the full SciFact corpus into its own collection (one-off, ~3 min on CPU)."""
    from veritas.ingest import ingest_text

    if veritas.index.count() > 0:
        return veritas.index.count()
    chunks = [ch for did, d in load_corpus().items() for ch in ingest_text(doc_name(did), doc_text(d))]
    veritas.index.add(chunks)
    return veritas.index.count()
