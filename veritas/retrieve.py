"""Phase 3a: two-stage retrieval.

Stage 1 - bi-encoder dense search (fast, recall-oriented).
Stage 2 - cross-encoder re-ranking (slow, precision-oriented): the transformer
reads query and passage together and scores their relevance jointly.
"""

from dataclasses import dataclass, asdict
from functools import lru_cache

from .config import settings
from .index import VectorIndex
from .ingest import Chunk


@dataclass
class Evidence:
    sid: str            # citation handle shown to the LLM: S1, S2, ...
    chunk: Chunk
    dense_score: float
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["chunk"] = self.chunk.to_dict()
        return d


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def retrieve(index: VectorIndex, query: str, k: int | None = None,
             rerank_k: int | None = None, use_reranker: bool = True, docs: list[str] | None = None) -> list[Evidence]:
    k = k or settings.retrieve_k
    rerank_k = rerank_k or settings.rerank_k
    hits = index.query(query, k, docs)
    if not hits:
        return []
    if docs and len(docs) <= 3:
        # Broad questions ("what are the main findings?") match the abstract poorly by keywords,
        # so the first page of each selected document always competes in re-ranking.
        seen = {c.id for c, _ in hits}
        hits += [h for h in index.first_page(docs) if h[0].id not in seen]

    if use_reranker:
        scores = get_reranker().predict([(query, c.text) for c, _ in hits], show_progress_bar=False)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        candidates = [(c, d, float(s)) for (c, d), s in ranked]
    else:
        candidates = [(c, d, None) for c, d in hits]
    n_docs = len({c.doc_name for c, _, _ in candidates})
    # with a single selected document the per-document cap would starve the answer of evidence
    picked = _diversify(candidates, rerank_k, settings.max_per_doc if n_docs > 1 else rerank_k)

    return [Evidence(sid=f"S{i + 1}", chunk=c, dense_score=d, rerank_score=r)
            for i, (c, d, r) in enumerate(picked)]


def _diversify(candidates: list, k: int, max_per_doc: int) -> list:
    """Keep rank order but take at most max_per_doc passages per document, so the
    LLM sees several sources (needed for balanced answers and conflict detection)."""
    picked, per_doc, overflow = [], {}, []
    for cand in candidates:
        doc = cand[0].doc_name
        if per_doc.get(doc, 0) < max_per_doc:
            picked.append(cand)
            per_doc[doc] = per_doc.get(doc, 0) + 1
        else:
            overflow.append(cand)
        if len(picked) == k:
            return picked
    return picked + overflow[:k - len(picked)]   # too few documents: fill with the best of the rest


def format_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{e.sid}] ({e.chunk.doc_name}, p.{e.chunk.page})\n{e.chunk.text}" for e in evidence)
