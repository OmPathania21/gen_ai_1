"""Phase 4c: confidence scoring + conflict detection.

confidence = w_nli * P(entail) + w_cons * consistency - penalty * P(contradict), clipped to [0, 1]
If self-consistency was not computed, the NLI term carries the full weight.
"""

from .config import settings


def confidence_score(entail: float, contradict: float, consistency: float | None) -> float:
    if consistency is None:
        raw = entail - settings.contradiction_penalty * contradict
    else:
        raw = (settings.w_nli * entail + settings.w_consistency * consistency
               - settings.contradiction_penalty * contradict)
    return max(0.0, min(1.0, raw))


def verdict(conf: float, entail: float, contradict: float) -> str:
    if contradict >= 0.5 and contradict > entail:
        return "contradicted"
    if conf >= settings.verified_threshold:
        return "verified"
    if conf < settings.unsupported_threshold:
        return "unsupported"
    return "uncertain"


def find_conflicts(results, evidence, threshold: float = 0.5):
    """A conflict = one source entails the claim while a source from a DIFFERENT document contradicts it."""
    from .report import Conflict

    doc = {e.sid: e.chunk.doc_name for e in evidence}
    conflicts = []
    for r in results:
        sup = [s for s, p in r.per_source.items() if p["entail"] >= threshold]
        con = [s for s, p in r.per_source.items() if p["contradict"] >= threshold]
        pair = next(((a, b) for a in sup for b in con if doc[a] != doc[b]), None)
        if pair:
            conflicts.append(Conflict(claim=r.text, support_sid=pair[0], contradict_sid=pair[1],
                                      support_doc=doc[pair[0]], contradict_doc=doc[pair[1]]))
    return conflicts
