"""Phase 4a: claim extraction.

The LLM decomposes its own answer into atomic, self-contained claims, each
carrying the [S#] citations it was written with. A regex sentence splitter is
the fallback if the LLM returns malformed JSON.
"""

import re
from dataclasses import dataclass, field

from . import prompts
from .ingest import split_sentences
from .llm import LLM

_TAG = re.compile(r"\[(S\d+)\]")
_INSUFFICIENT = re.compile(r"(evidence|sources?|passages?)\s+(is|are|was|were)\s+insufficient|(evidence|sources?|passages?)\s+(does|do|did)\s*(not|n't)|cannot be determined|not (stated|reported|provided) in the|\b(is|are|was|were) (not )?provided\b|\bprovided (evidence|sources?|passages?)\b", re.I)


@dataclass
class Claim:
    text: str
    citations: list[str] = field(default_factory=list)


def _fallback(answer: str) -> list[Claim]:
    out = []
    for sent in split_sentences(answer):
        if _INSUFFICIENT.search(sent):
            continue
        cites = _TAG.findall(sent)
        text = _TAG.sub("", sent).strip()
        if len(text.split()) >= 4:
            out.append(Claim(text=text, citations=cites))
    return out


def extract_claims(llm: LLM, answer: str) -> list[Claim]:
    if not answer.strip():
        return []
    try:
        data = llm.chat_json([{"role": "user", "content": prompts.CLAIM_EXTRACTION.format(answer=answer)}])
        claims = [
            Claim(text=_TAG.sub("", c["claim"]).strip(), citations=[s for s in c.get("citations", []) if _TAG.fullmatch(f"[{s}]")])
            for c in data.get("claims", []) if isinstance(c, dict) and c.get("claim")
        ]
        claims = [c for c in claims if not _INSUFFICIENT.search(c.text)]
        return claims or _fallback(answer)
    except Exception:
        return _fallback(answer)
