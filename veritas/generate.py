"""Phase 3c: grounded generation.

The decoder-only LLM writes an answer conditioned on the retrieved evidence.
Also provides the closed-book baseline (RQ1) and stochastic re-samples used by
the self-consistency check.
"""

import re

from . import prompts
from .config import settings
from .llm import LLM
from .retrieve import Evidence, format_evidence


def _messages(strategy: str, question: str, evidence: list[Evidence]) -> list[dict]:
    spec = prompts.ANSWER_STRATEGIES[strategy]
    user = spec["user"].format(question=question, evidence=format_evidence(evidence), rules=prompts.GROUNDING_RULES)
    msgs = [{"role": "system", "content": spec["system"]}] if spec["system"] else []
    return msgs + [{"role": "user", "content": user}]


_ALT_CITE = re.compile(r"【\s*(S\d+)[^】]*】")
_PAREN_CITE = re.compile(r"\(\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\)")


def normalize_citations(text: str) -> str:
    """Models sometimes cite as 【S3】, 【S3†L1】 or (S2, S5); convert all to the [S3] form the pipeline expects."""
    text = _ALT_CITE.sub(r"[\1]", text)
    return _PAREN_CITE.sub(lambda m: "".join(f"[{s}]" for s in re.findall(r"S\d+", m.group(1))), text)


def _strip_reasoning(text: str) -> str:
    """For chain-of-thought output keep only the final answer section."""
    marker = text.upper().rfind("ANSWER:")
    return text[marker + len("ANSWER:"):].strip() if marker != -1 else text.strip()


def generate_answer(llm: LLM, question: str, evidence: list[Evidence], strategy: str = "role_instruct",
                    temperature: float | None = None, sample_idx: int = 0) -> str:
    temperature = settings.gen_temperature if temperature is None else temperature
    out = llm.chat(_messages(strategy, question, evidence), temperature=temperature, sample_idx=sample_idx)
    return normalize_citations(_strip_reasoning(out) if strategy == "cot" else out.strip())


def sample_answers(llm: LLM, question: str, evidence: list[Evidence], n: int | None = None,
                   strategy: str = "role_instruct") -> list[str]:
    """N extra answers at high temperature for the self-consistency check."""
    n = settings.n_consistency_samples if n is None else n
    return [generate_answer(llm, question, evidence, strategy, settings.sample_temperature, sample_idx=i + 1)
            for i in range(n)]


def closed_book_answer(llm: LLM, question: str) -> str:
    msg = [{"role": "user", "content": prompts.CLOSED_BOOK.format(question=question)}]
    return llm.chat(msg, temperature=settings.gen_temperature).strip()
