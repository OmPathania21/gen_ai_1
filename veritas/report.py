"""Phase 4e: structured report (JSON + Markdown)."""

from dataclasses import dataclass, field, asdict

from .retrieve import Evidence

ICON = {"verified": "✅", "uncertain": "⚠️", "unsupported": "❌", "contradicted": "⛔"}


@dataclass
class ClaimResult:
    text: str
    citations: list[str]
    entail: float                   # best NLI entailment over all evidence
    neutral: float
    contradict: float               # strongest NLI contradiction over all evidence
    best_source: str | None         # sid of the passage that best supports the claim
    citation_correct: bool | None   # does at least one cited passage entail the claim? (None = no citation)
    consistency: float | None       # self-consistency score (None = not computed)
    confidence: float
    label: str                      # verified | uncertain | unsupported | contradicted
    per_source: dict = field(default_factory=dict)  # sid -> {entail, neutral, contradict}
    explanation: str | None = None


@dataclass
class Conflict:
    claim: str
    support_sid: str
    contradict_sid: str
    support_doc: str
    contradict_doc: str
    explanation: str | None = None


@dataclass
class Report:
    question: str
    answer: str
    evidence: list[Evidence]
    claims: list[ClaimResult]
    conflicts: list[Conflict] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)
    revised_answer: str | None = None
    revised_claims: list[ClaimResult] = field(default_factory=list)
    lay_summary: str | None = None
    strategy: str = "role_instruct"
    timings: dict = field(default_factory=dict)

    @staticmethod
    def stats(claims: list[ClaimResult]) -> dict:
        n = len(claims)
        count = {k: sum(c.label == k for c in claims) for k in ICON}
        cited = [c.citation_correct for c in claims if c.citation_correct is not None]
        return {
            "n_claims": n,
            **count,
            "hallucination_rate": (count["unsupported"] + count["contradicted"]) / n if n else 0.0,
            "mean_confidence": sum(c.confidence for c in claims) / n if n else 0.0,
            "citation_accuracy": sum(cited) / len(cited) if cited else None,
        }

    @property
    def summary(self) -> dict:
        s = {"original": self.stats(self.claims)}
        if self.revised_claims:
            s["revised"] = self.stats(self.revised_claims)
        return s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [e.to_dict() for e in self.evidence]
        d["summary"] = self.summary
        return d

    def to_markdown(self) -> str:
        src = {e.sid: f"{e.chunk.doc_name}, p.{e.chunk.page}" for e in self.evidence}
        s = self.summary["original"]
        lines = [f"# VERITAS Verified Report", "", f"**Question:** {self.question}", "",
                 "## Answer", "", self.revised_answer or self.answer, ""]
        if self.revised_answer:
            lines += ["<details><summary>Original (uncorrected) answer</summary>", "", self.answer, "", "</details>", ""]
        lines += ["## Claim verification", "", "| | Claim | Confidence | NLI entail | Consistency | Sources |",
                  "|---|---|---|---|---|---|"]
        for c in self.claims:
            cons = f"{c.consistency:.2f}" if c.consistency is not None else "–"
            lines.append(f"| {ICON[c.label]} | {c.text} | {c.confidence:.2f} | {c.entail:.2f} | {cons} | {', '.join(c.citations) or '–'} |")
        lines += ["", f"**Claims:** {s['n_claims']} · ✅ {s['verified']} · ⚠️ {s['uncertain']} · ❌ {s['unsupported']} · "
                      f"⛔ {s['contradicted']} · hallucination rate {s['hallucination_rate']:.0%}", ""]
        if self.conflicts:
            lines += ["## Conflicting evidence", ""]
            for cf in self.conflicts:
                lines.append(f"- **{cf.claim}** — supported by {cf.support_sid} ({cf.support_doc}), "
                             f"contradicted by {cf.contradict_sid} ({cf.contradict_doc}). {cf.explanation or ''}")
            lines.append("")
        if self.lay_summary:
            lines += ["## Plain-language summary", "", self.lay_summary, ""]
        lines += ["## Sources", ""] + [f"- [{sid}] {loc}" for sid, loc in src.items()]
        return "\n".join(lines)
