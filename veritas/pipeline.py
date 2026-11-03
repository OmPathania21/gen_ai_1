"""The VERITAS pipeline: one fixed, single-pass sequence of steps.

retrieve -> generate -> extract claims -> NLI + self-consistency -> score
-> conflicts -> (one correction pass) -> report

There is no planner, no tool selection and no loop: every question runs
through exactly the same steps.
"""

import time
from pathlib import Path

from . import prompts
from .claims import extract_claims
from .config import settings
from .generate import generate_answer, normalize_citations, sample_answers
from .index import VectorIndex
from .ingest import SUPPORTED, ingest_file
from .llm import LLM
from .report import ClaimResult, Report
from .retrieve import Evidence, format_evidence, retrieve
from .scoring import find_conflicts
from .verify import NLI, get_nli, verify_claims


class Veritas:
    def __init__(self, collection: str = "default", llm: LLM | None = None, nli: NLI | None = None):
        self.index = VectorIndex(collection)
        self._llm = llm
        self._nli = nli

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    @property
    def nli(self) -> NLI:
        if self._nli is None:
            self._nli = get_nli()
        return self._nli

    # ---------------------------------------------------------------- ingestion
    def ingest(self, paths) -> dict[str, int]:
        added = {}
        for p in map(Path, paths):
            if p.is_dir():
                added.update(self.ingest(sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED)))
            elif p.suffix.lower() in SUPPORTED:
                added[p.name] = self.index.add(ingest_file(p))
        return added

    # ---------------------------------------------------------------- main run
    def run(self, question: str, strategy: str = "role_instruct", n_samples: int | None = None,
            use_reranker: bool = True, correct: bool | None = None, explain_conflicts: bool = True,
            lay_summary: bool = False, progress=None, docs: list[str] | None = None) -> Report:
        """progress: optional callback(str) told which step is starting (used by the web UI)."""
        say = progress or (lambda msg: None)
        n_samples = settings.n_consistency_samples if n_samples is None else n_samples
        correct = settings.regenerate_weak_claims if correct is None else correct
        t: dict[str, float] = {}

        def tick(name, start):
            t[name] = round(time.perf_counter() - start, 2)

        say("Retrieving and re-ranking evidence")
        s = time.perf_counter()
        evidence = retrieve(self.index, question, use_reranker=use_reranker, docs=docs)
        tick("retrieve", s)
        if not evidence:
            return Report(question=question, answer="No documents are indexed yet. Upload documents first.",
                          evidence=[], claims=[], strategy=strategy, timings=t)

        say("Generating grounded answer")
        s = time.perf_counter()
        answer = generate_answer(self.llm, question, evidence, strategy)
        tick("generate", s)

        say("Extracting atomic claims")
        s = time.perf_counter()
        claims = extract_claims(self.llm, answer)
        tick("claims", s)

        if n_samples:
            say(f"Sampling {n_samples} answers for self-consistency")
        s = time.perf_counter()
        samples = sample_answers(self.llm, question, evidence, n_samples, strategy) if n_samples else []
        tick("samples", s)

        say("Verifying claims with NLI")
        s = time.perf_counter()
        results = verify_claims(self.nli, claims, evidence, samples)
        conflicts = find_conflicts(results, evidence)
        tick("verify", s)

        report = Report(question=question, answer=answer, evidence=evidence, claims=results,
                        conflicts=conflicts, samples=samples, strategy=strategy, timings=t)

        if explain_conflicts and conflicts:
            say("Explaining conflicting sources")
            s = time.perf_counter()
            self._explain_conflicts(report)
            tick("conflicts", s)

        weak = [r for r in results if r.label in ("unsupported", "contradicted")]
        if correct and weak:
            say("Correcting weak claims (single pass)")
            s = time.perf_counter()
            report.revised_answer = self._correct_once(question, answer, weak, evidence)
            report.revised_claims = verify_claims(self.nli, extract_claims(self.llm, report.revised_answer), evidence)
            tick("correct", s)

        if lay_summary:
            say("Writing plain-language summary")
            report.lay_summary = self.lay_summary(report)
        t["total"] = round(sum(t.values()), 2)
        return report

    # ---------------------------------------------------------------- helpers
    def _correct_once(self, question: str, answer: str, weak: list[ClaimResult], evidence: list[Evidence]) -> str:
        weak_txt = "\n".join(f"- {w.text} (verdict: {w.label})" for w in weak)
        msg = prompts.REGENERATE.format(question=question, answer=answer, weak_claims=weak_txt,
                                        evidence=format_evidence(evidence))
        return normalize_citations(self.llm.chat([{"role": "system", "content": prompts.SYSTEM_ROLE},
                                                  {"role": "user", "content": msg}], temperature=settings.gen_temperature).strip())

    def _explain_conflicts(self, report: Report, limit: int = 3) -> None:
        by_sid = {e.sid: e for e in report.evidence}
        for cf in report.conflicts[:limit]:
            a, b = by_sid[cf.support_sid], by_sid[cf.contradict_sid]
            msg = prompts.CONFLICT_EXPLAIN.format(
                claim=cf.claim, source_a=f"{a.chunk.doc_name}, p.{a.chunk.page}", text_a=a.chunk.text,
                source_b=f"{b.chunk.doc_name}, p.{b.chunk.page}", text_b=b.chunk.text)
            cf.explanation = self.llm.chat([{"role": "user", "content": msg}], temperature=0.2).strip()

    def explain_claim(self, report: Report, claim: ClaimResult) -> str:
        """Reasoning prompt: plain-language justification of one verdict (used on demand by the UI)."""
        ev = next((e for e in report.evidence if e.sid == claim.best_source), None)
        msg = prompts.CONFIDENCE_EXPLAIN.format(
            claim=claim.text, label=claim.label, confidence=claim.confidence, entail=claim.entail,
            contradict=claim.contradict,
            consistency=(f"{claim.consistency:.2f} across {len(report.samples)} re-generated answers"
                         if claim.consistency is not None else "not computed (fast mode)"),
            source=claim.best_source, evidence=ev.chunk.text if ev else "none")
        claim.explanation = self.llm.chat([{"role": "user", "content": msg}], temperature=0.2).strip()
        return claim.explanation

    def lay_summary(self, report: Report) -> str:
        claims = report.revised_claims or report.claims
        findings = "\n".join(f"- {c.text}" for c in claims if c.label == "verified")
        if not findings:
            return "No claims could be verified from the provided documents."
        msg = prompts.PLAIN_SUMMARY.format(findings=findings)
        return self.llm.chat([{"role": "user", "content": msg}], temperature=0.7).strip()
