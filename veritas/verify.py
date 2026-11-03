"""Phase 4b: NLI evidence verification + self-consistency.

NLI (natural language inference): a DeBERTa-v3 transformer encoder fine-tuned
on MNLI/FEVER/ANLI reads (premise, hypothesis) and outputs
P(entailment), P(neutral), P(contradiction).

- Evidence check:   premise = retrieved passage,  hypothesis = claim
- Self-consistency: premise = re-sampled answer,  hypothesis = claim
  (SelfCheckGPT-NLI: a claim the model believes will reappear in independent
  samples; a hallucinated one usually will not.)
"""

from functools import lru_cache

from .claims import Claim
from .config import settings
from .ingest import split_sentences
from .report import ClaimResult
from .retrieve import Evidence
from .scoring import confidence_score, verdict


class NLI:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        # A GPU (NVIDIA CUDA or Apple-silicon MPS) is ~6x faster than CPU for this model (same outputs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        name = model_name or settings.nli_model
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = device
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.idx = {"entail": labels["entailment"], "neutral": labels["neutral"], "contradict": labels["contradiction"]}
        self._cache: dict[tuple[str, str], dict] = {}

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[dict]:
        # length-sorted batches: short windows are not padded to the length of whole passages
        todo = sorted((p for p in dict.fromkeys(pairs) if p not in self._cache), key=lambda p: len(p[0]) + len(p[1]))
        for i in range(0, len(todo), batch_size):
            part = todo[i:i + batch_size]
            enc = self.tok([p for p, _ in part], [h for _, h in part], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for pair, pr in zip(part, probs):
                self._cache[pair] = {k: pr[i] for k, i in self.idx.items()}
        return [self._cache[p] for p in pairs]


@lru_cache(maxsize=1)
def get_nli() -> NLI:
    return NLI()


def premises(text: str) -> list[str]:
    """The whole passage plus overlapping sentence windows (SummaC-style, Laban et al. 2022).
    Specific facts are judged best against the 1-3 sentences that state them; claims that
    summarise a passage are judged best against the whole passage. We keep the best match."""
    n, stride = settings.nli_window_sentences, settings.nli_window_stride
    sents = split_sentences(text)
    if n <= 0 or len(sents) <= n:
        return [text]
    return [text] + [" ".join(sents[i:i + n]) for i in range(0, len(sents) - n + 1, stride)]


def verify_claims(nli: NLI, claims: list[Claim], evidence: list[Evidence],
                  samples: list[str] | None = None) -> list[ClaimResult]:
    if not claims:
        return []
    samples = samples or []
    by_sid = {e.sid: e for e in evidence}

    # stage 1 (cheap): every claim against every whole passage, plus every self-consistency sample
    whole = nli.predict([(e.chunk.text, c.text) for c in claims for e in evidence])
    sm_pairs = [(s, c.text) for c in claims for s in samples]
    sm_probs = nli.predict(sm_pairs) if sm_pairs else []
    per_sources = [{e.sid: whole[ci * len(evidence) + ei] for ei, e in enumerate(evidence)} for ci in range(len(claims))]

    # stage 2 (precise): sentence windows, only for the passages a claim cites plus its best whole-passage matches
    top = settings.nli_window_top or len(evidence)
    focus = [sorted({s for s in c.citations if s in by_sid} | set(sorted(ps, key=lambda s: -ps[s]["entail"])[:top]))
             for c, ps in zip(claims, per_sources)]
    win_pairs = [(w, c.text) for c, sids in zip(claims, focus) for sid in sids for w in premises(by_sid[sid].chunk.text)[1:]]
    win_probs = iter(nli.predict(win_pairs)) if win_pairs else iter(())
    for c, sids, ps in zip(claims, focus, per_sources):
        for sid in sids:
            for _ in premises(by_sid[sid].chunk.text)[1:]:
                p = next(win_probs)
                if p["entail"] > ps[sid]["entail"]:
                    ps[sid] = p

    results = []
    for ci, claim in enumerate(claims):
        per_source = per_sources[ci]
        best_sid = max(per_source, key=lambda s: per_source[s]["entail"])
        best = per_source[best_sid]
        # If a passage clearly supports the claim, judge it by that passage: other passages that merely
        # omit the fact are often scored "contradiction" by NLI. Genuine cross-document disagreement is
        # still caught by find_conflicts(). With no supporting passage, any contradiction counts.
        contradict = (best["contradict"] if best["entail"] >= 0.5
                      else max(p["contradict"] for p in per_source.values()))

        cited = [s for s in claim.citations if s in by_sid]
        citation_correct = any(per_source[s]["entail"] >= 0.5 for s in cited) if cited else None

        consistency = None
        if samples:
            sp = sm_probs[ci * len(samples):(ci + 1) * len(samples)]
            consistency = sum(p["entail"] for p in sp) / len(sp)

        conf = confidence_score(best["entail"], contradict, consistency)
        results.append(ClaimResult(
            text=claim.text, citations=claim.citations, entail=best["entail"], neutral=best["neutral"],
            contradict=contradict, best_source=best_sid, citation_correct=citation_correct,
            consistency=consistency, confidence=conf, label=verdict(conf, best["entail"], contradict),
            per_source=per_source,
        ))
    return results
