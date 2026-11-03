"""VERITAS HTTP API (FastAPI).

Run (dev):   uvicorn api.main:app --reload --port 8000
The React app in web/ talks to these /api routes; in production the built
frontend (web/dist) is served by this same server, so one container hosts both.

Long pipeline runs execute as background jobs on a single worker thread:
the NLI model and the LLM rate limit are shared resources, so runs are queued
rather than executed in parallel. Clients poll GET /api/jobs/{id}.
"""

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from veritas import prompts
from veritas.claims import extract_claims
from veritas.config import ROOT, settings
from veritas.generate import generate_answer
from veritas.ingest import SUPPORTED
from veritas.pipeline import Veritas
from veritas.report import Report
from veritas.retrieve import retrieve
from veritas.verify import verify_claims

COLLECTIONS = {"default": "Your documents", "scifact": "SciFact benchmark (5,183 abstracts)"}
MAX_UPLOAD_MB = 25
MAX_JOBS_KEPT = 200

app = FastAPI(title="VERITAS API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

worker = ThreadPoolExecutor(max_workers=1)   # one pipeline at a time (see module docstring)
_veritas: dict[str, Veritas] = {}


def get_veritas(collection: str) -> Veritas:
    if collection not in COLLECTIONS:
        raise HTTPException(404, f"Unknown collection '{collection}'")
    if collection not in _veritas:
        _veritas[collection] = Veritas(collection)
    return _veritas[collection]


async def on_worker(fn, *args, **kwargs):
    return await asyncio.wrap_future(worker.submit(fn, *args, **kwargs))


# ---------------------------------------------------------------- jobs
@dataclass
class Job:
    id: str
    kind: str                       # "ask" | "compare"
    params: dict
    status: str = "queued"          # queued | running | done | error
    steps: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    report: Report | None = None    # kept server-side for /explain


JOBS: dict[str, Job] = {}


def submit(kind: str, params: dict, fn) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
    JOBS[job.id] = job
    if len(JOBS) > MAX_JOBS_KEPT:  # drop the oldest finished jobs
        for old in sorted((j for j in JOBS.values() if j.status in ("done", "error")), key=lambda j: j.created)[:50]:
            JOBS.pop(old.id, None)

    def run():
        job.status, job.started = "running", time.time()
        try:
            fn(job)
            job.status = "done"
        except Exception as e:  # surface API/rate-limit errors to the client
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
        finally:
            job.finished = time.time()

    worker.submit(run)
    return job


def job_view(job: Job) -> dict:
    ahead = sum(1 for j in JOBS.values() if j.status in ("queued", "running") and j.created < job.created)
    return {"id": job.id, "kind": job.kind, "status": job.status, "steps": job.steps, "queue_position": ahead,
            "elapsed": round((job.finished or time.time()) - (job.started or time.time()), 1),
            "params": job.params, "result": job.result, "error": job.error}


def report_view(rep: Report) -> dict:
    d = rep.to_dict()
    d["summary"] = rep.summary
    return d


# ---------------------------------------------------------------- schemas
MODES = {
    "fast": {"n_samples": 0, "correct": False},
    "full": {"n_samples": settings.n_consistency_samples, "correct": True},
}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    collection: str = "default"
    mode: str = "fast"                       # fast | full | custom
    strategy: str = "role_instruct"
    n_samples: int | None = Field(default=None, ge=0, le=6)
    correct: bool | None = None
    rerank: bool = True
    lay_summary: bool = False
    docs: list[str] | None = None            # search only these documents (None = all)


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    collection: str = "default"
    strategies: list[str] = ["bad_vague", "role_instruct", "cot"]


class ExplainRequest(BaseModel):
    claim_index: int
    revised: bool = False


# ---------------------------------------------------------------- routes
@app.get("/api/info")
def info():
    return {
        "generator": settings.llm_model, "embedder": settings.embed_model, "reranker": settings.rerank_model,
        "verifier": settings.nli_model, "modes": MODES, "consistency_samples": settings.n_consistency_samples,
        "thresholds": {"verified": settings.verified_threshold, "unsupported": settings.unsupported_threshold},
        "weights": {"nli": settings.w_nli, "consistency": settings.w_consistency,
                    "contradiction_penalty": settings.contradiction_penalty},
        "collections": [{"id": c, "name": n, "chunks": get_veritas(c).index.count()} for c, n in COLLECTIONS.items()],
        "busy": sum(1 for j in JOBS.values() if j.status in ("queued", "running")),
    }


@app.get("/api/documents")
def documents(collection: str = "default"):
    return [{"name": n, "chunks": c} for n, c in get_veritas(collection).index.documents().items()]


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...), collection: str = "default"):
    if collection == "scifact":
        raise HTTPException(400, "The benchmark collection is read-only.")
    v = get_veritas(collection)
    folder = settings.data_dir / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"{name}: unsupported type (use {', '.join(sorted(SUPPORTED))})")
        data = await f.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"{name}: larger than {MAX_UPLOAD_MB} MB")
        (folder / name).write_bytes(data)
        saved.append(folder / name)
    try:
        added = await on_worker(v.ingest, saved)
    except Exception as e:
        raise HTTPException(422, f"Could not read file: {e}")
    return {"added": added, "total_chunks": v.index.count()}


@app.delete("/api/documents/{name}")
def delete_document(name: str, collection: str = "default"):
    if collection == "scifact":
        raise HTTPException(400, "The benchmark collection is read-only.")
    get_veritas(collection).index.delete_document(name)
    return {"deleted": name}


@app.post("/api/ask")
def ask(req: AskRequest):
    v = get_veritas(req.collection)
    if v.index.count() == 0:
        raise HTTPException(400, "No documents indexed yet. Upload documents first.")
    if req.strategy not in prompts.ANSWER_STRATEGIES:
        raise HTTPException(400, f"Unknown strategy '{req.strategy}'")
    preset = MODES.get(req.mode, MODES["full"])
    n_samples = req.n_samples if req.mode == "custom" and req.n_samples is not None else preset["n_samples"]
    correct = req.correct if req.mode == "custom" and req.correct is not None else preset["correct"]

    def run(job: Job):
        rep = v.run(req.question.strip(), strategy=req.strategy, n_samples=n_samples, use_reranker=req.rerank,
                    correct=correct, lay_summary=req.lay_summary, progress=job.steps.append,
                    docs=req.docs or None)
        job.report, job.result = rep, report_view(rep)

    job = submit("ask", req.model_dump() | {"n_samples": n_samples, "correct": correct}, run)
    return job_view(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (it may have expired).")
    return job_view(job)


@app.post("/api/jobs/{job_id}/explain")
async def explain(job_id: str, req: ExplainRequest):
    job = JOBS.get(job_id)
    if not job or not job.report:
        raise HTTPException(404, "Report not found.")
    claims = job.report.revised_claims if req.revised else job.report.claims
    if not 0 <= req.claim_index < len(claims):
        raise HTTPException(400, "claim_index out of range")
    v = get_veritas(job.params["collection"])
    text = await on_worker(v.explain_claim, job.report, claims[req.claim_index])
    return {"explanation": text}


@app.get("/api/jobs/{job_id}/report.md", response_class=PlainTextResponse)
def report_md(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.report:
        raise HTTPException(404, "Report not found.")
    return job.report.to_markdown()


@app.get("/api/prompts")
def prompt_catalog():
    return {"catalog": prompts.PROMPT_CATALOG,
            "strategies": [{"id": k, "type": v["type"], "system": v["system"], "user": v["user"]}
                           for k, v in prompts.ANSWER_STRATEGIES.items()],
            "grounding_rules": prompts.GROUNDING_RULES}


@app.post("/api/compare")
def compare(req: CompareRequest):
    v = get_veritas(req.collection)
    bad = [s for s in req.strategies if s not in prompts.ANSWER_STRATEGIES]
    if bad or not req.strategies:
        raise HTTPException(400, f"Unknown strategies: {bad}")

    def run(job: Job):
        job.steps.append("Retrieving shared evidence")
        evidence = retrieve(v.index, req.question)
        rows = []
        for s in req.strategies:
            job.steps.append(f"Generating and verifying: {s}")
            ans = generate_answer(v.llm, req.question, evidence, s)
            res = verify_claims(v.nli, extract_claims(v.llm, ans), evidence)
            rows.append({"strategy": s, "type": prompts.ANSWER_STRATEGIES[s]["type"], "answer": ans,
                         "claims": [{"text": c.text, "label": c.label, "confidence": c.confidence} for c in res],
                         **Report.stats(res)})
        job.result = {"question": req.question, "rows": rows}

    return job_view(submit("compare", req.model_dump(), run))


@app.get("/api/evaluation")
def evaluation():
    out = {}
    for name in ("rq2_verifier", "prompt_compare", "rq_generation_summary"):
        p = settings.output_dir / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


@app.get("/api/health")
def health():
    return {"ok": True}


# ---------------------------------------------------------------- frontend (production build)
DIST = ROOT / "web" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
