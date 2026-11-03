import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type Claim, type Evidence, type Info, type Report } from "../api";
import { ErrorBox, Meter, Spinner, Stat, VERDICT, VERDICT_FILL, VerdictBadge, VerdictBar, pct, useJob } from "../components/ui";

const DEMO_QUESTIONS = [
  "How can hallucinations in large language models be detected, and how effective are these methods?",
  "What is TruthfulQA, and how did larger language models perform on it?",
];
const GENERIC_QUESTIONS = [
  "What are the main findings of this document?",
  "What methods or approach does the document use?",
  "What limitations or open problems does it mention?",
];

type Mode = "fast" | "full";
type Doc = { name: string; chunks: number };

function expectedSteps(mode: Mode, samples: number) {
  const s = ["Retrieving and re-ranking evidence", "Generating grounded answer", "Extracting atomic claims"];
  if (mode === "full") s.push(`Sampling ${samples} answers for self-consistency`);
  s.push("Verifying claims with NLI");
  if (mode === "full") s.push("Correcting weak claims (single pass)");
  return s;
}

export default function Ask({ info, onLibraryChange }: { info: Info | null; onLibraryChange: () => void }) {
  const collection = "default";
  const [docs, setDocs] = useState<Doc[] | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [drag, setDrag] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("fast");
  // ?job=<id> in the URL reopens an earlier result (shareable link)
  const [jobId, setJobIdState] = useState<string | null>(
    () => new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("job"));
  const setJobId = (id: string) => {
    setJobIdState(id);
    history.replaceState(null, "", `#/ask?job=${id}`);
  };
  const [submitError, setSubmitError] = useState<string | null>(null);
  const { job, error } = useJob<Report>(jobId);
  const running = job?.status === "queued" || job?.status === "running";
  const samples = info?.consistency_samples ?? 4;

  const loadDocs = useCallback(async (select?: string[]) => {
    const list = await api.documents(collection);
    setDocs(list);
    setSelected((prev) => select ?? (prev.length ? prev.filter((n) => list.some((d) => d.name === n)) : list.map((d) => d.name)));
  }, []);
  useEffect(() => { loadDocs().catch((e) => setSubmitError(e.message)); }, [loadDocs]);

  useEffect(() => {  // reopened result: restore its question, mode and document selection
    if (!job?.params?.question || question) return;
    setQuestion(String(job.params.question));
    if (job.params.mode === "full" || job.params.mode === "fast") setMode(job.params.mode);
    if (Array.isArray(job.params.docs) && job.params.docs.length) setSelected(job.params.docs as string[]);
  }, [job?.params?.question]);  // eslint-disable-line react-hooks/exhaustive-deps

  async function upload(files: File[]) {
    if (!files.length) return;
    setUploading(true); setSubmitError(null); setUploadMsg(null);
    try {
      const r = await api.upload(collection, files);
      const names = Object.keys(r.added).length ? Object.keys(r.added) : files.map((f) => f.name);
      await loadDocs(names);  // focus the question on the new upload
      setUploadMsg(`Added ${names.join(", ")}. Questions will be answered from ${names.length > 1 ? "these documents" : "this document"} only.`);
      onLibraryChange();
    } catch (e) {
      setSubmitError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function remove(name: string) {
    if (!confirm(`Remove "${name}"?`)) return;
    await api.deleteDocument(collection, name);
    await loadDocs();
    onLibraryChange();
  }

  const toggle = (name: string) =>
    setSelected(selected.includes(name) ? selected.filter((n) => n !== name) : [...selected, name]);

  async function run(q = question) {
    if (!q.trim() || !selected.length) return;
    setSubmitError(null);
    try {
      const all = docs && selected.length === docs.length;
      const j = await api.ask({ question: q.trim(), collection, mode, docs: all ? null : selected });
      setJobId(j.id);
    } catch (e) {
      setSubmitError((e as Error).message);
    }
  }

  const demoSelected = selected.some((n) => DEMO_DOCS.has(n));
  const suggestions = demoSelected && selected.every((n) => DEMO_DOCS.has(n)) ? DEMO_QUESTIONS : GENERIC_QUESTIONS;

  return (
    <div className="space-y-6">
      <section className="mx-auto max-w-3xl">
        <h1 className="text-center font-serif text-3xl font-bold tracking-tight sm:text-[2.6rem] sm:leading-tight">
          Ask your documents. <span className="text-amber">Trust every answer.</span>
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-center text-muted">
          Upload a PDF, ask a question, and VERITAS answers only from your document. Then it checks every claim against the source.
        </p>

        <div className="card mt-8 overflow-hidden shadow-[0_8px_30px_-12px_rgba(20,33,61,0.18)]">
          {/* ---- step 1: documents */}
          <div className="p-5 sm:p-6">
            <div className="flex items-center justify-between">
              <h2 className="flex items-center gap-2 font-semibold">
                <span className="grid h-6 w-6 place-items-center rounded-full bg-ink text-xs text-white">1</span> Your documents
              </h2>
              {docs && docs.length > 1 && (
                <button className="text-xs font-medium text-muted hover:text-ink"
                        onClick={() => setSelected(selected.length === docs.length ? [] : docs.map((d) => d.name))}>
                  {selected.length === docs.length ? "Clear selection" : "Select all"}
                </button>
              )}
            </div>

            <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
                 onDrop={(e) => { e.preventDefault(); setDrag(false); upload([...e.dataTransfer.files]); }}
                 onClick={() => !uploading && fileInput.current?.click()} role="button" tabIndex={0}
                 onKeyDown={(e) => e.key === "Enter" && fileInput.current?.click()}
                 className={`mt-4 flex cursor-pointer items-center gap-4 rounded-xl border-2 border-dashed px-4 py-4 transition ${
                   drag ? "border-amber bg-amber-soft" : "border-line hover:border-ink/30 hover:bg-paper/60"}`}>
              <input ref={fileInput} type="file" multiple accept=".pdf,.docx,.txt,.md" className="hidden"
                     onChange={(e) => { upload([...(e.target.files ?? [])]); e.target.value = ""; }} />
              <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-amber-soft text-lg text-amber">
                {uploading ? <Spinner className="text-amber" /> : "↑"}
              </span>
              <span className="text-sm">
                <span className="font-semibold">{uploading ? "Reading and indexing your file…" : "Upload a PDF"}</span>
                <span className="block text-xs text-muted">
                  {uploading ? "This takes a few seconds per document." : "Drop it here or click to browse · PDF, DOCX, TXT, MD · up to 25 MB"}
                </span>
              </span>
            </div>
            {uploadMsg && <p className="mt-3 text-sm text-verified">{uploadMsg}</p>}

            {docs && docs.length > 0 && (
              <div className="mt-4">
                <p className="mb-2 text-xs text-muted">
                  Answer from <strong className="text-ink">{selected.length}</strong> of {docs.length} documents. Click to include or exclude.
                </p>
                <div className="flex flex-wrap gap-2">
                  {docs.map((d) => {
                    const on = selected.includes(d.name);
                    return (
                      <span key={d.name} className={`group inline-flex max-w-full items-center rounded-full border text-xs transition ${
                        on ? "border-ink bg-ink text-white" : "border-line bg-white text-muted hover:border-ink/40"}`}>
                        <button onClick={() => toggle(d.name)} className="flex min-w-0 items-center gap-1.5 py-1.5 pl-3 pr-1" title={`${d.chunks} passages`}>
                          <span aria-hidden>{on ? "✓" : "+"}</span><span className="truncate">{d.name}</span>
                        </button>
                        <button onClick={() => remove(d.name)} aria-label={`Remove ${d.name}`}
                                className={`mr-1 rounded-full px-1.5 py-0.5 opacity-50 hover:opacity-100 ${on ? "hover:bg-white/15" : "hover:bg-unsupported-soft hover:text-unsupported"}`}>
                          ×
                        </button>
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* ---- step 2: question */}
          <div className="border-t border-line bg-paper/40 p-5 sm:p-6">
            <h2 className="flex items-center gap-2 font-semibold">
              <span className="grid h-6 w-6 place-items-center rounded-full bg-ink text-xs text-white">2</span> Your question
            </h2>
            <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={3}
                      onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }}
                      placeholder="Ask anything about the selected documents, e.g. What are the main findings?"
                      className="mt-4 w-full resize-none rounded-xl border border-line bg-white px-4 py-3 text-[15px] outline-none focus:border-ink/40" />
            <div className="mt-2 flex flex-wrap gap-2">
              {suggestions.map((ex) => (
                <button key={ex} onClick={() => setQuestion(ex)}
                        className="max-w-full truncate rounded-full border border-line bg-white px-3 py-1 text-xs text-muted hover:border-ink/30 hover:text-ink">
                  {ex}
                </button>
              ))}
            </div>

            <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="inline-flex rounded-xl border border-line bg-white p-1" role="radiogroup" aria-label="Verification mode">
                {([["fast", "Quick check", "~30 s"], ["full", "Deep check", "~1–2 min"]] as const).map(([m, label, hint]) => (
                  <button key={m} role="radio" aria-checked={mode === m} onClick={() => setMode(m)}
                          className={`rounded-lg px-3.5 py-1.5 text-sm font-medium transition ${
                            mode === m ? "bg-ink text-white" : "text-muted hover:text-ink"}`}>
                    {label} <span className={`ml-1 text-xs font-normal ${mode === m ? "text-white/60" : "text-muted"}`}>{hint}</span>
                  </button>
                ))}
              </div>
              <button className="btn-primary !px-6 !py-2.5" onClick={() => run()}
                      disabled={!question.trim() || running || !selected.length}>
                {running ? <><Spinner /> Checking…</> : "Get verified answer →"}
              </button>
            </div>
            <p className="mt-2 text-xs text-muted">
              {mode === "fast" ? "Answers from your documents and checks every claim against its source."
                : `Also re-asks the model ${samples} times to test consistency, and corrects weak claims once.`}
              {docs && !selected.length && <span className="ml-1 font-medium text-unsupported">Select at least one document.</span>}
            </p>
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-6xl space-y-6">
        {submitError && <ErrorBox message={submitError} />}
        {error && <ErrorBox message={error} />}
        {job && running && <Progress steps={job.steps} expected={expectedSteps(mode, samples)}
                                     queue={job.queue_position} status={job.status} elapsed={job.elapsed} />}
        {job?.status === "error" && <ErrorBox message={job.error ?? "The run failed."} />}
        {job?.status === "done" && job.result && <Result report={job.result} jobId={job.id} elapsed={job.elapsed} />}
      </div>
    </div>
  );
}

const DEMO_DOCS = new Set([
  "chain_of_verification_2023.pdf", "factscore_2023.pdf", "finetuning_new_knowledge_2024.pdf",
  "hallucination_survey_ji_2023.pdf", "rag_lewis_2020.pdf", "selfcheckgpt_2023.pdf", "truthfulqa_2022.pdf",
]);

function Progress({ steps, expected, queue, status, elapsed }:
  { steps: string[]; expected: string[]; queue: number; status: string; elapsed: number }) {
  const [now, setNow] = useState(elapsed);
  useEffect(() => { setNow(elapsed); const t = setInterval(() => setNow((x) => x + 1), 1000); return () => clearInterval(t); }, [elapsed]);
  return (
    <section className="card p-5" aria-live="polite">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold">{status === "queued" ? `Waiting in queue (${queue} ahead)…` : "Running the pipeline"}</h2>
        <span className="font-mono text-sm text-muted">{Math.round(now)} s</span>
      </div>
      <ol className="mt-4 space-y-2.5">
        {expected.map((s, i) => {
          const done = steps.length > i + 1;
          const active = steps.length === i + 1;
          const optional = s.startsWith("Correcting");
          return (
            <li key={s} className="flex items-center gap-3 text-sm">
              <span className={`grid h-6 w-6 place-items-center rounded-full text-xs font-bold ${
                done ? "bg-verified text-white" : active ? "bg-amber text-white" : "bg-line text-muted"}`}>
                {done ? "✓" : i + 1}
              </span>
              <span className={done ? "text-muted" : active ? "font-medium" : "text-muted/70"}>
                {s}{optional && <span className="ml-1 text-xs">(only if weak claims are found)</span>}
              </span>
              {active && <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-amber" />}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/* ------------------------------------------------------------------ results */

function Result({ report, jobId, elapsed }: { report: Report; jobId: string; elapsed: number }) {
  const [showOriginal, setShowOriginal] = useState(false);
  const [activeSid, setActiveSid] = useState<string | null>(null);
  const corrected = !!report.revised_answer;
  const useRevised = corrected && !showOriginal;
  const claims = useRevised ? report.revised_claims : report.claims;
  const stats = (useRevised ? report.summary.revised : report.summary.original) ?? report.summary.original;

  return (
    <div className="space-y-6">
      <section className="card grid grid-cols-2 gap-6 p-5 lg:grid-cols-5">
        <Stat label="Claims checked" value={String(stats.n_claims)} />
        <Stat label="Hallucination rate" value={pct(stats.hallucination_rate)}
              hint={corrected ? `was ${pct(report.summary.original.hallucination_rate)} before correction` : "unsupported + contradicted"} />
        <Stat label="Mean confidence" value={stats.mean_confidence.toFixed(2)} />
        <Stat label="Citation accuracy" value={pct(stats.citation_accuracy)} hint="cited passage supports claim" />
        <Stat label="Time" value={`${Math.round(elapsed)} s`} hint={`${report.evidence.length} sources`} />
        <div className="col-span-2 lg:col-span-5"><VerdictBar stats={stats} /></div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-w-0 space-y-6">
          <section className="card p-5 sm:p-6">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="label">{useRevised ? "Answer · corrected" : "Answer"}</h2>
              <div className="flex gap-2">
                {corrected && (
                  <button className="btn-ghost !px-3 !py-1 text-xs" onClick={() => setShowOriginal(!showOriginal)}>
                    {showOriginal ? "Show corrected" : "Show original"}
                  </button>
                )}
                <a className="btn-ghost !px-3 !py-1 text-xs" href={api.reportUrl(jobId)} download="veritas_report.md">Download report</a>
              </div>
            </div>
            <AnswerText text={useRevised ? report.revised_answer! : report.answer} activeSid={activeSid} onCite={setActiveSid} />
            {report.lay_summary && (
              <div className="mt-5 rounded-xl bg-amber-soft p-4 text-sm leading-relaxed">
                <div className="label mb-1 !text-amber">Plain-language summary</div>{report.lay_summary}
              </div>
            )}
          </section>

          {report.conflicts.length > 0 && (
            <section className="card border-uncertain/40 p-5">
              <h2 className="label !text-uncertain">Conflicting evidence</h2>
              <ul className="mt-3 space-y-3">
                {report.conflicts.map((c, i) => (
                  <li key={i} className="text-sm">
                    <p className="font-medium">{c.claim}</p>
                    <p className="mt-1 text-muted">Supported by <Cite sid={c.support_sid} onClick={setActiveSid} /> {c.support_doc} ·
                      contradicted by <Cite sid={c.contradict_sid} onClick={setActiveSid} /> {c.contradict_doc}</p>
                    {c.explanation && <p className="mt-1.5 leading-relaxed">{c.explanation}</p>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          <section>
            <h2 className="label mb-3">Claim-level verification</h2>
            <div className="space-y-3">
              {claims.map((c, i) => (
                <ClaimCard key={`${useRevised}-${i}`} claim={c} index={i} jobId={jobId} revised={useRevised}
                           evidence={report.evidence} onCite={setActiveSid} />
              ))}
              {claims.length === 0 && <p className="text-sm text-muted">No factual claims were found in the answer.</p>}
            </div>
          </section>
        </div>

        <aside className="space-y-3 lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:overflow-y-auto">
          <h2 className="label">Sources</h2>
          {report.evidence.map((e) => <SourceCard key={e.sid} ev={e} active={activeSid === e.sid}
                                                  onClick={() => setActiveSid(activeSid === e.sid ? null : e.sid)} />)}
          {report.samples.length > 0 && (
            <details className="card p-4 text-sm">
              <summary className="cursor-pointer font-medium">Self-consistency samples ({report.samples.length})</summary>
              <ol className="mt-3 space-y-3 text-muted">
                {report.samples.map((s, i) => <li key={i} className="whitespace-pre-line leading-relaxed">{i + 1}. {s}</li>)}
              </ol>
            </details>
          )}
        </aside>
      </div>
    </div>
  );
}

function Cite({ sid, onClick, active }: { sid: string; onClick: (s: string) => void; active?: boolean }) {
  return (
    <button onClick={() => onClick(sid)}
            className={`mx-0.5 rounded-md px-1.5 py-px align-baseline font-mono text-[11px] font-medium transition ${
              active ? "bg-amber text-white" : "bg-ink/8 text-ink hover:bg-amber hover:text-white"}`}>
      {sid}
    </button>
  );
}

function AnswerText({ text, activeSid, onCite }: { text: string; activeSid: string | null; onCite: (s: string) => void }) {
  const parts = useMemo(() => text.split(/(\[S\d+\])/g), [text]);
  return (
    <p className="mt-3 whitespace-pre-line font-serif text-[17px] leading-[1.75]">
      {parts.map((p, i) => {
        const m = p.match(/^\[(S\d+)\]$/);
        return m ? <Cite key={i} sid={m[1]} onClick={onCite} active={activeSid === m[1]} /> : <Fragment key={i}>{p.replace(/\*\*/g, "")}</Fragment>;
      })}
    </p>
  );
}

function ClaimCard({ claim, index, jobId, revised, evidence, onCite }:
  { claim: Claim; index: number; jobId: string; revised: boolean; evidence: Evidence[]; onCite: (s: string) => void }) {
  const [open, setOpen] = useState(false);
  const [explanation, setExplanation] = useState<string | null>(claim.explanation);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const v = VERDICT[claim.label];
  const best = evidence.find((e) => e.sid === claim.best_source);

  async function explain() {
    setLoading(true); setErr(null);
    try { setExplanation((await api.explain(jobId, index, revised)).explanation); }
    catch (e) { setErr((e as Error).message); }
    finally { setLoading(false); }
  }

  return (
    <article className="card overflow-hidden">
      <button className="flex w-full items-start gap-3 p-4 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
        <VerdictBadge label={claim.label} />
        <span className="flex-1 text-[15px] leading-snug">{claim.text}</span>
        <span className={`font-serif text-lg font-bold tabular-nums ${v.fg}`}>{claim.confidence.toFixed(2)}</span>
        <span className={`mt-1 text-muted transition ${open ? "rotate-180" : ""}`} aria-hidden>⌄</span>
      </button>
      <div className="grid gap-x-6 gap-y-2 border-t border-line/70 bg-paper/50 px-4 py-3 text-xs sm:grid-cols-3">
        <div><span className="label">Evidence (NLI)</span><Meter value={claim.entail} color="bg-verified" label="entailment" /></div>
        <div><span className="label">Self-consistency</span>
          {claim.consistency == null ? <div className="pt-0.5 text-muted">not computed (fast mode)</div>
            : <Meter value={claim.consistency} color="bg-ink" label="consistency" />}</div>
        <div><span className="label">Contradiction</span><Meter value={claim.contradict} color="bg-unsupported" label="contradiction" /></div>
      </div>
      {open && (
        <div className="space-y-4 border-t border-line/70 px-4 py-4 text-sm">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
            Cited: {claim.citations.length ? claim.citations.map((s) => <Cite key={s} sid={s} onClick={onCite} />) : "none"}
            {claim.citation_correct === true && <span className="text-verified">· citation supports the claim</span>}
            {claim.citation_correct === false && <span className="text-unsupported">· cited passage does not support it</span>}
          </div>
          {best && (
            <blockquote className="rounded-xl border-l-4 border-amber/70 bg-amber-soft/60 px-4 py-3 leading-relaxed">
              <div className="label mb-1">Best evidence · {best.sid} · {best.chunk.doc_name}, p.{best.chunk.page}</div>
              {best.chunk.text}
            </blockquote>
          )}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead><tr className="text-left text-muted"><th className="py-1 font-medium">Source</th><th>Entail</th><th>Neutral</th><th>Contradict</th></tr></thead>
              <tbody className="font-mono tabular-nums">
                {Object.entries(claim.per_source).map(([sid, p]) => (
                  <tr key={sid} className={sid === claim.best_source ? "font-semibold" : ""}>
                    <td className="py-0.5">{sid}</td><td>{p.entail.toFixed(2)}</td><td>{p.neutral.toFixed(2)}</td><td>{p.contradict.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {explanation ? (
            <div className="rounded-xl bg-ink/5 p-3 leading-relaxed whitespace-pre-line"><div className="label mb-1">Why this verdict</div>{explanation}</div>
          ) : (
            <button className="btn-ghost !py-1.5 text-xs" onClick={explain} disabled={loading}>
              {loading ? <><Spinner /> Explaining…</> : "Explain this verdict"}
            </button>
          )}
          {err && <ErrorBox message={err} />}
          <div className={`h-1 rounded-full ${VERDICT_FILL[claim.label]} opacity-30`} />
        </div>
      )}
    </article>
  );
}

function SourceCard({ ev, active, onClick }: { ev: Evidence; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick}
            className={`card block w-full p-4 text-left text-sm transition ${active ? "border-amber ring-2 ring-amber/30" : "hover:border-ink/25"}`}>
      <div className="flex items-center gap-2">
        <span className={`rounded-md px-1.5 py-px font-mono text-[11px] font-semibold ${active ? "bg-amber text-white" : "bg-ink/8"}`}>{ev.sid}</span>
        <span className="truncate font-medium">{ev.chunk.doc_name}</span>
        <span className="ml-auto shrink-0 text-xs text-muted">p.{ev.chunk.page}</span>
      </div>
      <p className={`mt-2 leading-relaxed text-muted ${active ? "" : "line-clamp-3"}`}>{ev.chunk.text}</p>
      <div className="mt-2 font-mono text-[11px] text-muted">
        similarity {ev.dense_score.toFixed(2)}{ev.rerank_score != null && ` · re-rank ${ev.rerank_score.toFixed(1)}`}
      </div>
    </button>
  );
}
