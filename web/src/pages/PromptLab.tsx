import { useEffect, useState } from "react";
import { api, type CompareRow, type PromptCatalog } from "../api";
import { ErrorBox, Spinner, VERDICT, VerdictBar, pct, useJob } from "../components/ui";

const TYPE_COLORS: Record<string, string> = {
  "Role-play": "bg-[#e3ecfb] text-[#1f5fbf]",
  Instructional: "bg-verified-soft text-verified",
  Information: "bg-[#e6eef7] text-[#1f618d]",
  "Completion + Few-shot": "bg-[#fbf3d6] text-[#8a6d00]",
  Classification: "bg-[#efe4f6] text-[#7d3c98]",
  Reasoning: "bg-amber-soft text-amber",
  Comparison: "bg-unsupported-soft text-unsupported",
  Creative: "bg-[#ddf3ee] text-[#148f77]",
};

export default function PromptLab({ collection }: { collection: string }) {
  const [catalog, setCatalog] = useState<PromptCatalog | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [question, setQuestion] = useState("How can hallucinations in large language models be detected, and how effective are these methods?");
  const [picked, setPicked] = useState<string[]>(["bad_vague", "role_instruct", "cot"]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const { job } = useJob<{ question: string; rows: CompareRow[] }>(jobId);
  const running = job?.status === "queued" || job?.status === "running";

  useEffect(() => { api.prompts().then(setCatalog).catch((e) => setErr(e.message)); }, []);

  async function compare() {
    setErr(null);
    try { setJobId((await api.compare({ question, collection, strategies: picked })).id); }
    catch (e) { setErr((e as Error).message); }
  }

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <header>
        <h1 className="font-serif text-3xl font-bold tracking-tight sm:text-4xl">Prompt Lab</h1>
        <p className="mt-1.5 text-muted">Every prompt VERITAS uses, labelled by prompt-engineering type, and a live experiment comparing prompt designs.</p>
      </header>

      <section>
        <h2 className="label mb-3">Prompts in the pipeline</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {catalog?.catalog.map((p) => (
            <article key={p.name} className="card flex flex-col p-4">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-semibold">{p.name}</h3>
                <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-[11px] font-semibold ${TYPE_COLORS[p.type] ?? "bg-ink/6"}`}>{p.type}</span>
              </div>
              <p className="mt-0.5 text-xs text-muted">Stage: {p.stage}</p>
              <pre className={`mt-3 overflow-hidden whitespace-pre-wrap rounded-lg bg-paper p-3 font-mono text-[11.5px] leading-relaxed text-ink/80 ${
                open === p.name ? "" : "max-h-28"}`}>{p.template}</pre>
              <button className="mt-2 self-start text-xs font-medium text-amber hover:underline" onClick={() => setOpen(open === p.name ? null : p.name)}>
                {open === p.name ? "Collapse" : "Show full prompt"}
              </button>
            </article>
          ))}
        </div>
      </section>

      <section className="card p-5 sm:p-6">
        <h2 className="font-serif text-xl font-bold">Experiment: same question, same evidence, different prompt</h2>
        <p className="mt-1 text-sm text-muted">Each strategy answers from identical retrieved passages; every claim is then verified by NLI.</p>
        <input value={question} onChange={(e) => setQuestion(e.target.value)}
               className="mt-4 w-full rounded-xl border border-line bg-paper/60 px-4 py-2.5 text-sm outline-none focus:border-ink/40 focus:bg-white" />
        <div className="mt-3 flex flex-wrap gap-2">
          {catalog?.strategies.map((s) => {
            const on = picked.includes(s.id);
            return (
              <button key={s.id} onClick={() => setPicked(on ? picked.filter((x) => x !== s.id) : [...picked, s.id])}
                      title={s.type}
                      className={`rounded-full border px-3 py-1 text-xs font-medium transition ${
                        on ? "border-ink bg-ink text-white" : "border-line text-muted hover:border-ink/40"}`}>
                {s.id}
              </button>
            );
          })}
          <button className="btn-primary ml-auto" disabled={running || !picked.length || question.length < 3} onClick={compare}>
            {running ? <><Spinner /> {job?.steps.at(-1) ?? "Queued"}…</> : "Run comparison"}
          </button>
        </div>
        {err && <div className="mt-4"><ErrorBox message={err} /></div>}
        {job?.status === "error" && <div className="mt-4"><ErrorBox message={job.error ?? "Failed"} /></div>}

        {job?.status === "done" && job.result && (
          <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {job.result.rows.map((r) => (
              <article key={r.strategy} className={`rounded-2xl border p-4 ${r.strategy === "bad_vague" ? "border-unsupported/30 bg-unsupported-soft/40" : "border-line"}`}>
                <div className="flex items-baseline justify-between">
                  <h3 className="font-mono text-sm font-semibold">{r.strategy}</h3>
                  <span className={`font-serif text-2xl font-bold ${r.hallucination_rate > 0.3 ? "text-unsupported" : "text-verified"}`}>
                    {pct(r.hallucination_rate)}
                  </span>
                </div>
                <p className="text-xs text-muted">{r.type} · hallucination rate</p>
                <div className="mt-3"><VerdictBar stats={r} /></div>
                <p className="mt-3 text-xs text-muted">{r.n_claims} claims · citations {pct(r.citation_accuracy)} correct</p>
                <details className="mt-3 text-sm">
                  <summary className="cursor-pointer text-xs font-medium text-amber">Answer and claims</summary>
                  <p className="mt-2 whitespace-pre-line leading-relaxed">{r.answer}</p>
                  <ul className="mt-3 space-y-1 text-xs">
                    {r.claims.map((c, i) => (
                      <li key={i}><span className={`font-semibold ${VERDICT[c.label].fg}`}>{VERDICT[c.label].icon}</span> {c.text}</li>
                    ))}
                  </ul>
                </details>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
