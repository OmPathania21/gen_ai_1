import { useEffect, useState } from "react";
import { api } from "../api";
import { ErrorBox, Stat, pct } from "../components/ui";

const LABELS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"];
const SHORT: Record<string, string> = { SUPPORT: "Supported", CONTRADICT: "Contradicted", NOT_ENOUGH_INFO: "Not enough info" };
const STRATEGY_NAME: Record<string, string> = {
  bad_vague: "Bad prompt (vague)", zero_shot: "Zero-shot instructions", role_instruct: "Role + instructions",
  few_shot: "Few-shot", cot: "Chain-of-thought",
};

export default function Evaluation() {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.evaluation().then(setData).catch((e) => setErr(e.message)); }, []);

  const rq2 = data?.rq2_verifier;
  const pc = data?.prompt_compare;

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      <header>
        <h1 className="font-serif text-3xl font-bold tracking-tight sm:text-4xl">Evaluation</h1>
        <p className="mt-1.5 text-muted">Results of the research experiments. Reproduce them with the scripts in <code className="font-mono text-sm">eval/</code>.</p>
      </header>
      {err && <ErrorBox message={err} />}
      {data && !rq2 && !pc && <p className="text-muted">No results yet. Run <code className="font-mono">python -m eval.rq2_verifier</code>.</p>}

      {rq2 && (
        <section className="card p-5 sm:p-6">
          <div className="label !text-amber">RQ2</div>
          <h2 className="font-serif text-2xl font-bold">Can a transformer NLI model verify scientific claims?</h2>
          <p className="mt-1 text-sm text-muted">
            DeBERTa-v3 NLI vs expert labels on SciFact ({rq2.n} claim–abstract pairs). Chance level is 0.33 accuracy and 0.50 AUROC.
          </p>
          <div className="mt-6 grid grid-cols-2 gap-6 sm:grid-cols-4">
            <Stat label="Accuracy" value={rq2.nli.accuracy.toFixed(2)} />
            <Stat label="Macro-F1" value={rq2.nli.macro_f1.toFixed(2)} />
            <Stat label="AUROC" value={rq2.nli_auroc_supported.toFixed(2)} hint="supported vs not" />
            <Stat label="Pairs" value={String(rq2.n)} />
          </div>

          <div className="mt-8 grid gap-8 lg:grid-cols-2">
            <div>
              <h3 className="label mb-3">Per-class F1</h3>
              <div className="space-y-3">
                {LABELS.map((l) => {
                  const c = rq2.nli.per_class[l];
                  return (
                    <div key={l}>
                      <div className="flex justify-between text-sm"><span>{SHORT[l]}</span>
                        <span className="font-mono text-xs text-muted">P {c.precision.toFixed(2)} · R {c.recall.toFixed(2)} · F1 {c["f1-score"].toFixed(2)}</span></div>
                      <div className="mt-1 h-2.5 rounded-full bg-line"><div className="h-full rounded-full bg-ink" style={{ width: `${c["f1-score"] * 100}%` }} /></div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div>
              <h3 className="label mb-3">Confusion matrix (rows = expert label)</h3>
              <ConfusionMatrix m={rq2.nli.confusion_matrix} />
            </div>
          </div>
          <p className="mt-6 rounded-xl bg-paper p-4 text-sm leading-relaxed">
            <strong>Analysis.</strong> Most errors are cautious ones: supported claims labelled "not enough info". Contradictions are the
            hardest class (recall {rq2.nli.per_class.CONTRADICT.recall.toFixed(2)}), the main target for fine-tuning the verifier on SciFact.
          </p>
        </section>
      )}

      {pc && (
        <section className="card p-5 sm:p-6">
          <div className="label !text-amber">RQ4 · preview</div>
          <h2 className="font-serif text-2xl font-bold">Which prompt design hallucinates least?</h2>
          <p className="mt-1 text-sm text-muted">
            {pc._meta?.questions?.length ?? 3} questions × 5 prompt designs, same retrieved evidence, claims verified by NLI.
          </p>
          <div className="mt-6 space-y-4">
            {Object.entries(pc).filter(([k]) => !k.startsWith("_")).map(([k, r]: [string, any]) => (
              <div key={k} className="grid items-center gap-2 sm:grid-cols-[200px_1fr_160px]">
                <span className={`text-sm font-medium ${k === "bad_vague" ? "text-unsupported" : ""}`}>{STRATEGY_NAME[k] ?? k}</span>
                <div className="h-6 rounded-lg bg-line">
                  <div className={`flex h-full items-center justify-end rounded-lg px-2 text-xs font-semibold text-white ${
                    k === "bad_vague" ? "bg-unsupported" : "bg-ink"}`} style={{ width: `${Math.max(r.hallucination_rate * 100, 7)}%` }}>
                    {pct(r.hallucination_rate)}
                  </div>
                </div>
                <span className="text-xs text-muted">{r.n_claims} claims · cites {pct(r.citation_accuracy)} correct</span>
              </div>
            ))}
          </div>
          <p className="mt-6 rounded-xl bg-paper p-4 text-sm leading-relaxed">
            <strong>Analysis.</strong> With identical evidence, prompt design alone moves the hallucination rate from
            {" "}{pct(pc.bad_vague?.hallucination_rate)} (vague prompt) to {pct(pc.role_instruct?.hallucination_rate)} (role + grounding rules).
            Chain-of-thought is also safe but produces far fewer claims. This is a 3-question preview; the full RQ4 run uses SciFact.
          </p>
        </section>
      )}
    </div>
  );
}

function ConfusionMatrix({ m }: { m: number[][] }) {
  const max = Math.max(...m.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr><th />{LABELS.map((l) => <th key={l} className="px-2 pb-2 font-medium text-muted">{SHORT[l]}</th>)}</tr>
        </thead>
        <tbody>
          {m.map((row, i) => (
            <tr key={i}>
              <th className="pr-3 text-right font-medium text-muted">{SHORT[LABELS[i]]}</th>
              {row.map((v, j) => (
                <td key={j} className="p-1">
                  <div className={`grid h-14 w-20 place-items-center rounded-lg font-mono text-sm font-semibold ${
                    v / max > 0.5 ? "text-white" : "text-ink"}`}
                       style={{ background: i === j ? `rgba(30,142,90,${0.15 + 0.85 * v / max})` : `rgba(209,67,47,${0.08 + 0.7 * v / max})` }}>
                    {v}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
