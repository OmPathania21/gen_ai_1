import { useEffect, useRef, useState } from "react";
import { api, type Job, type Stats, type Verdict } from "../api";

export const VERDICT: Record<Verdict, { label: string; icon: string; fg: string; bg: string }> = {
  verified: { label: "Verified", icon: "✓", fg: "text-verified", bg: "bg-verified-soft" },
  uncertain: { label: "Uncertain", icon: "!", fg: "text-uncertain", bg: "bg-uncertain-soft" },
  unsupported: { label: "Unsupported", icon: "✕", fg: "text-unsupported", bg: "bg-unsupported-soft" },
  contradicted: { label: "Contradicted", icon: "⊘", fg: "text-contradicted", bg: "bg-contradicted-soft" },
};
export const VERDICT_FILL: Record<Verdict, string> = {
  verified: "bg-verified", uncertain: "bg-uncertain", unsupported: "bg-unsupported", contradicted: "bg-contradicted",
};

export const pct = (x: number | null | undefined, digits = 0) =>
  x == null ? "–" : `${(x * 100).toFixed(digits)}%`;

export function VerdictBadge({ label, size = "sm" }: { label: Verdict; size?: "sm" | "md" }) {
  const v = VERDICT[label];
  return (
    <span className={`inline-flex shrink-0 items-center gap-1.5 rounded-full font-semibold ${v.bg} ${v.fg} ${
      size === "md" ? "px-3 py-1 text-xs" : "px-2.5 py-0.5 text-[11px]"}`}>
      <span aria-hidden>{v.icon}</span>{v.label}
    </span>
  );
}

export function Meter({ value, color = "bg-ink", label }: { value: number; color?: string; label?: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-line" role="meter" aria-valuenow={value}
           aria-valuemin={0} aria-valuemax={1} aria-label={label}>
        <div className={`h-full rounded-full ${color}`} style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
      <span className="w-9 text-right font-mono text-xs tabular-nums">{value.toFixed(2)}</span>
    </div>
  );
}

/** Stacked bar showing how many claims got each verdict. */
export function VerdictBar({ stats }: { stats: Stats }) {
  const order: Verdict[] = ["verified", "uncertain", "unsupported", "contradicted"];
  if (!stats.n_claims) return null;
  return (
    <div>
      <div className="flex h-2.5 overflow-hidden rounded-full bg-line">
        {order.map((k) => stats[k] > 0 && (
          <div key={k} className={VERDICT_FILL[k]} style={{ width: `${(stats[k] / stats.n_claims) * 100}%` }}
               title={`${VERDICT[k].label}: ${stats[k]}`} />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        {order.map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${VERDICT_FILL[k]}`} />{VERDICT[k].label} {stats[k]}
          </span>
        ))}
      </div>
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="mt-1 font-serif text-2xl font-bold tabular-nums">{value}</div>
      {hint && <div className="text-xs text-muted">{hint}</div>}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return <div className="rounded-xl border border-unsupported/30 bg-unsupported-soft px-4 py-3 text-sm text-unsupported">{message}</div>;
}

export function Spinner({ className = "" }: { className?: string }) {
  return <span className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent ${className}`} />;
}

/** Poll a background job until it finishes. */
export function useJob<R>(jobId: string | null) {
  const [job, setJob] = useState<Job<R> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => {
    setJob(null); setError(null);
    if (!jobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const j = await api.job<R>(jobId);
        if (cancelled) return;
        setJob(j);
        if (j.status === "queued" || j.status === "running") timer.current = window.setTimeout(tick, 900);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    };
    tick();
    return () => { cancelled = true; window.clearTimeout(timer.current); };
  }, [jobId]);
  return { job, error };
}
