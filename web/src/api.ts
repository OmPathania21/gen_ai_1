// Typed client for the VERITAS FastAPI backend (api/main.py).

export type Verdict = "verified" | "uncertain" | "unsupported" | "contradicted";

export interface Chunk { id: string; doc_id: string; doc_name: string; page: number; text: string }
export interface Evidence { sid: string; chunk: Chunk; dense_score: number; rerank_score: number | null }
export interface NliProbs { entail: number; neutral: number; contradict: number }

export interface Claim {
  text: string;
  citations: string[];
  entail: number;
  neutral: number;
  contradict: number;
  best_source: string | null;
  citation_correct: boolean | null;
  consistency: number | null;
  confidence: number;
  label: Verdict;
  per_source: Record<string, NliProbs>;
  explanation: string | null;
}

export interface Conflict {
  claim: string; support_sid: string; contradict_sid: string;
  support_doc: string; contradict_doc: string; explanation: string | null;
}

export interface Stats {
  n_claims: number; verified: number; uncertain: number; unsupported: number; contradicted: number;
  hallucination_rate: number; mean_confidence: number; citation_accuracy: number | null;
}

export interface Report {
  question: string;
  answer: string;
  evidence: Evidence[];
  claims: Claim[];
  conflicts: Conflict[];
  samples: string[];
  revised_answer: string | null;
  revised_claims: Claim[];
  lay_summary: string | null;
  strategy: string;
  timings: Record<string, number>;
  summary: { original: Stats; revised?: Stats };
}

export interface CompareRow extends Stats {
  strategy: string; type: string; answer: string;
  claims: { text: string; label: Verdict; confidence: number }[];
}

export interface Job<R = unknown> {
  id: string;
  kind: "ask" | "compare";
  status: "queued" | "running" | "done" | "error";
  steps: string[];
  queue_position: number;
  elapsed: number;
  params: Record<string, unknown>;
  result: R | null;
  error: string | null;
}

export interface Info {
  generator: string; embedder: string; reranker: string; verifier: string;
  consistency_samples: number;
  thresholds: { verified: number; unsupported: number };
  weights: { nli: number; consistency: number; contradiction_penalty: number };
  collections: { id: string; name: string; chunks: number }[];
  busy: number;
}

export interface PromptCatalog {
  catalog: { name: string; type: string; stage: string; template: string }[];
  strategies: { id: string; type: string; system: string | null; user: string }[];
  grounding_rules: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  const text = await res.text();
  const body = text && res.headers.get("content-type")?.includes("json") ? JSON.parse(text) : text;
  if (!res.ok) {
    const detail = (body as { detail?: unknown })?.detail;
    const msg = Array.isArray(detail) ? detail.map((d: { msg: string }) => d.msg).join("; ") : detail;
    throw new Error(typeof msg === "string" ? msg : `Request failed (${res.status})`);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  info: () => request<Info>("/info"),
  documents: (collection: string) =>
    request<{ name: string; chunks: number }[]>(`/documents?collection=${collection}`),
  upload: (collection: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return request<{ added: Record<string, number>; total_chunks: number }>(
      `/documents?collection=${collection}`, { method: "POST", body: fd });
  },
  deleteDocument: (collection: string, name: string) =>
    request(`/documents/${encodeURIComponent(name)}?collection=${collection}`, { method: "DELETE" }),
  ask: (body: Record<string, unknown>) => request<Job<Report>>("/ask", json(body)),
  compare: (body: Record<string, unknown>) => request<Job<{ question: string; rows: CompareRow[] }>>("/compare", json(body)),
  job: <R,>(id: string) => request<Job<R>>(`/jobs/${id}`),
  explain: (id: string, claim_index: number, revised: boolean) =>
    request<{ explanation: string }>(`/jobs/${id}/explain`, json({ claim_index, revised })),
  reportUrl: (id: string) => `/api/jobs/${id}/report.md`,
  prompts: () => request<PromptCatalog>("/prompts"),
  evaluation: () => request<Record<string, any>>("/evaluation"),
};
