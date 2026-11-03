import { useCallback, useEffect, useState } from "react";
import { api, type Info } from "./api";
import Ask from "./pages/Ask";
import PromptLab from "./pages/PromptLab";
import Evaluation from "./pages/Evaluation";

const PAGES = [
  { id: "ask", label: "Workspace" },
  { id: "prompts", label: "Prompt Lab" },
  { id: "evaluation", label: "Evaluation" },
] as const;
type PageId = (typeof PAGES)[number]["id"];

const readHash = (): PageId => {
  const h = window.location.hash.replace("#/", "").split("?")[0] as PageId;
  return PAGES.some((p) => p.id === h) ? h : "ask";
};

export default function App() {
  const [page, setPage] = useState<PageId>(readHash);
  const [info, setInfo] = useState<Info | null>(null);
  const [offline, setOffline] = useState(false);

  const refreshInfo = useCallback(() => {
    api.info().then((i) => { setInfo(i); setOffline(false); }).catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    refreshInfo();
    const onHash = () => setPage(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [refreshInfo]);

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-10 border-b border-line bg-paper/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3 sm:px-8">
          <a href="#/ask" className="flex items-center gap-2">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="font-serif text-xl font-bold tracking-tight">VERITAS</span>
          </a>
          <nav className="ml-auto flex gap-1">
            {PAGES.map((p) => (
              <a key={p.id} href={`#/${p.id}`}
                 className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                   page === p.id ? "bg-ink text-white" : "text-muted hover:bg-ink/5 hover:text-ink"}`}>
                {p.label}
              </a>
            ))}
          </nav>
        </div>
      </header>

      <main className="w-full flex-1 px-4 py-8 sm:px-8 sm:py-12">
        {offline && (
          <div className="mx-auto mb-6 max-w-3xl rounded-xl border border-unsupported/30 bg-unsupported-soft px-4 py-3 text-sm text-unsupported">
            Cannot reach the VERITAS server. Start it with <code className="font-mono">uvicorn api.main:app --port 8000</code>.
          </div>
        )}
        {page === "ask" && <Ask info={info} onLibraryChange={refreshInfo} />}
        {page === "prompts" && <div className="mx-auto max-w-6xl"><PromptLab collection="default" /></div>}
        {page === "evaluation" && <Evaluation />}
      </main>

      <footer className="border-t border-line px-4 py-5 text-center text-xs text-muted">
        VERITAS · grounded generation with claim-level verification
        {info && <> · {info.generator.split("/").pop()} + {info.verifier.split("/").pop()?.split("-mnli")[0]} NLI</>}
      </footer>
    </div>
  );
}
