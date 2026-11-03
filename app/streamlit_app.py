"""VERITAS web app.  Run:  streamlit run app/streamlit_app.py"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from veritas import prompts  # noqa: E402
from veritas.claims import extract_claims  # noqa: E402
from veritas.config import settings  # noqa: E402
from veritas.generate import generate_answer  # noqa: E402
from veritas.pipeline import Veritas  # noqa: E402
from veritas.report import ICON, Report  # noqa: E402
from veritas.retrieve import retrieve  # noqa: E402
from veritas.verify import verify_claims  # noqa: E402

st.set_page_config(page_title="VERITAS", page_icon="🔎", layout="wide")

COLOR = {"verified": "#1e8e3e", "uncertain": "#e37400", "unsupported": "#d93025", "contradicted": "#8b0000"}
UPLOAD_DIR = settings.data_dir / "uploads"


@st.cache_resource(show_spinner="Loading models…")
def get_veritas(collection: str) -> Veritas:
    v = Veritas(collection)
    _ = v.nli  # load the NLI model once
    return v


def badge(label: str) -> str:
    return (f"<span style='background:{COLOR[label]};color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:0.8em'>{ICON[label]} {label.upper()}</span>")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔎 VERITAS")
st.sidebar.caption("Evidence-Driven Generative AI for Reliable Research")
page = st.sidebar.radio("Page", ["Ask", "Library", "Prompt Lab", "Evaluation"])
collection = st.sidebar.selectbox("Collection", ["default", "scifact"],
                                  help="'default' = your uploads, 'scifact' = benchmark corpus")
v = get_veritas(collection)
st.sidebar.metric("Indexed chunks", v.index.count())
st.sidebar.caption(f"Generator: `{settings.llm_model}`  \nVerifier: `{settings.nli_model.split('/')[-1]}`")


# ---------------------------------------------------------------- Library
def page_library() -> None:
    st.header("📚 Document library")
    st.write("Upload the documents VERITAS should answer from. Only these documents count as evidence.")
    files = st.file_uploader("PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if files and st.button(f"Index {len(files)} file(s)", type="primary"):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in files:
            p = UPLOAD_DIR / f.name
            p.write_bytes(f.getbuffer())
            paths.append(p)
        with st.spinner("Extracting, chunking and embedding…"):
            added = v.ingest(paths)
        st.success(" · ".join(f"{k}: {n} chunks" for k, n in added.items()))

    docs = v.index.documents()
    if not docs:
        st.info("No documents yet.")
        return
    st.subheader(f"{len(docs)} documents")
    if collection == "scifact":
        st.caption("Benchmark corpus: 5,183 SciFact abstracts.")
        return
    for name, n in docs.items():
        c1, c2 = st.columns([5, 1])
        c1.write(f"📄 **{name}** — {n} chunks")
        if c2.button("Remove", key=f"rm-{name}"):
            v.index.delete_document(name)
            st.rerun()


# ---------------------------------------------------------------- Ask
def render_claims(rep: Report, claims, key: str) -> None:
    by_sid = {e.sid: e for e in rep.evidence}
    for i, c in enumerate(claims):
        with st.container(border=True):
            st.markdown(f"{badge(c.label)} &nbsp; **{c.text}**", unsafe_allow_html=True)
            cols = st.columns(4)
            cols[0].progress(c.confidence, text=f"Confidence {c.confidence:.2f}")
            cols[1].write(f"NLI entail **{c.entail:.2f}** · contradict **{c.contradict:.2f}**")
            cols[2].write("Consistency " + (f"**{c.consistency:.2f}**" if c.consistency is not None else "–"))
            cite = {True: "✔ correct", False: "✘ wrong source", None: "none"}[c.citation_correct]
            cols[3].write(f"Cited {', '.join(c.citations) or '–'} ({cite})")
            with st.expander("Evidence & reasoning"):
                ev = by_sid.get(c.best_source)
                if ev:
                    st.caption(f"Best evidence [{ev.sid}] — {ev.chunk.doc_name}, p.{ev.chunk.page}")
                    st.write(ev.chunk.text)
                st.dataframe(pd.DataFrame(c.per_source).T.round(3), width="stretch")
                if st.button("Explain this verdict", key=f"{key}-exp-{i}"):
                    with st.spinner("Reasoning…"):
                        st.info(v.explain_claim(rep, c))


def page_ask() -> None:
    st.header("❓ Ask a research question")
    q = st.text_area("Question", placeholder="e.g. How can hallucinations in large language models be detected?")
    mode = st.radio("Mode", ["Fast", "Full", "Custom"], horizontal=True,
                    captions=["Answer + NLI evidence check (~15–30 s)",
                              f"+ {settings.n_consistency_samples} self-consistency samples + correction (~1–2 min)",
                              "Choose every setting"])
    strategy, rerank, lay = "role_instruct", True, False
    n_samples, correct = (0, False) if mode == "Fast" else (settings.n_consistency_samples, True)
    if mode == "Custom":
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            strategy = c1.selectbox("Prompt strategy", list(prompts.ANSWER_STRATEGIES), index=2,
                                    format_func=lambda s: f"{s} — {prompts.ANSWER_STRATEGIES[s]['type']}")
            n_samples = c2.slider("Self-consistency samples", 0, 6, settings.n_consistency_samples)
            rerank = c3.toggle("Cross-encoder re-ranking", True)
            correct = c1.toggle("Single-pass correction of weak claims", True)
            lay = c2.toggle("Plain-language summary", False)

    if st.button("Run VERITAS", type="primary", disabled=not q.strip()):
        if v.index.count() == 0:
            st.warning("Upload documents on the Library page first.")
            return
        with st.status("Running pipeline…", expanded=True) as s:
            def progress(msg: str) -> None:
                s.update(label=f"{msg}…")
                s.write(f"⏳ {msg}")
            try:
                rep = v.run(q.strip(), strategy=strategy, n_samples=n_samples, use_reranker=rerank,
                            correct=correct, lay_summary=lay, progress=progress)
            except Exception as e:  # most often an API rate limit or a missing key
                s.update(label="Failed", state="error")
                st.error(f"{type(e).__name__}: {e}")
                return
            s.update(label=f"Done in {rep.timings.get('total', 0):.1f}s", state="complete", expanded=False)
        st.session_state["report"] = rep

    rep: Report | None = st.session_state.get("report")
    if not rep:
        return
    stats = rep.summary["original"]
    m = st.columns(5)
    m[0].metric("Claims", stats["n_claims"])
    m[1].metric("Verified", stats["verified"])
    m[2].metric("Hallucination rate", f"{stats['hallucination_rate']:.0%}",
                delta=(f"{rep.summary['revised']['hallucination_rate'] - stats['hallucination_rate']:+.0%} after correction"
                       if "revised" in rep.summary else None), delta_color="inverse")
    m[3].metric("Mean confidence", f"{stats['mean_confidence']:.2f}")
    ca = stats["citation_accuracy"]
    m[4].metric("Citation accuracy", f"{ca:.0%}" if ca is not None else "–")

    tabs = st.tabs(["Answer", "Claims", "Conflicts", "Evidence", "Self-consistency samples", "Export"])
    with tabs[0]:
        if rep.revised_answer:
            st.subheader("Corrected answer")
            st.write(rep.revised_answer)
            with st.expander("Original answer (before correction)"):
                st.write(rep.answer)
        else:
            st.write(rep.answer)
        if rep.lay_summary:
            st.subheader("Plain-language summary")
            st.info(rep.lay_summary)
    with tabs[1]:
        render_claims(rep, rep.claims, "orig")
        if rep.revised_claims:
            st.subheader("Claims after correction")
            render_claims(rep, rep.revised_claims, "rev")
    with tabs[2]:
        if not rep.conflicts:
            st.success("No conflicting sources detected.")
        for cf in rep.conflicts:
            st.warning(f"**{cf.claim}**\n\nSupported by [{cf.support_sid}] {cf.support_doc} · "
                       f"contradicted by [{cf.contradict_sid}] {cf.contradict_doc}\n\n{cf.explanation or ''}")
    with tabs[3]:
        for e in rep.evidence:
            rr = f" · re-rank {e.rerank_score:.2f}" if e.rerank_score is not None else ""
            with st.expander(f"[{e.sid}] {e.chunk.doc_name}, p.{e.chunk.page} — similarity {e.dense_score:.2f}{rr}"):
                st.write(e.chunk.text)
    with tabs[4]:
        for i, s in enumerate(rep.samples, 1):
            with st.expander(f"Sample {i} (temperature {settings.sample_temperature})"):
                st.write(s)
    with tabs[5]:
        st.download_button("Download report (Markdown)", rep.to_markdown(), "veritas_report.md")
        st.download_button("Download report (JSON)", json.dumps(rep.to_dict(), indent=2, default=str), "veritas_report.json")
        st.json(rep.timings)


# ---------------------------------------------------------------- Prompt Lab
def page_prompt_lab() -> None:
    st.header("🧪 Prompt Lab")
    st.write("Every prompt VERITAS uses, labelled by prompt-engineering type, plus a live comparison of answer strategies.")
    for p in prompts.PROMPT_CATALOG:
        with st.expander(f"**{p['name']}** · {p['type']} · stage: {p['stage']}"):
            st.code(p["template"], language="text")

    st.subheader("Compare prompt strategies on the same evidence")
    q = st.text_input("Question", key="lab_q")
    picked = st.multiselect("Strategies", list(prompts.ANSWER_STRATEGIES), default=["bad_vague", "role_instruct", "cot"])
    if st.button("Compare", disabled=not (q and picked)):
        evidence = retrieve(v.index, q)
        rows = []
        for s in picked:
            with st.spinner(f"{s}…"):
                ans = generate_answer(v.llm, q, evidence, s)
                res = verify_claims(v.nli, extract_claims(v.llm, ans), evidence)
            stats = Report.stats(res)
            rows.append({"strategy": s, "type": prompts.ANSWER_STRATEGIES[s]["type"], "claims": stats["n_claims"],
                         "hallucination rate": stats["hallucination_rate"], "mean confidence": stats["mean_confidence"],
                         "citation accuracy": stats["citation_accuracy"]})
            with st.expander(f"{s}: answer"):
                st.write(ans)
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch")


# ---------------------------------------------------------------- Evaluation
def page_evaluation() -> None:
    st.header("📊 Evaluation")
    out = settings.output_dir
    rq2 = out / "rq2_verifier.json"
    if rq2.exists():
        r = json.loads(rq2.read_text())
        st.subheader("RQ2 · Claim verification on SciFact")
        rows = [{"verifier": k, "accuracy": r[k]["accuracy"], "macro F1": r[k]["macro_f1"]} for k in ("nli", "llm_judge") if k in r]
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch")
        labels = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
        st.caption("NLI confusion matrix (rows = gold, columns = predicted)")
        st.dataframe(pd.DataFrame(r["nli"]["confusion_matrix"], index=labels, columns=labels))
    gen = out / "rq_generation_summary.json"
    if gen.exists():
        s = json.loads(gen.read_text())
        meta = s.pop("_meta", {})
        st.subheader(f"RQ1 / RQ4 · Hallucination by condition ({meta.get('n_questions')} questions, {meta.get('llm')})")
        df = pd.DataFrame(s).T[["n_claims", "hallucination_rate", "mean_confidence", "citation_accuracy"]]
        st.dataframe(df.astype(float).round(3), width="stretch")
        st.bar_chart(df["hallucination_rate"].astype(float))
    pc = out / "prompt_compare.json"
    if pc.exists():
        s = json.loads(pc.read_text())
        meta = s.pop("_meta", {})
        st.subheader(f"RQ4 preview · Prompt design vs hallucination ({len(meta.get('questions', []))} questions)")
        df = pd.DataFrame(s).T[["type", "n_claims", "hallucination_rate", "citation_accuracy"]]
        st.dataframe(df, width="stretch")
        st.bar_chart(df["hallucination_rate"].astype(float), y_label="hallucination rate")
    if not (rq2.exists() or gen.exists() or pc.exists()):
        st.info("No results yet. Run `python -m eval.rq2_verifier` and `python -m eval.rq_generation`.")


{"Ask": page_ask, "Library": page_library, "Prompt Lab": page_prompt_lab, "Evaluation": page_evaluation}[page]()
