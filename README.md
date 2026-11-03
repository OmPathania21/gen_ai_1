# VERITAS: Evidence-Driven Generative AI for Reliable Research

VERITAS answers research questions from documents you upload, then checks **every claim in its own answer** against the evidence. Each claim is labelled verified, uncertain, unsupported or contradicted, with a confidence score. Where sources disagree, the conflict is flagged.

Course project for 21CSE306P Applied Generative AI (Team 13).

## Pipeline (fixed, single pass)

```
Upload docs → chunk + embed (MiniLM) → ChromaDB
Question → dense retrieval → cross-encoder re-rank → grounded prompt → LLM answer with [S#] citations
        → claim extraction → DeBERTa-v3 NLI vs evidence + self-consistency (N re-samples)
        → confidence score + conflict detection → one correction pass → verified report
```

---

## Quick start (step by step)

Works on **Windows, macOS and Linux**. Takes about 10–15 minutes the first time, mostly downloads.

### Step 1: Install the prerequisites (once per computer)

| Tool | Version | Download | Check it works |
|---|---|---|---|
| Git | any | https://git-scm.com/downloads | `git --version` |
| Python | **3.10 – 3.13** | https://www.python.org/downloads/ | `python3 --version` (Windows: `py --version`) |
| Node.js | **20 or newer** (LTS) | https://nodejs.org | `node --version` |

> **Windows:** when installing Python, tick **"Add python.exe to PATH"** on the first screen.

You need about 4 GB of free disk space (Python packages + three small local models).

### Step 2: Get the code

```bash
git clone https://github.com/OmPathania21/gen_ai_1.git
cd gen_ai_1
```

Already cloned? Run `git pull` inside the folder instead.

### Step 3: Get a free Groq API key

The answer-writing model (gpt-oss-120b) runs on Groq's free cloud API. Each teammate uses **their own** key.

1. Go to **https://console.groq.com/keys** and sign in (Google or GitHub login works).
2. Click **Create API Key**, type any name (e.g. `veritas`) and click **Submit**.
3. **Copy the key now**: it starts with `gsk_` and is shown only once.

Keep the key private: never paste it in chat, code or a commit. It goes only in a file called `.env`, which Git ignores.

### Step 4: Run the setup script

This creates the Python environment, installs everything, asks for your key, builds the web page, and downloads the demo papers and models.

**macOS / Linux**
```bash
python3 scripts/setup.py
```

**Windows** (Command Prompt or PowerShell)
```bat
py scripts\setup.py
```

When it asks `Groq API key:`, paste the key from Step 3 and press **Enter**. The input is hidden, so nothing appears as you paste; that is normal.

The script is safe to re-run. If a step fails (e.g. a network drop), run it again and finished steps are skipped.

<details>
<summary>Prefer to add the key by hand?</summary>

1. Copy the example file:
   - macOS / Linux: `cp .env.example .env`
   - Windows: `copy .env.example .env`
2. Open `.env` in any text editor (it is in the project's main folder; on macOS press `Cmd+Shift+.` in Finder to see hidden files).
3. Replace `gsk_your_key_here` with your key, with no spaces or quotes:
   ```
   GROQ_API_KEY=gsk_AbC123...
   ```
4. Save the file.
</details>

### Step 5: Start VERITAS

**macOS / Linux**
```bash
./start.sh
```

**Windows**
```bat
start.bat
```

Open **http://localhost:8000** in your browser. Press `Ctrl+C` in the terminal to stop the app.

### Step 6: Try it

1. On the **Workspace** page, click **Upload a PDF** and choose the papers in `data/sample_docs/` (downloaded in Step 4), or any PDF, DOCX, TXT or MD file of your own.
2. Click a suggested question, or type your own.
3. Pick **Quick check** (~30 s) or **Deep check** (~1–2 min, adds self-consistency and one correction pass).
4. Click **Get verified answer**.

The first question takes longer because the local models load into memory.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is missing` | The key is not in `.env`. Redo Step 3 and Step 4, or edit `.env` by hand (see the box under Step 4). Restart the app afterwards. |
| `401` / `Invalid API Key` | The key was mistyped or deleted. Create a new one at console.groq.com/keys and replace it in `.env`. |
| `429` / rate limit | The free tier allows a limited number of tokens per minute. Wait a minute, or use Quick check. |
| `python3` / `py` not found | Install Python (Step 1). On Windows, reinstall and tick "Add python.exe to PATH". |
| `npm` not found | Install Node.js LTS (Step 1), then re-run the setup script. |
| `./start.sh: Permission denied` | Run `chmod +x start.sh` once. |
| Port 8000 already in use | macOS/Linux: `PORT=8001 ./start.sh`, then open http://localhost:8001. |
| `No documents indexed yet` | Upload at least one document first (Step 6). |
| Slow answers | Normal on CPU-only laptops. An NVIDIA GPU (CUDA) or Apple-silicon Mac is used automatically if present. |

---

## For developers

Hot reload while editing (two terminals):

```bash
.venv/bin/python -m uvicorn api.main:app --reload --port 8000   # backend (Windows: .venv\Scripts\python ...)
cd web && npm run dev                                             # frontend at http://localhost:5173
```

Command line:

```bash
.venv/bin/python -m veritas ingest data/sample_docs             # index the demo papers
.venv/bin/python -m veritas ask "How can hallucinations in LLMs be detected?"
.venv/bin/python -m veritas docs                                 # list indexed documents
```

Reports are saved to `outputs/` as Markdown and JSON. Settings (models, thresholds, weights) are in `veritas/config.py`.

## Research questions and experiments

| RQ | Question | Script |
|---|---|---|
| RQ1 | Does grounding reduce hallucination vs. a closed-book LLM? | `python -m eval.rq_generation --n 20` |
| RQ2 | Can NLI reliably verify scientific claims? (SciFact) | `python -m eval.rq2_verifier [--judge]` |
| RQ3 | Does self-consistency add signal beyond NLI? | `python -m eval.rq_generation --judge` (per-claim CSV) |
| RQ4 | Which prompt design hallucinates least? | `python -m eval.prompt_compare` |
| RQ5 | Do model size and retrieval settings matter? | `GROQ_MODEL=openai/gpt-oss-20b python -m eval.rq_generation` |

The RQ2 and RQ4 results are committed in `outputs/` so the Evaluation page works right after cloning.

## Layout

```
api/       main.py  (FastAPI: documents, ask jobs, explain, prompts, evaluation)
web/       React + Vite + Tailwind frontend (Workspace, Prompt Lab, Evaluation)
app/       streamlit_app.py  (original prototype UI)
veritas/   config  ingest  index  retrieve  llm  prompts  generate
           claims  verify  scoring  report  pipeline  __main__
eval/      scifact.py  rq2_verifier.py  rq_generation.py  prompt_compare.py
scripts/   setup.py  download_data.py
gen_ai/    review decks, poster, documentation
data/      sample_docs/  scifact/  (downloaded)   chroma/  cache/  uploads/  (created at run time)
outputs/   reports and experiment results
```
