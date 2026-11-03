"""One-time setup for VERITAS. Works on macOS, Linux and Windows.

    python scripts/setup.py              (Windows: py scripts\\setup.py)

Steps: check Python -> create .venv -> install Python packages -> save your Groq key in .env
       -> build the web UI -> download demo data -> download the three local models.
Safe to re-run: finished steps are skipped. Options: --skip-data, --skip-models.
"""

import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ENV_FILE = ROOT / ".env"
PLACEHOLDER = "gsk_your_key_here"


def step(n: int, text: str) -> None:
    print(f"\n[{n}/7] {text}")


def run(cmd: list, cwd: Path = ROOT) -> None:
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def current_key() -> str:
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text().splitlines():
        if line.strip().startswith("GROQ_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            return "" if key == PLACEHOLDER else key
    return ""


def save_key(key: str) -> None:
    lines = (ENV_FILE if ENV_FILE.exists() else ROOT / ".env.example").read_text().splitlines()
    out = [f"GROQ_API_KEY={key}" if l.strip().startswith("GROQ_API_KEY=") else l for l in lines]
    if not any(l.startswith("GROQ_API_KEY=") for l in out):
        out.insert(0, f"GROQ_API_KEY={key}")
    ENV_FILE.write_text("\n".join(out) + "\n")


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)  # keep our messages in order with the tools' output
    args = set(sys.argv[1:])

    step(1, "Checking Python version")
    if sys.version_info < (3, 10):
        print(f"Python 3.10 or newer is needed (you have {sys.version.split()[0]}). Install it from python.org.")
        return 1
    print(f"OK: Python {sys.version.split()[0]}")
    if sys.version_info >= (3, 14):
        print("Note: tested on Python 3.10-3.13. If a package fails to install, use Python 3.13.")

    step(2, "Creating the virtual environment (.venv)")
    if VENV_PY.exists():
        print("OK: .venv already exists")
    else:
        run([sys.executable, "-m", "venv", VENV])

    step(3, "Installing Python packages (first time: several minutes, ~2 GB)")
    run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    run([VENV_PY, "-m", "pip", "install", "-r", ROOT / "requirements.txt"])

    step(4, "Groq API key")
    if current_key():
        print("OK: .env already has a key (edit .env to change it)")
    else:
        print("VERITAS needs a free Groq API key for the language model.")
        print("  1. Open https://console.groq.com/keys and sign in (Google/GitHub works)")
        print("  2. Click 'Create API Key', give it any name, and copy the key (starts with gsk_)")
        print("  3. Paste it below and press Enter (the input is hidden; press Enter alone to skip)")
        key = getpass.getpass("Groq API key: ").strip()
        if key:
            if not key.startswith("gsk_"):
                print("Warning: Groq keys usually start with 'gsk_'. Saving it anyway.")
            save_key(key)
            print("Saved to .env (this file is never committed)")
        else:
            if not ENV_FILE.exists():
                shutil.copy(ROOT / ".env.example", ENV_FILE)
            print("Skipped. Later, open .env and replace gsk_your_key_here with your key.")

    step(5, "Building the web interface (React)")
    npm = shutil.which("npm")
    if not npm:
        print("Node.js not found: install the LTS version from https://nodejs.org, then re-run this script.")
        print("(Without it the API still runs, but there is no web page.)")
    else:
        if not (ROOT / "web" / "node_modules").exists():
            run([npm, "install"], cwd=ROOT / "web")
        run([npm, "run", "build"], cwd=ROOT / "web")

    step(6, "Downloading demo papers and the SciFact benchmark")
    if "--skip-data" in args:
        print("Skipped (--skip-data)")
    else:
        subprocess.run([str(VENV_PY), str(ROOT / "scripts" / "download_data.py")], cwd=ROOT)

    step(7, "Downloading the local models (embedder, re-ranker, NLI verifier; ~1 GB, once)")
    if "--skip-models" in args:
        print("Skipped (--skip-models): they download on the first question instead")
    else:
        run([VENV_PY, "-c",
             "from veritas.config import settings as s;"
             "from sentence_transformers import SentenceTransformer, CrossEncoder;"
             "from transformers import AutoTokenizer, AutoModelForSequenceClassification;"
             "SentenceTransformer(s.embed_model); CrossEncoder(s.rerank_model);"
             "AutoTokenizer.from_pretrained(s.nli_model); AutoModelForSequenceClassification.from_pretrained(s.nli_model);"
             "print('OK: models ready')"])

    start = "start.bat" if os.name == "nt" else "./start.sh"
    print(f"\nSetup complete. Start VERITAS with:  {start}   then open http://localhost:8000")
    if not current_key():
        print("Reminder: add your Groq key to .env before asking questions.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nA step failed (exit code {e.returncode}). Fix the error above and re-run: python scripts/setup.py")
        sys.exit(1)
