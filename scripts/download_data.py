"""Download the SciFact benchmark and the demo paper collection (LLM hallucination literature).

Works on macOS, Linux and Windows (standard library only):
    python scripts/download_data.py
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
PAPERS = {
    "2005.11401": "rag_lewis_2020.pdf",
    "2109.07958": "truthfulqa_2022.pdf",
    "2202.03629": "hallucination_survey_ji_2023.pdf",
    "2303.08896": "selfcheckgpt_2023.pdf",
    "2305.14251": "factscore_2023.pdf",
    "2309.11495": "chain_of_verification_2023.pdf",
    "2405.05904": "finetuning_new_knowledge_2024.pdf",
}


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    scifact, docs = ROOT / "data" / "scifact", ROOT / "data" / "sample_docs"
    scifact.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    failed = []
    if not (scifact / "data" / "corpus.jsonl").exists():
        print("Downloading SciFact benchmark ...")
        archive = scifact / "data.tar.gz"
        try:
            fetch(SCIFACT_URL, archive)
            with tarfile.open(archive) as t:
                if hasattr(tarfile, "data_filter"):  # safe extraction on newer Pythons
                    t.extractall(scifact, filter="data")
                else:
                    t.extractall(scifact)
        except Exception as e:
            failed.append(f"SciFact ({e})")

    for arxiv_id, name in PAPERS.items():
        dest = docs / name
        if dest.exists():
            continue
        print(f"Downloading {name} ...")
        try:
            fetch(f"https://arxiv.org/pdf/{arxiv_id}", dest)
        except Exception as e:
            failed.append(f"{name} ({e})")

    if failed:
        print("Could not download: " + "; ".join(failed) + ". Re-run this script to retry.")
        return 1
    print("Data ready: data/scifact and data/sample_docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
