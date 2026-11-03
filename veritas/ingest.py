"""Phase 2a: document ingestion.

Reads PDF / DOCX / TXT / MD files, cleans the text, and splits it into
sentence-aware overlapping chunks that keep their document name and page
number so every citation can point to an exact location.
"""

import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import settings

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
_REF_HEAD = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography|works cited)\s*$", re.I)
_BIB_CUE = re.compile(r"\b(19|20)\d{2}[a-z]?\b|\bet al\.|\barXiv\b|\bProceedings\b|\bpp\. ?\d")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_name: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _doc_id(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:10]


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    text = re.sub(r"\s*\n\s*", " ", text)                   # flatten layout newlines
    text = re.sub(r"\[\d+(?:[,–-]\s*\d+)*\]", "", text)  # drop in-text refs like [12] or [3-5]
    return re.sub(r"\s{2,}", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def read_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, raw_text)] for a file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz  # pymupdf

        with fitz.open(path) as pdf:
            return [(i + 1, page.get_text()) for i, page in enumerate(pdf)]
    if suffix == ".docx":
        import docx

        paras = [p.text for p in docx.Document(path).paragraphs if p.text.strip()]
        # DOCX has no fixed pages; group every 25 paragraphs as a pseudo-page.
        return [(i // 25 + 1, "\n".join(paras[i:i + 25])) for i in range(0, len(paras), 25)]
    if suffix in {".txt", ".md"}:
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {suffix} (supported: {sorted(SUPPORTED)})")


def drop_references(pages: list[tuple[int, str]], block: int = 6) -> list[tuple[int, str]]:
    """Remove bibliography entries. Reference lists are full of paper titles, so they would
    otherwise match almost any query. After the LAST 'References' heading (earlier ones are
    usually a table of contents), keep only line blocks that read like prose (e.g. appendices)."""
    starts = [(i, j) for i, (_, raw) in enumerate(pages) for j, line in enumerate(raw.splitlines()) if _REF_HEAD.match(line)]
    if not starts:
        return pages
    ref_page, ref_line = starts[-1]
    out = []
    for i, (page, raw) in enumerate(pages):
        lines = raw.splitlines()
        if i < ref_page:
            out.append((page, raw))
            continue
        head, tail = (lines[:ref_line], lines[ref_line + 1:]) if i == ref_page else ([], lines)
        kept = [ln for k in range(0, len(tail), block)
                for ln in tail[k:k + block] if not _dense_bib("\n".join(tail[k:k + block]), 4)]
        out.append((page, "\n".join(head + kept)))
    return out


def _dense_bib(text: str, per_100_words: float) -> bool:
    return len(_BIB_CUE.findall(text)) / max(len(text.split()), 1) * 100 >= per_100_words


def looks_like_bibliography(text: str) -> bool:
    """Backup for reference lists without a heading: very dense in years / 'et al.' / venue names."""
    return _dense_bib(text, 10)


def chunk_pages(pages: list[tuple[int, str]], doc_name: str,
                chunk_words: int | None = None, overlap_words: int | None = None) -> list[Chunk]:
    chunk_words = chunk_words or settings.chunk_words
    overlap_words = overlap_words or settings.chunk_overlap_words
    doc_id = _doc_id(doc_name)
    chunks: list[Chunk] = []

    for page, raw in drop_references(pages):
        sentences = split_sentences(clean_text(raw))
        window: list[str] = []
        n_words = 0
        for sent in sentences:
            window.append(sent)
            n_words += len(sent.split())
            if n_words >= chunk_words:
                chunks.append(_make_chunk(doc_id, doc_name, page, window, len(chunks)))
                # carry trailing sentences forward as overlap
                carry, carried = [], 0
                for s in reversed(window):
                    if carried >= overlap_words:
                        break
                    carry.insert(0, s)
                    carried += len(s.split())
                window, n_words = carry, carried
        if window and n_words > 15:  # skip tiny page fragments (headers, page numbers)
            chunks.append(_make_chunk(doc_id, doc_name, page, window, len(chunks)))
    return chunks


def _keep(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if not looks_like_bibliography(c.text)]


def _make_chunk(doc_id: str, doc_name: str, page: int, sents: list[str], idx: int) -> Chunk:
    return Chunk(id=f"{doc_id}-{idx:04d}", doc_id=doc_id, doc_name=doc_name, page=page, text=" ".join(sents))


def ingest_file(path: str | Path) -> list[Chunk]:
    path = Path(path)
    return _keep(chunk_pages(read_pages(path), doc_name=path.name))


def ingest_text(name: str, text: str) -> list[Chunk]:
    """Ingest raw text (used for benchmark corpora such as SciFact abstracts)."""
    return chunk_pages([(1, text)], doc_name=name)
