"""Central configuration. Every tunable value lives here so experiments are reproducible."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    # Paths
    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / "data" / "chroma"
    cache_dir: Path = ROOT / "data" / "cache"
    output_dir: Path = ROOT / "outputs"

    # LLM (generator)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    gen_temperature: float = 0.1          # grounded answer: near-deterministic
    sample_temperature: float = 0.8       # self-consistency samples: stochastic
    max_tokens: int = 1024
    reasoning_effort: str = "low"         # gpt-oss models: keep hidden reasoning short (free tier = 8k tokens/min)

    # Models (small enough for an 8 GB laptop)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    # Chunking
    chunk_words: int = 220                # ~300 tokens
    chunk_overlap_words: int = 40

    # Retrieval
    retrieve_k: int = 20                  # dense candidates
    rerank_k: int = 5                     # passages sent to the LLM
    max_per_doc: int = 2                  # source diversity: at most N passages from one document

    # Verification
    nli_window_sentences: int = 3         # also test each claim against 3-sentence windows (0 = whole passage only)
    nli_window_stride: int = 2
    nli_window_top: int = 0               # window-check cited passages + N best matches (0 = all; use 2-3 on CPU-only hosts)
    n_consistency_samples: int = 4
    w_nli: float = 0.6                    # confidence = w_nli*entail + w_cons*consistency - penalty
    w_consistency: float = 0.4
    contradiction_penalty: float = 0.5
    verified_threshold: float = 0.70
    unsupported_threshold: float = 0.40
    regenerate_weak_claims: bool = True   # single fixed pass, never a loop

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_dir, self.cache_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
