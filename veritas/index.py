"""Phase 2b: dense vector index.

Embeds chunks with a Sentence-Transformer bi-encoder and stores them in a
persistent ChromaDB collection (cosine similarity).
"""

import threading
from functools import lru_cache

from .config import settings
from .ingest import Chunk

_client = None
_client_lock = threading.Lock()


def get_client():
    """One shared ChromaDB client per process: creating two at once on the same path crashes ChromaDB."""
    global _client
    with _client_lock:
        if _client is None:
            import chromadb

            settings.ensure_dirs()
            _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        return _client


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


def embed(texts: list[str]) -> list[list[float]]:
    vecs = get_embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()


class VectorIndex:
    def __init__(self, collection: str = "default"):
        self.client = get_client()
        self.name = collection
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], batch: int = 256) -> int:
        existing = set(self.col.get(ids=[c.id for c in chunks])["ids"]) if chunks else set()
        new = [c for c in chunks if c.id not in existing]
        for i in range(0, len(new), batch):
            part = new[i:i + batch]
            self.col.add(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=embed([c.text for c in part]),
                metadatas=[{"doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page} for c in part],
            )
        return len(new)

    def query(self, text: str, k: int, docs: list[str] | None = None) -> list[tuple[Chunk, float]]:
        """docs: optional list of doc_names to search within (None = whole collection)."""
        if self.count() == 0:
            return []
        where = None
        if docs:
            where = {"doc_name": docs[0]} if len(docs) == 1 else {"doc_name": {"$in": list(docs)}}
        res = self.col.query(query_embeddings=embed([text]), n_results=min(k, self.count()), where=where)
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(id=cid, doc_id=meta["doc_id"], doc_name=meta["doc_name"], page=int(meta["page"]), text=doc)
            out.append((chunk, 1.0 - dist))  # cosine distance -> similarity
        return out

    def first_page(self, docs: list[str]) -> list[tuple[Chunk, float]]:
        """Page-1 chunks (title, abstract, introduction) of the given documents."""
        where = {"$and": [{"doc_name": {"$in": list(docs)}}, {"page": 1}]}
        res = self.col.get(where=where, include=["documents", "metadatas"])
        return [(Chunk(id=i, doc_id=m["doc_id"], doc_name=m["doc_name"], page=1, text=d), 0.0)
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def count(self) -> int:
        return self.col.count()

    def documents(self) -> dict[str, int]:
        """doc_name -> number of chunks."""
        metas = self.col.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for m in metas:
            counts[m["doc_name"]] = counts.get(m["doc_name"], 0) + 1
        return dict(sorted(counts.items()))

    def delete_document(self, doc_name: str) -> None:
        self.col.delete(where={"doc_name": doc_name})

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})
