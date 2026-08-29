"""
Long-term vector memory store: embed short facts/turns, store them with a
FAISS index, and retrieve the ones relevant to the current query.

Design choice: memory is per-session-id but persisted to disk, so it
survives across separate conversations with the same user -- this is
the piece that separates "memory" from "just a longer context window".
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class MemoryStore:
    def __init__(self, store_dir: str, embedding_model: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

        self.index_path = self.store_dir / "memory.index"
        self.meta_path = self.store_dir / "memory_meta.json"

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path) as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = []  # list of {"text": ..., "session_id": ...}

    def add(self, text: str, session_id: str):
        emb = self.model.encode([text], normalize_embeddings=True).astype("float32")
        self.index.add(emb)
        self.meta.append({"text": text, "session_id": session_id})
        self._save()

    def retrieve(self, query: str, session_id: str, top_k: int, threshold: float) -> list[str]:
        if self.index.ntotal == 0:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q_emb, min(top_k * 3, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or score < threshold:
                continue
            entry = self.meta[idx]
            if entry["session_id"] == session_id:
                results.append(entry["text"])
            if len(results) >= top_k:
                break
        return results

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f)
