"""
Ingest raw documents -> chunk -> build a dense (FAISS) index and a BM25 index.

Why both indexes are built here (not just one):
- FAISS holds dense embeddings for semantic/paraphrase matching.
- BM25 holds a sparse term-frequency index for exact lexical matching
  (entity names, numbers, jargon that embeddings often blur together).
retrieve.py combines scores from both at query time.
"""
import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def read_documents(input_dir: str) -> list[dict]:
    """Read .txt and .pdf files into {'source': path, 'text': content} dicts."""
    docs = []
    for path in Path(input_dir).glob("**/*"):
        if path.suffix.lower() == ".txt":
            docs.append({"source": str(path), "text": path.read_text(errors="ignore")})
        elif path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append({"source": str(path), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple sliding-window word-based chunker.

    Overlap matters: without it, an answer that spans a chunk boundary
    gets split and neither chunk alone is enough context to answer from.
    """
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def build_indexes(chunks: list[str], embed_model_name: str, out_dir: Path):
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / "dense.index"))

    tokenized = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    with open(out_dir / "chunks.json", "w") as f:
        json.dump(chunks, f)

    print(f"Indexed {len(chunks)} chunks -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = read_documents(args.input_dir)
    if not docs:
        raise SystemExit(f"No .txt/.pdf files found in {args.input_dir}")

    all_chunks = []
    for doc in docs:
        all_chunks.extend(
            chunk_text(doc["text"], cfg["chunking"]["chunk_size"], cfg["chunking"]["chunk_overlap"])
        )

    build_indexes(all_chunks, cfg["embedding_model"], out_dir)


if __name__ == "__main__":
    main()
