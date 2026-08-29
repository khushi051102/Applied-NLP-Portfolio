"""
Hybrid retrieval: combine dense (FAISS) and sparse (BM25) scores, then
rerank the merged candidate set with a cross-encoder.

Score fusion approach: min-max normalize each score list to [0,1] independently
(they're on incomparable scales -- cosine similarity vs. BM25 term-weight sums),
then take a weighted sum via `hybrid_alpha`. This is simpler and more transparent
than reciprocal-rank fusion and works well for small/medium corpora.
"""
import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def minmax(scores: np.ndarray) -> np.ndarray:
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_retrieve(query: str, index_dir: Path, cfg: dict) -> list[dict]:
    with open(index_dir / "chunks.json") as f:
        chunks = json.load(f)
    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    dense_index = faiss.read_index(str(index_dir / "dense.index"))

    embed_model = SentenceTransformer(cfg["embedding_model"])
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")

    r = cfg["retrieval"]
    dense_scores, dense_ids = dense_index.search(q_emb, r["dense_top_k"])
    dense_scores, dense_ids = dense_scores[0], dense_ids[0]

    bm25_scores_all = np.asarray(bm25.get_scores(query.split()))
    bm25_top_ids = np.argsort(bm25_scores_all)[::-1][: r["bm25_top_k"]]
    bm25_scores = bm25_scores_all[bm25_top_ids]

    candidate_ids = sorted(set(dense_ids.tolist()) | set(bm25_top_ids.tolist()))

    dense_lookup = dict(zip(dense_ids.tolist(), minmax(dense_scores)))
    bm25_lookup = dict(zip(bm25_top_ids.tolist(), minmax(bm25_scores)))

    alpha = r["hybrid_alpha"]
    fused = []
    for cid in candidate_ids:
        d = dense_lookup.get(cid, 0.0)
        b = bm25_lookup.get(cid, 0.0)
        fused.append((cid, alpha * d + (1 - alpha) * b))
    fused.sort(key=lambda x: x[1], reverse=True)

    top_candidates = [{"id": cid, "text": chunks[cid], "fused_score": float(s)} for cid, s in fused[:20]]

    # Rerank with cross-encoder: scores (query, doc) pairs jointly instead of
    # comparing independent embeddings, which is much better at judging
    # "does this chunk actually answer the question" vs. "is this topically similar".
    reranker = CrossEncoder(cfg["reranker_model"])
    pairs = [[query, c["text"]] for c in top_candidates]
    rerank_scores = reranker.predict(pairs)
    for c, s in zip(top_candidates, rerank_scores):
        c["rerank_score"] = float(s)

    top_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return top_candidates[: r["rerank_top_k"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = hybrid_retrieve(args.query, Path(args.index_dir), cfg)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] rerank={r['rerank_score']:.3f} fused={r['fused_score']:.3f}")
        print(r["text"][:300])


if __name__ == "__main__":
    main()
