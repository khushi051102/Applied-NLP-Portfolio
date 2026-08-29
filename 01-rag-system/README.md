# Domain RAG System

A retrieval-augmented generation pipeline over a domain-specific document set (default: research papers / PDFs — swap in any corpus). Goes beyond naive top-k RAG: hybrid retrieval (dense + BM25), cross-encoder reranking, and quantitative evaluation with RAGAS.

## Why this project (not just "a RAG demo")

Most RAG tutorials stop at "embed chunks, cosine similarity, done" — that's the part every candidate has. This project demonstrates the parts that actually matter in production:
- **Hybrid retrieval** — dense embeddings alone miss exact keyword/entity matches (product names, IDs, rare terms); BM25 alone misses semantic paraphrase. Combining both closes that gap.
- **Reranking** — a cross-encoder re-scores the top-k candidates using full query-document attention, far more accurate than the bi-encoder similarity used for initial retrieval, at the cost of only reranking a small candidate set.
- **Evaluation, not vibes** — faithfulness, answer relevancy, and context precision measured with RAGAS instead of eyeballing a few outputs.

## Pipeline

```
raw docs -> chunking (src/ingest.py) -> embeddings + BM25 index
        -> query -> hybrid retrieve (src/retrieve.py) -> rerank
        -> generate (src/generate.py) -> RAGAS eval (src/evaluate.py)
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # or point generate.py at a local model
python src/ingest.py --input_dir data/raw --index_dir data/index
python src/retrieve.py --query "your question" --index_dir data/index
python src/evaluate.py --qa_file data/eval/qa_pairs.json
```

## What to fill in yourself

- `data/raw/` — your chosen corpus (pick something you can talk about in an interview: your own coursework papers, a company's public docs, a dataset from Kaggle/arXiv)
- `data/eval/qa_pairs.json` — 20-30 question/answer pairs for evaluation
- Swap `generate.py`'s model call for a local/open-source model if you want to avoid API costs

## Results (fill in after running)

| Metric | Naive RAG | Hybrid + Rerank |
|--------|-----------|------------------|
| Faithfulness | | |
| Answer Relevancy | | |
| Context Precision | | |
