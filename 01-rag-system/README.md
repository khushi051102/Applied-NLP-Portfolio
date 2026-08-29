# 01 — RAG System: Hybrid Retrieval + Generation

A retrieval-augmented generation (RAG) pipeline built from scratch: hybrid dense + sparse
retrieval, cross-encoder reranking, and LLM-based answer generation, evaluated against
real SQuAD question-answer pairs.

## Why I built this

To understand RAG beyond the "call a library" level — specifically why hybrid retrieval
(dense + BM25) outperforms either alone, why reranking with a cross-encoder matters versus
just trusting embedding similarity, and how to actually measure whether a RAG system is
retrieving the right information vs. just sounding confident.

## Architecture
Raw documents (.txt/.pdf)
│
▼
ingest.py ──► chunk (512 words, 64 overlap)
│ │
│ ▼
│ sentence-transformers embeddings
│ │
▼ ▼
FAISS dense index BM25 sparse index
│ │
└──────┬───────┘
▼
retrieve.py — hybrid fusion (min-max normalize + weighted sum)
│
▼
cross-encoder reranking (top 20 → top 5)
│
▼
generate.py — LLM answers using ONLY retrieved context
│
▼
evaluate.py — exact match / F1 / retrieval hit rate vs. gold answers

**Why hybrid retrieval, not just dense embeddings:** dense embeddings are good at semantic/
paraphrase matching but blur exact terms — names, numbers, jargon. BM25 catches those.
Fusing both, then reranking the merged set with a cross-encoder (which scores query+chunk
pairs jointly, not independently), gives noticeably better top-1 accuracy than either alone.

## Tech stack

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector index:** FAISS (flat, cosine similarity)
- **Sparse index:** BM25 (`rank_bm25`)
- **Generation:** local `Qwen/Qwen2.5-1.5B-Instruct` (swappable to OpenAI via config)
- **Dataset:** [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) (Wikipedia paragraphs +
  human-written QA pairs) — used both as the knowledge base and as ground truth for evaluation

## Setup

```bash
cd 01-rag-system
pip install -r requirements.txt
```

## Usage

**1. Prepare data** — pulls a slice of SQuAD, writes it as per-article `.txt` files:
```bash
python data/prepare_data.py
```

**2. Build indexes** — chunks documents, builds FAISS + BM25 indexes:
```bash
python src/ingest.py --input_dir data/raw --index_dir data/index --config config.yaml
```

**3. Query the retriever directly** (inspect what gets retrieved, before generation):
```bash
python src/retrieve.py --query "Which NFL team represented the AFC at Super Bowl 50?" --index_dir data/index --config config.yaml
```

**4. Ask a full question** (retrieve → rerank → generate):
```bash
python src/generate.py --query "Which NFL team represented the AFC at Super Bowl 50?" --index_dir data/index --config config.yaml
```

**5. Evaluate** against gold SQuAD answers (exact match, F1, retrieval hit rate):
```bash
python src/evaluate.py --qa_file data/qa_pairs.json --index_dir data/index --config config.yaml --limit 50
```

## Example

**Query:** *"Which NFL team represented the AFC at Super Bowl 50?"*

**Top retrieved chunk** (rerank score 6.86, next-highest 1.05):
> "Super Bowl 50 was an American football game to determine the champion of the National
> Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion
> Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers
> 24–10 to earn their third Super Bowl title."

**Generated answer:** *"The Denver Broncos represented the AFC at Super Bowl 50."*

## Evaluation results

*(n=5 questions — small sample due to local CPU inference constraints; see note below)*

| Metric | Score |
|---|---|
| Exact Match | 0.20 |
| F1 | 0.35 |
| Retrieval Hit Rate | 1.00 |

**Key finding:** retrieval hit rate was perfect (1.00) — the correct answer was present in
retrieved context for every question. Exact-match/F1 were lower, indicating generation
(not retrieval) is the current bottleneck: the local `Qwen2.5-1.5B-Instruct` model tends to
paraphrase or add extra words rather than extracting terse exact-match spans, a known
tradeoff of small local models vs. larger API-based ones. This separation of retrieval vs.
generation quality — rather than one blended score — was the point of building custom
metrics instead of a single end-to-end accuracy number.

## Config

All pipeline behavior (chunk size, retrieval weights, generation backend/model) is controlled
via `config.yaml` — no hardcoded values in the pipeline scripts.

## Notes on generation backend

`generate.py` supports both a local Hugging Face model (`backend: local`, default — no API
key needed, works fully offline) and OpenAI (`backend: openai`, requires `OPENAI_API_KEY`).
Local is the default so the project runs out-of-the-box for anyone cloning it.