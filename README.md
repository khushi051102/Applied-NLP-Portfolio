# NLP Portfolio — Applied Projects

Three end-to-end applied NLP projects built to demonstrate production-relevant skills: retrieval-augmented generation, parameter-efficient fine-tuning, and agentic conversational systems with memory and tool use.

## Projects

| # | Project | Core Skills | Stack |
|---|---------|-------------|-------|
| 1 | [Domain RAG System](01-rag-system/) | Chunking, embeddings, hybrid retrieval, reranking, RAG evaluation | LangChain, FAISS, sentence-transformers, RAGAS |
| 2 | [LoRA Fine-Tuning Pipeline](02-llm-finetuning/) | PEFT/LoRA, quantization, instruction tuning, eval harness | HuggingFace Transformers, PEFT, bitsandbytes |
| 3 | [Agentic Chatbot with Memory](03-agentic-chatbot/) | Multi-turn dialogue, tool calling, long-term memory, deployment | LangGraph/FastAPI, vector memory store |

## Why these three

Each project targets a different piece of the applied-NLP stack that shows up in ML engineering job descriptions: **retrieval systems**, **model adaptation**, and **agent orchestration**. Together they cover the pipeline from raw data → retrieval/inference → deployed, evaluatable system, rather than three isolated notebooks.

## Suggested resume bullets

- Built a hybrid retrieval-augmented generation system over [N] domain documents, combining dense + BM25 retrieval with cross-encoder reranking, improving answer faithfulness by [X]% (RAGAS) over naive RAG.
- Fine-tuned a [7B] open-source LLM using LoRA/QLoRA on a domain-specific instruction dataset, achieving [X]% improvement on [task] while reducing trainable parameters by >99%.
- Designed and deployed an agentic chatbot with persistent vector-based memory and external tool calling, served via FastAPI with sub-[X]s p95 latency.

> Fill in real numbers once you've run experiments — don't put placeholder metrics on the actual resume.

## Setup

Each subproject has its own `requirements.txt` and `README.md` with run instructions. Recommended: one virtualenv per project to avoid dependency conflicts (transformers/peft versions vs. langchain versions can clash).

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r 01-rag-system/requirements.txt
```

## Repo structure

```
nlp-portfolio/
├── 01-rag-system/
├── 02-llm-finetuning/
├── 03-agentic-chatbot/
├── .gitignore
└── LICENSE
```
