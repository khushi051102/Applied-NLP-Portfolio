cat > README.md << 'EOF'
# Applied NLP Portfolio

Three applied NLP/LLM projects built end-to-end — retrieval, fine-tuning, and agentic tool-use — each with a working pipeline, custom evaluation, and honest documentation of what does and doesn't work.

Built to go deeper than "call an API": understanding *why* hybrid retrieval beats dense-only search, what QLoRA actually changes about training memory, and how tool-calling and long-term memory work without relying on a single all-in-one framework.

## Projects

| | What it does | Key techniques |
|---|---|---|
| **[01 — RAG System](./01-rag-system)** | Hybrid dense + sparse retrieval with cross-encoder reranking, then grounded LLM generation over SQuAD | FAISS, BM25, cross-encoder reranking, custom EM/F1/retrieval-hit-rate eval |
| **[02 — LLM Fine-Tuning](./02-llm-finetuning)** | QLoRA fine-tune for structured text-to-JSON extraction, with before/after eval | 4-bit QLoRA, PEFT/LoRA, synthetic data generation, exact-match eval |
| **[03 — Agentic Chatbot](./03-agentic-chatbot)** | Conversational agent with autonomous tool-calling and persistent long-term memory | Structured function-calling, FAISS vector memory, FastAPI |

## Design decisions common across all three

- **Same base model where local generation is used** (`Qwen/Qwen2.5-1.5B-Instruct`) — a deliberate choice for consistency and to keep everything runnable without paid API access, not an oversight.
- **Custom evaluation over off-the-shelf metrics, when the off-the-shelf option doesn't fit the task** — e.g. exact-match/F1/retrieval-hit-rate instead of RAGAS (which needs an LLM judge) in 01, and parse-rate/exact-match/field-accuracy instead of ROUGE (built for free text, not structured JSON) in 02.
- **Config-driven, not hardcoded** — model choice, hyperparameters, and backend (local vs. API) live in each project's `config.yaml`, not scattered through the code.
- **Honest limitations documented, not hidden** — each project's README calls out what doesn't work well and why (e.g. 01's generation bottleneck, 02's date-field data bug, 03's session-scoped retrieval heuristic), rather than presenting only the best-case numbers.

## Repo structure

\`\`\`
Applied-NLP-Portfolio/
├── 01-rag-system/
├── 02-llm-finetuning/
├── 03-agentic-chatbot/
├── .gitignore
└── LICENSE
\`\`\`

Each project is self-contained with its own \`requirements.txt\`, \`config.yaml\`, and README — see the individual project READMEs (linked above) for setup, usage, architecture, and full results.
EOF