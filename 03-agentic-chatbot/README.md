03 — Agentic Chatbot: Tool-Calling + Long-Term Memory

A conversational agent that can (1) decide on its own when to call external tools and (2) recall facts from earlier conversations via a persistent vector memory store — not just whatever fits in the current context window.

What it does
Tool-calling: the LLM decides, per message, whether it needs a tool (calculator, search) or can answer directly — no hardcoded keyword routing. Uses structured function-calling (Groq's OpenAI-compatible tools= API), not prompt-based parsing.
Long-term memory: every turn is embedded and stored in a FAISS vector index, scoped by session_id, persisted to disk. On each new message, the most relevant past facts (above a similarity threshold) are retrieved and injected into the system prompt — so the agent can recall something from a previous session, not just the current one.
Architecture
User message
     │
     ▼
MemoryStore.retrieve()  ──►  relevant past facts (FAISS cosine similarity search,
     │                        filtered to this session_id)
     ▼
Inject facts into system prompt
     │
     ▼
LLM call (Groq, tools=[calculator, search], tool_choice="auto")
     │
     ├── model picks a tool ──► execute tool ──► feed result back ──► LLM produces final answer
     └── model answers directly ──► done
     │
     ▼
MemoryStore.add()  ──►  compact summary of this turn written back to the index
     │                   (trivial turns like "thanks"/"ok" are filtered out)
     ▼
Response returned via FastAPI /chat endpoint
Tech stack
Component	Choice
Generation	Groq API (openai/gpt-oss-120b), OpenAI-compatible tool-calling
Embeddings	sentence-transformers/all-MiniLM-L6-v2 (same as 01-rag-system)
Vector store	FAISS IndexFlatIP (cosine similarity via normalized embeddings)
API layer	FastAPI + Pydantic
Tool sandbox	AST-based safe expression evaluator (no eval())
Setup
bash
conda activate tensorflow_env   # or your preferred env
cd 03-agentic-chatbot
pip install -r requirements.txt

Get a free Groq API key (no credit card required) at console.groq.com, then:

bash
export GROQ_API_KEY="your-key-here"
Usage
bash
cd src
uvicorn app:app --reload

In a separate terminal:

bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "What is 8471 * 39264, exactly?"}'
Example run

Tool-calling (forces a real calculator call — too large to compute reliably from model knowledge alone):

> What is 8471 * 39264, exactly?
< 8471 × 39,264 = 332,605,344

Memory recall (fact stated in one message, correctly recalled in a later, separate message — same session_id):

> My favorite programming language is Rust.
< That's a great choice! Rust's focus on safety, performance, and concurrency...

> What is my favorite programming language?
< Your favorite programming language is Rust.
Design decisions & known limitations
Groq free tier over OpenAI: avoids billing entirely while keeping the same structured tools= function-calling contract as OpenAI's API (Groq's SDK is intentionally OpenAI-compatible). Trade-off: free-tier model availability changes without much notice — this project originally targeted llama-3.3-70b-versatile, which Groq deprecated mid-project, requiring a one-line config change to openai/gpt-oss-120b. The model name lives entirely in config.yaml for exactly this reason.
Session-scoped FAISS retrieval is a heuristic, not a guarantee: IndexFlatIP has no native metadata filtering, so retrieve() searches the global index for the top k*3 nearest neighbors, then filters down to the current session_id. This works well for a single-session demo but would under-retrieve for a session that's a small minority of a large, multi-user index. A production version would use per-session indices or a vector DB with native filtering (e.g. Qdrant, Weaviate).
Filler-turn filtering is a simple stoplist, not a classifier — good enough to keep obvious noise ("thanks", "ok") out of the memory store, not a robust relevance filter.
search tool is a stub — returns a placeholder string rather than hitting a real API. Swapping in a free option (Wikipedia REST API, DuckDuckGo Instant Answer) is natural future work.
Tool-call cap handling: if the model requests more tool calls in one turn than max_tool_calls_per_turn allows, calls past the cap still receive a placeholder "tool" response rather than being silently dropped — required by the API's contract that every tool_call_id gets a matching response.
Future work
Real search tool (Wikipedia/DuckDuckGo API)
Per-session FAISS indices for correct retrieval filtering at scale
Optional local-model backend (e.g. Qwen2.5-1.5B) as a fallback path, for demonstrating tool-calling without any external API dependency