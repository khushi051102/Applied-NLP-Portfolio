# Agentic Chatbot with Memory and Tool Use

A multi-turn conversational agent that maintains long-term memory across sessions (not just the current context window) and can call external tools, served as a FastAPI app.

## Why this is different from a basic chatbot wrapper

A basic chatbot is `messages.append(); call_llm()`. This project adds two things that turn it into an "agent" rather than a chat wrapper:
- **Persistent memory** — facts from past turns/sessions are embedded and stored in a vector store, then retrieved and injected into context when relevant, so the agent remembers user preferences and past conversations beyond the current context window (which real assistants need, since context windows are finite and expensive).
- **Tool calling** — the agent decides, based on the query, whether it needs to call a tool (calculator, web search stub, database lookup) rather than answering from parametric knowledge alone, and incorporates the tool result into its final answer.

## Architecture

```
user message -> memory retrieval (src/memory.py) -> agent reasoning (src/agent.py)
             -> [optional tool call (src/tools.py)] -> response
             -> write new facts back to memory store
             -> served via src/app.py (FastAPI)
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
uvicorn src.app:app --reload
# then POST to http://localhost:8000/chat with {"session_id": "...", "message": "..."}
```

## What to fill in yourself

- `src/tools.py` — currently has a calculator and a stub search tool; swap in something real relevant to a portfolio story (a weather API, a small SQLite product DB, a scraper)
- Memory store default is a local FAISS index (`data/memory/`); swap for a hosted vector DB (Pinecone/Weaviate/Qdrant) if you want to show you can work with managed infra
- Add a small eval set of multi-turn conversations to test that memory retrieval actually pulls the right facts back (this is the piece that's easy to get wrong silently)

## Talking points for interviews

- How you decided what to write to memory vs. what's just conversational noise
- The tradeoff between always retrieving memory (slower, more context) vs. deciding when it's needed
- How tool-call routing works: the model outputs a structured decision, not a fixed if/else
