"""
Generate an answer from retrieved context. Backend is chosen via
config.yaml's generation.backend ("local" or "openai") -- the rest
of the pipeline doesn't care which one is used.
"""
import argparse
from pathlib import Path

import yaml
from retrieve import hybrid_retrieve, load_config

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

_local_pipeline = None  # cached so the model loads once, not per-query


def _call_llm_local(prompt: str, cfg: dict) -> str:
    """
    Run generation on a small local HF instruction model instead of a paid API.
    Cached at module level because loading the model (~3GB download first time,
    then several seconds to load into memory) is too slow to repeat per query.
    """
    global _local_pipeline
    from transformers import pipeline

    if _local_pipeline is None:
        _local_pipeline = pipeline(
            "text-generation",
            model=cfg["generation"]["local_model"],
            device_map="auto",
        )

    messages = [{"role": "user", "content": prompt}]
    out = _local_pipeline(
        messages,
        max_new_tokens=cfg["generation"]["max_tokens"],
        temperature=cfg["generation"]["temperature"],
        do_sample=cfg["generation"]["temperature"] > 0,
    )
    # pipeline returns the full conversation; the model's reply is the last turn
    return out[0]["generated_text"][-1]["content"]


def _call_llm_openai(prompt: str, cfg: dict) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=cfg["generation"]["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg["generation"]["max_tokens"],
        temperature=cfg["generation"]["temperature"],
    )
    return resp.choices[0].message.content


def call_llm(prompt: str, cfg: dict) -> str:
    """Dispatches to a local HF model or the OpenAI API based on config.yaml."""
    backend = cfg["generation"].get("backend", "local")
    if backend == "local":
        return _call_llm_local(prompt, cfg)
    elif backend == "openai":
        return _call_llm_openai(prompt, cfg)
    else:
        raise ValueError(f"Unknown generation backend: {backend}")


def answer_question(question: str, index_dir: Path, cfg: dict) -> dict:
    retrieved = hybrid_retrieve(question, index_dir, cfg)
    context = "\n\n---\n\n".join(r["text"] for r in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = call_llm(prompt, cfg)
    return {"question": question, "answer": answer, "contexts": [r["text"] for r in retrieved]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = answer_question(args.query, Path(args.index_dir), cfg)
    print("\nANSWER:\n", result["answer"])


if __name__ == "__main__":
    main()