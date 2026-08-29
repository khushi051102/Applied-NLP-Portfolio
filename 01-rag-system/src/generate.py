"""
Generate an answer from retrieved context. Swap `call_llm` for a local
HF model if you want to avoid API costs -- the rest of the pipeline
doesn't care which one you use.
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


def call_llm(prompt: str, cfg: dict) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=cfg["generation"]["model"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg["generation"]["max_tokens"],
        temperature=cfg["generation"]["temperature"],
    )
    return resp.choices[0].message.content


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
