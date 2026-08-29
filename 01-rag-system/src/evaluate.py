"""
Evaluate the RAG pipeline with two interpretable, fast metrics:

- Retrieval hit rate: does the gold answer string appear anywhere in the
  retrieved chunks? Isolates retrieval quality from generation quality.
- Exact match / F1: standard SQuAD-style string comparison between the
  generated answer and the gold answer. Isolates generation quality.

Why not an LLM-as-judge framework (e.g. RAGAS) here: those metrics are
validated against strong judge models (GPT-4 class); a small local model
acting as its own judge is unreliable, and doubles the number of slow
local LLM calls per question. Exact-match/F1 against gold answers is the
same style of metric SQuAD itself uses, and it's cheap and reproducible.
"""
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

from generate import answer_question
from retrieve import load_config


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_hit(contexts: list[str], gold: str) -> bool:
    gold_norm = normalize(gold)
    return any(gold_norm in normalize(c) for c in contexts)


def run_eval(qa_file: str, index_dir: Path, cfg: dict, limit: int = None):
    with open(qa_file) as f:
        qa_pairs = json.load(f)

    if limit:
        qa_pairs = qa_pairs[:limit]

    em_scores, f1_scores, hit_scores = [], [], []
    per_question = []

    for i, pair in enumerate(qa_pairs, 1):
        result = answer_question(pair["question"], index_dir, cfg)
        gold = pair["answer"]

        em = exact_match(result["answer"], gold)
        f1 = f1_score(result["answer"], gold)
        hit = retrieval_hit(result["contexts"], gold)

        em_scores.append(em)
        f1_scores.append(f1)
        hit_scores.append(hit)

        per_question.append({
            "question": pair["question"],
            "gold": gold,
            "predicted": result["answer"],
            "exact_match": em,
            "f1": round(f1, 3),
            "retrieval_hit": hit,
        })

        print(f"[{i}/{len(qa_pairs)}] EM={em} F1={f1:.2f} hit={hit} | {pair['question'][:60]}")

    summary = {
        "n_questions": len(qa_pairs),
        "exact_match": round(sum(em_scores) / len(em_scores), 3),
        "f1": round(sum(f1_scores) / len(f1_scores), 3),
        "retrieval_hit_rate": round(sum(hit_scores) / len(hit_scores), 3),
    }
    return summary, per_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--index_dir", default="data/index")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions to evaluate (default 50, for speed)")
    parser.add_argument("--out", default="data/eval/results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary, per_question = run_eval(args.qa_file, Path(args.index_dir), cfg, limit=args.limit)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_question": per_question}, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()