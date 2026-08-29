"""
Evaluate the RAG pipeline with RAGAS: faithfulness (is the answer
grounded in the retrieved context, not hallucinated), answer relevancy
(does it actually address the question), and context precision (is
retrieval pulling in the right chunks, not noise).

qa_file format: JSON list of {"question": ..., "ground_truth": ...}
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from generate import answer_question
from retrieve import load_config


def build_eval_dataset(qa_file: str, index_dir: Path, cfg: dict) -> Dataset:
    with open(qa_file) as f:
        qa_pairs = json.load(f)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for pair in qa_pairs:
        result = answer_question(pair["question"], index_dir, cfg)
        rows["question"].append(pair["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append(result["contexts"])
        rows["ground_truth"].append(pair["ground_truth"])

    return Dataset.from_dict(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--index_dir", default="data/index")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = build_eval_dataset(args.qa_file, Path(args.index_dir), cfg)

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    print(results)


if __name__ == "__main__":
    main()
