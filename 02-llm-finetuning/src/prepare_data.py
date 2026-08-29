"""
Turn a raw JSONL dataset into instruction-formatted train/val/test splits.

Expected raw format: {"input": ..., "output": ...} per line.
Reformats into a single instruction-tuned prompt string per example, which
is what train_lora.py trains the causal LM on directly (next-token prediction
over the whole formatted string, with the prompt portion masked out of the loss
so the model isn't penalized for "predicting" the question it was given).
"""
import argparse
import json
import random
from pathlib import Path

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_example(instruction: str, example: dict) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction, input=example["input"], output=example["output"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--instruction", default="Complete the following task based on the input.")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.raw_file) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    random.Random(args.seed).shuffle(examples)
    n = len(examples)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)

    splits = {
        "test": examples[:n_test],
        "val": examples[n_test : n_test + n_val],
        "train": examples[n_test + n_val :],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_examples in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_examples:
                text = format_example(args.instruction, ex)
                f.write(json.dumps({"text": text, "output": ex["output"]}) + "\n")
        print(f"{split_name}: {len(split_examples)} examples -> {path}")


if __name__ == "__main__":
    main()
