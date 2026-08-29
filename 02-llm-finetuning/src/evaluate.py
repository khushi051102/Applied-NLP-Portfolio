"""
Compare base model vs. LoRA fine-tuned model on a held-out test set.
Swap the metric in `score` for whatever fits your task (exact match /
F1 for extraction, ROUGE for summarization, accuracy for classification).
"""
import argparse
import json

from evaluate import load as load_metric
from inference import load_config, load_model, generate
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def load_base_only(cfg: dict):
    bnb_config = BitsAndBytesConfig(load_in_4bit=cfg["quantization"]["load_in_4bit"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def run_eval(model, tokenizer, test_examples: list[dict]) -> list[str]:
    return [generate(model, tokenizer, ex["text"]) for ex in test_examples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(args.test_file) as f:
        test_examples = [json.loads(line) for line in f]

    rouge = load_metric("rouge")

    print("Running base model...")
    base_model, tokenizer = load_base_only(cfg)
    base_preds = run_eval(base_model, tokenizer, test_examples)
    del base_model
    torch.cuda.empty_cache()

    print("Running fine-tuned model...")
    ft_model, tokenizer = load_model(args.adapter_dir, cfg)
    ft_preds = run_eval(ft_model, tokenizer, test_examples)

    refs = [ex["output"] for ex in test_examples]
    base_scores = rouge.compute(predictions=base_preds, references=refs)
    ft_scores = rouge.compute(predictions=ft_preds, references=refs)

    print("\nBase model:", base_scores)
    print("Fine-tuned model:", ft_scores)


if __name__ == "__main__":
    main()
