"""
Compare base model vs. LoRA fine-tuned model on the held-out text-to-JSON
extraction test set.

Metric rationale: this is a structured-extraction task, not free-text
generation, so ROUGE/BLEU don't fit -- a JSON output is either correct or
it isn't. We report:
  - parse_rate: fraction of outputs that are valid, parseable JSON at all
    (the base model, never trained on this exact format, often fails here --
    trailing commentary, markdown code fences, malformed brackets, etc.)
  - exact_match: fraction where the parsed JSON exactly equals the ground
    truth dict (all 5 keys, all values, e.g. amount must be an int not a
    string "350")
  - field_accuracy: average per-field correctness across customer/item/
    quantity/amount/date, for parseable outputs -- a softer signal that
    shows *how close* wrong answers were, not just pass/fail
"""
import argparse
import json

from inference import load_config, load_model, generate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

EXPECTED_KEYS = ["customer", "item", "quantity", "amount", "date"]


def load_base_only(cfg: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb_config, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def extract_json(raw_output: str, prompt: str) -> dict | None:
    """
    generate() returns the full decoded sequence including the prompt, so
    strip that off first, then try to parse whatever comes after
    '### Response:' as JSON. Returns None if it isn't parseable -- that's
    itself a meaningful result (base model often fails to even produce
    valid JSON), not an error to crash on.
    """
    completion = raw_output[len(prompt):] if raw_output.startswith(prompt) else raw_output
    # Model sometimes wraps output in a markdown code fence; strip that if present.
    completion = completion.strip().strip("`").strip()
    if completion.startswith("json"):
        completion = completion[4:].strip()
    # Take the first {...} block in case there's trailing commentary.
    start = completion.find("{")
    end = completion.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(completion[start : end + 1])
    except json.JSONDecodeError:
        return None


def score(predicted: dict | None, reference: dict) -> dict:
    if predicted is None:
        return {"parsed": False, "exact_match": False, "field_correct": {k: False for k in EXPECTED_KEYS}}
    field_correct = {k: (predicted.get(k) == reference.get(k)) for k in EXPECTED_KEYS}
    exact_match = all(field_correct.values())
    return {"parsed": True, "exact_match": exact_match, "field_correct": field_correct}


def run_eval(model, tokenizer, test_examples: list[dict]) -> list[dict]:
    results = []
    for ex in test_examples:
        prompt = ex["text"].split("### Response:")[0] + "### Response:\n"
        raw_output = generate(model, tokenizer, prompt)
        predicted = extract_json(raw_output, prompt)
        reference = json.loads(ex["output"])
        results.append(score(predicted, reference))
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    parse_rate = sum(r["parsed"] for r in results) / n
    exact_match = sum(r["exact_match"] for r in results) / n
    field_totals = {k: 0 for k in EXPECTED_KEYS}
    for r in results:
        for k in EXPECTED_KEYS:
            field_totals[k] += r["field_correct"][k]
    field_accuracy = {k: v / n for k, v in field_totals.items()}
    return {"parse_rate": parse_rate, "exact_match": exact_match, "field_accuracy": field_accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(args.test_file) as f:
        test_examples = [json.loads(line) for line in f]

    print(f"Running base model on {len(test_examples)} test examples...")
    base_model, tokenizer = load_base_only(cfg)
    base_results = run_eval(base_model, tokenizer, test_examples)
    del base_model
    torch.cuda.empty_cache()

    print("Running fine-tuned model...")
    ft_model, tokenizer = load_model(args.adapter_dir, cfg)
    ft_results = run_eval(ft_model, tokenizer, test_examples)

    base_summary = summarize(base_results)
    ft_summary = summarize(ft_results)

    print("\n=== Base model ===")
    print(json.dumps(base_summary, indent=2))
    print("\n=== Fine-tuned model ===")
    print(json.dumps(ft_summary, indent=2))

    with open("eval_results.json", "w") as f:
        json.dump({"base": base_summary, "fine_tuned": ft_summary}, f, indent=2)
    print("\nSaved to eval_results.json")


if __name__ == "__main__":
    main()
