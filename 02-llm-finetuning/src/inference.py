"""
Load the frozen base model + trained LoRA adapter for generation.
The adapter is merged at load time via PEFT so inference looks like
normal HF generation.
"""
import argparse

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(adapter_dir: str, cfg: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model(args.adapter_dir, cfg)
    print(generate(model, tokenizer, args.prompt))


if __name__ == "__main__":
    main()
