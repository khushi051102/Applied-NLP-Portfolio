"""
Train LoRA adapters on top of a frozen, 4-bit quantized base model.

Key mechanism notes (for interview-readiness, not just to make it run):
- 4-bit quantization (QLoRA) compresses the frozen base weights to fit in
  memory; LoRA adapters themselves are trained in higher precision, so
  quality loss from quantization is minimal since the base model isn't
  the part being updated.
- LoRA decomposes a weight update dW into two low-rank matrices A (r x d)
  and B (d x r), so dW = B @ A has rank r << d. Only A and B are trained;
  the original weight W stays frozen. At inference, W + B@A is used.
- `target_modules` controls which weight matrices get adapters -- attention
  Q/K/V/O projections are the standard choice since that's where most of
  the task-specific adaptation happens.

Note on trl version: as of trl >= 0.16, dataset_text_field / max_length and
other SFT-specific settings live on SFTConfig (not TrainingArguments/
SFTTrainer directly), and the tokenizer is passed as processing_class=
instead of tokenizer=. This script targets that current API. If you're on
an older trl (< 0.16) pinned elsewhere, these lines will need adjusting back.
"""
import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb_config, device_map="auto"
    )

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=cfg["lora"]["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # sanity check: should be << 1% of base params

    dataset = load_dataset(
        "json",
        data_files={"train": f"{args.data_dir}/train.jsonl", "validation": f"{args.data_dir}/val.jsonl"},
    )

    t = cfg["training"]
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),  # defensive cast
        warmup_steps=t["warmup_ratio"],  # transformers v5+ removed warmup_ratio; warmup_steps now accepts a float < 1 as a ratio
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        eval_strategy="epoch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
