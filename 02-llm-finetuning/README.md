# LoRA Fine-Tuning Pipeline

Parameter-efficient fine-tuning (LoRA) of an open-source LLM on a domain-specific instruction/task dataset, with a proper before/after evaluation harness. Built to demonstrate *why* PEFT is used, not just how to call `.train()`.

## Why LoRA instead of full fine-tuning

Full fine-tuning updates every weight in a multi-billion-parameter model — expensive in GPU memory and easy to overfit on small datasets, and you end up storing a full copy of the model per task. LoRA freezes the base model and injects small trainable low-rank matrices into the attention layers, so:
- Trainable parameters drop by >99% (e.g. millions instead of billions)
- You can train on a single consumer GPU with 4-bit quantization (QLoRA)
- The result is a small adapter file (a few MB) you can swap in/out of the frozen base model

## Pipeline

```
raw dataset -> format as instruction pairs (src/prepare_data.py)
            -> load base model in 4-bit (src/train_lora.py)
            -> LoRA adapters trained on attention projections
            -> src/inference.py loads base + adapter for generation
            -> src/evaluate.py compares base vs. fine-tuned on held-out set
```

## Setup

```bash
pip install -r requirements.txt
python src/prepare_data.py --raw_file data/raw_dataset.jsonl --out_dir data/processed
python src/train_lora.py --data_dir data/processed --output_dir models/adapter
python src/inference.py --adapter_dir models/adapter --prompt "your prompt"
python src/evaluate.py --adapter_dir models/adapter --test_file data/processed/test.jsonl
```

## What to fill in yourself

- Pick a real, narrow task: domain summarization, sentiment on a specific product category, structured info extraction from a niche document type. "Fine-tune on everything" is a weak story in interviews — "fine-tuned on 3k support tickets to classify urgency" is a strong one.
- `data/raw_dataset.jsonl` — your dataset (HF Hub dataset, scraped/labeled data, or a Kaggle set)
- Base model in `config.yaml` — default assumes a ~3-7B open model; adjust to what your GPU (or Colab) can hold in 4-bit

## Results (fill in after running)

| | Base model | LoRA fine-tuned |
|---|---|---|
| Task metric (accuracy/ROUGE/etc.) | | |
| Trainable params | 0 | |
| Adapter size (MB) | — | |
