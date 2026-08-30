"""
Build instruction-formatted train/val/test splits for the text-to-JSON
extraction LoRA fine-tuning task.

Task: given a messy natural-language sentence describing an order
("John ordered 2 pizzas for Rs.450 on March 3rd, 2024"), extract the
structured facts as JSON: {"customer", "item", "quantity", "amount", "date"}.

Data is fully synthetic (generated here, not downloaded) — we control a
"ground truth" order record, render it into varied natural-language
phrasing, and know the exact correct JSON output because we built it
ourselves. This sidesteps dataset licensing/cleaning entirely and makes
before/after eval unambiguous (JSON either matches the ground truth or
it doesn't).

Reformats into the same ### Instruction / ### Input / ### Response
template used across this portfolio, which train_lora.py trains the
causal LM on directly.
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

DEFAULT_INSTRUCTION = (
    "Extract the order details from the text as JSON with exactly these keys: "
    "customer, item, quantity, amount, date (format: YYYY-MM-DD)."
)

CUSTOMERS = [
    "John", "Priya", "Wei", "Fatima", "Carlos", "Aisha", "Liam", "Sana",
    "Diego", "Emma", "Ravi", "Zara", "Noah", "Meera", "Yusuf", "Lucia",
]

ITEMS = {
    "pizza": 225, "burger": 150, "coffee": 90, "notebook": 60,
    "t-shirt": 350, "backpack": 900, "headphones": 1200, "book": 280,
    "candle": 175, "mug": 120,
}

CURRENCY_SYMBOLS = ["Rs.", "₹", "$"]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

SENTENCE_TEMPLATES = [
    "{customer} ordered {quantity} {item}{plural} for {currency}{amount} on {date_phrase}",
    "{customer} bought {quantity} {item}{plural} on {date_phrase} for {currency}{amount}",
    "On {date_phrase}, {customer} purchased {quantity} {item}{plural} worth {currency}{amount}",
    "{customer} placed an order for {quantity} {item}{plural} costing {currency}{amount}, dated {date_phrase}",
    "{quantity} {item}{plural} were ordered by {customer} on {date_phrase} totaling {currency}{amount}",
]


def random_date(rng: random.Random):
    month_idx = rng.randint(1, 12)
    day = rng.randint(1, 28)
    year = rng.choice([2023, 2024])
    return year, month_idx, day


def date_phrase(rng: random.Random, year: int, month_idx: int, day: int) -> str:
    month_name = MONTHS[month_idx - 1]
    style = rng.choice(["month_day_year", "day_month", "numeric", "ordinal"])
    if style == "month_day_year":
        return f"{month_name} {day}, {year}"
    if style == "day_month":
        return f"{day} {month_name}"
    if style == "numeric":
        return f"{month_idx:02d}/{day:02d}/{year}"
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{month_name} {day}{suffix}"


def make_example(rng: random.Random) -> dict:
    customer = rng.choice(CUSTOMERS)
    item = rng.choice(list(ITEMS.keys()))
    quantity = rng.randint(1, 5)
    amount = ITEMS[item] * quantity
    year, month_idx, day = random_date(rng)

    template = rng.choice(SENTENCE_TEMPLATES)
    sentence = template.format(
        customer=customer,
        quantity=quantity,
        item=item,
        plural="s" if quantity > 1 else "",
        currency=rng.choice(CURRENCY_SYMBOLS),
        amount=amount,
        date_phrase=date_phrase(rng, year, month_idx, day),
    )

    output_json = {
        "customer": customer,
        "item": item,
        "quantity": quantity,
        "amount": amount,
        "date": f"{year:04d}-{month_idx:02d}-{day:02d}",
    }
    return {"sentence": sentence, "output_json": output_json}


def format_example(instruction: str, example: dict) -> dict:
    output_str = json.dumps(example["output_json"])
    text = PROMPT_TEMPLATE.format(instruction=instruction, input=example["sentence"], output=output_str)
    return {"text": text, "output": output_str}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n_examples", type=int, default=500, help="Total examples across train+val+test")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    examples = [make_example(rng) for _ in range(args.n_examples)]

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
                f.write(json.dumps(format_example(args.instruction, ex)) + "\n")
        print(f"{split_name}: {len(split_examples)} examples -> {path}")


if __name__ == "__main__":
    main()
