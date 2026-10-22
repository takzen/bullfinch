#!/usr/bin/env python3
"""Train Bullfinch with Unsloth's FastDecisionModel + DecisionTrainer (a head over the hidden
states, all questions in one pass) on exactly the rows ``train_qlora.py`` samples for the same
--examples, --n, --mix and --seed, so the two recipes can be compared on dev and on the panel.

    PYTHONPATH=src python scripts/train_unsloth.py --examples data/examples/*.jsonl \
        --n 8000 --mix polemo=0.14,... --out artifacts/unsloth1

* Settings follow Unsloth's recommendation: 4-bit LoRA r 16, lr 2e-4 (head 1e-4), effective
  batch 16, one pass; dev loss every --eval-every steps, the best checkpoint is kept and
  training stops after --patience evals without improvement.
* Saves ``<out>/model`` (LoRA + head); score it with ``evaluate.py --unsloth <out>/model``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bullfinch.unsloth_head import to_row  # noqa: E402
from train_qlora import load_examples, sample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", help="same base as configs/lora.yaml")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--dev-n", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batch", type=int, default=1, help="padding a batch of long dev rows is what makes evals slow")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from unsloth import DecisionTrainer, FastDecisionModel, is_bfloat16_supported  # noqa: I001
    from datasets import Dataset
    from transformers import EarlyStoppingCallback, TrainingArguments

    mix = {k: float(v) for k, v in (p.split("=") for p in args.mix.split(","))}
    print("loading examples ...")
    rows = load_examples(args.examples)
    train = sample([ex for ex, s in rows if s == "train"], args.n, mix, random.Random(args.seed))
    dev = sample([ex for ex, s in rows if s == "dev"], args.dev_n, mix, random.Random(args.seed + 1))

    model, tokenizer = FastDecisionModel.from_pretrained(args.model, revision=args.revision,
                                                       max_seq_length=args.max_tokens, load_in_4bit=True)
    model = FastDecisionModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0,
                                             use_gradient_checkpointing="unsloth", random_state=args.seed)
    train_items, rep = FastDecisionModel.build_dataset(Dataset.from_list([to_row(e) for e in train]), tokenizer, model)
    dev_items, rep_dev = FastDecisionModel.build_dataset(Dataset.from_list([to_row(e) for e in dev]), tokenizer, model)
    print(f"train {len(train_items)} (skipped {rep['skipped']}), dev {len(dev_items)} (skipped {rep_dev['skipped']})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                          "n_train": len(train_items), "n_dev": len(dev_items)},
                                                         indent=2), encoding="utf-8")
    trainer = DecisionTrainer(
        model=model, processing_class=tokenizer, train_dataset=train_items, eval_dataset=dev_items,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        args=TrainingArguments(
            per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=1,
            per_device_eval_batch_size=args.eval_batch,
            warmup_steps=max(1, round(0.03 * math.ceil(len(train_items) / 16))),  # no warmup_ratio in transformers 5
            learning_rate=args.lr, lr_scheduler_type="cosine", weight_decay=0.01,
            bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
            eval_strategy="steps", eval_steps=args.eval_every, save_strategy="steps", save_steps=args.eval_every,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, logging_steps=10, output_dir=str(args.out / "checkpoints"),
            report_to="none", seed=args.seed,
        ),
    )
    trainer.train()
    model.save_pretrained(str(args.out / "model"))
    tokenizer.save_pretrained(str(args.out / "model"))
    metrics = FastDecisionModel.evaluate(model, tokenizer, dev_items)
    print("dev:", metrics)
    (args.out / "dev_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
