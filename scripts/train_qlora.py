#!/usr/bin/env python3
"""QLoRA training of Bullfinch on typed decisions (configs/lora.yaml).

    uv run python scripts/train_qlora.py \
        --examples data/examples/polemo.jsonl data/examples/dyk.jsonl data/examples/psc.jsonl \
        --n 5000 --mix polemo=0.4,dyk=0.3,psc=0.3 \
        --out artifacts/pilot

* Inputs are DecisionExample JSONL written by the converters. Rows without a ``split`` get
  one from their group (``bullfinch.splits.split_for``).
* Every training example is rendered fresh: shuffled option order, Polish or English frame.
* Loss: negative log-likelihood of the gold option code under a softmax over the allowed
  codes only (soft targets when present), the same distribution the engine reads.
* Every ``--eval-every`` optimizer steps the dev panel is scored with a fixed option order;
  the adapter with the lowest dev NLL is kept, and training stops after ``--patience`` evals
  without improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from bullfinch.engine import frame_for
from bullfinch.questions import ANSWER_PREFILL, DecisionExample, TypedQuestion, render
from bullfinch.sources import split_for


def load_examples(paths: list[Path]) -> list[tuple[DecisionExample, str]]:
    out = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.pop("split", None) or split_for(row.get("group", row["id"]))
            row["question"] = TypedQuestion(**row["question"])
            out.append((DecisionExample(**row), split))
    return out


def family_bucket(family: str, mix: dict[str, float]) -> str | None:
    for name in mix:
        if family == name or family.startswith(name + ":"):
            return name
    return None


def sample(pool: list[DecisionExample], n: int, mix: dict[str, float], rng: random.Random) -> list[DecisionExample]:
    by_bucket: dict[str, list[DecisionExample]] = defaultdict(list)
    for ex in pool:
        bucket = family_bucket(ex.family, mix)
        if bucket:
            by_bucket[bucket].append(ex)
    chosen: list[DecisionExample] = []
    for bucket, share in mix.items():
        items = by_bucket.get(bucket, [])
        want = min(len(items), round(n * share))
        chosen += rng.sample(items, want)
        print(f"  {bucket}: {want} of {len(items)} available")
    rng.shuffle(chosen)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, nargs="+", required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/lora.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5000, help="training examples to sample")
    ap.add_argument("--mix", required=True, help="family=share,... (shares of --n)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="skip longer prompts")
    ap.add_argument("--dev-n", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50, help="optimizer steps")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mix = {k: float(v) for k, v in (p.split("=") for p in args.mix.split(","))}
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print("loading examples ...")
    rows = load_examples(args.examples)
    train_pool = [ex for ex, split in rows if split == "train"]
    dev_pool = [ex for ex, split in rows if split == "dev"]
    train = sample(train_pool, args.n, mix, rng)
    dev = sample(dev_pool, args.dev_n, mix, random.Random(args.seed + 1))

    model_name, revision = cfg["base_model"], cfg["base_revision"]
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    q = cfg["quantization"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"]),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))

    prefix = tok.encode(ANSWER_PREFILL, add_special_tokens=False)
    code_ids = {c: tok.encode(ANSWER_PREFILL + c, add_special_tokens=False)[len(prefix)] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

    def encode(ex: DecisionExample, frame: str, order: list[str] | None):
        messages, codes = render(ex.state, ex.question, frame=frame, order=order)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text + ANSWER_PREFILL, return_tensors="pt", add_special_tokens=False).input_ids
        return ids, codes

    def loss_for(ex: DecisionExample, ids, codes) -> tuple[torch.Tensor, bool]:
        logits = model(input_ids=ids.to(0), logits_to_keep=1).logits[0, -1].float()
        sel = torch.tensor([code_ids[c] for c in codes], device=logits.device)
        logp = torch.log_softmax(logits[sel], dim=0)
        keys = list(codes.values())
        if ex.soft_targets:
            target = torch.tensor([ex.soft_targets.get(k, 0.0) for k in keys], device=logits.device)
            loss = -(target * logp).sum()
        else:
            loss = -logp[keys.index(ex.answer)]
        correct = keys[int(logp.argmax())] == ex.answer
        return loss, correct

    def evaluate() -> dict:
        model.eval()
        nll, hits, per = 0.0, 0, Counter()
        with torch.inference_mode():
            for ex in dev:
                ids, codes = encode(ex, frame_for(ex.question), None)
                loss, ok = loss_for(ex, ids, codes)
                nll += float(loss)
                hits += ok
                per[(family_bucket(ex.family, mix), ok)] += 1
        model.train()
        fam = {b: round(per[(b, True)] / max(1, per[(b, True)] + per[(b, False)]), 3) for b in mix}
        return {"dev_nll": round(nll / len(dev), 4), "dev_acc": round(hits / len(dev), 4), "dev_acc_by_family": fam}

    tr = cfg["training"]
    accum = tr["gradient_accumulation_steps"]
    steps_total = math.ceil(len(train) / accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tr["learning_rate"])
    sched = get_cosine_schedule_with_warmup(opt, int(steps_total * tr["warmup_ratio"]), steps_total)
    log = (args.out / "metrics.jsonl").open("a", encoding="utf-8")
    (args.out / "run_config.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()}, "config": cfg, "n_train": len(train),
        "n_dev": len(dev), "train_families": Counter(family_bucket(e.family, mix) for e in train),
    }, indent=2, default=str), encoding="utf-8")

    base = evaluate()
    print("before training:", base)
    log.write(json.dumps({"step": 0, **base}) + "\n")
    best, bad, step, seen, skipped = base["dev_nll"], 0, 0, 0, 0
    running, t0 = [], time.perf_counter()
    model.train()
    for ex in train:
        keys = ex.question.option_keys()
        order = keys[:] if ex.question.type == "score" else rng.sample(keys, len(keys))
        ids, codes = encode(ex, rng.choice(("pl", "en")), order)
        if ids.shape[1] > args.max_tokens:
            skipped += 1
            continue
        loss, _ = loss_for(ex, ids, codes)
        (loss / accum).backward()
        running.append(float(loss))
        seen += 1
        if seen % accum:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            rate = seen / (time.perf_counter() - t0)
            print(f"step {step}/{steps_total} loss {sum(running[-accum * 10:]) / len(running[-accum * 10:]):.4f} "
                  f"({rate:.2f} ex/s, skipped {skipped})", flush=True)
        if step % args.eval_every == 0:
            m = evaluate()
            log.write(json.dumps({"step": step, "seen": seen, **m}) + "\n"); log.flush()
            print(f"eval @ {step}: {m}", flush=True)
            if m["dev_nll"] < best:
                best, bad = m["dev_nll"], 0
                model.save_pretrained(args.out / "adapter")
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stop: dev NLL stopped improving")
                    break
    final = evaluate()
    log.write(json.dumps({"step": step, "seen": seen, "final": True, **final}) + "\n")
    if final["dev_nll"] < best or not (args.out / "adapter").exists():
        model.save_pretrained(args.out / "adapter")
    print("final:", final, f"| best dev NLL {min(best, final['dev_nll'])} | skipped {skipped} long prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
