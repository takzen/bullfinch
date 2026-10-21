"""Decision engine: option-code scoring of typed questions (template ``decision-v1``).

One forward pass per question. The chat prompt (thinking disabled) is followed by the
``{"answer": "`` prefill; the logits of the allowed option codes at the next position are
softmaxed over those codes only, optionally divided by a per-question-type temperature from
a calibration file. No text is generated.

Model loading needs the ``inference`` extra; the answer-shaping helpers below do not, so they
are unit-tested without a GPU.

Copied from Serin 4B (axiomjudge); keep the two in step by hand.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from .questions import ANSWER_PREFILL, SCORE_STEPS, TEMPLATE_VERSION, TypedQuestion, render

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_POLISH = re.compile(r"[ąćęłńóśźż]", re.IGNORECASE)


def frame_for(question: TypedQuestion) -> str:
    """Frame language: Polish when the question itself is written in Polish."""

    text = question.instructions + " " + " ".join(question.criteria.values())
    return "pl" if _POLISH.search(text) else "en"


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [v / temperature for v in values]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def shape_answer(question: TypedQuestion, probs: dict[str, float]) -> dict[str, Any]:
    """Map option-key probabilities to the Jev/basal answer shape for the question type."""

    if question.type == "noul":
        return {"noul": probs["true"]}
    if question.type == "score":
        return {"score": sum(SCORE_STEPS[int(k)] * p for k, p in probs.items())}
    return {"probabilities": probs}


def parse_question(raw: dict[str, Any]) -> TypedQuestion:
    qtype = raw.get("type", "choice")
    if qtype not in ("choice", "noul", "score"):
        raise ValueError(f"unsupported question type {qtype!r}")
    criteria = raw.get("criteria") or {}
    if not isinstance(criteria, dict):
        raise ValueError("criteria must be an object")
    if qtype == "noul":
        criteria = {"true": criteria.get("true", "yes"), "false": criteria.get("false", "no")}
    if qtype == "choice" and len(criteria) < 2:
        raise ValueError("a choice question needs at least two criteria")
    return TypedQuestion(qtype, str(raw.get("instructions", "")), {str(k): str(v) for k, v in criteria.items()})


class DecisionEngine:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str = DEFAULT_REVISION,
        adapter: str | None = None,
        calibration: Path | None = None,
        max_length: int = 4096,
        tta: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.model_name, self.revision, self.adapter = model, revision, adapter
        self.max_length = max_length
        self.tta = tta
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        net = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16
        )
        if adapter:
            from peft import PeftModel

            net = PeftModel.from_pretrained(net, adapter)
        self.model = net.eval()
        self.temperatures: dict[str, float] = {}
        self.calibration_name = "uncalibrated"
        if calibration:
            data = json.loads(Path(calibration).read_text(encoding="utf-8"))
            self.temperatures = {k: float(v) for k, v in data.get("temperature", {}).items()}
            self.calibration_name = str(calibration)
        prefix_ids = self.tokenizer.encode(ANSWER_PREFILL, add_special_tokens=False)
        self.code_ids = {}
        for code in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            ids = self.tokenizer.encode(ANSWER_PREFILL + code, add_special_tokens=False)
            if ids[: len(prefix_ids)] != prefix_ids or len(ids) != len(prefix_ids) + 1:
                raise RuntimeError(f"option code {code} is not a single token after the prefill")
            self.code_ids[code] = ids[-1]

    def prompt_ids(self, state: str, question: TypedQuestion, frame: str | None = None,
                   order: list[str] | None = None):
        messages, codes = render(state, question, frame=frame or frame_for(question), order=order)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tokenizer(text + ANSWER_PREFILL, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] > self.max_length:
            raise ValueError(f"prompt is {ids.shape[1]} tokens, above max_length {self.max_length}")
        return ids, codes

    def option_logprobs(self, state: str, question: TypedQuestion, frame: str | None = None,
                        order: list[str] | None = None) -> dict[str, float]:
        """Log-probability of every option key from one forward pass."""

        ids, codes = self.prompt_ids(state, question, frame, order)
        with self.torch.inference_mode():
            logits = self.model(input_ids=ids.to(0), logits_to_keep=1).logits[0, -1].float()
        values = [float(logits[self.code_ids[c]]) for c in codes]
        probs = softmax(values, self.temperatures.get(question.type, 1.0))
        return {key: math.log(max(p, 1e-12)) for key, p in zip(codes.values(), probs)}

    def logprobs(self, state: str, question: TypedQuestion) -> dict[str, float]:
        """Option log-probabilities behind the verdict: one pass, or the --tta average."""

        if not self.tta or question.type == "score":
            return self.option_logprobs(state, question)
        # Test-time averaging: every cyclic rotation of the options in both frame
        # languages, so neither a letter's position nor the frame tips the verdict.
        keys = question.option_keys()
        runs = [self.option_logprobs(state, question, frame, keys[i:] + keys[:i])
                for frame in ("pl", "en") for i in range(len(keys))]
        return {k: sum(r[k] for r in runs) / len(runs) for k in keys}

    def decide(self, state: str, question: TypedQuestion) -> dict[str, Any]:
        logp = self.logprobs(state, question)
        probs = softmax(list(logp.values()))
        return shape_answer(question, dict(zip(logp, probs)))

    def systemone(self, body: dict[str, Any]) -> dict[str, Any]:
        """Answer a /v1/systemone request body."""

        start = time.perf_counter()
        state = str(body.get("state", ""))
        questions = body.get("questions") or {}
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        answers = {name: self.decide(state, parse_question(raw)) for name, raw in questions.items()}
        return {
            "answers": answers,
            "model": f"bullfinch/{self.adapter or 'base'}",
            "template": TEMPLATE_VERSION,
            "calibration": self.calibration_name,
            "latency_ms": round((time.perf_counter() - start) * 1000),
        }
