"""Bullfinch decisions in the format of Unsloth's ``FastDecisionModel`` (a trained head over the
hidden states instead of option-letter logits).

* ``to_row`` turns a ``DecisionExample`` into one typed-decisions row (state, questions, gold as
  JSON strings, one question named ``q``) for ``FastDecisionModel.build_dataset``.
* ``UnslothEngine`` scores a saved model with the same ``logprobs`` interface as
  ``DecisionEngine``, so ``scripts/evaluate.py --unsloth DIR`` puts it on the same panel.
"""

from __future__ import annotations

import json
import math

from .questions import SCORE_STEPS, DecisionExample, TypedQuestion


def question_json(q: TypedQuestion) -> dict:
    """Our question as Unsloth expects it; a score lists one label per step, low to high."""
    if q.type == "score":
        low, high = q.criteria.get("low", ""), q.criteria.get("high", "")
        steps = len(SCORE_STEPS)
        labels = [f"{i + 1}" for i in range(steps)]
        labels[0], labels[-1] = f"1: {low}", f"{steps}: {high}"
        return {"type": "score", "instructions": q.instructions, "criteria": labels}
    criteria = dict(q.criteria) if q.type == "choice" else {
        "true": q.criteria.get("true", "yes"), "false": q.criteria.get("false", "no")}
    return {"type": q.type, "instructions": q.instructions, "criteria": criteria}


def to_row(ex: DecisionExample) -> dict:
    keys = ex.question.option_keys()
    probs = dict(ex.soft_targets) if ex.soft_targets else {k: float(k == ex.answer) for k in keys}
    probs = {k: probs.get(k, 0.0) for k in keys}
    gold = {"type": ex.question.type, "label": max(probs, key=probs.get), "probabilities": probs}
    if ex.question.type == "noul":
        gold["noul"] = probs["true"]
    return {"id": ex.id, "state": ex.state, "questions": json.dumps({"q": question_json(ex.question)}, ensure_ascii=False),
            "gold": json.dumps({"q": gold}, ensure_ascii=False)}


class UnslothEngine:
    def __init__(self, path: str, max_length: int = 4096) -> None:
        from unsloth import FastDecisionModel

        self._fdm = FastDecisionModel
        self.model, self.tokenizer = FastDecisionModel.from_pretrained(path, max_seq_length=max_length,
                                                                       load_in_4bit=True)
        FastDecisionModel.for_inference(self.model)
        self.model_name, self.revision, self.adapter, self.tta = path, "unsloth", path, False

    def logprobs(self, state: str, question: TypedQuestion) -> dict[str, float]:
        answer = self._fdm.predict(self.model, self.tokenizer, state, {"q": question_json(question)})["q"]
        raw = {str(k).lower(): float(v) for k, v in answer["probabilities"].items()}
        probs = {k: raw.get(k, 0.0) for k in question.option_keys()}
        total = sum(probs.values()) or 1.0
        return {k: math.log(max(v / total, 1e-12)) for k, v in probs.items()}
