"""Typed questions and their prompt rendering (template ``decision-v1``).

A *decision example* is a state, one typed question (Jev / basal contract) and the gold
answer key. Rendering turns it into chat messages whose options carry single-letter codes,
plus the code map needed to read the answer: the model's assistant turn is prefilled with
``{"answer": "`` and the decision is a softmax over the option-code logits that follow.

Robustness to phrasing is a training-data property, not an inference trick: the same record
is rendered with several instruction phrasings, frame languages and option orders
(:func:`variants`). Evaluation and serving render with a fixed order.

Copied from Serin 4B (axiomjudge); keep the two in step by hand, the template is shared.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Literal

TEMPLATE_VERSION = "decision-v1"
ANSWER_PREFILL = '{"answer": "'
CODES = string.ascii_uppercase
QuestionType = Literal["choice", "noul", "score"]

# 5-point scale for `score`, low to high; the expected value over it is mapped to [0, 1].
SCORE_STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)
_SCORE_LABELS = {
    "pl": ("bardzo nisko", "nisko", "pośrodku", "wysoko", "bardzo wysoko"),
    "en": ("very low", "low", "middle", "high", "very high"),
}

_FRAME = {
    "pl": {
        "state": "Stan",
        "question": "Pytanie",
        "options": "Opcje",
        "reply": 'Odpowiedz wyłącznie w formacie JSON: {"answer": "<litera opcji>"}.',
        "scale": "Skala od {low} do {high}",
    },
    "en": {
        "state": "State",
        "question": "Question",
        "options": "Options",
        "reply": 'Reply only with JSON: {"answer": "<option letter>"}.',
        "scale": "Scale from {low} to {high}",
    },
}


@dataclass(frozen=True)
class TypedQuestion:
    type: QuestionType
    instructions: str
    criteria: dict[str, str]

    def option_keys(self) -> list[str]:
        if self.type == "noul":
            return ["true", "false"]
        if self.type == "score":
            return [str(i) for i in range(len(SCORE_STEPS))]
        return list(self.criteria)


@dataclass(frozen=True)
class DecisionExample:
    """One supervised decision. `answer` is an option key (or a score step index)."""

    id: str
    state: str
    question: TypedQuestion
    answer: str
    family: str
    language: str  # language of the state text
    source: str
    provenance: str = "external"
    group: str = ""  # dedup / split unit (same group never crosses splits)
    soft_targets: dict[str, float] | None = None
    meta: dict[str, str] = field(default_factory=dict)


def _option_lines(question: TypedQuestion, keys: list[str], frame: str) -> list[str]:
    if question.type == "score":
        low, high = question.criteria.get("low", ""), question.criteria.get("high", "")
        labels = _SCORE_LABELS[frame]
        lines = [f"{labels[int(k)]}" for k in keys]
        lines[0] += f" ({low})" if low else ""
        lines[-1] += f" ({high})" if high else ""
        return lines
    return [f"{k}: {question.criteria[k]}" if question.type == "choice" else question.criteria[k] for k in keys]


def render(
    state: str,
    question: TypedQuestion,
    *,
    frame: str = "pl",
    order: list[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Return (chat messages, {code: option_key}) for one question.

    `order` is the option-key order to display (default: the question's own order). The score
    scale is never shuffled, its order carries meaning.
    """

    if frame not in _FRAME:
        raise ValueError(f"unsupported frame language {frame!r}")
    keys = question.option_keys()
    if order is not None and question.type != "score":
        if sorted(order) != sorted(keys):
            raise ValueError("order must be a permutation of the option keys")
        keys = list(order)
    if not 2 <= len(keys) <= len(CODES):
        raise ValueError(f"a question needs 2..{len(CODES)} options, got {len(keys)}")
    f = _FRAME[frame]
    codes = dict(zip(CODES, keys))
    lines = _option_lines(question, keys, frame)
    options = "\n".join(f"{code}) {line}" for code, line in zip(codes, lines))
    content = (
        f"{f['state']}:\n{state}\n\n"
        f"{f['question']}: {question.instructions}\n\n"
        f"{f['options']}:\n{options}\n\n"
        f"{f['reply']}"
    )
    return [{"role": "user", "content": content}], codes


def variants(example: DecisionExample, n: int, rng: random.Random) -> list[dict]:
    """Render `n` training variants: random frame language and shuffled option order.

    Instruction phrasing variety comes from the converters, which emit several questions per
    record; this adds frame and order variety on top. Returns JSON-ready rows.
    """

    rows = []
    keys = example.question.option_keys()
    for i in range(n):
        frame = rng.choice(("pl", "en"))
        order = keys[:]
        if example.question.type != "score":
            rng.shuffle(order)
        messages, codes = render(example.state, example.question, frame=frame, order=order)
        answer_code = next(code for code, key in codes.items() if key == example.answer)
        rows.append(
            {
                "id": f"{example.id}~v{i}",
                "messages": messages,
                "prefill": ANSWER_PREFILL,
                "codes": codes,
                "answer_code": answer_code,
                "soft_targets": example.soft_targets,
                "family": example.family,
                "language": example.language,
                "frame": frame,
                "source": example.source,
                "provenance": example.provenance,
                "group": example.group,
                "template": TEMPLATE_VERSION,
            }
        )
    return rows
