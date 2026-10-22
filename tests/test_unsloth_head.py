import json

from bullfinch.questions import DecisionExample, TypedQuestion
from bullfinch.unsloth_head import to_row


def ex(question: TypedQuestion, answer: str, **kw) -> DecisionExample:
    return DecisionExample(id="x-1", state="Tekst.", question=question, answer=answer, family="t", language="pl",
                           source="test", provenance="external", group="g", **kw)


def test_choice_and_noul_rows_carry_one_hot_gold() -> None:
    row = to_row(ex(TypedQuestion("choice", "Który?", {"a": "pierwszy", "b": "drugi"}), "b"))
    q, gold = json.loads(row["questions"])["q"], json.loads(row["gold"])["q"]
    assert q["criteria"] == {"a": "pierwszy", "b": "drugi"}
    assert gold["label"] == "b" and gold["probabilities"] == {"a": 0.0, "b": 1.0}
    gold = json.loads(to_row(ex(TypedQuestion("noul", "Czy?", {"true": "tak", "false": "nie"}), "true"))["gold"])["q"]
    assert gold["label"] == "true" and gold["noul"] == 1.0


def test_score_lists_one_label_per_step_and_keeps_soft_targets() -> None:
    q = TypedQuestion("score", "Oceń.", {"low": "źle", "high": "dobrze"})
    row = to_row(ex(q, "3", soft_targets={"2": 0.25, "3": 0.75}))
    crit = json.loads(row["questions"])["q"]["criteria"]
    gold = json.loads(row["gold"])["q"]
    assert crit[0] == "1: źle" and crit[-1] == "5: dobrze" and len(crit) == 5
    assert gold["label"] == "3" and gold["probabilities"]["0"] == 0.0
