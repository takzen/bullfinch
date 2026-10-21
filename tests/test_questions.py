import random

import pytest

from bullfinch.questions import ANSWER_PREFILL, DecisionExample, TypedQuestion, render, variants
from bullfinch.splits import split_for

Q = TypedQuestion("choice", "Jaki wydźwięk ma ta recenzja?", {
    "positive": "pozytywny",
    "negative": "negatywny",
})


def test_render_codes_follow_the_requested_order() -> None:
    messages, codes = render("Świetny hotel.", Q, frame="pl", order=["negative", "positive"])
    assert codes == {"A": "negative", "B": "positive"}
    text = messages[0]["content"]
    assert "A) negative: negatywny" in text and "B) positive: pozytywny" in text
    assert text.index("Stan:") < text.index("Pytanie:") < text.index("Opcje:")


def test_render_rejects_a_bad_order_and_keeps_the_score_scale() -> None:
    with pytest.raises(ValueError):
        render("s", Q, order=["positive"])
    score = TypedQuestion("score", "Jak bardzo?", {"low": "wcale", "high": "całkowicie"})
    _, codes = render("s", score, frame="pl", order=["4", "3", "2", "1", "0"])
    assert codes == {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}  # never shuffled


def test_variants_point_the_answer_code_at_the_gold_option() -> None:
    ex = DecisionExample(id="e1", state="s", question=Q, answer="negative", family="polemo",
                         language="pl", source="test")
    rows = variants(ex, 6, random.Random(7))
    assert len({r["id"] for r in rows}) == 6
    for r in rows:
        assert r["codes"][r["answer_code"]] == "negative"
        assert r["prefill"] == ANSWER_PREFILL
    assert {r["frame"] for r in rows} <= {"pl", "en"}


def test_split_for_is_deterministic_and_mostly_train() -> None:
    assert split_for("polemo:123") == split_for("polemo:123")
    splits = [split_for(f"g{i}") for i in range(2000)]
    assert set(splits) == {"train", "dev", "calibration"}
    assert 0.85 < splits.count("train") / len(splits) < 0.95
