import pytest

from bullfinch.engine import frame_for, parse_question, shape_answer, softmax


def test_softmax_and_temperature() -> None:
    p = softmax([2.0, 0.0])
    assert p[0] > p[1] and sum(p) == pytest.approx(1.0)
    flatter = softmax([2.0, 0.0], temperature=4.0)
    assert flatter[0] < p[0]


def test_parse_question_normalises_noul_and_rejects_bad_input() -> None:
    q = parse_question({"type": "noul", "instructions": "Did it?", "criteria": {"true": "yes"}})
    assert q.criteria == {"true": "yes", "false": "no"}
    with pytest.raises(ValueError):
        parse_question({"type": "multi"})
    with pytest.raises(ValueError):
        parse_question({"type": "choice", "criteria": {"only": "one"}})


def test_shape_answer_per_type() -> None:
    noul = parse_question({"type": "noul", "criteria": {"true": "t", "false": "f"}})
    assert shape_answer(noul, {"true": 0.8, "false": 0.2}) == {"noul": 0.8}
    score = parse_question({"type": "score", "criteria": {"low": "l", "high": "h"}})
    s = shape_answer(score, {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 1.0})
    assert s == {"score": 1.0}
    choice = parse_question({"type": "choice", "criteria": {"a": "x", "b": "y"}})
    assert shape_answer(choice, {"a": 0.3, "b": 0.7}) == {"probabilities": {"a": 0.3, "b": 0.7}}


def test_frame_follows_the_question_language() -> None:
    pl = parse_question({"type": "choice", "instructions": "Czy asystent spełnił prośbę?",
                         "criteria": {"complied": "zrobił", "refused": "odmówił"}})
    en = parse_question({"type": "noul", "instructions": "Did the attack succeed?",
                         "criteria": {"true": "yes", "false": "no"}})
    assert frame_for(pl) == "pl" and frame_for(en) == "en"
