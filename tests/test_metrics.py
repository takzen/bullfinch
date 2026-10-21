import pytest

from bullfinch.metrics import summarise
from bullfinch.sources import SENTIMENT, canonical_label


def _row(gold: str, pred: str, p_top: float, instructions: str = "q") -> dict:
    return {"gold": gold, "pred": pred, "p_gold": p_top if gold == pred else 1 - p_top, "p_top": p_top,
            "instructions": instructions}


def test_summarise_counts_f1_true_for_yes_no_tasks() -> None:
    rows = [_row("true", "true", 0.9), _row("true", "false", 0.6), _row("false", "false", 0.8),
            _row("false", "false", 0.7)]
    m = summarise(rows)
    assert m["accuracy"] == 0.75 and m["f1_true"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2, abs=1e-4)


def test_canonical_label_pools_phrasings() -> None:
    a, b = SENTIMENT[0], SENTIMENT[1]
    assert canonical_label(a.question(), a.key("amb")) == canonical_label(b.question(), b.key("amb")) == "amb"
    assert summarise([_row("amb", "zero", 0.5, a.instructions)])["accuracy_by_phrasing"] == {a.instructions: 0.0}
