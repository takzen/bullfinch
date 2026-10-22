from bullfinch.engine import frame_for
from bullfinch.sources import ANSWERS_QUESTION, SAME_ARTICLE, SENTIMENT, from_dyk, from_polemo_in, from_psc


def test_polish_phrasings_get_the_polish_frame() -> None:
    for phrasings in (SENTIMENT, ANSWERS_QUESTION, SAME_ARTICLE):
        frames = [frame_for(p.question()) for p in phrasings]
        assert frames[:-1] == ["pl"] * (len(phrasings) - 1) and frames[-1] == "en"
        assert all(p.options.keys() == phrasings[0].options.keys() for p in phrasings)


def test_polemo_maps_every_label_and_skips_unknown_ones() -> None:
    labels = {}
    for i, target in enumerate(("minus_m", "plus_m", "amb", "zero")):
        (ex,) = from_polemo_in({"sentence": " Dobry lekarz . ", "target": f"__label__meta_{target}"}, i, "train")
        labels[target] = ex.question.criteria[ex.answer]
        assert ex.state == "Dobry lekarz ." and ex.answer in ex.question.option_keys()
    assert len(set(labels.values())) == 4
    assert list(from_polemo_in({"sentence": "x", "target": "__label__other"}, 0, "train")) == []


def test_dyk_and_psc_turn_labels_into_noul_answers() -> None:
    (yes,) = from_dyk({"q_id": "c1", "question": "kto?", "answer": "On.", "target": 1}, 0, "train")
    (no,) = from_dyk({"q_id": "c1", "question": "kto?", "answer": "Inny.", "target": 0}, 1, "train")
    assert (yes.answer, no.answer) == ("true", "false") and yes.group == no.group == "dyk:c1"
    (ex,) = from_psc({"extract_text": "A.", "summary_text": "B.", "label": 1}, 0, "test")
    assert ex.answer == "true" and "Streszczenie A: A." in ex.state
    assert list(from_psc({"extract_text": "A.", "summary_text": "", "label": 1}, 0, "test")) == []


def test_eval_only_converters() -> None:
    from bullfinch.sources import from_belebele, from_include, from_opendecisions

    row = {"flores_passage": "Tekst.", "question": "Co?", "correct_answer_num": "3",
           **{f"mc_answer{i}": f"o{i}" for i in range(1, 5)}}
    (b,) = from_belebele(row, 0, "test")
    assert b.answer == "3" and b.question.criteria["3"] == "o3"
    (inc,) = from_include({"question": "Pytanie?", "answer": 2, **{f"option_{k}": k for k in "abcd"}}, 0, "test")
    assert inc.answer == "c"
    base = {"id": "x", "language": "pl", "state": {"a": "żółw"}}
    (noul,) = from_opendecisions({**base, "expected": "yes", "question": {
        "type": "noul", "instructions": "Czy?", "criteria": {"true": "t", "false": "f"}}}, 0, "test")
    assert noul.answer == "true" and "żółw" in noul.state
    (score,) = from_opendecisions({**base, "expected": 1, "question": {
        "type": "score", "instructions": "Oceń.", "criteria": ["niski", "średni", "wysoki"]}}, 0, "test")
    assert score.question.type == "choice" and score.answer == "1"
    assert list(from_opendecisions({**base, "language": "en"}, 0, "test")) == []
