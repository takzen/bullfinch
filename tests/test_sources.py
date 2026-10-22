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


def test_massive_asks_scenario_then_intent_within_it() -> None:
    from bullfinch.sources import INTENTS, SCENARIOS, from_massive

    assert all(k.split("_")[0] in SCENARIOS for k in INTENTS) and len(INTENTS) == 60
    out = list(from_massive({"id": "7", "utt": "obudź mnie o siódmej", "scenario_name": "alarm",
                             "intent_name": "alarm_set"}, 0, "train"))
    assert [e.family for e in out] == ["massive:scenario", "massive:intent"]
    assert out[1].question.option_keys() == ["alarm_query", "alarm_remove", "alarm_set"]
    assert out[0].group == out[1].group == "massive:7"
    only = list(from_massive({"id": "8", "utt": "jaka pogoda", "scenario_name": "weather",
                              "intent_name": "weather_query"}, 1, "test"))
    assert [e.family for e in only] == ["massive:scenario"]


def test_polqa_and_exams() -> None:
    from bullfinch.sources import SOURCES, from_exams, from_polqa, overlap_key

    (p,) = from_polqa({"question_id": 3, "question": "Kto?", "passage_title": "Alfa", "passage_text": "Tekst.",
                       "relevant": True}, 0, "train")
    assert p.answer == "true" and p.state.endswith("Fragment (Alfa): Tekst.") and p.group == "polqa:3"
    row = {"id": "e1", "answerKey": "B", "question": {"stem": "Ile?", "choices": {
        "label": ["A", "B", "C", "D"], "text": ["1", "2", "3", "4"]}}}
    (e,) = from_exams(row, 0, "train")
    assert e.answer == "b" and e.question.criteria["b"] == "2"
    assert SOURCES["exams"].pool == SOURCES["include"].pool
    assert SOURCES["exams"].texts(row) == ["Ile?"]
    assert overlap_key("Ile, to jest?") == overlap_key("ile to  jest")


def test_pllum_pairs_and_ratings() -> None:
    from bullfinch.sources import from_pllum

    user = {"role": "user", "content": "Ile to 2+2?"}
    row = {"id": "q_1", "chosen": [user, {"role": "assistant", "content": "4"}],
           "rejected": [user, {"role": "assistant", "content": "5"}],
           "chosen_rating_details": {"correctness": 5, "helpfulness": 4, "safety": 5},
           "rejected_rating_details": {"correctness": 1, "helpfulness": 2}}
    out = {e.family: e for e in from_pllum(row, 0, "train") if e.id.endswith(("pair", "chosen-correctness"))}
    pair = out["pllum:pair"]
    good_first = pair.state.index("Odpowiedź 1:\n4") > 0 if "Odpowiedź 1:\n4" in pair.state else False
    assert pair.meta["label"] == ("first" if good_first else "second")
    assert out["pllum:correctness"].answer == "4" and out["pllum:correctness"].question.type == "score"
    assert len(list(from_pllum(row, 0, "train"))) == 5
    same = {**row, "rejected": row["chosen"]}
    assert list(from_pllum(same, 0, "train")) == []
