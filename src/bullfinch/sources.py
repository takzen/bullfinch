"""Convert labelled Polish datasets into typed decision examples.

Each converter takes one raw row (a dict, as yielded by ``datasets``) and returns zero or
more :class:`DecisionExample` objects. Rows without a usable label yield nothing; we never
guess a label.

Every task has several phrasings, mostly Polish with one English, each with its own option
keys and descriptions. A row gets one phrasing, chosen deterministically from its id, so the
corpus as a whole covers all of them and the model cannot lean on one wording or key name.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .questions import DecisionExample, QuestionType, TypedQuestion


@dataclass(frozen=True)
class Phrasing:
    """One wording of a task: instructions plus {label: (option key, description)}."""

    type: QuestionType
    instructions: str
    options: dict[str, tuple[str, str]]

    def question(self) -> TypedQuestion:
        return TypedQuestion(self.type, self.instructions, dict(self.options.values()))

    def key(self, label: str) -> str:
        return self.options[label][0]


def _noul(instructions: str, yes: str, no: str) -> Phrasing:
    return Phrasing("noul", instructions, {"true": ("true", yes), "false": ("false", no)})


# ---------------------------------------------------------------- question phrasings

SENTIMENT = [
    Phrasing("choice", "Jaki wydźwięk ma ta opinia?", {
        "minus_m": ("negatywny", "autor ocenia źle, przeważają wady"),
        "plus_m": ("pozytywny", "autor ocenia dobrze, przeważają zalety"),
        "amb": ("mieszany", "są wyraźne zalety i wyraźne wady"),
        "zero": ("neutralny", "tekst nie wyraża oceny, tylko informuje"),
    }),
    Phrasing("choice", "Jak autor tekstu ocenia to, o czym pisze?", {
        "minus_m": ("źle", "jest niezadowolony"),
        "plus_m": ("dobrze", "jest zadowolony"),
        "amb": ("różnie", "chwali jedno, krytykuje drugie"),
        "zero": ("bez oceny", "nie wyraża opinii"),
    }),
    Phrasing("choice", "Określ sentyment recenzji.", {
        "minus_m": ("negatywny", "negatywny"),
        "plus_m": ("pozytywny", "pozytywny"),
        "amb": ("ambiwalentny", "ambiwalentny: i plusy, i minusy"),
        "zero": ("neutralny", "neutralny lub brak opinii"),
    }),
    Phrasing("choice", "Czy ta wypowiedź jest pochwałą, krytyką, czymś pomiędzy, czy nie zawiera oceny?", {
        "minus_m": ("krytyka", "przeważa krytyka"),
        "plus_m": ("pochwała", "przeważa pochwała"),
        "amb": ("pomiędzy", "pochwała i krytyka są wyważone"),
        "zero": ("brak oceny", "tekst jest rzeczowy, bez oceny"),
    }),
    Phrasing("choice", "What is the sentiment of this Polish review?", {
        "minus_m": ("negative", "mostly negative"),
        "plus_m": ("positive", "mostly positive"),
        "amb": ("mixed", "clear positives and clear negatives"),
        "zero": ("neutral", "no opinion, purely informative"),
    }),
]

ANSWERS_QUESTION = [
    _noul("Czy fragment odpowiada na pytanie?",
          "tak, fragment zawiera odpowiedź", "nie, fragment nie odpowiada na pytanie"),
    _noul("Czy na podstawie tego fragmentu można odpowiedzieć na zadane pytanie?",
          "tak, można", "nie, nie można"),
    _noul("Czy ten fragment jest trafną odpowiedzią na pytanie użytkownika?",
          "tak, jest trafny", "nie, jest nietrafiony lub nie na temat"),
    _noul("Czy w tym fragmencie jest odpowiedź na pytanie?",
          "tak, jest", "nie, nie ma jej"),
    _noul("Does the passage answer the question?",
          "yes, it contains the answer", "no, it does not answer it"),
]

SAME_ARTICLE = [
    _noul("Czy oba streszczenia dotyczą tego samego artykułu?",
          "tak, to ten sam artykuł", "nie, to różne artykuły"),
    _noul("Czy te dwa teksty streszczają ten sam artykuł prasowy?",
          "tak, ten sam", "nie, różne"),
    _noul("Czy streszczenie B dotyczy tego samego tekstu źródłowego co streszczenie A?",
          "tak, tego samego", "nie, innego"),
    _noul("Czy obydwa streszczenia mówią o tych samych wydarzeniach z tego samego źródła?",
          "tak", "nie"),
    _noul("Do both summaries describe the same news article?",
          "yes, the same article", "no, different articles"),
]


PHRASINGS = {p.instructions: p for p in SENTIMENT + ANSWERS_QUESTION + SAME_ARTICLE}


def canonical_label(question: TypedQuestion, key: str) -> str:
    """Source label behind an option key, so metrics pool every phrasing of a task."""

    p = PHRASINGS.get(question.instructions)
    if p is None:
        return key
    return next(label for label, (k, _) in p.options.items() if k == key)


def _pick(options: list[Phrasing], row_id: str) -> Phrasing:
    digest = hashlib.sha256(row_id.encode()).digest()
    return options[digest[0] % len(options)]


def text_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()[:16]


def overlap_key(text: str) -> str:
    """Looser than text_key: case, spacing and punctuation do not hide a repeated text."""

    return hashlib.sha256(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()[:16]


def _ok(text: object) -> bool:
    return isinstance(text, str) and bool(text.strip())


# ------------------------------------------------------------------------- converters


def _polemo(row: dict, idx: int, split: str, name: str, family: str) -> Iterable[DecisionExample]:
    text, label = row.get("sentence"), str(row.get("target", "")).removeprefix("__label__meta_")
    if not _ok(text) or label not in SENTIMENT[0].options:
        return
    rid = f"{name}-{split}-{idx:05d}"
    p = _pick(SENTIMENT, rid)
    yield DecisionExample(
        id=rid, state=text.strip(), question=p.question(), answer=p.key(label), family=family,
        language="pl", source=f"allegro/klej-{name}", group=f"{name}:{text_key(text)}",
        meta={"label": label},
    )


def from_polemo_in(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """allegro/klej-polemo2-in: review sentiment (medicine, hotels), 4 options."""

    return _polemo(row, idx, split, "polemo2-in", "polemo")


def from_polemo_out(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """allegro/klej-polemo2-out: the same task in other domains (evaluation only)."""

    return _polemo(row, idx, split, "polemo2-out", "polemo:out")


def from_dyk(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """allegro/klej-dyk: does a Wikipedia sentence answer a "Did you know" question."""

    q, a, target = row.get("question"), row.get("answer"), row.get("target")
    if not (_ok(q) and _ok(a)) or target not in (0, 1):
        return
    rid = f"dyk-{split}-{idx:05d}"
    p = _pick(ANSWERS_QUESTION, rid)
    yield DecisionExample(
        id=rid, state=f"Pytanie użytkownika: {q.strip()}\n\nFragment: {a.strip()}",
        question=p.question(), answer=p.key("true" if target == 1 else "false"), family="dyk",
        language="pl", source="allegro/klej-dyk", group=f"dyk:{row.get('q_id') or text_key(q)}",
        meta={"label": str(target)},
    )


def from_psc(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """allegro/klej-psc: do an extractive and an abstractive summary share an article."""

    a, b, label = row.get("extract_text"), row.get("summary_text"), row.get("label")
    if not (_ok(a) and _ok(b)) or str(label) not in ("0", "1"):
        return
    rid = f"psc-{split}-{idx:05d}"
    p = _pick(SAME_ARTICLE, rid)
    yield DecisionExample(
        id=rid, state=f"Streszczenie A: {a.strip()}\n\nStreszczenie B: {b.strip()}",
        question=p.question(), answer=p.key("true" if str(label) == "1" else "false"), family="psc",
        language="pl", source="allegro/klej-psc", group=f"psc:{text_key(b)}",
        meta={"label": str(label)},
    )


SCENARIOS = {
    "alarm": "budzik: ustawianie, sprawdzanie, usuwanie alarmów",
    "audio": "głośność: głośniej, ciszej, wyciszenie",
    "calendar": "kalendarz: wydarzenia, spotkania, przypomnienia",
    "cooking": "gotowanie: przepisy i pytania kulinarne",
    "datetime": "data i czas: która godzina, jaki dzień, strefy czasowe",
    "email": "poczta e-mail i kontakty",
    "general": "rozmowa ogólna: powitania, żarty, luźne pytania",
    "iot": "inteligentny dom: światła, gniazdka, odkurzacz, ekspres do kawy",
    "lists": "listy: tworzenie, dodawanie, usuwanie, sprawdzanie",
    "music": "preferencje i informacje o muzyce: co gra, lubię / nie lubię, tryb odtwarzania",
    "news": "wiadomości i serwisy informacyjne",
    "play": "włączenie odtwarzania: muzyka, radio, podcast, audiobook, gra",
    "qa": "pytania o wiedzę: fakty, definicje, obliczenia, kursy walut, giełda",
    "recommendation": "polecenia: filmy, miejsca, wydarzenia",
    "social": "media społecznościowe: posty, sprawdzanie, skargi do firm",
    "takeaway": "jedzenie na wynos: zamówienia i pytania do lokali",
    "transport": "transport: bilety, połączenia, taksówki, korki",
    "weather": "pogoda",
}

INTENTS = {
    "alarm_query": "sprawdzenie ustawionych alarmów", "alarm_remove": "usunięcie alarmu",
    "alarm_set": "ustawienie alarmu lub budzika",
    "audio_volume_down": "ściszenie", "audio_volume_mute": "wyciszenie",
    "audio_volume_other": "inna zmiana ustawień dźwięku", "audio_volume_up": "pogłośnienie",
    "calendar_query": "sprawdzenie kalendarza", "calendar_remove": "usunięcie wydarzenia z kalendarza",
    "calendar_set": "dodanie wydarzenia lub przypomnienia",
    "cooking_query": "pytanie kulinarne, np. jak długo gotować", "cooking_recipe": "prośba o przepis",
    "datetime_convert": "przeliczenie czasu między strefami", "datetime_query": "pytanie o datę lub godzinę",
    "email_addcontact": "dodanie kontaktu", "email_query": "sprawdzenie poczty",
    "email_querycontact": "pytanie o dane kontaktu", "email_sendemail": "wysłanie e-maila",
    "general_greet": "powitanie, grzecznościowa pogawędka", "general_joke": "prośba o żart",
    "general_quirky": "dziwne lub luźne pytanie spoza innych kategorii",
    "iot_cleaning": "uruchomienie odkurzacza lub sprzątania", "iot_coffee": "zrobienie kawy",
    "iot_hue_lightchange": "zmiana koloru światła", "iot_hue_lightdim": "przyciemnienie światła",
    "iot_hue_lightoff": "wyłączenie światła", "iot_hue_lighton": "włączenie światła",
    "iot_hue_lightup": "rozjaśnienie światła", "iot_wemo_off": "wyłączenie gniazdka lub urządzenia",
    "iot_wemo_on": "włączenie gniazdka lub urządzenia",
    "lists_createoradd": "utworzenie listy lub dodanie pozycji", "lists_query": "sprawdzenie listy",
    "lists_remove": "usunięcie listy lub pozycji",
    "music_dislikeness": "utwór się nie podoba", "music_likeness": "utwór się podoba, zapamiętaj gust",
    "music_query": "pytanie o grany utwór lub wykonawcę", "music_settings": "tryb odtwarzania: losowo, powtarzanie",
    "news_query": "pytanie o wiadomości",
    "play_audiobook": "włączenie audiobooka", "play_game": "włączenie gry", "play_music": "włączenie muzyki",
    "play_podcasts": "włączenie podcastu", "play_radio": "włączenie radia",
    "qa_currency": "kurs walut", "qa_definition": "definicja słowa lub pojęcia", "qa_factoid": "pytanie o fakt",
    "qa_maths": "obliczenie matematyczne", "qa_stock": "notowania giełdowe",
    "recommendation_events": "polecenie wydarzeń", "recommendation_locations": "polecenie miejsc, np. restauracji",
    "recommendation_movies": "polecenie filmów",
    "social_post": "publikacja posta lub skarga do firmy w mediach", "social_query": "sprawdzenie mediów społecznościowych",
    "takeaway_order": "złożenie zamówienia na wynos", "takeaway_query": "pytanie o zamówienie lub ofertę lokalu",
    "transport_query": "pytanie o połączenie lub rozkład", "transport_taxi": "zamówienie taksówki",
    "transport_ticket": "rezerwacja biletu", "transport_traffic": "informacja o korkach",
    "weather_query": "pytanie o pogodę",
}

SCENARIO_INSTRUCTIONS = [
    "Do jakiego obszaru należy to polecenie dla asystenta głosowego?",
    "Której usługi asystenta dotyczy ta wypowiedź użytkownika?",
    "Przypisz polecenie do właściwej kategorii.",
    "Który moduł asystenta powinien obsłużyć tę prośbę?",
    "Which assistant domain does this Polish voice command belong to?",
]

INTENT_INSTRUCTIONS = [
    "Czego dokładnie chce użytkownik?",
    "Jaką intencję ma to polecenie?",
    "Którą akcję asystent powinien wykonać?",
    "Rozpoznaj intencję użytkownika w tej wypowiedzi.",
    "What exactly does the user want the assistant to do?",
]


def _pick_text(options: list[str], row_id: str) -> str:
    return options[hashlib.sha256(row_id.encode()).digest()[0] % len(options)]


def from_massive(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """AmazonScience/massive pl-PL: voice-assistant scenario (18 options), then the intent
    among that scenario's intents (60 do not fit the 26 option letters)."""

    utt, scenario, intent = row.get("utt"), row.get("scenario_name"), row.get("intent_name")
    if not _ok(utt) or scenario not in SCENARIOS or intent not in INTENTS:
        return
    rid = f"massive-{split}-{idx:05d}"
    base = dict(state=f"Polecenie: {utt.strip()}", language="pl", source="AmazonScience/massive",
                group=f"massive:{row.get('id') or text_key(utt)}")
    yield DecisionExample(
        id=f"{rid}-scenario", question=TypedQuestion("choice", _pick_text(SCENARIO_INSTRUCTIONS, rid), dict(SCENARIOS)),
        answer=scenario, family="massive:scenario", meta={"label": scenario}, **base,
    )
    siblings = sorted(k for k in INTENTS if k.startswith(scenario + "_"))
    if len(siblings) >= 2:
        yield DecisionExample(
            id=f"{rid}-intent", question=TypedQuestion("choice", _pick_text(INTENT_INSTRUCTIONS, rid),
                                                       {k: INTENTS[k] for k in siblings}),
            answer=intent, family="massive:intent", meta={"label": intent}, **base,
        )


def from_polqa(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """ipipan/polqa pairs: is a Wikipedia passage relevant to (answers) the question."""

    q, passage, relevant = row.get("question"), row.get("passage_text"), row.get("relevant")
    if not (_ok(q) and _ok(passage)) or relevant not in (True, False):
        return
    rid = f"polqa-{split}-{idx:05d}"
    p = _pick(ANSWERS_QUESTION, rid)
    title = (row.get("passage_title") or "").strip()
    head = f"Fragment ({title})" if title else "Fragment"
    yield DecisionExample(
        id=rid, state=f"Pytanie użytkownika: {q.strip()}\n\n{head}: {passage.strip()}",
        question=p.question(), answer=p.key("true" if relevant else "false"), family="polqa",
        language="pl", source="ipipan/polqa", group=f"polqa:{row.get('question_id')}",
        meta={"label": "true" if relevant else "false", "passage_source": str(row.get("passage_source"))},
    )


EXAMS_INSTRUCTIONS = [
    "Która odpowiedź jest poprawna?",
    "Wybierz poprawne dokończenie lub odpowiedź.",
    "Które z poniższych jest prawidłowe?",
    "Wskaż właściwą odpowiedź na pytanie egzaminacyjne.",
    "Which answer is correct?",
]


def from_exams(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """mhardalov/exams crosslingual_pl: Polish vocational exam questions, 4 options."""

    q = row.get("question") or {}
    stem, choices, gold = q.get("stem"), q.get("choices") or {}, row.get("answerKey")
    labels, texts = choices.get("label") or [], choices.get("text") or []
    if not _ok(stem) or gold not in labels or len(labels) != len(texts) or not all(map(_ok, texts)):
        return
    rid = f"exams-{split}-{idx:05d}"
    yield DecisionExample(
        id=rid, state=stem.strip(),
        question=TypedQuestion("choice", _pick_text(EXAMS_INSTRUCTIONS, rid),
                               {k.lower(): v.strip() for k, v in zip(labels, texts)}),
        answer=gold.lower(), family="exams", language="pl", source="mhardalov/exams",
        group=f"exams:{row.get('id') or text_key(stem)}", meta={"label": gold.lower()},
    )


# ------------------------------------------------- evaluation-only sets (never trained on)

INCLUDE_INSTRUCTIONS = "Która odpowiedź na to pytanie jest poprawna?"


def from_belebele(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """facebook/belebele pol_Latn: reading comprehension, 4 options; the item's own question."""

    passage, q, gold = row.get("flores_passage"), row.get("question"), str(row.get("correct_answer_num"))
    options = {str(i): row.get(f"mc_answer{i}") for i in range(1, 5)}
    if not (_ok(passage) and _ok(q) and all(map(_ok, options.values()))) or gold not in options:
        return
    yield DecisionExample(
        id=f"belebele-{split}-{idx:04d}", state=passage.strip(),
        question=TypedQuestion("choice", q.strip(), {k: v.strip() for k, v in options.items()}),
        answer=gold, family="belebele", language="pl", source="facebook/belebele",
        group=f"belebele:{text_key(passage)}", meta={"label": gold},
    )


def from_include(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """CohereLabs/include-base-44 Polish: regional exam questions, 4 options."""

    q, gold = row.get("question"), str(row.get("answer"))
    options = {k: row.get(f"option_{k}") for k in "abcd"}
    key = "abcd"[int(gold)] if gold in "0123" else None
    if not (_ok(q) and all(map(_ok, options.values()))) or key is None:
        return
    yield DecisionExample(
        id=f"include-{split}-{idx:04d}", state=q.strip(),
        question=TypedQuestion("choice", INCLUDE_INSTRUCTIONS, {k: v.strip() for k, v in options.items()}),
        answer=key, family="include", language="pl", source="CohereLabs/include-base-44",
        group=f"include:{text_key(q)}", meta={"label": key, "subject": str(row.get("subject"))},
    )


def from_opendecisions(row: dict, idx: int, split: str) -> Iterable[DecisionExample]:
    """jgeuter/OpenDecisions, Polish items only. Its score questions list N levels, which
    decision-v1 cannot express, so they are scored as a choice of level ("0".."N-1")."""

    if row.get("language") != "pl":
        return
    q, expected, state = row.get("question") or {}, row.get("expected"), row.get("state")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, indent=1)
    qtype, criteria = q.get("type"), q.get("criteria")
    if qtype == "noul" and isinstance(criteria, dict) and expected in ("yes", "no"):
        question = TypedQuestion("noul", q["instructions"], {"true": criteria.get("true", "tak"),
                                                             "false": criteria.get("false", "nie")})
        answer = "true" if expected == "yes" else "false"
    elif qtype == "choice" and isinstance(criteria, dict) and str(expected) in criteria:
        question, answer = TypedQuestion("choice", q["instructions"], dict(criteria)), str(expected)
    elif qtype == "score" and isinstance(criteria, list) and 0 <= int(expected) < len(criteria):
        question = TypedQuestion("choice", q["instructions"], {str(i): c for i, c in enumerate(criteria)})
        answer = str(int(expected))
    else:
        return
    yield DecisionExample(
        id=f"opendecisions-{row['id']}", state=state.strip(), question=question, answer=answer,
        family="opendecisions", language="pl", source="jgeuter/OpenDecisions",
        group=f"opendecisions:{row['id']}", meta={"label": answer, "type": str(qtype), "tier": str(row.get("tier"))},
    )


@dataclass(frozen=True)
class Source:
    convert: Callable[[dict, int, str], Iterable[DecisionExample]]
    texts: Callable[[dict], list[str]]  # raw texts checked for overlap with test rows
    pool: str  # sources in one pool are checked against each other's test rows


def fields(*names: str) -> Callable[[dict], list[str]]:
    return lambda row: [row[n] for n in names if _ok(row.get(n))]


SOURCES = {
    "polemo_in": Source(from_polemo_in, fields("sentence"), "polemo"),
    "polemo_out": Source(from_polemo_out, fields("sentence"), "polemo"),
    "dyk": Source(from_dyk, fields("question", "answer"), "dyk"),
    "psc": Source(from_psc, fields("extract_text", "summary_text"), "psc"),
    "massive": Source(from_massive, fields("utt"), "massive"),
    "polqa": Source(from_polqa, fields("question", "passage_text"), "polqa"),
    "exams": Source(from_exams, lambda r: [r["question"]["stem"]], "exams"),
    "belebele": Source(from_belebele, fields("flores_passage"), "belebele"),
    "include": Source(from_include, fields("question"), "exams"),  # EXAMS pl repeats INCLUDE items
    "opendecisions": Source(from_opendecisions, fields(), "opendecisions"),
}
