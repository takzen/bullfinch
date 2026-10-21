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


def _pick(options: list[Phrasing], row_id: str) -> Phrasing:
    digest = hashlib.sha256(row_id.encode()).digest()
    return options[digest[0] % len(options)]


def text_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()[:16]


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


@dataclass(frozen=True)
class Source:
    convert: Callable[[dict, int, str], Iterable[DecisionExample]]
    texts: tuple[str, ...]  # raw fields checked for overlap with test rows
    pool: str               # sources in one pool are checked against each other's test rows


SOURCES = {
    "polemo_in": Source(from_polemo_in, ("sentence",), "polemo"),
    "polemo_out": Source(from_polemo_out, ("sentence",), "polemo"),
    "dyk": Source(from_dyk, ("question", "answer"), "dyk"),
    "psc": Source(from_psc, ("extract_text", "summary_text"), "psc"),
}
