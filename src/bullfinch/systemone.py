"""Client for any decision server that speaks ``POST /v1/systemone`` (basal, Jev, ours).

Each model is asked the way its API expects: the raw state and the typed question go in as
they are, with no Bullfinch prompt template, and the option probabilities come back. Only
``choice`` and ``noul`` are scored; our 5-step ``score`` has no shared definition across
servers.
"""

from __future__ import annotations

import json
import math
import urllib.request

from .questions import TypedQuestion


class SystemOneEngine:
    def __init__(self, url: str, key: str | None = None, model: str | None = None, timeout: float = 120) -> None:
        self.url = url.rstrip("/") + "/v1/systemone"
        self.key, self.timeout, self.model = key, timeout, model
        self.model_name, self.revision, self.adapter, self.tta = model or url, "systemone", None, False

    def logprobs(self, state: str, question: TypedQuestion) -> dict[str, float]:
        if question.type not in ("choice", "noul"):
            raise ValueError(f"{question.type} questions are not scored over /v1/systemone")
        q = {"type": question.type, "instructions": question.instructions, "criteria": dict(question.criteria)}
        body = {"state": state, "questions": {"q": q}}
        if self.model:
            body["model"] = self.model
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            answer = json.load(resp)["answers"]["q"]
        if question.type == "noul":
            p = float(answer["noul"] if "noul" in answer else answer["probabilities"]["true"])
            probs = {"true": p, "false": 1.0 - p}
        else:
            probs = {k: float(answer["probabilities"].get(k, 0.0)) for k in question.criteria}
        total = sum(probs.values()) or 1.0
        return {k: math.log(max(v / total, 1e-12)) for k, v in probs.items()}
