"""Option scoring through Ollama, for baselines such as Bielik (template ``decision-v1``).

The same prompt as :class:`bullfinch.engine.DecisionEngine`, wrapped in ChatML and sent raw
with the ``{"answer": "`` prefill; one token is generated and its ``top_logprobs`` give the
option-code scores. A code missing from the top list gets one nat below the lowest listed
token, so it keeps a small, honest share instead of zero.
"""

from __future__ import annotations

import json
import math
import urllib.request

from .engine import DecisionEngine, frame_for, softmax
from .questions import ANSWER_PREFILL, TypedQuestion, render


class OllamaEngine:
    def __init__(self, model: str, host: str = "http://localhost:11434", tta: bool = False,
                 top: int = 20, num_ctx: int = 4096) -> None:
        self.model_name, self.revision, self.adapter = model, "ollama", None
        self.host, self.tta, self.top, self.num_ctx = host.rstrip("/"), tta, top, num_ctx

    def _top_logprobs(self, prompt: str) -> list[dict]:
        body = {"model": self.model_name, "prompt": prompt, "raw": True, "stream": False,
                "logprobs": True, "top_logprobs": self.top, "keep_alive": "30m",
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": self.num_ctx}}
        req = urllib.request.Request(f"{self.host}/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.load(resp)
        if data.get("prompt_eval_count", 0) >= self.num_ctx:
            raise ValueError(f"prompt does not fit num_ctx {self.num_ctx}")
        return data["logprobs"][0]["top_logprobs"]

    def option_logprobs(self, state: str, question: TypedQuestion, frame: str | None = None,
                        order: list[str] | None = None) -> dict[str, float]:
        messages, codes = render(state, question, frame=frame or frame_for(question), order=order)
        prompt = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{ANSWER_PREFILL}"
        tops = self._top_logprobs(prompt)
        seen: dict[str, float] = {}
        for t in tops:
            tok = t["token"].strip()
            if tok in codes and tok not in seen:
                seen[tok] = t["logprob"]
        floor = min(t["logprob"] for t in tops) - 1.0
        probs = softmax([seen.get(c, floor) for c in codes])
        return {key: math.log(max(p, 1e-12)) for key, p in zip(codes.values(), probs)}

    logprobs = DecisionEngine.logprobs
