"""Bullfinch 4B: a small decision model for Polish (typed questions, option-letter logits)."""

from .questions import ANSWER_PREFILL, TEMPLATE_VERSION, DecisionExample, TypedQuestion, render

__all__ = ["ANSWER_PREFILL", "TEMPLATE_VERSION", "DecisionExample", "TypedQuestion", "render"]
