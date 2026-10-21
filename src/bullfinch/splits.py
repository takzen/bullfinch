"""Deterministic dev / calibration carve-outs for sets that ship without a validation split.

Converters keep the official split of every row (train, validation as ``dev``, test). Rows
from a train split can be carved by group so the same source record, with all its phrasings,
lands in exactly one split. Official test rows never pass through here.
"""

from __future__ import annotations

import hashlib

SPLIT_BUCKETS = (("calibration", 0.05), ("dev", 0.05))


def split_for(group: str) -> str:
    """Deterministic split for a group: the same record never lands in two splits."""

    x = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, share in SPLIT_BUCKETS:
        edge += share
        if x < edge:
            return name
    return "train"
