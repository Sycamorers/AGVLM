"""Stable split assignment helpers."""

import hashlib
from typing import Dict, Iterable, Tuple


def stable_fraction(value: str, salt: str) -> float:
    payload = ("%s::%s" % (salt, value)).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def assign_hash_split(value: str, salt: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    fraction = stable_fraction(value, salt)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "validation"
    return "test"


def assign_holdout(value: str, salt: str, holdout_ratio: float) -> bool:
    return stable_fraction(value, salt) < holdout_ratio


def grouped_assignments(values: Iterable[str], salt: str, holdout_ratio: float) -> Dict[str, str]:
    return {
        value: ("holdout" if assign_holdout(value, salt, holdout_ratio) else "train") for value in values
    }
