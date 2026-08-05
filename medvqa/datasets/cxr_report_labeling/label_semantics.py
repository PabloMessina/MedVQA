"""Label collapses and Normal-class rules."""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional, Set

LLM_LABELS = (
    "definitely true",
    "likely true",
    "unknown",
    "likely false",
    "definitely false",
)

UNMENTIONED = "Unmentioned"

# Binary gold for regex tuning
LLM_POSITIVE_LABELS: FrozenSet[str] = frozenset({"definitely true", "likely true"})

# Paper compatibility with CheXpert-like 4-way
CHEXPERT_LIKE_COLLAPSE = {
    "definitely true": "positive",
    "likely true": "uncertain",
    "likely false": "negative",
    "definitely false": "negative",
    "unknown": "unmentioned",
    UNMENTIONED.lower(): "unmentioned",
    "unmentioned": "unmentioned",
}

SUPPORT_DEVICES_CLASS = "Support Devices"
NORMAL_CLASS = "Normal"


def llm_label_to_binary_positive(label: Optional[str]) -> bool:
    if label is None:
        return False
    return label.strip().lower() in LLM_POSITIVE_LABELS


def to_chexpert_like(label: Optional[str]) -> str:
    if label is None:
        return "unmentioned"
    key = label.strip().lower()
    return CHEXPERT_LIKE_COLLAPSE.get(key, "unmentioned")


def is_normal_disqualifying_positive(
    positive_classes: Iterable[str],
    *,
    support_devices_ok: bool = True,
) -> bool:
    """
    Return True if the report should NOT be treated as Normal-eligible.

    Support-Devices-only positives do not disqualify Normal when support_devices_ok.
    """
    positives: Set[str] = {c for c in positive_classes if c}
    if support_devices_ok:
        positives.discard(SUPPORT_DEVICES_CLASS)
    return len(positives) > 0


def final_label_for_class(
    *,
    regex_matched: bool,
    llm_label: Optional[str],
) -> str:
    """Released label: regex miss -> Unmentioned; regex hit -> LLM 5-way label."""
    if not regex_matched:
        return UNMENTIONED
    if llm_label is None:
        raise ValueError("regex matched but llm_label is missing")
    normalized = llm_label.strip().lower()
    if normalized not in LLM_LABELS:
        raise ValueError(f"Invalid LLM label: {llm_label}")
    # Preserve title-case style for release
    return {
        "definitely true": "Definitely True",
        "likely true": "Likely True",
        "unknown": "Unknown",
        "likely false": "Likely False",
        "definitely false": "Definitely False",
    }[normalized]
