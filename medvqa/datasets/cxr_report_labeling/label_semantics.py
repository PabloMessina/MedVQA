"""Label collapses and Normal-class rules."""

from __future__ import annotations

from typing import FrozenSet, Iterable, Mapping, Optional, Sequence

LLM_LABELS = (
    "definitely true",
    "likely true",
    "unknown",
    "likely false",
    "definitely false",
)

UNMENTIONED = "Unmentioned"
RELEASED_UNMENTIONED = "unmentioned"

# Binary gold for regex tuning
LLM_POSITIVE_LABELS: FrozenSet[str] = frozenset({"definitely true", "likely true"})

# Compact / sparse / npz released 6-way (lowercase)
PRESENCE_RANK = {
    "definitely true": 2,
    "likely true": 1,
}

NPZ_LABEL_TO_UINT8 = {
    RELEASED_UNMENTIONED: 0,
    "definitely true": 1,
    "likely true": 2,
    "unknown": 3,
    "likely false": 4,
    "definitely false": 5,
}

RESOLUTION_OWN_LLM = "own_llm"
RESOLUTION_PRESENCE_ENTAILMENT = "presence_entailment"
RESOLUTION_DESCENDANTS_NONE_POSITIVE = "descendants_matched_none_positive"
RESOLUTION_LLM_FAILED = "llm_failed"
RESOLUTION_NO_FINDING_AUTO = "no_finding_auto"
RESOLUTION_NO_FINDING_LLM = "no_finding_llm"

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

SUPPORT_DEVICES_CLASS = "support_device_present"
NORMAL_CLASS = "no_finding"


def llm_label_to_binary_positive(label: Optional[str]) -> bool:
    if label is None:
        return False
    return label.strip().lower() in LLM_POSITIVE_LABELS


def to_chexpert_like(label: Optional[str]) -> str:
    if label is None:
        return "unmentioned"
    key = label.strip().lower()
    return CHEXPERT_LIKE_COLLAPSE.get(key, "unmentioned")


def is_normal_disqualifying_positive(positive_classes: Iterable[str]) -> bool:
    """
    Return True if the report should NOT be treated as no_finding-eligible.

    Any positive modeled class disqualifies, including support devices / device abnormality
    (aligned with canonical no_finding semantics).
    """
    return any(c for c in positive_classes if c)


def normalize_nli_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    key = label.strip().lower()
    if key == "unmentioned":
        return RELEASED_UNMENTIONED
    if key in LLM_LABELS:
        return key
    return None


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
    normalized = normalize_nli_label(llm_label)
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


def resolve_presence_entailment(
    class_id: str,
    *,
    own_matched: bool,
    matched_descendants: Sequence[str],
    llm_by_class: Mapping[str, Optional[str]],
) -> dict:
    """
    Presence entailment among ``{C} ∪ matched descendants``.

    ``definitely true`` > ``likely true``. Else C's own LLM if C matched.
    Else descendant-only non-positive → ``unknown``. Child negation does not
    veto a parent. Missing LLM values are treated as not positive.
    """
    considered = [class_id, *list(matched_descendants)]
    best_rank = 0
    winners: list[str] = []
    for cid in considered:
        lab = normalize_nli_label(llm_by_class.get(cid))
        rank = PRESENCE_RANK.get(lab or "", 0)
        if rank > best_rank:
            best_rank = rank
            winners = [cid]
        elif rank == best_rank and rank > 0:
            winners.append(cid)

    if best_rank > 0:
        if class_id in winners:
            dominated_by = class_id
            resolution = RESOLUTION_OWN_LLM
        else:
            dominated_by = sorted(winners)[0]
            resolution = RESOLUTION_PRESENCE_ENTAILMENT
        label = "definitely true" if best_rank == 2 else "likely true"
        return {
            "label": label,
            "dominated_by": dominated_by,
            "resolution": resolution,
        }

    if own_matched:
        own = normalize_nli_label(llm_by_class.get(class_id))
        if own is None:
            return {
                "label": None,
                "dominated_by": None,
                "resolution": RESOLUTION_LLM_FAILED,
            }
        return {
            "label": own,
            "dominated_by": class_id,
            "resolution": RESOLUTION_OWN_LLM,
        }

    return {
        "label": "unknown",
        "dominated_by": None,
        "resolution": RESOLUTION_DESCENDANTS_NONE_POSITIVE,
    }


def resolve_no_finding_from_other_labels(
    other_labels: Mapping[str, str],
) -> Optional[tuple[str, str]]:
    """
    Auto-false ``no_finding`` from other released labels.

    Any ``definitely true`` → ``definitely false``; else any ``likely true`` →
    ``likely false``. Returns ``(label, dominated_by)`` or ``None`` if LLM is needed.
    """
    def_true = sorted(
        cid
        for cid, lab in other_labels.items()
        if cid != NORMAL_CLASS and normalize_nli_label(lab) == "definitely true"
    )
    if def_true:
        return "definitely false", def_true[0]
    likely_true = sorted(
        cid
        for cid, lab in other_labels.items()
        if cid != NORMAL_CLASS and normalize_nli_label(lab) == "likely true"
    )
    if likely_true:
        return "likely false", likely_true[0]
    return None
