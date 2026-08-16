"""Regex match helpers for prompt classes (with hierarchy)."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import ClassRegistry
from medvqa.datasets.regular_expressions.cxr_patterns import (
    _CLASS_NAME_TO_REGEX_PATTERNS,
    collect_reports_matching_class,
)


def match_span(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    if not m:
        return None
    return m.group(0)


def collect_match_spans_for_class(
    text: str,
    regex_class_name: str,
) -> List[str]:
    """Return matched substrings for a class (including subclass patterns)."""
    spans: List[str] = []
    definition = _CLASS_NAME_TO_REGEX_PATTERNS[regex_class_name]
    if isinstance(definition, list):
        for item in definition:
            if isinstance(item, re.Pattern):
                s = match_span(item, text)
                if s:
                    spans.append(s)
            elif isinstance(item, str):
                spans.extend(collect_match_spans_for_class(text, item))
    else:
        s = match_span(definition, text)
        if s:
            spans.append(s)
    # unique preserve order
    seen = set()
    out = []
    for s in spans:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _iter_stem_spans(text: str, regex_class_name: str) -> List[Tuple[str, str]]:
    """Yield ``(stem, span)`` for this composition node and matching descendants.

    A compiled pattern attached to ``regex_class_name`` is the residual
    ``<stem>.txt`` arm (``stem == regex_class_name``). String children recurse.
    """
    definition = _CLASS_NAME_TO_REGEX_PATTERNS[regex_class_name]
    hits: List[Tuple[str, str]] = []
    if isinstance(definition, list):
        for item in definition:
            if isinstance(item, re.Pattern):
                s = match_span(item, text)
                if s:
                    hits.append((regex_class_name, s))
            elif isinstance(item, str):
                hits.extend(_iter_stem_spans(text, item))
            else:
                raise TypeError(
                    f"Expected pattern or class name under {regex_class_name!r}, "
                    f"got {type(item)}"
                )
    else:
        s = match_span(definition, text)
        if s:
            hits.append((regex_class_name, s))
    return hits


def explain_regex_hits(
    text: str,
    class_ids: Sequence[str],
) -> List[dict]:
    """Named live hits for prompt classes (typically precomputed ``positive_classes``).

    Each row is ``{class_id, stem, span, via_child}``:

    - ``class_id``: prompt class being explained
    - ``stem``: composition leaf / residual whose pattern fired
    - ``span``: first matched substring for that stem (same as
      :func:`collect_match_spans_for_class`)
    - ``via_child``: True when ``stem != class_id`` (parent listed because a
      child regex fired)

    Classes missing from the composition table are skipped. A class in
    ``class_ids`` with no live hit is omitted (caller can contrast with the
    precomputed list after regex edits).
    """
    rows: List[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for class_id in class_ids:
        if class_id not in _CLASS_NAME_TO_REGEX_PATTERNS:
            continue
        for stem, span in _iter_stem_spans(text, class_id):
            key = (class_id, stem, span)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "class_id": class_id,
                    "stem": stem,
                    "span": span,
                    "via_child": stem != class_id,
                }
            )
    return rows


def report_matches_prompt_class(
    text: str,
    class_id: str,
    registry: ClassRegistry,
) -> Tuple[bool, List[str]]:
    entry = registry.by_id()[class_id]
    if not entry.regex_names:
        return False, []
    all_spans: List[str] = []
    for regex_name in entry.regex_names:
        all_spans.extend(collect_match_spans_for_class(text, regex_name))
    return (len(all_spans) > 0), all_spans


def batch_match_prompt_classes(
    texts: Sequence[str],
    registry: ClassRegistry,
    class_ids: Optional[Sequence[str]] = None,
    num_processes: int = 1,
) -> Dict[str, Set[int]]:
    cleaned = [" ".join(t.lower().split()) for t in texts]
    cache: Dict[str, List[int]] = {}
    ids = list(class_ids) if class_ids is not None else registry.regex_backed_class_ids()
    out: Dict[str, Set[int]] = {}
    by_id = registry.by_id()
    for class_id in ids:
        entry = by_id[class_id]
        if not entry.regex_names:
            out[class_id] = set()
            continue
        idxs: Set[int] = set()
        for regex_name in entry.regex_names:
            matched = collect_reports_matching_class(
                reports=cleaned,
                class_name=regex_name,
                class_match_cache=cache,
                num_processes=num_processes,
            )
            idxs.update(matched)
        out[class_id] = idxs
    return out
