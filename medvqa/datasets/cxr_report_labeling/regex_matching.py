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
