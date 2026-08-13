#!/usr/bin/env python3
"""Run per-class mini-report fixtures against canonical CXR regexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medvqa.datasets.cxr_report_labeling.class_registry import build_class_registry
from medvqa.datasets.cxr_report_labeling.regex_matching import (
    report_matches_prompt_class,
)
from medvqa.datasets.regular_expressions.cxr_patterns import (
    _CLASS_NAME_TO_REGEX_PATTERNS,
)
from medvqa.utils.constants import CANONICAL_CLASS_TO_REGEX_CLASSES

MINI_TESTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "regular_expressions"
    / "cxr_class_mini_tests"
)
MINI_TESTS_BY_CLASS_DIR = MINI_TESTS_DIR / "by_class"


def load_all_fixtures(directory: Path = MINI_TESTS_BY_CLASS_DIR) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No mini-test fixtures in {directory}")
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            file_rows = data
        elif isinstance(data, dict):
            file_rows = [data]
        else:
            raise TypeError(f"Expected object or list in {path}")
        for row in file_rows:
            cid = row["class_id"]
            if cid in seen:
                raise ValueError(f"Duplicate class_id {cid} in fixtures")
            if path.stem != cid:
                raise ValueError(f"Fixture filename {path.name} != class_id {cid}")
            seen.add(cid)
            rows.append(row)
    return rows


def validate_fixture_coverage(rows: list[dict]) -> None:
    expected = {cid for cid, regs in CANONICAL_CLASS_TO_REGEX_CLASSES.items() if regs is not None}
    got = {r["class_id"] for r in rows}
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise AssertionError(f"Fixture coverage mismatch: missing={missing} extra={extra}")
    for row in rows:
        assert len(row.get("positives") or []) >= 3, row["class_id"]
        assert len(row.get("negatives") or []) >= 3, row["class_id"]


def run_mini_tests(*, strict_soft: bool = False) -> int:
    for cid, regs in CANONICAL_CLASS_TO_REGEX_CLASSES.items():
        if regs is None:
            continue
        for name in regs:
            assert name in _CLASS_NAME_TO_REGEX_PATTERNS, (cid, name)

    registry = build_class_registry()
    rows = load_all_fixtures()
    validate_fixture_coverage(rows)

    hard_failures: list[str] = []
    soft_warnings: list[str] = []

    for row in rows:
        cid = row["class_id"]
        for text in row["positives"]:
            ok, spans = report_matches_prompt_class(text, cid, registry)
            if not ok:
                hard_failures.append(f"POS miss [{cid}]: {text!r}")
        for text in row["negatives"]:
            ok, spans = report_matches_prompt_class(text, cid, registry)
            if ok:
                hard_failures.append(f"NEG hit [{cid}]: {text!r} spans={spans}")
        for text in row.get("negation_soft") or []:
            ok, spans = report_matches_prompt_class(text, cid, registry)
            if ok:
                msg = f"SOFT negation still matches [{cid}]: {text!r} spans={spans}"
                if strict_soft:
                    hard_failures.append(msg)
                else:
                    soft_warnings.append(msg)

    print(f"classes tested: {len(rows)}")
    print(f"hard failures: {len(hard_failures)}")
    print(f"soft warnings: {len(soft_warnings)}")
    for msg in hard_failures[:50]:
        print("FAIL:", msg)
    if len(hard_failures) > 50:
        print(f"... and {len(hard_failures) - 50} more hard failures")
    for msg in soft_warnings[:20]:
        print("WARN:", msg)
    if len(soft_warnings) > 20:
        print(f"... and {len(soft_warnings) - 20} more soft warnings")
    return 1 if hard_failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-soft",
        action="store_true",
        help="Treat negation_soft hits as hard failures",
    )
    args = parser.parse_args(argv)
    return run_mini_tests(strict_soft=args.strict_soft)


if __name__ == "__main__":
    sys.exit(main())
