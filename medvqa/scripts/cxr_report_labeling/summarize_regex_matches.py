"""Inventory full-corpus regex matches and remaining unique LLM content_keys.

CPU-only; no API. Prints IU first. Excludes ``no_finding`` from candidate pairs.
"""

from __future__ import annotations

import argparse
import json
import logging

from medvqa.datasets.cxr_report_labeling.paths import DEFAULT_RUN_ID
from medvqa.datasets.cxr_report_labeling.report_loaders import ALL_DATASETS
from medvqa.datasets.cxr_report_labeling.verify_jobs import summarize_regex_matches
from medvqa.scripts.cxr_report_labeling.annotate_reports_with_llm import (
    DEFAULT_MODEL_NAME,
)
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _format_dataset_block(row: dict) -> str:
    if row.get("missing"):
        return (
            f"  {row['dataset']}: MISSING {row.get('regex_matches_path')}"
        )
    per_class = row.get("per_class") or {}
    top = sorted(per_class.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    top_s = ", ".join(f"{k}={v}" for k, v in top)
    return (
        f"  {row['dataset']}: n_reports={row['n_reports']} n_pos={row['n_pos']} "
        f"density={row['density']} empty={row['empty_pct']}% "
        f"candidates={row['n_candidate_pairs']} "
        f"remaining_content_keys={row['n_remaining_content_keys']} "
        f"ceil(N/{row['chunk_size']})={row['n_chunks']}\n"
        f"    top classes: {top_s or '(none)'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=list(ALL_DATASETS),
        help="Default: all four corpora (IU printed first)",
    )
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--skip_annotation_paths",
        nargs="+",
        default=None,
        help="Extra successful annotation jsonl(s) whose content_keys are skipped",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the full summary dict as JSON (includes per-class counts)",
    )
    args = parser.parse_args()
    setup_logging()
    summary = summarize_regex_matches(
        datasets=args.datasets,
        run_id=args.run_id,
        chunk_size=args.chunk_size,
        model_name=args.model_name,
        skip_annotation_paths=args.skip_annotation_paths,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2))
        return
    print(f"run_id={summary['run_id']} model={summary['model_name']}")
    for row in summary["datasets"]:
        print(_format_dataset_block(row))
    tot = summary["total"]
    print(
        f"  TOTAL: n_reports={tot['n_reports']} n_pos={tot['n_pos']} "
        f"density={tot['density']} empty={tot['empty_pct']}% "
        f"candidates={tot['n_candidate_pairs']} "
        f"remaining_content_keys={tot['n_remaining_content_keys']} "
        f"ceil(N/{tot['chunk_size']})={tot['n_chunks']}"
    )


if __name__ == "__main__":
    main()
