"""Apply regex patterns to a full dataset; write sparse positive matches."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Set

from medvqa.datasets.cxr_report_labeling.class_registry import ClassRegistry, build_class_registry
from medvqa.datasets.cxr_report_labeling.paths import DEFAULT_RUN_ID, get_cache_run_dir, get_results_run_dir
from medvqa.datasets.cxr_report_labeling.regex_matching import batch_match_prompt_classes, collect_match_spans_for_class
from medvqa.datasets.cxr_report_labeling.report_loaders import (
    DATASET_CHEXPERT_PLUS,
    DATASET_IUXRAY,
    DATASET_MIMICCXR,
    DATASET_REXGRADIENT,
    load_chexpert_plus_reports,
    load_iuxray_reports,
    load_mimiccxr_reports,
    load_rexgradient_reports,
)
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _load_dataset(dataset: str, limit: Optional[int] = None) -> List[dict]:
    if dataset == DATASET_MIMICCXR:
        return load_mimiccxr_reports(limit=limit)
    if dataset == DATASET_CHEXPERT_PLUS:
        return load_chexpert_plus_reports(limit=limit)
    if dataset == DATASET_REXGRADIENT:
        return load_rexgradient_reports(limit_per_split=limit)
    if dataset == DATASET_IUXRAY:
        return load_iuxray_reports(limit=limit)
    raise ValueError(dataset)


def apply_regex_to_dataset(
    *,
    dataset: str,
    run_id: str = DEFAULT_RUN_ID,
    class_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    num_processes: int = 1,
    include_spans: bool = False,
) -> str:
    setup_logging()
    results_dir = get_results_run_dir(run_id)
    cache_dir = get_cache_run_dir(run_id)
    out_dir = os.path.join(cache_dir, "full_regex_matches")
    os.makedirs(out_dir, exist_ok=True)

    registry_path = os.path.join(results_dir, "class_registry.json")
    registry = ClassRegistry.load(registry_path) if os.path.exists(registry_path) else build_class_registry()
    reports = _load_dataset(dataset, limit=limit)
    texts = [r["report_text"] for r in reports]
    ids = list(class_ids) if class_ids is not None else registry.regex_backed_class_ids()

    matches = batch_match_prompt_classes(
        texts, registry, class_ids=ids, num_processes=num_processes
    )
    # invert: index -> positive classes
    idx_to_pos: Dict[int, List[str]] = {i: [] for i in range(len(reports))}
    for class_id, idxs in matches.items():
        for i in idxs:
            idx_to_pos[i].append(class_id)

    out_path = os.path.join(out_dir, f"{dataset}.jsonl")
    by_id = registry.by_id()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(reports):
            pos = sorted(idx_to_pos[i])
            row = {
                "uid": r["uid"],
                "dataset": dataset,
                "report_id": r["report_id"],
                "positive_classes": pos,
            }
            if include_spans and pos:
                span_map = {}
                for class_id in pos:
                    entry = by_id[class_id]
                    spans = []
                    for regex_name in entry.regex_names or []:
                        spans.extend(collect_match_spans_for_class(r["report_text"], regex_name))
                    span_map[class_id] = spans
                row["match_spans"] = span_map
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote regex matches for %s (%d reports) to %s", dataset, len(reports), out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=[DATASET_MIMICCXR, DATASET_CHEXPERT_PLUS, DATASET_REXGRADIENT, DATASET_IUXRAY])
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--include_spans", action="store_true")
    args = parser.parse_args()
    apply_regex_to_dataset(
        dataset=args.dataset,
        run_id=args.run_id,
        class_ids=args.classes,
        limit=args.limit,
        num_processes=args.num_processes,
        include_spans=args.include_spans,
    )


if __name__ == "__main__":
    main()
