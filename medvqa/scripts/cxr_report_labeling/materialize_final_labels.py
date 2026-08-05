"""Materialize final Unmentioned / 5-way labels per dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import ClassRegistry, build_class_registry
from medvqa.datasets.cxr_report_labeling.label_semantics import (
    NORMAL_CLASS,
    SUPPORT_DEVICES_CLASS,
    UNMENTIONED,
    final_label_for_class,
    is_normal_disqualifying_positive,
)
from medvqa.datasets.cxr_report_labeling.paths import DEFAULT_RUN_ID, get_cache_run_dir, get_results_run_dir
from medvqa.scripts.cxr_report_labeling.eval_regex_vs_llm import load_annotations_flexible
from medvqa.utils.files_utils import load_jsonl, save_json
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _collect_llm_labels(paths: Sequence[str]) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        ann = load_annotations_flexible(path)
        for k, v in ann.items():
            out[k] = v["label"]
    return out


def materialize_final_labels(
    *,
    dataset: str,
    run_id: str = DEFAULT_RUN_ID,
    regex_matches_path: Optional[str] = None,
    llm_annotation_paths: Optional[Sequence[str]] = None,
    model_name: str = "gemini-2.5-flash-lite-preview-09-2025",
) -> str:
    setup_logging()
    cache_dir = get_cache_run_dir(run_id)
    results_dir = get_results_run_dir(run_id)
    if regex_matches_path is None:
        regex_matches_path = os.path.join(cache_dir, "full_regex_matches", f"{dataset}.jsonl")
    matches = load_jsonl(regex_matches_path)

    registry_path = os.path.join(results_dir, "class_registry.json")
    registry = ClassRegistry.load(registry_path) if os.path.exists(registry_path) else build_class_registry()
    class_ids = registry.class_ids()

    safe_model = model_name.replace("/", "_")
    if llm_annotation_paths is None:
        llm_annotation_paths = [
            os.path.join(cache_dir, "llm_annotations", safe_model, "sample_annotations.jsonl"),
            os.path.join(cache_dir, "llm_annotations", safe_model, f"verify_{dataset}.jsonl"),
        ]
    llm_labels = _collect_llm_labels(llm_annotation_paths)

    out_dir = os.path.join(results_dir, "final_labels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}.jsonl")

    n_missing_llm = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in matches:
            uid = row["uid"]
            positives = set(row.get("positive_classes") or [])
            labels: Dict[str, str] = {}

            # Non-Normal classes first
            for class_id in class_ids:
                if class_id == NORMAL_CLASS:
                    continue
                entry = registry.by_id()[class_id]
                if not entry.has_regex:
                    labels[class_id] = UNMENTIONED
                    continue
                if class_id not in positives:
                    labels[class_id] = UNMENTIONED
                else:
                    llm = llm_labels.get((uid, class_id))
                    if llm is None:
                        n_missing_llm += 1
                        labels[class_id] = UNMENTIONED  # conservative until verified
                    else:
                        labels[class_id] = final_label_for_class(regex_matched=True, llm_label=llm)

            # Normal last
            binary_pos_classes = [
                c for c, lab in labels.items()
                if lab in ("Definitely True", "Likely True")
            ]
            if is_normal_disqualifying_positive(binary_pos_classes, support_devices_ok=True):
                labels[NORMAL_CLASS] = "Definitely False"
            else:
                llm = llm_labels.get((uid, NORMAL_CLASS))
                if llm is not None:
                    labels[NORMAL_CLASS] = final_label_for_class(regex_matched=True, llm_label=llm)
                else:
                    labels[NORMAL_CLASS] = UNMENTIONED

            out = {
                "uid": uid,
                "dataset": dataset,
                "report_id": row.get("report_id"),
                "labels": labels,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    meta = {
        "dataset": dataset,
        "n_reports": len(matches),
        "n_missing_llm_for_regex_positives": n_missing_llm,
        "path": out_path,
    }
    save_json(meta, os.path.join(out_dir, f"{dataset}_manifest.json"))
    logger.info("Wrote final labels to %s (missing LLM for %d positives)", out_path, n_missing_llm)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--regex_matches_path", default=None)
    parser.add_argument("--model_name", default="gemini-2.5-flash-lite-preview-09-2025")
    args = parser.parse_args()
    materialize_final_labels(
        dataset=args.dataset,
        run_id=args.run_id,
        regex_matches_path=args.regex_matches_path,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
