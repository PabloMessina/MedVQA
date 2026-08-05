"""LLM-verify regex-positive (report, class) pairs on full datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    build_class_registry,
    load_system_prompt_text,
)
from medvqa.datasets.cxr_report_labeling.paths import DEFAULT_RUN_ID, get_cache_run_dir, get_results_run_dir
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
from medvqa.datasets.cxr_report_labeling.schemas import annotation_resume_key, build_nli_query
from medvqa.scripts.cxr_report_labeling.annotate_reports_with_llm import (
    estimate_jobs,
    parse_llm_model_output,
)
from medvqa.utils.files_utils import load_jsonl
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.openai_api_utils import orchestrate_api_calls

logger = logging.getLogger(__name__)


def _load_reports_by_uid(dataset: str) -> Dict[str, dict]:
    if dataset == DATASET_MIMICCXR:
        reports = load_mimiccxr_reports()
    elif dataset == DATASET_CHEXPERT_PLUS:
        reports = load_chexpert_plus_reports()
    elif dataset == DATASET_REXGRADIENT:
        reports = load_rexgradient_reports()
    elif dataset == DATASET_IUXRAY:
        reports = load_iuxray_reports()
    else:
        raise ValueError(dataset)
    return {r["uid"]: r for r in reports}


def _load_sample_gold(path: Optional[str]) -> Dict[Tuple[str, str], str]:
    if path is None or not os.path.exists(path):
        return {}
    from medvqa.scripts.cxr_report_labeling.eval_regex_vs_llm import load_annotations_flexible

    ann = load_annotations_flexible(path)
    return {k: v["label"] for k, v in ann.items()}


def run_verify_positives(
    *,
    dataset: str,
    run_id: str = DEFAULT_RUN_ID,
    regex_matches_path: Optional[str] = None,
    sample_annotations_path: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-lite-preview-09-2025",
    api_key_name: str = "GOOGLE_API_KEY",
    api_type: str = "gemini",
    max_queries: Optional[int] = 50,
    class_ids: Optional[Sequence[str]] = None,
    estimate_only: bool = False,
    dry_run: bool = False,
) -> dict:
    setup_logging()
    cache_dir = get_cache_run_dir(run_id)
    results_dir = get_results_run_dir(run_id)
    if regex_matches_path is None:
        regex_matches_path = os.path.join(cache_dir, "full_regex_matches", f"{dataset}.jsonl")
    matches = load_jsonl(regex_matches_path)
    registry_path = os.path.join(results_dir, "class_registry.json")
    registry = ClassRegistry.load(registry_path) if os.path.exists(registry_path) else build_class_registry()
    by_id = registry.by_id()

    if sample_annotations_path is None:
        # try default sample annotations
        safe_model = model_name.replace("/", "_")
        sample_annotations_path = os.path.join(
            cache_dir, "llm_annotations", safe_model, "sample_annotations.jsonl"
        )
    gold = _load_sample_gold(sample_annotations_path)

    class_filter = set(class_ids) if class_ids is not None else None
    reports = _load_reports_by_uid(dataset)

    save_path = Path(cache_dir) / "llm_annotations" / model_name.replace("/", "_") / f"verify_{dataset}.jsonl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: Set[str] = set()
    if save_path.exists():
        for row in load_jsonl(str(save_path)):
            meta = row.get("metadata") or {}
            if meta.get("resume_key"):
                done_keys.add(meta["resume_key"])

    jobs = []
    reused = 0
    for row in matches:
        uid = row["uid"]
        report = reports.get(uid)
        if report is None:
            continue
        for class_id in row.get("positive_classes") or []:
            if class_filter is not None and class_id not in class_filter:
                continue
            if (uid, class_id) in gold:
                reused += 1
                continue
            entry = by_id[class_id]
            if not entry.has_regex:
                continue
            resume_key = annotation_resume_key(
                uid, class_id, entry.prompt_hash, registry.system_prompt_hash, model_name
            )
            if resume_key in done_keys:
                continue
            jobs.append(
                {
                    "query": build_nli_query(report["report_text"], entry.prompt_text),
                    "metadata": {
                        "resume_key": resume_key,
                        "uid": uid,
                        "dataset": dataset,
                        "class_id": class_id,
                        "prompt_hash": entry.prompt_hash,
                        "system_prompt_hash": registry.system_prompt_hash,
                        "model_name": model_name,
                        "stage": "verify_positive",
                    },
                }
            )

    if max_queries is not None:
        jobs = jobs[:max_queries]

    info = {
        "dataset": dataset,
        "n_match_rows": len(matches),
        "n_reused_sample_gold": reused,
        "n_jobs_queued": len(jobs),
        "save_path": str(save_path),
        "estimate": estimate_jobs(len(jobs)),
    }
    logger.info("%s", info)
    if estimate_only or dry_run or not jobs:
        return info

    query_to_meta = {j["query"]: j["metadata"] for j in jobs}
    texts = [j["query"] for j in jobs]
    orchestrate_api_calls(
        texts=texts,
        system_instructions=load_system_prompt_text(),
        model_name=model_name,
        api_key_name=api_key_name,
        api_type=api_type,
        max_requests_per_minute=100,
        max_tokens_per_minute=100_000,
        max_tokens_per_request=512,
        temperature=0.0,
        parse_output=parse_llm_model_output,
        save_filepath=str(save_path),
        tmp_dir=os.path.join(cache_dir, "api_tmp"),
    )
    rows = load_jsonl(str(save_path))
    with open(save_path, "w", encoding="utf-8") as f:
        for row in rows:
            meta = dict(row.get("metadata") or {})
            q = meta.get("query")
            if q in query_to_meta:
                meta.update(query_to_meta[q])
            row = dict(row)
            row["metadata"] = meta
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info["n_rows"] = len(rows)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=[DATASET_MIMICCXR, DATASET_CHEXPERT_PLUS, DATASET_REXGRADIENT, DATASET_IUXRAY])
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--regex_matches_path", default=None)
    parser.add_argument("--sample_annotations_path", default=None)
    parser.add_argument("--model_name", default="gemini-2.5-flash-lite-preview-09-2025")
    parser.add_argument("--api_key_name", default="GOOGLE_API_KEY")
    parser.add_argument("--api_type", default="gemini")
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--estimate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_verify_positives(
        dataset=args.dataset,
        run_id=args.run_id,
        regex_matches_path=args.regex_matches_path,
        sample_annotations_path=args.sample_annotations_path,
        model_name=args.model_name,
        api_key_name=args.api_key_name,
        api_type=args.api_type,
        max_queries=args.max_queries,
        class_ids=args.classes,
        estimate_only=args.estimate_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
