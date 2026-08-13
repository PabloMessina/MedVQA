"""Compact legacy/bloated sample_annotations.jsonl into namespaced query_hash cache.

Groups by recomputed query_hash, keeps one valid parsed_response per hash, then
fan-outs to current samples.jsonl uids that share report_text_hash + class_id.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    build_class_registry,
)
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_cache_run_dir,
    get_results_run_dir,
    get_sample_annotations_path,
)
from medvqa.datasets.cxr_report_labeling.schemas import (
    annotation_config_fingerprint,
    annotation_config_slug,
    annotation_content_key,
    annotation_query_hash,
    report_text_hash,
)
from medvqa.scripts.cxr_report_labeling.annotate_reports_with_llm import (
    DEFAULT_MODEL_NAME,
    _parsed_response_ok,
)
from medvqa.utils.files_utils import load_jsonl
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_REPORT_RE = re.compile(r"<report>(.*?)</report>", re.DOTALL | re.IGNORECASE)


def _extract_report_from_query(query: str) -> Optional[str]:
    if not query:
        return None
    m = _REPORT_RE.search(query)
    if not m:
        return None
    return m.group(1).strip()


def compact_annotations(
    *,
    run_id: str = DEFAULT_RUN_ID,
    samples_path: Optional[str] = None,
    legacy_annotations_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0.0,
    max_tokens_per_request: int = 2048,
    config_extra: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    setup_logging()
    results_dir = get_results_run_dir(run_id)
    cache_dir = get_cache_run_dir(run_id)
    if samples_path is None:
        samples_path = os.path.join(results_dir, "samples.jsonl")
    samples = load_jsonl(samples_path)

    registry_path = os.path.join(results_dir, "class_registry.json")
    registry = (
        ClassRegistry.load(registry_path)
        if os.path.exists(registry_path)
        else build_class_registry()
    )
    by_id = registry.by_id()

    safe_model = model_name.replace("/", "_")
    if legacy_annotations_path is None:
        legacy_annotations_path = os.path.join(
            cache_dir, "llm_annotations", safe_model, "sample_annotations.jsonl"
        )

    config_fingerprint = annotation_config_fingerprint(
        temperature=temperature,
        max_tokens_per_request=max_tokens_per_request,
        extra=config_extra,
    )
    config_slug = annotation_config_slug(
        registry.system_prompt_hash,
        temperature=temperature,
        max_tokens_per_request=max_tokens_per_request,
        extra=config_extra,
    )
    out_path = get_sample_annotations_path(run_id, model_name, config_slug)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    legacy = Path(legacy_annotations_path)
    backup_path = legacy.with_name(
        f"sample_annotations.before_query_hash_compact_{stamp}.jsonl"
    )

    # report_text_hash → list of sample records
    by_rhash: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        h = s.get("report_text_hash") or report_text_hash(s["report_text"])
        by_rhash[h].append(s)

    # prompt_hash → class_id (prompt texts should be unique)
    prompt_to_class = {e.prompt_hash: e.class_id for e in registry.classes}

    n_legacy = 0
    n_bad_parse = 0
    n_unresolved = 0
    best_by_qh: Dict[str, dict] = {}

    logger.info("Reading legacy annotations from %s", legacy_annotations_path)
    with open(legacy_annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            n_legacy += 1
            row = json.loads(line)
            parsed = row.get("parsed_response")
            if not _parsed_response_ok(parsed):
                n_bad_parse += 1
                continue
            meta = row.get("metadata") or {}
            class_id = meta.get("class_id")
            prompt_hash = meta.get("prompt_hash")
            if not class_id and prompt_hash:
                class_id = prompt_to_class.get(prompt_hash)
            rhash = meta.get("report_text_hash")
            if not rhash:
                report = _extract_report_from_query(meta.get("query") or "")
                if report:
                    rhash = report_text_hash(report)
            if not class_id or not rhash or class_id not in by_id:
                n_unresolved += 1
                continue
            entry = by_id[class_id]
            sys_hash = meta.get("system_prompt_hash") or registry.system_prompt_hash
            content_key = annotation_content_key(rhash, entry.prompt_hash, sys_hash)
            query_hash = annotation_query_hash(
                content_key, model_name, config_fingerprint
            )
            if query_hash not in best_by_qh:
                best_by_qh[query_hash] = {
                    "parsed_response": parsed,
                    "content_key": content_key,
                    "query_hash": query_hash,
                    "class_id": class_id,
                    "prompt_hash": entry.prompt_hash,
                    "system_prompt_hash": sys_hash,
                    "model_name": model_name,
                    "config_fingerprint": config_fingerprint,
                    "report_text_hash": rhash,
                    "query": meta.get("query"),
                }

    # Fan-out to current samples
    out_rows: List[dict] = []
    for qh, payload in best_by_qh.items():
        rhash = payload["report_text_hash"]
        class_id = payload["class_id"]
        for sample in by_rhash.get(rhash, []):
            meta = {
                "query_hash": payload["query_hash"],
                "content_key": payload["content_key"],
                "uid": sample["uid"],
                "dataset": sample["dataset"],
                "class_id": class_id,
                "prompt_hash": payload["prompt_hash"],
                "system_prompt_hash": payload["system_prompt_hash"],
                "model_name": model_name,
                "config_fingerprint": config_fingerprint,
                "report_text_hash": rhash,
            }
            if payload.get("query"):
                meta["query"] = payload["query"]
            out_rows.append(
                {"metadata": meta, "parsed_response": payload["parsed_response"]}
            )

    # Remaining unique queries for current samples
    n_needed = 0
    n_covered = 0
    done_qh = set(best_by_qh)
    for sample in samples:
        rhash = sample.get("report_text_hash") or report_text_hash(sample["report_text"])
        for class_id, entry in by_id.items():
            n_needed += 1
            ck = annotation_content_key(
                rhash, entry.prompt_hash, registry.system_prompt_hash
            )
            qh = annotation_query_hash(ck, model_name, config_fingerprint)
            if qh in done_qh:
                n_covered += 1

    summary = {
        "legacy_path": legacy_annotations_path,
        "backup_path": str(backup_path),
        "out_path": out_path,
        "config_slug": config_slug,
        "n_legacy_rows": n_legacy,
        "n_bad_parse": n_bad_parse,
        "n_unresolved": n_unresolved,
        "n_unique_query_hashes_kept": len(best_by_qh),
        "n_fanout_rows": len(out_rows),
        "n_sample_class_pairs": n_needed,
        "n_covered_pairs": n_covered,
        "n_remaining_unique_approx": n_needed - n_covered,
        "dry_run": dry_run,
    }
    logger.info("%s", summary)

    if dry_run:
        return summary

    if legacy.exists() and not backup_path.exists():
        import shutil

        logger.info("Backing up legacy file to %s", backup_path)
        shutil.copy2(str(legacy), str(backup_path))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows to %s", len(out_rows), out_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--legacy_annotations_path", default=None)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens_per_request", type=int, default=2048)
    parser.add_argument("--config_extra", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    summary = compact_annotations(
        run_id=args.run_id,
        samples_path=args.samples_path,
        legacy_annotations_path=args.legacy_annotations_path,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens_per_request=args.max_tokens_per_request,
        config_extra=args.config_extra,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
