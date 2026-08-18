"""Regex-positive / no_finding LLM-verify job intents and inventory stats."""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import ClassRegistry
from medvqa.datasets.cxr_report_labeling.label_semantics import (
    NORMAL_CLASS,
    resolve_no_finding_from_other_labels,
)
from medvqa.datasets.cxr_report_labeling.ontology_dag import (
    OntologyDag,
    load_ontology_dag,
)
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_full_regex_matches_path,
    get_results_run_dir,
    resolve_sample_annotations_path,
    resolve_verify_annotations_path,
)
from medvqa.datasets.cxr_report_labeling.report_loaders import (
    ALL_DATASETS,
    DATASET_IUXRAY,
    load_reports_by_uid,
)
from medvqa.datasets.cxr_report_labeling.schemas import (
    annotation_content_key,
    annotation_query_hash,
    report_text_hash,
)
from medvqa.utils.files_utils import iter_jsonl, load_jsonl

logger = logging.getLogger(__name__)

JOB_SOURCE_SAMPLES = "samples"
JOB_SOURCE_REGEX_POSITIVES = "regex_positives"
JOB_SOURCE_NO_FINDING = "no_finding"
JOB_SOURCES = (
    JOB_SOURCE_SAMPLES,
    JOB_SOURCE_REGEX_POSITIVES,
    JOB_SOURCE_NO_FINDING,
)

_OK_LABELS = {
    "definitely true",
    "likely true",
    "unknown",
    "likely false",
    "definitely false",
}


def parsed_response_ok(parsed: object) -> bool:
    if not isinstance(parsed, dict):
        return False
    label = parsed.get("label")
    return isinstance(label, str) and label.strip().lower() in _OK_LABELS


def expand_candidate_classes(
    positive_classes: Iterable[str],
    dag: OntologyDag,
    *,
    skip_classes: Optional[Set[str]] = None,
) -> Set[str]:
    """Jsonl hits union DAG ancestors; drop ``no_finding`` (and any ``skip_classes``)."""
    skip = set(skip_classes) if skip_classes is not None else {NORMAL_CLASS}
    out: Set[str] = set()
    for cid in positive_classes:
        if not cid or cid in skip:
            continue
        out.add(cid)
        if cid in dag.class_ids:
            out.update(dag.ancestors(cid))
    return {c for c in out if c not in skip}


def load_ok_content_keys(paths: Optional[Sequence[str]]) -> Set[str]:
    skip: Set[str] = set()
    if not paths:
        return skip
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for item in load_jsonl(path):
            if not parsed_response_ok(item.get("parsed_response")):
                continue
            meta = item.get("metadata") or {}
            ck = meta.get("content_key")
            if ck:
                skip.add(ck)
    return skip


def load_failed_query_hashes(path: Optional[str]) -> Set[str]:
    out: Set[str] = set()
    if not path or not os.path.exists(path):
        return out
    for item in iter_jsonl(path):
        qh = item.get("query_hash") if isinstance(item, dict) else None
        if not qh and isinstance(item, dict):
            qh = item.get("custom_id")
        if isinstance(qh, str) and qh:
            out.add(qh)
    return out


def load_failed_query_hashes_union(paths: Sequence[Optional[str]]) -> Set[str]:
    out: Set[str] = set()
    seen: Set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        out.update(load_failed_query_hashes(path))
    return out


def append_failed_query_hashes(path: str, query_hashes: Sequence[str]) -> int:
    if not query_hashes:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for qh in query_hashes:
            f.write(json.dumps({"query_hash": qh, "ts": ts}, ensure_ascii=False) + "\n")
            n += 1
    return n


def matched_via_tags(class_id: str, jsonl_positives: Set[str], dag: OntologyDag) -> List[str]:
    tags: List[str] = []
    if class_id in jsonl_positives:
        tags.append("own")
    if class_id in dag.class_ids:
        if any(d in jsonl_positives for d in dag.descendants(class_id)):
            tags.append("descendant")
    return tags


def _class_filter_ids(
    registry: ClassRegistry,
    class_ids: Optional[Sequence[str]],
    *,
    exclude_no_finding: bool,
) -> List[str]:
    if class_ids is not None:
        ids = list(class_ids)
    elif exclude_no_finding:
        ids = list(registry.regex_backed_class_ids())
    else:
        ids = list(registry.class_ids())
    if exclude_no_finding:
        ids = [c for c in ids if c != NORMAL_CLASS]
    return ids


def iter_regex_positive_intents(
    *,
    dataset: str,
    matches: Sequence[dict],
    reports_by_uid: Dict[str, dict],
    dag: OntologyDag,
    registry: ClassRegistry,
    class_ids: Optional[Sequence[str]] = None,
) -> List[dict]:
    """One intent per ``(uid, candidate class)`` (jsonl hits ∪ DAG ancestors)."""
    allowed = set(_class_filter_ids(registry, class_ids, exclude_no_finding=True))
    intents: List[dict] = []
    missing_reports = 0
    for row in matches:
        uid = row["uid"]
        report = reports_by_uid.get(uid)
        if report is None:
            missing_reports += 1
            continue
        positives = [c for c in (row.get("positive_classes") or []) if c]
        candidates = expand_candidate_classes(positives, dag)
        rhash = report.get("report_text_hash") or report_text_hash(report["report_text"])
        for class_id in sorted(candidates):
            if class_id not in allowed:
                continue
            if class_id not in registry.by_id():
                continue
            intents.append(
                {
                    "uid": uid,
                    "dataset": dataset,
                    "class_id": class_id,
                    "report_text": report["report_text"],
                    "report_text_hash": rhash,
                    "jsonl_positives": positives,
                }
            )
    if missing_reports:
        logger.warning(
            "regex_positives: %d match rows had no report text for dataset=%s",
            missing_reports,
            dataset,
        )
    return intents


def compact_needs_no_finding_llm(labels: Dict[str, str]) -> bool:
    """True when no other class is definitely/likely true (queue ``no_finding``)."""
    return resolve_no_finding_from_other_labels(labels) is None


def iter_no_finding_intents(
    *,
    dataset: str,
    compact_rows: Sequence[dict],
    reports_by_uid: Dict[str, dict],
) -> Tuple[List[dict], dict]:
    """Queue ``no_finding`` NLI only for reports that are not auto-false."""
    intents: List[dict] = []
    n_auto = 0
    n_missing_report = 0
    for row in compact_rows:
        uid = row["uid"]
        labels = row.get("labels") or {}
        if not compact_needs_no_finding_llm(labels):
            n_auto += 1
            continue
        report = reports_by_uid.get(uid)
        if report is None:
            n_missing_report += 1
            continue
        rhash = report.get("report_text_hash") or report_text_hash(report["report_text"])
        intents.append(
            {
                "uid": uid,
                "dataset": dataset,
                "class_id": NORMAL_CLASS,
                "report_text": report["report_text"],
                "report_text_hash": rhash,
            }
        )
    stats = {
        "n_compact_rows": len(compact_rows),
        "n_auto_false": n_auto,
        "n_missing_report": n_missing_report,
        "n_no_finding_intents": len(intents),
    }
    return intents, stats


def remaining_content_keys_for_intents(
    intents: Sequence[dict],
    registry: ClassRegistry,
    *,
    skip_content_keys: Optional[Set[str]] = None,
) -> Set[str]:
    skip = skip_content_keys or set()
    by_id = registry.by_id()
    remaining: Set[str] = set()
    for intent in intents:
        entry = by_id.get(intent["class_id"])
        if entry is None:
            continue
        rhash = intent.get("report_text_hash") or report_text_hash(intent["report_text"])
        ck = annotation_content_key(rhash, entry.prompt_hash, registry.system_prompt_hash)
        if ck not in skip:
            remaining.add(ck)
    return remaining


def _per_class_counts(matches: Sequence[dict], *, skip_no_finding: bool = True) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in matches:
        for cid in row.get("positive_classes") or []:
            if skip_no_finding and cid == NORMAL_CLASS:
                continue
            if cid:
                counts[cid] += 1
    return dict(counts)


def summarize_dataset_regex_matches(
    *,
    dataset: str,
    run_id: str = DEFAULT_RUN_ID,
    registry: ClassRegistry,
    dag: OntologyDag,
    chunk_size: int = 2000,
    model_name: str = "gemini-2.5-flash-lite",
    skip_annotation_paths: Optional[Sequence[str]] = None,
    regex_matches_path: Optional[str] = None,
) -> dict:
    path = regex_matches_path or get_full_regex_matches_path(run_id, dataset)
    if not os.path.exists(path):
        return {
            "dataset": dataset,
            "regex_matches_path": path,
            "missing": True,
        }
    matches = load_jsonl(path)
    n_reports = len(matches)
    n_empty = sum(1 for r in matches if not (r.get("positive_classes") or []))
    n_pos = 0
    for r in matches:
        for cid in r.get("positive_classes") or []:
            if cid and cid != NORMAL_CLASS:
                n_pos += 1
    density = (n_pos / n_reports) if n_reports else 0.0
    empty_pct = (100.0 * n_empty / n_reports) if n_reports else 0.0

    reports_by_uid = load_reports_by_uid(dataset)
    intents = iter_regex_positive_intents(
        dataset=dataset,
        matches=matches,
        reports_by_uid=reports_by_uid,
        dag=dag,
        registry=registry,
    )

    skip_paths: List[str] = list(skip_annotation_paths or [])
    sample_path = resolve_sample_annotations_path(run_id, model_name)
    if sample_path not in skip_paths:
        skip_paths.append(sample_path)
    verify_path = resolve_verify_annotations_path(run_id, model_name, dataset)
    if os.path.isfile(verify_path) and verify_path not in skip_paths:
        skip_paths.append(verify_path)
    skip_keys = load_ok_content_keys(skip_paths)
    remaining = remaining_content_keys_for_intents(
        intents, registry, skip_content_keys=skip_keys
    )
    n_remaining = len(remaining)
    n_chunks = int(math.ceil(n_remaining / chunk_size)) if chunk_size > 0 else 0

    return {
        "dataset": dataset,
        "regex_matches_path": path,
        "missing": False,
        "n_reports": n_reports,
        "n_pos": n_pos,
        "density": round(density, 4),
        "n_empty": n_empty,
        "empty_pct": round(empty_pct, 2),
        "n_candidate_pairs": len(intents),
        "n_skip_content_keys": len(skip_keys),
        "n_remaining_content_keys": n_remaining,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "per_class": _per_class_counts(matches),
        "sample_annotations_path": sample_path if os.path.isfile(sample_path) else None,
        "verify_annotations_path": verify_path if os.path.isfile(verify_path) else None,
    }


def summarize_regex_matches(
    *,
    datasets: Optional[Sequence[str]] = None,
    run_id: str = DEFAULT_RUN_ID,
    chunk_size: int = 2000,
    model_name: str = "gemini-2.5-flash-lite",
    skip_annotation_paths: Optional[Sequence[str]] = None,
    registry: Optional[ClassRegistry] = None,
    dag: Optional[OntologyDag] = None,
) -> dict:
    from medvqa.datasets.cxr_report_labeling.class_registry import (
        ClassRegistry as _CR,
    )
    from medvqa.datasets.cxr_report_labeling.class_registry import (
        build_class_registry,
    )

    ds_list = list(datasets) if datasets else list(ALL_DATASETS)
    # IU first even if the caller passed a custom order that puts it later.
    if DATASET_IUXRAY in ds_list:
        ds_list = [DATASET_IUXRAY] + [d for d in ds_list if d != DATASET_IUXRAY]
    results_dir = get_results_run_dir(run_id)
    registry_path = os.path.join(results_dir, "class_registry.json")
    if registry is None:
        registry = _CR.load(registry_path) if os.path.exists(registry_path) else build_class_registry()
    if dag is None:
        dag = load_ontology_dag()

    per_ds = []
    for dataset in ds_list:
        per_ds.append(
            summarize_dataset_regex_matches(
                dataset=dataset,
                run_id=run_id,
                registry=registry,
                dag=dag,
                chunk_size=chunk_size,
                model_name=model_name,
                skip_annotation_paths=skip_annotation_paths,
            )
        )

    present = [d for d in per_ds if not d.get("missing")]
    total = {
        "n_reports": sum(d["n_reports"] for d in present),
        "n_pos": sum(d["n_pos"] for d in present),
        "n_empty": sum(d["n_empty"] for d in present),
        "n_candidate_pairs": sum(d["n_candidate_pairs"] for d in present),
        "n_remaining_content_keys": sum(d["n_remaining_content_keys"] for d in present),
    }
    if total["n_reports"]:
        total["density"] = round(total["n_pos"] / total["n_reports"], 4)
        total["empty_pct"] = round(100.0 * total["n_empty"] / total["n_reports"], 2)
    else:
        total["density"] = 0.0
        total["empty_pct"] = 0.0
    total["chunk_size"] = chunk_size
    total["n_chunks"] = (
        int(math.ceil(total["n_remaining_content_keys"] / chunk_size)) if chunk_size > 0 else 0
    )
    return {
        "run_id": run_id,
        "model_name": model_name,
        "datasets": per_ds,
        "total": total,
    }


def annotation_keys_for_intent(
    intent: dict,
    registry: ClassRegistry,
    model_name: str,
    config_fingerprint: str,
) -> Tuple[str, str]:
    entry = registry.by_id()[intent["class_id"]]
    rhash = intent.get("report_text_hash") or report_text_hash(intent["report_text"])
    content_key = annotation_content_key(
        rhash, entry.prompt_hash, registry.system_prompt_hash
    )
    query_hash = annotation_query_hash(content_key, model_name, config_fingerprint)
    return content_key, query_hash
