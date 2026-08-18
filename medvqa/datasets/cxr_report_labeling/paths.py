"""Cache / results layout for the CXR report labeling pipeline."""

from __future__ import annotations

import os
from typing import Optional

from medvqa.settings import CACHE_DIR, RESULTS_DIR

PIPELINE_SUBDIR = "cxr_report_labeling"
DEFAULT_RUN_ID = "v1_k1200_iu400"


def get_cache_run_dir(run_id: str = DEFAULT_RUN_ID) -> str:
    path = os.path.join(CACHE_DIR, PIPELINE_SUBDIR, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_results_run_dir(run_id: str = DEFAULT_RUN_ID) -> str:
    path = os.path.join(RESULTS_DIR, PIPELINE_SUBDIR, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_srr_bert_cache_dir(run_id: Optional[str] = None) -> str:
    """Shared SRR-BERT label cache (class-agnostic; optionally under a run)."""
    if run_id is None:
        path = os.path.join(CACHE_DIR, PIPELINE_SUBDIR, "srr_bert_leaves")
    else:
        path = os.path.join(get_cache_run_dir(run_id), "srr_bert_leaves")
    os.makedirs(path, exist_ok=True)
    return path


def get_llm_annotations_dir(
    run_id: str,
    model_name: str,
    config_slug: str,
) -> str:
    """
    ``cache/.../llm_annotations/<safe_model>/<config_slug>/``

    One directory per (model, system-prompt, decoding-config).
    """
    safe_model = model_name.replace("/", "_")
    path = os.path.join(
        get_cache_run_dir(run_id), "llm_annotations", safe_model, config_slug
    )
    os.makedirs(path, exist_ok=True)
    return path


def get_sample_annotations_path(
    run_id: str,
    model_name: str,
    config_slug: str,
) -> str:
    return os.path.join(
        get_llm_annotations_dir(run_id, model_name, config_slug),
        "sample_annotations.jsonl",
    )


def resolve_sample_annotations_path(
    run_id: str,
    model_name: str,
    *,
    config_slug: Optional[str] = None,
    prefer_legacy: bool = True,
) -> str:
    """
    Resolve annotations jsonl for a model.

    If ``config_slug`` is given, return that namespaced path.
    Else prefer an existing namespaced ``*/sample_annotations.jsonl`` (newest mtime),
    falling back to legacy flat ``llm_annotations/<model>/sample_annotations.jsonl``.
    """
    safe_model = model_name.replace("/", "_")
    model_dir = os.path.join(get_cache_run_dir(run_id), "llm_annotations", safe_model)
    if config_slug:
        return get_sample_annotations_path(run_id, model_name, config_slug)

    candidates: list[str] = []
    if os.path.isdir(model_dir):
        for name in os.listdir(model_dir):
            sub = os.path.join(model_dir, name)
            if os.path.isdir(sub):
                p = os.path.join(sub, "sample_annotations.jsonl")
                if os.path.isfile(p):
                    candidates.append(p)
    legacy = os.path.join(model_dir, "sample_annotations.jsonl")
    if prefer_legacy and os.path.isfile(legacy):
        # Only use legacy if no namespaced files exist yet.
        if not candidates:
            return legacy
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    # Default path even if missing (callers create on write)
    if config_slug is None:
        return legacy
    return get_sample_annotations_path(run_id, model_name, config_slug)


def get_full_regex_matches_path(run_id: str, dataset: str) -> str:
    path = os.path.join(get_cache_run_dir(run_id), "full_regex_matches")
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"{dataset}.jsonl")


def get_verify_annotations_path(
    run_id: str,
    model_name: str,
    config_slug: str,
    dataset: str,
    *,
    no_finding: bool = False,
) -> str:
    name = (
        f"verify_no_finding_{dataset}.jsonl"
        if no_finding
        else f"verify_{dataset}.jsonl"
    )
    return os.path.join(
        get_llm_annotations_dir(run_id, model_name, config_slug),
        name,
    )


def get_legacy_failed_query_hashes_path(
    run_id: str,
    model_name: str,
    config_slug: str,
) -> str:
    """Unscoped sidecar used by parallel loops before per-dataset split (read-only)."""
    return os.path.join(
        get_llm_annotations_dir(run_id, model_name, config_slug),
        "failed_query_hashes.jsonl",
    )


def get_failed_query_hashes_path(
    run_id: str,
    model_name: str,
    config_slug: str,
    dataset: Optional[str] = None,
    *,
    job_source: Optional[str] = None,
) -> str:
    """Failed ``query_hash`` sidecar for this config + dataset/job_source (write target)."""
    if job_source == "no_finding" and dataset:
        name = f"failed_query_hashes_no_finding_{dataset}.jsonl"
    elif dataset:
        name = f"failed_query_hashes_{dataset}.jsonl"
    else:
        name = "failed_query_hashes_samples.jsonl"
    return os.path.join(
        get_llm_annotations_dir(run_id, model_name, config_slug),
        name,
    )


def get_final_labels_dir(run_id: str = DEFAULT_RUN_ID) -> str:
    path = os.path.join(get_results_run_dir(run_id), "final_labels")
    os.makedirs(path, exist_ok=True)
    return path


def resolve_verify_annotations_path(
    run_id: str,
    model_name: str,
    dataset: str,
    *,
    config_slug: Optional[str] = None,
    no_finding: bool = False,
) -> str:
    """Resolve ``verify_*.jsonl``; prefer ``config_slug``, else newest mtime."""
    name = (
        f"verify_no_finding_{dataset}.jsonl"
        if no_finding
        else f"verify_{dataset}.jsonl"
    )
    if config_slug:
        return get_verify_annotations_path(
            run_id, model_name, config_slug, dataset, no_finding=no_finding
        )
    safe_model = model_name.replace("/", "_")
    model_dir = os.path.join(get_cache_run_dir(run_id), "llm_annotations", safe_model)
    candidates: list[str] = []
    if os.path.isdir(model_dir):
        for sub_name in os.listdir(model_dir):
            sub = os.path.join(model_dir, sub_name)
            if os.path.isdir(sub):
                p = os.path.join(sub, name)
                if os.path.isfile(p):
                    candidates.append(p)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    # Default even if missing
    return os.path.join(model_dir, name)
