"""Record schemas and helpers for CXR report labeling artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def normalize_report_text(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split()).strip()


def report_text_hash(report_text: str) -> str:
    return hashlib.sha256(normalize_report_text(report_text).encode("utf-8")).hexdigest()


def make_uid(dataset: str, report_id: str) -> str:
    return f"{dataset}:{report_id}"


def annotation_resume_key(
    uid: str,
    class_id: str,
    prompt_hash: str,
    system_prompt_hash: str,
    model_name: str,
) -> str:
    """Legacy uid-scoped row id. Prefer content_key / query_hash for API dedupe."""
    payload = f"{uid}|{class_id}|{prompt_hash}|{system_prompt_hash}|{model_name}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def annotation_content_key(
    report_text_hash_value: str,
    prompt_hash: str,
    system_prompt_hash: str,
) -> str:
    """Model-agnostic identity of (report, class hypothesis, system instructions)."""
    payload = f"{report_text_hash_value}|{prompt_hash}|{system_prompt_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def annotation_config_fingerprint(
    *,
    temperature: float = 0.0,
    max_tokens_per_request: int = 2048,
    extra: str | None = None,
) -> str:
    """Canonical decoding-config string embedded in query_hash / config_slug."""
    # Normalize temperature so 0 and 0.0 match.
    temp_s = f"{float(temperature):g}"
    parts = [f"t{temp_s}", f"mt{int(max_tokens_per_request)}"]
    if extra:
        parts.append(str(extra).strip().replace("/", "_").replace(" ", "_"))
    return "_".join(parts)


def annotation_config_slug(
    system_prompt_hash: str,
    *,
    temperature: float = 0.0,
    max_tokens_per_request: int = 2048,
    extra: str | None = None,
) -> str:
    """Directory name under llm_annotations/<model>/."""
    fp = annotation_config_fingerprint(
        temperature=temperature,
        max_tokens_per_request=max_tokens_per_request,
        extra=extra,
    )
    return f"sys_{system_prompt_hash[:8]}_{fp}"


def annotation_query_hash(
    content_key: str,
    model_name: str,
    config_fingerprint: str,
) -> str:
    """Inference-cache key: content + model + decoding config."""
    payload = f"{content_key}|{model_name}|{config_fingerprint}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_record(
    *,
    uid: str,
    dataset: str,
    report_id: str,
    report_text: str,
    source_refs: dict[str, Any],
    sampling_method: str,
    complexity_score: float,
    split: str | None = None,
    strata_labels: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "uid": uid,
        "dataset": dataset,
        "report_id": report_id,
        "report_text": normalize_report_text(report_text),
        "report_text_hash": report_text_hash(report_text),
        "source_refs": source_refs or {},
        "sampling": {
            "method": sampling_method,
            "complexity_score": float(complexity_score),
            "strata_labels": strata_labels or {},
        },
    }
    if split is not None:
        rec["split"] = split
    if extra:
        rec.update(extra)
    return rec


def dumps_jsonl_row(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def build_nli_query(report_text: str, hypothesis: str) -> str:
    report = normalize_report_text(report_text)
    hyp = hypothesis.strip()
    return f"<report>{report}</report>\n<hypothesis>{hyp}</hypothesis>"
