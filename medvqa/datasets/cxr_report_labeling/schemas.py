"""Record schemas and helpers for CXR report labeling artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def normalize_report_text(text: Optional[str]) -> str:
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
    payload = "|".join([uid, class_id, prompt_hash, system_prompt_hash, model_name])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_record(
    *,
    uid: str,
    dataset: str,
    report_id: str,
    report_text: str,
    source_refs: Dict[str, Any],
    sampling_method: str,
    complexity_score: float,
    split: Optional[str] = None,
    strata_labels: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
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


def dumps_jsonl_row(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def build_nli_query(report_text: str, hypothesis: str) -> str:
    report = normalize_report_text(report_text)
    hyp = hypothesis.strip()
    return f"<report>{report}</report>\n<hypothesis>{hyp}</hypothesis>"
