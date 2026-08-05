"""Unified report loaders for MIMIC-CXR, CheXpert Plus, ReXGradient-160K, IU X-ray."""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from medvqa.datasets.cxr_report_labeling.schemas import make_uid, normalize_report_text
from medvqa.settings import (
    CHEXPERT_PLUS_CSV_PATH,
    IUXRAY_REPORTS_MIN_JSON_PATH,
    REXGRADIENT_160K_TEST_METADATA_CSV_PATH,
    REXGRADIENT_160K_TRAIN_METADATA_CSV_PATH,
    REXGRADIENT_160K_VAL_METADATA_CSV_PATH,
)
from medvqa.utils.files_utils import load_json

logger = logging.getLogger(__name__)

DATASET_MIMICCXR = "mimiccxr"
DATASET_CHEXPERT_PLUS = "chexpert_plus"
DATASET_REXGRADIENT = "rexgradient"
DATASET_IUXRAY = "iuxray"

_STUDY_ID_FROM_PATH = re.compile(r"/s(\d+)\.txt$", re.IGNORECASE)


def _base_record(
    dataset: str,
    report_id: str,
    report_text: str,
    source_refs: dict,
    split: Optional[str] = None,
) -> dict:
    text = normalize_report_text(report_text)
    rec = {
        "uid": make_uid(dataset, report_id),
        "dataset": dataset,
        "report_id": str(report_id),
        "report_text": text,
        "source_refs": source_refs,
    }
    if split is not None:
        rec["split"] = split
    return rec


def load_mimiccxr_reports(limit: Optional[int] = None) -> List[dict]:
    from medvqa.datasets.mimiccxr import get_path_to_report_text_dict

    path_to_text = get_path_to_report_text_dict()
    records: List[dict] = []
    for path, text in path_to_text.items():
        m = _STUDY_ID_FROM_PATH.search(path.replace("\\", "/"))
        if m:
            report_id = m.group(1)
            study_id = int(report_id)
        else:
            basename = os.path.basename(path)
            report_id = basename.split(".")[0].lstrip("s")
            study_id = int(report_id) if report_id.isdigit() else None
        rec = _base_record(
            DATASET_MIMICCXR,
            report_id,
            text,
            source_refs={"report_path_key": path, "study_id": study_id},
        )
        if study_id is not None:
            rec["study_id"] = study_id
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    logger.info("Loaded %d MIMIC-CXR reports", len(records))
    return records


def load_chexpert_plus_reports(limit: Optional[int] = None) -> List[dict]:
    usecols = [
        "path_to_image",
        "deid_patient_id",
        "patient_report_date_order",
        "report",
        "split",
    ]
    df = pd.read_csv(CHEXPERT_PLUS_CSV_PATH, usecols=usecols)
    records: List[dict] = []
    for row in df.itertuples(index=False):
        report_id = f"{row.deid_patient_id}_{row.patient_report_date_order}"
        rec = _base_record(
            DATASET_CHEXPERT_PLUS,
            report_id,
            row.report,
            source_refs={
                "deid_patient_id": row.deid_patient_id,
                "patient_report_date_order": int(row.patient_report_date_order),
                "path_to_image": row.path_to_image,
            },
            split=str(row.split) if pd.notna(row.split) else None,
        )
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    logger.info("Loaded %d CheXpert Plus reports", len(records))
    return records


def _load_rexgradient_split(csv_path: str, split: str, limit: Optional[int]) -> List[dict]:
    df = pd.read_csv(csv_path)
    records: List[dict] = []
    for row in df.itertuples(index=False):
        findings = "" if pd.isna(row.Findings) else str(row.Findings)
        impression = "" if pd.isna(row.Impression) else str(row.Impression)
        text = normalize_report_text(f"{findings} {impression}")
        report_id = str(row.id)
        rec = _base_record(
            DATASET_REXGRADIENT,
            report_id,
            text,
            source_refs={
                "id": report_id,
                "AccessionNumber": getattr(row, "AccessionNumber", None),
                "StudyInstanceUid": getattr(row, "StudyInstanceUid", None),
            },
            split=split,
        )
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    return records


def load_rexgradient_reports(
    splits: Sequence[str] = ("train", "val", "test"),
    limit_per_split: Optional[int] = None,
) -> List[dict]:
    split_to_path = {
        "train": REXGRADIENT_160K_TRAIN_METADATA_CSV_PATH,
        "val": REXGRADIENT_160K_VAL_METADATA_CSV_PATH,
        "valid": REXGRADIENT_160K_VAL_METADATA_CSV_PATH,
        "test": REXGRADIENT_160K_TEST_METADATA_CSV_PATH,
    }
    records: List[dict] = []
    for split in splits:
        key = "val" if split == "valid" else split
        path = split_to_path[key]
        part = _load_rexgradient_split(path, key, limit_per_split)
        records.extend(part)
        logger.info("Loaded %d ReXGradient reports from %s", len(part), key)
    return records


def load_iuxray_reports(limit: Optional[int] = None) -> List[dict]:
    data = load_json(IUXRAY_REPORTS_MIN_JSON_PATH)
    records: List[dict] = []
    for filename, obj in data.items():
        findings = obj.get("findings") or ""
        impression = obj.get("impression") or ""
        text = normalize_report_text(f"{findings} {impression}")
        report_id = filename
        image_ids = [im.get("id") for im in (obj.get("images") or []) if isinstance(im, dict)]
        rec = _base_record(
            DATASET_IUXRAY,
            report_id,
            text,
            source_refs={"filename": filename, "image_ids": image_ids},
        )
        records.append(rec)
        if limit is not None and len(records) >= limit:
            break
    logger.info("Loaded %d IU X-ray reports", len(records))
    return records


def load_all_datasets(
    *,
    mimic_limit: Optional[int] = None,
    chexpert_limit: Optional[int] = None,
    rex_limit_per_split: Optional[int] = None,
    iu_limit: Optional[int] = None,
    datasets: Optional[Iterable[str]] = None,
) -> Dict[str, List[dict]]:
    wanted = set(datasets) if datasets is not None else {
        DATASET_MIMICCXR,
        DATASET_CHEXPERT_PLUS,
        DATASET_REXGRADIENT,
        DATASET_IUXRAY,
    }
    out: Dict[str, List[dict]] = {}
    if DATASET_MIMICCXR in wanted:
        out[DATASET_MIMICCXR] = load_mimiccxr_reports(limit=mimic_limit)
    if DATASET_CHEXPERT_PLUS in wanted:
        out[DATASET_CHEXPERT_PLUS] = load_chexpert_plus_reports(limit=chexpert_limit)
    if DATASET_REXGRADIENT in wanted:
        out[DATASET_REXGRADIENT] = load_rexgradient_reports(limit_per_split=rex_limit_per_split)
    if DATASET_IUXRAY in wanted:
        out[DATASET_IUXRAY] = load_iuxray_reports(limit=iu_limit)
    return out
