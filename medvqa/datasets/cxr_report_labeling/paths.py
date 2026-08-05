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
