"""Run SRR-BERT-Leaves labeling over reports (intended for GPU notebook / CLI)."""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Sequence

from medvqa.datasets.cxr_report_labeling.paths import get_srr_bert_cache_dir
from medvqa.datasets.cxr_report_labeling.report_loaders import (
    DATASET_CHEXPERT_PLUS,
    DATASET_IUXRAY,
    DATASET_MIMICCXR,
    DATASET_REXGRADIENT,
    load_all_datasets,
)
from medvqa.datasets.cxr_report_labeling.schemas import report_text_hash
from medvqa.datasets.cxr_report_labeling.srr_bert_leaves import SRRBertLeavesLabeler
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def run_srr_bert_on_datasets(
    *,
    datasets: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    limits: Optional[dict] = None,
    save_every: int = 2000,
) -> dict:
    setup_logging()
    cache_dir = cache_dir or get_srr_bert_cache_dir()
    limits = limits or {}
    loaded = load_all_datasets(
        mimic_limit=limits.get(DATASET_MIMICCXR),
        chexpert_limit=limits.get(DATASET_CHEXPERT_PLUS),
        rex_limit_per_split=limits.get(DATASET_REXGRADIENT),
        iu_limit=limits.get(DATASET_IUXRAY),
        datasets=datasets,
    )
    labeler = SRRBertLeavesLabeler(
        device=device,
        default_batch_size=batch_size,
        cache_dir=cache_dir,
        verbose=True,
    )
    stats = {}
    for dataset, reports in loaded.items():
        texts = [r["report_text"] for r in reports]
        hashes = [report_text_hash(t) for t in texts]
        logger.info("Labeling %s (%d reports) with SRR-BERT-Leaves", dataset, len(texts))
        vectors = labeler.get_labels_for_reports(
            texts, report_hashes=hashes, batch_size=batch_size, save_every=save_every
        )
        n_pos = sum(int(v.sum() > 0) for v in vectors)
        stats[dataset] = {"n_reports": len(texts), "n_with_any_positive": n_pos}
        logger.info("%s done: %s", dataset, stats[dataset])
    labeler.save_caches()
    return {"cache_dir": cache_dir, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iu_limit", type=int, default=None)
    parser.add_argument("--mimic_limit", type=int, default=None)
    parser.add_argument("--chexpert_limit", type=int, default=None)
    parser.add_argument("--rex_limit_per_split", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=2000)
    args = parser.parse_args()
    run_srr_bert_on_datasets(
        datasets=args.datasets,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
        limits={
            DATASET_IUXRAY: args.iu_limit,
            DATASET_MIMICCXR: args.mimic_limit,
            DATASET_CHEXPERT_PLUS: args.chexpert_limit,
            DATASET_REXGRADIENT: args.rex_limit_per_split,
        },
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
