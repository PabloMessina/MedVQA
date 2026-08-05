"""Sample representative reports (K1/K2/K3) across CXR datasets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

from medvqa.datasets.cxr_report_labeling.class_registry import build_class_registry
from medvqa.datasets.cxr_report_labeling.complexity import score_reports
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_cache_run_dir,
    get_results_run_dir,
    get_srr_bert_cache_dir,
)
from medvqa.datasets.cxr_report_labeling.report_loaders import (
    DATASET_CHEXPERT_PLUS,
    DATASET_IUXRAY,
    DATASET_MIMICCXR,
    DATASET_REXGRADIENT,
    load_all_datasets,
)
from medvqa.datasets.cxr_report_labeling.schemas import dumps_jsonl_row
from medvqa.datasets.cxr_report_labeling.stratified_sampling import DEFAULT_K, sample_dataset
from medvqa.utils.files_utils import save_json
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _load_srr_vectors_if_available(
    reports: List[dict],
    cache_dir: str,
):
    """Load report-hash → labels from SRR cache if present; else return None."""
    import pickle

    from medvqa.datasets.cxr_report_labeling.schemas import report_text_hash
    from medvqa.datasets.cxr_report_labeling.srr_bert_leaves import (
        SRR_BERT_LEAVES_CLASS_NAMES,
    )

    path = os.path.join(cache_dir, "report_hash_to_labels.pkl")
    if not os.path.exists(path):
        logger.warning("SRR report cache not found at %s — sampling without SRR strata", path)
        return None, None
    with open(path, "rb") as f:
        report_hash_to_labels = pickle.load(f)
    vectors = []
    missing = 0
    for r in reports:
        h = report_text_hash(r["report_text"])
        if h in report_hash_to_labels:
            vectors.append(report_hash_to_labels[h])
        else:
            missing += 1
            vectors.append(None)
    if missing:
        logger.warning(
            "%d/%d reports missing from SRR cache; those contribute no SRR strata",
            missing,
            len(reports),
        )
        # Replace None with zero vector without No Finding? Better skip SRR for those indices
        # by using empty positives only for present ones — handled in sample via None filter
        import numpy as np
        from medvqa.datasets.cxr_report_labeling.srr_bert_leaves import NUM_LABELS, NO_FINDING_IDX

        filled = []
        for v in vectors:
            if v is None:
                z = np.zeros(NUM_LABELS, dtype=np.int8)
                # unknown — do not mark No Finding
                filled.append(z)
            else:
                filled.append(v)
        vectors = filled
    return vectors, SRR_BERT_LEAVES_CLASS_NAMES


def run_sampling(
    run_id: str = DEFAULT_RUN_ID,
    datasets: Optional[Sequence[str]] = None,
    seed: int = 0,
    num_regex_processes: int = 1,
    limits: Optional[Dict[str, Optional[int]]] = None,
    require_srr: bool = False,
) -> str:
    """
    Sample reports and write results/<run_id>/samples.jsonl.
    Returns path to samples.jsonl.
    """
    setup_logging()
    registry = build_class_registry()
    results_dir = get_results_run_dir(run_id)
    cache_dir = get_cache_run_dir(run_id)
    registry_path = os.path.join(results_dir, "class_registry.json")
    registry.save(registry_path)
    logger.info("Saved class registry to %s", registry_path)

    limits = limits or {}
    loaded = load_all_datasets(
        mimic_limit=limits.get(DATASET_MIMICCXR),
        chexpert_limit=limits.get(DATASET_CHEXPERT_PLUS),
        rex_limit_per_split=limits.get(DATASET_REXGRADIENT),
        iu_limit=limits.get(DATASET_IUXRAY),
        datasets=datasets,
    )

    srr_cache = get_srr_bert_cache_dir()  # shared class-agnostic cache
    all_samples: List[dict] = []
    for dataset, reports in loaded.items():
        k1, k2, k3 = DEFAULT_K[dataset]
        scores = score_reports(reports)
        # persist complexity for reuse
        score_path = os.path.join(cache_dir, f"complexity_scores_{dataset}.json")
        with open(score_path, "w", encoding="utf-8") as f:
            json.dump(
                {"uids": [r["uid"] for r in reports], "scores": scores},
                f,
            )
        srr_vectors, srr_names = _load_srr_vectors_if_available(reports, srr_cache)
        if require_srr and srr_vectors is None:
            raise FileNotFoundError(
                f"require_srr=True but no SRR cache at {srr_cache}. "
                "Run the SRR-BERT notebook section first."
            )
        samples = sample_dataset(
            reports,
            dataset=dataset,
            registry=registry,
            k1=k1,
            k2=k2,
            k3=k3,
            complexity_scores=scores,
            srr_label_vectors=srr_vectors,
            srr_class_names=srr_names,
            seed=seed,
            num_regex_processes=num_regex_processes,
        )
        all_samples.extend(samples)

    samples_path = os.path.join(results_dir, "samples.jsonl")
    with open(samples_path, "w", encoding="utf-8") as f:
        for rec in all_samples:
            f.write(dumps_jsonl_row(rec) + "\n")

    summary_path = os.path.join(results_dir, "samples_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uid", "dataset", "method", "complexity_score", "preview"],
        )
        writer.writeheader()
        for rec in all_samples:
            writer.writerow(
                {
                    "uid": rec["uid"],
                    "dataset": rec["dataset"],
                    "method": rec["sampling"]["method"],
                    "complexity_score": rec["sampling"]["complexity_score"],
                    "preview": rec["report_text"][:160],
                }
            )

    manifest = {
        "run_id": run_id,
        "n_samples": len(all_samples),
        "by_dataset": {
            ds: sum(1 for s in all_samples if s["dataset"] == ds) for ds in loaded
        },
        "samples_path": samples_path,
        "summary_path": summary_path,
        "registry_path": registry_path,
    }
    save_json(manifest, os.path.join(results_dir, "samples_manifest.json"))
    logger.info("Wrote %d samples to %s", len(all_samples), samples_path)
    return samples_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_regex_processes", type=int, default=1)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=[DATASET_MIMICCXR, DATASET_CHEXPERT_PLUS, DATASET_REXGRADIENT, DATASET_IUXRAY],
    )
    parser.add_argument("--require_srr", action="store_true")
    parser.add_argument("--iu_limit", type=int, default=None, help="Debug limit for IU")
    parser.add_argument("--mimic_limit", type=int, default=None)
    parser.add_argument("--chexpert_limit", type=int, default=None)
    parser.add_argument("--rex_limit_per_split", type=int, default=None)
    args = parser.parse_args()
    limits = {
        DATASET_IUXRAY: args.iu_limit,
        DATASET_MIMICCXR: args.mimic_limit,
        DATASET_CHEXPERT_PLUS: args.chexpert_limit,
        DATASET_REXGRADIENT: args.rex_limit_per_split,
    }
    run_sampling(
        run_id=args.run_id,
        datasets=args.datasets,
        seed=args.seed,
        num_regex_processes=args.num_regex_processes,
        limits=limits,
        require_srr=args.require_srr,
    )


if __name__ == "__main__":
    main()
