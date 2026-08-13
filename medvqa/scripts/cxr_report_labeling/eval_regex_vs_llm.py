"""Evaluate regexes against LLM sample annotations; dump FP/FN examples.

Matching reuses :func:`batch_match_prompt_classes` (same multiprocessing path as
``apply_regex_to_dataset`` / full-dataset precompute). Use ``--lightweight`` for
metrics-only scans (no per-class FP/FN jsonl dumps).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    build_class_registry,
)
from medvqa.datasets.cxr_report_labeling.label_semantics import (
    llm_label_to_binary_positive,
)
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_cache_run_dir,
    get_results_run_dir,
    resolve_sample_annotations_path,
)
from medvqa.datasets.cxr_report_labeling.regex_matching import (
    batch_match_prompt_classes,
    report_matches_prompt_class,
)
from medvqa.datasets.regular_expressions import cxr_patterns
from medvqa.settings import PROJECT_ROOT, REGULAR_EXPRESSIONS_FOLDER
from medvqa.utils.files_utils import load_jsonl, save_json
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


# Connected components among prompt classes sharing regex hierarchy (from plan)
REFINEMENT_GROUPS: List[List[str]] = [
    [
        "Atelectasis", "Bulla", "Consolidation", "Edema", "Granuloma", "Infiltration",
        "Interstitial Lung Disease", "Lobar Atelectasis", "Lung Cavity", "Lung Cyst",
        "Lung Lesion", "Lung Opacity", "Lung Tumor", "Mass", "Nodule", "Pneumonia",
        "Pulmonary Fibrosis", "Rounded Atelectasis",
    ],
    ["Aortic Calcification", "Calcification", "Pleural Calcification", "Pleural Thickening"],
    ["Bone Fracture", "Clavicle Fracture", "Rib Fracture"],
    ["Aortic Enlargement", "Aortic Tortuosity"],
    ["Cardiomegaly", "Enlarged Cardiomediastinum"],
    ["Fissural Effusion", "Pleural Effusion"],
    ["Hydropneumothorax", "Pneumothorax"],
]


def backup_regex_classes(dest_dir: Optional[str] = None) -> str:
    src = os.path.join(REGULAR_EXPRESSIONS_FOLDER, "cxr_classes")
    if dest_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = os.path.join(str(PROJECT_ROOT), ".agent", f"cxr_classes_backup_{stamp}")
    if os.path.exists(dest_dir):
        raise FileExistsError(dest_dir)
    shutil.copytree(src, dest_dir)
    logger.info("Backed up regex classes to %s", dest_dir)
    return dest_dir


def clear_pattern_cache() -> None:
    cxr_patterns.clear_pattern_cache()


def _index_annotations(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    """(uid, class_id) -> best annotation row (parsed)."""
    out: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        meta = row.get("metadata") or {}
        uid = meta.get("uid")
        class_id = meta.get("class_id")
        parsed = row.get("parsed_response") or row.get("parsed_output") or row
        # orchestrate_api_calls structure may nest differently
        if "label" not in parsed and isinstance(row.get("parsed_response"), dict):
            parsed = row["parsed_response"]
        if "label" not in parsed and "response" in row:
            # try common shapes
            resp = row["response"]
            if isinstance(resp, dict) and "label" in resp:
                parsed = resp
        if uid and class_id and isinstance(parsed, dict) and "label" in parsed:
            out[(uid, class_id)] = {
                "label": parsed["label"],
                "relevant_quote": parsed.get("relevant_quote"),
                "reasoning": parsed.get("reasoning"),
                "uses_hedging_language": parsed.get("uses_hedging_language"),
                "raw": row,
            }
    return out


def load_annotations_flexible(path: str) -> Dict[Tuple[str, str], dict]:
    rows = load_jsonl(path)
    indexed = _index_annotations(rows)
    if indexed:
        return indexed
    # Fallback: some saves put parsed fields at top level after enrichment
    out = {}
    for row in rows:
        meta = row.get("metadata") or {}
        uid, class_id = meta.get("uid"), meta.get("class_id")
        label = None
        if "parsed_response" in row and isinstance(row["parsed_response"], dict):
            label = row["parsed_response"].get("label")
            parsed = row["parsed_response"]
        elif "label" in row:
            label = row["label"]
            parsed = row
        else:
            continue
        if uid and class_id and label:
            out[(uid, class_id)] = {
                "label": label,
                "relevant_quote": parsed.get("relevant_quote"),
                "reasoning": parsed.get("reasoning"),
                "uses_hedging_language": parsed.get("uses_hedging_language"),
                "raw": row,
            }
    return out


def _metrics_entry(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_annotated": tp + fp + fn + tn,
        "n_positive": tp + fn,
    }


def evaluate_regex_vs_llm(
    *,
    run_id: str = DEFAULT_RUN_ID,
    samples_path: Optional[str] = None,
    annotations_path: str,
    class_ids: Optional[Sequence[str]] = None,
    backup: bool = False,
    num_processes: int = 1,
    lightweight: bool = False,
    out_dir: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-lite",
) -> dict:
    setup_logging()
    clear_pattern_cache()
    if backup:
        backup_regex_classes()

    results_dir = get_results_run_dir(run_id)
    cache_dir = get_cache_run_dir(run_id)
    eval_dir = out_dir or os.path.join(cache_dir, "regex_eval")
    os.makedirs(eval_dir, exist_ok=True)

    if samples_path is None:
        samples_path = os.path.join(results_dir, "samples.jsonl")
    sample_rows = load_jsonl(samples_path)
    uids = [s["uid"] for s in sample_rows]
    texts = [s["report_text"] for s in sample_rows]
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}
    samples_by_uid = {s["uid"]: s for s in sample_rows}

    registry_path = os.path.join(results_dir, "class_registry.json")
    registry = ClassRegistry.load(registry_path) if os.path.exists(registry_path) else build_class_registry()

    ann = load_annotations_flexible(annotations_path)
    ids = list(class_ids) if class_ids is not None else [
        c for c in registry.class_ids() if registry.by_id()[c].has_regex
    ]

    logger.info(
        "Matching %d classes × %d samples (num_processes=%d, lightweight=%s)",
        len(ids),
        len(texts),
        num_processes,
        lightweight,
    )
    # Same batch + Pool path as apply_regex_to_dataset / full_regex_matches precompute.
    matched_by_class = batch_match_prompt_classes(
        texts,
        registry,
        class_ids=ids,
        num_processes=num_processes,
    )

    metrics = {}
    for class_id in ids:
        matched_idxs = matched_by_class.get(class_id) or set()
        tp = fp = fn = tn = 0
        fp_uids: List[str] = []
        fn_uids: List[str] = []
        for uid, idx in uid_to_idx.items():
            key = (uid, class_id)
            if key not in ann:
                continue
            gold_pos = llm_label_to_binary_positive(ann[key]["label"])
            matched = idx in matched_idxs
            if matched and gold_pos:
                tp += 1
            elif matched and not gold_pos:
                fp += 1
                if not lightweight:
                    fp_uids.append(uid)
            elif (not matched) and gold_pos:
                fn += 1
                if not lightweight:
                    fn_uids.append(uid)
            else:
                tn += 1
        metrics[class_id] = _metrics_entry(tp, fp, fn, tn)

        if lightweight:
            continue

        stem = class_id.lower().replace("/", "_").replace(" ", "_")
        fp_path = os.path.join(eval_dir, f"{stem}_fp.jsonl")
        fn_path = os.path.join(eval_dir, f"{stem}_fn.jsonl")
        with open(fp_path, "w", encoding="utf-8") as f:
            for uid in fp_uids:
                _, spans = report_matches_prompt_class(
                    samples_by_uid[uid]["report_text"], class_id, registry
                )
                f.write(
                    json.dumps(
                        {
                            "uid": uid,
                            "class_id": class_id,
                            "report_text": samples_by_uid[uid]["report_text"],
                            "llm": ann[(uid, class_id)],
                            "regex_spans": spans,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        with open(fn_path, "w", encoding="utf-8") as f:
            for uid in fn_uids:
                f.write(
                    json.dumps(
                        {
                            "uid": uid,
                            "class_id": class_id,
                            "report_text": samples_by_uid[uid]["report_text"],
                            "llm": ann[(uid, class_id)],
                            "regex_spans": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    out_path = os.path.join(eval_dir, "metrics_per_class.json")
    save_json(metrics, out_path)
    groups_path = os.path.join(eval_dir, "refinement_groups.json")
    save_json(REFINEMENT_GROUPS, groups_path)
    meta_path = os.path.join(eval_dir, "eval_meta.json")
    save_json(
        {
            "run_id": run_id,
            "model_name": model_name,
            "samples_path": os.path.abspath(samples_path),
            "annotations_path": os.path.abspath(annotations_path),
            "n_samples": len(texts),
            "n_annotation_keys": len(ann),
            "n_classes": len(ids),
            "num_processes": num_processes,
            "lightweight": lightweight,
            "class_ids": ids,
        },
        meta_path,
    )
    logger.info("Wrote metrics to %s (lightweight=%s)", out_path, lightweight)
    return {"metrics": metrics, "eval_dir": eval_dir, "refinement_groups": REFINEMENT_GROUPS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument(
        "--annotations_path",
        default=None,
        help=(
            "LLM sample annotations jsonl. Default: resolve namespaced path for "
            "--model_name under the run cache."
        ),
    )
    parser.add_argument("--model_name", default="gemini-2.5-flash-lite")
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--backup", action="store_true")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Workers for batch_match_prompt_classes (same as apply_regex_to_dataset).",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Write metrics_per_class.json only (skip per-class FP/FN jsonl dumps).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Override eval output dir (default: cache/.../regex_eval).",
    )
    args = parser.parse_args()
    annotations_path = args.annotations_path
    if annotations_path is None:
        annotations_path = resolve_sample_annotations_path(args.run_id, args.model_name)
    evaluate_regex_vs_llm(
        run_id=args.run_id,
        samples_path=args.samples_path,
        annotations_path=annotations_path,
        class_ids=args.classes,
        backup=args.backup,
        num_processes=args.num_processes,
        lightweight=args.lightweight,
        out_dir=args.out_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
