"""Evaluate regexes against LLM sample annotations; dump FP/FN examples."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import ClassRegistry, build_class_registry
from medvqa.datasets.cxr_report_labeling.label_semantics import llm_label_to_binary_positive
from medvqa.datasets.cxr_report_labeling.paths import DEFAULT_RUN_ID, get_cache_run_dir, get_results_run_dir
from medvqa.datasets.cxr_report_labeling.regex_matching import report_matches_prompt_class
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
    cxr_patterns._PATTERN_CACHE.clear()


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


def evaluate_regex_vs_llm(
    *,
    run_id: str = DEFAULT_RUN_ID,
    samples_path: Optional[str] = None,
    annotations_path: str,
    class_ids: Optional[Sequence[str]] = None,
    backup: bool = False,
) -> dict:
    setup_logging()
    clear_pattern_cache()
    if backup:
        backup_regex_classes()

    results_dir = get_results_run_dir(run_id)
    cache_dir = get_cache_run_dir(run_id)
    eval_dir = os.path.join(cache_dir, "regex_eval")
    os.makedirs(eval_dir, exist_ok=True)

    if samples_path is None:
        samples_path = os.path.join(results_dir, "samples.jsonl")
    samples = {s["uid"]: s for s in load_jsonl(samples_path)}
    registry_path = os.path.join(results_dir, "class_registry.json")
    registry = ClassRegistry.load(registry_path) if os.path.exists(registry_path) else build_class_registry()

    ann = load_annotations_flexible(annotations_path)
    ids = list(class_ids) if class_ids is not None else [
        c for c in registry.class_ids() if registry.by_id()[c].has_regex
    ]

    metrics = {}
    for class_id in ids:
        tp = fp = fn = tn = 0
        fp_rows = []
        fn_rows = []
        for uid, sample in samples.items():
            key = (uid, class_id)
            if key not in ann:
                continue
            gold_pos = llm_label_to_binary_positive(ann[key]["label"])
            matched, spans = report_matches_prompt_class(sample["report_text"], class_id, registry)
            if matched and gold_pos:
                tp += 1
            elif matched and not gold_pos:
                fp += 1
                fp_rows.append(
                    {
                        "uid": uid,
                        "class_id": class_id,
                        "report_text": sample["report_text"],
                        "llm": ann[key],
                        "regex_spans": spans,
                    }
                )
            elif (not matched) and gold_pos:
                fn += 1
                fn_rows.append(
                    {
                        "uid": uid,
                        "class_id": class_id,
                        "report_text": sample["report_text"],
                        "llm": ann[key],
                        "regex_spans": spans,
                    }
                )
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[class_id] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "n_annotated": tp + fp + fn + tn,
        }
        stem = class_id.lower().replace("/", "_").replace(" ", "_")
        with open(os.path.join(eval_dir, f"{stem}_fp.jsonl"), "w", encoding="utf-8") as f:
            for row in fp_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(eval_dir, f"{stem}_fn.jsonl"), "w", encoding="utf-8") as f:
            for row in fn_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_path = os.path.join(eval_dir, "metrics_per_class.json")
    save_json(metrics, out_path)
    groups_path = os.path.join(eval_dir, "refinement_groups.json")
    save_json(REFINEMENT_GROUPS, groups_path)
    logger.info("Wrote metrics to %s", out_path)
    return {"metrics": metrics, "eval_dir": eval_dir, "refinement_groups": REFINEMENT_GROUPS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--annotations_path", required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()
    evaluate_regex_vs_llm(
        run_id=args.run_id,
        samples_path=args.samples_path,
        annotations_path=args.annotations_path,
        class_ids=args.classes,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()
