"""DAG-aware regex-vs-LLM eval for an agent-owned class subgraph.

Loads leaf patterns from a workdir ``cxr_classes`` folder (not production OR
composition). Applies ontology label + match propagation within the class set.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from medvqa.datasets.cxr_report_labeling.label_semantics import (
    llm_label_to_binary_positive,
)
from medvqa.datasets.cxr_report_labeling.ontology_dag import (
    OntologyDag,
    load_ontology_dag,
)
from medvqa.datasets.regular_expressions.cxr_patterns import (
    clear_pattern_cache,
    load_pattern,
)
from medvqa.scripts.cxr_report_labeling.eval_regex_vs_llm import (
    load_annotations_flexible,
)
from medvqa.utils.files_utils import load_jsonl, save_json
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _leaf_match(
    text: str,
    class_id: str,
    regex_classes_dir: str,
) -> Tuple[bool, List[str]]:
    path = os.path.join(regex_classes_dir, f"{class_id}.txt")
    if not os.path.isfile(path):
        return False, []
    pat = load_pattern(class_id, directory=regex_classes_dir)
    cleaned = " ".join(text.lower().split())
    m = pat.search(cleaned)
    if not m:
        return False, []
    return True, [m.group(0)]


def _propagated_match(
    *,
    text: str,
    class_id: str,
    dag: OntologyDag,
    class_ids: Set[str],
    regex_classes_dir: str,
    leaf_cache: Dict[str, Tuple[bool, List[str]]],
) -> Tuple[bool, List[str], List[str]]:
    """True if this class or any descendant in the set matches (leaf patterns).

    Returns ``(matched, spans, matched_via_classes)`` where
    ``matched_via_classes`` lists leaf ids in the set that fired.
    """
    spans: List[str] = []
    matched_via: List[str] = []
    matched = False
    nodes = [class_id] + sorted(dag.descendants(class_id) & class_ids)
    for nid in nodes:
        if nid not in leaf_cache:
            leaf_cache[nid] = _leaf_match(text, nid, regex_classes_dir)
        ok, sp = leaf_cache[nid]
        if ok:
            matched = True
            matched_via.append(nid)
            for s in sp:
                if s not in spans:
                    spans.append(s)
    return matched, spans, matched_via


def _llm_payload_compact(payload: Optional[dict]) -> dict:
    """Keep the fields agents need when auditing disagreements."""
    if not payload or not isinstance(payload, dict):
        return {}
    out = {}
    for key in ("label", "relevant_quote", "reasoning", "uses_hedging_language"):
        if key in payload:
            out[key] = payload[key]
    return out


def _gold_via_classes(
    *,
    class_id: str,
    dag: OntologyDag,
    id_set: Set[str],
    pos_map: Dict[str, bool],
) -> List[str]:
    """Direct-positive class ids in the set that justify parent gold_positive."""
    via: List[str] = []
    if pos_map.get(class_id, False):
        via.append(class_id)
    for d in sorted(dag.descendants(class_id) & id_set):
        if pos_map.get(d, False):
            via.append(d)
    return via


def _llm_by_class_for_row(
    *,
    uid: str,
    class_id: str,
    gold_via: Sequence[str],
    matched_via: Sequence[str],
    llm_by_pair: Dict[Tuple[str, str], dict],
) -> Dict[str, dict]:
    """Attach compact LLM payloads for evaluated + gold_via + matched_via classes."""
    want = list(dict.fromkeys([class_id, *gold_via, *matched_via]))
    out: Dict[str, dict] = {}
    for cid in want:
        payload = llm_by_pair.get((uid, cid))
        compact = _llm_payload_compact(payload)
        if compact:
            out[cid] = compact
    return out


def evaluate_regex_subgraph(
    *,
    class_ids: Sequence[str],
    regex_classes_dir: str,
    annotations_path: str,
    samples_path: str,
    out_dir: str,
    dag: Optional[OntologyDag] = None,
) -> dict:
    """Evaluate leaf regexes (+ ontology propagation) vs LLM gold.

    Writes ``metrics_per_class.json``, ``indices.json``, and per-class
    ``*_fp.jsonl`` / ``*_fn.jsonl`` under ``out_dir``.
    """
    setup_logging()
    clear_pattern_cache()
    class_ids = list(dict.fromkeys(class_ids))
    if not class_ids:
        raise ValueError("class_ids must be non-empty")
    full_dag = dag or load_ontology_dag()
    if full_dag.class_ids != frozenset(class_ids):
        # Allow either a pre-induced subgraph or the full DAG.
        if frozenset(class_ids).issubset(full_dag.class_ids):
            sub = full_dag.induce_subgraph(class_ids)
        else:
            raise KeyError("class_ids not contained in ontology")
    else:
        sub = full_dag
    id_set = set(class_ids)

    samples = {s["uid"]: s for s in load_jsonl(samples_path)}
    ann = load_annotations_flexible(annotations_path)

    # Direct gold from annotations for classes in set (only annotated pairs).
    # uid -> class_id -> optional bool (None = no annotation for that pair)
    direct_gold: Dict[str, Dict[str, Optional[bool]]] = defaultdict(dict)
    llm_by_pair: Dict[Tuple[str, str], dict] = {}
    for (uid, cid), payload in ann.items():
        if cid not in id_set:
            continue
        if uid not in samples:
            continue
        direct_gold[uid][cid] = llm_label_to_binary_positive(payload.get("label"))
        llm_by_pair[(uid, cid)] = payload

    uids = sorted(direct_gold.keys())
    os.makedirs(out_dir, exist_ok=True)

    metrics: Dict[str, Any] = {}
    indices: Dict[str, Dict[str, List[str]]] = {}
    leaf_cache_by_uid: Dict[str, Dict[str, Tuple[bool, List[str]]]] = {}

    for class_id in class_ids:
        tp = fp = fn = tn = 0
        fp_rows: List[dict] = []
        fn_rows: List[dict] = []
        fp_uids: List[str] = []
        fn_uids: List[str] = []

        for uid in uids:
            # Gold for this class: annotated label with ancestor propagation from
            # any positive descendant annotation on the same report.
            pos_map = {
                cid: bool(direct_gold[uid][cid])
                for cid in id_set
                if cid in direct_gold[uid] and direct_gold[uid][cid] is not None
            }
            # Only evaluate class_id if we have a direct annotation OR a
            # propagating descendant annotation that makes gold defined.
            has_direct = class_id in direct_gold[uid]
            desc_pos = any(
                pos_map.get(d, False)
                for d in (sub.descendants(class_id) & id_set)
                if d in pos_map
            )
            if not has_direct and not desc_pos:
                # No gold signal for this class on this uid
                continue
            if has_direct:
                gold_pos = bool(direct_gold[uid][class_id]) or desc_pos
            else:
                gold_pos = desc_pos

            gold_via = _gold_via_classes(
                class_id=class_id,
                dag=sub,
                id_set=id_set,
                pos_map=pos_map,
            )
            # If gold_pos is True only via descendant but pos_map missed the
            # evaluated class (direct false), gold_via still lists descendants.
            # If gold_pos is False, gold_via should be empty.
            if not gold_pos:
                gold_via = []

            if uid not in leaf_cache_by_uid:
                leaf_cache_by_uid[uid] = {}
            report_text = samples[uid]["report_text"]
            matched, spans, matched_via = _propagated_match(
                text=report_text,
                class_id=class_id,
                dag=sub,
                class_ids=id_set,
                regex_classes_dir=regex_classes_dir,
                leaf_cache=leaf_cache_by_uid[uid],
            )

            llm = llm_by_pair.get((uid, class_id)) or {}
            llm_by_class = _llm_by_class_for_row(
                uid=uid,
                class_id=class_id,
                gold_via=gold_via,
                matched_via=matched_via,
                llm_by_pair=llm_by_pair,
            )
            gold_via_descendant = any(c != class_id for c in gold_via)
            row_base = {
                "uid": uid,
                "class_id": class_id,
                "report_text": report_text,
                "llm": llm,
                "llm_by_class": llm_by_class,
                "regex_spans": spans,
                "matched_via_classes": matched_via,
                "gold_positive": gold_pos,
                "gold_via_classes": gold_via,
                "gold_via_descendant": gold_via_descendant,
            }

            if matched and gold_pos:
                tp += 1
            elif matched and not gold_pos:
                fp += 1
                fp_uids.append(uid)
                fp_rows.append(row_base)
            elif (not matched) and gold_pos:
                fn += 1
                fn_uids.append(uid)
                fn_rows.append(row_base)
            else:
                tn += 1

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[class_id] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_evaluated": tp + fp + fn + tn,
        }
        indices[class_id] = {"fp_uids": fp_uids, "fn_uids": fn_uids}

        stem = class_id.replace("/", "_")
        with open(os.path.join(out_dir, f"{stem}_fp.jsonl"), "w", encoding="utf-8") as f:
            for row in fp_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"{stem}_fn.jsonl"), "w", encoding="utf-8") as f:
            for row in fn_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(out_dir, "metrics_per_class.json")
    indices_path = os.path.join(out_dir, "indices.json")
    save_json(metrics, metrics_path)
    save_json(indices, indices_path)
    meta = {
        "class_ids": class_ids,
        "regex_classes_dir": os.path.abspath(regex_classes_dir),
        "annotations_path": os.path.abspath(annotations_path),
        "samples_path": os.path.abspath(samples_path),
        "dag": sub.as_parent_child_dict(),
        "metrics_path": metrics_path,
        "indices_path": indices_path,
    }
    save_json(meta, os.path.join(out_dir, "eval_meta.json"))
    logger.info("Wrote subgraph eval metrics to %s", metrics_path)
    return {"metrics": metrics, "indices": indices, "out_dir": out_dir, "meta": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--regex_classes_dir", required=True)
    parser.add_argument("--annotations_path", required=True)
    parser.add_argument("--samples_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    result = evaluate_regex_subgraph(
        class_ids=args.classes,
        regex_classes_dir=args.regex_classes_dir,
        annotations_path=args.annotations_path,
        samples_path=args.samples_path,
        out_dir=args.out_dir,
    )
    print(json.dumps({"out_dir": result["out_dir"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
