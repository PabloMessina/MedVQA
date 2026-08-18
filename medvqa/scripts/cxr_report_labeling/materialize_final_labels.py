"""Materialize released 6-way labels from regex matches + LLM verify.

Candidate gate: class C is a candidate if C or any ontology descendant is in
``positive_classes`` (jsonl hits ∪ DAG ancestors). Non-candidates are
``regex_unmatched`` even if sample LLM gold exists.

Presence entailment among ``{C} ∪ matched descendants``: ``definitely true`` >
``likely true``. Else C's own LLM if C matched. Else descendant-only
non-positive → ``unknown``. Child negation does not veto a parent.

``no_finding`` (``--stage final``): any other class ``definitely true`` →
``definitely false``; else any ``likely true`` → ``likely false``; else LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    build_class_registry,
)
from medvqa.datasets.cxr_report_labeling.label_semantics import (
    LLM_LABELS,
    NORMAL_CLASS,
    NPZ_LABEL_TO_UINT8,
    RELEASED_REGEX_UNMATCHED,
    RESOLUTION_LLM_FAILED,
    RESOLUTION_NO_FINDING_AUTO,
    RESOLUTION_NO_FINDING_LLM,
    normalize_nli_label,
    resolve_no_finding_from_other_labels,
    resolve_presence_entailment,
)
from medvqa.datasets.cxr_report_labeling.ontology_dag import load_ontology_dag
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_failed_query_hashes_path,
    get_final_labels_dir,
    get_full_regex_matches_path,
    get_legacy_failed_query_hashes_path,
    get_results_run_dir,
    resolve_sample_annotations_path,
    resolve_verify_annotations_path,
)
from medvqa.datasets.cxr_report_labeling.report_loaders import (
    ALL_DATASETS,
    load_reports_by_uid,
)
from medvqa.datasets.cxr_report_labeling.schemas import (
    annotation_config_fingerprint,
    annotation_content_key,
    annotation_query_hash,
    report_text_hash,
)
from medvqa.datasets.cxr_report_labeling.verify_jobs import (
    expand_candidate_classes,
    load_failed_query_hashes_union,
    matched_via_tags,
    parsed_response_ok,
)
from medvqa.scripts.cxr_report_labeling.annotate_reports_with_llm import (
    DEFAULT_MODEL_NAME,
)
from medvqa.utils.files_utils import load_jsonl, save_json
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

STAGE_REGEX_BACKED = "regex_backed"
STAGE_FINAL = "final"


class LlmIndex:
    """Join LLM rows by (uid, class_id), then by content_key."""

    def __init__(self) -> None:
        self.by_uid_class: Dict[Tuple[str, str], dict] = {}
        self.by_content_key: Dict[str, dict] = {}

    def add_row(self, row: dict, expected_dataset: Optional[str] = None) -> None:
        parsed = row.get("parsed_response")
        if not parsed_response_ok(parsed):
            return
        meta = row.get("metadata") or {}
        ds = meta.get("dataset")
        if expected_dataset and ds and ds != expected_dataset:
            return
        uid, class_id = meta.get("uid"), meta.get("class_id")
        rec = {
            "label": normalize_nli_label(parsed.get("label")),
            "relevant_quote": parsed.get("relevant_quote"),
            "reasoning": parsed.get("reasoning"),
            "uses_hedging_language": parsed.get("uses_hedging_language"),
            "content_key": meta.get("content_key"),
            "query_hash": meta.get("query_hash"),
            "model_name": meta.get("model_name"),
            "config_fingerprint": meta.get("config_fingerprint"),
        }
        if uid and class_id and rec["label"]:
            self.by_uid_class[(uid, class_id)] = rec
        ck = meta.get("content_key")
        if ck and rec["label"] and ck not in self.by_content_key:
            self.by_content_key[ck] = rec

    def lookup(self, uid: str, class_id: str, content_key: Optional[str]) -> Optional[dict]:
        hit = self.by_uid_class.get((uid, class_id))
        if hit is not None:
            return hit
        if content_key:
            return self.by_content_key.get(content_key)
        return None


def _load_llm_index(
    paths: Sequence[str], expected_dataset: Optional[str] = None
) -> LlmIndex:
    index = LlmIndex()
    for path in paths:
        if not path or not os.path.exists(path):
            logger.info("LLM path missing (skipped): %s", path)
            continue
        n = 0
        n_skip_ds = 0
        for row in load_jsonl(path):
            meta = (row.get("metadata") or {})
            ds = meta.get("dataset")
            if expected_dataset and ds and ds != expected_dataset:
                n_skip_ds += 1
                continue
            index.add_row(row, expected_dataset=expected_dataset)
            n += 1
        logger.info(
            "Loaded %d rows from %s (skipped_other_dataset=%d)",
            n,
            path,
            n_skip_ds,
        )
    return index


def _infer_model_config(paths: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            meta = row.get("metadata") or {}
            model = meta.get("model_name")
            fp = meta.get("config_fingerprint")
            if model and fp:
                return str(model), str(fp)
    return None, None


def _default_llm_paths(
    *,
    run_id: str,
    dataset: str,
    model_name: str,
    stage: str,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    paths: List[str] = [
        resolve_sample_annotations_path(run_id, model_name),
        resolve_verify_annotations_path(run_id, model_name, dataset),
    ]
    if stage == STAGE_FINAL:
        paths.append(
            resolve_verify_annotations_path(
                run_id, model_name, dataset, no_finding=True
            )
        )
    for p in extra or []:
        if p and p not in paths:
            paths.append(p)
    return paths


def _sidecar_for_verify(run_id: str, model_name: str, dataset: str) -> str:
    verify = resolve_verify_annotations_path(run_id, model_name, dataset)
    return os.path.join(os.path.dirname(verify), "failed_query_hashes.jsonl")


def _regex_backed_class_ids(registry: ClassRegistry) -> List[str]:
    return [c for c in registry.class_ids() if c != NORMAL_CLASS and registry.by_id()[c].has_regex]


def materialize_final_labels(
    *,
    dataset: str,
    run_id: str = DEFAULT_RUN_ID,
    stage: str = STAGE_REGEX_BACKED,
    regex_matches_path: Optional[str] = None,
    llm_annotation_paths: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    config_slug: Optional[str] = None,
    write_npz: bool = True,
) -> dict:
    setup_logging()
    if stage not in (STAGE_REGEX_BACKED, STAGE_FINAL):
        raise ValueError(f"stage must be {STAGE_REGEX_BACKED!r} or {STAGE_FINAL!r}")

    results_dir = get_results_run_dir(run_id)
    matches_path = regex_matches_path or get_full_regex_matches_path(run_id, dataset)
    matches = load_jsonl(matches_path)

    registry_path = os.path.join(results_dir, "class_registry.json")
    registry = (
        ClassRegistry.load(registry_path)
        if os.path.exists(registry_path)
        else build_class_registry()
    )
    dag = load_ontology_dag()
    by_id = registry.by_id()
    regex_classes = _regex_backed_class_ids(registry)

    if llm_annotation_paths is None:
        llm_annotation_paths = _default_llm_paths(
            run_id=run_id,
            dataset=dataset,
            model_name=model_name,
            stage=stage,
        )
    llm_annotation_paths = list(llm_annotation_paths)
    llm_index = _load_llm_index(llm_annotation_paths, expected_dataset=dataset)

    inferred_model, inferred_fp = _infer_model_config(llm_annotation_paths)
    use_model = inferred_model or model_name
    config_fingerprint = inferred_fp or annotation_config_fingerprint()

    sidecar_paths: List[Optional[str]] = []
    if config_slug:
        sidecar_paths.append(
            get_failed_query_hashes_path(
                run_id, model_name, config_slug, dataset=dataset
            )
        )
        sidecar_paths.append(
            get_legacy_failed_query_hashes_path(run_id, model_name, config_slug)
        )
    else:
        sidecar_paths.append(_sidecar_for_verify(run_id, model_name, dataset))
        nf_verify = resolve_verify_annotations_path(
            run_id, model_name, dataset, no_finding=True
        )
        alt = os.path.join(os.path.dirname(nf_verify), "failed_query_hashes.jsonl")
        sidecar_paths.append(alt)
    failed_hashes = load_failed_query_hashes_union(sidecar_paths)
    logger.info(
        "Fail sidecar union %s (%d hashes)", sidecar_paths, len(failed_hashes)
    )

    reports_by_uid = load_reports_by_uid(dataset)
    out_dir = get_final_labels_dir(run_id)
    compact_path = os.path.join(out_dir, f"{dataset}.jsonl")
    sparse_path = os.path.join(out_dir, f"{dataset}_sparse.jsonl")

    missing_required: List[dict] = []
    n_fail_leftover = 0
    n_candidates = 0
    compact_rows: List[dict] = []
    sparse_rows: List[dict] = []

    for row in matches:
        uid = row["uid"]
        jsonl_pos = {c for c in (row.get("positive_classes") or []) if c and c != NORMAL_CLASS}
        candidates = expand_candidate_classes(jsonl_pos, dag)
        report = reports_by_uid.get(uid)
        rhash = None
        if report is not None:
            rhash = report.get("report_text_hash") or report_text_hash(report["report_text"])

        labels: Dict[str, str] = {}
        llm_by_class: Dict[str, Optional[str]] = {}
        rec_by_class: Dict[str, dict] = {}

        def _lookup(cid: str) -> Optional[dict]:
            entry = by_id.get(cid)
            ck = None
            qh = None
            if entry is not None and rhash is not None:
                ck = annotation_content_key(
                    rhash, entry.prompt_hash, registry.system_prompt_hash
                )
                qh = annotation_query_hash(ck, use_model, config_fingerprint)
            rec = llm_index.lookup(uid, cid, ck)
            return rec, ck, qh

        # Preload LLM for candidates + their matched descendants
        needed: Set[str] = set(candidates)
        for cid in list(candidates):
            if cid in dag.class_ids:
                needed.update(d for d in dag.descendants(cid) if d in jsonl_pos)
        for cid in needed:
            rec, ck, qh = _lookup(cid)
            rec_by_class[cid] = {"rec": rec, "content_key": ck, "query_hash": qh}
            llm_by_class[cid] = rec["label"] if rec else None

        for class_id in regex_classes:
            own_matched = class_id in jsonl_pos
            is_cand = class_id in candidates
            if not is_cand:
                labels[class_id] = RELEASED_REGEX_UNMATCHED
                continue

            n_candidates += 1
            matched_desc = []
            if class_id in dag.class_ids:
                matched_desc = sorted(
                    d for d in dag.descendants(class_id) if d in jsonl_pos
                )
            via = matched_via_tags(class_id, jsonl_pos, dag)

            # Missing LLM: required unless fail-sidecar leftover
            cell_info = rec_by_class.get(class_id) or {}
            rec = cell_info.get("rec")
            qh = cell_info.get("query_hash")
            considered = [class_id, *matched_desc]
            missing_not_failed = []
            leftover = False
            for cid in considered:
                info = rec_by_class.get(cid) or {}
                if info.get("rec") is not None:
                    continue
                qh_c = info.get("query_hash")
                if qh_c and qh_c in failed_hashes:
                    leftover = True
                    continue
                missing_not_failed.append(cid)
            if missing_not_failed:
                missing_required.append(
                    {
                        "uid": uid,
                        "class_id": class_id,
                        "missing_classes": missing_not_failed,
                    }
                )
                continue

            resolved = resolve_presence_entailment(
                class_id,
                own_matched=own_matched,
                matched_descendants=matched_desc,
                llm_by_class=llm_by_class,
            )
            if leftover and resolved.get("resolution") == RESOLUTION_LLM_FAILED:
                n_fail_leftover += 1
                sparse_rows.append(
                    {
                        "uid": uid,
                        "dataset": dataset,
                        "class_id": class_id,
                        "matched_via": via,
                        "jsonl_positive": own_matched,
                        "own_llm_label": llm_by_class.get(class_id),
                        "own_relevant_quote": (rec or {}).get("relevant_quote") if rec else None,
                        "dominated_by": None,
                        "resolution": RESOLUTION_LLM_FAILED,
                        "label": None,
                    }
                )
                continue

            lab = resolved["label"]
            if lab is None:
                n_fail_leftover += 1
                sparse_rows.append(
                    {
                        "uid": uid,
                        "dataset": dataset,
                        "class_id": class_id,
                        "matched_via": via,
                        "jsonl_positive": own_matched,
                        "own_llm_label": None,
                        "own_relevant_quote": None,
                        "dominated_by": None,
                        "resolution": RESOLUTION_LLM_FAILED,
                        "label": None,
                    }
                )
                continue

            labels[class_id] = lab
            sparse_rows.append(
                {
                    "uid": uid,
                    "dataset": dataset,
                    "class_id": class_id,
                    "matched_via": via,
                    "jsonl_positive": own_matched,
                    "own_llm_label": llm_by_class.get(class_id),
                    "own_relevant_quote": (rec or {}).get("relevant_quote") if rec else None,
                    "dominated_by": resolved.get("dominated_by"),
                    "resolution": resolved.get("resolution"),
                    "label": lab,
                }
            )

        compact_rows.append(
            {
                "uid": uid,
                "dataset": dataset,
                "report_id": row.get("report_id"),
                "labels": labels,
            }
        )

    if missing_required:
        raise RuntimeError(
            f"Refuse to release {dataset}: {len(missing_required)} regex-candidate "
            f"cell(s) lack LLM and are not fail-sidecar leftovers. "
            f"Examples: {missing_required[:8]!r}. "
            "Do not map missing → regex_unmatched. Finish verify or retry leftovers "
            "with a new model/config."
        )

    # ----- no_finding (final stage) -----
    n_nf_auto = 0
    n_nf_llm = 0
    n_nf_fail = 0
    if stage == STAGE_FINAL:
        for crow in compact_rows:
            uid = crow["uid"]
            labels = crow["labels"]
            auto = resolve_no_finding_from_other_labels(labels)
            report = reports_by_uid.get(uid)
            rhash = None
            if report is not None:
                rhash = report.get("report_text_hash") or report_text_hash(
                    report["report_text"]
                )
            entry = by_id[NORMAL_CLASS]
            ck = (
                annotation_content_key(
                    rhash, entry.prompt_hash, registry.system_prompt_hash
                )
                if rhash is not None
                else None
            )
            qh = (
                annotation_query_hash(ck, use_model, config_fingerprint)
                if ck
                else None
            )
            if auto is not None:
                lab, dominated = auto
                labels[NORMAL_CLASS] = lab
                n_nf_auto += 1
                sparse_rows.append(
                    {
                        "uid": uid,
                        "dataset": dataset,
                        "class_id": NORMAL_CLASS,
                        "matched_via": [],
                        "jsonl_positive": False,
                        "own_llm_label": None,
                        "own_relevant_quote": None,
                        "dominated_by": dominated,
                        "resolution": RESOLUTION_NO_FINDING_AUTO,
                        "label": lab,
                    }
                )
                continue
            rec = llm_index.lookup(uid, NORMAL_CLASS, ck)
            if rec is None:
                if qh and qh in failed_hashes:
                    n_nf_fail += 1
                    n_fail_leftover += 1
                    sparse_rows.append(
                        {
                            "uid": uid,
                            "dataset": dataset,
                            "class_id": NORMAL_CLASS,
                            "matched_via": [],
                            "jsonl_positive": False,
                            "own_llm_label": None,
                            "own_relevant_quote": None,
                            "dominated_by": None,
                            "resolution": RESOLUTION_LLM_FAILED,
                            "label": None,
                        }
                    )
                    continue
                missing_required.append(
                    {"uid": uid, "class_id": NORMAL_CLASS, "missing_classes": [NORMAL_CLASS]}
                )
                continue
            lab = rec["label"]
            labels[NORMAL_CLASS] = lab
            n_nf_llm += 1
            sparse_rows.append(
                {
                    "uid": uid,
                    "dataset": dataset,
                    "class_id": NORMAL_CLASS,
                    "matched_via": [],
                    "jsonl_positive": False,
                    "own_llm_label": lab,
                    "own_relevant_quote": rec.get("relevant_quote"),
                    "dominated_by": NORMAL_CLASS,
                    "resolution": RESOLUTION_NO_FINDING_LLM,
                    "label": lab,
                }
            )
        if missing_required:
            raise RuntimeError(
                f"Refuse to release {dataset} no_finding: {len(missing_required)} "
                f"report(s) need LLM and are not fail-sidecar leftovers. "
                f"Examples: {missing_required[:8]!r}."
            )

    with open(compact_path, "w", encoding="utf-8") as f:
        for crow in compact_rows:
            f.write(json.dumps(crow, ensure_ascii=False) + "\n")
    with open(sparse_path, "w", encoding="utf-8") as f:
        for srow in sparse_rows:
            f.write(json.dumps(srow, ensure_ascii=False) + "\n")

    npz_path = None
    uids_path = None
    class_ids_path = None
    if write_npz:
        if stage == STAGE_FINAL:
            npz_class_ids = [
                c
                for c in registry.class_ids()
                if c == NORMAL_CLASS or registry.by_id()[c].has_regex
            ]
        else:
            npz_class_ids = list(regex_classes)
        n_u = len(compact_rows)
        n_c = len(npz_class_ids)
        mat = np.zeros((n_u, n_c), dtype=np.uint8)
        col = {cid: i for i, cid in enumerate(npz_class_ids)}
        for i, crow in enumerate(compact_rows):
            labs = crow.get("labels") or {}
            for cid, j in col.items():
                lab = labs.get(cid)
                if lab is None:
                    # Fail-sidecar leftover (or no_finding pending): not regex_unmatched
                    mat[i, j] = 255
                else:
                    mat[i, j] = NPZ_LABEL_TO_UINT8.get(lab, 255)
        npz_path = os.path.join(out_dir, f"{dataset}_labels.npz")
        np.savez_compressed(npz_path, labels=mat)
        uids_path = os.path.join(out_dir, f"{dataset}_uids.jsonl")
        with open(uids_path, "w", encoding="utf-8") as f:
            for crow in compact_rows:
                f.write(json.dumps({"uid": crow["uid"]}) + "\n")
        class_ids_path = os.path.join(out_dir, f"{dataset}_class_ids.json")
        save_json(
            {
                "class_ids": npz_class_ids,
                "uint8": {**NPZ_LABEL_TO_UINT8, "unresolved": 255},
                "nli_order": list(LLM_LABELS),
            },
            class_ids_path,
        )

    manifest = {
        "dataset": dataset,
        "stage": stage,
        "n_reports": len(compact_rows),
        "n_candidate_cells": n_candidates,
        "n_sparse_rows": len(sparse_rows),
        "n_fail_sidecar_leftovers": n_fail_leftover,
        "n_failed_query_hashes": len(failed_hashes),
        "fail_sidecar_paths": [p for p in sidecar_paths if p],
        "n_no_finding_auto": n_nf_auto,
        "n_no_finding_llm": n_nf_llm,
        "n_no_finding_fail": n_nf_fail,
        "compact_path": compact_path,
        "sparse_path": sparse_path,
        "npz_path": npz_path,
        "uids_path": uids_path,
        "class_ids_path": class_ids_path,
        "llm_annotation_paths": llm_annotation_paths,
        "regex_matches_path": matches_path,
        "model_name": use_model,
        "config_fingerprint": config_fingerprint,
    }
    save_json(manifest, os.path.join(out_dir, f"{dataset}_manifest.json"))
    logger.info(
        "Wrote %s (%d reports, %d candidates, %d fail leftovers)",
        compact_path,
        len(compact_rows),
        n_candidates,
        n_fail_leftover,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(ALL_DATASETS))
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--stage",
        default=STAGE_REGEX_BACKED,
        choices=[STAGE_REGEX_BACKED, STAGE_FINAL],
        help="regex_backed omits no_finding; final merges no_finding auto-false + LLM",
    )
    parser.add_argument("--regex_matches_path", default=None)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--config_slug", default=None)
    parser.add_argument(
        "--llm_annotation_paths",
        nargs="+",
        default=None,
        help="Override default sample gold + verify jsonl path(s)",
    )
    parser.add_argument(
        "--no_npz",
        action="store_true",
        help="Skip iuxray_labels.npz / uids.jsonl / class_ids.json",
    )
    args = parser.parse_args()
    manifest = materialize_final_labels(
        dataset=args.dataset,
        run_id=args.run_id,
        stage=args.stage,
        regex_matches_path=args.regex_matches_path,
        llm_annotation_paths=args.llm_annotation_paths,
        model_name=args.model_name,
        config_slug=args.config_slug,
        write_npz=not args.no_npz,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "llm_annotation_paths"}, indent=2))
    print("llm_annotation_paths:")
    for p in manifest.get("llm_annotation_paths") or []:
        print(f"  {p}")


if __name__ == "__main__":
    main()
