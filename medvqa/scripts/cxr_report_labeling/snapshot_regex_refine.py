"""Freeze LLM annotations + regex classes under .agent/regex_refine/<stamp>/."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from medvqa.datasets.cxr_report_labeling.ontology_dag import load_ontology_dag
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_cache_run_dir,
    get_results_run_dir,
    resolve_sample_annotations_path,
)
from medvqa.settings import PROJECT_ROOT, REGULAR_EXPRESSIONS_FOLDER
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _expand_eval_class_ids(
    owned_class_ids: Sequence[str],
    *,
    include_ontology_descendants: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(owned, readonly, eval)`` class id lists.

    When ``include_ontology_descendants`` is True, ontology descendants of each
    owned stem are added as read-only eval members (true leaves are unchanged).
    """
    owned = list(dict.fromkeys(owned_class_ids))
    if not owned:
        raise ValueError("owned_class_ids must be non-empty")
    dag = load_ontology_dag()
    unknown = [cid for cid in owned if cid not in dag.class_ids]
    if unknown:
        raise KeyError(f"Owned class_id(s) not in ontology: {unknown}")
    readonly: list[str] = []
    if include_ontology_descendants:
        desc: set[str] = set()
        for cid in owned:
            desc |= dag.descendants(cid)
        readonly = sorted(desc - set(owned))
    eval_ids = owned + readonly
    return owned, readonly, eval_ids


def _materialize_group(
    *,
    root: Path,
    group_id: str,
    class_ids: Sequence[str],
    include_ontology_descendants: bool = True,
) -> dict:
    """Create ``groups/<group_id>/`` with owned + optional readonly descendant leaves.

    ``class_ids`` are the **owned** stems the coordinator assigned (promotable).
    Ontology descendants are copied as read-only eval companions by default.
    """
    owned, readonly, eval_ids = _expand_eval_class_ids(
        class_ids,
        include_ontology_descendants=include_ontology_descendants,
    )
    dag = load_ontology_dag()
    subgraph = dag.induce_subgraph(eval_ids)
    pristine = root / "cxr_classes_pristine"
    group_dir = root / "groups" / group_id
    if group_dir.exists():
        raise FileExistsError(group_dir)
    group_classes = group_dir / "cxr_classes"
    group_eval = group_dir / "eval"
    group_dir.mkdir(parents=True, exist_ok=False)
    group_classes.mkdir(parents=True, exist_ok=False)
    missing = []
    for cid in eval_ids:
        src = pristine / f"{cid}.txt"
        if not src.is_file():
            missing.append(cid)
            continue
        shutil.copy2(src, group_classes / f"{cid}.txt")
    if missing:
        raise FileNotFoundError(
            f"Eval class regex file(s) missing under {pristine}: {missing}"
        )
    owned_payload = {
        "group_id": group_id,
        "owned_class_ids": owned,
        "readonly_class_ids": readonly,
        "eval_class_ids": eval_ids,
        "class_ids": eval_ids,
        "include_ontology_descendants": include_ontology_descendants,
    }
    owned_path = group_dir / "owned_classes.json"
    with open(owned_path, "w", encoding="utf-8") as f:
        json.dump(owned_payload, f, indent=2)
        f.write("\n")
    group_eval.mkdir(parents=True, exist_ok=True)
    return {
        "group_id": group_id,
        "owned_class_ids": owned,
        "readonly_class_ids": readonly,
        "eval_class_ids": eval_ids,
        "class_ids": eval_ids,
        "include_ontology_descendants": include_ontology_descendants,
        "dag": subgraph.as_parent_child_dict(),
        "paths": {
            "group_dir": str(group_dir),
            "group_cxr_classes": str(group_classes),
            "group_owned_classes": str(owned_path),
            "group_eval": str(group_eval),
        },
    }


def add_regex_refine_group(
    *,
    stamp: str,
    class_ids: Sequence[str],
    group_id: Optional[str] = None,
    include_ontology_descendants: bool = True,
) -> dict:
    """Add a group under an existing snapshot stamp."""
    setup_logging()
    class_ids = list(dict.fromkeys(class_ids))
    group_id = group_id or (
        "_".join(class_ids) if len(class_ids) <= 3 else f"group_{len(class_ids)}"
    )
    root = Path(PROJECT_ROOT) / ".agent" / "regex_refine" / stamp
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot stamp not found: {root}")
    group_info = _materialize_group(
        root=root,
        group_id=group_id,
        class_ids=class_ids,
        include_ontology_descendants=include_ontology_descendants,
    )
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"stamp": stamp, "paths": {"root": str(root)}}
    groups = manifest.setdefault("groups", [])
    groups.append(group_info)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info(
        "Added group %s to snapshot %s (owned=%s readonly=%s)",
        group_id,
        stamp,
        group_info["owned_class_ids"],
        group_info["readonly_class_ids"],
    )
    return group_info


def snapshot_regex_refine(
    *,
    run_id: str = DEFAULT_RUN_ID,
    model_name: str = DEFAULT_MODEL_NAME,
    class_ids: Sequence[str],
    group_id: Optional[str] = None,
    annotations_path: Optional[str] = None,
    samples_path: Optional[str] = None,
    stamp: Optional[str] = None,
    extra_groups: Optional[Sequence[tuple[str, Sequence[str]]]] = None,
    include_ontology_descendants: bool = True,
) -> dict:
    """Create a frozen refine workspace under ``.agent/regex_refine/<stamp>/``.

    ``extra_groups`` is a sequence of ``(group_id, class_ids)`` to materialize
    alongside the primary group (shared annotations/samples/pristine).

    By default, ontology descendants of each owned stem are copied into the
    group workdir as **read-only** eval companions so ``eval_regex_subgraph``
    can propagate gold + matches within the closed subgraph.
    """
    setup_logging()
    class_ids = list(dict.fromkeys(class_ids))
    if not class_ids:
        raise ValueError("class_ids must be non-empty")
    group_id = group_id or ("_".join(class_ids) if len(class_ids) <= 3 else f"group_{len(class_ids)}")

    if annotations_path is None:
        annotations_path = resolve_sample_annotations_path(run_id, model_name)
    if samples_path is None:
        samples_path = os.path.join(get_results_run_dir(run_id), "samples.jsonl")

    for path, label in ((annotations_path, "annotations"), (samples_path, "samples")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    stamp = stamp or _stamp()
    root = Path(PROJECT_ROOT) / ".agent" / "regex_refine" / stamp
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)

    ann_dst = root / "annotations.snapshot.jsonl"
    samples_dst = root / "samples.snapshot.jsonl"
    pristine = root / "cxr_classes_pristine"

    shutil.copy2(annotations_path, ann_dst)
    shutil.copy2(samples_path, samples_dst)
    src_classes = os.path.join(REGULAR_EXPRESSIONS_FOLDER, "cxr_classes")
    shutil.copytree(src_classes, pristine)

    primary = _materialize_group(
        root=root,
        group_id=group_id,
        class_ids=class_ids,
        include_ontology_descendants=include_ontology_descendants,
    )
    groups = [primary]
    for gid, cids in extra_groups or []:
        groups.append(
            _materialize_group(
                root=root,
                group_id=gid,
                class_ids=list(cids),
                include_ontology_descendants=include_ontology_descendants,
            )
        )

    manifest = {
        "stamp": stamp,
        "run_id": run_id,
        "model_name": model_name,
        "group_id": group_id,
        "owned_class_ids": primary["owned_class_ids"],
        "readonly_class_ids": primary["readonly_class_ids"],
        "eval_class_ids": primary["eval_class_ids"],
        "class_ids": primary["eval_class_ids"],
        "include_ontology_descendants": include_ontology_descendants,
        "dag": primary["dag"],
        "groups": groups,
        "paths": {
            "root": str(root),
            "annotations_snapshot": str(ann_dst),
            "samples_snapshot": str(samples_dst),
            "cxr_classes_pristine": str(pristine),
            "group_dir": primary["paths"]["group_dir"],
            "group_cxr_classes": primary["paths"]["group_cxr_classes"],
            "group_owned_classes": primary["paths"]["group_owned_classes"],
            "group_eval": primary["paths"]["group_eval"],
            "source_annotations": os.path.abspath(annotations_path),
            "source_samples": os.path.abspath(samples_path),
        },
        "n_annotation_lines": sum(1 for _ in open(ann_dst, encoding="utf-8")),
        "n_sample_lines": sum(1 for _ in open(samples_dst, encoding="utf-8")),
        "promote": (
            "Copy each stem in owned_class_ids from "
            "groups/<group_id>/cxr_classes/<id>.txt into "
            "medvqa/datasets/regular_expressions/cxr_classes/ after review. "
            "Do not promote readonly_class_ids (descendant companions)."
        ),
    }
    manifest_path = root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("Wrote regex-refine snapshot to %s (%d groups)", root, len(groups))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Owned class_ids for the primary refine group (promotable stems)",
    )
    parser.add_argument("--group_id", default=None)
    parser.add_argument("--annotations_path", default=None)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--stamp", default=None)
    parser.add_argument(
        "--extra_group",
        action="append",
        nargs="+",
        metavar="GROUP_OR_CLASS",
        help=(
            "Additional group: GROUP_ID OWNED_CLASS_ID [OWNED_CLASS_ID ...]. "
            "Repeat flag for multiple groups. Ontology descendants are added "
            "as read-only eval companions unless --no_include_ontology_descendants."
        ),
    )
    parser.add_argument(
        "--add_to_stamp",
        default=None,
        help="If set, only add --classes as a new group under this existing stamp",
    )
    parser.add_argument(
        "--no_include_ontology_descendants",
        action="store_true",
        help=(
            "Do not auto-expand ontology descendants into the group eval set "
            "(leaf-only / singleton mid-node metrics)."
        ),
    )
    args = parser.parse_args()
    include_desc = not args.no_include_ontology_descendants
    if args.add_to_stamp:
        info = add_regex_refine_group(
            stamp=args.add_to_stamp,
            class_ids=args.classes,
            group_id=args.group_id,
            include_ontology_descendants=include_desc,
        )
        print(json.dumps(info, indent=2))
        return
    extra = []
    for spec in args.extra_group or []:
        if len(spec) < 2:
            raise SystemExit("--extra_group needs GROUP_ID and at least one CLASS_ID")
        extra.append((spec[0], spec[1:]))
    manifest = snapshot_regex_refine(
        run_id=args.run_id,
        model_name=args.model_name,
        class_ids=args.classes,
        group_id=args.group_id,
        annotations_path=args.annotations_path,
        samples_path=args.samples_path,
        stamp=args.stamp,
        extra_groups=extra,
        include_ontology_descendants=include_desc,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
