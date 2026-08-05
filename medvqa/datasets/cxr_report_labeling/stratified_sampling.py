"""K1 complexity / K2 round-robin stratified / K3 random sampling."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    resolve_regex_pattern_name,
)
from medvqa.datasets.cxr_report_labeling.complexity import score_reports
from medvqa.datasets.cxr_report_labeling.label_semantics import (
    NORMAL_CLASS,
    SUPPORT_DEVICES_CLASS,
    is_normal_disqualifying_positive,
)
from medvqa.datasets.cxr_report_labeling.schemas import sample_record
from medvqa.datasets.regular_expressions.cxr_patterns import collect_reports_matching_class

logger = logging.getLogger(__name__)

DEFAULT_K = {
    "mimiccxr": (200, 800, 200),
    "chexpert_plus": (200, 800, 200),
    "rexgradient": (200, 800, 200),
    "iuxray": (50, 250, 100),
}


def compute_regex_positive_sets(
    reports: Sequence[dict],
    registry: ClassRegistry,
    text_key: str = "report_text",
    num_processes: int = 1,
) -> Dict[str, Set[int]]:
    """Map prompt class_id -> set of report indices matching its regex hierarchy."""
    texts = [r[text_key].lower() for r in reports]
    # Slight cleanup already done; lowercase helps IGNORECASE patterns stay consistent
    texts = [" ".join(t.split()) for t in texts]
    cache: Dict[str, List[int]] = {}
    class_to_idxs: Dict[str, Set[int]] = {}
    for entry in registry.classes:
        if not entry.regex_names:
            continue
        idxs: Set[int] = set()
        for regex_name in entry.regex_names:
            matched = collect_reports_matching_class(
                reports=texts,
                class_name=regex_name,
                class_match_cache=cache,
                num_processes=num_processes,
            )
            idxs.update(matched)
        class_to_idxs[entry.class_id] = idxs
        logger.info("Regex stratum %s: %d matches", entry.class_id, len(idxs))
    return class_to_idxs


def compute_srr_positive_sets(
    srr_label_vectors: Sequence[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, Set[int]]:
    """Map SRR leaf name -> report indices predicted positive."""
    out: Dict[str, Set[int]] = {name: set() for name in class_names}
    for i, vec in enumerate(srr_label_vectors):
        for j, flag in enumerate(vec):
            if flag:
                out[class_names[j]].add(i)
    return out


def normal_proxy_indices(
    n_reports: int,
    regex_positives: Dict[str, Set[int]],
    srr_positives: Optional[Dict[str, Set[int]]] = None,
) -> Set[int]:
    """
    Normal candidates: SRR No Finding, or no non-Support-Devices regex hits.
    Support-Devices-only still eligible.
    """
    disqualified: Set[int] = set()
    for class_id, idxs in regex_positives.items():
        if class_id == SUPPORT_DEVICES_CLASS:
            continue
        disqualified |= idxs
    candidates = set(range(n_reports)) - disqualified
    if srr_positives and "No Finding" in srr_positives:
        candidates |= set(srr_positives["No Finding"])
        # Still remove clearly abnormal regex hits
        candidates -= disqualified
    return candidates


def _round_robin_sample(
    strata: Dict[str, List[int]],
    k: int,
    already_chosen: Set[int],
) -> List[Tuple[int, str]]:
    """
    Round-robin over strata (rarest-first). Each stratum list must be sorted
    by complexity descending. Returns (index, stratum_name) pairs.
    """
    # Sort strata by size ascending (rarest first) for tie-friendly coverage
    names = sorted(strata.keys(), key=lambda n: (len(strata[n]), n))
    pointers = {n: 0 for n in names}
    chosen: List[Tuple[int, str]] = []
    chosen_set = set(already_chosen)

    progress = True
    while len(chosen) < k and progress:
        progress = False
        for name in names:
            if len(chosen) >= k:
                break
            lst = strata[name]
            p = pointers[name]
            while p < len(lst) and lst[p] in chosen_set:
                p += 1
            pointers[name] = p
            if p < len(lst):
                idx = lst[p]
                pointers[name] = p + 1
                chosen.append((idx, name))
                chosen_set.add(idx)
                progress = True
    return chosen


def sample_dataset(
    reports: List[dict],
    *,
    dataset: str,
    registry: ClassRegistry,
    k1: int,
    k2: int,
    k3: int,
    complexity_scores: Optional[Sequence[float]] = None,
    regex_positives: Optional[Dict[str, Set[int]]] = None,
    srr_label_vectors: Optional[Sequence[np.ndarray]] = None,
    srr_class_names: Optional[Sequence[str]] = None,
    seed: int = 0,
    num_regex_processes: int = 1,
) -> List[dict]:
    """
    Sample K1+K2+K3 reports without replacement.
    """
    n = len(reports)
    if k1 + k2 + k3 > n:
        raise ValueError(f"Requested {k1}+{k2}+{k3} > {n} reports for {dataset}")

    if complexity_scores is None:
        complexity_scores = score_reports(reports)
    scores = list(complexity_scores)
    order_by_complexity = sorted(range(n), key=lambda i: scores[i], reverse=True)

    if regex_positives is None:
        regex_positives = compute_regex_positive_sets(
            reports, registry, num_processes=num_regex_processes
        )

    srr_positives: Dict[str, Set[int]] = {}
    if srr_label_vectors is not None:
        assert srr_class_names is not None
        srr_positives = compute_srr_positive_sets(srr_label_vectors, srr_class_names)

    # --- K1 ---
    k1_idxs = order_by_complexity[:k1]
    chosen: Set[int] = set(k1_idxs)
    samples: List[dict] = []
    for idx in k1_idxs:
        r = reports[idx]
        samples.append(
            sample_record(
                uid=r["uid"],
                dataset=dataset,
                report_id=r["report_id"],
                report_text=r["report_text"],
                source_refs=r.get("source_refs", {}),
                sampling_method="complexity",
                complexity_score=scores[idx],
                split=r.get("split"),
                extra={k: r[k] for k in ("study_id",) if k in r},
            )
        )

    # --- K2 strata ---
    strata: Dict[str, List[int]] = {}
    for class_id, idxs in regex_positives.items():
        remaining = [i for i in idxs if i not in chosen]
        remaining.sort(key=lambda i: scores[i], reverse=True)
        strata[f"regex:{class_id}"] = remaining

    for leaf, idxs in srr_positives.items():
        remaining = [i for i in idxs if i not in chosen]
        remaining.sort(key=lambda i: scores[i], reverse=True)
        strata[f"srr:{leaf}"] = remaining

    normal_idxs = normal_proxy_indices(n, regex_positives, srr_positives)
    normal_remaining = [i for i in normal_idxs if i not in chosen]
    normal_remaining.sort(key=lambda i: scores[i], reverse=True)
    strata[f"proxy:{NORMAL_CLASS}"] = normal_remaining

    # Drop empty strata
    strata = {k: v for k, v in strata.items() if v}

    k2_picks = _round_robin_sample(strata, k2, chosen)
    for idx, stratum in k2_picks:
        chosen.add(idx)
        r = reports[idx]
        samples.append(
            sample_record(
                uid=r["uid"],
                dataset=dataset,
                report_id=r["report_id"],
                report_text=r["report_text"],
                source_refs=r.get("source_refs", {}),
                sampling_method="stratified",
                complexity_score=scores[idx],
                split=r.get("split"),
                strata_labels={"stratum": stratum},
                extra={k: r[k] for k in ("study_id",) if k in r},
            )
        )

    # --- K3 ---
    remaining = [i for i in range(n) if i not in chosen]
    rng = random.Random(seed)
    if k3 > len(remaining):
        raise ValueError(f"Not enough remaining reports for K3 on {dataset}")
    k3_idxs = rng.sample(remaining, k3)
    for idx in k3_idxs:
        chosen.add(idx)
        r = reports[idx]
        samples.append(
            sample_record(
                uid=r["uid"],
                dataset=dataset,
                report_id=r["report_id"],
                report_text=r["report_text"],
                source_refs=r.get("source_refs", {}),
                sampling_method="random",
                complexity_score=scores[idx],
                split=r.get("split"),
                extra={k: r[k] for k in ("study_id",) if k in r},
            )
        )

    logger.info(
        "%s sampled %d (K1=%d K2=%d K3=%d)",
        dataset,
        len(samples),
        k1,
        len(k2_picks),
        k3,
    )
    return samples


def top_up_samples_for_strata(
    reports: List[dict],
    existing_samples: List[dict],
    *,
    dataset: str,
    new_strata: Dict[str, List[int]],
    complexity_scores: Sequence[float],
    max_new: int,
) -> List[dict]:
    """Append-only top-up: pick highest-complexity unused reports for new strata via round-robin."""
    existing_uids = {s["uid"] for s in existing_samples}
    uid_to_idx = {r["uid"]: i for i, r in enumerate(reports)}
    already = {uid_to_idx[u] for u in existing_uids if u in uid_to_idx}
    strata = {
        name: sorted([i for i in idxs if i not in already], key=lambda i: complexity_scores[i], reverse=True)
        for name, idxs in new_strata.items()
    }
    strata = {k: v for k, v in strata.items() if v}
    picks = _round_robin_sample(strata, max_new, already)
    out: List[dict] = []
    for idx, stratum in picks:
        r = reports[idx]
        out.append(
            sample_record(
                uid=r["uid"],
                dataset=dataset,
                report_id=r["report_id"],
                report_text=r["report_text"],
                source_refs=r.get("source_refs", {}),
                sampling_method="stratified_topup",
                complexity_score=float(complexity_scores[idx]),
                split=r.get("split"),
                strata_labels={"stratum": stratum},
            )
        )
    return out
