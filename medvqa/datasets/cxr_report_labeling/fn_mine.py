"""Read-only full-corpus FN-mining kit for CXR regex hunt waves.

Join ``full_regex_matches/<dataset>.jsonl`` to report loaders. Load each
dataset's texts at most once per process. Sub-agents must use this module
instead of calling ``load_mimiccxr_reports()`` (or siblings) in a loop.

Also provides sample-gold FN lookup (stratum D) via ``sample_gold_fns`` /
``sample_gold_fns.pkl`` under ``.agent/regex_refine/_fn_mine/``.
"""

from __future__ import annotations

import bisect
import json
import os
import pickle
import random
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from collections.abc import Iterator, Sequence
from pathlib import Path

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    build_class_registry,
)
from medvqa.datasets.cxr_report_labeling.ontology_dag import (
    OntologyDag,
    load_ontology_dag,
)
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_cache_run_dir,
)
from medvqa.datasets.cxr_report_labeling.regex_matching import report_matches_prompt_class
from medvqa.datasets.cxr_report_labeling.report_loaders import (
    DATASET_CHEXPERT_PLUS,
    DATASET_IUXRAY,
    DATASET_MIMICCXR,
    DATASET_REXGRADIENT,
    load_chexpert_plus_reports,
    load_iuxray_reports,
    load_mimiccxr_reports,
    load_rexgradient_reports,
)
from medvqa.datasets.cxr_report_labeling.schemas import make_uid
from medvqa.settings import PROJECT_ROOT, REGULAR_EXPRESSIONS_FOLDER
from medvqa.utils.constants import CANONICAL_CLASS_TO_REGEX_CLASSES
from medvqa.utils.files_utils import load_json

DATASETS = (
    DATASET_MIMICCXR,
    DATASET_CHEXPERT_PLUS,
    DATASET_REXGRADIENT,
    DATASET_IUXRAY,
)

RelatedHits = bool | Sequence[str] | None

_INDEX_CACHE_DIR = Path(PROJECT_ROOT) / ".agent" / "regex_refine" / "_fn_mine"
_RECALL_SCAN_GLOB = "_recall_scan*"
SAMPLE_GOLD_FNS_PATH = _INDEX_CACHE_DIR / "sample_gold_fns.pkl"
FN_MINE_STATUS_PATH = Path(PROJECT_ROOT) / ".agent" / "regex_refine" / "fn_mine_status.json"
SKIP_UNLESS_ASKED = ("tuberculosis", "pulmonary_vascular_congestion")
MINI_TESTS_BY_CLASS_DIR = (
    Path(REGULAR_EXPRESSIONS_FOLDER) / "cxr_class_mini_tests" / "by_class"
)

_index: FullRegexIndex | None = None
_report_store: ReportStore | None = None
_keyword_index: KeywordIndex | None = None
_registry: ClassRegistry | None = None
_dag: OntologyDag | None = None
_sample_gold_fns: dict | None = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_INDEX_TOKEN_LEN = 3


def _clean_text(text: str) -> str:
    return " ".join(text.lower().split())


def _parse_uid(uid: str) -> tuple[str, str]:
    if ":" not in uid:
        raise ValueError(f"uid must be dataset:report_id, got {uid!r}")
    dataset, report_id = uid.split(":", 1)
    return dataset, report_id


def _registry_cached() -> ClassRegistry:
    global _registry
    if _registry is None:
        _registry = build_class_registry()
    return _registry


def _dag_cached() -> OntologyDag:
    global _dag
    if _dag is None:
        _dag = load_ontology_dag()
    return _dag


def ontology_related_class_ids(class_id: str) -> list[str]:
    """Parents ∪ siblings ∪ children, excluding ``class_id``."""
    dag = _dag_cached()
    if class_id not in dag.class_ids:
        raise KeyError(class_id)
    related: set[str] = set()
    parents = list(dag.parents.get(class_id, []))
    related.update(parents)
    related.update(dag.children.get(class_id, []))
    for p in parents:
        related.update(dag.children.get(p, []))
    related.discard(class_id)
    return sorted(related)


def _resolve_related(class_id: str | None, related_hits: RelatedHits) -> set[str] | None:
    if related_hits is None or related_hits is False:
        return None
    if related_hits is True:
        if not class_id:
            raise ValueError("related_hits=True requires regex_negative_for")
        return set(ontology_related_class_ids(class_id))
    return set(related_hits)


class FullRegexIndex:
    """Sparse positive-class index over ``full_regex_matches`` jsonl (no texts)."""

    def __init__(self, run_id: str = DEFAULT_RUN_ID):
        self.run_id = run_id
        self.uid_to_pos: dict[str, frozenset[str]] = {}
        self.uid_to_meta: dict[str, tuple[str, str]] = {}
        self.dataset_uids: dict[str, list[str]] = {}
        self.dataset_zero: dict[str, list[str]] = {}

    def classes_for(self, uid: str) -> frozenset[str]:
        return self.uid_to_pos.get(uid, frozenset())

    def uids_for_dataset(self, dataset: str) -> list[str]:
        if dataset not in self.dataset_uids:
            raise KeyError(f"Unknown dataset {dataset!r}")
        return self.dataset_uids[dataset]


def _match_jsonl_paths(run_id: str) -> dict[str, Path]:
    cache_dir = Path(get_cache_run_dir(run_id)) / "full_regex_matches"
    out: dict[str, Path] = {}
    for dataset in DATASETS:
        path = cache_dir / f"{dataset}.jsonl"
        if path.is_file():
            out[dataset] = path
    if not out:
        raise FileNotFoundError(f"No full_regex_matches jsonl under {cache_dir}")
    return out


def _index_signature(run_id: str, paths: dict[str, Path]) -> dict:
    return {
        "run_id": run_id,
        "files": {
            ds: {"path": str(p), "mtime_ns": p.stat().st_mtime_ns, "size": p.stat().st_size}
            for ds, p in sorted(paths.items())
        },
    }


def _load_index_from_jsonl(run_id: str, paths: dict[str, Path]) -> FullRegexIndex:
    idx = FullRegexIndex(run_id=run_id)
    for dataset, path in paths.items():
        uids: list[str] = []
        zeros: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = row["uid"]
                pos = frozenset(row.get("positive_classes") or [])
                report_id = str(row.get("report_id") or _parse_uid(uid)[1])
                idx.uid_to_pos[uid] = pos
                idx.uid_to_meta[uid] = (dataset, report_id)
                uids.append(uid)
                if not pos:
                    zeros.append(uid)
        idx.dataset_uids[dataset] = uids
        idx.dataset_zero[dataset] = zeros
    return idx


def load_index(*, run_id: str = DEFAULT_RUN_ID, force_reload: bool = False) -> FullRegexIndex:
    global _index
    if _index is not None and not force_reload and _index.run_id == run_id:
        return _index
    paths = _match_jsonl_paths(run_id)
    sig = _index_signature(run_id, paths)
    cache_path = _INDEX_CACHE_DIR / f"index_{run_id}.pkl"
    if not force_reload and cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("signature") == sig:
                raw = payload["index"]
                if isinstance(raw, FullRegexIndex):
                    _index = raw
                else:
                    idx = FullRegexIndex(run_id=raw["run_id"])
                    idx.uid_to_pos = raw["uid_to_pos"]
                    idx.uid_to_meta = raw["uid_to_meta"]
                    idx.dataset_uids = raw["dataset_uids"]
                    idx.dataset_zero = raw["dataset_zero"]
                    _index = idx
                return _index
        except (OSError, pickle.UnpicklingError, KeyError, EOFError, AttributeError, TypeError):
            pass
    idx = _load_index_from_jsonl(run_id, paths)
    _INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "signature": sig,
                    "index": {
                        "run_id": idx.run_id,
                        "uid_to_pos": idx.uid_to_pos,
                        "uid_to_meta": idx.uid_to_meta,
                        "dataset_uids": idx.dataset_uids,
                        "dataset_zero": idx.dataset_zero,
                    },
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except OSError:
        pass
    _index = idx
    return idx


class ReportStore:
    """Load each dataset at most once; then O(1) ``uid → text``."""

    def __init__(self):
        self._by_dataset: dict[str, dict[str, str]] = {}

    def _load_dataset(self, dataset: str) -> dict[str, str]:
        if dataset == DATASET_MIMICCXR:
            records = load_mimiccxr_reports()
        elif dataset == DATASET_CHEXPERT_PLUS:
            records = load_chexpert_plus_reports()
        elif dataset == DATASET_REXGRADIENT:
            records = load_rexgradient_reports()
        elif dataset == DATASET_IUXRAY:
            records = load_iuxray_reports()
        else:
            raise ValueError(dataset)
        return {r["uid"]: r["report_text"] for r in records}

    def texts_for_dataset(self, dataset: str) -> dict[str, str]:
        cached = self._by_dataset.get(dataset)
        if cached is None:
            cached = self._load_dataset(dataset)
            self._by_dataset[dataset] = cached
        return cached

    def text(self, uid: str) -> str:
        dataset, report_id = _parse_uid(uid)
        mapping = self.texts_for_dataset(dataset)
        if uid in mapping:
            return mapping[uid]
        alt = make_uid(dataset, report_id)
        if alt in mapping:
            return mapping[alt]
        raise KeyError(f"No report text for {uid}")


def load_report_store() -> ReportStore:
    global _report_store
    if _report_store is None:
        _report_store = ReportStore()
    return _report_store


def report_text(uid: str) -> str:
    return load_report_store().text(uid)


def _sample_uids(uids: Sequence[str], *, limit: int, seed: int) -> list[str]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    pool = list(uids)
    if len(pool) <= limit:
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, limit)


def _filter_uids(
    dataset: str,
    *,
    regex_negative_for: str | None,
    related_hits: RelatedHits,
    zero_hits: bool,
) -> list[str]:
    idx = load_index()
    if zero_hits:
        pool = list(idx.dataset_zero.get(dataset, []))
    else:
        pool = list(idx.uids_for_dataset(dataset))
    related = _resolve_related(regex_negative_for, related_hits)
    out: list[str] = []
    for uid in pool:
        pos = idx.classes_for(uid)
        if regex_negative_for is not None and regex_negative_for in pos:
            continue
        if related is not None and not (pos & related):
            continue
        out.append(uid)
    return out


def iter_reports(
    dataset: str,
    *,
    regex_negative_for: str | None = None,
    related_hits: RelatedHits = None,
    zero_hits: bool = False,
    limit: int = 50,
    seed: int = 0,
) -> Iterator[dict]:
    """Yield up to ``limit`` reports for one dataset (texts loaded lazily).

    ``related_hits=True`` means ontology parent/sibling/child hit (and X miss
    when ``regex_negative_for`` is set). Pass an explicit class-id list for
    cross-branch mid-node strata. ``zero_hits`` is stratum A.
    """
    idx = load_index()
    store = load_report_store()
    uids = _filter_uids(
        dataset,
        regex_negative_for=regex_negative_for,
        related_hits=related_hits,
        zero_hits=zero_hits,
    )
    chosen = _sample_uids(uids, limit=limit, seed=seed)
    texts = store.texts_for_dataset(dataset)
    for uid in chosen:
        dataset_i, report_id = idx.uid_to_meta.get(uid, _parse_uid(uid))
        text = texts.get(uid)
        if text is None:
            continue
        yield {
            "uid": uid,
            "dataset": dataset_i,
            "report_id": report_id,
            "report_text": text,
            "positive_classes": sorted(idx.classes_for(uid)),
        }


def _term_pattern(term: str) -> re.Pattern[str]:
    parts = [p for p in re.split(r"\s+", term.strip()) if p]
    if not parts:
        raise ValueError("empty keyword term")
    body = r"\s+".join(re.escape(p.lower()) for p in parts)
    # Allow a short inflection suffix so "fracture" hits "fractures".
    return re.compile(rf"\b{body}\w*")


def _compile_term_patterns(terms: Sequence[str]) -> list[tuple[str, re.Pattern[str]]]:
    return [(t, _term_pattern(t)) for t in terms]


def matched_keyword_terms(
    text: str,
    terms: Sequence[str],
    *,
    compiled: Sequence[tuple[str, re.Pattern[str]]] | None = None,
) -> list[str]:
    cleaned = _clean_text(text)
    pairs = list(compiled) if compiled is not None else _compile_term_patterns(terms)
    return [t for t, pat in pairs if pat.search(cleaned)]


def _tokenize_report(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(_clean_text(text)))


def _term_index_anchors(term: str) -> list[str]:
    """Alphanumeric anchors (len ≥ 3) used for inverted-index candidate generation."""
    return [t for t in _TOKEN_RE.findall(term.lower()) if len(t) >= _MIN_INDEX_TOKEN_LEN]


def _postings_from_chunk(
    chunk: list[tuple[int, str]],
) -> dict[str, list[int]]:
    """Worker: map tokens → local uid indices for one chunk of reports."""
    local: dict[str, list[int]] = {}
    for i, text in chunk:
        for tok in _tokenize_report(text):
            local.setdefault(tok, []).append(i)
    return local


class KeywordIndex:
    """Per-dataset token → uid-index postings for fast stratum-C retrieval."""

    def __init__(self, run_id: str = DEFAULT_RUN_ID):
        self.run_id = run_id
        # dataset -> sorted uid list (stable ids into postings)
        self.uid_lists: dict[str, list[str]] = {}
        # dataset -> token -> sorted uid indices
        self.postings: dict[str, dict[str, list[int]]] = {}
        # dataset -> sorted token vocabulary (prefix search)
        self.vocab: dict[str, list[str]] = {}

    def tokens_with_prefix(self, dataset: str, prefix: str) -> list[str]:
        vocab = self.vocab.get(dataset) or []
        if not prefix:
            return []
        i = bisect.bisect_left(vocab, prefix)
        out: list[str] = []
        while i < len(vocab) and vocab[i].startswith(prefix):
            out.append(vocab[i])
            i += 1
        return out

    def uid_indices_for_anchor(self, dataset: str, anchor: str) -> set[int]:
        postings = self.postings.get(dataset) or {}
        hits: set[int] = set()
        for tok in self.tokens_with_prefix(dataset, anchor):
            hits.update(postings.get(tok) or ())
        return hits

    def candidate_uids_for_term(self, dataset: str, term: str) -> set[str] | None:
        """Return candidate uids for ``term``, or None if the term is not indexable."""
        anchors = _term_index_anchors(term)
        if not anchors:
            return None
        uid_list = self.uid_lists.get(dataset) or []
        # Intersect anchors (all must appear as some inflected token prefix).
        common: set[int] | None = None
        for anchor in anchors:
            part_hits = self.uid_indices_for_anchor(dataset, anchor)
            if common is None:
                common = part_hits
            else:
                common &= part_hits
            if not common:
                return set()
        assert common is not None
        return {uid_list[i] for i in common if 0 <= i < len(uid_list)}


def _keyword_index_cache_path(run_id: str) -> Path:
    return _INDEX_CACHE_DIR / f"keyword_index_{run_id}.pkl"


def _keyword_index_signature(run_id: str, store: ReportStore) -> dict:
    match_paths = _match_jsonl_paths(run_id)
    match_sig = _index_signature(run_id, match_paths)
    texts_sig: dict[str, dict] = {}
    for dataset in DATASETS:
        texts = store.texts_for_dataset(dataset)
        texts_sig[dataset] = {
            "n": len(texts),
            "chars": sum(len(t) for t in texts.values()),
        }
    return {"run_id": run_id, "match": match_sig, "texts": texts_sig}


def _merge_postings(parts: list[dict[str, list[int]]]) -> dict[str, list[int]]:
    merged: dict[str, list[int]] = {}
    for part in parts:
        for tok, ids in part.items():
            bucket = merged.setdefault(tok, [])
            bucket.extend(ids)
    for tok, ids in merged.items():
        ids.sort()
        # Chunks are disjoint; still uniquify defensively.
        dedup: list[int] = []
        prev = -1
        for i in ids:
            if i != prev:
                dedup.append(i)
                prev = i
        merged[tok] = dedup
    return merged


def build_keyword_index(
    *,
    run_id: str = DEFAULT_RUN_ID,
    n_workers: int | None = None,
    force: bool = False,
) -> KeywordIndex:
    """Build (or reload) the token inverted index; multiprocess over reports."""
    global _keyword_index
    if (
        not force
        and _keyword_index is not None
        and _keyword_index.run_id == run_id
    ):
        return _keyword_index

    store = load_report_store()
    match_idx = load_index(run_id=run_id)
    sig = _keyword_index_signature(run_id, store)
    cache_path = _keyword_index_cache_path(run_id)
    if not force and cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("signature") == sig:
                raw = payload["index"]
                kw = KeywordIndex(run_id=raw["run_id"])
                kw.uid_lists = raw["uid_lists"]
                kw.postings = raw["postings"]
                kw.vocab = raw["vocab"]
                _keyword_index = kw
                return _keyword_index
        except (OSError, pickle.UnpicklingError, KeyError, EOFError, AttributeError, TypeError):
            pass

    if n_workers is None:
        n_workers = max(1, min(8, os.cpu_count() or 1))

    kw = KeywordIndex(run_id=run_id)
    for dataset in DATASETS:
        texts = store.texts_for_dataset(dataset)
        # Prefer match-index uid order so regex-negative filters stay aligned.
        uid_list = [u for u in match_idx.uids_for_dataset(dataset) if u in texts]
        seen_uids = set(uid_list)
        for u in texts:
            if u not in seen_uids:
                uid_list.append(u)
                seen_uids.add(u)
        kw.uid_lists[dataset] = uid_list

        chunk_size = max(500, (len(uid_list) + n_workers - 1) // max(n_workers, 1))
        chunks: list[list[tuple[int, str]]] = []
        for start in range(0, len(uid_list), chunk_size):
            sl = uid_list[start : start + chunk_size]
            chunks.append([(start + j, texts[u]) for j, u in enumerate(sl)])

        if n_workers <= 1 or len(chunks) <= 1:
            parts = [_postings_from_chunk(c) for c in chunks]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                parts = list(pool.map(_postings_from_chunk, chunks, chunksize=1))
        postings = _merge_postings(parts)
        kw.postings[dataset] = postings
        kw.vocab[dataset] = sorted(postings.keys())

    _INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("wb") as f:
            pickle.dump(
                {
                    "signature": sig,
                    "index": {
                        "run_id": kw.run_id,
                        "uid_lists": kw.uid_lists,
                        "postings": kw.postings,
                        "vocab": kw.vocab,
                    },
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except OSError:
        pass
    _keyword_index = kw
    return _keyword_index


def load_keyword_index(
    *,
    run_id: str = DEFAULT_RUN_ID,
    build_if_missing: bool = True,
    n_workers: int | None = None,
) -> KeywordIndex:
    global _keyword_index
    if _keyword_index is not None and _keyword_index.run_id == run_id:
        return _keyword_index
    cache_path = _keyword_index_cache_path(run_id)
    store = load_report_store()
    sig = _keyword_index_signature(run_id, store)
    if cache_path.is_file():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            if payload.get("signature") == sig:
                raw = payload["index"]
                kw = KeywordIndex(run_id=raw["run_id"])
                kw.uid_lists = raw["uid_lists"]
                kw.postings = raw["postings"]
                kw.vocab = raw["vocab"]
                _keyword_index = kw
                return _keyword_index
        except (OSError, pickle.UnpicklingError, KeyError, EOFError, AttributeError, TypeError):
            pass
    if not build_if_missing:
        raise FileNotFoundError(
            f"Keyword index missing/stale at {cache_path}; "
            "run: python -m medvqa.datasets.cxr_report_labeling.fn_mine --build-keyword-index"
        )
    return build_keyword_index(run_id=run_id, n_workers=n_workers, force=True)


def keyword_candidates(
    dataset: str,
    class_id: str,
    terms: Sequence[str],
    *,
    regex_negative: bool = True,
    limit: int = 50,
    seed: int = 0,
    use_keyword_index: bool = True,
) -> list[dict]:
    """Substring/token retrieval on X-negative reports. Not a second regex.

    Uses a token inverted index when available (built on first call or via
    ``--build-keyword-index``). Term patterns are compiled once per call.
    """
    if not terms:
        raise ValueError("terms must be non-empty")
    idx = load_index()
    store = load_report_store()
    texts = store.texts_for_dataset(dataset)
    compiled = _compile_term_patterns(terms)

    candidate_uids: list[str] | None = None
    if use_keyword_index:
        try:
            kw = load_keyword_index(build_if_missing=True)
        except Exception:
            kw = None
        if kw is not None:
            union: set[str] = set()
            unindexed = 0
            for term in terms:
                part = kw.candidate_uids_for_term(dataset, term)
                if part is None:
                    unindexed += 1
                else:
                    union |= part
            # If every term was indexable, restrict to the union; otherwise fall
            # back to a full scan so short anchors (e.g. "S sign") are not lost.
            if unindexed == 0:
                candidate_uids = [u for u in union if u in texts]

    if candidate_uids is None:
        candidate_uids = [u for u in idx.uids_for_dataset(dataset) if u in texts]

    hits: list[str] = []
    for uid in candidate_uids:
        if regex_negative and class_id in idx.classes_for(uid):
            continue
        text = texts.get(uid)
        if not text:
            continue
        if matched_keyword_terms(text, terms, compiled=compiled):
            hits.append(uid)

    chosen = _sample_uids(hits, limit=limit, seed=seed)
    out: list[dict] = []
    for uid in chosen:
        dataset_i, report_id = idx.uid_to_meta.get(uid, _parse_uid(uid))
        out.append(
            {
                "uid": uid,
                "dataset": dataset_i,
                "report_id": report_id,
                "report_text": texts[uid],
                "positive_classes": sorted(idx.classes_for(uid)),
                "matched_terms": matched_keyword_terms(
                    texts[uid], terms, compiled=compiled
                ),
            }
        )
    return out


def latest_recall_scan_dir() -> Path | None:
    root = Path(PROJECT_ROOT) / ".agent" / "regex_refine"
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for path in root.glob(_RECALL_SCAN_GLOB):
        if path.is_dir() and (path / "metrics_per_class.json").is_file():
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def sample_gold_metrics(
    class_id: str,
    *,
    scan_dir: str | Path | None = None,
) -> dict:
    """Sample-gold P/R from the latest production-OR recall scan (not live jsonl)."""
    path_dir = Path(scan_dir) if scan_dir is not None else latest_recall_scan_dir()
    if path_dir is None:
        raise FileNotFoundError(
            "No _recall_scan*/metrics_per_class.json under .agent/regex_refine/"
        )
    metrics = load_json(str(path_dir / "metrics_per_class.json"))
    if class_id not in metrics:
        raise KeyError(f"{class_id} not in {path_dir / 'metrics_per_class.json'}")
    entry = dict(metrics[class_id])
    entry["class_id"] = class_id
    entry["scan_dir"] = str(path_dir)
    return entry


def sample_gold_fns_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else SAMPLE_GOLD_FNS_PATH


def load_sample_gold_fns(
    *,
    path: str | Path | None = None,
    force_reload: bool = False,
) -> dict:
    """Load the sample-gold FN pickle (LLM+ & hierarchy-aware regex−).

    Default path: ``.agent/regex_refine/_fn_mine/sample_gold_fns.pkl``.
    Rebuild with ``python -m medvqa.datasets.cxr_report_labeling.fn_mine
    --build-sample-gold-fn-index``.
    """
    global _sample_gold_fns
    resolved = sample_gold_fns_path(path)
    if (
        not force_reload
        and _sample_gold_fns is not None
        and Path(_sample_gold_fns.get("meta", {}).get("path") or resolved) == resolved
    ):
        return _sample_gold_fns
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Missing sample-gold FN index at {resolved}. Build with: "
            "python -m medvqa.datasets.cxr_report_labeling.fn_mine "
            "--build-sample-gold-fn-index"
        )
    with resolved.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "fns" not in payload:
        raise ValueError(f"Invalid sample-gold FN pickle: {resolved}")
    meta = dict(payload.get("meta") or {})
    meta["path"] = str(resolved)
    payload = {**payload, "meta": meta}
    _sample_gold_fns = payload
    return payload


def sample_gold_fn_count(class_id: str, *, path: str | Path | None = None) -> int:
    data = load_sample_gold_fns(path=path)
    counts = data.get("fn_counts") or {}
    if class_id in counts:
        return int(counts[class_id])
    by_ds = (data.get("fns") or {}).get(class_id) or {}
    return sum(len(v) for v in by_ds.values())


def sample_gold_fns(
    class_id: str,
    *,
    dataset: str | None = None,
    datasets: Sequence[str] | None = None,
    limit: int | None = None,
    path: str | Path | None = None,
) -> list[dict]:
    """Sample-gold FNs for ``class_id`` (hunt stratum D).

    Each row: ``uid``, ``dataset``, ``label``, ``relevant_quote``, …
    Prefer ``relevant_quote`` for prompt-fidelity judgment; call ``report_text``
    only when needed. Still require ``production_misses`` before appending
    mini-test positives (regex may have moved since the pickle was built).
    """
    data = load_sample_gold_fns(path=path)
    by_ds = (data.get("fns") or {}).get(class_id) or {}
    if dataset is not None and datasets is not None:
        raise ValueError("pass dataset or datasets, not both")
    if dataset is not None:
        order = [dataset]
    elif datasets is not None:
        order = list(datasets)
    else:
        order = [ds for ds in DATASETS if ds in by_ds] + [
            ds for ds in sorted(by_ds) if ds not in DATASETS
        ]
    out: list[dict] = []
    for ds in order:
        for row in by_ds.get(ds) or []:
            item = dict(row)
            item.setdefault("dataset", ds)
            item.setdefault("class_id", class_id)
            out.append(item)
            if limit is not None and len(out) >= limit:
                return out
    return out


def build_sample_gold_fn_index(
    *,
    run_id: str = DEFAULT_RUN_ID,
    model_name: str = "gemini-2.5-flash-lite",
    out_path: str | Path | None = None,
    num_processes: int = 16,
    class_ids: Sequence[str] | None = None,
) -> Path:
    """Recompute sample-gold FN pickle via live hierarchy-aware regex matching.

    Independent of ``full_regex_matches`` precompute. Writes
    ``sample_gold_fns.pkl`` under ``_fn_mine/`` by default.
    """
    from collections import defaultdict
    from datetime import timezone

    from medvqa.datasets.cxr_report_labeling.label_semantics import (
        llm_label_to_binary_positive,
    )
    from medvqa.datasets.cxr_report_labeling.paths import (
        get_results_run_dir,
        resolve_sample_annotations_path,
    )
    from medvqa.datasets.cxr_report_labeling.regex_matching import (
        batch_match_prompt_classes,
    )
    from medvqa.datasets.regular_expressions import cxr_patterns
    from medvqa.scripts.cxr_report_labeling.eval_regex_vs_llm import (
        load_annotations_flexible,
    )
    from medvqa.utils.files_utils import load_jsonl

    global _sample_gold_fns

    resolved_out = sample_gold_fns_path(out_path)
    results_dir = Path(get_results_run_dir(run_id))
    samples_path = results_dir / "samples.jsonl"
    ann_path = resolve_sample_annotations_path(run_id, model_name)

    cxr_patterns.clear_pattern_cache()
    sample_rows = load_jsonl(str(samples_path))
    uids = [s["uid"] for s in sample_rows]
    texts = [s["report_text"] for s in sample_rows]
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}

    registry_path = results_dir / "class_registry.json"
    registry = (
        ClassRegistry.load(str(registry_path))
        if registry_path.is_file()
        else build_class_registry()
    )
    ids = list(class_ids) if class_ids is not None else [
        c for c in registry.class_ids() if registry.by_id()[c].has_regex
    ]

    ann = load_annotations_flexible(ann_path)
    matched_by_class = batch_match_prompt_classes(
        texts, registry, class_ids=ids, num_processes=num_processes
    )

    fns: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, int] = {}
    for cid in ids:
        matched = matched_by_class.get(cid) or set()
        n_fn = 0
        for (uid, class_id), rec in ann.items():
            if class_id != cid:
                continue
            if not llm_label_to_binary_positive(rec.get("label")):
                continue
            idx = uid_to_idx.get(uid)
            if idx is None or idx in matched:
                continue
            n_fn += 1
            ds, _ = _parse_uid(uid)
            fns[cid][ds].append(
                {
                    "uid": uid,
                    "label": rec.get("label"),
                    "relevant_quote": rec.get("relevant_quote"),
                    "uses_hedging_language": rec.get("uses_hedging_language"),
                }
            )
        counts[cid] = n_fn
        fns[cid] = dict(fns[cid])

    payload = {
        "meta": {
            "run_id": run_id,
            "model_name": model_name,
            "samples_path": str(samples_path),
            "annotations_path": str(ann_path),
            "match_source": "live_regex_batch_match_prompt_classes",
            "hierarchy": "composed_or_via_report_matches_prompt_class",
            "n_samples": len(uids),
            "n_annotation_keys": len(ann),
            "n_classes": len(ids),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "path": str(resolved_out),
            "note": (
                "FN = LLM gold positive & hierarchy-aware regex miss on sample. "
                "Independent of full_regex_matches precompute."
            ),
        },
        "fns": dict(fns),
        "fn_counts": counts,
    }
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    with resolved_out.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    _sample_gold_fns = payload
    return resolved_out


def mini_tests_for(class_id: str) -> dict:
    path = MINI_TESTS_BY_CLASS_DIR / f"{class_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"Expected one fixture in {path}")
        data = data[0]
    if data.get("class_id") != class_id:
        raise ValueError(f"class_id mismatch in {path}")
    return data


def production_misses(class_id: str, sentence: str) -> bool:
    """True when current production regex does **not** fire on ``sentence``."""
    ok, _spans = report_matches_prompt_class(sentence, class_id, _registry_cached())
    return not ok


def production_hits(class_id: str, sentence: str) -> tuple[bool, list[str]]:
    return report_matches_prompt_class(sentence, class_id, _registry_cached())


def append_mini_test_positive(
    class_id: str,
    sentence: str,
    *,
    require_miss: bool = True,
) -> Path:
    """Append a short hard positive to ``by_class/<id>.json``. Hunt-phase write."""
    text = " ".join(sentence.split()).strip()
    if not text:
        raise ValueError("empty sentence")
    if require_miss and not production_misses(class_id, text):
        raise ValueError(f"production regex already matches [{class_id}]: {text!r}")
    path = MINI_TESTS_BY_CLASS_DIR / f"{class_id}.json"
    data = mini_tests_for(class_id)
    positives = list(data.get("positives") or [])
    if text in positives:
        return path
    positives.append(text)
    data["positives"] = positives
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _empty_phase() -> dict:
    return {"status": "pending", "wave": None, "stamp": None, "n_accepted": None, "notes": ""}


def _regex_backed_class_ids() -> list[str]:
    return [cid for cid, regs in CANONICAL_CLASS_TO_REGEX_CLASSES.items() if regs is not None]


def _topology_entry(class_id: str) -> dict:
    dag = _dag_cached()
    children = list(dag.children.get(class_id, []))
    descendants = sorted(dag.descendants(class_id))
    return {
        "n_children": len(children),
        "n_descendants": len(descendants),
        "children": children,
        "descendants": descendants,
        "is_leaf": len(children) == 0,
        "skip": class_id in SKIP_UNLESS_ASKED,
        "hunt": _empty_phase(),
        "fit": _empty_phase(),
    }


def load_fn_mine_status(*, path: Path | None = None) -> dict:
    """Per-class hunt/fit tracker (``.agent/regex_refine/fn_mine_status.json``)."""
    path = path or FN_MINE_STATUS_PATH
    if path.is_file():
        status = json.loads(path.read_text(encoding="utf-8"))
    else:
        status = {
            "schema": 1,
            "policy": {
                "hunt_order": "leaves_first_then_toposort_by_n_descendants",
                "fit_after": "hunt_done_and_all_descendants_fit",
            },
            "skip_unless_asked": list(SKIP_UNLESS_ASKED),
            "classes": {},
        }
    classes = status.setdefault("classes", {})
    for cid in _regex_backed_class_ids():
        topo = _topology_entry(cid)
        if cid not in classes:
            classes[cid] = topo
            continue
        entry = classes[cid]
        for key in ("n_children", "n_descendants", "children", "descendants", "is_leaf", "skip"):
            entry[key] = topo[key]
        entry.setdefault("hunt", _empty_phase())
        entry.setdefault("fit", _empty_phase())
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return status


def save_fn_mine_status(status: dict, *, path: Path | None = None) -> Path:
    path = path or FN_MINE_STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _phase_status(entry: dict, phase: str) -> str:
    return (entry.get(phase) or {}).get("status") or "pending"


def mark_fn_mine_phase(
    class_id: str,
    phase: str,
    *,
    status: str,
    wave: str | None = None,
    stamp: str | None = None,
    n_accepted: int | None = None,
    notes: str = "",
) -> dict:
    if phase not in ("hunt", "fit"):
        raise ValueError(phase)
    tracker = load_fn_mine_status()
    if class_id not in tracker["classes"]:
        raise KeyError(class_id)
    tracker["classes"][class_id][phase] = {
        "status": status,
        "wave": wave,
        "stamp": stamp,
        "n_accepted": n_accepted,
        "notes": notes,
    }
    save_fn_mine_status(tracker)
    return tracker["classes"][class_id]


def next_hunt_class_ids(*, limit: int = 5, include_skipped: bool = False) -> list[str]:
    """Leaves with hunt still pending, then mid-nodes whose children are all hunt-done.

    Mid-nodes are not returned until every child has ``hunt.status == done``.
    """
    tracker = load_fn_mine_status()
    ready: list[tuple[int, int, str]] = []
    for cid, entry in tracker["classes"].items():
        if not include_skipped and entry.get("skip"):
            continue
        if _phase_status(entry, "hunt") != "pending":
            continue
        children = entry.get("children") or []
        if children:
            child_entries = tracker["classes"]
            if not all(
                _phase_status(child_entries[c], "hunt") == "done" for c in children
            ):
                continue
        ready.append((entry.get("n_descendants", 0), entry.get("n_children", 0), cid))
    ready.sort()
    return [cid for _d, _c, cid in ready[:limit]]


def next_fit_class_ids(*, limit: int = 5, include_skipped: bool = False) -> list[str]:
    """Hunt-done classes whose every descendant has already been fit (leaves first)."""
    tracker = load_fn_mine_status()
    ready: list[tuple[int, int, str]] = []
    for cid, entry in tracker["classes"].items():
        if not include_skipped and entry.get("skip"):
            continue
        if _phase_status(entry, "hunt") != "done":
            continue
        if _phase_status(entry, "fit") != "pending":
            continue
        descendants = entry.get("descendants") or []
        if any(_phase_status(tracker["classes"][d], "fit") != "done" for d in descendants):
            continue
        ready.append((entry.get("n_descendants", 0), entry.get("n_children", 0), cid))
    ready.sort()
    return [cid for _d, _c, cid in ready[:limit]]


def fn_mine_status_summary() -> dict:
    tracker = load_fn_mine_status()
    counts: dict[str, int] = {
        "n_classes": 0,
        "hunt_pending": 0,
        "hunt_done": 0,
        "fit_pending": 0,
        "fit_done": 0,
        "skipped": 0,
    }
    for entry in tracker["classes"].values():
        counts["n_classes"] += 1
        if entry.get("skip"):
            counts["skipped"] += 1
        hs = _phase_status(entry, "hunt")
        fs = _phase_status(entry, "fit")
        counts["hunt_pending"] += int(hs == "pending")
        counts["hunt_done"] += int(hs == "done")
        counts["fit_pending"] += int(fs == "pending")
        counts["fit_done"] += int(fs == "done")
    return {
        "path": str(FN_MINE_STATUS_PATH),
        "counts": counts,
        "next_hunt": next_hunt_class_ids(limit=10),
        "next_fit": next_fit_class_ids(limit=10),
    }


def smoke_patient33343(*, limit: int = 50) -> dict:
    """Coordinator sanity for the motivating CheXpert FN (no full-corpus keyword scan)."""
    uid = "chexpert_plus:patient33343_5"
    idx = load_index()
    pos = sorted(idx.classes_for(uid))
    text = report_text(uid)
    device_terms = ["fixation", "hardware", "drain", "pectus", "nuss", "bar", "radiopaque"]
    fracture_terms = ["fracture", "sternal", "apparent"]
    wire_terms = ["sternal wire", "sternal wires", "cerclage", "sternal cables"]
    tube_terms = ["drain", "jp", "radiopaque drain"]
    zero_pool = _filter_uids(
        DATASET_CHEXPERT_PLUS,
        regex_negative_for="support_device_present",
        related_hits=None,
        zero_hits=True,
    )
    related_pool = _filter_uids(
        DATASET_CHEXPERT_PLUS,
        regex_negative_for="support_device_present",
        related_hits=["bone_fracture"],
        zero_hits=False,
    )
    classes = (
        "support_device_present",
        "bone_fracture",
        "sternotomy_wires",
        "support_tube_present",
    )
    return {
        "uid": uid,
        "positive_classes": pos,
        "zero_hits": pos == [],
        "in_stratum_a_device": uid in zero_pool,
        "in_stratum_b_device_via_fracture": uid in related_pool,
        "n_stratum_a_chexpert_device": len(zero_pool),
        "text_preview": text[:400],
        "keyword_terms_on_uid": {
            "support_device_present": matched_keyword_terms(text, device_terms),
            "bone_fracture": matched_keyword_terms(text, fracture_terms),
            "sternotomy_wires": matched_keyword_terms(text, wire_terms),
            "support_tube_present": matched_keyword_terms(text, tube_terms),
        },
        "production_misses": {cid: production_misses(cid, text) for cid in classes},
        "sample_gold": {cid: sample_gold_metrics(cid) for cid in classes},
        "mini_test_n_pos": {cid: len(mini_tests_for(cid).get("positives") or []) for cid in classes},
        "iter_reports_cap": {
            "limit": limit,
            "n_yielded": sum(
                1
                for _ in iter_reports(
                    DATASET_CHEXPERT_PLUS,
                    regex_negative_for="support_device_present",
                    zero_hits=True,
                    limit=limit,
                    seed=0,
                )
            ),
        },
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run patient33343_5 sanity check")
    parser.add_argument("--status", action="store_true", help="Print hunt/fit tracker summary")
    parser.add_argument(
        "--build-keyword-index",
        action="store_true",
        help="Build/refresh the token inverted index for stratum-C keyword_candidates",
    )
    parser.add_argument(
        "--build-sample-gold-fn-index",
        action="store_true",
        help=(
            "Build/refresh sample_gold_fns.pkl (LLM+ & regex− on annotated samples) "
            "for hunt stratum D"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process workers for index builders (default: min(8, cpus) for keyword; "
        "16 for sample-gold FN)",
    )
    parser.add_argument(
        "--model_name",
        default="gemini-2.5-flash-lite",
        help="LLM annotation namespace for --build-sample-gold-fn-index",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for --build-sample-gold-fn-index (default: _fn_mine/sample_gold_fns.pkl)",
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        result = smoke_patient33343(limit=args.limit)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.status:
        save_fn_mine_status(load_fn_mine_status())
        print(json.dumps(fn_mine_status_summary(), indent=2, ensure_ascii=False))
        return 0
    if args.build_keyword_index:
        t0 = time.time()
        kw = build_keyword_index(n_workers=args.workers, force=True)
        elapsed = time.time() - t0
        summary = {
            "path": str(_keyword_index_cache_path(kw.run_id)),
            "elapsed_s": round(elapsed, 2),
            "n_workers": args.workers
            if args.workers is not None
            else max(1, min(8, os.cpu_count() or 1)),
            "datasets": {
                ds: {
                    "n_uids": len(kw.uid_lists.get(ds) or []),
                    "n_tokens": len(kw.vocab.get(ds) or []),
                }
                for ds in DATASETS
            },
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    if args.build_sample_gold_fn_index:
        t0 = time.time()
        n_workers = args.workers if args.workers is not None else 16
        out = build_sample_gold_fn_index(
            model_name=args.model_name,
            out_path=args.out,
            num_processes=n_workers,
        )
        data = load_sample_gold_fns(path=out, force_reload=True)
        counts = data.get("fn_counts") or {}
        summary = {
            "path": str(out),
            "elapsed_s": round(time.time() - t0, 2),
            "n_workers": n_workers,
            "n_classes": len(counts),
            "n_fn_total": int(sum(counts.values())),
            "meta": data.get("meta"),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
