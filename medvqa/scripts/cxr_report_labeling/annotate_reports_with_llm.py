"""LLM-annotate sampled reports × classes (notebook-friendly, capped).

Supports interactive Gemini calls and pause-gated Gemini hybrid Batch:
  estimate / dry_run → batch_prepare → batch_submit → batch_collect
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    build_class_registry,
    load_system_prompt_text,
)
from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_cache_run_dir,
    get_results_run_dir,
    get_sample_annotations_path,
)
from medvqa.datasets.cxr_report_labeling.schemas import (
    annotation_config_fingerprint,
    annotation_config_slug,
    annotation_content_key,
    annotation_query_hash,
    build_nli_query,
    report_text_hash,
)
from medvqa.utils.files_utils import load_json, load_jsonl, save_json
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.openai_api_utils import (
    GPT_IS_ACTING_WEIRD_REGEX,
    orchestrate_api_calls,
    resolve_gemini_reasoning_effort,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"

_POSSIBLE_LABELS = {
    "definitely true",
    "likely true",
    "unknown",
    "likely false",
    "definitely false",
}

_BATCH_MODES = frozenset({"prepare", "submit", "collect"})

_ANNOTATION_JSON_KEYS = (
    "relevant_quote",
    "reasoning",
    "uses_hedging_language",
    "label",
)

_CODE_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    m = _CODE_FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    # Partial / noisy fences (common with LLMs)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _unescape_json_string_content(value: str) -> str:
    """Unescape common JSON string escapes; leave literal quotes as-is."""
    out = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == "r":
                out.append("\r")
            elif nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            elif nxt == "/":
                out.append("/")
            else:
                out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_annotation_fields_loosely(blob: str) -> dict:
    """Extract the four annotation fields when json.loads fails (e.g. raw quotes in values)."""
    out = {}
    for i, key in enumerate(_ANNOTATION_JSON_KEYS):
        m = re.search(rf'"{re.escape(key)}"\s*:\s*', blob)
        if not m:
            raise ValueError(f"Missing key {key!r} in model output")
        rest = blob[m.end() :].lstrip()
        if not rest.startswith('"'):
            raise ValueError(f"Expected string value for {key!r}")
        rest = rest[1:]  # drop opening quote
        if i + 1 < len(_ANNOTATION_JSON_KEYS):
            next_key = _ANNOTATION_JSON_KEYS[i + 1]
            end_m = re.search(rf'"\s*,\s*"{re.escape(next_key)}"', rest)
            if not end_m:
                raise ValueError(f"Could not find end of value for {key!r}")
            raw = rest[: end_m.start()]
        else:
            end_m = re.search(r'"\s*}', rest)
            if not end_m:
                raise ValueError(f"Could not find end of value for {key!r}")
            raw = rest[: end_m.start()]
        out[key] = _unescape_json_string_content(raw)
    return out


def _load_annotation_json_obj(text: str) -> dict:
    cleaned = _strip_code_fences(text)
    start_idx = cleaned.index("{")
    end_idx = cleaned.rindex("}")
    blob = cleaned[start_idx : end_idx + 1]
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return _extract_annotation_fields_loosely(blob)


def parse_llm_model_output(text: str) -> dict:
    assert isinstance(text, str)
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    json_obj = _load_annotation_json_obj(text)
    for key in _ANNOTATION_JSON_KEYS:
        assert key in json_obj, f"Missing {key} in {text}"
    label = str(json_obj["label"]).strip().lower()
    assert label in _POSSIBLE_LABELS, f"Invalid label: {label}"
    return {
        "relevant_quote": str(json_obj["relevant_quote"]).strip(),
        "reasoning": str(json_obj["reasoning"]).strip(),
        "uses_hedging_language": str(json_obj["uses_hedging_language"]).strip(),
        "label": label,
    }


def _config_extra_with_reasoning(
    config_extra: Optional[str], reasoning_effort: Optional[str]
) -> Optional[str]:
    parts: List[str] = []
    if config_extra and str(config_extra).strip():
        parts.append(str(config_extra).strip())
    if reasoning_effort:
        token = f"re{reasoning_effort}"
        if token not in parts:
            parts.append(token)
    return "_".join(parts) or None


def _annotations_path(
    run_id: str,
    model_name: str,
    *,
    system_prompt_hash: str,
    temperature: float = 0.0,
    max_tokens_per_request: int = 2048,
    config_extra: Optional[str] = None,
) -> Path:
    slug = annotation_config_slug(
        system_prompt_hash,
        temperature=temperature,
        max_tokens_per_request=max_tokens_per_request,
        extra=config_extra,
    )
    path = Path(
        get_sample_annotations_path(run_id, model_name, slug)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _batch_dir(run_id: str) -> Path:
    path = Path(get_cache_run_dir(run_id)) / "api_tmp" / "gemini" / "batch_api"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_job_pointer_path(run_id: str, model_name: str) -> Path:
    safe_model = model_name.replace("/", "_")
    return _batch_dir(run_id) / f"latest_job_{safe_model}.json"


def _parsed_response_ok(parsed: object) -> bool:
    if not isinstance(parsed, dict):
        return False
    label = parsed.get("label")
    return isinstance(label, str) and label.strip().lower() in _POSSIBLE_LABELS


def _load_done_query_hashes(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    for item in load_jsonl(str(path)):
        if not _parsed_response_ok(item.get("parsed_response")):
            continue
        meta = item.get("metadata") or {}
        key = meta.get("query_hash")
        if key:
            done.add(key)
    return done


def _load_skip_content_keys(paths: Optional[Sequence[str]]) -> Set[str]:
    skip: Set[str] = set()
    if not paths:
        return skip
    for path in paths:
        if not path or not os.path.exists(path):
            logger.warning("skip_annotation_path missing: %s", path)
            continue
        for item in load_jsonl(path):
            if not _parsed_response_ok(item.get("parsed_response")):
                continue
            meta = item.get("metadata") or {}
            ck = meta.get("content_key")
            if ck:
                skip.add(ck)
    return skip


def _load_query_hash_cache(path: Path) -> Dict[str, dict]:
    """query_hash → {parsed_response, metadata_template} for local fan-out."""
    cache: Dict[str, dict] = {}
    if not path.exists():
        return cache
    for item in load_jsonl(str(path)):
        parsed = item.get("parsed_response")
        if not _parsed_response_ok(parsed):
            continue
        meta = item.get("metadata") or {}
        qh = meta.get("query_hash")
        if qh and qh not in cache:
            cache[qh] = {"parsed_response": parsed, "metadata": meta}
    return cache


def _existing_uid_class_pairs(path: Path) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    if not path.exists():
        return pairs
    for item in load_jsonl(str(path)):
        if not _parsed_response_ok(item.get("parsed_response")):
            continue
        meta = item.get("metadata") or {}
        uid = meta.get("uid")
        class_id = meta.get("class_id")
        if uid and class_id:
            pairs.add((uid, class_id))
    return pairs


def build_annotation_jobs(
    samples: List[dict],
    registry: ClassRegistry,
    model_name: str,
    *,
    config_fingerprint: str,
    class_ids: Optional[Sequence[str]] = None,
    uids: Optional[Sequence[str]] = None,
    already_done_query_hashes: Optional[Set[str]] = None,
    skip_content_keys: Optional[Set[str]] = None,
) -> Tuple[List[dict], dict]:
    """
    Build unique API jobs keyed by query_hash.

    Returns (jobs, stats) where each job has query, custom_id, metadata (with fanout).
    """
    already_done_query_hashes = already_done_query_hashes or set()
    skip_content_keys = skip_content_keys or set()
    uid_filter = set(uids) if uids is not None else None
    class_filter = list(class_ids) if class_ids is not None else registry.class_ids()
    by_id = registry.by_id()

    # query_hash → job accumulator
    by_qh: Dict[str, dict] = {}
    n_uid_intents = 0
    n_skipped_done = 0
    n_skipped_content = 0

    for sample in samples:
        if uid_filter is not None and sample["uid"] not in uid_filter:
            continue
        rhash = sample.get("report_text_hash") or report_text_hash(sample["report_text"])
        for class_id in class_filter:
            entry = by_id[class_id]
            n_uid_intents += 1
            content_key = annotation_content_key(
                rhash, entry.prompt_hash, registry.system_prompt_hash
            )
            if content_key in skip_content_keys:
                n_skipped_content += 1
                continue
            query_hash = annotation_query_hash(
                content_key, model_name, config_fingerprint
            )
            if query_hash in already_done_query_hashes:
                n_skipped_done += 1
                continue
            query = build_nli_query(sample["report_text"], entry.prompt_text)
            job = by_qh.get(query_hash)
            if job is None:
                job = {
                    "query": query,
                    "custom_id": query_hash,
                    "metadata": {
                        "query_hash": query_hash,
                        "content_key": content_key,
                        "class_id": class_id,
                        "prompt_hash": entry.prompt_hash,
                        "system_prompt_hash": registry.system_prompt_hash,
                        "model_name": model_name,
                        "config_fingerprint": config_fingerprint,
                        "report_text_hash": rhash,
                        "query": query,
                        "fanout": [],
                    },
                }
                by_qh[query_hash] = job
            job["metadata"]["fanout"].append(
                {"uid": sample["uid"], "dataset": sample["dataset"]}
            )

    jobs = list(by_qh.values())
    # Deduplicate fanout uids within each job
    for job in jobs:
        seen_uids: Set[str] = set()
        uniq = []
        for item in job["metadata"]["fanout"]:
            if item["uid"] in seen_uids:
                continue
            seen_uids.add(item["uid"])
            uniq.append(item)
        job["metadata"]["fanout"] = uniq

    stats = {
        "n_uid_intents": n_uid_intents,
        "n_unique_queries": len(jobs),
        "n_skipped_done_query_hash": n_skipped_done,
        "n_skipped_skip_content_key": n_skipped_content,
    }
    return jobs, stats


def ensure_local_fanout_rows(
    samples: List[dict],
    registry: ClassRegistry,
    model_name: str,
    *,
    save_path: Path,
    config_fingerprint: str,
    class_ids: Optional[Sequence[str]] = None,
    uids: Optional[Sequence[str]] = None,
) -> dict:
    """Append uid rows for query_hashes already cached but missing this sample uid."""
    cache = _load_query_hash_cache(save_path)
    if not cache:
        return {"n_fanout_appended": 0}
    existing = _existing_uid_class_pairs(save_path)
    uid_filter = set(uids) if uids is not None else None
    class_filter = list(class_ids) if class_ids is not None else registry.class_ids()
    by_id = registry.by_id()
    appended = 0
    with open(save_path, "a", encoding="utf-8") as f:
        for sample in samples:
            if uid_filter is not None and sample["uid"] not in uid_filter:
                continue
            rhash = sample.get("report_text_hash") or report_text_hash(
                sample["report_text"]
            )
            for class_id in class_filter:
                if (sample["uid"], class_id) in existing:
                    continue
                entry = by_id[class_id]
                content_key = annotation_content_key(
                    rhash, entry.prompt_hash, registry.system_prompt_hash
                )
                query_hash = annotation_query_hash(
                    content_key, model_name, config_fingerprint
                )
                hit = cache.get(query_hash)
                if hit is None:
                    continue
                meta = {
                    k: v
                    for k, v in hit["metadata"].items()
                    if k not in ("fanout", "fanout_uids", "uid", "dataset")
                }
                meta["uid"] = sample["uid"]
                meta["dataset"] = sample["dataset"]
                meta["class_id"] = class_id
                meta["query_hash"] = query_hash
                meta["content_key"] = content_key
                row = {"metadata": meta, "parsed_response": hit["parsed_response"]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.add((sample["uid"], class_id))
                appended += 1
    if appended:
        logger.info("Appended %d local fan-out rows to %s", appended, save_path)
    return {"n_fanout_appended": appended}


def estimate_jobs(n_jobs: int, tokens_per_job: int = 800) -> dict:
    return {
        "n_jobs": n_jobs,
        "approx_input_tokens": n_jobs * tokens_per_job,
        "note": "Rough estimate only; actual cost depends on model pricing and report length.",
    }


def _resolve_batch_job_record_path(
    run_id: str,
    model_name: str,
    batch_job_record_path: Optional[str],
) -> str:
    if batch_job_record_path:
        return batch_job_record_path
    pointer = _latest_job_pointer_path(run_id, model_name)
    assert pointer.exists(), (
        f"No batch_job_record_path given and no pointer at {pointer}. "
        "Run batch_mode='prepare' first."
    )
    pointer_obj = load_json(str(pointer))
    path = pointer_obj["batch_job_record_path"]
    assert os.path.exists(path), f"Pointer refers to missing job record: {path}"
    return path


def run_annotation(
    *,
    run_id: str = DEFAULT_RUN_ID,
    samples_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key_name: str = "GOOGLE_API_KEY",
    api_type: str = "gemini",
    max_queries: Optional[int] = 50,
    class_ids: Optional[Sequence[str]] = None,
    uids: Optional[Sequence[str]] = None,
    estimate_only: bool = False,
    dry_run: bool = False,
    batch_mode: Optional[str] = None,
    batch_job_record_path: Optional[str] = None,
    batch_wait: bool = False,
    batch_description: Optional[str] = None,
    batch_poll_interval_s: float = 30.0,
    temperature: float = 0.0,
    max_tokens_per_request: int = 2048,
    max_requests_per_minute: float = 100,
    max_tokens_per_minute: float = 100_000,
    skip_annotation_paths: Optional[Sequence[str]] = None,
    config_extra: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> dict:
    """Annotate samples with LLM (content-addressed query_hash cache).

    batch_mode:
      - None: interactive (throttled) API calls
      - \"prepare\": write local OpenAI-format batch JSONL (no API)
      - \"submit\": upload + create Gemini batch (uses latest prepare unless path given)
      - \"collect\": poll/download/parse into sample_annotations.jsonl
    """
    setup_logging()
    if batch_mode is not None and batch_mode not in _BATCH_MODES:
        raise ValueError(
            f"batch_mode must be one of {sorted(_BATCH_MODES)} or None, got {batch_mode!r}"
        )
    if batch_mode is not None and api_type != "gemini":
        raise ValueError("batch_mode currently only supported for api_type='gemini'")

    results_dir = get_results_run_dir(run_id)
    if samples_path is None:
        samples_path = os.path.join(results_dir, "samples.jsonl")
    samples = load_jsonl(samples_path)
    registry_path = os.path.join(results_dir, "class_registry.json")
    if os.path.exists(registry_path):
        registry = ClassRegistry.load(registry_path)
    else:
        registry = build_class_registry()
        registry.save(registry_path)

    if api_type == "gemini":
        reasoning_effort = resolve_gemini_reasoning_effort(
            model_name, reasoning_effort
        )
    elif reasoning_effort is not None and not str(reasoning_effort).strip():
        reasoning_effort = None
    config_extra = _config_extra_with_reasoning(config_extra, reasoning_effort)
    config_fingerprint = annotation_config_fingerprint(
        temperature=temperature,
        max_tokens_per_request=max_tokens_per_request,
        extra=config_extra,
    )
    config_slug = annotation_config_slug(
        registry.system_prompt_hash,
        temperature=temperature,
        max_tokens_per_request=max_tokens_per_request,
        extra=config_extra,
    )
    save_path = _annotations_path(
        run_id,
        model_name,
        system_prompt_hash=registry.system_prompt_hash,
        temperature=temperature,
        max_tokens_per_request=max_tokens_per_request,
        config_extra=config_extra,
    )
    tmp_dir = os.path.join(get_cache_run_dir(run_id), "api_tmp")
    pending_path = save_path.with_suffix(".pending.jsonl")

    # ----- collect / submit: do not rebuild the job list from samples -----
    if batch_mode in ("submit", "collect"):
        job_record_path = _resolve_batch_job_record_path(
            run_id, model_name, batch_job_record_path
        )
        # Prefer save_path from pointer if present (matches prepare config).
        pointer = _latest_job_pointer_path(run_id, model_name)
        if pointer.exists():
            pointer_obj = load_json(str(pointer))
            if pointer_obj.get("save_path"):
                save_path = Path(pointer_obj["save_path"])
                pending_path = Path(
                    pointer_obj.get("pending_path")
                    or str(save_path.with_suffix(".pending.jsonl"))
                )
        info = {
            "n_samples": len(samples),
            "save_path": str(save_path),
            "config_slug": config_slug,
            "batch_mode": batch_mode,
            "batch_job_record_path": job_record_path,
        }
        if batch_mode == "submit":
            batch_info = orchestrate_api_calls(
                use_batch_api=True,
                api_type="gemini",
                batch_stage="submit",
                batch_job_record_path=job_record_path,
                api_key_name=api_key_name,
                batch_description=batch_description,
                tmp_dir=tmp_dir,
                model_name=model_name,
            )
            info["batch"] = batch_info
            return info

        before_count = len(load_jsonl(str(save_path))) if save_path.exists() else 0
        # Ensure job record points at pending metadata for custom_id join / fan-out.
        job_rec = load_json(job_record_path)
        if not job_rec.get("pending_metadata_path") and pending_path.exists():
            job_rec["pending_metadata_path"] = str(pending_path)
            save_json(job_rec, job_record_path)
        batch_info = orchestrate_api_calls(
            use_batch_api=True,
            api_type="gemini",
            batch_stage="collect",
            batch_job_record_path=job_record_path,
            api_key_name=api_key_name,
            parse_output=parse_llm_model_output,
            save_filepath=str(save_path),
            batch_wait=batch_wait,
            batch_poll_interval_s=batch_poll_interval_s,
            tmp_dir=tmp_dir,
            model_name=model_name,
        )
        info["batch"] = batch_info
        after_count = len(load_jsonl(str(save_path))) if save_path.exists() else 0
        info["n_rows_after"] = after_count
        info["n_new_rows"] = after_count - before_count
        return info

    # Local fan-out for query_hashes already done but missing sample uids.
    fanout_info = ensure_local_fanout_rows(
        samples,
        registry,
        model_name,
        save_path=save_path,
        config_fingerprint=config_fingerprint,
        class_ids=class_ids,
        uids=uids,
    )

    done = _load_done_query_hashes(save_path)
    skip_keys = _load_skip_content_keys(skip_annotation_paths)
    jobs, job_stats = build_annotation_jobs(
        samples,
        registry,
        model_name=model_name,
        config_fingerprint=config_fingerprint,
        class_ids=class_ids,
        uids=uids,
        already_done_query_hashes=done,
        skip_content_keys=skip_keys,
    )
    if max_queries is not None:
        jobs = jobs[:max_queries]

    info = {
        "n_samples": len(samples),
        "n_already_done_query_hashes": len(done),
        "n_skip_content_keys": len(skip_keys),
        "n_uid_intents": job_stats["n_uid_intents"],
        "n_jobs_queued": len(jobs),
        "n_unique_queries": len(jobs),
        "job_stats": job_stats,
        "fanout": fanout_info,
        "save_path": str(save_path),
        "config_slug": config_slug,
        "config_fingerprint": config_fingerprint,
        "estimate": estimate_jobs(len(jobs)),
        "batch_mode": batch_mode,
        "max_tokens_per_request": max_tokens_per_request,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }
    logger.info("%s", info)
    if estimate_only or dry_run or not jobs:
        if dry_run and jobs:
            logger.info("Dry run example query:\n%s", jobs[0]["query"][:500])
        return info

    with open(pending_path, "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    info["pending_path"] = str(pending_path)

    texts = [j["query"] for j in jobs]
    custom_ids = [j["custom_id"] for j in jobs]
    system_instructions = load_system_prompt_text()
    description = batch_description or f"cxr_annotate_{run_id}_{model_name}"

    if batch_mode == "prepare":
        batch_info = orchestrate_api_calls(
            texts=texts,
            system_instructions=system_instructions,
            model_name=model_name,
            api_key_name=api_key_name,
            api_type="gemini",
            use_batch_api=True,
            batch_stage="prepare",
            batch_description=description,
            max_tokens_per_request=max_tokens_per_request,
            temperature=temperature,
            tmp_dir=tmp_dir,
            save_filepath=str(save_path),
            custom_ids=custom_ids,
            pending_metadata_path=str(pending_path),
            reasoning_effort=reasoning_effort,
        )
        pointer = {
            "batch_job_record_path": batch_info["batch_job_record_path"],
            "batch_requests_filepath": batch_info["batch_requests_filepath"],
            "n_requests": batch_info["n_requests"],
            "model_name": model_name,
            "config_slug": config_slug,
            "config_fingerprint": config_fingerprint,
            "pending_path": str(pending_path),
            "save_path": str(save_path),
        }
        pointer_path = _latest_job_pointer_path(run_id, model_name)
        save_json(pointer, str(pointer_path))
        info["batch"] = batch_info
        info["latest_job_pointer"] = str(pointer_path)
        info["batch_job_record_path"] = batch_info["batch_job_record_path"]
        info["batch_requests_filepath"] = batch_info["batch_requests_filepath"]
        logger.info(
            "Prepared batch JSONL at %s — inspect before submit",
            batch_info["batch_requests_filepath"],
        )
        return info

    # Interactive (non-batch): call API once per unique query, then fan-out locally.
    before_count = len(load_jsonl(str(save_path))) if save_path.exists() else 0
    # Temporary file for raw unique responses; we fan-out into save_path.
    raw_path = save_path.with_suffix(".interactive_raw.jsonl")
    if raw_path.exists():
        raw_path.unlink()

    custom_id_to_job = {j["custom_id"]: j for j in jobs}

    # Interactive path only stores query in metadata; map back via pending.
    orchestrate_api_calls(
        texts=texts,
        system_instructions=system_instructions,
        model_name=model_name,
        api_key_name=api_key_name,
        api_type=api_type,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        max_tokens_per_request=max_tokens_per_request,
        temperature=temperature,
        parse_output=parse_llm_model_output,
        save_filepath=str(raw_path),
        tmp_dir=tmp_dir,
        reasoning_effort=reasoning_effort,
    )

    n_appended = 0
    with open(save_path, "a", encoding="utf-8") as out_f:
        for row in load_jsonl(str(raw_path)):
            meta = row.get("metadata") or {}
            q = meta.get("query")
            # Match by query text to pending job (unique within chunk).
            job = next((j for j in jobs if j["query"] == q), None)
            if job is None:
                logger.warning("Interactive response query not found in pending jobs")
                continue
            parsed = row.get("parsed_response")
            for item in job["metadata"]["fanout"]:
                row_meta = {
                    k: v
                    for k, v in job["metadata"].items()
                    if k not in ("fanout", "fanout_uids")
                }
                row_meta["uid"] = item["uid"]
                row_meta["dataset"] = item["dataset"]
                out_f.write(
                    json.dumps(
                        {"metadata": row_meta, "parsed_response": parsed},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_appended += 1
    if raw_path.exists():
        raw_path.unlink()

    after_count = len(load_jsonl(str(save_path))) if save_path.exists() else 0
    info["n_rows_after"] = after_count
    info["n_new_rows"] = after_count - before_count
    info["n_fanout_appended"] = n_appended
    del custom_id_to_job  # silence unused if jobs empty edge
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api_key_name", default="GOOGLE_API_KEY")
    parser.add_argument("--api_type", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--uids", nargs="+", default=None)
    parser.add_argument("--estimate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--batch_mode",
        default=None,
        choices=["prepare", "submit", "collect"],
        help="Gemini hybrid batch stage (omit for interactive calls)",
    )
    parser.add_argument("--batch_job_record_path", default=None)
    parser.add_argument(
        "--batch_wait",
        action="store_true",
        help="On collect, poll until the batch finishes",
    )
    parser.add_argument("--batch_description", default=None)
    parser.add_argument(
        "--max_tokens_per_request",
        type=int,
        default=2048,
        help="Completion token budget per annotation (raise if long reports truncate JSON)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip_annotation_paths",
        nargs="+",
        default=None,
        help="Prior annotation jsonl(s); skip successful content_keys",
    )
    parser.add_argument(
        "--config_extra",
        default=None,
        help="Optional extra token in config fingerprint/slug (e.g. reasoning)",
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        help=(
            "OpenAI-compat reasoning_effort (maps to Gemini 3 thinking_level). "
            "Default: minimal for gemini-3.5-flash-lite; omitted otherwise."
        ),
    )
    args = parser.parse_args()
    run_annotation(
        run_id=args.run_id,
        samples_path=args.samples_path,
        model_name=args.model_name,
        api_key_name=args.api_key_name,
        api_type=args.api_type,
        max_queries=args.max_queries,
        class_ids=args.classes,
        uids=args.uids,
        estimate_only=args.estimate_only,
        dry_run=args.dry_run,
        batch_mode=args.batch_mode,
        batch_job_record_path=args.batch_job_record_path,
        batch_wait=args.batch_wait,
        batch_description=args.batch_description,
        max_tokens_per_request=args.max_tokens_per_request,
        temperature=args.temperature,
        skip_annotation_paths=args.skip_annotation_paths,
        config_extra=args.config_extra,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
