"""LLM-annotate sampled reports × classes (notebook-friendly, capped)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from medvqa.datasets.cxr_report_labeling.class_registry import (
    ClassRegistry,
    build_class_registry,
    load_system_prompt_text,
)
from medvqa.datasets.cxr_report_labeling.paths import DEFAULT_RUN_ID, get_cache_run_dir, get_results_run_dir
from medvqa.datasets.cxr_report_labeling.schemas import (
    annotation_resume_key,
    build_nli_query,
)
from medvqa.utils.files_utils import load_jsonl
from medvqa.utils.logging_utils import setup_logging
from medvqa.utils.openai_api_utils import GPT_IS_ACTING_WEIRD_REGEX, orchestrate_api_calls

logger = logging.getLogger(__name__)

_POSSIBLE_LABELS = {
    "definitely true",
    "likely true",
    "unknown",
    "likely false",
    "definitely false",
}


def parse_llm_model_output(text: str) -> dict:
    assert isinstance(text, str)
    if GPT_IS_ACTING_WEIRD_REGEX.search(text):
        raise RuntimeError(f"GPT is protesting: {text}")
    start_idx = text.index("{")
    end_idx = text.rindex("}")
    json_obj = json.loads(text[start_idx : end_idx + 1])
    for key in ("relevant_quote", "reasoning", "uses_hedging_language", "label"):
        assert key in json_obj, f"Missing {key} in {text}"
    label = str(json_obj["label"]).strip().lower()
    assert label in _POSSIBLE_LABELS, f"Invalid label: {label}"
    return {
        "relevant_quote": str(json_obj["relevant_quote"]).strip(),
        "reasoning": str(json_obj["reasoning"]).strip(),
        "uses_hedging_language": str(json_obj["uses_hedging_language"]).strip(),
        "label": label,
    }


def _annotations_path(run_id: str, model_name: str) -> Path:
    cache_dir = get_cache_run_dir(run_id)
    safe_model = model_name.replace("/", "_")
    path = Path(cache_dir) / "llm_annotations" / safe_model / "sample_annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_done_keys(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    for item in load_jsonl(str(path)):
        meta = item.get("metadata") or {}
        key = meta.get("resume_key")
        if key:
            done.add(key)
    return done


def build_annotation_jobs(
    samples: List[dict],
    registry: ClassRegistry,
    model_name: str,
    class_ids: Optional[Sequence[str]] = None,
    uids: Optional[Sequence[str]] = None,
    already_done: Optional[Set[str]] = None,
) -> List[dict]:
    """Each job: query text + metadata including resume_key."""
    already_done = already_done or set()
    uid_filter = set(uids) if uids is not None else None
    class_filter = list(class_ids) if class_ids is not None else registry.class_ids()
    by_id = registry.by_id()
    jobs = []
    for sample in samples:
        if uid_filter is not None and sample["uid"] not in uid_filter:
            continue
        for class_id in class_filter:
            entry = by_id[class_id]
            resume_key = annotation_resume_key(
                sample["uid"],
                class_id,
                entry.prompt_hash,
                registry.system_prompt_hash,
                model_name,
            )
            if resume_key in already_done:
                continue
            query = build_nli_query(sample["report_text"], entry.prompt_text)
            jobs.append(
                {
                    "query": query,
                    "metadata": {
                        "resume_key": resume_key,
                        "uid": sample["uid"],
                        "dataset": sample["dataset"],
                        "class_id": class_id,
                        "prompt_hash": entry.prompt_hash,
                        "system_prompt_hash": registry.system_prompt_hash,
                        "model_name": model_name,
                    },
                }
            )
    return jobs


def estimate_jobs(n_jobs: int, tokens_per_job: int = 800) -> dict:
    return {
        "n_jobs": n_jobs,
        "approx_input_tokens": n_jobs * tokens_per_job,
        "note": "Rough estimate only; actual cost depends on model pricing and report length.",
    }


def run_annotation(
    *,
    run_id: str = DEFAULT_RUN_ID,
    samples_path: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-lite-preview-09-2025",
    api_key_name: str = "GOOGLE_API_KEY",
    api_type: str = "gemini",
    max_queries: Optional[int] = 50,
    class_ids: Optional[Sequence[str]] = None,
    uids: Optional[Sequence[str]] = None,
    estimate_only: bool = False,
    dry_run: bool = False,
    temperature: float = 0.0,
    max_tokens_per_request: int = 512,
    max_requests_per_minute: float = 100,
    max_tokens_per_minute: float = 100_000,
) -> dict:
    setup_logging()
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

    save_path = _annotations_path(run_id, model_name)
    done = _load_done_keys(save_path)
    jobs = build_annotation_jobs(
        samples,
        registry,
        model_name=model_name,
        class_ids=class_ids,
        uids=uids,
        already_done=done,
    )
    if max_queries is not None:
        jobs = jobs[:max_queries]

    info = {
        "n_samples": len(samples),
        "n_already_done_keys": len(done),
        "n_jobs_queued": len(jobs),
        "save_path": str(save_path),
        "estimate": estimate_jobs(len(jobs)),
    }
    logger.info("%s", info)
    if estimate_only or dry_run or not jobs:
        if dry_run and jobs:
            logger.info("Dry run example query:\n%s", jobs[0]["query"][:500])
        return info

    # orchestrate_api_calls expects texts; we save richer metadata via a wrapper
    # Store metadata alongside by writing a sidecar map custom_id — simplest approach:
    # pass queries and after the fact we cannot recover metadata from query alone if prompts collide.
    # So we embed a sentinel and also write a pending_jobs.jsonl before calling.
    pending_path = save_path.with_suffix(".pending.jsonl")
    with open(pending_path, "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    texts = [j["query"] for j in jobs]
    system_instructions = load_system_prompt_text()

    # Custom parse that attaches metadata by matching query string
    query_to_meta = {j["query"]: j["metadata"] for j in jobs}

    def parse_and_tag(text: str) -> dict:
        parsed = parse_llm_model_output(text)
        return parsed

    # Monkey-patch: orchestrate saves {parsed, metadata: {query: ...}}
    # We'll post-process the jsonl to inject resume metadata.
    before_count = len(load_jsonl(str(save_path))) if save_path.exists() else 0

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
        parse_output=parse_and_tag,
        save_filepath=str(save_path),
        tmp_dir=os.path.join(get_cache_run_dir(run_id), "api_tmp"),
    )

    # Enrich newly appended rows with structured metadata
    rows = load_jsonl(str(save_path))
    enriched = []
    for row in rows:
        meta = dict(row.get("metadata") or {})
        q = meta.get("query")
        if q in query_to_meta:
            meta.update(query_to_meta[q])
        row = dict(row)
        row["metadata"] = meta
        enriched.append(row)
    with open(save_path, "w", encoding="utf-8") as f:
        for row in enriched:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    info["n_rows_after"] = len(enriched)
    info["n_new_rows"] = len(enriched) - before_count
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--model_name", default="gemini-2.5-flash-lite-preview-09-2025")
    parser.add_argument("--api_key_name", default="GOOGLE_API_KEY")
    parser.add_argument("--api_type", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--uids", nargs="+", default=None)
    parser.add_argument("--estimate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
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
    )


if __name__ == "__main__":
    main()
