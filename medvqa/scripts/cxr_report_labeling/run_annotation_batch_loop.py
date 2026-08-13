"""Tier-1 Gemini Batch annotation loop (prepare → submit w/ 429 backoff → collect).

Designed for overnight Slurm: one chunk at a time under ~3M enqueued-token quota.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from openai import RateLimitError

from medvqa.datasets.cxr_report_labeling.paths import (
    DEFAULT_RUN_ID,
    get_cache_run_dir,
    get_results_run_dir,
)
from medvqa.scripts.cxr_report_labeling.annotate_reports_with_llm import (
    DEFAULT_MODEL_NAME,
    run_annotation,
)
from medvqa.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress_path(run_id: str) -> Path:
    path = (
        Path(get_cache_run_dir(run_id))
        / "api_tmp"
        / "gemini"
        / "batch_api"
        / "loop_progress.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_progress(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


def submit_with_backoff(
    *,
    run_id: str,
    samples_path: Optional[str],
    model_name: str,
    api_key_name: str,
    batch_description: Optional[str],
    backoff_initial_s: float,
    backoff_max_s: float,
    temperature: float = 0.0,
    max_tokens_per_request: int = 2048,
    skip_annotation_paths: Optional[list] = None,
    config_extra: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    """Call batch_mode=submit, retrying on 429 with exponential backoff."""
    sleep_s = backoff_initial_s
    attempt = 0
    n_429 = 0
    while True:
        attempt += 1
        try:
            info = run_annotation(
                run_id=run_id,
                samples_path=samples_path,
                model_name=model_name,
                api_key_name=api_key_name,
                api_type="gemini",
                batch_mode="submit",
                batch_description=batch_description,
                temperature=temperature,
                max_tokens_per_request=max_tokens_per_request,
                skip_annotation_paths=skip_annotation_paths,
                config_extra=config_extra,
                reasoning_effort=reasoning_effort,
            )
            info["submit_attempts"] = attempt
            info["n_429_waits"] = n_429
            return info
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            n_429 += 1
            logger.warning(
                "Submit hit rate limit (attempt=%d n_429=%d); sleeping %.0fs then retry. error=%s",
                attempt,
                n_429,
                sleep_s,
                e,
            )
            time.sleep(sleep_s)
            sleep_s = min(sleep_s * 2.0, backoff_max_s)


def run_batch_loop(
    *,
    run_id: str = DEFAULT_RUN_ID,
    samples_path: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key_name: str = "GEMINI_API_KEY",
    chunk_size: int = 2000,
    max_chunks: Optional[int] = None,
    submit_backoff_initial_s: float = 60.0,
    submit_backoff_max_s: float = 900.0,
    post_collect_sleep_s: float = 60.0,
    batch_poll_interval_s: float = 30.0,
    max_tokens_per_request: int = 2048,
    temperature: float = 0.0,
    skip_annotation_paths: Optional[list] = None,
    config_extra: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    setup_logging()
    if samples_path is None:
        samples_path = os.path.join(get_results_run_dir(run_id), "samples.jsonl")

    progress_path = _progress_path(run_id)
    logger.info(
        "Starting annotation batch loop run_id=%s model=%s chunk_size=%d "
        "max_tokens_per_request=%d temperature=%s progress=%s skip_paths=%s "
        "reasoning_effort=%s",
        run_id,
        model_name,
        chunk_size,
        max_tokens_per_request,
        temperature,
        progress_path,
        skip_annotation_paths,
        reasoning_effort,
    )

    chunks_done = 0
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "max_tokens_per_request": max_tokens_per_request,
        "temperature": temperature,
        "skip_annotation_paths": list(skip_annotation_paths or []),
        "reasoning_effort": reasoning_effort,
        "progress_path": str(progress_path),
        "chunks": [],
    }

    while True:
        if max_chunks is not None and chunks_done >= max_chunks:
            logger.info("Reached max_chunks=%d; stopping", max_chunks)
            break

        chunk_idx = chunks_done + 1
        t0 = time.time()
        description = f"cxr_loop_{run_id}_chunk{chunk_idx}"

        prep = run_annotation(
            run_id=run_id,
            samples_path=samples_path,
            model_name=model_name,
            api_key_name=api_key_name,
            api_type="gemini",
            max_queries=chunk_size,
            batch_mode="prepare",
            batch_description=description,
            max_tokens_per_request=max_tokens_per_request,
            temperature=temperature,
            skip_annotation_paths=skip_annotation_paths,
            config_extra=config_extra,
            reasoning_effort=reasoning_effort,
        )
        n_queued = int(prep.get("n_jobs_queued") or 0)
        n_uid = prep.get("n_uid_intents")
        logger.info(
            "Chunk %d prepare n_unique_queries=%d n_uid_intents=%s config_slug=%s",
            chunk_idx,
            n_queued,
            n_uid,
            prep.get("config_slug"),
        )
        if n_queued == 0:
            logger.info("No remaining annotation jobs; loop complete")
            _append_progress(
                progress_path,
                {
                    "ts": _utc_now_iso(),
                    "event": "done",
                    "chunk_idx": chunk_idx,
                    "n_jobs_queued": 0,
                    "n_uid_intents": n_uid,
                    "reason": "no_jobs_left",
                    "save_path": prep.get("save_path"),
                    "config_slug": prep.get("config_slug"),
                },
            )
            break

        job_record_path = prep.get("batch_job_record_path")
        logger.info(
            "Chunk %d prepared n_jobs=%d job_record=%s",
            chunk_idx,
            n_queued,
            job_record_path,
        )

        t_submit0 = time.time()
        submit_info = submit_with_backoff(
            run_id=run_id,
            samples_path=samples_path,
            model_name=model_name,
            api_key_name=api_key_name,
            batch_description=description,
            backoff_initial_s=submit_backoff_initial_s,
            backoff_max_s=submit_backoff_max_s,
            temperature=temperature,
            max_tokens_per_request=max_tokens_per_request,
            skip_annotation_paths=skip_annotation_paths,
            config_extra=config_extra,
            reasoning_effort=reasoning_effort,
        )
        batch = submit_info.get("batch") or {}
        batch_id = batch.get("batch_id")
        logger.info(
            "Chunk %d submitted batch_id=%s attempts=%s n_429=%s (%.1fs)",
            chunk_idx,
            batch_id,
            submit_info.get("submit_attempts"),
            submit_info.get("n_429_waits"),
            time.time() - t_submit0,
        )

        t_collect0 = time.time()
        collect_info = run_annotation(
            run_id=run_id,
            samples_path=samples_path,
            model_name=model_name,
            api_key_name=api_key_name,
            api_type="gemini",
            batch_mode="collect",
            batch_wait=True,
            batch_job_record_path=job_record_path,
            batch_poll_interval_s=batch_poll_interval_s,
            temperature=temperature,
            max_tokens_per_request=max_tokens_per_request,
            skip_annotation_paths=skip_annotation_paths,
            config_extra=config_extra,
            reasoning_effort=reasoning_effort,
        )
        collect_batch = collect_info.get("batch") or {}
        n_saved = collect_batch.get("n_saved")
        status = collect_batch.get("batch_status")
        logger.info(
            "Chunk %d collect status=%s n_saved=%s (%.1fs)",
            chunk_idx,
            status,
            n_saved,
            time.time() - t_collect0,
        )

        elapsed = time.time() - t0
        progress_row = {
            "ts": _utc_now_iso(),
            "event": "chunk_complete",
            "chunk_idx": chunk_idx,
            "n_jobs_queued": n_queued,
            "n_uid_intents": n_uid,
            "batch_job_record_path": job_record_path,
            "batch_id": batch_id,
            "batch_status": status,
            "n_saved": n_saved,
            "n_new_rows": collect_info.get("n_new_rows"),
            "submit_attempts": submit_info.get("submit_attempts"),
            "n_429_waits": submit_info.get("n_429_waits"),
            "elapsed_s": round(elapsed, 2),
            "save_path": collect_info.get("save_path") or prep.get("save_path"),
            "config_slug": prep.get("config_slug"),
        }
        _append_progress(progress_path, progress_row)
        summary["chunks"].append(progress_row)
        chunks_done += 1

        if not collect_batch.get("done"):
            logger.error("Collect did not finish cleanly; stopping loop. info=%s", collect_info)
            break
        if status in ("failed", "cancelled", "canceled", "expired"):
            logger.error("Batch ended with status=%s; stopping loop", status)
            break

        if max_chunks is not None and chunks_done >= max_chunks:
            logger.info("Reached max_chunks=%d after chunk %d", max_chunks, chunk_idx)
            break

        if post_collect_sleep_s > 0:
            logger.info("Post-collect sleep %.0fs before next chunk", post_collect_sleep_s)
            time.sleep(post_collect_sleep_s)

    summary["chunks_done"] = chunks_done
    summary["finished_at"] = _utc_now_iso()
    logger.info("Loop finished chunks_done=%d", chunks_done)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--samples_path", default=None)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api_key_name", default="GEMINI_API_KEY")
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Stop after this many chunks (default: run until no jobs left)",
    )
    parser.add_argument("--submit_backoff_initial_s", type=float, default=60.0)
    parser.add_argument("--submit_backoff_max_s", type=float, default=900.0)
    parser.add_argument("--post_collect_sleep_s", type=float, default=60.0)
    parser.add_argument("--batch_poll_interval_s", type=float, default=30.0)
    parser.add_argument(
        "--max_tokens_per_request",
        type=int,
        default=2048,
        help="Completion token budget written into each prepare request (default 2048)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip_annotation_paths",
        nargs="+",
        default=None,
        help="Prior annotation jsonl(s); skip successful content_keys (leftover-only runs)",
    )
    parser.add_argument("--config_extra", default=None)
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        help="OpenAI-compat reasoning_effort (Gemini 3 thinking_level). Default minimal for gemini-3.5-flash-lite.",
    )
    args = parser.parse_args()
    summary = run_batch_loop(
        run_id=args.run_id,
        samples_path=args.samples_path,
        model_name=args.model_name,
        api_key_name=args.api_key_name,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        submit_backoff_initial_s=args.submit_backoff_initial_s,
        submit_backoff_max_s=args.submit_backoff_max_s,
        post_collect_sleep_s=args.post_collect_sleep_s,
        batch_poll_interval_s=args.batch_poll_interval_s,
        max_tokens_per_request=args.max_tokens_per_request,
        temperature=args.temperature,
        skip_annotation_paths=args.skip_annotation_paths,
        config_extra=args.config_extra,
        reasoning_effort=args.reasoning_effort,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "chunks"}, indent=2))


if __name__ == "__main__":
    main()
