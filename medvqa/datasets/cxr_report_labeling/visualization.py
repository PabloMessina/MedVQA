"""Visualization helpers for regex vs LLM metrics (used by notebooks)."""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd

from medvqa.utils.files_utils import load_json, load_jsonl

PathLike = Union[str, Path]


def ranked_indices_by_score(
    scores: Sequence[float],
    *,
    descending: bool = True,
) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=descending)


def format_complexity_inspection(
    report: dict,
    score: float,
    *,
    rank: int,
    n_total: int,
    dataset: str,
    max_report_chars: Optional[int] = None,
) -> str:
    """Plain-text block for one ranked report (notebook / CLI friendly)."""
    text = report.get("report_text") or ""
    if max_report_chars is not None and len(text) > max_report_chars:
        text = text[:max_report_chars] + "\n... [truncated]"
    lines = [
        "=" * 80,
        f"dataset={dataset}  rank={rank + 1}/{n_total}  score={score:.6g}",
        f"uid={report.get('uid')}  chars={len(report.get('report_text') or '')}",
        "--- report ---",
        text,
    ]
    return "\n".join(lines)


def inspect_complexity_interactively(
    reports_by_dataset: Dict[str, Sequence[dict]],
    complexity_by_dataset: Dict[str, Sequence[float]],
    *,
    default_dataset: Optional[str] = None,
    max_report_chars: Optional[int] = None,
):
    """ipywidgets browser: pick dataset, highest/lowest, and rank slider."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    datasets = [ds for ds in reports_by_dataset if ds in complexity_by_dataset]
    if not datasets:
        print("No overlapping datasets between reports and complexity scores")
        return None

    if default_dataset is None or default_dataset not in datasets:
        default_dataset = datasets[0]

    dataset_dd = widgets.Dropdown(options=datasets, value=default_dataset, description="dataset")
    order_dd = widgets.Dropdown(
        options=[("highest complexity", "high"), ("lowest complexity", "low")],
        value="high",
        description="order",
    )
    rank_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(len(complexity_by_dataset[default_dataset]) - 1, 0),
        step=1,
        description="rank",
        continuous_update=False,
    )
    out = widgets.Output()

    def _ranked(ds: str, order: str) -> List[int]:
        return ranked_indices_by_score(
            complexity_by_dataset[ds],
            descending=(order == "high"),
        )

    def _sync_slider(*_args):
        ds = dataset_dd.value
        n = len(complexity_by_dataset[ds])
        rank_slider.max = max(n - 1, 0)
        if rank_slider.value > rank_slider.max:
            rank_slider.value = rank_slider.max

    def _render(*_args):
        ds = dataset_dd.value
        order = order_dd.value
        ranked = _ranked(ds, order)
        if not ranked:
            with out:
                clear_output(wait=True)
                print(f"No reports for {ds}")
            return
        rank = int(rank_slider.value)
        idx = ranked[rank]
        report = reports_by_dataset[ds][idx]
        score = float(complexity_by_dataset[ds][idx])
        with out:
            clear_output(wait=True)
            print(
                format_complexity_inspection(
                    report,
                    score,
                    rank=rank,
                    n_total=len(ranked),
                    dataset=ds,
                    max_report_chars=max_report_chars,
                )
            )

    dataset_dd.observe(_sync_slider, names="value")
    order_dd.observe(_sync_slider, names="value")
    for w in (dataset_dd, order_dd, rank_slider):
        w.observe(_render, names="value")

    controls = widgets.HBox([dataset_dd, order_dd, rank_slider])
    display(controls, out)
    _render()
    return controls


def metrics_to_dataframe(metrics: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for class_id, m in metrics.items():
        rows.append({"class_id": class_id, **m})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("f1")
    return df


def plot_prf1_bars(metrics: Dict[str, dict], figsize: Optional[tuple] = None):
    df = metrics_to_dataframe(metrics)
    if df.empty:
        print("No metrics to plot")
        return None
    if figsize is None:
        figsize = (10, max(6, len(metrics) * 0.25))
    fig, ax = plt.subplots(figsize=figsize)
    y = range(len(df))
    ax.barh([i - 0.25 for i in y], df["precision"], height=0.25, label="precision")
    ax.barh(y, df["recall"], height=0.25, label="recall")
    ax.barh([i + 0.25 for i in y], df["f1"], height=0.25, label="f1")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["class_id"])
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.set_title("Regex vs LLM (binary) — P / R / F1")
    plt.tight_layout()
    return fig


def show_fp_fn_example(row: dict, max_report_chars: int = 1200) -> None:
    print("=" * 80)
    print(f"uid={row.get('uid')}  class={row.get('class_id')}")
    llm = row.get("llm") or {}
    print(f"LLM label: {llm.get('label')}")
    print(f"quote: {llm.get('relevant_quote')}")
    print(f"reasoning: {llm.get('reasoning')}")
    print(f"regex_spans: {row.get('regex_spans')}")
    text = row.get("report_text") or ""
    print("--- report ---")
    print(text[:max_report_chars] + ("..." if len(text) > max_report_chars else ""))


def resolve_regex_refine_group(group_dir: PathLike) -> Dict[str, Any]:
    """Resolve paths for a regex-refine agent group workdir.

    Expects::

        .agent/regex_refine/<stamp>/groups/<group_id>/
          owned_classes.json
          cxr_classes/
          eval/ | eval_baseline/   (optional at resolve time)
        and stamp-level ``annotations.snapshot.jsonl`` / ``samples.snapshot.jsonl``.

    ``owned_classes.json`` may list:

    - ``eval_class_ids`` — full closed subgraph for hierarchy-aware eval
    - ``owned_class_ids`` — promotable / editable stems
    - ``readonly_class_ids`` — ontology descendants (do not edit / promote)
    - legacy ``class_ids`` — treated as the eval set
    """
    group_dir = Path(group_dir).expanduser().resolve()
    if not group_dir.is_dir():
        raise FileNotFoundError(f"group_dir not found: {group_dir}")

    owned_path = group_dir / "owned_classes.json"
    if not owned_path.is_file():
        raise FileNotFoundError(
            f"Not a regex-refine group dir (missing owned_classes.json): {group_dir}"
        )
    owned = load_json(str(owned_path))
    owned_class_ids = list(owned.get("owned_class_ids") or [])
    readonly_class_ids = list(owned.get("readonly_class_ids") or [])
    eval_class_ids = list(
        owned.get("eval_class_ids")
        or owned.get("class_ids")
        or owned_class_ids
    )
    if not owned_class_ids:
        # Legacy snapshots: class_ids were the owned (and only) stems.
        owned_class_ids = list(owned.get("class_ids") or eval_class_ids)
    if not eval_class_ids:
        raise ValueError(f"No eval/owned class_ids in {owned_path}")
    if not owned_class_ids:
        owned_class_ids = list(eval_class_ids)

    stamp_dir = group_dir.parent.parent
    ann = stamp_dir / "annotations.snapshot.jsonl"
    samples = stamp_dir / "samples.snapshot.jsonl"
    pattern_dir = group_dir / "cxr_classes"

    eval_dirs: Dict[str, Path] = {}
    for name in ("eval", "eval_baseline"):
        p = group_dir / name
        if p.is_dir():
            eval_dirs[name] = p

    return {
        "group_dir": group_dir,
        "stamp_dir": stamp_dir,
        "owned": owned,
        "class_ids": eval_class_ids,
        "eval_class_ids": eval_class_ids,
        "owned_class_ids": owned_class_ids,
        "readonly_class_ids": readonly_class_ids,
        "pattern_dir": pattern_dir,
        "annotations_path": ann if ann.is_file() else None,
        "samples_path": samples if samples.is_file() else None,
        "eval_dirs": eval_dirs,
    }


def _discover_class_ids_in_dump_dir(dump_dir: Path) -> List[str]:
    stems: set[str] = set()
    for p in dump_dir.glob("*_fp.jsonl"):
        stems.add(p.name[: -len("_fp.jsonl")])
    for p in dump_dir.glob("*_fn.jsonl"):
        stems.add(p.name[: -len("_fn.jsonl")])
    return sorted(stems)


def load_fp_fn_dumps(
    dump_dir: PathLike,
    class_id: str,
) -> Dict[str, Any]:
    """Load ``*_fp.jsonl`` / ``*_fn.jsonl`` and optional metrics for one class."""
    dump_dir = Path(dump_dir).expanduser().resolve()
    stem = class_id.replace("/", "_")
    fp_path = dump_dir / f"{stem}_fp.jsonl"
    fn_path = dump_dir / f"{stem}_fn.jsonl"
    fp = load_jsonl(str(fp_path)) if fp_path.is_file() else []
    fn = load_jsonl(str(fn_path)) if fn_path.is_file() else []

    metrics: Optional[dict] = None
    metrics_path = dump_dir / "metrics_per_class.json"
    if metrics_path.is_file():
        raw = load_json(str(metrics_path))
        # Support both flat {class: metrics} and wrapped {"metrics": {...}}.
        if isinstance(raw, dict) and "metrics" in raw and isinstance(raw["metrics"], dict):
            raw = raw["metrics"]
        metrics = raw.get(class_id) if isinstance(raw, dict) else None

    return {
        "dump_dir": dump_dir,
        "class_id": class_id,
        "fp": fp,
        "fn": fn,
        "metrics": metrics,
        "fp_path": fp_path if fp_path.is_file() else None,
        "fn_path": fn_path if fn_path.is_file() else None,
    }


def _highlight_report_html(
    text: str,
    *,
    spans: Sequence[str],
    quote: Optional[str] = None,
    max_chars: Optional[int] = None,
) -> str:
    """Escape report text and highlight regex spans (yellow) + LLM quote (cyan)."""
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"

    def _flex_pattern(fragment: str) -> str:
        # Spans come from whitespace-normalized match text; allow flexible gaps.
        parts = fragment.split()
        if not parts:
            return ""
        return r"\s+".join(re.escape(p) for p in parts)

    # Collect case-insensitive non-overlapping mark intervals.
    intervals: List[tuple[int, int, str]] = []
    for span in spans or []:
        if not span:
            continue
        pat = _flex_pattern(span)
        if not pat:
            continue
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            intervals.append((m.start(), m.end(), "span"))
    if quote:
        q = quote.strip()
        pat = _flex_pattern(q) if q else ""
        if pat:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                intervals.append((m.start(), m.end(), "quote"))

    if not intervals:
        return f"<pre style='white-space:pre-wrap;font-family:monospace'>{html.escape(text)}</pre>"

    intervals.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged: List[tuple[int, int, str]] = []
    for start, end, kind in intervals:
        if merged and start < merged[-1][1]:
            # Prefer earlier longer mark; skip overlaps.
            continue
        merged.append((start, end, kind))

    colors = {
        "span": ("#fff3a3", "regex span"),
        "quote": ("#b3e5fc", "LLM quote"),
    }
    parts: List[str] = []
    cursor = 0
    for start, end, kind in merged:
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        bg, title = colors.get(kind, ("#eee", kind))
        parts.append(
            f"<mark title='{html.escape(title)}' "
            f"style='background:{bg};padding:0 2px'>"
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    return (
        "<pre style='white-space:pre-wrap;font-family:monospace'>"
        + "".join(parts)
        + "</pre>"
    )


def format_disagreement_case(
    row: dict,
    *,
    kind: str,
    index: int,
    n_total: int,
    metrics: Optional[dict] = None,
    max_report_chars: Optional[int] = None,
    as_html: bool = False,
) -> str:
    """Format one FP/FN disagreement for notebook display."""
    llm = row.get("llm") or {}
    if not isinstance(llm, dict):
        llm = {}
    spans = row.get("regex_spans") or []
    matched_via = row.get("matched_via_classes") or []
    gold_via = row.get("gold_via_classes") or []
    llm_by_class = row.get("llm_by_class") or {}
    if not isinstance(llm_by_class, dict):
        llm_by_class = {}
    metrics_line = ""
    if metrics:
        metrics_line = (
            f"P={metrics.get('precision', 0):.3f}  "
            f"R={metrics.get('recall', 0):.3f}  "
            f"F1={metrics.get('f1', 0):.3f}  "
            f"TP={metrics.get('tp')} FP={metrics.get('fp')} "
            f"FN={metrics.get('fn')} TN={metrics.get('tn')}  "
            f"n={metrics.get('n_evaluated')}"
        )

    header = (
        f"{kind.upper()}  {index + 1}/{n_total}  "
        f"uid={row.get('uid')}  class={row.get('class_id')}"
    )

    def _llm_by_class_lines() -> List[str]:
        if not llm_by_class:
            return []
        lines = ["llm_by_class:"]
        for cid, payload in llm_by_class.items():
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"  [{cid}] label={payload.get('label')!r}  "
                f"quote={payload.get('relevant_quote')!r}"
            )
        return lines

    if as_html:
        report_html = _highlight_report_html(
            row.get("report_text") or "",
            spans=spans,
            quote=llm.get("relevant_quote"),
            max_chars=max_report_chars,
        )
        blocks = [
            f"<h3 style='margin:0.2em 0'>{html.escape(header)}</h3>",
        ]
        if metrics_line:
            blocks.append(f"<div><b>class metrics:</b> {html.escape(metrics_line)}</div>")
        blocks.extend(
            [
                f"<div><b>LLM label:</b> {html.escape(str(llm.get('label')))}</div>",
                f"<div><b>LLM quote:</b> {html.escape(str(llm.get('relevant_quote') or ''))}</div>",
                f"<div><b>LLM reasoning:</b> {html.escape(str(llm.get('reasoning') or ''))}</div>",
                f"<div><b>gold_positive:</b> {html.escape(str(row.get('gold_positive')))}  "
                f"<b>gold_via_descendant:</b> {html.escape(str(row.get('gold_via_descendant')))}</div>",
                f"<div><b>gold_via_classes:</b> {html.escape(json.dumps(gold_via, ensure_ascii=False))}</div>",
                f"<div><b>matched_via_classes:</b> {html.escape(json.dumps(matched_via, ensure_ascii=False))}</div>",
                f"<div><b>regex_spans:</b> {html.escape(json.dumps(spans, ensure_ascii=False))}</div>",
            ]
        )
        if llm_by_class:
            blocks.append("<div><b>llm_by_class:</b><ul style='margin:0.2em 0'>")
            for cid, payload in llm_by_class.items():
                if not isinstance(payload, dict):
                    continue
                blocks.append(
                    "<li>"
                    f"<code>{html.escape(str(cid))}</code>: "
                    f"label={html.escape(str(payload.get('label')))}; "
                    f"quote={html.escape(str(payload.get('relevant_quote') or ''))}"
                    "</li>"
                )
            blocks.append("</ul></div>")
        blocks.extend(
            [
                "<hr/><div><b>report</b> "
                "(yellow = regex span, cyan = LLM quote)</div>",
                report_html,
            ]
        )
        return "\n".join(blocks)

    text = row.get("report_text") or ""
    if max_report_chars is not None and len(text) > max_report_chars:
        text = text[:max_report_chars] + "\n... [truncated]"
    lines = [
        "=" * 80,
        header,
    ]
    if metrics_line:
        lines.append(f"class metrics: {metrics_line}")
    lines.extend(
        [
            f"LLM label: {llm.get('label')}",
            f"LLM quote: {llm.get('relevant_quote')}",
            f"LLM reasoning: {llm.get('reasoning')}",
            f"gold_positive: {row.get('gold_positive')}  "
            f"gold_via_descendant: {row.get('gold_via_descendant')}",
            f"gold_via_classes: {gold_via}",
            f"matched_via_classes: {matched_via}",
            f"regex_spans: {spans}",
        ]
    )
    lines.extend(_llm_by_class_lines())
    lines.extend(
        [
            "--- report ---",
            text,
        ]
    )
    return "\n".join(lines)


def recompute_group_eval(
    group_dir: PathLike,
    *,
    out_name: str = "eval",
) -> Dict[str, Any]:
    """Re-run subgraph eval for a refine group; write dumps under ``out_name``."""
    from medvqa.scripts.cxr_report_labeling.eval_regex_subgraph import (
        evaluate_regex_subgraph,
    )

    info = resolve_regex_refine_group(group_dir)
    if info["annotations_path"] is None or info["samples_path"] is None:
        raise FileNotFoundError(
            "Stamp snapshots missing "
            f"(need annotations.snapshot.jsonl + samples.snapshot.jsonl under {info['stamp_dir']})"
        )
    if not info["pattern_dir"].is_dir():
        raise FileNotFoundError(f"Missing pattern dir: {info['pattern_dir']}")

    out_dir = info["group_dir"] / out_name
    return evaluate_regex_subgraph(
        class_ids=info["class_ids"],
        regex_classes_dir=str(info["pattern_dir"]),
        annotations_path=str(info["annotations_path"]),
        samples_path=str(info["samples_path"]),
        out_dir=str(out_dir),
    )


def inspect_regex_llm_disagreements(
    path: PathLike,
    *,
    default_source: Optional[str] = None,
    default_class_id: Optional[str] = None,
    default_kind: str = "fn",
    max_report_chars: Optional[int] = 4000,
    recompute: bool = False,
    as_html: bool = True,
):
    """ipywidgets browser for regex vs LLM FP/FN disagreements.

    ``path`` may be:

    - a regex-refine **group** dir (has ``owned_classes.json``), or
    - a dump dir containing ``*_fp.jsonl`` / ``*_fn.jsonl``.

    For group dirs, use the source dropdown (``eval`` / ``eval_baseline``).
    Set ``recompute=True`` to refresh ``eval/`` via subgraph eval first.
    """
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    path = Path(path).expanduser().resolve()
    group_info: Optional[Dict[str, Any]] = None
    dump_only = False

    if (path / "owned_classes.json").is_file():
        group_info = resolve_regex_refine_group(path)
        if recompute:
            recompute_group_eval(path, out_name="eval")
            group_info = resolve_regex_refine_group(path)
        source_options = list(group_info["eval_dirs"].keys())
        if not source_options:
            raise FileNotFoundError(
                f"No eval/ or eval_baseline/ under {path}. "
                "Pass recompute=True or run eval_regex_subgraph first."
            )
        class_ids = list(group_info["class_ids"])
    elif path.is_dir() and (
        list(path.glob("*_fp.jsonl")) or list(path.glob("*_fn.jsonl"))
    ):
        dump_only = True
        source_options = ["dump"]
        class_ids = _discover_class_ids_in_dump_dir(path)
        if not class_ids:
            raise FileNotFoundError(f"No FP/FN dumps in {path}")
    else:
        raise FileNotFoundError(
            f"Expected a regex-refine group dir or an FP/FN dump dir: {path}"
        )

    if default_kind not in ("fp", "fn"):
        default_kind = "fn"
    if default_class_id is None or default_class_id not in class_ids:
        default_class_id = class_ids[0]
    if default_source is None or default_source not in source_options:
        # Prefer post-agent eval over baseline when both exist.
        default_source = (
            "eval"
            if "eval" in source_options
            else source_options[0]
        )

    source_dd = widgets.Dropdown(
        options=source_options,
        value=default_source,
        description="source",
        layout=widgets.Layout(width="180px"),
    )
    class_dd = widgets.Dropdown(
        options=class_ids,
        value=default_class_id,
        description="class",
        layout=widgets.Layout(width="320px"),
    )
    kind_dd = widgets.Dropdown(
        options=[("false negatives", "fn"), ("false positives", "fp")],
        value=default_kind,
        description="kind",
        layout=widgets.Layout(width="200px"),
    )
    index_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="index",
        continuous_update=False,
        layout=widgets.Layout(width="360px"),
    )
    recompute_btn = widgets.Button(
        description="Recompute eval",
        button_style="warning",
        disabled=dump_only or group_info is None,
        tooltip="Re-run subgraph eval into group eval/ then reload",
    )
    out = widgets.Output()

    state: Dict[str, Any] = {"bundle": None}

    def _dump_dir_for_source(source: str) -> Path:
        if dump_only:
            return path
        assert group_info is not None
        return group_info["eval_dirs"][source]

    def _load_bundle():
        dump_dir = _dump_dir_for_source(source_dd.value)
        # Refresh group eval_dirs after recompute.
        nonlocal group_info
        if group_info is not None and not dump_only:
            group_info = resolve_regex_refine_group(path)
            if source_dd.value not in group_info["eval_dirs"]:
                raise FileNotFoundError(
                    f"source {source_dd.value!r} missing under {path}"
                )
            dump_dir = group_info["eval_dirs"][source_dd.value]
        state["bundle"] = load_fp_fn_dumps(dump_dir, class_dd.value)

    def _rows() -> List[dict]:
        bundle = state["bundle"] or {}
        return list(bundle.get(kind_dd.value) or [])

    def _sync_slider(*_args):
        rows = _rows()
        index_slider.max = max(len(rows) - 1, 0)
        if index_slider.value > index_slider.max:
            index_slider.value = index_slider.max

    def _render(*_args):
        rows = _rows()
        with out:
            clear_output(wait=True)
            if not rows:
                print(
                    f"No {kind_dd.value.upper()} rows for "
                    f"{class_dd.value} in {source_dd.value}"
                )
                return
            idx = int(index_slider.value)
            idx = max(0, min(idx, len(rows) - 1))
            row = rows[idx]
            metrics = (state["bundle"] or {}).get("metrics")
            formatted = format_disagreement_case(
                row,
                kind=kind_dd.value,
                index=idx,
                n_total=len(rows),
                metrics=metrics,
                max_report_chars=max_report_chars,
                as_html=as_html,
            )
            if as_html:
                display(HTML(formatted))
            else:
                print(formatted)

    def _on_source_or_class(*_args):
        _load_bundle()
        _sync_slider()
        _render()

    def _on_recompute(_btn):
        with out:
            clear_output(wait=True)
            print("Recomputing subgraph eval → eval/ ...")
        recompute_group_eval(path, out_name="eval")
        # Ensure source dropdown includes eval.
        if "eval" not in list(source_dd.options):
            source_dd.options = list(source_dd.options) + ["eval"]
        source_dd.value = "eval"
        _on_source_or_class()

    recompute_btn.on_click(_on_recompute)
    source_dd.observe(_on_source_or_class, names="value")
    class_dd.observe(_on_source_or_class, names="value")
    kind_dd.observe(_sync_slider, names="value")
    kind_dd.observe(_render, names="value")
    index_slider.observe(_render, names="value")

    controls = widgets.HBox([source_dd, class_dd, kind_dd, index_slider, recompute_btn])
    display(controls, out)
    _on_source_or_class()
    return controls
