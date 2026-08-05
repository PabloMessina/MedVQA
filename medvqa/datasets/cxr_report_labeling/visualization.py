"""Visualization helpers for regex vs LLM metrics (used by notebooks)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


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
