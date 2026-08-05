#!/usr/bin/env python3
"""Regenerate ontology presentation HTML from the YAML, RadLex CSV, and prompts.

Writes:
  - presentation_hierarchy.html       (category-grouped tree)
  - presentation_hierarchy_dag.html   (is-a / propagation DAG of all classes)
  - presentation_mapping_table.html   (RadLex/dataset mappings + definitions)

Reads:
  - canonical_cxr_ontology.yaml (this directory)
  - radlex_mappings.csv (this directory)
  - medvqa/prompts/cxr_classes/<id>.txt (full NLI hypotheses)

Usage (from repo root):
  python3 medvqa/datasets/cxr_report_labeling/ontology/generate_presentations.py
  python3 medvqa/datasets/cxr_report_labeling/ontology/generate_presentations.py \\
    --also-copy-to .agent/canonical_classes
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
YAML_PATH = HERE / "canonical_cxr_ontology.yaml"
RADLEX_MAPPINGS_PATH = HERE / "radlex_mappings.csv"
PROMPT_DIR = REPO_ROOT / "medvqa" / "prompts" / "cxr_classes"


def load_ontology():
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    classes = data["classes"]
    categories = data["categories"]
    return data, classes, categories


def load_prompt(class_id: str) -> str:
    path = PROMPT_DIR / f"{class_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def load_radlex_mappings() -> dict[str, dict[str, str]]:
    with RADLEX_MAPPINGS_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mappings = {row["class_id"]: row for row in rows}
    if len(mappings) != len(rows):
        raise ValueError(f"Duplicate class_id in {RADLEX_MAPPINGS_PATH}")
    return mappings


def radlex_value_html(value: str) -> str:
    """Render RadLex text while linking every RID to its BioPortal class."""
    if not value:
        return '<span class="none">—</span>'
    parts: list[str] = []
    last = 0
    for match in re.finditer(r"\bRID\d+\b", value):
        parts.append(html.escape(value[last:match.start()]))
        rid = match.group()
        url = (
            "https://bioportal.bioontology.org/ontologies/RADLEX"
            "?p=classes&conceptid=http%3A%2F%2Fwww.radlex.org%2FRID%2F"
            f"{rid}"
        )
        parts.append(
            f'<a class="radlex-link" href="{url}" target="_blank" '
            f'rel="noopener noreferrer">{rid}</a>'
        )
        last = match.end()
    parts.append(html.escape(value[last:]))
    return "".join(parts)


def radlex_cell(mapping: dict[str, str]) -> str:
    match = (mapping.get("radlex_match") or "none").strip()
    needs_review = (mapping.get("needs_review") or "").strip().lower() == "yes"
    notes = (mapping.get("notes") or "").strip()
    review_badge = (
        '<span class="radlex-review">Needs review</span>' if needs_review else ""
    )
    notes_html = (
        f'<div class="radlex-notes">{html.escape(notes)}</div>' if notes else ""
    )
    return (
        '<div class="radlex-meta">'
        f'<span class="radlex-match radlex-{html.escape(match)}">'
        f"{html.escape(match)}</span>{review_badge}</div>"
        f'<div class="radlex-value">{radlex_value_html(mapping.get("radlex", ""))}</div>'
        f"{notes_html}"
    )


def badge_kind(kind: str) -> str:
    return f'<span class="badge kind">{html.escape(kind)}</span>'


def chip(rel: str, name: str) -> str:
    colors = {
        "exact": ("#1b7f4e", "#1b7f4e"),
        "near": ("#2f6fed", "#2f6fed"),
        "broader": ("#a15c00", "#a15c00"),
        "narrower": ("#6b4fbb", "#6b4fbb"),
        "partial": ("#9a3412", "#9a3412"),
    }
    border, color = colors.get(rel, ("#78716c", "#78716c"))
    label = name if rel == "exact" else f"{name} · {rel}"
    return (
        f'<span class="chip" style="border-color:{border};color:{color}">'
        f"{html.escape(label)}</span>"
    )


def maps_html(maps) -> str:
    if not maps:
        return '<span class="none">—</span>'
    return " ".join(chip(m.get("relationship", "exact"), m["class_name"]) for m in maps)


def definition_cell(c: dict) -> str:
    """Short YAML definition always visible; full NLI prompt behind <details>."""
    short = (c.get("preliminary_definition") or "").strip()
    full = load_prompt(c["id"])
    short_html = html.escape(short) if short else '<span class="none">—</span>'
    if not full:
        return f'<div class="def-short">{short_html}</div>'
    # If full prompt equals or closely mirrors short, still offer expand for the prompt wording.
    return (
        f'<div class="def-short">{short_html}</div>'
        f'<details class="def-full">'
        f"<summary>Full NLI prompt</summary>"
        f'<pre class="def-prompt">{html.escape(full)}</pre>'
        f"</details>"
    )


def build_hierarchy(data, classes, categories) -> str:
    ont = data["ontology"]
    by_id = {c["id"]: c for c in classes}
    id_to_name = {c["id"]: c["name"] for c in classes}
    children: dict[str, list[str]] = defaultdict(list)
    for c in classes:
        for p in c.get("parents") or []:
            children[p].append(c["id"])
    for p in children:
        children[p].sort(key=lambda cid: id_to_name[cid].casefold())

    multi_parent = [c["id"] for c in classes if len(c.get("parents") or []) > 1]
    parts: list[str] = []
    parts.append(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Canonical CXR Ontology — Hierarchy</title>
<style>
  :root {{
    --bg:#0f172a; --panel:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --line:#334155;
    --accent:#38bdf8;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:20px 28px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#1e293b,#0f172a); }}
  header h1 {{ margin:0 0 6px; font-size:22px; font-weight:650; }}
  header p {{ margin:0; color:var(--muted); font-size:13px; }}
  .stats {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:12px; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 12px; font-size:12px; }}
  .stat b {{ color:var(--accent); font-size:16px; display:block; }}
  main {{ padding:18px 28px 40px; display:grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap:14px; }}
  section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  h2 {{ margin:0 0 4px; font-size:14px; letter-spacing:.02em; }}
  .desc {{ color:var(--muted); font-size:11px; margin:0 0 8px; }}
  ul.tree, ul.tree ul {{ list-style:none; margin:4px 0 4px 14px; padding-left:12px; border-left:1px solid #334155; }}
  li.node {{ margin:3px 0; }}
  .name {{ font-weight:550; }}
  .id {{ color:var(--muted); font-size:10px; margin-left:6px; font-family: ui-monospace, monospace; }}
  .badge {{ display:inline-block; font-size:10px; padding:1px 6px; border-radius:999px; margin-left:6px; border:1px solid; }}
  .badge.kind {{ color:#94a3b8; border-color:#475569; }}
  .multi {{ color:#f472b6; font-size:10px; margin-left:4px; }}
  footer {{ padding:12px 28px; color:var(--muted); font-size:11px; border-top:1px solid var(--line); }}
</style>
</head>
<body>
<header>
  <h1>Canonical CXR classifier ontology</h1>
  <p>{html.escape(ont["name"])} · v{html.escape(ont["version"])} · status: {html.escape(ont["status"])} · parent = is-a / label inheritance</p>
  <div class="stats">
    <div class="stat"><b>{len(classes)}</b> classes</div>
    <div class="stat"><b>{len(categories)}</b> categories</div>
    <div class="stat"><b>{len(multi_parent)}</b> multi-parent classes</div>
  </div>
</header>
<main>
"""
    )

    for cat in categories:
        cat_classes = [c for c in classes if c["category"] == cat["id"]]
        cat_ids = {c["id"] for c in cat_classes}
        roots = sorted(
            [c for c in cat_classes if not any(p in cat_ids for p in (c.get("parents") or []))],
            key=lambda c: c["name"].casefold(),
        )

        def build_ul_cross(cid: str, section_cat: str) -> str:
            c = by_id[cid]
            ps = c.get("parents") or []
            multi = '<span class="multi">⟵ multi-parent</span>' if len(ps) > 1 else ""
            section_ids = {x["id"] for x in classes if x["category"] == section_cat}
            if (
                ps
                and not any(p in section_ids for p in ps)
                and c["category"] == section_cat
            ):
                foreign = [id_to_name[p] for p in ps if p not in section_ids]
                open_li = (
                    f'<li class="node"><span class="name">{html.escape(c["name"])}</span> '
                    f'<span class="id">{cid}</span>'
                    f'<span class="multi">parents: {html.escape(", ".join(foreign))}</span>'
                )
            else:
                open_li = (
                    f'<li class="node"><span class="name">{html.escape(c["name"])}</span> '
                    f'<span class="id">{cid}</span>{badge_kind(c["kind"])}{multi}'
                )
            kids = sorted(
                [
                    k
                    for k in children.get(cid, [])
                    if by_id[k]["category"] == section_cat
                    or by_id[cid]["category"] == section_cat
                ],
                key=lambda k: by_id[k]["name"].casefold(),
            )
            if section_cat == "diagnostic_label":
                kids = []
            if section_cat == "focal_lesion_findings":
                kids = [k for k in kids if by_id[k]["category"] == section_cat]
            if not kids:
                return open_li + "</li>"
            return (
                open_li
                + "<ul>"
                + "".join(build_ul_cross(k, section_cat) for k in kids)
                + "</ul></li>"
            )

        tree_items = [build_ul_cross(r["id"], cat["id"]) for r in roots]
        parts.append(
            f'<section>\n  <h2>{html.escape(cat["name"])} '
            f'<span style="color:var(--muted);font-weight:400;font-size:12px">({len(cat_classes)})</span></h2>\n'
            f'  <p class="desc">{html.escape(cat["description"])}</p>\n'
            f'  <ul class="tree">{"".join(tree_items)}</ul>\n</section>\n'
        )

    parts.append(
        "</main>\n<footer>Generated by generate_presentations.py from canonical_cxr_ontology.yaml"
        "</footer>\n</body></html>\n"
    )
    return "".join(parts)


def build_mapping_table(data, classes, categories) -> str:
    cat_by_id = {c["id"]: c for c in categories}
    id_to_name = {c["id"]: c["name"] for c in classes}
    radlex_mappings = load_radlex_mappings()
    class_ids = set(id_to_name)
    radlex_ids = set(radlex_mappings)
    if class_ids != radlex_ids:
        missing = sorted(class_ids - radlex_ids)
        extra = sorted(radlex_ids - class_ids)
        raise ValueError(
            f"RadLex mappings do not match ontology classes; missing={missing}, extra={extra}"
        )
    cat_options = "".join(
        f'<option value="{html.escape(c["id"])}">{html.escape(c["name"])}</option>'
        for c in categories
    )

    rows: list[str] = []
    for i, c in enumerate(classes, 1):
        dm = c.get("dataset_mappings") or {}
        parents = ", ".join(id_to_name[p] for p in (c.get("parents") or [])) or "—"
        mapped = (
            "1"
            if any(dm.get(k) for k in ("puccxr", "cxrlt2024", "vindr_cxr"))
            else "0"
        )
        radlex = radlex_mappings[c["id"]]
        radlex_review = (
            "1" if (radlex.get("needs_review") or "").strip().lower() == "yes" else "0"
        )
        rows.append(
            f'<tr data-cat="{html.escape(c["category"])}" data-mapped="{mapped}" '
            f'data-radlex-review="{radlex_review}">'
            f"<td>{i}</td>"
            f'<td><div class="cname">{html.escape(c["name"])}</div>'
            f'<div class="cid">{c["id"]}</div>'
            f'<div class="cat">{html.escape(cat_by_id[c["category"]]["name"])} · '
            f'{html.escape(c["kind"])}</div>'
            f'<div class="cat">parents: {html.escape(parents)}</div></td>'
            f'<td class="def">{definition_cell(c)}</td>'
            f'<td class="radlex">{radlex_cell(radlex)}</td>'
            f'<td class="maps">{maps_html(dm.get("puccxr") or [])}</td>'
            f'<td class="maps">{maps_html(dm.get("cxrlt2024") or [])}</td>'
            f'<td class="maps">{maps_html(dm.get("vindr_cxr") or [])}</td>'
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Canonical CXR Ontology — Dataset mapping</title>
<style>
  :root {{ --bg:#f7f5f1; --text:#1c1917; --muted:#78716c; --line:#d6d3d1; --header:#1e293b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:18px 24px; background:var(--header); color:#f8fafc; }}
  header h1 {{ margin:0 0 6px; font-size:20px; }}
  header p {{ margin:0; opacity:.85; font-size:13px; }}
  .legend {{ display:flex; gap:10px; flex-wrap:wrap; padding:12px 24px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:3; }}
  .radlex-legend {{ position:static; }}
  .chip {{ display:inline-block; border:1.5px solid; border-radius:6px; padding:1px 7px; font-size:11px; margin:1px 2px; background:#fff; white-space:nowrap; }}
  .none {{ color:#a8a29e; }}
  .controls {{ padding:10px 24px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; font-size:13px; }}
  .controls select, .controls input {{ padding:6px 8px; border:1px solid var(--line); border-radius:6px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; background:#fff; }}
  th, td {{ border-bottom:1px solid var(--line); padding:8px 10px; vertical-align:top; text-align:left; }}
  th {{ background:#f5f5f4; position:sticky; top:52px; z-index:2; font-size:11px; text-transform:uppercase; letter-spacing:.03em; color:#57534e; }}
  tr:hover td {{ background:#fafaf9; }}
  .cat {{ color:var(--muted); font-size:11px; }}
  .cname {{ font-weight:600; }}
  .cid {{ font-family:ui-monospace,monospace; font-size:10px; color:#a8a29e; }}
  td.maps {{ max-width:260px; }}
  td.radlex {{ min-width:260px; max-width:340px; }}
  .radlex-meta {{ display:flex; gap:5px; align-items:center; flex-wrap:wrap; margin-bottom:5px; }}
  .radlex-match, .radlex-review {{
    display:inline-block; border-radius:999px; padding:2px 7px;
    font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.03em;
  }}
  .radlex-exact {{ color:#166534; background:#dcfce7; }}
  .radlex-close {{ color:#1d4ed8; background:#dbeafe; }}
  .radlex-broader {{ color:#92400e; background:#fef3c7; }}
  .radlex-composite {{ color:#6b21a8; background:#f3e8ff; }}
  .radlex-none {{ color:#57534e; background:#e7e5e4; }}
  .radlex-review {{ color:#9f1239; background:#ffe4e6; }}
  .radlex-value {{ font-size:12px; font-weight:600; line-height:1.4; }}
  .radlex-link {{ color:#0369a1; text-decoration:none; }}
  .radlex-link:hover {{ text-decoration:underline; }}
  .radlex-notes {{ margin-top:5px; color:#57534e; font-size:11px; line-height:1.4; }}
  td.def {{ max-width:420px; font-size:12px; line-height:1.45; color:#44403c; }}
  .def-short {{ margin-bottom:4px; }}
  details.def-full {{ margin-top:2px; }}
  details.def-full > summary {{
    cursor:pointer; color:#1d4ed8; font-size:11px; font-weight:600;
    list-style:none; user-select:none;
  }}
  details.def-full > summary::-webkit-details-marker {{ display:none; }}
  details.def-full > summary::before {{ content:"▸ "; }}
  details.def-full[open] > summary::before {{ content:"▾ "; }}
  details.def-full > summary:hover {{ text-decoration:underline; }}
  pre.def-prompt {{
    margin:8px 0 0; padding:10px 12px; white-space:pre-wrap; word-break:break-word;
    background:#f5f5f4; border:1px solid var(--line); border-radius:8px;
    font-family:"IBM Plex Sans", "Segoe UI", sans-serif; font-size:11.5px; line-height:1.45;
    max-height:280px; overflow:auto; color:#1c1917;
  }}
</style>
</head>
<body>
<header>
  <h1>Canonical CXR ontology mappings</h1>
  <p>{len(classes)} classes · RadLex · PUC-CXR · CXR-LT 2024 · VinDr-CXR · expand definitions for full NLI prompts</p>
</header>
<div class="legend">
  <span><b>Dataset mappings:</b></span>
  <span class="chip" style="border-color:#1b7f4e;color:#1b7f4e">Exact match</span>
  <span class="chip" style="border-color:#2f6fed;color:#2f6fed">Near · near</span>
  <span class="chip" style="border-color:#a15c00;color:#a15c00">Source broader · broader</span>
  <span class="chip" style="border-color:#6b4fbb;color:#6b4fbb">Source narrower · narrower</span>
  <span class="chip" style="border-color:#9a3412;color:#9a3412">Partial · partial</span>
  <span class="none">— no useful match</span>
</div>
<div class="legend radlex-legend">
  <span><b>RadLex:</b></span>
  <span class="radlex-match radlex-exact">Exact</span>
  <span class="radlex-match radlex-close">Close</span>
  <span class="radlex-match radlex-broader">Broader</span>
  <span class="radlex-match radlex-composite">Composite</span>
  <span class="radlex-match radlex-none">None</span>
  <span class="radlex-review">Needs review</span>
</div>
<div class="controls">
  <label>Filter category
    <select id="catFilter" onchange="filterRows()">
      <option value="">All</option>{cat_options}
    </select>
  </label>
  <label><input type="checkbox" id="onlyUnmapped" onchange="filterRows()"/> Only fully unmapped (all three —)</label>
  <label><input type="checkbox" id="onlyRadlexReview" onchange="filterRows()"/> Needs RadLex review</label>
  <label><input type="checkbox" id="expandAll" onchange="toggleExpandAll()"/> Expand all full prompts</label>
  <span id="count" style="color:#78716c"></span>
</div>
<table id="mapTable">
<thead>
<tr>
  <th>#</th>
  <th>Ontology class</th>
  <th>Definition</th>
  <th>RadLex mapping</th>
  <th>PUC-CXR</th>
  <th>CXR-LT 2024</th>
  <th>VinDr-CXR</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<script>
function filterRows() {{
  const cat = document.getElementById('catFilter').value;
  const onlyUnmapped = document.getElementById('onlyUnmapped').checked;
  const onlyRadlexReview = document.getElementById('onlyRadlexReview').checked;
  let n = 0, shown = 0;
  for (const tr of document.querySelectorAll('#mapTable tbody tr')) {{
    n++;
    const okCat = !cat || tr.dataset.cat === cat;
    const okMap = !onlyUnmapped || tr.dataset.mapped === '0';
    const okRadlex = !onlyRadlexReview || tr.dataset.radlexReview === '1';
    const show = okCat && okMap && okRadlex;
    tr.style.display = show ? '' : 'none';
    if (show) shown++;
  }}
  document.getElementById('count').textContent = shown + ' / ' + n + ' rows';
}}
function toggleExpandAll() {{
  const open = document.getElementById('expandAll').checked;
  for (const d of document.querySelectorAll('details.def-full')) {{
    d.open = open;
  }}
}}
filterRows();
</script>
</body></html>
"""


# Distinct category colors for DAG nodes (light fills on dark or light bg).
_CATEGORY_COLORS = [
    "#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa",
    "#fb923c", "#2dd4bf", "#e879f9", "#94a3b8", "#f87171",
    "#4ade80", "#60a5fa", "#c084fc", "#facc15", "#22d3ee",
    "#eab308", "#86efac", "#fda4af",
]


def build_hierarchy_dag(data, classes, categories) -> str:
    """Interactive DAG: child→parent edges (propagation), roots at top."""
    ont = data["ontology"]
    id_to_name = {c["id"]: c["name"] for c in classes}
    cat_by_id = {c["id"]: c for c in categories}
    cat_color = {
        cat["id"]: _CATEGORY_COLORS[i % len(_CATEGORY_COLORS)]
        for i, cat in enumerate(categories)
    }

    nodes = []
    edges = []
    for c in classes:
        parents = list(c.get("parents") or [])
        nodes.append(
            {
                "data": {
                    "id": c["id"],
                    "label": c["name"],
                    "kind": c["kind"],
                    "category": c["category"],
                    "category_name": cat_by_id[c["category"]]["name"],
                    "color": cat_color[c["category"]],
                    "parents": parents,
                    "parent_names": [id_to_name[p] for p in parents],
                    "short_def": (c.get("preliminary_definition") or "").strip(),
                    "full_prompt": load_prompt(c["id"]),
                    "is_root": len(parents) == 0,
                    "is_multi": len(parents) > 1,
                }
            }
        )
        for p in parents:
            # Propagation direction: child → parent. Layout uses rankDir BT
            # so sinks (parents/roots) sit at the top.
            edges.append(
                {
                    "data": {
                        "id": f"{c['id']}__to__{p}",
                        "source": c["id"],
                        "target": p,
                    }
                }
            )

    graph = {"nodes": nodes, "edges": edges}
    # Escape < so embedded JSON cannot prematurely close the <script> tag.
    graph_json = json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c")
    legend_items = "".join(
        f'<span class="leg"><i style="background:{cat_color[c["id"]]}"></i>'
        f'{html.escape(c["name"])}</span>'
        for c in categories
    )
    cat_options = "".join(
        f'<option value="{html.escape(c["id"])}">{html.escape(c["name"])}</option>'
        for c in categories
    )
    n_edges = len(edges)
    n_multi = sum(1 for c in classes if len(c.get("parents") or []) > 1)
    n_roots = sum(1 for c in classes if not (c.get("parents") or []))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Canonical CXR Ontology — Propagation DAG</title>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  :root {{
    --bg:#0b1220; --panel:#111827; --panel2:#1f2937; --text:#e5e7eb; --muted:#9ca3af;
    --line:#374151; --accent:#38bdf8; --accent2:#34d399;
  }}
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; height:100%; background:var(--bg); color:var(--text);
    font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; }}
  body {{ display:flex; flex-direction:column; }}
  header {{ padding:14px 18px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#1f2937,#0b1220); }}
  header h1 {{ margin:0 0 4px; font-size:18px; font-weight:650; }}
  header p {{ margin:0; color:var(--muted); font-size:12px; }}
  .stats {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }}
  .stat {{ background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:6px 10px; font-size:11px; }}
  .stat b {{ color:var(--accent); font-size:14px; display:block; }}
  .toolbar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding:10px 18px;
    border-bottom:1px solid var(--line); background:var(--panel); font-size:13px; }}
  .toolbar input[type=search], .toolbar select {{ padding:6px 8px; border-radius:6px; border:1px solid var(--line);
    background:#0f172a; color:var(--text); min-width:160px; }}
  .toolbar button {{ padding:6px 10px; border-radius:6px; border:1px solid var(--line);
    background:#0f172a; color:var(--text); cursor:pointer; }}
  .toolbar button:hover {{ border-color:var(--accent); color:#fff; }}
  .legend {{ display:flex; gap:8px; flex-wrap:wrap; padding:8px 18px; border-bottom:1px solid var(--line);
    background:#0f172a; font-size:11px; color:var(--muted); }}
  .leg {{ display:inline-flex; align-items:center; gap:5px; margin-right:4px; }}
  .leg i {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
  .main {{ flex:1; display:grid; grid-template-columns: 1fr 340px; min-height:0; }}
  #cy {{ width:100%; height:100%; background:
    radial-gradient(circle at 20% 0%, #1e293b 0%, transparent 40%),
    radial-gradient(circle at 80% 100%, #0f172a 0%, transparent 35%),
    var(--bg); }}
  aside {{ border-left:1px solid var(--line); background:var(--panel); padding:14px 16px; overflow:auto; }}
  aside h2 {{ margin:0 0 8px; font-size:15px; }}
  aside .id {{ font-family:ui-monospace,monospace; font-size:11px; color:var(--muted); }}
  aside .meta {{ font-size:12px; color:var(--muted); margin:8px 0; line-height:1.5; }}
  aside .short {{ font-size:13px; line-height:1.45; margin:10px 0; }}
  aside details {{ margin-top:8px; }}
  aside summary {{ cursor:pointer; color:var(--accent); font-size:12px; font-weight:600; }}
  aside pre {{ white-space:pre-wrap; word-break:break-word; background:#0f172a; border:1px solid var(--line);
    border-radius:8px; padding:10px; font-size:11.5px; line-height:1.45; max-height:360px; overflow:auto;
    font-family:"IBM Plex Sans","Segoe UI",sans-serif; }}
  .placeholder {{ color:var(--muted); font-size:13px; line-height:1.5; }}
  .chip {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:1px 8px;
    font-size:11px; margin:2px 2px 2px 0; background:#0f172a; }}
  @media (max-width: 960px) {{
    .main {{ grid-template-columns: 1fr; grid-template-rows: 65vh auto; }}
    aside {{ border-left:none; border-top:1px solid var(--line); }}
  }}
</style>
</head>
<body>
<header>
  <h1>Canonical CXR classifier ontology — propagation DAG</h1>
  <p>{html.escape(ont["name"])} · v{html.escape(ont["version"])} · edges: child → parent (label inheritance) · roots at top</p>
  <div class="stats">
    <div class="stat"><b>{len(classes)}</b> classes</div>
    <div class="stat"><b>{n_edges}</b> inheritance edges</div>
    <div class="stat"><b>{n_roots}</b> roots</div>
    <div class="stat"><b>{n_multi}</b> multi-parent</div>
  </div>
</header>
<div class="toolbar">
  <label>Search <input type="search" id="search" placeholder="name or id…" oninput="onSearch()"/></label>
  <label>Category
    <select id="catFilter" onchange="onFilter()">
      <option value="">All</option>{cat_options}
    </select>
  </label>
  <button type="button" onclick="fitView()">Fit</button>
  <button type="button" onclick="runLayout()">Re-layout</button>
  <button type="button" onclick="clearSelection()">Clear selection</button>
  <span id="status" style="color:var(--muted)"></span>
</div>
<div class="legend"><span><b>Category:</b></span>{legend_items}</div>
<div class="main">
  <div id="cy"></div>
  <aside id="panel">
    <p class="placeholder">Click a class to inspect it. Edges point <b>child → parent</b> (propagation). Use search/category to focus; selection highlights ancestors and descendants.</p>
  </aside>
</div>
<script type="application/json" id="graph-data">{graph_json}</script>
<script>
const GRAPH = JSON.parse(document.getElementById('graph-data').textContent);
cytoscape.use(cytoscapeDagre);

const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: [...GRAPH.nodes, ...GRAPH.edges],
  style: [
    {{
      selector: 'node',
      style: {{
        'label': 'data(label)',
        'background-color': 'data(color)',
        'color': '#0b1220',
        'text-outline-width': 0,
        'font-size': 10,
        'font-weight': 600,
        'text-valign': 'center',
        'text-halign': 'center',
        'text-wrap': 'wrap',
        'text-max-width': 108,
        'width': 118,
        'height': 42,
        'shape': 'round-rectangle',
        'border-width': 1.5,
        'border-color': '#e5e7eb',
        'padding': '4px',
      }}
    }},
    {{
      selector: 'node[kind = "device"]',
      style: {{ 'shape': 'round-rectangle', 'border-style': 'dashed' }}
    }},
    {{
      selector: 'node[kind = "diagnosis"]',
      style: {{ 'shape': 'round-hexagon' }}
    }},
    {{
      selector: 'node[kind = "attribute"]',
      style: {{ 'shape': 'ellipse', 'width': 100, 'height': 36 }}
    }},
    {{
      selector: 'node[kind = "state"]',
      style: {{ 'shape': 'diamond', 'width': 90, 'height': 90, 'text-max-width': 80 }}
    }},
    {{
      selector: 'node[?is_root]',
      style: {{ 'border-width': 3, 'border-color': '#f8fafc' }}
    }},
    {{
      selector: 'node[?is_multi]',
      style: {{ 'border-color': '#f472b6', 'border-width': 2.5 }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 1.6,
        'line-color': '#64748b',
        'target-arrow-color': '#94a3b8',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'arrow-scale': 0.9,
        'opacity': 0.85,
      }}
    }},
    {{
      selector: 'node.faded, edge.faded',
      style: {{ 'opacity': 0.12 }}
    }},
    {{
      selector: 'node.highlight',
      style: {{ 'border-width': 3, 'border-color': '#fbbf24', 'z-index': 999 }}
    }},
    {{
      selector: 'node.selected',
      style: {{ 'border-width': 4, 'border-color': '#38bdf8', 'z-index': 1000 }}
    }},
    {{
      selector: 'edge.highlight',
      style: {{ 'width': 2.5, 'line-color': '#fbbf24', 'target-arrow-color': '#fbbf24', 'opacity': 1, 'z-index': 998 }}
    }},
  ],
  layout: {{ name: 'preset' }},
  wheelSensitivity: 0.25,
  minZoom: 0.15,
  maxZoom: 3,
}});

function runLayout() {{
  cy.layout({{
    name: 'dagre',
    rankDir: 'BT',  // child→parent edges; roots/sinks toward the top
    nodeSep: 28,
    edgeSep: 16,
    rankSep: 70,
    padding: 24,
    animate: false,
  }}).run();
  fitView();
}}

function fitView() {{
  cy.fit(cy.elements(':visible'), 40);
}}

function clearHighlights() {{
  cy.elements().removeClass('faded highlight selected');
}}

function clearSelection() {{
  clearHighlights();
  document.getElementById('panel').innerHTML =
    '<p class="placeholder">Click a class to inspect it. Edges point <b>child → parent</b> (propagation).</p>';
  document.getElementById('status').textContent = '';
}}

function escapeHtml(s) {{
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',\"'\":'&#39;'}}[c]
  ));
}}

function showPanel(n) {{
  const d = n.data();
  const parents = (d.parent_names || []).map(p => `<span class="chip">${{escapeHtml(p)}}</span>`).join(' ') || '—';
  document.getElementById('panel').innerHTML = `
    <h2>${{escapeHtml(d.label)}}</h2>
    <div class="id">${{escapeHtml(d.id)}}</div>
    <div class="meta">
      <div><b>Kind:</b> ${{escapeHtml(d.kind)}}</div>
      <div><b>Category:</b> ${{escapeHtml(d.category_name)}}</div>
      <div><b>Propagates to:</b> ${{parents}}</div>
    </div>
    <div class="short">${{escapeHtml(d.short_def || '—')}}</div>
    <details>
      <summary>Full NLI prompt</summary>
      <pre>${{escapeHtml(d.full_prompt || '')}}</pre>
    </details>
  `;
}}

function selectNode(n) {{
  clearHighlights();
  const ancestors = n.successors('node');  // follow child→parent
  const ancestorEdges = n.successors('edge');
  const descendants = n.predecessors('node');
  const descendantEdges = n.predecessors('edge');
  const keep = n.union(ancestors).union(descendants).union(ancestorEdges).union(descendantEdges);
  cy.elements().addClass('faded');
  keep.removeClass('faded').addClass('highlight');
  n.removeClass('highlight').addClass('selected');
  showPanel(n);
  document.getElementById('status').textContent =
    `${{dlabel(n)}} · ${{ancestors.length}} ancestor(s), ${{descendants.length}} descendant(s)`;
}}

function dlabel(n) {{ return n.data('label'); }}

cy.on('tap', 'node', (evt) => selectNode(evt.target));
cy.on('tap', (evt) => {{ if (evt.target === cy) clearSelection(); }});

function onFilter() {{
  const cat = document.getElementById('catFilter').value;
  const q = document.getElementById('search').value.trim().toLowerCase();
  cy.nodes().forEach((n) => {{
    const d = n.data();
    const okCat = !cat || d.category === cat;
    const okQ = !q || d.label.toLowerCase().includes(q) || d.id.toLowerCase().includes(q);
    if (okCat && okQ) n.style('display', 'element');
    else n.style('display', 'none');
  }});
  cy.edges().forEach((e) => {{
    const show = e.source().style('display') !== 'none' && e.target().style('display') !== 'none';
    e.style('display', show ? 'element' : 'none');
  }});
  document.getElementById('status').textContent =
    `${{cy.nodes(':visible').length}} / ${{cy.nodes().length}} classes visible`;
}}

function onSearch() {{ onFilter(); }}

runLayout();
onFilter();
</script>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--also-copy-to",
        type=Path,
        nargs="*",
        default=[],
        help="Optional extra directories to write the same HTML files into (e.g. .agent/canonical_classes)",
    )
    args = parser.parse_args()

    data, classes, categories = load_ontology()
    missing = [c["id"] for c in classes if not load_prompt(c["id"])]
    if missing:
        raise SystemExit(f"Missing prompt .txt for {len(missing)} classes: {missing[:10]}")

    hier = build_hierarchy(data, classes, categories)
    dag = build_hierarchy_dag(data, classes, categories)
    mapping = build_mapping_table(data, classes, categories)

    targets = [HERE, *args.also_copy_to]
    for out_dir in targets:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "presentation_hierarchy.html").write_text(hier, encoding="utf-8")
        (out_dir / "presentation_hierarchy_dag.html").write_text(dag, encoding="utf-8")
        (out_dir / "presentation_mapping_table.html").write_text(mapping, encoding="utf-8")
        print(f"Wrote presentations to {out_dir}")


if __name__ == "__main__":
    main()
