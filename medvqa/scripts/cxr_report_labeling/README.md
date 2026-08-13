# CXR report labeling scripts

Notebook-friendly entry points used by
`medvqa/notebooks/datasets/CXR report labeling pipeline.ipynb`.

Canonical class ontology (YAML + reviews/HTML):  
`medvqa/datasets/cxr_report_labeling/ontology/`  
NLI class prompts: `medvqa/prompts/cxr_classes/`.

| Script | Role |
|--------|------|
| `run_srr_bert_leaves.py` | GPU SRR-BERT labeling + disk cache |
| `sample_reports.py` | K1/K2/K3 sampling → `results/.../samples.jsonl` |
| `annotate_reports_with_llm.py` | Capped sample LLM annotation (interactive or Gemini hybrid Batch: prepare → submit → collect) |
| `run_annotation_batch_loop.py` | Tier-1 overnight loop: chunked prepare → submit (429 backoff) → collect |
| `compact_sample_annotations.py` | Migrate legacy flat annotations → namespaced `query_hash` cache + fan-out |
| `eval_regex_vs_llm.py` | Metrics (+ optional FP/FN dumps) vs LLM gold; uses `batch_match_prompt_classes` (`--num_processes`, `--lightweight`) |

Slurm: `medvqa/slurm/launch_cxr_eval_regex_vs_llm.sh` (CPU-only lightweight recall scan; default 16 CPUs).
| `snapshot_regex_refine.py` | Freeze annotations/samples/regexes under `.agent/regex_refine/<stamp>/` |
| `eval_regex_subgraph.py` | Agent API: leaf regexes from a workdir + ontology label/match propagation |
| `run_regex_mini_tests.py` | Per-class regex mini-report sanity tests (`cxr_class_mini_tests/by_class/`) |
| `apply_regex_to_dataset.py` | Full-dataset regex matches |
| `verify_positive_matches_with_llm.py` | LLM verify regex positives |
| `materialize_final_labels.py` | Final Unmentioned / 5-way labels |

### Overnight Tier-1 Gemini Batch loop (Slurm)

CPU-only job; one chunk of unique `query_hash`es at a time (fits ~3M enqueued-token quota). Progress also goes to `cache/cxr_report_labeling/<run_id>/api_tmp/gemini/batch_api/loop_progress.jsonl`.

**Sampling:** `sample_reports` / K1–K3 picks are content-unique by `report_text_hash` across datasets.

**Annotation cache layout:**

```text
cache/.../llm_annotations/<model>/sys_<8hex>_t<temp>_mt<max_tokens>/
  sample_annotations.jsonl
```

- `content_key` = hash(report_text_hash | prompt_hash | system_prompt_hash) — model-agnostic
- `query_hash` = hash(content_key | model | config) — inference resume key
- Leftover-only retry (same model, new config): point `--skip_annotation_paths` at the prior jsonl; writes go to a new `sys_…_mt…` dir
- System-prompt change → new `sys_<hash>` dir and full re-annotate (content_keys all change)

```bash
# Smoke one chunk:
MAX_CHUNKS=1 TIME=4:00:00 ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh

# Full multi-day run:
TIME=3-00:00:00 CHUNK_SIZE=2000 ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh

# Leftover-only with a new max_tokens config:
MAX_TOKENS_PER_REQUEST=8192 \
SKIP_ANNOTATION_PATHS="/path/to/sys_…_t0_mt2048/sample_annotations.jsonl" \
./medvqa/slurm/launch_cxr_annotate_batch_loop.sh

# Leftover-only with Gemini 3.5 Flash Lite (thinking_level=minimal via reasoning_effort):
MODEL_NAME=gemini-3.5-flash-lite \
MAX_TOKENS_PER_REQUEST=8192 \
MAX_CHUNKS=1 TIME=1:00:00 CHUNK_SIZE=50 \
SKIP_ANNOTATION_PATHS="/path/to/gemini-2.5-flash-lite/sys_…_t0_mt2048/sample_annotations.jsonl" \
./medvqa/slurm/launch_cxr_annotate_batch_loop.sh
# Writes llm_annotations/gemini-3.5-flash-lite/sys_…_t0_mt8192_reminimal/ (not the 2.5 jsonl).

# Or locally / interactive:
python -m medvqa.scripts.cxr_report_labeling.run_annotation_batch_loop \
  --run_id v1_k1200_iu400 --chunk_size 2000 --max_chunks 1 --api_key_name GEMINI_API_KEY
```

Requires `GEMINI_API_KEY` in the environment (repo `.env` loaded by `medvqa.settings`).

Compact a legacy flat `llm_annotations/<model>/sample_annotations.jsonl`:

```bash
python -m medvqa.scripts.cxr_report_labeling.compact_sample_annotations \
  --run_id v1_k1200_iu400 --dry_run
# then without --dry_run after reviewing the summary
```

### Regex refine (agent playbook)

**Full handoff for the next agent:** [`REGEX_REFINE.md`](REGEX_REFINE.md) (status, layout, mid-node hierarchy, partition rules, promote checklist).

Use this while Slurm may still be appending live annotations. Agents must **not** edit the live jsonl or production `medvqa/datasets/regular_expressions/cxr_classes/`; work only under `.agent/regex_refine/`.

1. **Snapshot** (coordinator) — freeze LLM gold; copy **owned** stems plus auto **readonly ontology descendants** into the group workdir (`owned_classes.json`: `owned_class_ids` = promote/edit; `eval_class_ids` = closed subgraph for eval):

```bash
python -m medvqa.scripts.cxr_report_labeling.snapshot_regex_refine \
  --run_id v1_k1200_iu400 --classes breast_implant --group_id breast_implant
# Mid-nodes: same command — descendants are added as readonly by default.
# Leaf-only escape hatch: --no_include_ontology_descendants
```

2. **Baseline eval** on the group workdir using **`eval_class_ids`** (not owned-only):

```bash
python -m medvqa.scripts.cxr_report_labeling.eval_regex_subgraph \
  --classes breast_implant \
  --regex_classes_dir .agent/regex_refine/<stamp>/groups/breast_implant/cxr_classes \
  --annotations_path .agent/regex_refine/<stamp>/annotations.snapshot.jsonl \
  --samples_path .agent/regex_refine/<stamp>/samples.snapshot.jsonl \
  --out_dir .agent/regex_refine/<stamp>/groups/breast_implant/eval
```

3. **Launch sub-agents** — **fill the paste-ready [Sub-agent brief template](REGEX_REFINE.md#sub-agent-brief-template)** in `REGEX_REFINE.md` (one Task prompt per group). That template forces agents to read `REGEX_REFINE.md`, README, and `medvqa/prompts/cxr_classes/<id>.txt`. Edit only `owned_class_ids`; re-eval with full `eval_class_ids`. Mid-nodes: residual umbrella phrasing after children. Do **not** edit `cxr_patterns.py` OR-composition.

4. **Stop for review** — write `groups/<group_id>/REPORT.md` (before/after metrics, diff notes). Coordinator promotes **only** `owned_class_ids` into production `cxr_classes/` after human approval (never the whole folder / readonly descendants). Mini-tests (`run_regex_mini_tests`) validate **production** only — update fixtures after promote.

See **Mid-nodes / hierarchy-aware refine** and **Sub-agent brief template** in [`REGEX_REFINE.md`](REGEX_REFINE.md). Partition rule: disjoint ontology subgraphs; umbrellas / `cxr_patterns.py` stay coordinator-owned after children finish.

### Regex FN mining (full-corpus hunt then fit)

Separate from sample-gold refine: find prompt-true regex misses on
`full_regex_matches/*.jsonl` (strata A/B/C) and optionally sample-gold FNs via
`sample_gold_fns.pkl` (stratum D), turn them into short mini-test positives, then
fit owned leaves. Playbook + paste-ready hunt/fit briefs: [`FN_MINE.md`](FN_MINE.md).
Kit: `medvqa.datasets.cxr_report_labeling.fn_mine`. Coordinator stop/go between
waves; no extra LLM annotation. Rebuild D index:
`python -m medvqa.datasets.cxr_report_labeling.fn_mine --build-sample-gold-fn-index`.
