# CXR report labeling scripts

[`medvqa/notebooks/datasets/CXR report labeling pipeline.ipynb`](../../notebooks/datasets/CXR%20report%20labeling%20pipeline.ipynb)
is a **runbook and interactive QA** notebook (paths, Slurm/CLI copy-paste, complexity / P/R/F1 / per-report regex browsers). It is **not** the job driver: Batch annotate, full-corpus regex, eval, verify, and materialize run from the scripts and Slurm wrappers below.

How `v1_k1200_iu400` labels were actually produced (models, leftover merge, output paths): **[Released labels (`v1_k1200_iu400`)](#released-labels-v1_k1200_iu400)** below. The rest of this file is the generic pipeline how-to.

Canonical class ontology (YAML + reviews/HTML):  
`medvqa/datasets/cxr_report_labeling/ontology/`  
NLI class prompts: `medvqa/prompts/cxr_classes/`.

| Script | Role |
|--------|------|
| `run_srr_bert_leaves.py` | GPU SRR-BERT labeling + disk cache |
| `sample_reports.py` | K1/K2/K3 sampling → `results/.../samples.jsonl` |
| `summarize_regex_matches.py` | Inventory full-corpus regex matches + remaining unique `content_key`s / `ceil(N/CHUNK_SIZE)` (IU first) |
| `annotate_reports_with_llm.py` | LLM annotation: sample cartesian **or** `--job_source regex_positives` / `no_finding` (Gemini hybrid Batch: prepare → submit → collect) |
| `run_annotation_batch_loop.py` | Overnight loop: chunked prepare → submit (429 backoff) → collect; threads `JOB_SOURCE` / `DATASETS` |
| `compact_sample_annotations.py` | Migrate legacy flat annotations → namespaced `query_hash` cache + fan-out |
| `eval_regex_vs_llm.py` | Metrics (+ optional FP/FN dumps) vs LLM gold; uses `batch_match_prompt_classes` (`--num_processes`, `--lightweight`). Default class list includes `no_finding` as a regex-empty proxy |
| `snapshot_regex_refine.py` | Freeze annotations/samples/regexes under `.agent/regex_refine/<stamp>/` |
| `eval_regex_subgraph.py` | Agent API: leaf regexes from a workdir + ontology label/match propagation |
| `run_regex_mini_tests.py` | Per-class regex mini-report sanity tests (`cxr_class_mini_tests/by_class/`) |
| `apply_regex_to_dataset.py` | Full-dataset regex matches |
| `verify_positive_matches_with_llm.py` | **Deprecated** (`orchestrate_api_calls`, `--max_queries 50`). Use `JOB_SOURCE=regex_positives` on the Batch loop. |
| `materialize_final_labels.py` | Integrate regex-backed 6-way labels (`--stage regex_backed` then `--stage final` for `no_finding`) |
| Slurm: `launch_cxr_materialize_final_labels.sh` | Rematerialize `final_labels/` (32G+; full verify jsonl load) |

Slurm: `medvqa/slurm/launch_cxr_eval_regex_vs_llm.sh` (CPU-only lightweight recall scan; default 16 CPUs).
Slurm annotate loop: `medvqa/slurm/launch_cxr_annotate_batch_loop.sh`.

### Overnight Tier-1 Gemini Batch loop (Slurm)

CPU-only job; one chunk of unique `query_hash`es at a time (fits ~3M enqueued-token quota). Progress also goes to `cache/cxr_report_labeling/<run_id>/api_tmp/gemini/batch_api/loop_progress_<job_source>_<dataset>.jsonl`.

**Sampling:** `sample_reports` / K1–K3 picks are content-unique by `report_text_hash` across datasets.

**Annotation cache layout:**

```text
cache/.../llm_annotations/<model>/sys_<8hex>_t<temp>_mt<max_tokens>/
  sample_annotations.jsonl
  verify_<dataset>.jsonl
  verify_no_finding_<dataset>.jsonl
  failed_query_hashes_<dataset>.jsonl
  failed_query_hashes_no_finding_<dataset>.jsonl
  failed_query_hashes_samples.jsonl
  failed_query_hashes.jsonl   # legacy unscoped; read-only
```

- `content_key` = hash(report_text_hash | prompt_hash | system_prompt_hash) — model-agnostic
- `query_hash` = hash(content_key | model | config) — inference resume key
- Leftover-only retry (same model, new config): point `--skip_annotation_paths` at the prior jsonl; writes go to a new `sys_…_mt…` dir
- System-prompt change → new `sys_<hash>` dir and full re-annotate (content_keys all change)
- Same-config parse/length/missing-batch failures go in `failed_query_hashes_<dataset>.jsonl` (writes are dataset-scoped) and are **not** retried in later chunks of that config. A legacy unscoped `failed_query_hashes.jsonl` may still be **read**. Leftover retry = new `MODEL_NAME` / `MAX_TOKENS_PER_REQUEST` + `SKIP_ANNOTATION_PATHS` on **successes only** (not the fail sidecar).
- Batch tmp files (`batch_job_prepare_*.json`, `batch_requests_*.jsonl`) and `loop_progress_*.jsonl` are tagged by job_source + dataset so parallel loops cannot clobber each other.
- The overnight loop preloads reports, regex matches, intents, sample-gold skip keys, and remaining unique jobs once; each chunk only re-scans verify/fail sidecars, fans out, and slices ``CHUNK_SIZE``.
- Full-corpus verify writes `verify_<dataset>.jsonl` — do **not** append those rows into `sample_annotations.jsonl`.

```bash
# Smoke one chunk:
MAX_CHUNKS=1 TIME=4:00:00 ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh

# Full multi-day run:
TIME=3-00:00:00 CHUNK_SIZE=2000 ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh

# Leftover-only: skip successes (sample gold + 2.5 verify), not the fail sidecar.
ANN25="$MEDVQA_WORKSPACE_DIR/cache/cxr_report_labeling/v1_k1200_iu400/llm_annotations/gemini-2.5-flash-lite/sys_6d686537_t0_mt2048"
MAX_TOKENS_PER_REQUEST=8192 \
SKIP_ANNOTATION_PATHS="$ANN25/sample_annotations.jsonl $ANN25/verify_iuxray.jsonl" \
./medvqa/slurm/launch_cxr_annotate_batch_loop.sh

# Leftover-only with Gemini 3.5 Flash Lite (thinking_level=minimal via reasoning_effort):
MODEL_NAME=gemini-3.5-flash-lite \
MAX_TOKENS_PER_REQUEST=8192 \
MAX_CHUNKS=1 TIME=1:00:00 CHUNK_SIZE=50 \
SKIP_ANNOTATION_PATHS="$ANN25/sample_annotations.jsonl $ANN25/verify_mimiccxr.jsonl" \
./medvqa/slurm/launch_cxr_annotate_batch_loop.sh
# Writes llm_annotations/gemini-3.5-flash-lite/sys_…_t0_mt8192_reminimal/ (not the 2.5 jsonl).
# Then merge recovered rows into the 2.5 verify jsonl by content_key (see release notes).

# Or locally / interactive:
python -m medvqa.scripts.cxr_report_labeling.run_annotation_batch_loop \
  --run_id v1_k1200_iu400 --chunk_size 2000 --max_chunks 1 --api_key_name GEMINI_API_KEY
```

Requires `GEMINI_API_KEY` in the environment (repo `.env` loaded by `medvqa.settings`).

### Released labels (`v1_k1200_iu400`)

Provenance for the four-corpus 6-way labels under `$MEDVQA_WORKSPACE_DIR/results/cxr_report_labeling/v1_k1200_iu400/final_labels/` (2026-08). This is **not** a second pipeline — same scripts as below, with leftover recovery and a later rename of the non-candidate sentinel.

**Do not rematerialize while parse-fail leftovers are unresolved.** `--stage final` overwrites the same `{dataset}.jsonl` / `_sparse.jsonl` / `_labels.npz` paths as `--stage regex_backed` (stage is only in `{dataset}_manifest.json`).

#### Inputs

| Piece | Location |
|-------|----------|
| Run | `v1_k1200_iu400` |
| Regex matches | `cache/.../full_regex_matches/{iuxray,mimiccxr,chexpert_plus,rexgradient}.jsonl` |
| Primary LLM | `gemini-2.5-flash-lite` / `sys_6d686537_t0_mt2048` (`ANN25`) |
| Leftover LLM | `gemini-3.5-flash-lite` / `sys_6d686537_t0_mt8192_reminimal` (`ANN35`; `thinking_level=minimal`) |
| Sample gold | `$ANN25/sample_annotations.jsonl` |
| Verify (regex positives) | `$ANN25/verify_<dataset>.jsonl` (includes merged 3.5 leftovers) |
| Verify (`no_finding`) | `$ANN25/verify_no_finding_<dataset>.jsonl` |

Shorthand:

```text
ANN25=.../llm_annotations/gemini-2.5-flash-lite/sys_6d686537_t0_mt2048
ANN35=.../llm_annotations/gemini-3.5-flash-lite/sys_6d686537_t0_mt8192_reminimal
```

#### Sequence that produced the files

1. **Sample cartesian gold** (2.5) → `sample_annotations.jsonl`.
2. **Full-corpus regex** already on disk; inventory with `summarize_regex_matches` (32G+ for large corpora; login-node `load_jsonl` OOMs).
3. **`JOB_SOURCE=regex_positives`** overnight Batch loops per dataset (2.5). Writes `verify_<dataset>.jsonl`. Original 2.5 loops auto-skip sample gold when `model_name=gemini-2.5-flash-lite`.
4. **`--stage regex_backed`** materialize (regex classes only; no `no_finding`).
5. **`JOB_SOURCE=no_finding`** (auto-false reports skip the API) → `verify_no_finding_<dataset>.jsonl`.
6. **Parse-fail leftovers** retried with Gemini 3.5 Flash Lite. `SKIP_ANNOTATION_PATHS` must list **both** `$ANN25/sample_annotations.jsonl` **and** `$ANN25/verify_<dataset>.jsonl` (successes only — not the fail sidecar). Writes go to `ANN35`, not 2.5 jsonl.
7. **Merge 3.5 → 2.5 verify** by `content_key`: append rows whose key is not already in 2.5 verify and not in sample gold. Do **not** `cat` the whole 3.5 file (a bad leftover run can re-label sample gold). Backups:
   - `verify_mimiccxr.before_gemini35_merge_20260817T221008Z.jsonl`
   - `verify_chexpert_plus.before_gemini35_merge_20260817T231000Z.jsonl`
   - `verify_rexgradient.before_gemini35_merge_20260818T014338Z.jsonl`
8. **`--stage final`** (regex-backed + `no_finding` auto-false / LLM). Rematerialize after the `unmentioned` → `regex_unmatched` rename (IU locally; MIMIC / CheXpert / ReX Slurm job `cxr_materialize-94277`).

3.5 leftover rows merged into 2.5 verify: **MIMIC 80**, **CheXpert Plus 175**, **ReX 18**. IU regex leftovers were handled earlier on sample gold (~23 rows). `$ANN35/verify_mimiccxr.jsonl` has extra rows from a bad skip-path run; the merge filter kept 80.

#### Released outputs

`results/cxr_report_labeling/v1_k1200_iu400/final_labels/`

| File | Role |
|------|------|
| `{dataset}.jsonl` | Compact: every regex class (+ `no_finding`); non-candidates are `regex_unmatched` |
| `{dataset}_sparse.jsonl` | Candidate (and `no_finding`) cells only — no `regex_unmatched` rows |
| `{dataset}_labels.npz` | uint8 matrix; `0=regex_unmatched` … `5=definitely false`; `255=unresolved` (unused on this release) |
| `{dataset}_uids.jsonl` / `{dataset}_class_ids.json` | Row / column order |
| `{dataset}_manifest.json` | Counts, stage, LLM paths |

Counts from the `regex_unmatched` rematerialize (`stage=final`, `n_fail_sidecar_leftovers=0`):

| Dataset | Reports | Candidate cells | `no_finding` auto / LLM |
|---------|--------:|----------------:|------------------------:|
| IU | 3,955 | 8,783 | 2,419 / 1,536 |
| MIMIC | 227,835 | 1,094,091 | 193,276 / 34,559 |
| CheXpert Plus | 223,462 | 1,364,946 | 212,720 / 10,742 |
| ReXGradient | 160,000 | 456,093 | 104,739 / 55,261 |

Manifests still report `n_failed_query_hashes: 257` from the **legacy unscoped** `$ANN25/failed_query_hashes.jsonl` union (read-only). That sidecar did not block release.

#### Incidents worth knowing (do not repeat)

- **`SKIP_ANNOTATION_PATHS` drop:** the Slurm launcher used to emit repeated `--skip_annotation_paths` flags; argparse `nargs='+'` kept only the last path and dropped sample gold. MIMIC leftover job queued ~8.4k junk instead of ~80. Fix: one flag, all paths (`launch_cxr_annotate_batch_loop.sh` + `flatten_skip_annotation_paths`). Always pass sample gold **and** the 2.5 verify jsonl.
- **Parser `\bSure\b`:** `GPT_IS_ACTING_WEIRD_REGEX` matched “Blood Sure” inside report quotes (CheXpert 5/175 false parse fails). Check removed from `parse_llm_model_output`; five valid rows recovered from saved batch output (no re-API).
- **MIMIC verify contamination:** CheXpert rows landed in `verify_mimiccxr.jsonl` (timestamp collision). Foreign burst dropped to `verify_mimiccxr.dropped_chexpert_plus_20260817.jsonl`; MIMIC regex-positive loop retried.
- **CheXpert collect OOM:** full-file `load_jsonl` on the login node; relaunch with **32G+**.
- **ReX `no_finding` LLM share** is high (~34.5% of reports vs ~5% CheXpert), correlated with lower regex candidate density — not from an external prior.

#### Semantics (released 6-way)

Class C is a candidate if C or any ontology descendant is in regex `positive_classes` (jsonl ∪ DAG ancestors). Else **`regex_unmatched`** (legacy compact name `unmentioned`; not “clinically unmentioned” — lookbehinds can miss negated mentions). Among `{C} ∪ matched descendants`: `definitely true` > `likely true`; else C’s own LLM if C matched; else `unknown`. After regex-backed integrate, `no_finding` is auto-false if any other class is `definitely true` / `likely true`; otherwise LLM.

### IU-first regex-positive verify + integrate

Do **not** start MIMIC / CheXpert / ReX until IU integrate + `no_finding` have been eyeballed. Regex matches are already in `cache/.../full_regex_matches/`. Do **not** use `verify_positive_matches_with_llm.py`.

**Semantics:** class C is a candidate if C or any ontology descendant is in `positive_classes` (jsonl ∪ DAG ancestors). Else released `regex_unmatched` (legacy alias `unmentioned`). Presence entailment among `{C} ∪ matched descendants`: `definitely true` > `likely true`; else C's own LLM if C matched; else `unknown` (`resolution=descendants_matched_none_positive`). After regex-backed integrate, `no_finding` is auto-false if any other class is `definitely true` / `likely true`; otherwise LLM.

1. Inventory (CPU, no API) — remaining unique `content_key`s after sample gold, `ceil(N/2000)`:

```bash
python -m medvqa.scripts.cxr_report_labeling.summarize_regex_matches \
  --run_id v1_k1200_iu400 --datasets iuxray
# optionally omit --datasets to print all four (IU first)
```

2. Smoke one IU regex-positive chunk:

```bash
JOB_SOURCE=regex_positives DATASETS=iuxray MAX_CHUNKS=1 TIME=4:00:00 \
  ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh
```

3. Full IU loop (`MAX_CHUNKS` unset) after smoke looks healthy:

```bash
JOB_SOURCE=regex_positives DATASETS=iuxray TIME=3-00:00:00 \
  ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh
```

4. Materialize regex-backed 6-way labels; spot-check a few uids (empties + a regex-hit report):

```bash
python -m medvqa.scripts.cxr_report_labeling.materialize_final_labels \
  --dataset iuxray --stage regex_backed
# writes results/.../final_labels/iuxray.jsonl, iuxray_sparse.jsonl, iuxray_labels.npz
```

5. Second wave `no_finding` (auto-false reports skip the API), then final merge:

```bash
JOB_SOURCE=no_finding DATASETS=iuxray TIME=1-00:00:00 \
  ./medvqa/slurm/launch_cxr_annotate_batch_loop.sh

python -m medvqa.scripts.cxr_report_labeling.materialize_final_labels \
  --dataset iuxray --stage final
```

Leftover-only (failed hashes from this config): new `mt` / `MODEL_NAME` + `SKIP_ANNOTATION_PATHS` pointing at successful `verify_iuxray.jsonl` and `sample_annotations.jsonl` — not the fail sidecar.

**Rematerialize all corpora** (after label-semantics changes; no API). Backup `final_labels/` first if needed. Large datasets need Slurm **32G+**:

```bash
# one corpus
DATASET=mimiccxr MEM=64G TIME=2:00:00 ./medvqa/slurm/launch_cxr_materialize_final_labels.sh

# all four (final stage overwrites same paths)
DATASETS="iuxray mimiccxr chexpert_plus rexgradient" MEM=64G TIME=4:00:00 \
  ./medvqa/slurm/launch_cxr_materialize_final_labels.sh
```

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
`full_regex_matches/*.jsonl` (stratum C via `keyword_candidates`; A/B optional)
and optionally sample-gold FNs via `sample_gold_fns.pkl` (stratum D). Default
new waves are **S-first** (invent holes, then D+C; `similar_tokens` expands C
typos). Turn accepted misses into short mini-test positives, then fit owned
leaves. Playbook + paste-ready hunt/fit briefs: [`FN_MINE.md`](FN_MINE.md).
Kit: `medvqa.datasets.cxr_report_labeling.fn_mine`. Coordinator stop/go between
waves; no extra LLM annotation. Rebuild D index:
`python -m medvqa.datasets.cxr_report_labeling.fn_mine --build-sample-gold-fn-index`.
