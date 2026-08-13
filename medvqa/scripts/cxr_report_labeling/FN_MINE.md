# Full-corpus regex FN mining (hunt then fit)

Coordinator + sub-agent workflow for finding **prompt-true** findings that
production regex never fires on, then fitting owned `.txt` files to new
mini-test positives **without** extra LLM annotation.

**Interactive stop/go:** do not auto-chain waves. After hunt, after fit, and
before promote / the next class list, **stop and wait for an explicit human OK**.

Sample-gold P/R + mini-tests are the metrics. Full-dataset P/R is unknowable
without LLM — do not claim it.

Related: sample-gold refine (different job) lives in `[REGEX_REFINE.md](REGEX_REFINE.md)`.

## Hunt vs fit


| Phase      | Goal                                                        | Allowed writes                                                                        | Forbidden                                                                                                       |
| ---------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **1 Hunt** | Genuine prompt-true FNs → **short** mini-test positives     | `cxr_class_mini_tests/by_class/<owned_id>.json` positives only; `groups/<id>/HUNT.md` | production `cxr_classes/`, `cxr_patterns.py`, prompts, ontology, live annotation jsonl, other classes' fixtures |
| **2 Fit**  | Make hard mini-test positives pass; hold sample-gold floors | workdir owned `.txt` + `REPORT.md`                                                    | production until human promote; `cxr_patterns.py`; readonly descendants; prompts; live jsonl                    |


FNs on the full corpora are uids where class X is **absent** from
`cache/.../full_regex_matches/<dataset>.jsonl` (`positive_classes` is sparse).
Judge prompt fidelity on the report text. Extra LLM gold is unnecessary.

A parallel Slurm track verifies **regex positives**; it cannot discover these FNs.

## Search kit (required)

Module: `[medvqa/datasets/cxr_report_labeling/fn_mine.py](../../datasets/cxr_report_labeling/fn_mine.py)`

Sub-agents **may write Python** but must use this API — never
`load_mimiccxr_reports()` / `load_all_datasets()` in a hot loop.

**One Python process per hunt:** do A/B/C, `production_misses` checks, and
`append_mini_test_positive` in a **single** `python <<'PY'` (or one REPL).
Do **not** spawn a new interpreter per stratum — each cold start reloads the
match index (~120MB) and report texts.

**Keyword index (stratum C):** `keyword_candidates` uses a token inverted index
(built once, pickled under `.agent/regex_refine/_fn_mine/`). Prefetch before a
wave if the cache is missing:

```bash
python -m medvqa.datasets.cxr_report_labeling.fn_mine --build-keyword-index --workers 8
```

**Sample-gold FN index (stratum D):** `sample_gold_fns` loads
`.agent/regex_refine/_fn_mine/sample_gold_fns.pkl` (LLM+ & regex− on the
annotated sample). Prefetch / refresh after promotes:

```bash
python -m medvqa.datasets.cxr_report_labeling.fn_mine --build-sample-gold-fn-index --workers 16
```

Prefer **short retrieval anchors** as C terms (`airless`, `Luftsichel`,
`atelectasis/collapse`); use long phrases as post-filters on the shortlist, not
as `keyword_candidates` terms.

```python
from medvqa.datasets.cxr_report_labeling.fn_mine import (
    iter_reports,           # join match jsonl → lazy texts
    keyword_candidates,     # retrieval hint, not a second regex
    sample_gold_metrics,    # latest _recall_scan* production-OR P/R
    sample_gold_fns,        # stratum D: LLM+ & regex− on annotated sample
    sample_gold_fn_count,
    mini_tests_for,
    production_misses,      # True iff current production regex does not fire
    report_text,
    append_mini_test_positive,
    ontology_related_class_ids,
)

# Stratum A: zero regex hits
for row in iter_reports("mimiccxr", regex_negative_for="bone_fracture",
                        zero_hits=True, limit=50, seed=0):
    ...

# Stratum B leaves: X miss + ontology parent/sibling/child hit
for row in iter_reports("chexpert_plus", regex_negative_for="sternotomy_wires",
                        related_hits=True, limit=50, seed=0):
    ...

# Stratum B mid-nodes: production OR already folds children into X, so pass
# an explicit cross-branch list (brief).
for row in iter_reports("chexpert_plus", regex_negative_for="support_device_present",
                        related_hits=["bone_fracture"], limit=50, seed=0):
    ...

# Stratum C: keyword filter on X-negative reports
rows = keyword_candidates("chexpert_plus", "support_device_present",
                          terms=["fixation device", "nuss", "radiopaque drain"],
                          regex_negative=True, limit=50)

# Stratum D: sample-gold FNs (LLM positive, hierarchy-aware regex miss)
# Prefetch pickle: python -m medvqa.datasets.cxr_report_labeling.fn_mine \
#   --build-sample-gold-fn-index
for row in sample_gold_fns("pacemaker_or_icd", limit=50):
    quote = row.get("relevant_quote") or ""
    # judge prompt fidelity on quote; append short mini-test positive if True
    ...
```

**Caps (every hunt brief):** ≤50 uids/dataset, ≤4 datasets, stop after **k=15**
accepted TPs (or earlier if the lexicon is exhausted). Prefer MIMIC + CheXpert.
When stratum D is enabled, prefer D first (annotated FNs), then A/B/C for residual
full-corpus lexicon holes.

**Mini-test positives:** short sentences *inspired by* the report — not a 2k-char
dump. Current production regex **must** miss (`production_misses` / `require_miss=True`).
Do not delete existing negatives.

Fixtures: `[cxr_class_mini_tests/by_class/<class_id>.json](../../datasets/regular_expressions/cxr_class_mini_tests/by_class/)`.
One file per class so parallel agents do not merge-conflict.

```bash
python -m medvqa.scripts.cxr_report_labeling.run_regex_mini_tests
```

Mini-tests validate **production**. Hunt positives are expected to **fail** until
a later fit is promoted.

**Sample-gold FN index (stratum D):** pickle at
`.agent/regex_refine/_fn_mine/sample_gold_fns.pkl`. Rebuild after regex promotes
(or when annotations change):

```bash
python -m medvqa.datasets.cxr_report_labeling.fn_mine --build-sample-gold-fn-index
# optional: --workers 16 --out .agent/regex_refine/_fn_mine/sample_gold_fns.pkl
```

FN = LLM gold positive (`definitely`/`likely true`) and hierarchy-aware regex miss
on the annotated sample. Independent of `full_regex_matches` precompute. Many D
rows are bad gold / prompt-unfaithful — reject those; do not chase 100% sample
recall.

## Program loop

**Leaves-first hunt, then bottom-up fit.** Do not fit a mid-node until every
descendant has been hunted **and** fitted. Tracker:

`.agent/regex_refine/fn_mine_status.json`

```bash
python -m medvqa.datasets.cxr_report_labeling.fn_mine --status
```

`next_hunt_class_ids()` / `next_fit_class_ids()` implement Kahn-style readiness
(fewest descendants first). Each wave ≤5 disjoint `owned_class_ids`.

1. Hunt **leaves** (`n_children == 0`) until that queue is empty (skip TB / PVC
   unless asked). Park mid-node hunt TPs if a smoke wave already wrote them.
2. Then hunt mid-nodes whose **children are all hunt-done** (residual parent
   phrasing only).
3. **Fit** only when `hunt=done` and every descendant has `fit=done` (leaves
   become fit-ready as soon as their own hunt is reviewed).
4. Zero accepted TPs still gets `HUNT.md`; fit may be skipped (`fit=skipped`).
5. Skip unless asked: `tuberculosis`, `pulmonary_vascular_congestion`.
6. Do not reopen CPA / tube “prompt ceiling” FNs from old `REPORT.md` unless hunt
   finds **new full-corpus lexicon** holes.

Wave 0 hunted two mid-nodes (`support_device_present`, `support_tube_present`,
`bone_fracture`) as smoke. Those hunts are recorded as done; **fit stays blocked**
until the rest of each subtree is hunted and fitted.

Env: `module load conda && conda activate py313`. Canonical run: `v1_k1200_iu400`.

## Coordinator: Hunt brief template

Copy into each hunt Task prompt. Replace `<…>`. One agent ↔ one owned class.
Disjoint `owned_class_ids` across agents in the wave.

```markdown
You are a CXR regex **FN-hunt** sub-agent in `/home/pamessina/medvqa`.

### Read first (required)
1. `medvqa/scripts/cxr_report_labeling/FN_MINE.md` — hunt vs fit, kit API, caps, mid-node rules.
2. `medvqa/scripts/cxr_report_labeling/REGEX_REFINE.md` — prompt fidelity, mid-node residual phrasing (do not re-lexicon children).
3. Class prompt (semantic source of truth, **read-only**):
   `medvqa/prompts/cxr_classes/<owned_class_id>.txt`
4. Production regex (read-only): `medvqa/datasets/regular_expressions/cxr_classes/<owned_class_id>.txt`
5. Mini-tests: `medvqa/datasets/regular_expressions/cxr_class_mini_tests/by_class/<owned_class_id>.json`
6. Env: `module load conda && conda activate py313`. Canonical run: `v1_k1200_iu400`.

### Ownership (STRICT)
- Wave: <wave_id>
- **Owned class (only this one):** `<owned_class_id>`
- Notes dir (optional): `.agent/regex_refine/<stamp>/groups/<group_id>/` — write `HUNT.md` here if the dir exists; otherwise write `medvqa/datasets/regular_expressions/cxr_class_mini_tests/by_class/<owned_class_id>.HUNT.md`
- **Allowed write:** append short **positives** to `by_class/<owned_class_id>.json` via `append_mini_test_positive` (or equivalent). Do not delete negatives.
- **FORBIDDEN:** production `cxr_classes/*.txt`, `cxr_patterns.py`, prompts, ontology, live `sample_annotations.jsonl`, other classes' mini-tests, other groups.

### Kit (required)
Use `medvqa.datasets.cxr_report_labeling.fn_mine` only for corpus access.
Never call `load_mimiccxr_reports()` / `load_chexpert_plus_reports()` / `load_all_datasets()` in a loop.
`keyword_candidates` **loads** the prebuilt token inverted index pickle (do **not**
run `--build-keyword-index` unless the coordinator says the cache is missing).
`sample_gold_fns` **loads** `.agent/regex_refine/_fn_mine/sample_gold_fns.pkl`
(do **not** run `--build-sample-gold-fn-index` unless the coordinator says to).
Prefer short C-term anchors; long phrases = post-filters on the shortlist.
**One** `python <<'PY'` for the whole hunt.

### Caps
- Datasets (in order): <datasets, default mimiccxr chexpert_plus rexgradient iuxray>
- ≤50 uids per dataset per stratum helper call
- Stop after **k=15** accepted TPs (or sooner if lexicon exhausted)
- New positives must be **short**; `production_misses(<owned_class_id>, sentence)` must be True
- **One** `python <<'PY'` for the whole hunt (reuse process caches). Prefer short C-term anchors.

### Strata
- **D (when enabled — prefer first):** `sample_gold_fns(<owned_class_id>, datasets=[...], limit=50)`
  Use `relevant_quote` for prompt judgment; reject bad gold / prompt-unfaithful.
  Still require `production_misses` before append (pickle may predate last promote).
- A: `iter_reports(..., regex_negative_for=<owned_class_id>, zero_hits=True, limit=50, seed=<seed>)`
- B: `iter_reports(..., regex_negative_for=<owned_class_id>, related_hits=<True | [class_ids]>, limit=50, seed=<seed>)`
  Coordinator related_hits: <related_hits>
- C: `keyword_candidates(..., terms=<terms>, regex_negative=True, limit=50)`
  Keyword terms: <comma-separated terms>
Coordinator: enable_stratum_D=<yes|no, default no for legacy waves; yes for sample-gold closing passes>

### Hunt focus
<one-paragraph focus, e.g. residual umbrella fixation-device / unnamed drain; lookbehind on apparent sternal fractures; true sternal wires only — reject pectus bar>

### Prompt-fidelity rejects (do not add as positives)
<e.g. pectus bar / NUSS as sternotomy_wires; indication-only ETT; history-only; device fracture as bone_fracture>

### Mid-node (if owned is a parent)
Optimize **residual parent phrasing** children miss. Do not re-implement readonly child lexicons. Production OR already counts child hits as parent hits on the full-match jsonl.

### Deliverable
1. Append accepted short positives to `by_class/<owned_class_id>.json`.
2. Write `HUNT.md` with:
   - accepted TPs: sentence, source uid, stratum (D/A/B/C), why prompt-true, confirmation that production misses
   - rejected candidates: uid + reason (prompt-unfaithful / already covered / too long / bad gold)
   - stance: lexicon hole vs prompt ceiling vs empty hunt
Return a short summary (n accepted / n rejected). Do **not** edit regex. Do **not** ask to promote.
```



## Coordinator: Fit brief template

Use only **after** human review of hunt TPs. Snapshot first (`snapshot_regex_refine`).
One agent ↔ one group; disjoint `owned_class_ids`. Nested mid-nodes: barrier
(do not fit parent while a child is in flight).

```markdown
You are a CXR regex **FN-fit** sub-agent in `/home/pamessina/medvqa`.

### Read first (required)
1. `medvqa/scripts/cxr_report_labeling/FN_MINE.md` — fit floors, promote rules.
2. `medvqa/scripts/cxr_report_labeling/REGEX_REFINE.md` — gold semantics, mid-node hierarchy, promote pitfalls.
3. Short pointer: `medvqa/scripts/cxr_report_labeling/README.md` (“Regex refine”).
4. Class prompt(s) — **read-only**: `medvqa/prompts/cxr_classes/<owned_class_id>.txt`
5. Hunt notes: `groups/<group_id>/HUNT.md` (approved TPs only; ignore rejected).
6. Env: `module load conda && conda activate py313`. Canonical run: `v1_k1200_iu400`.

### Ownership (STRICT)
- Stamp: `.agent/regex_refine/<stamp>/`
- Group: `groups/<group_id>/`
- **Edit ONLY** owned leaves in `owned_classes.json` → `owned_class_ids`:
  `groups/<group_id>/cxr_classes/<owned_class_id>.txt`
- **READONLY:** `readonly_class_ids` (ontology descendants for eval), production `cxr_classes/`, `cxr_patterns.py`, ontology YAML, prompts, live annotation jsonl, other groups.
- Do not copy readonly descendant `.txt` out of the group folder.

### Eval set (hierarchy-aware)
- Use **all** `eval_class_ids` from `owned_classes.json`.
- Gold and regex matches propagate within that set (`matched_via_classes`, `gold_via_classes`, `llm_by_class`).
- This is **not** production OR-composition. Residual parent phrasing only; do not re-implement readonly child lexicons.

### Baseline (from coordinator)
- Parent/owned metrics: P=<…> R=<…> TP/FP/FN=<…>/<…>/<…> (n=<…>)
- FN dump: `groups/<group_id>/eval_baseline/<class_id>_fn.jsonl`
- FP dump: `groups/<group_id>/eval_baseline/<class_id>_fp.jsonl`

### Goal
- Fit owned `.txt` so **all hard mini-test positives** pass (old + **approved** hunt TPs in `by_class/<owned_class_id>.json`).
- Floors vs frozen sample snapshot: **R ≥ baseline**; **P ≳ 0.85** **or** **FP ≤ 1.5× baseline** unless leftover FPs are documented bad gold.
- Prompt fidelity > noisy sample LLM gold. Document bad-gold FNs rather than chasing 100% recall.
- Do not claim full-corpus P/R.

### Re-eval (after each edit)
```bash
cd /home/pamessina/medvqa
module load conda && conda activate py313
python -m medvqa.scripts.cxr_report_labeling.eval_regex_subgraph \
  --classes <space-separated eval_class_ids> \
  --regex_classes_dir .agent/regex_refine/<stamp>/groups/<group_id>/cxr_classes \
  --annotations_path .agent/regex_refine/<stamp>/annotations.snapshot.jsonl \
  --samples_path .agent/regex_refine/<stamp>/samples.snapshot.jsonl \
  --out_dir .agent/regex_refine/<stamp>/groups/<group_id>/eval
```

You may sanity-check mini-test sentences against the **workdir** pattern via
`load_pattern(<owned>, directory=.../groups/<group_id>/cxr_classes)`. Production
`run_regex_mini_tests` will still fail until promote.

### Deliverable
Write `groups/<group_id>/REPORT.md` with:
- Before/after P/R and TP/FP/FN for **each owned** class
- What changed (regex strategy); which hunt TPs now match
- Leftover FNs/FPs with bad-gold attribution; cite `matched_via_classes` / `gold_via_classes` / `llm_by_class` when relevant
- Promote stance per owned stem (PROMOTE / HOLD)

Return a short summary with metrics + promote recommendation.
**Do not promote yourself.** Wait for the coordinator/human.
```



## Promote (coordinator, after human OK)

```bash
cp .agent/regex_refine/<stamp>/groups/<id>/cxr_classes/<class_id>.txt \
   medvqa/datasets/regular_expressions/cxr_classes/<class_id>.txt
python -m medvqa.scripts.cxr_report_labeling.run_regex_mini_tests
```

Only `owned_class_ids` stems. Never copy `readonly_class_ids`.