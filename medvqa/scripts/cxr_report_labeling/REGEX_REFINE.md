# Sub-agent regex refinement (handoff)

Coordinator + sub-agent workflow for improving leaf regexes under
`medvqa/datasets/regular_expressions/cxr_classes/` using frozen LLM annotations
as gold. **Do not re-implement** the tooling unless something is broken.

**Full-corpus FN mining** (hunt missed true positives, then fit) is a separate
loop: [`FN_MINE.md`](FN_MINE.md) (kit + hunt/fit brief templates). Use that
file’s templates for hunt/fit waves; use **this** file’s template for
sample-gold refine.

**For the next coordinator (you):** read this file first, then run the
[Coordinator loop](#coordinator-loop-how-to-continue). When launching sub-agents,
**always fill and paste the [Sub-agent brief template](#sub-agent-brief-template)**
(one brief per group). Do not invent a thinner brief — sub-agents only know what
you put in the Task prompt.

## Goals

- Maximize **recall** (ideal ~100%) vs LLM gold (`definitely true` / `likely true`).
- Keep **precision** reasonable (do not flood FPs).
- Parallelize safely: each sub-agent owns a **disjoint** set of class leaves.
- Never edit live Slurm annotation outputs or `cxr_patterns.py` OR-composition from a sub-agent.

## Key code

| Path | Role |
|------|------|
| [`snapshot_regex_refine.py`](snapshot_regex_refine.py) | Freeze annotations/samples; create group workdirs under `.agent/regex_refine/<stamp>/` with **owned** stems + auto **readonly ontology descendants** |
| [`eval_regex_subgraph.py`](eval_regex_subgraph.py) | Leaf match from a workdir + ontology label/match propagation within `--classes`; writes metrics + FP/FN dumps (`matched_via_classes`, `gold_via_classes`, `llm_by_class`) |
| [`ontology_dag.py`](../../datasets/cxr_report_labeling/ontology_dag.py) | Parent/child DAG from `canonical_cxr_ontology.yaml` |
| [`cxr_patterns.load_pattern(name, directory=...)`](../../datasets/regular_expressions/cxr_patterns.py) | Load leaf `.txt` from an arbitrary directory; call `clear_pattern_cache()` after edits |
| [`run_regex_mini_tests.py`](run_regex_mini_tests.py) | Per-class fixtures under `cxr_class_mini_tests/by_class/` (uses **production** patterns, not workdirs) |
| [`fn_mine.py`](../../datasets/cxr_report_labeling/fn_mine.py) | Read-only full-corpus FN hunt kit (match jsonl + lazy texts) |
| [`FN_MINE.md`](FN_MINE.md) | Hunt then fit: kit API, caps, paste-ready hunt/fit briefs |
| Notebook §9 | Older `eval_regex_vs_llm` path (composed production match) — complementary, not the agent API |
| Notebook §10b / `inspect_regex_llm_disagreements` | FP/FN browser (shows hierarchy via fields) |

Short pointer also lives in [`README.md`](README.md) (“Regex refine”).

## Workspace layout

```text
.agent/regex_refine/<stamp>/
  manifest.json
  annotations.snapshot.jsonl      # frozen LLM gold (do not use live cache while refining)
  samples.snapshot.jsonl          # uid → report_text
  cxr_classes_pristine/            # full copy at freeze time (reference only)
  groups/<group_id>/
    owned_classes.json            # owned_class_ids (promote) + readonly_class_ids + eval_class_ids
    cxr_classes/<class_id>.txt    # owned + readonly descendant leaves
    baseline_metrics.json
    eval_baseline/                # pre-agent metrics + FP/FN
    eval/                         # agent re-evals
    REPORT.md                     # required agent deliverable
```

`.agent/` is gitignored. Production patterns live in
`medvqa/datasets/regular_expressions/cxr_classes/`.

## Environment

```bash
cd /home/pamessina/medvqa
module load conda && conda activate py313
# GEMINI_API_KEY only needed for annotation jobs, not for regex refine eval
```

Canonical run id: `v1_k1200_iu400`.  
Live annotations (Slurm may still append):  
`$MEDVQA_WORKSPACE_DIR/cache/cxr_report_labeling/v1_k1200_iu400/llm_annotations/gemini-2.5-flash-lite/sample_annotations.jsonl`

## Mid-nodes / hierarchy-aware refine

**Default snapshot behavior:** for each owned stem, ontology **descendants** are copied into the group as **read-only** eval companions (`readonly_class_ids`). True leaves are unchanged (empty readonly). Escape hatch: `--no_include_ontology_descendants` (leaf-only metrics; avoid for mid-nodes).

`owned_classes.json` shape:

```json
{
  "group_id": "support_tube_present",
  "owned_class_ids": ["support_tube_present"],
  "readonly_class_ids": ["chest_tube", "endotracheal_tube", "enteric_tube"],
  "eval_class_ids": ["support_tube_present", "chest_tube", "endotracheal_tube", "enteric_tube"],
  "class_ids": ["…same as eval_class_ids…"],
  "include_ontology_descendants": true
}
```

**Eval semantics (within `eval_class_ids`):**

- **Regex match:** parent hits if own `.txt` **or** any descendant leaf in the set hits → `matched_via_classes`.
- **Gold:** parent positive if direct LLM positive **or** any descendant in the set is positive → `gold_via_classes` + `llm_by_class`.
- Does **not** use `cxr_patterns.py` OR-composition (aliases outside the ontology tree are invisible here).

**Sub-agent rules for mid-nodes:**

- Edit **only** `owned_class_ids` `.txt` files.
- Never edit `readonly_class_ids`, ontology, prompts, production, live jsonl, or `cxr_patterns.py`.
- Optimize **composed** mid-node P/R: recover residual umbrella phrasing children miss; do **not** re-implement child lexicons unless `matched_via_classes` / FNs show parent-only gaps after children already fire.
- Before widening the parent pattern, inspect FP/FN rows: `matched_via_classes`, `gold_via_classes`, `llm_by_class` (child label/quote when gold propagated).

**Baseline / re-eval command** (pass full `eval_class_ids`):

```bash
# Read eval_class_ids from owned_classes.json, then:
python -m medvqa.scripts.cxr_report_labeling.eval_regex_subgraph \
  --classes support_tube_present endotracheal_tube enteric_tube chest_tube \
  --regex_classes_dir .agent/regex_refine/<stamp>/groups/support_tube_present/cxr_classes \
  --annotations_path .agent/regex_refine/<stamp>/annotations.snapshot.jsonl \
  --samples_path .agent/regex_refine/<stamp>/samples.snapshot.jsonl \
  --out_dir .agent/regex_refine/<stamp>/groups/support_tube_present/eval
```

Or use notebook `recompute_group_eval(group_dir)` — it uses `eval_class_ids` automatically.

**Promote:** only `owned_class_ids` stems. Never copy readonly descendant `.txt` from the group folder (those are pristine/production companions for eval).

**Ontology ≠ OR-composition:** optional coordinator production-OR sanity check is separate and out of sub-agent scope.

**Historical note:** stamp `20260809_131207` agent metrics were collected **leaf-only** (singleton groups before this default). Hierarchy-aware re-measure is recorded in that stamp’s wave table / `HIER_AWARE_REMEASURE.json` — use composed numbers for mid-node promote decisions.

## Coordinator loop (how to continue)

1. **Pick classes** with enough gold positives and room to improve.
   Prefer ontology **leaves**, or a **mid-node** (snapshot auto-includes descendants as readonly).
   Do **not** assign shared umbrellas (e.g. `lung_opacity`) while children are still being edited by other agents.
2. **Snapshot** (shared freeze + multiple groups):

```bash
python -m medvqa.scripts.cxr_report_labeling.snapshot_regex_refine \
  --run_id v1_k1200_iu400 \
  --classes <class_a> --group_id <class_a> \
  --extra_group <class_b> <class_b> \
  --extra_group <class_c> <class_c>
```

   Or add a group later: `--add_to_stamp <stamp> --classes ... --group_id ...`  
   Leaf-only escape hatch: `--no_include_ontology_descendants`.

3. **Baseline eval** per group using **`eval_class_ids`** → save `baseline_metrics.json` and seed `eval/`.
4. **Launch one sub-agent per group** using the **[Sub-agent brief template](#sub-agent-brief-template)** below (fill every placeholder; paste as the Task prompt). Do not skip the “Read first” block — that is how agents learn prompt paths and hierarchy rules.
5. **Human/coordinator review** `REPORT.md` + spot-check FP/FN jsonl.
6. **Promote** only approved stems from `owned_class_ids`:

```bash
cp .agent/regex_refine/<stamp>/groups/<id>/cxr_classes/<class_id>.txt \
   medvqa/datasets/regular_expressions/cxr_classes/<class_id>.txt
```

   Update mini-test fixtures if new phrasings should be hard positives.
   Run: `python -m medvqa.scripts.cxr_report_labeling.run_regex_mini_tests`

7. **Umbrellas / OR-composition** (`cxr_patterns.py`): coordinator-only, **after** child leaves stabilize (see lung_opacity partition notes below).

## Sub-agent brief template

**Coordinator rule:** copy this into each sub-agent Task prompt and replace
`<…>` placeholders. One agent ↔ one group; disjoint `owned_class_ids` across agents.

````markdown
You are a regex refine sub-agent in `/home/pamessina/medvqa`.

### Read first (required)
1. `medvqa/scripts/cxr_report_labeling/REGEX_REFINE.md` — layout, gold semantics, mid-node hierarchy, promote rules, pitfalls.
2. Short pointer: `medvqa/scripts/cxr_report_labeling/README.md` (“Regex refine”).
3. **Class prompt(s) — semantic source of truth (read-only):**
   `medvqa/prompts/cxr_classes/<owned_class_id>.txt`
   (one file per owned stem; do not edit prompts).
4. Env: `module load conda && conda activate py313`. Canonical run: `v1_k1200_iu400`.

### Ownership (STRICT)
- Stamp: `.agent/regex_refine/<stamp>/`
- Group: `groups/<group_id>/`
- **Edit ONLY** owned leaves listed in `owned_classes.json` → `owned_class_ids`:
  `groups/<group_id>/cxr_classes/<owned_class_id>.txt`
- **READONLY** (do not edit): `readonly_class_ids` in the same JSON (ontology descendants for eval), plus:
  production `medvqa/datasets/regular_expressions/cxr_classes/`,
  `medvqa/datasets/regular_expressions/cxr_patterns.py`,
  ontology YAML, prompts, live Slurm annotation jsonl, other groups.

### Eval set (hierarchy-aware)
- Use **all** `eval_class_ids` from `owned_classes.json` (owned ∪ readonly descendants).
- Gold and regex matches **propagate within that set** (`matched_via_classes`, `gold_via_classes`, `llm_by_class` on FP/FN rows).
- This is **not** production OR-composition. Prefer residual parent phrasing children miss; do not re-implement readonly child lexicons unless composed FNs are parent-only.

### Baseline (from coordinator)
- Parent/owned metrics: P=<…> R=<…> TP/FP/FN=<…>/<…>/<…> (n=<…>)
- (If multi-owned, list each owned class.)
- FN dump: `groups/<group_id>/eval_baseline/<class_id>_fn.jsonl`
- FP dump: `groups/<group_id>/eval_baseline/<class_id>_fp.jsonl`
- Optional browser: notebook §10b / `inspect_regex_llm_disagreements(group_dir)`

### Goal
- <recall-first | precision-first | residual mid-node polish>
- Precision floor: P ≳ <…>; recall floor: R ≳ <…>
- Prefer prompt fidelity; document bad-gold FNs rather than chasing 100% recall.

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
(Or `recompute_group_eval(group_dir)` — uses `eval_class_ids` automatically.)

### Deliverable
Write `groups/<group_id>/REPORT.md` with:
- Before/after P/R and TP/FP/FN for **each owned** class
- What changed (regex strategy)
- Leftover FNs/FPs with bad-gold attribution; cite `matched_via_classes` / `gold_via_classes` / `llm_by_class` when relevant
- Promote stance per owned stem (PROMOTE / HOLD)

Return a short summary with metrics + promote recommendation.
````

**Multi-owned closed subgraph** (e.g. `hernia` + `hiatal_hernia`): put every editable stem in `owned_class_ids` (snapshot `--classes a b --group_id …`). Brief must list **both** prompt paths and require per-class metrics; still forbid editing anything not owned.

## Gold / metrics semantics

- Positive gold: `definitely true` or `likely true` only (`llm_label_to_binary_positive`).
- `unknown` / false → negative for P/R. Many “TN” rows are `unknown`, not strong false.
- Eval joins reports via `uid` from `samples.snapshot.jsonl` (not `<report>` scraping).
- Subgraph eval uses **leaf** `.txt` files + **ontology** propagation when multiple classes are in the set (`eval_class_ids`). It does **not** use production OR-composition lists during agent tuning.
- FP/FN dumps include `matched_via_classes`, `gold_via_classes`, and `llm_by_class` so agents can see descendant-driven disagreements.

## OR-composition vs ontology (domino risk)

- **Ontology parents** (YAML): label inheritance for released semantics / eval propagation.
- **Regex OR-composition** (`cxr_patterns.py` lists): candidate-finding wiring (parent matches if child leaf matches).
- Editing a child `.txt` changes composed parent hits automatically — upward coupling.
- Never “fix parent FPs” by editing another agent’s child. Assign disjoint subgraphs; coordinator owns umbrellas last.

### Later multi-agent example: `lung_opacity`

Phase 1 parallel agents own disjoint branches (e.g. atelectasis+children; nodule+calcified child; singleton leaves…).  
Phase 2 coordinator tunes `lung_opacity.txt` + OR edges after the barrier.

## Status as of 2026-08-08 / 2026-08-09 (for the next agent)

### Annotation job

- Slurm loop: `medvqa/slurm/launch_cxr_annotate_batch_loop.sh` / job family `cxr_annotate_loop`
- Log pattern: `/mnt/workspace/$USER/slurm-out/cxr_annotate_loop-<jobid>.out`
- Progress: `cache/.../api_tmp/gemini/batch_api/loop_progress.jsonl`
- Model: `gemini-2.5-flash-lite`; key env: `GEMINI_API_KEY`; chunk_size 2000
- Working agreement: agents author code; user runs Slurm/API
- Live annotations ≈**192k** lines (still growing — chunk 102 in flight as of 2026-08-09 ~09:34 local; stamp `20260809_014008` froze ≈132k)

### Regex refine history

| Stamp | Groups | Outcome |
|-------|--------|---------|
| `20260808_202258` | `breast_implant` | Promoted to production (R→1.0, P=1.0) |
| `20260808_204257` | `focal_pulmonary_scarring`, `hilar_enlargement`, `costophrenic_angle_blunting` | scarring + hilar **promoted**; CPA **held** |
| `20260808_231118` | `costophrenic_angle_blunting` | Round 2 **promoted** to production (2026-08-08) |
| `20260808_232545` | 5 leaves (see below) | **All five promoted** (2026-08-08) |
| `20260808_234539` | 5 leaves (see below) | **All five promoted** (2026-08-08) |
| `20260809_000419` | 5 leaves (see below) | **All five promoted** (2026-08-09) |
| `20260809_002524` | 5 leaves (see below) | **All five promoted** (2026-08-09) |
| `20260809_004155` | 5 leaves (see below) | **All five promoted** (2026-08-09) |
| `20260809_010512` | 5 leaves (see below) | **All five promoted** (2026-08-09; pneumonia via pass-2 stamp) |
| `20260809_014008` | `pneumonia` pass-2 | **Promoted** to production (2026-08-09) |
| `20260809_094147` | 5 leaves (see below) | **All five promoted** (2026-08-09) |
| `20260809_113744` | 5 leaves (see below) | **All five promoted** (2026-08-09) |
| `20260809_120812` | 5 leaves (see below) | **All five promoted** (2026-08-09) |
| `20260809_123113` | 5 leaves (see below) | **3 promoted**; hydro + aortic held for recall pass-2 |
| `20260809_125536` | `hydropneumothorax`, `aortic_calcification` recall pass-2 | **Both promoted** (2026-08-09) |
| `20260809_131207` | 5 mid/leaves (see below) | **All five promoted** (2026-08-09); mid-nodes re-opened on next stamp |
| `20260809_140649` | 4 mid-nodes hierarchy-aware pass-2 | **All four promoted** (2026-08-09) |
| `20260809_150542` | precision mid-nodes (+ hernia/hiatal co-owned) | **All five promoted** (2026-08-09) |
| `20260809_153116` | light leaves + atelectasis FP trim | **All three promoted** (2026-08-09) |
| `20260809_155546` | TB / osteopenia / calcification / CVC leaf | **All four promoted** (2026-08-09) |
| `20260809_165840` | support_line / LLV / PVC / device_abnormality | **All four promoted** (2026-08-09) |
| `20260809_232706` | `support_device_present` hierarchy-aware | **Promoted** (2026-08-09) |
| `20260809_235220` | `lung_opacity` Phase A hierarchy-aware | **Promoted** (2026-08-10) |
| `20260810_003937` | `lung_opacity` Phase B1 helper merge | **Promoted** (2026-08-10); helpers removed from OR |
| `20260810_101035` | `central_venous_catheter` Phase B2 helper merge | **Promoted** (2026-08-10); CVC site/reservoir helpers removed from OR |
| `20260810_102230` | `bone_fracture` Phase B3 helper merge | **Promoted** (2026-08-10); callus_rib + humeral helpers removed from OR |
| `20260810_200934` | `support_device_present` Phase B4 helper merge | **Promoted** (2026-08-10); metal/prosthesis/humeral_prosthesis helpers removed from OR |
| `20260810_202229` | `support_tube_present` Phase B5 helper merge | **Promoted** (2026-08-10); tracheostomy_tube helper removed from OR |
| `20260810_202928` | `pleural_thickening` Phase B6 helper merge | **Promoted** (2026-08-10); apical_cap / pleural_scarring helpers removed from OR |
| `20260810_203427` | `pleural_effusion` + loculated Phase B7 helper merge | **Promoted** (2026-08-10); empyema / pseudotumor helpers removed from OR |
| `20260810_204445` | `pulmonary_nodule_or_mass` Phase B8 helper merge | **Promoted** (2026-08-10); pseudonodule / thoracic_masslike helpers removed from OR |
| `20260810_225816` | CPA / support_tube / device_abnormality / support_line recall wave | **All four promoted** (2026-08-10) |
| `20260810_232835` | lobar / graft / osseous / pneumonia recall wave #2 | **All four promoted** (2026-08-10) |
| `20260811_000435` | consolidation / FB / mediastinal mass / cyst / pleural nodule recall wave #3 | **All five promoted** (2026-08-11) |
| `20260811_002344` | scarring / ILD / mediastinal shift / fibrotic / hilar recall wave #4 | **All five promoted** (2026-08-11) |
| `20260811_011334` | fracture / DJD / rounded atelectasis / widening / EPA recall wave #5 | **All five promoted** (2026-08-11) |
| `20260811_013434` | pacer / reticular / tracheal deviation / loculated / sternotomy recall wave #6 | **All five promoted** (2026-08-11) |
| `20260811_014802` | pleural thickening / VCF / chest tube / aortic ectasia / prosthetic valve recall wave #7 | **All five promoted** (2026-08-11) |
| `20260811_020736` | CVC / LLV / BWT / elevated hemidiaphragm / rib fracture recall wave #8 | **All five promoted** (2026-08-11) |
| `20260811_023520` | cardiac silhouette / enteric / nodal / ETT / cavity recall wave #9 | **All five promoted** (2026-08-11) |
| `20260811_025217` | Kerley / GGO / calcified granuloma / flattened diaphragm / cardiac calc recall wave #10 | **All five promoted** (2026-08-11) |
| `20260811_030248` | support-device / ECS / effusion mid-nodes + emphysema / pneumomediastinum wave #11 | **All five promoted** (2026-08-11) |
| `20260811_042446` | hernia / breast implant / scoliosis / plaque / calcification wave #12 | **All five promoted** (2026-08-11) |
| `20260811_044912` | P-trim edema / nodule / ECS / lobar / hydro | **4 promoted**; ECS **held** (2026-08-11) |
| `20260811_051641` | P-trim consolidation / pneumonia / enteric / bone_fracture | **All four promoted** (2026-08-11) |

**Wave `20260808_204257` (≈81k annotation lines at freeze):**

| Class | Baseline P/R | After P/R | Promote? |
|-------|--------------|-----------|----------|
| `focal_pulmonary_scarring` | 0.75 / 0.11 | 1.00 / 0.93 | **Promoted** to production (2026-08-08) |
| `hilar_enlargement` | 1.00 / 0.19 | 0.89 / 0.88 | **Promoted** to production (2026-08-08) |
| `costophrenic_angle_blunting` | 0.96 / 0.17 | 0.72 / 0.40 | **Hold** — see round 2 |

### Costophrenic round 2 — stamp `20260808_231118` (≈106k lines, n=1348)

| Stage | P / R | TP / FP / FN |
|-------|-------|--------------|
| Production baseline | 0.971 / 0.169 | 34 / 1 / 167 |
| Seeded prior (+ proxies) | 0.690 / 0.398 | 80 / 36 / 121 |
| **Round-2 revision** | **0.976 / 0.199** | **40 / 1 / 161** |

- Dropped section-6 effusion proxies; hardened negation + clause-local gaps; kept sulcus/recess fluid.
- Leftover FNs ≈ effusion-only bad gold (prompt forbids that inference).
- **Promoted** workdir leaf → `medvqa/datasets/regular_expressions/cxr_classes/costophrenic_angle_blunting.txt` (2026-08-08). Mini-tests updated in `cxr_class_mini_tests/pleura.json`.

FP/FN browser: notebook §10b / `inspect_regex_llm_disagreements(group_dir)`.

Report: `.agent/regex_refine/20260808_231118/groups/costophrenic_angle_blunting/REPORT.md`  
GPT feedback: `.../FEEDBACK_gpt56.md`

### Parallel wave — stamp `20260808_232545` (≈108k annotation lines)

| Class | Baseline P / R | TP / FP / FN | Status |
|-------|----------------|--------------|--------|
| `aortic_tortuosity_or_elongation` | 1.00 / 0.34 → **0.99 / 0.99** | 28/0/54 → **81/1/1** | **promoted** |
| `vascular_graft_or_stent` | 0.82 / 0.15 → **0.87 / 0.21** | 9/2/53 → **13/2/49** | **promoted** (modest; leftover FN mostly bad gold) |
| `loculated_or_fissural_pleural_effusion` | 0.86 / 0.37 → **0.95 / 0.71** | 18/3/31 → **35/2/14** | **promoted** |
| `focal_osseous_lesion` | 0.63 / 0.12 → **1.00 / 0.40** | 5/3/38 → **17/0/26** | **promoted** |
| `enlarged_pulmonary_artery` | 0.88 / 0.23 → **1.00 / 0.57** | 7/1/23 → **17/0/13** | **promoted** |

All five **promoted** to `medvqa/datasets/regular_expressions/cxr_classes/` (2026-08-08). Mini-test fixtures updated in `cardiomediastinal.json`, `musculoskeletal.json`, `pleura.json`, `support_device.json`.

### Parallel wave — stamp `20260808_234539` (≈112k annotation lines)

| Class | Baseline P / R | TP / FP / FN | Status |
|-------|----------------|--------------|--------|
| `nontherapeutic_foreign_body` | 0.75 / 0.15 → **1.00 / 0.44** | 6/2/35 → **18/0/23** | **promoted** |
| `pleural_nodule_or_mass` | 0.25 / 0.18 → **1.00 / 0.18** | 4/12/18 → **4/0/18** | **promoted** (precision cleanup) |
| `interstitial_lung_disease` | 0.50 / 0.21 → **1.00 / 0.31** | 6/6/23 → **9/0/20** | **promoted** |
| `aortic_enlargement_or_ectasia` | 0.64 / 0.28 → **1.00 / 0.84** | 7/4/18 → **21/0/4** | **promoted** |
| `pulmonary_cyst_or_bulla` | 0.88 / 0.29 → **0.92 / 0.46** | 7/1/17 → **11/1/13** | **promoted** |

All five **promoted** to production (2026-08-08). Mini-tests updated (`diagnostic_label`, `pleura`, `other`, `focal_lesion_findings`, `cardiomediastinal`).

### Parallel wave — stamp `20260809_000419` (≈114k annotation lines)

| Class | Baseline P / R | TP / FP / FN | Status |
|-------|----------------|--------------|--------|
| `mediastinal_mass` | 0.33 / 0.20 → **1.00 / 0.55** | 4/8/16 → **11/0/9** | **promoted** |
| `rounded_atelectasis` | 1.00 / 0.32 → **1.00 / 0.32** | 6/0/13 (flat) | **promoted** (semantic harden; FNs bad gold) |
| `fibrotic_lung_change` | 0.67 / 0.48 → **1.00 / 0.68** | 12/6/13 → **17/0/8** | **promoted** |
| `bronchial_wall_thickening` | 1.00 / 0.52 → **1.00 / 0.76** | 15/0/14 → **22/0/7** | **promoted** |
| `tracheal_deviation` | 0.93 / 0.54 → **1.00 / 0.73** | 14/1/12 → **19/0/7** | **promoted** |

All five **promoted** (2026-08-09). Mini-tests updated (`cardiomediastinal`, `airway`, `pulmonary_opacity`).

### Parallel wave — stamp `20260809_002524` (≈117k annotation lines)

| Class | Baseline P / R | TP / FP / FN | Status |
|-------|----------------|--------------|--------|
| `mediastinal_widening` | 0.337 / 0.615 → **0.923 / 0.692** | 32/63/20 → **36/3/16** | **promoted** |
| `mediastinal_shift` | 0.525 / 0.620 → **1.000 / 0.620** | 31/28/19 → **31/0/19** | **promoted** |
| `vertebral_compression_fracture` | 0.591 / 0.565 → **0.944 / 0.739** | 13/9/10 → **17/1/6** | **promoted** |
| `pacemaker_or_icd` | 0.920 / 0.585 → **1.000 / 0.831** | 69/6/49 → **98/0/20** | **promoted** |
| `endotracheal_tube` | 0.882 / 0.612 → **0.989 / 0.918** | 120/16/76 → **180/2/16** | **promoted** |

All five **promoted** to production (2026-08-09). Mini-tests updated (`cardiomediastinal`, `bone_fracture_findings`, `support_device`).

### Parallel wave — stamp `20260809_004155` (≈121k annotation lines)

| Class | Baseline P / R | TP / FP / FN | Status |
|-------|----------------|--------------|--------|
| `enlarged_cardiac_silhouette` | 0.963 / 0.589 → **0.985 / 0.962** | 235/9/164 → **384/6/15** | **promoted** |
| `prosthetic_cardiac_valve` | 0.818 / 0.621 → **1.000 / 0.828** | 18/4/11 → **24/0/5** | **promoted** |
| `degenerative_osseous_change` | 0.949 / 0.659 → **0.969 / 0.729** | 56/3/29 → **62/2/23** | **promoted** |
| `lobar_atelectasis` | 0.654 / 0.662 → **0.905 / 0.467** | 149/79/76 → **105/11/120** | **promoted** |
| `reticular_interstitial_opacity_pattern` | 0.867 / 0.634 → **1.000 / 0.659** | 26/4/15 → **27/0/14** | **promoted** |

All five **promoted** to production (2026-08-09). Mini-tests updated (`cardiomediastinal`, `support_device`, `musculoskeletal`, `pulmonary_opacity`).

### Parallel wave — stamp `20260809_010512` (≈123k annotation lines)

| Class | Baseline P / R | TP / FP / FN | Status |
|-------|----------------|--------------|--------|
| `pneumonia` | 0.462 / 0.615 → **0.894 / 0.652** (pass-1); pass-2 on `014008` **0.889 / 0.732** | see pass-2 table | **promoted** (from `014008`) |
| `consolidation` | 0.441 / 0.661 → **0.911 / 0.653** | 238/302/122 → **235/23/125** | **promoted** |
| `elevated_or_eventrated_hemidiaphragm` | 0.962 / 0.758 → **1.000 / 0.879** | 50/2/16 → **58/0/8** | **promoted** |
| `hyperinflation` | 0.949 / 0.767 → **0.986 / 0.986** | 56/3/17 → **72/1/1** | **promoted** |
| `enteric_tube` | 0.869 / 0.821 → **0.974 / 0.944** | 192/29/42 → **221/6/13** | **promoted** |

All five **promoted** (2026-08-09). Pneumonia from pass-2 stamp `20260809_014008`; others from `20260809_010512`. Mini-tests updated (`diagnostic_label`, `pulmonary_opacity`, `diaphragm`, `lung_volume`, `support_device`).

### Pneumonia pass-2 — stamp `20260809_014008` (≈132k annotation lines)

| Stage | P / R | TP / FP / FN | Notes |
|-------|-------|--------------|-------|
| Production on new gold | 0.468 / 0.597 | 221/251/149 | old leaf |
| Seeded pass-1 on new gold | 0.889 / 0.649 | 240/30/130 | starting point |
| **Pass-2** | **0.889 / 0.732** | **271/34/99** | **promoted** (16/18 attribution FNs recovered) |

Feedback: `groups/pneumonia/FEEDBACK_pass2.md`.

### Parallel wave — stamp `20260809_094147` (≈194k annotation lines)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `kerley_lines` | 0.933 / 0.636 | **1.000 / 0.773** | 14/1/8 → **17/0/5** | **promoted** |
| `calcified_pulmonary_nodule_or_granuloma` | 1.000 / 0.714 | **1.000 / 0.943** | 25/0/10 → **33/0/2** | **promoted** |
| `ground_glass_opacity_pattern` | 0.800 / 0.696 | **1.000 / 0.913** | 16/4/7 → **21/0/2** | **promoted** |
| `clavicle_fracture` | 0.833 / 0.758 | **1.000 / 0.970** | 25/5/8 → **32/0/1** | **promoted** |
| `cardiac_or_valvular_calcification` | 0.824 / 0.700 | **1.000 / 0.950** | 14/3/6 → **19/0/1** | **promoted** |

All five **promoted** to production (2026-08-09). Mini-tests updated (`pulmonary_opacity`, `focal_lesion_findings`, `bone_fracture_findings`, `cardiomediastinal`); `run_regex_mini_tests` hard failures = 0.

### Parallel wave — stamp `20260809_113744` (≈206k annotation lines)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `pleural_plaque` | 0.950 / 0.760 | **0.955 / 0.840** | 19/1/6 → **21/1/4** | **promoted** |
| `sternotomy_wires` | 0.977 / 0.808 | **1.000 / 0.872** | 126/3/30 → **136/0/20** | **promoted** |
| `hilar_or_mediastinal_nodal_enlargement` | 0.355 / 0.812 | **0.952 / 0.833** | 39/71/9 → **40/2/8** | **promoted** |
| `flattened_diaphragm` | 0.870 / 0.833 | **1.000 / 0.917** | 20/3/4 → **22/0/2** | **promoted** |
| `chest_tube` | 0.713 / 0.878 | **0.964 / 0.855** | 273/110/38 → **266/10/45** | **promoted** |

All five **promoted** to production (2026-08-09). Mini-tests updated (`pleura`, `support_device`, `diaphragm`, `cardiomediastinal`); post-promote fix on `pleural_plaque` blocking `… without plaque`; `run_regex_mini_tests` hard failures = 0.

### Parallel wave — stamp `20260809_120812` (≈211k annotation lines)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `rib_fracture` | 0.625 / 0.878 | **0.892 / 0.892** | 130/78/18 → **132/16/16** | **promoted** |
| `pulmonary_edema` | 0.581 / 0.989 | **0.880 / 0.952** | 540/389/6 → **520/71/26** | **promoted** |
| `pneumomediastinum` | 0.646 / 0.886 | **1.000 / 0.886** | 31/17/4 → **31/0/4** | **promoted** |
| `pulmonary_cavity` | 0.434 / 0.956 | **0.976 / 0.889** | 43/56/2 → **40/1/5** | **promoted** |
| `emphysema` | 0.192 / 0.929 | **0.927 / 0.905** | 39/164/3 → **38/3/4** | **promoted** |

All five **promoted** to production (2026-08-09). Mini-tests updated (`bone_fracture_findings`, `pulmonary_opacity`, `cardiomediastinal`, `focal_lesion_findings`, `pulmonary_parenchyma`); `run_regex_mini_tests` hard failures = 0.

### Parallel wave — stamp `20260809_123113` (≈211k annotation lines)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `hydropneumothorax` | 0.091 / 0.930 | **0.811 / 0.860** | 93/926/7 → **86/20/14** | **held** — recall pass-2 |
| `pneumoperitoneum` | 0.491 / 0.966 | **0.967 / 1.000** | 28/29/1 → **29/1/0** | **promoted** |
| `kyphosis` | 0.409 / 0.900 | **0.818 / 0.900** | 18/26/2 → **18/4/2** | **promoted** (+ `hyperkyphosis` word-boundary fix) |
| `aortic_calcification` | 0.847 / 0.963 | **0.990 / 0.945** | 105/19/4 → **103/1/6** | **held** — recall pass-2 |
| `bronchiectasis` | 0.810 / 0.919 | **1.000 / 0.946** | 34/8/3 → **35/0/2** | **promoted** |

**Promoted** (2026-08-09): `pneumoperitoneum`, `kyphosis`, `bronchiectasis`. Mini-tests updated (`upper_abdomen`, `musculoskeletal`, `airway`); hard failures = 0.  
**Held then** for recall pass-2: `hydropneumothorax`, `aortic_calcification` → stamp `20260809_125536` (promoted there).

### Recall pass-2 — stamp `20260809_125536` (≈213k annotation lines; seeded pass-1 leaves)

| Class | Pass-1 seed P / R | Pass-2 P / R | TP / FP / FN | Status |
|-------|-------------------|--------------|--------------|--------|
| `hydropneumothorax` | 0.813 / 0.861 | **0.836 / 0.960** | 87/20/14 → **97/19/4** | **promoted** |
| `aortic_calcification` | 0.991 / 0.938 | **0.991 / 0.946** | 105/1/7 → **106/1/6** | **promoted** |

- Hydro: recovered 10 same-side / bilateral-effusion+unilateral-PTX FNs; OR-list catastrophe stayed closed (0 OR-list FPs); leftover 4 FNs are OR-uncertainty / bad gold / indication-only.
- Aortic: added `calcinos\w*` (“Mild aortic calcinosis”); leftover 6 FNs bad/out-of-scope gold; single FP still history-section leakage.
- **Promoted** (2026-08-09) to `medvqa/datasets/regular_expressions/cxr_classes/`. Mini-tests updated (`pleura`, `cardiomediastinal`); `run_regex_mini_tests` hard failures = 0.

### Parallel wave — stamp `20260809_131207` (≈217k annotation lines)

**Leaf-only** agent metrics (singleton `--classes`; pre-hierarchy default — do not use alone for mid-node promote):

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `pleural_thickening` | 0.966 / 0.464 | **0.920 / 0.823** | 84/3/97 → **149/13/32** | **promoted** |
| `pulmonary_nodule_or_mass` | 0.751 / 0.606 | **0.780 / 0.795** | 154/51/100 → **202/57/52** | **promoted** |
| `support_tube_present` | 0.934 / 0.547 | **0.983 / 0.576** | 747/53/618 → **786/14/579** | **promoted** |
| `enlarged_cardiomediastinal_silhouette` | 0.688 / 0.169 | **0.965 / 0.895** | 110/50/540 → **582/21/68** | **promoted** |
| `scoliosis` | 0.894 / 0.875 | **1.000 / 0.958** | 42/5/6 → **46/0/2** | **promoted** |

**Hierarchy-aware re-measure** (owned leaf ∨ ontology descendants; gold propagated; dumps under `eval_hier*` / `HIER_AWARE_REMEASURE.json`):

| Class | Hier baseline P / R | Hier after P / R | TP / FP / FN | Notes |
|-------|---------------------|------------------|--------------|-------|
| `pleural_thickening` | 0.966 / 0.621 | **0.920 / 0.824** | 113/4/69 → **150/13/32** | Clear residual R gain |
| `pulmonary_nodule_or_mass` | 0.766 / 0.618 | **0.789 / 0.795** | 160/49/99 → **206/55/53** | Clear residual R gain |
| `support_tube_present` | 0.942 / 0.585 | **0.980 / 0.577** | 803/49/569 → **791/16/581** | P↑; R flat (leaf-only R was mostly child overlap / gold noise) |
| `enlarged_cardiomediastinal_silhouette` | 0.923 / 0.910 | **0.983 / 0.893** | 638/53/63 → **626/11/75** | Children already carried R; after = P cleanup (leaf-only R jump was child duplication) |
| `scoliosis` | 0.894 / 0.875 | **1.000 / 0.958** | 42/5/6 → **46/0/2** | Leaf (no descendants) |

Groups now have `owned_classes.json` with `readonly_class_ids` seeded; workdirs: `.agent/regex_refine/20260809_131207/groups/<class_id>/`.

**Promoted** (2026-08-09) all five owned stems to production. Mini-tests updated (`pleura`, `focal_lesion_findings`, `support_device`, `cardiomediastinal`, `musculoskeletal`); hard failures = 0.

### Hierarchy-aware mid-node pass-2 — stamp `20260809_140649` (≈224k annotation lines)

Composed eval (`owned` ∨ ontology descendants). Seeded from promoted production leaves.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `pleural_thickening` | 0.922 / 0.811 | **0.914 / 0.837** | 154/13/36 → **159/15/31** | **promoted** |
| `pulmonary_nodule_or_mass` | 0.789 / 0.789 | **0.827 / 0.937** | 213/57/57 → **253/53/17** | **promoted** |
| `support_tube_present` | 0.979 / 0.576 | **0.983 / 0.582** | 809/17/595 → **817/14/587** | **promoted** |
| `enlarged_cardiomediastinal_silhouette` | 0.983 / 0.891 | **0.983 / 0.901** | 640/11/78 → **647/11/71** | **promoted** |

**Promoted** (2026-08-09) owned stems only to `medvqa/datasets/regular_expressions/cxr_classes/`. Mini-tests updated (`pleura`, `focal_lesion_findings`, `support_device`, `cardiomediastinal`); hard failures = 0.

### Precision wave — stamp `20260809_150542` (≈230k annotation lines; hierarchy-aware)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `pneumothorax` | 0.292 / 0.993 | **0.940 / 0.986** | 286/695/2 → **284/18/4** | **promoted** |
| `pleural_effusion` | 0.530 / 0.972 | **0.987 / 0.945** | 1015/899/29 → **987/13/57** | **promoted** |
| `bone_fracture` | 0.602 / 0.863 | **0.928 / 0.759** | 265/175/42 → **233/18/74** | **promoted** |
| `hernia` | 0.650 / 0.897 | **1.000 / 0.879** | 52/28/6 → **51/0/7** | **promoted** (co-owned) |
| `hiatal_hernia` | 0.846 / 0.917 | **1.000 / 0.938** | 44/8/4 → **45/0/3** | **promoted** (co-owned) |

All five **promoted** to `medvqa/datasets/regular_expressions/cxr_classes/` (2026-08-09). Mini-tests updated (`pleura`, `bone_fracture_findings`, `other`); `run_regex_mini_tests` hard failures = 0.

Workdirs: `.agent/regex_refine/20260809_150542/groups/{pneumothorax,pleural_effusion,bone_fracture,hernia}/`.  
`hernia` group owned both `hernia` + `hiatal_hernia`. Others kept children readonly.

### Light leaves + atelectasis FP trim — stamp `20260809_153116` (≈231k annotation lines; hierarchy-aware)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `subcutaneous_emphysema` | 0.908 / 0.931 | **1.000 / 0.991** | 108/11/8 → **115/0/1** | **promoted** |
| `pleural_calcification` | 0.950 / 0.905 | **1.000 / 0.952** | 38/2/4 → **40/0/2** | **promoted** |
| `atelectasis` (composed) | 0.905 / 0.963 | **0.989 / 0.945** | 952/100/37 → **935/10/54** | **promoted** |

- Leaves: SQ emphysema negation/site polish; pleural calcific fibrothorax + typo + negation (leftover FNs = bare plaques bad gold).
- Mid-node `atelectasis`: precision-first; dropped bare volume/linear proxies; readonly `lobar_atelectasis` / `rounded_atelectasis` untouched. Leftover FNs mostly PTX/effusion inference or child-only gold.
- Coordinator re-verified all three via `eval_regex_subgraph` (`eval_verify/`).
- **Promoted** (2026-08-09) owned stems only to `medvqa/datasets/regular_expressions/cxr_classes/`. Mini-tests updated (`soft_tissue`, `pleura`, `pulmonary_opacity`); `run_regex_mini_tests` hard failures = 0.

Workdirs: `.agent/regex_refine/20260809_153116/groups/{subcutaneous_emphysema,pleural_calcification,atelectasis}/`.

### TB / osteopenia / calcification / CVC — stamp `20260809_155546` (≈233k annotation lines)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `tuberculosis` | 0.302 / 0.812 | **0.857 / 0.750** | 13/30/3 → **12/2/4** | **promoted** (small n+; post-promote drop of overbroad `(?<!\bmiliary\s)`) |
| `radiographic_osteopenia` | 0.917 / 0.985 | **1.000 / 0.985** | 66/6/1 → **66/0/1** | **promoted** |
| `calcification` (composed) | 0.942 / 0.915 | **0.993 / 0.937** | 292/18/27 → **299/2/20** | **promoted** |
| `central_venous_catheter` (leaf) | 0.898 / 0.545 | **1.000 / 0.866** | 469/53/391 → **745/0/115** | **promoted** |

- TB: precision-first; Ghon/Ranke; history/r/o guards; tiny gold set. Post-promote fix: removed `(?<!\bmiliary\s)` so prompt-positive `miliary tuberculosis` matches again.
- Osteopenia: negation/history + demineralization tighten; leftover FN gracile-only bad gold.
- Calcification mid-node: residual parent phrasing; readonly children untouched.
- CVC: **ontology leaf** (not OR-composition); IJ/port/PICC/dialysis lexicon; Swan-Ganz dropped. Site/reservoir OR aliases still coordinator-owned later.
- Coordinator re-verified all four via `eval_verify/`.
- **Promoted** (2026-08-09) owned stems only to `medvqa/datasets/regular_expressions/cxr_classes/`. Mini-tests updated (`diagnostic_label`, `musculoskeletal`, `calcific_findings`, `support_device`); `run_regex_mini_tests` hard failures = 0.

Workdirs: `.agent/regex_refine/20260809_155546/groups/{tuberculosis,radiographic_osteopenia,calcification,central_venous_catheter}/`.

### Support-line / LLV / PVC / device_abnormality — stamp `20260809_165840` (≈236k annotation lines)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `support_line_present` (composed) | 0.836 / 0.765 | **0.998 / 0.662** | 1013/199/311 → **877/2/447** | **promoted** (R floor unmet — mostly bad gold) |
| `low_lung_volume` | 1.000 / 0.549 | **0.998 / 0.894** | 290/0/238 → **472/1/56** | **promoted** |
| `pulmonary_vascular_congestion` | 0.606 / 0.430 | **0.984 / 0.673** | 117/76/155 → **183/3/89** | **promoted** |
| `device_abnormality` | 0.241 / 0.215 | **0.922 / 0.319** | 112/352/408 → **166/14/354** | **promoted** (R≢0.40 without prompt violation) |

- `support_line_present`: dropped bare line/catheter FPs; Swan-Ganz/PA + residual support-line phrasing; CVC readonly untouched. Leftover FNs mostly pacers/drains/ETT/enteric (prompt exclusions).
- LLV: subject-first + synonym expand; laterality/`small` guards kept P≈1.
- PVC: engorgement/redistribution/venous-pressure lexicon + negation; leftover FNs mostly edema-only bad gold.
- `device_abnormality`: precision-first; device-scoped abnormality only (no bare fracture/coil).
- Coordinator re-verified all four via `eval_verify/`.
- **Promoted** (2026-08-09) owned stems only to `medvqa/datasets/regular_expressions/cxr_classes/`. Mini-tests updated (`support_device`, `lung_volume`, `cardiomediastinal`); `run_regex_mini_tests` hard failures = 0.

Workdirs: `.agent/regex_refine/20260809_165840/groups/{support_line_present,low_lung_volume,pulmonary_vascular_congestion,device_abnormality}/`.

### `support_device_present` hierarchy-aware — stamp `20260809_232706` (≈270k annotation lines)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `support_device_present` (composed) | 0.992 / 0.849 | **0.995 / 0.905** | 1716/13/305 → **1828/9/193** | **promoted** |

- Residual parent polish only (clips/staples, sternotomy, ortho fixation, joint prostheses, stimulators, CABG, trach/ECMO/etc.); 12 ontology descendants readonly.
- Production OR-helpers (`metal`, `prosthesis`, `tracheostomy_tube`, …) out of subgraph eval — recovered as parent residual where needed.
- Leftover FNs ≈ half bad gold / prompt exclusions, half child-gap gold (`support_line`/`support_tube`/graft/`device_abnormality`).
- Coordinator re-verified via `eval_verify/`; readonly companions match production.
- **Promoted** (2026-08-09) owned stem only to `medvqa/datasets/regular_expressions/cxr_classes/`. Mini-tests updated (`support_device`); `run_regex_mini_tests` hard failures = 0.

Workdirs: `.agent/regex_refine/20260809_232706/groups/support_device_present/`.

### `lung_opacity` Phase A — stamp `20260809_235220` (≈272k annotation lines; hierarchy-aware)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `lung_opacity` (composed) | 0.931 / 0.951 | **0.963 / 0.967** | 2239/165/116 → **2277/87/78** | **promoted** |

- Precision-first residual parent polish (negation/`opac*`/`infiltrat*`/airspace/shadowing/interstitial); 13 ontology descendants readonly.
- Production OR-helpers out of scope (Phase B later: merge helpers into ontology leaves then delete).
- Leftover FPs mostly readonly children; leftover FNs mostly prompt exclusions / child gaps.
- Coordinator re-verified via `eval_verify/`.
- **Promoted** (2026-08-10) owned stem only to `medvqa/datasets/regular_expressions/cxr_classes/`. Mini-tests updated (`pulmonary_opacity`); `run_regex_mini_tests` hard failures = 0. Annotation snapshots freed post-promote for home quota.

Workdirs: `.agent/regex_refine/20260809_235220/groups/lung_opacity/` (owned `.txt` + REPORT retained).

### `lung_opacity` Phase B1 — stamp `20260810_003937` (helper merge)

| Class | Seed (Phase A) P / R | After merge P / R | TP / FP / FN | Status |
|-------|----------------------|-------------------|--------------|--------|
| `lung_opacity` (composed) | 0.962 / 0.967 | **0.962 / 0.967** | **2306 / 90 / 78** | **promoted** |

- Merged framed **air bronchogram**, framed **miliary** opacity/pattern/TB, framed **lung/opaque-wall abscess** into `lung_opacity.txt`.
- Discarded bare infiltrate / bare alveolar / bare abscess / bare miliary + micronodule lexicon (covered or FP-prone).
- Coordinator: removed OR helpers `infiltration`, `alveolar_pattern`, `air_bronchogram`, `abscess`, `miliary_opacities` from `cxr_patterns.py`; deleted their `.txt` files.
- Mini-tests: `pulmonary_opacity` positives for air bronchogram / miliary / framed abscess; hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_003937/groups/lung_opacity/` (`REPORT_B1.md` retained).

### `central_venous_catheter` Phase B2 — stamp `20260810_101035` (helper merge)

| Class | Baseline leaf P / R | After merge P / R | TP / FP / FN | Status |
|-------|---------------------|-------------------|--------------|--------|
| `central_venous_catheter` (leaf) | 0.997 / 0.859 | **0.997 / 0.868** | **864 / 3 / 131** | **promoted** |

- Merged framed IJ presence / via-jugular residual and framed port lexicon (`portacatheter`, `port-a-venous`, `power port`, chemo port, framed reservoir) into `central_venous_catheter.txt`.
- Discarded bare `ij|jugular`, bare `subclavian`, bare `port`/`reservoir` (OR FP spike: +69 FP for +36 TP).
- Coordinator: removed OR helpers `central_venous_catheter_via_jugular_vein`, `central_venous_catheter_via_subclavian_vein`, `reservoir_central_venous_catheter` from `cxr_patterns.py`; deleted their `.txt` files. CVC is now a single `_load_pattern`.
- Mini-tests hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_101035/groups/central_venous_catheter/` (`REPORT.md` retained).

### `bone_fracture` Phase B3 — stamp `20260810_102230` (helper merge)

| Class | Baseline composed P / R | After merge P / R | TP / FP / FN | Status |
|-------|-------------------------|-------------------|--------------|--------|
| `bone_fracture` (hierarchy-aware) | 0.928 / 0.736 | **0.929 / 0.741** | **312 / 24 / 109** | **promoted** |

- Merged framed humeral fracture residual (`head and neck`, metadiaphysis connectors) into `bone_fracture.txt`.
- Discarded `callus_rib_fracture` (0 unique residual; rib/parent callus covers) and bare humeral `deform*`/`dislocat*` (prompt-unfaithful / OR FP).
- Coordinator: removed OR helpers `callus_rib_fracture`, `humeral_fracture` from `cxr_patterns.py`; deleted their `.txt` files. Kept ontology children (`clavicle_fracture`, `rib_fracture`, `vertebral_compression_fracture`) in the OR list.
- Mini-tests hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_102230/groups/bone_fracture/` (`REPORT.md` retained).

### `support_device_present` Phase B4 — stamp `20260810_200934` (helper merge)

| Class | Baseline composed P / R | After merge P / R | TP / FP / FN | Status |
|-------|-------------------------|-------------------|--------------|--------|
| `support_device_present` (hierarchy-aware) | 0.995 / 0.888 | **0.996 / 0.893** | **1996 / 9 / 240** | **promoted** |

- Merged framed ortho residual (`internal fixation`, `fixation bars` / NUSS, `spinal rods`, `bone anchors`, anterior cervical fusion) into `support_device_present.txt`; small residual polish (`verterbroplast*`, epicardial/pacing leads, `consistent with CABG`, IU `surgical XXXX`).
- Discarded bare `metal` / `screw` / `plate` / `fixation`, bare `prosthe*` / `implant` / `graft`, and `humeral_prosthesis` helper (0 unique residual; joint/limb arm already covers).
- Coordinator: removed OR helpers `metal`, `prosthesis`, `humeral_prosthesis` from `cxr_patterns.py`; deleted their `.txt` files.
- Mini-tests: `support_device` positives for internal fixation / bone anchor / NUSS; hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_200934/groups/support_device_present/` (`REPORT.md` retained).

### `support_tube_present` Phase B5 — stamp `20260810_202229` (helper merge)

| Class | Baseline composed P / R | After merge P / R | TP / FP / FN | Status |
|-------|-------------------------|-------------------|--------------|--------|
| `support_tube_present` (hierarchy-aware) | 0.983 / 0.579 | **0.983 / 0.580** | **1022 / 18 / 740** | **promoted** |

- Merged framed trach residual (`tracheostomy is present` / `projects`, `has a tracheostomy` with defect/scar guards) into `support_tube_present.txt`.
- Discarded bare `tracheostomy_tube` helper (`trach(eostomy) + optional tube/cannula`) — parent already covers tube/cannula forms; bare form FP-prone on defect / not-visualized.
- Coordinator: removed OR helper `tracheostomy_tube` from `cxr_patterns.py`; deleted its `.txt` file.
- Mini-tests: `support_tube_present` positives for projects / has a tracheostomy; hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_202229/groups/support_tube_present/` (`REPORT.md` retained).

### `pleural_thickening` Phase B6 — stamp `20260810_202928` (helper merge)

| Class | Baseline composed P / R | After merge P / R | TP / FP / FN | Status |
|-------|-------------------------|-------------------|--------------|--------|
| `pleural_thickening` (hierarchy-aware) | 0.909 / 0.781 | **0.910 / 0.784** | **222 / 22 / 61** | **promoted** |

- Discarded bare `apical_cap` / `pleural_scarring` helpers (0 unique gold residual; full helper OR +2 FP / +0 TP). Parent already covers apical-cap / pleural-scar / fibrosis arms.
- Kept framed residual only: modifier `based` for pleural[- ]based thickening.
- Coordinator: removed OR helpers `apical_cap`, `pleural_scarring` from `cxr_patterns.py`; deleted their `.txt` files. Kept ontology children (`pleural_plaque`, `pleural_calcification`) in the OR list.
- Mini-tests: `pleura` positive for pleural-based thickening; hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_202928/groups/pleural_thickening/` (`REPORT.md` retained).

### `pleural_effusion` + loculated Phase B7 — stamp `20260810_203427` (helper merge)

| Class | Baseline P / R | After merge P / R | TP / FP / FN | Status |
|-------|----------------|-------------------|--------------|--------|
| `pleural_effusion` (hierarchy-aware) | 0.987 / 0.932 | **0.988 / 0.935** | **1228 / 15 / 85** | **promoted** |
| `loculated_or_fissural_pleural_effusion` | 0.925 / 0.639 | **0.933 / 0.723** | **112 / 8 / 43** | **promoted** |

- Discarded bare `empyema` helper (parent already has `empyema|pyothorax`; `empyema\w*` = 0 unique gold residual) + residual PE polish (`no other…change`, volume/similar, slash-list, framed pleural collection, years-old history guard).
- Discarded bare `pseudotumor` helper (section F already covers; hyphen KEEP was +1 FP / +0 TP) + loculated residual polish (`-sided` / subpulmonic / hydropneumothorax, `into` / tracking-into fissure paths, inverted fissure fluid).
- Coordinator: removed OR helpers `empyema`, `pseudotumor` from `cxr_patterns.py`; deleted their `.txt` files. Loculated is now a single `_load_pattern` (still OR’d under PE as ontology child).
- Mini-tests updated (`pleura`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_203427/groups/{pleural_effusion,loculated_or_fissural_pleural_effusion}/` (`REPORT.md` retained).

### `pulmonary_nodule_or_mass` Phase B8 — stamp `20260810_204445` (helper merge)

| Class | Baseline composed P / R | After merge P / R | TP / FP / FN | Status |
|-------|-------------------------|-------------------|--------------|--------|
| `pulmonary_nodule_or_mass` (hierarchy-aware) | 0.841 / 0.919 | **0.843 / 0.941** | **428 / 80 / 27** | **promoted** |

- **DISCARD** both helpers as prompt-unfaithful: `pseudonodule` (nipple/skin/bone-island; OR +0 TP / +17 FP) and `thoracic_masslike` (extrapulmonary mass / adenopathy; OR +1 unfaithful TP / +33 FP). Combined helper OR +50 FP.
- Residual intrapulmonary polish only (calcified/sized/replacing/hemithorax mass; nodular gap/perihilar/interstitial; mass hedges).
- Coordinator: removed OR helpers `pseudonodule`, `thoracic_masslike` from `cxr_patterns.py`; deleted their `.txt` files. Kept ontology child `calcified_pulmonary_nodule_or_granuloma` in the OR list.
- Mini-tests updated (`focal_lesion_findings`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_204445/groups/pulmonary_nodule_or_mass/` (`REPORT.md` retained).

### Global recall wave — stamp `20260810_225816` (hierarchy-aware where mid-node)

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `costophrenic_angle_blunting` | 0.992 / 0.213 | **0.981 / 0.260** | **157 / 3 / 448** | **promoted** |
| `support_tube_present` | 0.981 / 0.579 | **0.982 / 0.587** | **1069 / 20 / 753** | **promoted** |
| `device_abnormality` | 0.909 / 0.318 | **0.923 / 0.400** | **263 / 22 / 395** | **promoted** |
| `support_line_present` | 0.997 / 0.660 | **0.997 / 0.662** | **1097 / 3 / 560** | **promoted** |

- Shared freeze (334823 ann / 4849 samples); recall-first with precision floors; prompt-faithful FN only.
- CPA: sinus bug fix, IU `XXXX`, sulcus/CPA fluid with negation glue (slight P dip, still ≥0.97).
- Tube: residual mediastinal/JP/surgical drains + trach compound arms; 0 FP delta.
- Device: malposition/retract/coil/sideport/azygos/sternal-fracture residuals; tip-in-expected FP bugfix; P↑.
- Line: NOR/nares scan bug + Swan-___ / PA / angiocatheter residuals; P held.
- Mini-tests updated (`pleura`, `support_device`); hard failures = 0. Snapshot bulk freed post-promote.
- Post-promote fix (2026-08-10): JP drain presence lookahead gained `project(?:s|ing)?` (aligned with surgical-drain arm); restored mini-test `"Jackson-Pratt drain projects over the lower chest."`.

Workdirs: `.agent/regex_refine/20260810_225816/groups/{costophrenic_angle_blunting,support_tube_present,device_abnormality,support_line_present}/`.

### Global recall wave #2 — stamp `20260810_232835` (true leaves)

Shared freeze (344755 ann / 4849 samples); subgraph baselines; recall-first with class P floors. Fresh production-OR scan was `_recall_scan_20260810_232322`.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `lobar_atelectasis` | 0.838 / 0.438 | **0.839 / 0.442** | **198 / 38 / 250** | **promoted** |
| `vascular_graft_or_stent` | 0.952 / 0.192 | **0.944 / 0.245** | **51 / 3 / 157** | **promoted** |
| `focal_osseous_lesion` | 0.780 / 0.256 | **0.948 / 0.306** | **55 / 3 / 125** | **promoted** |
| `pneumonia` | 0.897 / 0.649 | **0.909 / 0.705** | **570 / 57 / 239** | **promoted** |

- Lobar: near-complete named-lobe collapse + mucous-plug RUL; no subtype/whole-lung re-widen (+2 TP / 0 FP).
- Graft: vessel-localized aorta/venous/device-first stents + CABG/bypass graft markers; leftover FN mostly PICC/valve/non-vascular-stent gold.
- Osseous: negation/non-bone FP trim + lucent/blastic/lysis/bony-lesion residuals; leftover FN mostly fracture/DJD gold already covered by other MSK classes.
- Pneumonia: COVID-19/cavitary/suggestive-for/consideration residual; no bare opacity arms (+45 TP / −3 FP).
- Mini-tests updated (`pulmonary_opacity`, `support_device`, `musculoskeletal`, `diagnostic_label`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260810_232835/groups/{lobar_atelectasis,vascular_graft_or_stent,focal_osseous_lesion,pneumonia}/`.

### Global recall wave #3 — stamp `20260811_000435` (true leaves)

Shared freeze (348723 ann / 4849 samples); subgraph baselines; recall-first with class P floors.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `consolidation` | 0.890 / 0.705 | **0.895 / 0.709** | **632 / 74 / 259** | **promoted** |
| `nontherapeutic_foreign_body` | 0.887 / 0.329 | **1.000 / 0.392** | **56 / 0 / 87** | **promoted** |
| `mediastinal_mass` | 0.778 / 0.257 | **1.000 / 0.294** | **32 / 0 / 77** | **promoted** |
| `pulmonary_cyst_or_bulla` | 0.952 / 0.331 | **0.978 / 0.372** | **45 / 1 / 76** | **promoted** |
| `pleural_nodule_or_mass` | 0.857 / 0.255 | **0.969 / 0.330** | **31 / 1 / 63** | **promoted** |

- Consolidation: over-negation exceptions + airspace density / alveolar+interstitial; no bare opacity/pneumonia reopen.
- FB: negation/bullet-shaped FPs cleared; buckshot/battery/glass/ingested residual. Post-promote: restored laterality + `thorax|chest` on `foreign body in`, and `unexpected foreign body` (mini-test positives had broken).
- Mediastinal mass: negation/history/`or`-list leaks closed; paratracheal / arising-from-mediastinum residual.
- Cyst/bulla: air cysts / thin-walled cystic spaces / cystic lucency; thick-walled-bulla cavity DDx blocked.
- Pleural nodule: history mesothelioma FPs closed; nodularity / pleural-based lesion residual (not contact-only).
- Mini-tests updated (`pulmonary_opacity`, `other`, `cardiomediastinal`, `focal_lesion_findings`, `pleura`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_000435/groups/{consolidation,nontherapeutic_foreign_body,mediastinal_mass,pulmonary_cyst_or_bulla,pleural_nodule_or_mass}/`.

### Global recall wave #4 — stamp `20260811_002344` (true leaves)

Shared freeze (356655 ann / 4849 samples); subgraph baselines; recall-first with class P floors. Scarring and fibrotic ran in parallel with prompt-split briefs.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `focal_pulmonary_scarring` | 0.958 / 0.585 | **0.990 / 0.759** | **205 / 2 / 65** | **promoted** |
| `interstitial_lung_disease` | 0.879 / 0.312 | **1.000 / 0.366** | **34 / 0 / 59** | **promoted** |
| `mediastinal_shift` | 0.864 / 0.543 | **1.000 / 0.543** | **89 / 0 / 75** | **promoted** (P-only) |
| `fibrotic_lung_change` | 0.977 / 0.422 | **1.000 / 0.549** | **56 / 0 / 46** | **promoted** |
| `hilar_enlargement` | 0.851 / 0.599 | **0.956 / 0.750** | **129 / 6 / 43** | **promoted** |

- Scarring: limited scar DDx/location; dropped pleural-only and extensive/diffuse fibrotic-change.
- ILD: history/transplant/CT-DDx FPs closed; UIP pattern / known UIP / idiopathic interstitial pneumonitis residual.
- Shift: negation/`or`-list/`projecting` FPs closed; leftover FN = trachea-only / inference.
- Fibrotic: explicit fibrosis / UIP / extensive scarring; history-only FP closed.
- Hilar: hilum-size/fullness/adenopathy/mass residual; congestion/density-without-enlargement trimmed.
- Mini-tests updated (`pulmonary_opacity`, `diagnostic_label`, `cardiomediastinal`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_002344/groups/{focal_pulmonary_scarring,interstitial_lung_disease,mediastinal_shift,fibrotic_lung_change,hilar_enlargement}/`.

### Global recall wave #5 — stamp `20260811_011334` (1 mid-node + 4 leaves)

Shared freeze (364580 ann / 4849 samples); subgraph baselines (hierarchy-aware for `bone_fracture`); recall-first with class P floors. Production-OR rank scan: `.agent/regex_refine/_recall_scan/` (job 93325). DJD and fracture ran in parallel with prompt-split briefs.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `bone_fracture` (composed) | 0.918 / 0.730 | **0.939 / 0.749** | **355 / 23 / 119** | **promoted** |
| `degenerative_osseous_change` | 0.985 / 0.761 | **0.989 / 0.809** | **355 / 4 / 84** | **promoted** |
| `rounded_atelectasis` | 0.778 / 0.311 | **0.824 / 0.311** | **14 / 3 / 31** | **promoted** (P-only) |
| `mediastinal_widening` | 0.806 / 0.608 | **0.966 / 0.654** | **85 / 3 / 45** | **promoted** |
| `enlarged_pulmonary_artery` | 0.957 / 0.512 | **1.000 / 0.593** | **51 / 0 / 35** | **promoted** |

- Fracture: residual sternum/shoulder/glenoid/tuberosity + redaction/pathologic; readonly rib/clavicle/VCF untouched. Leftover composed FNs mostly out-of-FOV / device / child-gap gold (resection, nonspecific height loss).
- DJD: spurring / degeneration-of / arthrosis / spondylitic; no fracture/osteopenia/kyphosis arms.
- Rounded: history/`?` FP trim; leftover FNs nonspecific atelectasis gold.
- Widening: named-region/contour residual; technique/adenopathy FPs trimmed; did not chase pneumomediastinum/shift/mass.
- EPA: bilateral/right-and-left laterality + pulmonary arterial enlargement; leftover FNs vessels/hila/edema gold.
- Mini-tests updated (`bone_fracture_findings`, `musculoskeletal`, `pulmonary_opacity`, `cardiomediastinal`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_011334/groups/{bone_fracture,degenerative_osseous_change,rounded_atelectasis,mediastinal_widening,enlarged_pulmonary_artery}/`.

### Global recall wave #6 — stamp `20260811_013434` (true leaves)

Shared freeze (372502 ann / 4849 samples); subgraph baselines; recall-first with class P floors.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `pacemaker_or_icd` | 0.984 / 0.759 | **1.000 / 0.771** | **253 / 0 / 75** | **promoted** |
| `reticular_interstitial_opacity_pattern` | 1.000 / 0.701 | **1.000 / 0.707** | **130 / 0 / 54** | **promoted** |
| `tracheal_deviation` | 1.000 / 0.559 | **1.000 / 0.676** | **46 / 0 / 22** | **promoted** |
| `loculated_or_fissural_pleural_effusion` | 0.930 / 0.717 | **1.000 / 0.735** | **122 / 0 / 44** | **promoted** |
| `sternotomy_wires` | 1.000 / 0.772 | **0.995 / 0.815** | **189 / 1 / 43** | **promoted** |

- Pacemaker: IU/paraphrase generator residual; dropped external pads; leftover FN epicardial/temporary/ports.
- Reticular: `reticulogranular` only; leftover FN edema/Kerley/generic interstitial.
- Trachea: retraction / shift-of-trachea / degree adverbs; leftover FN mostly hallucinated quotes.
- Loculated: air+fluid / localized collection; all 9 FPs cleared (loculated PTX/history).
- Sternotomy: incision wires + IU `sternotomy XXXX`; leftover FN bare post-sternotomy/CABG.
- Mini-tests updated (`support_device`, `pulmonary_opacity`, `airway`, `pleura`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_013434/groups/{pacemaker_or_icd,reticular_interstitial_opacity_pattern,tracheal_deviation,loculated_or_fissural_pleural_effusion,sternotomy_wires}/`.

### Global recall wave #7 — stamp `20260811_014802` (1 mid-node + 4 leaves)

Shared freeze (372502 ann / 4849 samples); subgraph baselines (hierarchy-aware for `pleural_thickening`); recall-first with class P floors.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `pleural_thickening` (composed) | 0.912 / 0.768 | **0.927 / 0.772** | **240 / 19 / 71** | **promoted** |
| `vertebral_compression_fracture` | 0.884 / 0.670 | **0.915 / 0.714** | **65 / 6 / 26** | **promoted** |
| `chest_tube` | 0.949 / 0.844 | **0.986 / 0.856** | **362 / 5 / 61** | **promoted** |
| `aortic_enlargement_or_ectasia` | 0.942 / 0.699 | **0.987 / 0.828** | **77 / 1 / 16** | **promoted** |
| `prosthetic_cardiac_valve` | 1.000 / 0.721 | **1.000 / 0.802** | **69 / 0 / 17** | **promoted** |

- Thickening: negation/r/o lookbehinds + framed compatible-with-thickening; plaque/calc readonly untouched.
- VCF: `wedge fractures` / partially compressed VB (wave-5 child-gap); leftover FN nonspecific height loss/collapse.
- Chest tube: thoracostomies-tube / pleural+mediastinal drains / pigtail-type; leftover FN generic lines-and-tubes gold.
- Aorta: `ectatic` vs `ectasia` token bug; leftover FN prominence/unfolding gold.
- Valve: valvular prosthesis / percutaneous aortic valve / named-valve lists; leftover FN history-only AVR/MVR.
- Mini-tests updated (`pleura`, `bone_fracture_findings`, `support_device`, `cardiomediastinal`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_014802/groups/{pleural_thickening,vertebral_compression_fracture,chest_tube,aortic_enlargement_or_ectasia,prosthetic_cardiac_valve}/`.

### Global recall wave #8 — stamp `20260811_020736` (5 leaves, incl. CVC)

Shared freeze (374481 ann / 4849 samples); subgraph baselines; recall-first with class P floors. `support_device_present` held while CVC was edited.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `central_venous_catheter` | 0.997 / 0.866 | **1.000 / 0.888** | **1034 / 0 / 130** | **promoted** |
| `low_lung_volume` | 0.994 / 0.883 | **0.997 / 0.890** | **683 / 2 / 84** | **promoted** |
| `bronchial_wall_thickening` | 0.974 / 0.725 | **1.000 / 0.873** | **89 / 0 / 13** | **promoted** |
| `elevated_or_eventrated_hemidiaphragm` | 0.988 / 0.832 | **1.000 / 0.891** | **180 / 0 / 22** | **promoted** |
| `rib_fracture` | 0.881 / 0.843 | **0.983 / 0.886** | **225 / 4 / 29** | **promoted** |

- CVC: central-line-stable / PIC-PCVC-UVC / IU port; no Swan-Ganz or generic lines; leftover FN mostly PA-catheter / vague-lines gold.
- LLV: global/inspiration paraphrases; leftover FN unilateral/atelectasis/technique.
- BWT: Brochial typo, optional-wall + airway/bronchiolar thickening; both negation FPs closed.
- Hemidiaphragm: unilateral elevation of the diaphragm + tenting/eventration; `decreased elevation` FPs gone.
- Rib: FP 29→4 (indication `rib fx`, CT-only, negation); no resection/osteotomy.
- Mini-tests updated (`support_device`, `lung_volume`, `airway`, `diaphragm`, `bone_fracture_findings`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_020736/groups/{central_venous_catheter,low_lung_volume,bronchial_wall_thickening,elevated_or_eventrated_hemidiaphragm,rib_fracture}/`.

### Global recall wave #9 — stamp `20260811_023520` (5 leaves)

Shared freeze (380415 ann / 4849 samples); subgraph baselines; mixed recall + FP trim. Held `support_device_present` / `support_tube_present` / `enlarged_cardiomediastinal_silhouette`.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `enlarged_cardiac_silhouette` | 0.983 / 0.930 | **0.992 / 0.957** | **938 / 8 / 42** | **promoted** |
| `enteric_tube` | 0.947 / 0.917 | **0.969 / 0.941** | **555 / 18 / 35** | **promoted** |
| `hilar_or_mediastinal_nodal_enlargement` | 0.827 / 0.794 | **0.977 / 0.824** | **84 / 2 / 18** | **promoted** |
| `endotracheal_tube` | 0.978 / 0.934 | **0.993 / 0.952** | **534 / 4 / 27** | **promoted** |
| `pulmonary_cavity` | 0.861 / 0.827 | **1.000 / 0.840** | **63 / 0 / 12** | **promoted** |

- Cardiac: LARGE HEART / increasing size / ULoN-to-borderline; leftover FN cardiomediastinal/prominent.
- Enteric: list lookbehinds, Replogle/suction; leftover FP mostly tip-off-FOV (kept as presence).
- Nodal: FP 17→2 (list/definite/`not excluded`); leftover FN bare hilar prominence.
- ETT: interval/mainstem intubation; leftover FN generic tubes/lines.
- Cavity: apex/lobe-first cavity + developing abscess; all 10 FPs cleared.
- Mini-tests updated (`cardiomediastinal`, `support_device`, `focal_lesion_findings`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_023520/groups/{enlarged_cardiac_silhouette,enteric_tube,hilar_or_mediastinal_nodal_enlargement,endotracheal_tube,pulmonary_cavity}/`.

### Global recall wave #10 — stamp `20260811_025217` (5 leaves)

Shared freeze (380415 ann / 4849 samples); subgraph baselines; mixed P trim + prompt-faithful R. Held `lung_opacity` / `calcification`.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `kerley_lines` | 0.710 / 0.667 | **0.960 / 0.727** | **24 / 1 / 9** | **promoted** |
| `ground_glass_opacity_pattern` | 0.947 / 0.750 | **1.000 / 0.750** | **36 / 0 / 12** | **promoted** (P-only) |
| `calcified_pulmonary_nodule_or_granuloma` | 0.978 / 0.848 | **1.000 / 0.914** | **96 / 0 / 9** | **promoted** |
| `flattened_diaphragm` | 1.000 / 0.815 | **1.000 / 0.833** | **45 / 0 / 9** | **promoted** |
| `cardiac_or_valvular_calcification` | 1.000 / 0.800 | **1.000 / 0.900** | **36 / 0 / 4** | **promoted** |

- Kerley: `XXXX B-lines` / curly B; leftover FN edema/reticular gold.
- GGO: prior-CT FPs closed; no hazy/RDS chase.
- Granuloma: lingular/lobe/base intrapulmonary; leftover FN hilar.
- Flattened: posterior diaphragm; leftover FN silhouette/eventration.
- Cardiac calc: native coronary / apical pericardial; leftover FN aorta-only.
- Mini-tests updated (`pulmonary_opacity`, `focal_lesion_findings`, `diaphragm`, `cardiomediastinal`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_025217/groups/{kerley_lines,ground_glass_opacity_pattern,calcified_pulmonary_nodule_or_granuloma,flattened_diaphragm,cardiac_or_valvular_calcification}/`.

### Global recall wave #11 — stamp `20260811_030248` (3 mid-nodes + 2 leaves)

Shared freeze (383048 ann / 4849 samples); hierarchy-aware subgraph for mid-nodes; residual parent polish + leaf P trim.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `support_device_present` (composed) | 0.996 / 0.899 | **0.997 / 0.908** | **2343 / 8 / 238** | **promoted** |
| `enlarged_cardiomediastinal_silhouette` (composed) | 0.978 / 0.912 | **0.994 / 0.914** | **1074 / 6 / 101** | **promoted** |
| `pleural_effusion` (composed) | 0.990 / 0.931 | **0.994 / 0.950** | **1395 / 9 / 73** | **promoted** |
| `emphysema` | 0.955 / 0.898 | **0.991 / 0.898** | **106 / 1 / 12** | **promoted** (P-only) |
| `pneumomediastinum` | 0.941 / 0.842 | **1.000 / 0.877** | **50 / 0 / 7** | **promoted** |

- Support device: unnamed hardware + generic current lines; leftover FN fracture/removal/piercing. Children readonly.
- ECS: parent enlarge/widen; leftover FN shift/prominence/technique.
- Effusion: questionable/compatible-with/change-in; hydro/loculated untouched.
- Emphysema: history/baseline FPs; leftover FN bullae/COPD-only.
- Pneumomediastinum: superior-mediastinum air; leftover FN shift/PTX/NG.
- Mini-tests updated (`support_device`, `cardiomediastinal`, `pleura`, `pulmonary_parenchyma`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_030248/groups/{support_device_present,enlarged_cardiomediastinal_silhouette,pleural_effusion,emphysema,pneumomediastinum}/`.

### Global recall wave #12 — stamp `20260811_042446` (2 mid-nodes + 3 leaves)

Shared freeze (383071 ann / 4849 samples, including 23 Gemini-3.5 leftovers merged into 2.5 gold); hierarchy-aware subgraph for `hernia` / `calcification`.

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `hernia` (composed) | 0.980 / 0.824 | **0.980 / 0.832** | **99 / 2 / 20** | **promoted** |
| `breast_implant` | 0.933 / 0.757 | **0.970 / 0.865** | **32 / 1 / 5** | **promoted** |
| `scoliosis` | 1.000 / 0.892 | **1.000 / 0.961** | **98 / 0 / 4** | **promoted** |
| `pleural_plaque` | 0.969 / 0.816 | **1.000 / 0.816** | **31 / 0 / 7** | **promoted** (P-only) |
| `calcification` (composed) | 0.986 / 0.936 | **0.993 / 0.939** | **555 / 4 / 36** | **promoted** |

- Hernia: parent-only lung-herniation-with-gap; hiatal readonly. Leftover FN history/eventration/Chilaiditi/pull-up.
- Breast implant: plural prostheses / expander / silicone / framed postmastectomy-with-implant; explantation FP closed.
- Scoliosis: dextroscoliotic / spaced dextro curvature / spinal curvature; P stayed 1.0.
- Plaque: DDx `can have a similar appearance` FP closed; leftover FN thickening/calc/asbestos gold.
- Calcification: costochondral mineralization / contain-calcium; leftover FN bone/granuloma/plaque/athero/stones (prompt ceiling). Children readonly.
- Mini-tests updated (`diaphragm`, `support_device`, `musculoskeletal`, `pleura`, `calcific_findings`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_042446/groups/{hernia,breast_implant,scoliosis,pleural_plaque,calcification}/`.

### Precision-trim wave — stamp `20260811_044912` (4 leaves + 1 mid-node HOLD)

Shared freeze (383071 ann / 4849 samples); hierarchy-aware subgraph; **hard R floor = baseline R** (must not drop).

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `pulmonary_edema` | 0.837 / 0.927 | **0.947 / 0.942** | **709 / 40 / 44** | **promoted** |
| `pulmonary_nodule_or_mass` (composed) | 0.831 / 0.927 | **0.881 / 0.927** | **473 / 64 / 37** | **promoted** (granuloma readonly) |
| `lobar_atelectasis` | 0.831 / 0.431 | **0.934 / 0.431** | **211 / 15 / 278** | **promoted** |
| `hydropneumothorax` | 0.819 / 0.910 | **0.886 / 0.917** | **132 / 17 / 12** | **promoted** |
| `enlarged_cardiomediastinal_silhouette` (composed) | 0.994 / 0.914 | 0.994 / 0.914 | 1074 / 6 / 101 | **held** (6 FPs child-driven) |

- Edema: clause-local scan vs list-negation; kept `or`/`vs` hedges; leftover FN CHF-only / bare overload.
- Nodule: residual parent guards (list negation, `e/o`, extrapulmonary prefixes, `noncalcified`); leftover FP CT-only / hilar / vague nodule.
- Lobar: laterality-aware negation + DDx gap skip; all 211 TPs held; leftover FN subtype/whole-lung/inference gold.
- Hydro: history/indication/AFL/`air bronchogram`/tight bilateral bind; OR-list stayed closed; leftover FP same-side coexistence with gold unknown.
- ECS: no parent edit; all 6 subgraph FPs child-driven; leftover FN prompt-unfaithful or readonly-child gaps.
- Mini-tests updated (`pulmonary_opacity`, `focal_lesion_findings`, `pleura`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_044912/groups/{pulmonary_edema,pulmonary_nodule_or_mass,enlarged_cardiomediastinal_silhouette,lobar_atelectasis,hydropneumothorax}/`.

### Precision-trim wave #2 — stamp `20260811_051641` (3 leaves + 1 mid-node)

Shared freeze (383071 ann / 4849 samples); hierarchy-aware subgraph for `bone_fracture`; **hard R floor = baseline R**. Interrupted mid-edit; reports closed from existing `eval/` dumps (pneumonia re-eval’d by coordinator).

| Class | Baseline P / R | After P / R | TP / FP / FN | Status |
|-------|----------------|-------------|--------------|--------|
| `consolidation` | 0.893 / 0.694 | **0.918 / 0.694** | **670 / 60 / 296** | **promoted** |
| `pneumonia` | 0.904 / 0.709 | **0.914 / 0.709** | **639 / 60 / 262** | **promoted** |
| `enteric_tube` | 0.969 / 0.941 | **0.970 / 0.941** | **555 / 17 / 35** | **promoted** |
| `bone_fracture` (composed) | 0.945 / 0.769 | **0.982 / 0.769** | **376 / 7 / 113** | **promoted** (children readonly) |

- Consolidation: DDx / unusual-new hedges / likely-atelectasis / indication air bronchograms; leftover FN opacity/pneumonia/atelectasis gold.
- Pneumonia: Pass-4 negation/resolution/indication/`no findings of`; leftover FN bare-opacity gold.
- Enteric: only `transesophageal drainage` blocked; leftover FP tip-off-FOV / concat (bad gold).
- Fracture: residual parent FP trim; leftover FPs mostly child-driven; children not copied.
- Mini-tests updated (`pulmonary_opacity`, `diagnostic_label`, `support_device`); hard failures = 0. Snapshot bulk freed post-promote.

Workdirs: `.agent/regex_refine/20260811_051641/groups/{consolidation,pneumonia,enteric_tube,bone_fracture}/`.

## Suggested next

Post-promote production-OR scan: `.agent/regex_refine/_recall_scan_20260811_054411/` (job 93333; 383071 ann / 4849 samples). P-trim waves landed with **R held** (edema also **R +0.015**). Residual top FN×(1−R) is still CPA / support-tube/line / device_abnormality / lobar / graft — prompt ceilings, do not reopen from sample gold alone. Remaining high-FP umbrellas (`lung_opacity` FP 216, ECS 68, `calcification` 62) are mostly OR extras / children. Skip PVC / TB.

**Full-corpus FN mining** (new lexicon holes the 4849 samples never showed): [`FN_MINE.md`](FN_MINE.md).

## Non-goals / pitfalls

- Do not copy entire `groups/*/cxr_classes/` into production — only `owned_class_ids` stems (never `readonly_class_ids`).
- Do not point agents at the live growing `sample_annotations.jsonl`.
- Mini-tests validate production; workdir improvements need promote (and fixture updates) before mini-tests reflect them.
- LLM gold is noisy; document leftover FNs that look like bad gold rather than forcing 100% recall.
- Mid-node leaf-only metrics overstate parent-pattern recall when children already cover gold — always use hierarchy-aware `eval_class_ids` for mid-node promote decisions.
