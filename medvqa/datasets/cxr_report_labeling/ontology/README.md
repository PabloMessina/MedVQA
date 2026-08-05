# Canonical CXR classifier ontology

Tracked source of truth for the 79-class chest radiograph multilabel ontology used with
`medvqa/prompts/cxr_classes/` (NLI hypotheses) and the CXR report-labeling pipeline.

| File | Role |
|------|------|
| `canonical_cxr_ontology.yaml` | Authoritative ontology (ids, parents, categories, dataset mappings, definitions) |
| `canonical_cxr_ontology.schema.json` | JSON Schema for the YAML |
| `canonical_cxr_classes.txt` | Display names only (one per line; alphabetical) |
| `canonical_cxr_classes_review.md` | Human-readable class table + hierarchy + def coverage |
| `canonical_cxr_classes_dataset_coverage_review.md` | Reverse audit vs PUC-CXR / CXR-LT 2024 / VinDr-CXR |
| `radlex_mappings.csv` | RadLex concepts, match quality, review flags, and notes by canonical class |
| `presentation_hierarchy.html` | Browseable category-grouped tree |
| `presentation_hierarchy_dag.html` | Interactive propagation DAG (child → parent, roots at top) |
| `presentation_mapping_table.html` | RadLex and dataset mappings + short/full definitions |
| `generate_presentations.py` | Regenerates the HTML presentations from the YAML, RadLex CSV, and prompt `.txt` files |
| `PUCCXR_classes.txt`, `CXRLT2024_classes.txt`, `VinDr-CXR_classes.txt` | Upstream label lists used by the coverage review |

Prompt hypotheses live in `medvqa/prompts/cxr_classes/<class_id>.txt` (basename = ontology `id`).

```bash
# From repo root:
python3 medvqa/datasets/cxr_report_labeling/ontology/generate_presentations.py
# Optional local scratch copy:
python3 medvqa/datasets/cxr_report_labeling/ontology/generate_presentations.py \
  --also-copy-to .agent/canonical_classes
```

`presentation_hierarchy_dag.html` loads Cytoscape / dagre from unpkg (needs network once when opening the file).

Local scratch under `.agent/canonical_classes/` is not tracked; edit files here instead.
