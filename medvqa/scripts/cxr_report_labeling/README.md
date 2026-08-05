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
| `annotate_reports_with_llm.py` | Capped sample LLM annotation |
| `eval_regex_vs_llm.py` | Metrics + FP/FN dumps vs LLM gold |
| `apply_regex_to_dataset.py` | Full-dataset regex matches |
| `verify_positive_matches_with_llm.py` | LLM verify regex positives |
| `materialize_final_labels.py` | Final Unmentioned / 5-way labels |
