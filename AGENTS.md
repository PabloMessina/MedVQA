# MedVQA — Agent Guide

Chest X-ray medical vision–language research codebase: VQA/QA, report generation, phrase grounding, multilabel classification, fact embeddings / NLI, and related NLP. Installable Python package: `medvqa` (`setup.py`).

This file is a **navigation map**. Prefer searching/opening the listed paths over reading large notebooks or dumping whole packages into context.

## Layout

| Path | Role |
|------|------|
| `medvqa/datasets/` | Dataset loaders and `*_dataset_management.py` per corpus/task |
| `medvqa/models/` | Model defs: `vqa/`, `qa/`, `vision/`, `report_generation/`, `phrase_grounding/`, `multimodal_encoding/`, `nlp/`, `checkpoint/`, `ensemble/` |
| `medvqa/training/` | Ignite train loops (`get_engine`, etc.) — imported by scripts, not CLIs |
| `medvqa/scripts/` | Runnable entry points: `training_scripts/`, `evaluation_scripts/`, per-dataset preprocessing |
| `medvqa/metrics/` | Ignite-attachable metrics (`attach_*` in `__init__.py`) |
| `medvqa/evaluation/` | Post-hoc eval helpers, aggregation, plots |
| `medvqa/losses/` | Losses, optimizers, schedulers |
| `medvqa/utils/` | Shared helpers (files, OpenAI API, constants, torch, logging) |
| `medvqa/configs/` | YAML configs (`training/`, `evaluation/`, `freezing/`) |
| `medvqa/prompts/` | LLM prompt text files (`LLM_PROMPTS_DIR`) |
| `medvqa/notebooks/` | Exploratory experiments — large; not the API source of truth |
| `medvqa/slurm/` | Cluster launchers (phrase grounding train/eval) |
| `medvqa/settings.py` | Loads repo-root `.env`; workspace + dataset path constants |
| `.agent/` | Local scratch (gitignored) — not package code |

## Paths and config

- Repo root `.env` (gitignored) → loaded by `medvqa/settings.py`.
- **Required:** `MEDVQA_WORKSPACE_DIR` → derives `experiments/`, `cache/`, `tmp/`, `results/`.
- Optional fast workspaces: `MEDVQA_FAST_WORKSPACE_DIR` / `MEDVQA_LARGE_FAST_WORKSPACE_DIR` (fall back to `WORKSPACE_DIR`) → `FAST_CACHE_DIR`, `LARGE_FAST_CACHE_DIR`, `FAST_TMP_DIR`.
- Dataset and tool paths are env vars in `settings.py` (some required, some warn if missing).
- YAML configs live under `medvqa/configs/`; relative paths resolve via `CONFIGS_FOLDER` / `files_utils.load_config_yaml`.
- Prefer path constants from `medvqa.settings` (e.g. `WORKSPACE_DIR`, `EXPERIMENTS_DIR`, `RESULTS_DIR`).

## Environment

Default conda env for this project (also the Slurm `CONDA_ENV` default):

```bash
module load conda
conda activate py313
```

Override with `CONDA_ENV=<name>` when using `medvqa/slurm/` launchers. Some external tools use separate envs configured in `.env` (e.g. CheXpert labeler, RadCliQ) — do not confuse those with the main `py313` env.

## How to run things

Typical pattern:

```bash
python medvqa/scripts/training_scripts/<train_*.py> [args]
python medvqa/scripts/evaluation_scripts/<eval_*.py> [args]
```

Phrase grounding often uses `--config_filepath` pointing at YAML under `medvqa/configs/`. Slurm wrappers in `medvqa/slurm/` call those scripts.

### Important CLIs

| Script | Task |
|--------|------|
| `scripts/training_scripts/train_vqa.py` | Open-ended VQA |
| `scripts/training_scripts/train_visual_module.py` | Vision / classification / bbox-style modules |
| `scripts/training_scripts/train_phrase_grounding.py` | Phrase grounding (YAML) |
| `scripts/training_scripts/train_image2report.py` | Image → report |
| `scripts/training_scripts/train_labels2report.py` | Labels → report |
| `scripts/training_scripts/train_qa.py` | Text-only QA |
| `scripts/training_scripts/train_seq2seq.py` | Seq2seq NLP |
| `scripts/training_scripts/train_fact_embedding.py` | Fact / CXR embeddings |
| `scripts/training_scripts/train_yolov{5,8,11}.py` | YOLO detectors |
| `scripts/evaluation_scripts/eval_vqa.py` | VQA eval |
| `scripts/evaluation_scripts/eval_report_generation.py` | Report-gen eval |
| `scripts/evaluation_scripts/eval_phrase_grounding.py` | Phrase grounding eval |
| `scripts/evaluation_scripts/eval_visual_module.py` | Visual-module eval |
| `scripts/evaluation_scripts/eval_multilabel_classification.py` | Multilabel classification eval |

Training logic for a task usually pairs as: `scripts/…/train_X.py` → `training/X.py` → `models/<area>/`.

## Where to look for X

| Goal | Start here |
|------|------------|
| Add / change a dataset | `datasets/<name>/` + paths in `settings.py` / `.env`; optional prep in `scripts/<name>/`; IDs in `utils/constants.py` |
| Add / change a model | `models/<task>/` (often also `vision/visual_modules.py`) |
| Train VQA | `train_vqa.py` → `training/vqa.py` → `models/vqa/` |
| Train report gen | `train_image2report.py` / `train_labels2report.py` → `models/report_generation/` |
| Train classification / vision | `train_visual_module.py`, `train_yolov*.py` |
| Train phrase grounding | `train_phrase_grounding.py` + `configs/training/` |
| CXR report labeling | `datasets/cxr_report_labeling/` + `scripts/cxr_report_labeling/` (see that folder’s `README.md`); ontology in `datasets/cxr_report_labeling/ontology/`; class prompts in `prompts/cxr_classes/`; regex patterns in `datasets/regular_expressions/`; outputs under `cache|results/cxr_report_labeling/` |
| Evaluate | Matching `eval_*.py` in `scripts/evaluation_scripts/`; helpers in `evaluation/` |
| Metrics | `metrics/{nlp,medical,classification,bbox,segmentation}/` |
| LLM / OpenAI usage | Prompts: `prompts/`; API: `utils/openai_api_utils.py`; callers: many `scripts/**/*openai*` / LLM scripts |
| Checkpoints / results | Under `$MEDVQA_WORKSPACE_DIR/experiments/` and `…/results/`; loading via `models/checkpoint/` |

## Conventions worth knowing

- Dataset managers often named `{corpus}_{task}_dataset_management.py` with trainer/evaluator classes.
- Stack: PyTorch + Ignite; metrics via `attach_*`; newer flows use `training.utils.run_common_boilerplate_code_and_start_training`.
- Shared vocab/label/metric names: `utils/constants.py`.
- OpenAI scripts take `--api_key_name` (env **var name**, not the key).
- Regex / clinical class patterns: `datasets/regular_expressions/` (distinct from `prompts/`).

## Do not casually edit

- `.env` — secrets and machine-local paths
- `.agent/`, `medvqa.egg-info/`, `__pycache__/`, weights (`*.pt`), media
- Bulk rewrites of `notebooks/`, `prompts/cxr_classes/`, or `datasets/regular_expressions/cxr_classes/`
- Hardcoded absolute paths in `slurm/*.sh` unless intentionally retargeting the cluster

## Search tips for agents

1. Start from this map, then open the matching `train_*` / `eval_*` script.
2. Follow imports into `training/`, `models/`, and `datasets/`.
3. Use filename/task keywords (`phrase_grounding`, `mimiccxr`, `image2report`) rather than reading entire directories.
4. Treat notebooks as historical experiments, not the primary reference.
