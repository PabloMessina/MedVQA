"""Minimal SRR-BERT-Leaves report labeler with sentence merge and disk cache."""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "StanfordAIMI/SRR-BERT-Leaves"
DEFAULT_TOKENIZER_PATH = "microsoft/BiomedVLP-CXR-BERT-general"
MAX_LENGTH = 128
NUM_LABELS = 55

_RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
_MAPPING_PATH = os.path.join(_RESOURCES_DIR, "leaves_mapping.json")


def _load_label_map(mapping_path: str = _MAPPING_PATH) -> Dict[str, int]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    if len(label_map) != NUM_LABELS:
        raise ValueError(f"Expected {NUM_LABELS} labels, found {len(label_map)}")
    return label_map


LABEL_MAP = _load_label_map()
SRR_BERT_LEAVES_CLASS_NAMES: List[str] = [
    name for name, _ in sorted(LABEL_MAP.items(), key=lambda kv: kv[1])
]
IDX2LABEL = {i: name for i, name in enumerate(SRR_BERT_LEAVES_CLASS_NAMES)}
LABEL2IDX = {name: i for i, name in enumerate(SRR_BERT_LEAVES_CLASS_NAMES)}
NO_FINDING_IDX = LABEL2IDX["No Finding"]


def ensure_nltk_punkt() -> None:
    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_text(text: Union[str, object]) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else " "


def merge_labels(labels_list: Sequence[np.ndarray]) -> np.ndarray:
    merged = np.zeros((NUM_LABELS,), dtype=np.int8)
    if not labels_list:
        merged[NO_FINDING_IDX] = 1
        return merged
    for labels in labels_list:
        arr = np.asarray(labels, dtype=np.int8)
        if arr.shape != (NUM_LABELS,):
            raise ValueError(f"Expected label vector of length {NUM_LABELS}, got {arr.shape}")
        merged = np.maximum(merged, arr)
    finding_mask = np.ones(NUM_LABELS, dtype=bool)
    finding_mask[NO_FINDING_IDX] = False
    if np.any(merged[finding_mask] == 1):
        merged[NO_FINDING_IDX] = 0
    elif not np.any(merged):
        merged[NO_FINDING_IDX] = 1
    return merged


def labels_to_names(labels: Sequence[int]) -> List[str]:
    return [IDX2LABEL[i] for i, flag in enumerate(labels) if flag]


class SRRBertLeavesLabeler:
    """Sentence-split + OR-merge multilabel labeler with report-hash disk cache."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        default_batch_size: int = 32,
        cache_dir: Optional[str] = None,
        threshold: float = 0.5,
        verbose: bool = True,
        model_path: str = DEFAULT_MODEL_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        max_length: int = MAX_LENGTH,
    ):
        ensure_nltk_punkt()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.default_batch_size = default_batch_size
        self.cache_dir = cache_dir
        self.threshold = threshold
        self.verbose = verbose
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.class_names = SRR_BERT_LEAVES_CLASS_NAMES

        if self.verbose:
            logger.info("Loading tokenizer from %s", tokenizer_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        if self.verbose:
            logger.info("Loading model from %s on %s", model_path, self.device)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=NUM_LABELS
        )
        self.model.to(self.device).eval()

        self.sentence_to_labels: Dict[str, np.ndarray] = {}
        self.report_hash_to_labels: Dict[str, np.ndarray] = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_caches()

    def _sentence_cache_path(self) -> str:
        assert self.cache_dir
        return os.path.join(self.cache_dir, "sentence_to_labels.pkl")

    def _report_cache_path(self) -> str:
        assert self.cache_dir
        return os.path.join(self.cache_dir, "report_hash_to_labels.pkl")

    def _load_caches(self) -> None:
        for path, attr in (
            (self._sentence_cache_path(), "sentence_to_labels"),
            (self._report_cache_path(), "report_hash_to_labels"),
        ):
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        setattr(self, attr, pickle.load(f))
                    if self.verbose:
                        logger.info("Loaded %s (%d entries)", path, len(getattr(self, attr)))
                except Exception as e:
                    logger.warning("Could not load %s: %s", path, e)

    def save_caches(self) -> None:
        if not self.cache_dir:
            return
        for path, data in (
            (self._sentence_cache_path(), self.sentence_to_labels),
            (self._report_cache_path(), self.report_hash_to_labels),
        ):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp, path)
            if self.verbose:
                logger.info("Saved %s (%d entries)", path, len(data))

    def _predict_sentences(self, sentences: List[str], batch_size: int) -> List[np.ndarray]:
        if not sentences:
            return []
        cached: List[Optional[np.ndarray]] = [self.sentence_to_labels.get(s) for s in sentences]
        missing = [i for i, v in enumerate(cached) if v is None]
        if not missing:
            return [np.asarray(v, dtype=np.int8) for v in cached]  # type: ignore

        to_process = [sentences[i] for i in missing]
        new_labels: List[np.ndarray] = []
        iterator = range(0, len(to_process), batch_size)
        if self.verbose:
            iterator = tqdm(iterator, desc="SRR-BERT sentence batches")
        with torch.no_grad():
            for i in iterator:
                batch = [clean_text(t) for t in to_process[i : i + batch_size]]
                inputs = self.tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                preds = (torch.sigmoid(logits) > self.threshold).to(torch.int8).cpu().numpy()
                new_labels.extend([row.copy() for row in preds])

        for i, orig_idx in enumerate(missing):
            cached[orig_idx] = new_labels[i]
            self.sentence_to_labels[to_process[i]] = new_labels[i]
        return [np.asarray(v, dtype=np.int8) for v in cached]  # type: ignore

    def get_labels_for_reports(
        self,
        reports: List[str],
        report_hashes: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        save_every: int = 2000,
    ) -> List[np.ndarray]:
        """Label reports via sentence tokenize + OR-merge; cache by report hash when provided."""
        from nltk.tokenize import sent_tokenize

        from medvqa.datasets.cxr_report_labeling.schemas import report_text_hash

        if batch_size is None:
            batch_size = self.default_batch_size
        if report_hashes is None:
            report_hashes = [report_text_hash(r) for r in reports]

        results: List[Optional[np.ndarray]] = [None] * len(reports)
        todo_indices: List[int] = []
        for i, h in enumerate(report_hashes):
            cached = self.report_hash_to_labels.get(h)
            if cached is not None:
                results[i] = np.asarray(cached, dtype=np.int8)
            else:
                todo_indices.append(i)

        if self.verbose:
            logger.info(
                "SRR-BERT report cache: %d hit / %d miss",
                len(reports) - len(todo_indices),
                len(todo_indices),
            )

        # Process misses in chunks to allow periodic cache saves
        for chunk_start in range(0, len(todo_indices), save_every):
            chunk = todo_indices[chunk_start : chunk_start + save_every]
            chunk_reports = [reports[i] for i in chunk]
            all_sentences: List[str] = []
            per_report_sents: List[List[str]] = []
            seen = set()
            unique_sents: List[str] = []
            for report in chunk_reports:
                sents = sent_tokenize(clean_text(report))
                per_report_sents.append(sents)
                for s in sents:
                    if s not in seen:
                        seen.add(s)
                        unique_sents.append(s)

            unique_labels = self._predict_sentences(unique_sents, batch_size=batch_size)
            sent2i = {s: i for i, s in enumerate(unique_sents)}
            for local_i, report_idx in enumerate(chunk):
                sents = per_report_sents[local_i]
                idxs = [sent2i[s] for s in sents if s in sent2i]
                merged = merge_labels([unique_labels[j] for j in idxs])
                results[report_idx] = merged
                self.report_hash_to_labels[report_hashes[report_idx]] = merged
            if self.cache_dir:
                self.save_caches()

        return [np.asarray(x, dtype=np.int8) for x in results]  # type: ignore
