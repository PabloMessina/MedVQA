import os
import re
import torch
import logging
import numpy as np
from tqdm import tqdm
from f1chexbert import F1CheXbert
from nltk.tokenize import sent_tokenize
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.constants import CHEXBERT_LABELS
from medvqa.utils.files_utils import get_cached_pickle_file, save_pickle
from medvqa.utils.hashing_utils import hash_string

logger = logging.getLogger(__name__)


def merge_labels(labels_list):        
    merged = np.zeros((len(CHEXBERT_LABELS),), np.int8)
    merged[-1] = 1 # default to no findings
    for labels in labels_list:
        if labels[-1] == 0: # there is a finding
            merged[-1] = 0
        for i in range(0, len(labels)-1): # iterate over all labels except the last one
            if labels[i] == 1:
                merged[i] = 1
    return merged    


class CheXbertLabeler(F1CheXbert):
    """
    An enhanced version of F1CheXbert that processes reports in batches
    for significantly faster inference, especially on GPU.

    Inherits from the original F1CheXbert, reusing its model loading,
    tokenizer, and basic structure. Overrides the forward method to
    implement batching.
    """
    def __init__(self, default_batch_size: int = 32, verbose: bool = False, **kwargs):
        """
        Initializes the F1CheXbertBatch instance.

        Args:
            default_batch_size: The default batch size to use if not specified
                                in the forward call. Defaults to 32.
            verbose: If True, enables verbose logging. Defaults to False.
            **kwargs: Arguments to pass to the parent F1CheXbert constructor
                      (e.g., refs_filename, hyps_filename, device).
        """
        # Initialize the parent class (loads model, tokenizer, etc.)
        super().__init__(**kwargs)
        self.default_batch_size = default_batch_size
        self.verbose = verbose
        self.model.eval() # Ensure model is in eval mode
        if self.verbose:
            logger.info(f"CheXbertLabeler initialized. Using device: {self.device}")
            logger.info(f"Default batch size: {self.default_batch_size}")
        
        self.labels_cache_path = os.path.join(CACHE_DIR, 'chexbert_labeler_cache.pkl')
        self.labels_cache = None
        if os.path.exists(self.labels_cache_path):
            self.labels_cache = get_cached_pickle_file(self.labels_cache_path)
        if self.labels_cache is None:
            self.labels_cache = dict()
        elif self.verbose:
            logger.info(f'Labels cache successfully loaded from {self.labels_cache_path}')

        self.embeddings_cache_path = os.path.join(CACHE_DIR, 'chexbert_embeddings_cache.pkl')
        self.embeddings_cache = None
        if os.path.exists(self.embeddings_cache_path):
            self.embeddings_cache = get_cached_pickle_file(self.embeddings_cache_path)
        if self.embeddings_cache is None:
            self.embeddings_cache = dict()
        elif self.verbose:
            logger.info(f'Embeddings cache successfully loaded from {self.embeddings_cache_path}')

    def get_labels(self,
                   reports: List[str],
                   batch_size: int = None,
                   mode="rrg") -> List[List[Union[int, str]]]:
        """
        Processes a list of reports in batches to generate CheXbert labels.

        Args:
            reports: A list of report strings.
            batch_size: The number of reports to process in each batch.
                        Uses instance default if None.
            mode: The labeling mode ('rrg' or 'classification'). Defaults to 'rrg'.

        Returns:
            A list of lists, where each inner list contains the 14 labels
            for the corresponding input report.
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        if not reports:
            return []

        all_labels = []
        self.model.eval()  # Ensure model is in eval mode

        if self.verbose:
            iterator = tqdm(range(0, len(reports), batch_size), desc="Processing batches")
        else:
            iterator = range(0, len(reports), batch_size)

        with torch.no_grad(): # Disable gradient calculations for inference
            for i in iterator:
                batch_reports = reports[i : i + batch_size]

                # Clean reports (simple cleaning)
                cleaned_batch = []
                for report in batch_reports:
                    if not isinstance(report, str): report = str(report) # Handle non-strings
                    report = report.strip().replace('\n', ' ')
                    report = re.sub(r'\s+', ' ', report).strip() # Replace multiple spaces
                    cleaned_batch.append(report if report else " ") # Handle empty strings

                # Tokenize batch using the tokenizer from the parent class
                inputs = self.tokenizer(
                    cleaned_batch,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                # Move tensors to the device specified in the parent class
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                # Get model outputs using the model from the parent class
                batch_outputs = self.model(input_ids, attention_mask=attention_mask)

                # Process outputs for the batch
                num_reports_in_batch = input_ids.shape[0]
                batch_results = [[] for _ in range(num_reports_in_batch)]

                for task_output in batch_outputs: # Iterate over each task output (14 classes)
                    predictions = task_output.argmax(dim=1) # shape: (num_reports_in_batch, num_classes)
                    for report_idx in range(num_reports_in_batch):
                        pred_class = predictions[report_idx].item()
                        if mode == "rrg":
                            label = 1 if pred_class == 1 or pred_class == 3 else 0
                            batch_results[report_idx].append(label)
                        elif mode == "classification":
                            # Map to -1, 0, 1, ''
                            if pred_class == 0: label = ''
                            elif pred_class == 1: label = 1
                            elif pred_class == 2: label = 0
                            elif pred_class == 3: label = -1
                            else: label = ''
                            batch_results[report_idx].append(label)
                        else:
                            raise NotImplementedError(f"Mode '{mode}' not implemented.")
                all_labels.extend(batch_results)

        return all_labels
    
    def _get_labels_with_cache(self, texts: List[str], batch_size: int = None,
                               update_cache_on_disk: bool = False) -> np.ndarray:
        hashes = [hash_string(x) for x in texts]
        labels = np.zeros((len(texts), len(CHEXBERT_LABELS)), dtype=np.int8)
        uncached_indices = []
        for i, h in enumerate(hashes):
            try:
                labels[i] = self.labels_cache[h]
            except KeyError:
                uncached_indices.append(i)
        if uncached_indices:
            if self.verbose:
                logger.info(f"Found {len(uncached_indices)} uncached texts.")
            uncached_texts = [texts[i] for i in uncached_indices]
            if self.verbose:
                logger.info("Generating CheXbert labels for uncached texts (batch)...")
            uncached_labels = self.get_labels(uncached_texts, batch_size=batch_size, mode="rrg")
            for i, labels_ in zip(uncached_indices, uncached_labels):
                labels[i] = labels_
                self.labels_cache[hashes[i]] = labels
            if update_cache_on_disk:
                save_pickle(self.labels_cache, self.labels_cache_path)
                if self.verbose:
                    logger.info(f"Cache successfully updated and saved to {self.labels_cache_path}")
        else:
            if self.verbose:
                logger.info("All texts found in cache, no need to invoke CheXbert labeler.")
        return labels

    
    def get_embeddings(
        self, texts: List[str], batch_size: int = None, update_cache_on_disk: bool = False
    ) -> np.ndarray:
        """
        Processes a list of text strings in batches to generate their embeddings
        using the BERT encoder from the CheXbert model. The [CLS] token
        embedding is used as the sentence embedding. If the embeddings
        already exist in the cache, they are retrieved from there.

        Args:
            texts: A list of text strings.
            batch_size: The number of texts to process in each batch.
                        Uses instance default if None.
            update_cache_on_disk: If True, updates the cache on disk after
                                    generating embeddings. Defaults to False.

        Returns:
            A numpy array of embeddings, where each row corresponds to the
            embedding of the input text string.
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        if not texts:
            return np.array([])
        
        text_hashes = [hash_string(x) for x in texts]
        embeddings = np.zeros((len(texts), 768), dtype=np.float32)
        uncached_indices = []
        for i, h in enumerate(text_hashes):
            try:
                embeddings[i] = self.embeddings_cache[h]
            except KeyError:
                uncached_indices.append(i)
        if uncached_indices:
            if self.verbose:
                logger.info(f"Found {len(uncached_indices)} uncached texts.")
            
            self.model.eval()  # Ensure model is in eval mode

            if self.verbose:
                iterator = tqdm(range(0, len(uncached_indices), batch_size), desc="Generating embeddings")
            else:
                iterator = range(0, len(uncached_indices), batch_size)

            with torch.no_grad():  # Disable gradient calculations for inference
                for i in iterator:
                    batch_indices = uncached_indices[i : i + batch_size]
                    batch_texts = [texts[j] for j in batch_indices]

                    # Clean texts (simple cleaning)
                    cleaned_batch = []
                    for text in batch_texts:
                        if not isinstance(text, str):
                            text = str(text)  # Handle non-strings
                        text = text.strip().replace("\n", " ")
                        text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces
                        cleaned_batch.append(text if text else " ")  # Handle empty strings

                    # Tokenize batch using the tokenizer from the parent class
                    # Return tensors for PyTorch
                    inputs = self.tokenizer(
                        cleaned_batch,
                        padding="longest",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )

                    # Move tensors to the device specified in the parent class
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)

                    # Get the outputs from the BERT encoder.
                    # The model attribute in CheXbert is the full model which
                    # includes the BERT encoder and the task-specific heads.
                    # We need to access the BERT encoder part.
                    bert_output = self.model.bert(
                        input_ids, attention_mask=attention_mask
                    )[0]  # [0] is the last hidden state

                    # Extract the [CLS] token embedding (first token)
                    cls_embeddings = bert_output[:, 0, :].squeeze(dim=1)

                    # Move embeddings to CPU and convert to numpy
                    cls_embeddings = cls_embeddings.cpu().numpy()

                    # Store the embeddings in the appropriate indices
                    for j, idx in enumerate(batch_indices):
                        embeddings[idx] = cls_embeddings[j]
                        self.embeddings_cache[text_hashes[idx]] = cls_embeddings[j]

            if update_cache_on_disk:
                save_pickle(self.embeddings_cache, self.embeddings_cache_path)
                if self.verbose:
                    logger.info(f"Cache successfully updated and saved to {self.embeddings_cache_path}")

        else:
            if self.verbose:
                logger.info("All texts found in cache, no need to invoke CheXbert labeler.")
        return embeddings

    def compute_cosine_similarity(
        self, hyps: List[str], refs: List[str], batch_size: int = None
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Computes the average sentence-level cosine similarity between hypothesis
        and reference reports.

        Steps:
        1. Split both hypothesis and reference reports into sentences.
        2. Identify all unique sentences across all reports.
        3. Generate embeddings for all unique sentences in batches.
        4. For each hypothesis/reference pair:
            a. Get embeddings for their respective sentences.
            b. Compute a pairwise cosine similarity matrix.
            c. Calculate the mean of row-wise maximums and the mean of
               column-wise maximums.
            d. The average of these two means is the score for the pair.
        5. Calculate the overall average score across all pairs.

        Args:
            hyps: A list of hypothesis report strings.
            refs: A list of reference report strings.
            batch_size: Batch size for embedding generation. Uses instance
                        default if None.

        Returns:
            A dictionary containing:
                - overall_mean_similarity: The average similarity across all pairs.
                - per_pair_similarity: A numpy array of similarity scores for each pair.
        """
        if len(hyps) != len(refs):
            raise ValueError(
                "The number of hypothesis reports and reference reports must be the same."
            )
        if batch_size is None:
            batch_size = self.default_batch_size

        if self.verbose:
            logger.info("Splitting reports into sentences for cosine similarity...")
        hyps_sentences = [sent_tokenize(report) for report in hyps]
        refs_sentences = [sent_tokenize(report) for report in refs]

        unique_sentences = set()
        for sentences in hyps_sentences:
            unique_sentences.update(sentences)
        for sentences in refs_sentences:
            unique_sentences.update(sentences)
        unique_sentences = list(unique_sentences)

        if self.verbose:
            logger.info(f"Unique sentences found: {len(unique_sentences)}")

        if not unique_sentences:
            if self.verbose:
                logger.warning(
                    "No unique sentences found. Returning zero similarity."
                )
            return {
                "overall_mean_similarity": 0.0,
                "per_pair_similarity": np.zeros(len(hyps), dtype=np.float32),
            }

        # Compute embeddings for unique sentences
        if self.verbose:
            logger.info("Generating embeddings for unique sentences (batch)...")
        unique_sentence_embeddings = self.get_embeddings(
            unique_sentences, batch_size=batch_size
        )
        sentence2embedding_index = {
            sentence: i for i, sentence in enumerate(unique_sentences)
        }

        # Compute similarity for each hyp/ref pair
        per_pair_similarity = []
        if self.verbose:
            pair_iterator = tqdm(
                zip(hyps_sentences, refs_sentences),
                total=len(hyps),
                desc="Computing pair-wise cosine similarity",
            )
        else:
            pair_iterator = zip(hyps_sentences, refs_sentences)

        for hyp_sents, ref_sents in pair_iterator:
            if not hyp_sents or not ref_sents:
                # Handle cases with empty sentences in either hyp or ref
                per_pair_similarity.append(0.0)
                continue

            # Get embeddings for the current hyp and ref sentences
            hyp_embeddings = np.array(
                [
                    unique_sentence_embeddings[sentence2embedding_index[s]]
                    for s in hyp_sents
                ]
            )
            ref_embeddings = np.array(
                [
                    unique_sentence_embeddings[sentence2embedding_index[s]]
                    for s in ref_sents
                ]
            )

            # Compute pairwise cosine similarity matrix
            # Shape: (num_hyp_sentences, num_ref_sentences)
            similarity_matrix = cosine_similarity(hyp_embeddings, ref_embeddings)

            # Compute mean of row-wise maximums (each hyp sentence's max similarity to any ref sentence)
            mean_row_max = np.mean(np.max(similarity_matrix, axis=1)) if similarity_matrix.shape[1] > 0 else 0.0

            # Compute mean of column-wise maximums (each ref sentence's max similarity to any hyp sentence)
            mean_col_max = np.mean(np.max(similarity_matrix, axis=0)) if similarity_matrix.shape[0] > 0 else 0.0

            # The average of these two metrics is the score for this pair
            pair_score = (mean_row_max + mean_col_max) / 2.0
            per_pair_similarity.append(pair_score)

        per_pair_similarity_np = np.array(per_pair_similarity, dtype=np.float32)
        overall_mean_similarity = np.mean(per_pair_similarity_np).item()

        if self.verbose:
            logger.info("Cosine similarity calculation complete.")

        return {
            "mean_similarity": overall_mean_similarity,
            "per_pair_similarity": per_pair_similarity_np,
        }

    # Override the forward method
    def forward(self, hyps: List[str], refs: List[str], batch_size: int = None,
                split_into_sentences_first: bool = True,
                update_cache_on_disk: bool = False) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculates F1CheXbert metrics using batch processing for label generation.

        Args:
            hyps: A list of hypothesis report strings.
            refs: A list of reference report strings.
            batch_size: Batch size for CheXbert label generation. Uses instance
                        default if None.
            split_into_sentences_first: If True, splits reports into sentences
                                        before processing, predicting labels for
                                        each sentence, and then merging them.
                                        This is useful for long reports with
                                        multiple findings. Defaults to True.
            update_cache_on_disk: If True, updates the cache on disk after
                                        generating labels. Defaults to False.

        Returns:
            A dictionary containing:
                - accuracy: Overall accuracy of the predictions.
                - per_element_accuracy: Accuracy for each pair of reference and hypothesis.
                - classification_report: Detailed classification report.
                - ref_labels: Reference labels as a numpy array.
                - hyp_labels: Hypothesis labels as a numpy array.
        """
        if batch_size is None:
            batch_size = self.default_batch_size

        # --- Preprocessing (if split_into_sentences_first is True) ---
        if split_into_sentences_first:
            if self.verbose:
                logger.info("Splitting reports into sentences...")
            hyps_sentences = [sent_tokenize(report) for report in hyps]
            refs_sentences = [sent_tokenize(report) for report in refs]
            unique_sentences = set()
            for sentences in hyps_sentences:
                unique_sentences.update(sentences)
            for sentences in refs_sentences:
                unique_sentences.update(sentences)
            unique_sentences = list(unique_sentences)
            if self.verbose:
                logger.info(f"Unique sentences found: {len(unique_sentences)}")
            sentence2index = {sentence: i for i, sentence in enumerate(unique_sentences)}
            unique_labels = self._get_labels_with_cache(unique_sentences, batch_size=batch_size,
                                                        update_cache_on_disk=update_cache_on_disk)

        # --- Reference Labels ---
        if split_into_sentences_first:
            refs_chexbert = []
            for sentences in refs_sentences:
                s_idxs = [sentence2index[s] for s in sentences if s in sentence2index]
                labels = merge_labels([unique_labels[i] for i in s_idxs])
                refs_chexbert.append(labels)
        else:
            if self.verbose:
                logger.info("Generating CheXbert labels for references (batch)...")
            # refs_chexbert = self.get_labels(refs, batch_size=batch_size, mode="rrg")
            refs_chexbert = self._get_labels_with_cache(refs, batch_size=batch_size,
                                                        update_cache_on_disk=update_cache_on_disk)

        # --- Hypothesis Labels ---
        if split_into_sentences_first:
            hyps_chexbert = []
            for sentences in hyps_sentences:
                s_idxs = [sentence2index[s] for s in sentences if s in sentence2index]
                labels = merge_labels([unique_labels[i] for i in s_idxs])
                hyps_chexbert.append(labels)
        else:
            if self.verbose:
                logger.info("Generating CheXbert labels for hypotheses (batch)...")
            # hyps_chexbert = self.get_labels(hyps, batch_size=batch_size, mode="rrg")
            hyps_chexbert = self._get_labels_with_cache(hyps, batch_size=batch_size,
                                                        update_cache_on_disk=update_cache_on_disk)

        # --- Calculations ---
        if self.verbose:
            logger.info("Calculating metrics...")
        refs_chexbert_np = np.array(refs_chexbert)
        hyps_chexbert_np = np.array(hyps_chexbert)

        # Basic shape validation
        if refs_chexbert_np.shape != hyps_chexbert_np.shape:
            raise ValueError(f"Shape mismatch between reference labels ({refs_chexbert_np.shape}) and hypothesis labels ({hyps_chexbert_np.shape}).")
        if refs_chexbert_np.ndim != 2 or refs_chexbert_np.shape[1] != len(self.target_names):
            raise ValueError(f"Unexpected shape for reference labels: {refs_chexbert_np.shape}. Expected (num_reports, {len(self.target_names)}).")

        # Accuracy and per-element accuracy
        accuracy = np.mean(refs_chexbert_np == hyps_chexbert_np)
        pe_accuracy = np.mean(refs_chexbert_np == hyps_chexbert_np, axis=1).astype(np.float32)

        # Classification report
        cr = classification_report(refs_chexbert_np, hyps_chexbert_np, target_names=self.target_names, output_dict=True, zero_division=0)
        if self.verbose:
            logger.info("Metrics calculation complete.")

        return {
            'accuracy': accuracy.item(),
            'per_element_accuracy': pe_accuracy,
            'classification_report': cr,
            'ref_labels': refs_chexbert_np,
            'hyp_labels': hyps_chexbert_np,
        }