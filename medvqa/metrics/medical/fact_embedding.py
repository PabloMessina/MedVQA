import os
from dotenv import load_dotenv
import numpy as np
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity
from medvqa.models.huggingface_utils import (
    CachedT5FactExtractor,
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)

load_dotenv()
T5_FACT_EXTRACTOR_DEFAULT_MODEL_NAME = os.environ['T5_FACT_EXTRACTOR_DEFAULT_MODEL_NAME']
T5_FACT_EXTRACTOR_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH = os.environ['T5_FACT_EXTRACTOR_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH_v2']
FACT_EMBEDDING_DEFAULT_MODEL_NAME = os.environ['FACT_EMBEDDING_DEFAULT_MODEL_NAME']
FACT_EMBEDDING_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH = os.environ['FACT_EMBEDDING_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH_v5']

def _compute_metrics(gen_embeddings, gt_embeddings, threshold=0.7, only_soft_score=False):
    """
    Compute precision, recall, and F1 score for a given threshold. Also compute a soft score.
    """
    assert type(gen_embeddings) == np.ndarray 
    assert type(gt_embeddings) == np.ndarray

    if gen_embeddings.shape[0] == 0 or gt_embeddings.shape[0] == 0:
        if gen_embeddings.shape == gt_embeddings.shape:
            return 1.0
        else:
            return -1.0

    # Compute cosine similarity between all pairs of embeddings
    # shape: (gen_embeddings.shape[0], gt_embeddings.shape[0])
    similarity_matrix = cosine_similarity(gen_embeddings, gt_embeddings) # mathematically, values are in [-1, 1]
    # Compute score
    gen_max_sims = np.max(similarity_matrix, axis=1)
    gt_max_sims = np.max(similarity_matrix, axis=0)
    soft_score = 0.5 * (np.mean(gen_max_sims) + np.mean(gt_max_sims)) # mean of max similarity for each generated sentence and ground truth sentence
    if only_soft_score:
        return soft_score
    precision = np.mean(gen_max_sims >= threshold) # percentage of generated sentences that have at least one ground truth sentence with similarity >= threshold
    recall = np.mean(gt_max_sims >= threshold) # percentage of ground truth sentences that have at least one generated sentence with similarity >= threshold
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return soft_score, precision, recall, f1

_shared_report_embeddings = None

def _compute_metrics__multiprocressing(args):
    i, j, only_soft_score = args
    gen_embeddings = _shared_report_embeddings[i]
    gt_embeddings = _shared_report_embeddings[j]
    return _compute_metrics(gen_embeddings, gt_embeddings, only_soft_score=only_soft_score)

class FactEmbeddingScorer:

    def __init__(self,
                 t5_model_name=T5_FACT_EXTRACTOR_DEFAULT_MODEL_NAME,
                 t5_fact_extractor_checkpoint_folder_path=T5_FACT_EXTRACTOR_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH,
                 fact_embedding_model_name=FACT_EMBEDDING_DEFAULT_MODEL_NAME,
                 fact_embedding_model_checkpoint_folder_path=FACT_EMBEDDING_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH,
                 verbose=False, device='GPU', batch_size=200, num_workers=4,
                 threshold=0.7
                 ):
        
        # Sanity check
        assert os.path.exists(t5_fact_extractor_checkpoint_folder_path)
        assert fact_embedding_model_name in SupportedHuggingfaceMedicalBERTModels.get_all()
        assert device in ['GPU', 'CPU']
        
        # T5 Fact Extractor
        self.t5_fact_extractor = CachedT5FactExtractor(
            model_name=t5_model_name,
            model_checkpoint_folder_path=t5_fact_extractor_checkpoint_folder_path,
            device=device, batch_size=batch_size, num_workers=num_workers,
        )
        
        # Fact Embedding Extractor
        self.embedding_extractor = CachedTextEmbeddingExtractor(
            model_name=fact_embedding_model_name,
            model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
            device=device, batch_size=batch_size, num_workers=num_workers,
        )
        self.verbose = verbose
        self.threshold = threshold

    def __call__(self, gen_reports, gt_reports, update_cache_on_disk=False, return_avg_score=True, only_soft_score=False,
                 skip_cache=False):
        assert type(gen_reports) in [list, tuple, np.ndarray]
        assert type(gt_reports) in [list, tuple, np.ndarray]
        assert len(gen_reports) == len(gt_reports)

        if self.verbose:
            print(f'(*) Computing fact embedding scores for {len(gen_reports)} reports ...')

        facts = []
        f2idx = {}
            
        gen_facts_list = self.t5_fact_extractor(
            gen_reports, update_cache_on_disk=update_cache_on_disk, skip_cache=skip_cache) # list of list of facts
        gen_fact_idxs_list = [None] * len(gen_facts_list)
        for i, gen_facts in enumerate(gen_facts_list):
            for f in gen_facts:
                if f not in f2idx:
                    f2idx[f] = len(facts)
                    facts.append(f)
            gen_fact_idxs = [f2idx[f] for f in gen_facts]
            gen_fact_idxs_list[i] = gen_fact_idxs
        
        gt_facts_list = self.t5_fact_extractor(
            gt_reports, update_cache_on_disk=update_cache_on_disk, skip_cache=skip_cache) # list of list of facts
        gt_fact_idxs_list = [None] * len(gt_facts_list)
        for i, gt_facts in enumerate(gt_facts_list):
            for f in gt_facts:
                if f not in f2idx:
                    f2idx[f] = len(facts)
                    facts.append(f)
            gt_fact_idxs = [f2idx[f] for f in gt_facts]
            gt_fact_idxs_list[i] = gt_fact_idxs

        embeddings = self.embedding_extractor.compute_text_embeddings(facts, update_cache_on_disk=update_cache_on_disk)
        assert embeddings.shape[0] == len(facts)

        # Compute scores
        if only_soft_score:
            scores = np.zeros(len(gen_reports), dtype=np.float32)
            for i in range(len(gen_reports)):
                gen_embeddings = embeddings[gen_fact_idxs_list[i]]
                gt_embeddings = embeddings[gt_fact_idxs_list[i]]
                scores[i] = _compute_metrics(gen_embeddings, gt_embeddings, threshold=self.threshold, only_soft_score=True)
            if return_avg_score:
                return np.mean(scores)
            else:
                return scores
        else:
            scores = np.zeros((4, len(gen_reports)), dtype=np.float32) # soft_score, precision, recall, f1
            for i in range(len(gen_reports)):
                gen_embeddings = embeddings[gen_fact_idxs_list[i]]
                gt_embeddings = embeddings[gt_fact_idxs_list[i]]
                scores[:, i] = _compute_metrics(gen_embeddings, gt_embeddings, threshold=self.threshold)
            if return_avg_score:
                return np.mean(scores, axis=1) # soft_score, precision, recall, f1
            else:
                return scores
        
    def compute_pairwise_scores(self, reports, update_cache_on_disk=False, only_soft_score=False):
        assert type(reports) in [list, tuple, np.ndarray]

        if self.verbose:
            print(f'(*) Computing fact embedding scores for all pairs of {len(reports)} reports ...')

        facts = []
        f2idx = {}

        facts_list = self.t5_fact_extractor(
              reports, update_cache_on_disk=update_cache_on_disk) # list of list of facts
        fact_idxs_list = [None] * len(facts_list)
        for i, _facts in enumerate(facts_list):
            for f in _facts:
                if f not in f2idx:
                    f2idx[f] = len(facts)
                    facts.append(f)
            fact_idxs = [f2idx[f] for f in _facts]
            fact_idxs_list[i] = fact_idxs

        embeddings = self.embedding_extractor.compute_text_embeddings(facts, update_cache_on_disk=update_cache_on_disk)
        assert embeddings.shape[0] == len(facts)

        # Compute scores for all pairs
        args = [(i, j, only_soft_score) for i in range(len(reports)) for j in range(i, len(reports))]
        global _shared_report_embeddings
        _shared_report_embeddings = [embeddings[fact_idxs_list[i]] for i in range(len(reports))]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            scores = pool.map(_compute_metrics__multiprocressing, args)
        if only_soft_score:
            score_matrix = np.zeros((len(reports), len(reports)), dtype=np.float32)
            for (i, j, _), score in zip(args, scores):
                score_matrix[i, j] = score
                score_matrix[j, i] = score
        else:
            score_matrix = np.zeros((len(reports), len(reports), 4), dtype=np.float32) # soft_score, precision, recall, f1
            for (i, j, _), score in zip(args, scores):
                s, p, r, f1 = score
                score_matrix[i, j] = [s, p, r, f1]
                score_matrix[j, i] = [s, r, p, f1] # swap precision and recall

        return score_matrix