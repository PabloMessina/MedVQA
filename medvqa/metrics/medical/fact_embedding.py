import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.models.huggingface_utils import (
    compute_text_embeddings_with_BiomedVLP_BioVilT,
    compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized,
)
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.files import get_cached_pickle_file, save_pickle
from medvqa.utils.hashing import hash_string

load_dotenv()
FACT_EMBEDDING_DEFAULT_MODEL_NAME = os.environ['FACT_EMBEDDING_DEFAULT_MODEL_NAME']
FACT_EMBEDDING_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH = os.environ['FACT_EMBEDDING_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH']

_ALLOWED_MODEL_NAMES = ['cxr-bert-specialized', 'biovil-t']

def _compute_score(gen_embeddings, gt_embeddings):
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
    score1 = np.mean(np.max(similarity_matrix, axis=1)) # mean of max similarity for each generated sentence
    score2 = np.mean(np.max(similarity_matrix, axis=0)) # mean of max similarity for each ground truth sentence
    score = 0.5 * (score1 + score2)
    return score

class FactEmbeddingScorer:

    def __init__(self, model_name=None, model_checkpoint_folder_path=None, verbose=False, device='GPU', batch_size=100, num_workers=4):
        if model_name is None:
            model_name = FACT_EMBEDDING_DEFAULT_MODEL_NAME
        if model_checkpoint_folder_path is None:
            model_checkpoint_folder_path = FACT_EMBEDDING_DEFAULT_MODEL_CHECKPOINT_FOLDER_PATH
        assert model_name in _ALLOWED_MODEL_NAMES
        assert device in ['GPU', 'CPU']
        self.model_name = model_name
        self.model_checkpoint_folder_path = model_checkpoint_folder_path
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        model_checkpoint_filepath = get_checkpoint_filepath(model_checkpoint_folder_path)
        model_hash = hash_string(model_checkpoint_filepath)
        self.cache_path = os.path.join(CACHE_DIR, f'fact_embedding_vectors_cache({model_name},{model_hash[0]},{model_hash[1]}).pkl')
        self.cache = get_cached_pickle_file(self.cache_path)
        self.verbose = verbose
        if self.cache is None:
            self.cache = dict()
        elif verbose:
            print(f'Cache successfully loaded from {self.cache_path}. It contains {len(self.cache)} entries.')

    def __call__(self, gen_reports, gt_reports, update_cache_on_disk=False, return_avg_score=True):
        assert type(gen_reports) == list
        assert type(gt_reports) == list
        assert len(gen_reports) == len(gt_reports)

        if self.verbose:
            print(f'(*) Computing fact embedding scores for {len(gen_reports)} reports ...')

        gen_sentence_hashes = [None] * len(gen_reports)
        gt_sentence_hashes = [None] * len(gt_reports)
        unlabeled_hashes_set = set()
        unlabeled_hashes = []
        unlabeled_sentences = []

        for i, (gen_report, gt_report) in enumerate(zip(gen_reports, gt_reports)):
            gen_sentences = sent_tokenize(gen_report)
            gt_sentences = sent_tokenize(gt_report)
            gen_sentence_hashes[i] = []
            gt_sentence_hashes[i] = []
            for s in gen_sentences:
                hash = hash_string(s)
                gen_sentence_hashes[i].append(hash)
                if hash not in self.cache and hash not in unlabeled_hashes_set:
                    unlabeled_hashes_set.add(hash)
                    unlabeled_hashes.append(hash)
                    unlabeled_sentences.append(s)
            for s in gt_sentences:
                hash = hash_string(s)
                gt_sentence_hashes[i].append(hash)
                if hash not in self.cache and hash not in unlabeled_hashes_set:
                    unlabeled_hashes_set.add(hash)
                    unlabeled_hashes.append(hash)
                    unlabeled_sentences.append(s)

        if len(unlabeled_sentences) > 0:
            if self.verbose:
                print(f'Computing embeddings for {len(unlabeled_sentences)} sentences not found in cache ...')
            if self.model_name == 'cxr-bert-specialized':
                embedding_func = compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized
            elif self.model_name == 'biovil-t':
                embedding_func = compute_text_embeddings_with_BiomedVLP_BioVilT
            else: assert False
            embeddings = embedding_func(
                texts=unlabeled_sentences, device=self.device, batch_size=self.batch_size, num_workers=self.num_workers,
                model_checkpoint_folder_path=self.model_checkpoint_folder_path
            )       
            print('embeddings.shape:', embeddings.shape)
            assert embeddings.shape[0] == len(unlabeled_sentences)
            # Update cache
            for hash, embedding in zip(unlabeled_hashes, embeddings):
                self.cache[hash] = embedding
            if update_cache_on_disk:
                save_pickle(self.cache, self.cache_path)
                if self.verbose:
                    print(f'Cache successfully updated and saved to {self.cache_path}')

        # Compute scores
        scores = np.zeros(len(gen_reports))
        for i in range(len(gen_reports)):
            gen_hashes = gen_sentence_hashes[i]
            gt_hashes = gt_sentence_hashes[i]
            gen_embeddings = np.array([self.cache[h] for h in gen_hashes])
            gt_embeddings = np.array([self.cache[h] for h in gt_hashes])
            scores[i] = _compute_score(gen_embeddings, gt_embeddings)
        if return_avg_score:
            return np.mean(scores)
        else:
            return scores