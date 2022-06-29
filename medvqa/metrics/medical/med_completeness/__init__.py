from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from medvqa.metrics.dataset_aware_metric import DatasetAwareMetric
from medvqa.utils.common import SOURCE_DIR, CACHE_DIR
from medvqa.utils.files import load_pickle
import os

MEDICAL_TERMS_PATH = os.path.join(SOURCE_DIR, 'medvqa', 'metrics', 'medical',
                                 'med_completeness', 'medical_terms.txt')
MEDICAL_SYNONYMS_PATH = os.path.join(SOURCE_DIR, 'medvqa', 'metrics', 'medical',
                                 'med_completeness', 'medical_synonyms.txt')
MEDICAL_TERMS_WEIGHTS_PATH = os.path.join(CACHE_DIR, 'medical_terms_weights.pkl')

class MedicalCompletenessBase:

    def _load_medical_terms(self, tokenizer):
        with open(MEDICAL_TERMS_PATH) as f:
            medical_terms = [line.strip() for line in f.readlines()]            
            assert len(set(medical_terms)) == len(medical_terms)
        self.medical_ids = set(tokenizer.token2id[x] for x in medical_terms if x in tokenizer.vocab)
    
    def _load_medical_synonyms(self, tokenizer):
        with open(MEDICAL_SYNONYMS_PATH) as f:
            self.medical_synonyms = [line.strip().split() for line in f.readlines()]
        self.id2synonym = { x:x for x in self.medical_ids }
        for row in self.medical_synonyms:
            synonym_ids = [tokenizer.token2id[x] for x in row if x in tokenizer.vocab]
            if len(synonym_ids) >= 2:        
                for i in range(1, len(row)):
                    self.id2synonym[synonym_ids[i]] = synonym_ids[0]
    
    def score(self, gt_s, gen_s):

        # ground truth sequence
        gt_s = [self.id2synonym[x] for x in gt_s if x in self.id2synonym]
        L = min(4, len(gt_s))
        gt_count = [dict() for _ in range(L)]
        for k in range(L):
            for i in range(len(gt_s) - k):                
                key = gt_s[i] if k == 0 else tuple(gt_s[i:i+k+1])                
                gt_count[k][key] = gt_count[k].get(key, 0) + 1
        
        # generated sequence
        gen_s = [self.id2synonym[x] for x in gen_s if x in self.id2synonym]
        gen_count = [dict() for _ in range(L)]
        inter_size = [0] * L
        for i in range(len(gen_s)):
            for k in range(min(L, len(gen_s) - i)):                
                key = gen_s[i] if k == 0 else tuple(gen_s[i:i+k+1])                
                gen_c = gen_count[k][key] = gen_count[k].get(key, 0) + 1
                gt_c = gt_count[k].get(key, 0)
                if gen_c <= gt_c:
                    inter_size[k] += 1
                elif gt_c == 0:
                    break        

        # final score
        score = 0
        for i in range(L):
            gt_size = len(gt_s) - i
            gen_size = max(len(gen_s) - i, 0)
            assert gt_size >= 0
            prec = inter_size[i] / gen_size if gen_size > 0 else 0
            rec = inter_size[i] / gt_size if gt_size > 0 else 0
            score += 2 * prec * rec / (prec + rec) if prec > 0 or rec > 0 else 0
        return score / L if L > 0 else 0
    
    def score__debug(self, gt_s, gen_s, tokenizer, verbose=True):

        # ground truth sequence
        print('------------------')
        print('ground truth')
        print('------------------')
        gt_ignored = [x for x in gt_s if x not in self.id2synonym]
        gt_s = [self.id2synonym[x] for x in gt_s if x in self.id2synonym]
        L = min(4, len(gt_s))
        gt_count = [dict() for _ in range(L)]
        for k in range(L):
            for i in range(len(gt_s) - k):                
                key = gt_s[i] if k == 0 else tuple(gt_s[i:i+k+1])                
                gt_count[k][key] = gt_count[k].get(key, 0) + 1
        print(f"1-grams ({len(gt_s)}): {', '.join(tokenizer.id2token[x] for x in gt_s)}")
        print(f"ignored ({len(gt_ignored)}): {', '.join(tokenizer.id2token[x] for x in gt_ignored)}")
        if verbose:
            print('gt_count = ', gt_count)
        
        # generated sequence
        print('------------------')
        print('generated')
        print('------------------')
        gen_ignored = [x for x in gen_s if x not in self.id2synonym]
        gen_s = [self.id2synonym[x] for x in gen_s if x in self.id2synonym]
        gen_count = [dict() for _ in range(L)]
        inter_size = [0] * L
        for i in range(len(gen_s)):
            for k in range(min(L, len(gen_s) - i)):                
                key = gen_s[i] if k == 0 else tuple(gen_s[i:i+k+1])                
                gen_c = gen_count[k][key] = gen_count[k].get(key, 0) + 1
                gt_c = gt_count[k].get(key, 0)
                if verbose:
                    print(f'key = {key}, gt_c = {gt_c}, gen_c = {gen_c}')
                if gen_c <= gt_c:
                    inter_size[k] += 1
                elif gt_c == 0:
                    break
        print(f"1-grams ({len(gen_s)}): {', '.join(tokenizer.id2token[x] for x in gen_s)}")
        print(f"ignored ({len(gen_ignored)}): {', '.join(tokenizer.id2token[x] for x in gen_ignored)}")
        if verbose:
            print('gen_count = ', gen_count)

        print("======")
        print('inter_size =', inter_size)

        # final score
        score = 0
        for i in range(L):
            gt_size = len(gt_s) - i
            gen_size = max(len(gen_s) - i, 0)
            assert gt_size >= 0
            prec = inter_size[i] / gen_size if gen_size > 0 else 0
            rec = inter_size[i] / gt_size if gt_size > 0 else 0
            score += 2 * prec * rec / (prec + rec) if prec > 0 or rec > 0 else 0
        return score / L if L > 0 else 0

class MedicalCompleteness(MedicalCompletenessBase, Metric):

    def __init__(self, tokenizer, output_transform=lambda x: x, device=None, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        self._load_medical_terms(tokenizer)
        self._load_medical_synonyms(tokenizer)
        if record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()

    def update(self, output):
        pred_sentences, gt_sentences = output
        for pred_s, gt_s in zip(pred_sentences, gt_sentences):
            score = self.score(gt_s, pred_s)
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += len(pred_sentences)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Medical Completness needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count


class WeightedMedicalCompletenessBase(MedicalCompletenessBase):
    
    def _load_weights(self, tokenizer):
        terms2weight = load_pickle(MEDICAL_TERMS_WEIGHTS_PATH)
        assert len(terms2weight) == 4
        ids2weight = [dict() for _ in range(4)]
        for k in range(4):
            for key in terms2weight[k].keys():
                try:
                    key2 = tokenizer.token2id[key] if k == 0 else tuple(tokenizer.token2id[x] for x in key)
                    ids2weight[k][key2] = terms2weight[k][key]
                except KeyError:
                    continue
        self.ids2weight = ids2weight
    
    def score(self, gt_s, gen_s):
        
        # ground truth sequence
        gt_s = [self.id2synonym[x] for x in gt_s if x in self.id2synonym]
        L = min(4, len(gt_s))
        gt_tot_weights = [0] * L
        gt_freqs = [dict() for _ in range(L)]
        for k in range(L):
            for i in range(len(gt_s) - k):
                if k == 0:
                    key = gt_s[i]
                else:
                    key = tuple(gt_s[i:i+k+1])
                gt_tot_weights[k] += self.ids2weight[k].get(key, 0)
                gt_freqs[k][key] = gt_freqs[k].get(key, 0) + 1
        
        # generated sequence
        gen_s = [self.id2synonym[x] for x in gen_s if x in self.id2synonym]
        gen_tot_weights = [0] * L
        gen_freqs = [dict() for _ in range(L)]
        inter_weights = [0] * L
        for k in range(min(L, len(gen_s))):
            for i in range(len(gen_s) - k):
                if k == 0:
                    key = gen_s[i]
                else:
                    key = tuple(gen_s[i:i+k+1])
                w = self.ids2weight[k].get(key, 0)
                gen_tot_weights[k] += w
                f = gen_freqs[k][key] = gen_freqs[k].get(key, 0) + 1
                if f <= gt_freqs[k].get(key, 0):
                    inter_weights[k] += w
        
        # final score
        # print(gt_tot_weights)
        # print(gen_tot_weights)
        # print(inter_weights)
        score = 0
        for i in range(L):
            prec = inter_weights[i] / gen_tot_weights[i] if gen_tot_weights[i] > 0 else 0
            rec = inter_weights[i] / gt_tot_weights[i] if gt_tot_weights[i] > 0 else 0
            score += 2 * prec * rec / (prec + rec) if prec > 0 or rec > 0 else 0
        return score / L if L > 0 else 0

class WeightedMedicalCompleteness(WeightedMedicalCompletenessBase, Metric):

    def __init__(self, tokenizer, output_transform=lambda x: x, device=None, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        self._load_medical_terms(tokenizer)
        self._load_medical_synonyms(tokenizer)
        self._load_weights(tokenizer)
        if record_scores:
            self._scores = []
        super().__init__(output_transform=output_transform, device=device)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()
        super().reset()

    def update(self, output):
        pred_sentences, gt_sentences = output
        for pred_s, gt_s in zip(pred_sentences, gt_sentences):
            score = self.score(gt_s, pred_s)
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += len(pred_sentences)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Weighted Med. Comp. needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count

class DatasetAwareWeightedMedicalCompleteness(WeightedMedicalCompletenessBase, DatasetAwareMetric):

    def __init__(self, tokenizer, output_transform, allowed_dataset_ids, record_scores=False):
        self._acc_score = 0
        self._count = 0
        self.record_scores = record_scores
        self._load_medical_terms(tokenizer)
        self._load_medical_synonyms(tokenizer)
        self._load_weights(tokenizer)
        if record_scores:
            self._scores = []
        super().__init__(output_transform, allowed_dataset_ids)
    
    def reset(self):
        self._acc_score = 0
        self._count = 0
        if self.record_scores:
            self._scores.clear()

    def update(self, output):
        pred_sentences, gt_sentences = output
        for pred_s, gt_s in zip(pred_sentences, gt_sentences):
            score = self.score(gt_s, pred_s)
            self._acc_score += score
            if self.record_scores:
                self._scores.append(score)
        self._count += len(pred_sentences)

    def compute(self):
        if self._count == 0:
            raise NotComputableError('Weighted Med. Comp. needs at least one example before it can be computed.')
        if self.record_scores:
            return self._scores
        return self._acc_score / self._count
