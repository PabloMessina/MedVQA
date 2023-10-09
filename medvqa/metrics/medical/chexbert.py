import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from f1chexbert import F1CheXbert
from f1chexbert.f1chexbert import tokenize, generate_attention_masks
from nltk.tokenize import sent_tokenize
from medvqa.utils.common import CACHE_DIR
from medvqa.utils.constants import CHEXBERT_LABELS
from medvqa.utils.files import get_cached_pickle_file, save_pickle
from medvqa.utils.hashing import hash_string

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

    def __init__(self, device=None, verbose=False):
        super().__init__(device=device)
        self.cache_path = os.path.join(CACHE_DIR, 'chexbert_labeler_cache.pkl')
        self.cache = get_cached_pickle_file(self.cache_path)
        self.verbose = verbose
        if self.cache is None:
            self.cache = dict()
        elif verbose:
            print(f'Cache successfully loaded from {self.cache_path}')

    def get_labels(self, texts, update_cache_on_disk=False):

        if self.verbose:
            print(f'(*) Chexbert: labeling {len(texts)} texts ...')

        output_labels = [None] * len(texts)
        dirty_count = 0
        
        for i, text in tqdm(enumerate(texts), mininterval=2):
            text_hash = hash_string(text)
            if text_hash in self.cache:
                output_labels[i] = self.cache[text_hash]
                continue
            sentences = sent_tokenize(text)
            sentence_labels = []
            for sentence in sentences:
                hash = hash_string(sentence)
                labels = self.cache.get(hash, None)
                if labels is None:
                    labels = self.get_label(sentence)
                    self.cache[hash] = labels
                    dirty_count += 1
                sentence_labels.append(labels)
            output_labels[i] = merge_labels(sentence_labels)
            self.cache[text_hash] = output_labels[i]            

        if dirty_count > 0:
            if self.verbose:
                print(f'Done labeling: {dirty_count} new labels found and cached')
            if update_cache_on_disk:
                save_pickle(self.cache, self.cache_path)
                if self.verbose:
                    print(f'Cache successfully updated and saved to {self.cache_path}')
        elif self.verbose:
            print('All labels found in cache, no need to invoke chexbert labeler')
        
        return np.array(output_labels)
    
    def get_embedding(self, text):
        assert type(text) == str
        text = pd.Series([text])
        out = tokenize(text, self.tokenizer)
        batch = torch.LongTensor([o for o in out])
        src_len = [b.shape[0] for b in batch]
        attn_mask = generate_attention_masks(batch, src_len, self.device)
        final_hidden = self.model.bert(batch.to(self.device), attention_mask=attn_mask)[0]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        return cls_hidden.cpu().detach().numpy().squeeze()
    
    def get_embeddings(self, texts):
        assert type(texts) == list or type(texts) == np.ndarray
        return np.array([self.get_embedding(text) for text in tqdm(texts, mininterval=2)])