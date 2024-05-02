import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.cluster import KMeans
from medvqa.datasets.text_data_utils import create_text_dataset_and_dataloader, remove_consecutive_repeated_words_from_text, sentence_tokenize_texts_in_parallel
from medvqa.models.checkpoint import get_checkpoint_filepath
from medvqa.models.common import load_model_state_dict
from medvqa.utils.common import LARGE_FAST_CACHE_DIR
from medvqa.utils.files import get_cached_pickle_file, get_file_path_with_hashing_if_too_long, load_jsonl, load_pickle, save_pickle
from medvqa.utils.hashing import compute_hashes_in_parallel, hash_string, hash_string_list, update_hash
from medvqa.utils.logging import print_bold

class SupportedHuggingfaceMedicalBERTModels:
    BiomedVLP_CXR_BERT_specialized = 'microsoft/BiomedVLP-CXR-BERT-specialized'
    BiomedVLP_BioViL_T = 'microsoft/BiomedVLP-BioViL-T'
    BioLinkBERT_large = 'michiyasunaga/BioLinkBERT-large'
    BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    Bio_ClinicalBERT = 'emilyalsentzer/Bio_ClinicalBERT'

    _model_name_to_embedding_size = {
        BiomedVLP_CXR_BERT_specialized: 128,
        BiomedVLP_BioViL_T: 128,
        BioLinkBERT_large: 1024,
        BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext: 768,
        Bio_ClinicalBERT: 768,
    }

    _model_name_to_hidden_size = {
        BiomedVLP_CXR_BERT_specialized: 768,
        BiomedVLP_BioViL_T: 768,
        BioLinkBERT_large: 1024,
        BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext: 768,
        Bio_ClinicalBERT: 768,
    }
    
    @classmethod
    def get_all(cls):
        return [
            cls.BiomedVLP_CXR_BERT_specialized,
            cls.BiomedVLP_BioViL_T,
            cls.BioLinkBERT_large,
            cls.BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext,
            cls.Bio_ClinicalBERT,
        ]
    
    @classmethod
    def get_all_cxr_bert_variants(cls):
        return [
            cls.BiomedVLP_CXR_BERT_specialized,
            cls.BiomedVLP_BioViL_T,
        ]
    
    @classmethod
    def get_models_with_pooler_output(cls):
        return [
            cls.BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext,
            cls.Bio_ClinicalBERT,
            cls.BioLinkBERT_large
        ]
    
    @classmethod
    def get_embedding_size(cls, model_name):
        return cls._model_name_to_embedding_size[model_name]
    
    @classmethod
    def get_hidden_size(cls, model_name):
        return cls._model_name_to_hidden_size[model_name]

def _adapt_checkpoint_keys(checkpoint):
    for key in list(checkpoint.keys()):
        if key.startswith('model.'):
            checkpoint[key[6:]] = checkpoint.pop(key)
    return checkpoint

def compute_text_embeddings(model_name, get_tokenizer_func, texts, device, batch_size=32, num_workers=0,
                            model_checkpoint_folder_path=None, model_checkpoint_filepath=None,
                            is_cxr_bert_variant=False, average_token_embeddings=False):

    if average_token_embeddings:
        assert not is_cxr_bert_variant, 'average_token_embeddings is not supported for CXR-BERT variants'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device.lower() in ['gpu', 'cuda'] else 'cpu')

    # Load model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)

    # Load pre-trained weights from checkpoint folder (if provided)
    if model_checkpoint_filepath is None:
        if model_checkpoint_folder_path is not None:
            model_checkpoint_filepath = get_checkpoint_filepath(model_checkpoint_folder_path)
    if model_checkpoint_filepath is not None:
        print(f'Loading model weights from {model_checkpoint_filepath}')
        checkpoint = torch.load(model_checkpoint_filepath, map_location=device)
        load_model_state_dict(model, _adapt_checkpoint_keys(checkpoint['model']), strict=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_func = get_tokenizer_func(tokenizer)

    # Create dataset and dataloader
    _, dataloader = create_text_dataset_and_dataloader(
        texts=texts,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_func=tokenizer_func,
    )

    # Run inference
    model.eval()
    embeddings = None
    offset = 0
    if is_cxr_bert_variant:
        forward_func = lambda x: model.get_projected_text_embeddings(
            input_ids=x['input_ids'], attention_mask=x['attention_mask']) # CXR-BERT variants
    else:
        if average_token_embeddings:
            forward_func = lambda x: model(**x).last_hidden_state.mean(dim=1)
        else:
            forward_func = lambda x: model(**x).pooler_output # BERT-like models
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), mininterval=2):
            encoding = batch['encoding']
            encoding_to_device = {}
            for k, v in encoding.items():
                encoding_to_device[k] = v.to(device)
            batch_embeddings = forward_func(encoding_to_device)
            batch_size = batch_embeddings.shape[0]
            if embeddings is None:
                embeddings = np.zeros((len(texts), batch_embeddings.shape[1]), dtype=np.float32) # Lazy initialization
            embeddings[offset:offset+batch_size] = batch_embeddings.cpu().numpy()
            offset += batch_size
    assert offset == len(texts)

    # Cleanup
    del model
    del tokenizer
    del dataloader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings

def _get_microsoft_BERT_tokenizer_func(tokenizer):
    return lambda x: tokenizer.batch_encode_plus(batch_text_or_text_pairs=x,
                                                add_special_tokens=True,
                                                padding='longest',
                                                return_tensors='pt')

def _get_default_BERT_tokenizer_func(tokenizer):
    return lambda x: tokenizer(x, padding='longest', return_tensors='pt')

def compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(texts, device, batch_size=32, num_workers=0,
                                                                model_checkpoint_folder_path=None,
                                                                model_checkpoint_filepath=None):
    return compute_text_embeddings(
        model_name=SupportedHuggingfaceMedicalBERTModels.BiomedVLP_CXR_BERT_specialized,
        get_tokenizer_func=_get_microsoft_BERT_tokenizer_func,
        texts=texts,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_checkpoint_folder_path=model_checkpoint_folder_path,
        model_checkpoint_filepath=model_checkpoint_filepath,
        is_cxr_bert_variant=True,
    )

def compute_text_embeddings_with_BiomedVLP_BioVilT(texts, device, batch_size=32, num_workers=0,
                                                                model_checkpoint_folder_path=None,
                                                                model_checkpoint_filepath=None):
    return compute_text_embeddings(
        model_name=SupportedHuggingfaceMedicalBERTModels.BiomedVLP_BioViL_T,
        get_tokenizer_func=_get_microsoft_BERT_tokenizer_func,
        texts=texts,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_checkpoint_folder_path=model_checkpoint_folder_path,
        model_checkpoint_filepath=model_checkpoint_filepath,
        is_cxr_bert_variant=True,
    )

def compute_text_embeddings_with_BERT_variant(model_name, texts, device, batch_size=32, num_workers=0,
                                                model_checkpoint_folder_path=None,
                                                model_checkpoint_filepath=None, average_token_embeddings=False):
    return compute_text_embeddings(
        model_name=model_name,
        get_tokenizer_func=_get_default_BERT_tokenizer_func,
        texts=texts,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_checkpoint_folder_path=model_checkpoint_folder_path,
        model_checkpoint_filepath=model_checkpoint_filepath,
        is_cxr_bert_variant=False,
        average_token_embeddings=average_token_embeddings,
    )

def generate_text_with_T5(input_texts, model_name, model_checkpoint_folder_path, max_len=None, num_beams=1, device='GPU', batch_size=32, num_workers=0):
    assert os.path.exists(model_checkpoint_folder_path)
    model_checkpoint_filepath = get_checkpoint_filepath(model_checkpoint_folder_path)
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'CPU')
    
    # Load model
    from transformers import T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print(f'Loading model weights from {model_checkpoint_filepath}')
    checkpoint = torch.load(model_checkpoint_filepath, map_location=device)
    load_model_state_dict(model, _adapt_checkpoint_keys(checkpoint['model']), strict=False)

    # Load tokenizer
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    # Create dataset and dataloader
    _, dataloader = create_text_dataset_and_dataloader(
        texts=input_texts,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_func=lambda x: tokenizer(x, padding='longest', return_tensors='pt'),
    )

    # Run inference
    output_texts = [None] * len(input_texts)
    offset = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), mininterval=2):
            encoding = batch['encoding']
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            if max_len is None:
                max_len = input_ids.shape[1] * 4 # 4x input length
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                max_new_tokens=max_len, num_beams=num_beams, early_stopping=True)
            output_texts_batch = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for i, output_text in enumerate(output_texts_batch):
                output_texts[offset+i] = output_text
            offset += len(output_texts_batch)
    assert offset == len(input_texts)
    assert None not in output_texts

    # Return output
    return output_texts

class CachedTextEmbeddingExtractor:
    def __init__(self, model_name, device='GPU', model_checkpoint_folder_path=None, batch_size=32, num_workers=0,
                 average_token_embeddings=False):
        assert model_name in SupportedHuggingfaceMedicalBERTModels.get_all()
        if average_token_embeddings:
            assert model_name not in SupportedHuggingfaceMedicalBERTModels.get_all_cxr_bert_variants()
        self.embedding_size = SupportedHuggingfaceMedicalBERTModels.get_embedding_size(model_name)
        self.model_name = model_name
        self.model_checkpoint_folder_path = model_checkpoint_folder_path
        if model_checkpoint_folder_path is not None:
            self.model_checkpoint_filepath = get_checkpoint_filepath(model_checkpoint_folder_path)
        else:
            self.model_checkpoint_filepath = None
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.average_token_embeddings = average_token_embeddings
        strings = [
            model_name,
            self.model_checkpoint_filepath or '',
        ]
        if average_token_embeddings:
            strings.append('average_token_embeddings')
        self.cache_path = get_file_path_with_hashing_if_too_long(
            folder_path=LARGE_FAST_CACHE_DIR,
            prefix=f'text_embeddings_cache',
            strings=strings,
            force_hashing=True,
        )
        self._cache = None
        self._hash2index = None

    @property
    def cache(self):
        if self._cache is None:
            self._load_cache()
        return self._cache
    
    @property
    def hash2index(self):
        if self._hash2index is None:
            self._load_cache()
        return self._hash2index

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            print(f'Loading cached text embeddings from {self.cache_path}')
            self._cache = load_pickle(self.cache_path)
            print(f'len(self.cache["hashes"]) = {len(self.cache["hashes"])}')
            print(f'self.cache["embeddings"].shape = {self.cache["embeddings"].shape}')
            self._hash2index = { h:i for i,h in enumerate(self.cache['hashes']) }
            assert len(self.hash2index) == len(self.cache['hashes'])
        else:
            self._cache = { 'hashes': [], 'embeddings': np.empty((0, self.embedding_size), dtype=np.float32) }
            self._hash2index = {}

    def _compute_embeddings(self, texts):
        if self.model_name == SupportedHuggingfaceMedicalBERTModels.BiomedVLP_CXR_BERT_specialized:
            embeddings = compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(
                texts=texts,
                device=self.device,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                model_checkpoint_folder_path=self.model_checkpoint_folder_path,
            )
        elif self.model_name == SupportedHuggingfaceMedicalBERTModels.BiomedVLP_BioViL_T:
            embeddings = compute_text_embeddings_with_BiomedVLP_BioVilT(
                texts=texts,
                device=self.device,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                model_checkpoint_folder_path=self.model_checkpoint_folder_path,
            )
        elif self.model_name in SupportedHuggingfaceMedicalBERTModels.get_models_with_pooler_output():
            embeddings = compute_text_embeddings_with_BERT_variant(
                model_name=self.model_name,
                texts=texts,
                device=self.device,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                model_checkpoint_folder_path=self.model_checkpoint_folder_path,
                average_token_embeddings=self.average_token_embeddings,
            )
        else:
            raise NotImplementedError(f'Unsupported model name: {self.model_name}')
        return embeddings
    
    def _add_to_cache(self, texts, embeddings, update_cache_on_disk=True):
        assert len(texts) == embeddings.shape[0]
        assert embeddings.shape[1] == self.embedding_size
        hashes = compute_hashes_in_parallel(texts)
        # Add new embeddings to cache
        self.cache['hashes'].extend(hashes)
        self.cache['embeddings'] = np.concatenate([self.cache['embeddings'], embeddings], axis=0)
        # Update hash2index
        n = len(self.cache['hashes'])
        offset = n - len(texts)
        for i in range(offset, n):
            h = hashes[i-offset]
            assert h not in self.hash2index
            self.hash2index[h] = i
        # Update cache on disk
        if update_cache_on_disk:
            print(f'Saving updated cache to {self.cache_path}')
            save_pickle(self.cache, self.cache_path)

    def compute_text_embeddings(self, texts, update_cache_on_disk=True):
        embedding_indices = [None] * len(texts)
        
        # Compute embeddings for new texts
        new_texts_set = set()
        hash_index_pairs = []
        hashes = compute_hashes_in_parallel(texts)
        for i, text in tqdm(enumerate(texts), mininterval=2, total=len(texts)):
            h = hashes[i]
            if h in self.hash2index:
                embedding_indices[i] = self.hash2index[h]
            else:
                new_texts_set.add(text)
                hash_index_pairs.append((h, i))
        if len(new_texts_set) > 0:
            new_texts = list(new_texts_set)
            print(f'Computing embeddings for {len(new_texts)} new texts')
            new_embeddings = self._compute_embeddings(new_texts)
            assert new_embeddings.shape == (len(new_texts), self.embedding_size)
            self._add_to_cache(new_texts, new_embeddings, update_cache_on_disk=update_cache_on_disk)

        # Retrieve embeddings from cache
        if len(hash_index_pairs) > 0:
            for h, i in tqdm(hash_index_pairs, mininterval=2, total=len(hash_index_pairs)):
                embedding_indices[i] = self.hash2index[h]
        assert None not in embedding_indices
        output = self.cache['embeddings'][embedding_indices]
        assert output.shape == (len(texts), self.embedding_size)
        return output
    
    def compute_kmeans_labels(self, texts, num_clusters, num_iterations=300, verbose=2, update_cache_on_disk=True, cache_kmeans_labels=True,
                              embeddings=None):
        if cache_kmeans_labels:
            h = hash_string_list(texts)
            h = update_hash(h, f'n_clusters={num_clusters}')
            h = update_hash(h, f'num_iterations={num_iterations}')
            h = update_hash(h, f'cache_path={self.cache_path}')
            save_path = os.path.join(LARGE_FAST_CACHE_DIR, f'kmeans_labels({h[0]},{h[1]}).pkl')
            if save_path in self.cache:
                return self.cache[save_path]
            if os.path.exists(save_path):
                print(f'Loading cached kmeans labels from {save_path}')
                output = self.cache[save_path] = load_pickle(save_path)
                return output
        if embeddings is None:
            embeddings = self.compute_text_embeddings(texts, update_cache_on_disk=update_cache_on_disk)
        else:
            assert embeddings.shape[0] == len(texts) # Sanity check
        print(f'Running KMeans clustering with k={num_clusters}')
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto', verbose=verbose, max_iter=num_iterations).fit(embeddings)
        labels = kmeans.labels_
        if cache_kmeans_labels:
            self.cache[save_path] = labels
            save_pickle(labels, save_path)
            print(f'Saved kmeans labels to {save_path}')
        return labels
    
class CachedT5FactExtractor:
    def __init__(self, model_name, model_checkpoint_folder_path, device='GPU', batch_size=32, num_workers=0):
        assert os.path.exists(model_checkpoint_folder_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.model_checkpoint_filepath = get_checkpoint_filepath(model_checkpoint_folder_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'CPU')
        self._model = None
        self._cache = None
        self._cache_path = os.path.join(LARGE_FAST_CACHE_DIR, f't5_facts_cache({hash_string_list([model_name, self.model_checkpoint_filepath])}).pkl')

    @property
    def t5_model(self):
        if self._model is None:
            from transformers import T5ForConditionalGeneration
            self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print(f'Loading model weights from {self.model_checkpoint_filepath}')
            checkpoint = torch.load(self.model_checkpoint_filepath, map_location=self.device)
            load_model_state_dict(self._model, _adapt_checkpoint_keys(checkpoint['model']), strict=False)
        return self._model
    
    @property
    def cache(self):
        if self._cache is None:
            if os.path.exists(self._cache_path):
                print(f'Loading cached T5 facts from {self._cache_path}')
                self._cache = load_pickle(self._cache_path)
            else:
                # Load environment variables from .env file
                from dotenv import load_dotenv
                load_dotenv()
                INTEGRATED_SENTENCE_FACTS_JSONL_PATH = os.getenv('INTEGRATED_SENTENCE_FACTS_JSONL_PATH')
                # Populate cache with facts from integrated sentence facts
                print(f'Populating cache with facts from {INTEGRATED_SENTENCE_FACTS_JSONL_PATH}')
                rows = load_jsonl(INTEGRATED_SENTENCE_FACTS_JSONL_PATH)
                cache = {}
                for row in tqdm(rows, mininterval=2):
                    cache[hash_string(row['sentence'])] = row['facts']
                # Save cache to disk
                self._cache = cache
                save_pickle(cache, self._cache_path)
                print(f'Saved T5 facts to {self._cache_path}')
        return self._cache
    
    def _compute_facts(self, sentences):

        # Load model
        model = self.t5_model

        # Load tokenizer
        from transformers import T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained(self.model_name)

        # Create dataset and dataloader
        _, dataloader = create_text_dataset_and_dataloader(
            texts=sentences,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            tokenizer_func=lambda x: tokenizer(x, padding='longest', return_tensors='pt'),
        )

        from medvqa.datasets.text_data_utils import parse_facts

        # Run inference
        output = [None] * len(sentences)
        offset = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), mininterval=2):
                encoding = batch['encoding']
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                max_len = input_ids.shape[1] * 4 # 4x input length
                output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                  max_new_tokens=max_len, num_beams=1, early_stopping=True)
                output_texts_batch = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for i, output_text in enumerate(output_texts_batch):
                    facts = parse_facts(output_text)
                    facts = [remove_consecutive_repeated_words_from_text(f) for f in facts]
                    output[offset+i] = facts
                offset += len(output_texts_batch)
        assert offset == len(sentences)
        assert None not in output

        # Return output
        return output
    
    def _aggregate_facts(self, facts_per_sentence):
        output = []
        seen = set()
        for facts in facts_per_sentence:
            for f in facts:
                if f not in seen:
                    seen.add(f)
                    output.append(f)
        return output

    def __call__(self, texts, update_cache_on_disk=True):
        assert type(texts) in [list, tuple, np.ndarray]
        assert isinstance(texts[0], str)
        sentences_per_text = sentence_tokenize_texts_in_parallel(texts)
        unique_sentences = []
        s2i = {}
        for sentences in sentences_per_text:
            for s in sentences:
                if s not in s2i:
                    s2i[s] = len(unique_sentences)
                    unique_sentences.append(s)
        hashes = compute_hashes_in_parallel(unique_sentences)
        facts_per_sentence = [None] * len(unique_sentences)
        cache = self.cache
        sentences_to_process = []
        for i, s in enumerate(unique_sentences):
            try:
                facts_per_sentence[i] = cache[hashes[i]]
            except KeyError:
                sentences_to_process.append((s, i)) # (sentence, index)
        if len(sentences_to_process) > 0:
            facts_per_sentence_to_process = self._compute_facts([s for s, _ in sentences_to_process])
            for i, facts in enumerate(facts_per_sentence_to_process):
                idx = sentences_to_process[i][1]
                facts_per_sentence[idx] = facts
                cache[hashes[idx]] = facts # update cache with new facts
            if update_cache_on_disk:
                save_pickle(cache, self._cache_path)
                print(f'Saved updated T5 facts to {self._cache_path}')
                print(f'New cache size = {len(cache)}')
        assert None not in facts_per_sentence
        output = [None] * len(texts)
        for i, sentences in enumerate(sentences_per_text):
            output[i] = self._aggregate_facts([facts_per_sentence[s2i[s]] for s in sentences])
        return output
    
class TripletRankingEvaluator:
    def __init__(self, triplets_filepath, model_name, device='GPU', model_checkpoint_folder_path=None, batch_size=32, num_workers=0,
                 average_token_embeddings=False):
        print(f'Loading triplets from {triplets_filepath}')
        self.triplets_data = get_cached_pickle_file(triplets_filepath)
        if model_name == 'CheXbert':
            from medvqa.metrics.medical.chexbert import CheXbertLabeler            
            chexbert_labeler =  CheXbertLabeler(device=device)
            self._get_embeddings_func =chexbert_labeler.get_embeddings
        else:
            embedding_extractor = CachedTextEmbeddingExtractor(
                model_name=model_name,
                device=device,
                model_checkpoint_folder_path=model_checkpoint_folder_path,
                batch_size=batch_size,
                num_workers=num_workers,
                average_token_embeddings=average_token_embeddings,
            )
            self._get_embeddings_func = embedding_extractor.compute_text_embeddings

    def evaluate_triplet_ranking(self, split, category, rule_index):
        assert split in ['train', 'val', 'test']
        assert category in ['observations', 'anatomical_locations']
        sentences = self.triplets_data['sentences']
        rule = self.triplets_data[split][category][rule_index]['rule']
        triplets = self.triplets_data[split][category][rule_index]['triplets']
        print(f'Evaluating triplet ranking on {split} split with category {category} and rule "{rule}"')
        print(f'triplets.shape = {triplets.shape}')
        anchors = [sentences[i] for i in triplets.T[0]]
        positives = [sentences[i] for i in triplets.T[1]]
        negatives = [sentences[i] for i in triplets.T[2]]
        embeddings = self._get_embeddings_func(anchors + positives + negatives)
        A = embeddings[:len(anchors)]
        P = embeddings[len(anchors):len(anchors)+len(positives)]
        N = embeddings[len(anchors)+len(positives):]
        AP = np.sum(A * P, axis=1)
        AN = np.sum(A * N, axis=1)
        correct = AP > AN
        accuracy = np.mean(correct)
        return {
            'accuracy': accuracy,
            'correct': correct,
            'AP': AP,
            'AN': AN,
            'anchors': anchors,
            'positives': positives,
            'negatives': negatives,
        }
    
    def evaluate_split(self, split):
        assert split in ['train', 'val', 'test']
        sentences = self.triplets_data['sentences']
        split_idxs = set()
        for category in ['observations', 'anatomical_locations']:
            for rule_index in range(len(self.triplets_data[split][category])):
                triplets = self.triplets_data[split][category][rule_index]['triplets']
                split_idxs.update(triplets.flatten())
        split_idxs = sorted(list(split_idxs))
        split_sentences = [sentences[i] for i in split_idxs]
        split_embeddings = self._get_embeddings_func(split_sentences)
        split_embeddings /= np.linalg.norm(split_embeddings, axis=1, keepdims=True) # Normalize embeddings
        idx2i = { idx:i for i,idx in enumerate(split_idxs) }
        assert len(idx2i) == len(split_idxs)
        
        for category in ['observations', 'anatomical_locations']:
            for rule_index in range(len(self.triplets_data[split][category])):
                triplets = self.triplets_data[split][category][rule_index]['triplets']
                rule = self.triplets_data[split][category][rule_index]['rule']
                print(f'Category: {category}, Rule: {rule}')
                print(f'triplets.shape = {triplets.shape}')
                anchors = [idx2i[i] for i in triplets.T[0]]
                positives = [idx2i[i] for i in triplets.T[1]]
                negatives = [idx2i[i] for i in triplets.T[2]]
                A = split_embeddings[anchors]
                P = split_embeddings[positives]
                N = split_embeddings[negatives]
                AP = np.sum(A * P, axis=1)
                AN = np.sum(A * N, axis=1)
                correct = AP > AN
                accuracy = np.mean(correct)
                print_bold(f'accuracy = {accuracy}')
                print('-'*80)

