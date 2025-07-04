import random
import math
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from medvqa.datasets.dataloading_utils import INFINITE_DATASET_LENGTH, CompositeInfiniteDataset
from medvqa.datasets.nli import MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH, RADNLI_DEV_JSONL_PATH, RADNLI_TEST_JSONL_PATH
from medvqa.datasets.seq2seq.seq2seq_dataset_management import _compute_input2paraphrases, _prepare_nli_data, _prepare_sentence_to_facts_data
from medvqa.utils.text_data_utils import sentence_tokenize_texts_in_parallel
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.files_utils import load_jsonl
from medvqa.utils.logging_utils import print_bold
from medvqa.utils.math_utils import rank_vectors_by_dot_product

_LABEL_TO_INDEX = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}
_INDEX_TO_LABEL = { v: k for k, v in _LABEL_TO_INDEX.items() }

class BertNLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, shuffle=False, infinite=False, merged=False, indices=None):
        assert len(premises) == len(hypotheses) == len(labels)        
        if merged:
            self.texts = [f'{p} [SEP] {h}' for p, h in zip(premises, hypotheses)]
        else:            
            self.premises = premises
            self.hypotheses = hypotheses
        self.merged = merged
        self.labels = labels
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(len(self.labels)))
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
        if shuffle:
            random.shuffle(self.indices) # shuffle in place
    
    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        idx = self.indices[i]
        l = self.labels[idx]
        if self.merged:
            t = self.texts[idx]
            output = { 't': t, 'l': l }
        else:
            p = self.premises[idx]
            h = self.hypotheses[idx]
            output = { 'p': p, 'h': h, 'l': l }
        return output

class EntailmentContradictionDataset(Dataset):
    def __init__(self, ent_premises, ent_hypotheses, con_premises, con_hypotheses, shuffle=False, infinite=False):
        print('EntailmentContradictionDataset: __init__')
        assert len(ent_premises) == len(ent_hypotheses)
        assert len(con_premises) == len(con_hypotheses)
        self.ent_premises = ent_premises
        self.ent_hypotheses = ent_hypotheses
        self.con_premises = con_premises
        self.con_hypotheses = con_hypotheses
        self.n_ent = len(ent_premises)
        self.n_con = len(con_premises)
        print(f'Number of entailment samples: {self.n_ent}')
        print(f'Number of contradiction samples: {self.n_con}')
        self.indices = list(range(self.n_con)) # iterate over contradiction samples and pick a random entailment sample
        self.infinite = infinite
        if infinite:
            self._len = INFINITE_DATASET_LENGTH
        else:
            self._len = len(self.indices)
        if shuffle:
            random.shuffle(self.indices) # shuffle in place

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self.infinite:
            i = i % len(self.indices)
        con_idx = self.indices[i]
        ent_idx = random.randint(0, self.n_ent-1)
        return {
            'ent_p': self.ent_premises[ent_idx],
            'ent_h': self.ent_hypotheses[ent_idx],
            'con_p': self.con_premises[con_idx],
            'con_h': self.con_hypotheses[con_idx],
        } # sim(ent_p, ent_h) > sim(con_p, con_h) (in words: sentences that are entailed should have higher similarity
        # than sentences that are contradictory)

class BertNLITrainer:

    def __init__(self, batch_size, num_workers, collate_batch_fn, integrated_nli_jsonl_filepath=None,
                 integrated_sentence_facts_jsonl_filepath=None, merged=False,
                 train_mode=True, test_mode=False, dev_mode=False, use_mscxrt=False, use_radnli_test=True):

        self.batch_size = batch_size
        self.num_workers = num_workers
        assert sum([train_mode, dev_mode, test_mode]) == 1 # only one mode must be True

        if use_mscxrt or use_radnli_test:
            assert train_mode or test_mode

        if train_mode:
            assert integrated_nli_jsonl_filepath is not None
            # Load data from MedNLI. This will be used to train the NLI model.
            nli_train_rows = load_jsonl(integrated_nli_jsonl_filepath)
            train_premises = [x['premise'] for x in nli_train_rows]
            train_hypotheses = [x['hypothesis'] for x in nli_train_rows]
            train_labels = [_LABEL_TO_INDEX[x['label']] for x in nli_train_rows]
            source2id = {}
            for row in nli_train_rows:
                source = row['source']
                if source not in source2id:
                    source2id[source] = len(source2id)
            id2source = { v: k for k, v in source2id.items() }
            train_sources = [source2id[x['source']] for x in nli_train_rows]
            print(f'Number of train samples: {len(train_premises)}')
            print_bold('Example train text:')
            _i = random.randint(0, len(train_premises)-1)
            print(f'Premise: {train_premises[_i]}')
            print(f'Hypothesis: {train_hypotheses[_i]}')
            print(f'Label: {_INDEX_TO_LABEL[train_labels[_i]]}')
            print(f'Source: {train_sources[_i]}')

            if integrated_sentence_facts_jsonl_filepath is not None:
                print(f'Loading integrated sentence facts from {integrated_sentence_facts_jsonl_filepath}...')
                integrated_sentence_facts = load_jsonl(integrated_sentence_facts_jsonl_filepath)
                print(f'Number of integrated sentence facts: {len(integrated_sentence_facts)}')
                entailment_id = _LABEL_TO_INDEX['entailment']
                source_id = len(source2id) # add new source
                id2source[source_id] = 'integrated_sentence_facts'
                len_bef = len(train_premises)
                for row in integrated_sentence_facts:
                    s = row['sentence']
                    for f in row['facts']:
                        if s != f:
                            train_premises.append(s)
                            train_hypotheses.append(f)
                            train_sources.append(source_id)
                            train_labels.append(entailment_id)
                print(f'Number of train entailment samples added: {len(train_premises) - len_bef}')
        
        if train_mode or test_mode:
            test_premises = []
            test_hypotheses = []
            test_labels = []
            if use_radnli_test:
                radnli_test_rows = load_jsonl(RADNLI_TEST_JSONL_PATH)
                test_premises.extend(x['sentence1'] for x in radnli_test_rows)
                test_hypotheses.extend(x['sentence2'] for x in radnli_test_rows)
                test_labels.extend(_LABEL_TO_INDEX[x['gold_label']] for x in radnli_test_rows)
                print(f'Number of test samples: {len(test_premises)}')
                print_bold('Example test text:')
                _i = random.randint(0, len(test_premises)-1)
                print(f'Premise: {test_premises[_i]}')
                print(f'Hypothesis: {test_hypotheses[_i]}')
                print(f'Label: {_INDEX_TO_LABEL[test_labels[_i]]}')
            if use_mscxrt: # MS_CXR_T
                df = pd.read_csv(MS_CXR_T_TEMPORAL_SENTENCE_SIMILARITY_V1_CSV_PATH)
                n = len(df)
                for premise, hypothesis, label in zip(df.sentence_1, df.sentence_2, df.category):
                    test_premises.append(premise)
                    test_hypotheses.append(hypothesis)
                    if label == 'paraphrase':
                        test_labels.append(_LABEL_TO_INDEX['entailment'])                
                    elif label == 'contradiction':
                        test_labels.append(_LABEL_TO_INDEX['contradiction'])
                    else:
                        raise ValueError(f'Unknown label {label}')
                print(f'Number of test samples: {len(test_premises)}')
                print_bold('Example test text:')
                _i = random.randint(len(test_premises)-n, len(test_premises)-1)
                print(f'Premise: {test_premises[_i]}')
                print(f'Hypothesis: {test_hypotheses[_i]}')
                print(f'Label: {_INDEX_TO_LABEL[test_labels[_i]]}')

        if dev_mode:
            radnli_dev_rows = load_jsonl(RADNLI_DEV_JSONL_PATH)
            dev_premises = [x['sentence1'] for x in radnli_dev_rows]
            dev_hypotheses = [x['sentence2'] for x in radnli_dev_rows]
            dev_labels = [_LABEL_TO_INDEX[x['gold_label']] for x in radnli_dev_rows]
            print(f'Number of dev samples: {len(dev_premises)}')
            print_bold('Example dev text:')
            _i = random.randint(0, len(dev_premises)-1)
            print(f'Premise: {dev_premises[_i]}')
            print(f'Hypothesis: {dev_hypotheses[_i]}')
            print(f'Label: {_INDEX_TO_LABEL[dev_labels[_i]]}')
        
        # Create datasets and dataloaders
        print('----')
        if train_mode:
            print_bold('Building train NLI dataset and dataloader...') 

            # Create a dataset for each label and source combination and weight them by log2(N)^3
            source_label_2_indices = {}
            unique_sources = set()
            unique_labels = set()
            for i, (source, label) in enumerate(zip(train_sources, train_labels)):
                unique_sources.add(source)
                unique_labels.add(label)
                key = (source, label)
                if key not in source_label_2_indices:
                    source_label_2_indices[key] = []
                source_label_2_indices[key].append(i)
            unique_sources = sorted(list(unique_sources))
            unique_labels = sorted(list(unique_labels))

            label_datasets = []
            label_weights = []
            label2weight = {
                'entailment': 1,
                'contradiction': 1,
                'neutral': 2, # 2 neutral samples will be sampled twice as much as the other labels
            }
            for label in unique_labels:
                _datasets = []
                _weights = []
                for source in unique_sources:
                    key = (source, label)
                    if key in source_label_2_indices:
                        indices = source_label_2_indices[key]
                        _datasets.append(BertNLIDataset(train_premises, train_hypotheses, train_labels, shuffle=True, infinite=True, indices=indices))
                        _weights.append(math.log2(len(indices))**3) # weight by log2(N)^3
                        # print(f'Source: {source} | Label: {label} -> {len(indices)} ({_weights[-1]:.2f})')
                        print(f'Label: {_INDEX_TO_LABEL[label]} | Source: {id2source[source]} -> {len(indices)} ({_weights[-1]:.2f})')
                label_datasets.append(CompositeInfiniteDataset(_datasets, _weights))
                label_weights.append(label2weight[_INDEX_TO_LABEL[label]])
            print(f'Number of label datasets: {len(label_datasets)}')
            print(f'Label weights: {label_weights}')
            
            self.train_dataset = CompositeInfiniteDataset(label_datasets, label_weights)
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
        if dev_mode:
            print_bold('Building dev NLI dataset and dataloader...')
            self.dev_dataset = BertNLIDataset(dev_premises, dev_hypotheses, dev_labels, merged=merged)
            self.dev_dataloader = DataLoader(
                self.dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
        if train_mode or test_mode:
            print_bold('Building test NLI dataset and dataloader...')
            self.test_dataset = BertNLIDataset(test_premises, test_hypotheses, test_labels, merged=merged)
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )

    @property
    def name(self):
        return 'NLI(BERT-based)'

def _get_nli_embedding_inputs_for_mlp(embeddings, sentence2idx, premise_sentences, hypothesis):
    h_emb = embeddings[sentence2idx[hypothesis]]
    if len(premise_sentences) == 0:
        zero_vector = np.zeros((embeddings.shape[1],), dtype=np.float32)
        p_most_sim_emb = zero_vector
        p_least_sim_emb = zero_vector
        p_max_emb = zero_vector
        p_avg_emb = zero_vector
    else:
        p_idxs = [sentence2idx[s] for s in premise_sentences]
        p_embs = embeddings[p_idxs]
        sorted_idxs = rank_vectors_by_dot_product(p_embs, h_emb)
        p_most_sim_emb = p_embs[sorted_idxs[0]]
        p_least_sim_emb = p_embs[sorted_idxs[-1]]
        p_max_emb = np.max(p_embs, axis=0)
        p_avg_emb = np.mean(p_embs, axis=0)
    return h_emb, p_most_sim_emb, p_least_sim_emb, p_max_emb, p_avg_emb

class EmbeddingNLIDatasetWrapper(Dataset):
    def __init__(self, seq2seq_dataset, embeddings, sentence2idx, whole_sentences, splittable_sentences):
        self._d = seq2seq_dataset
        self.embeddings = embeddings
        self.sentence2idx = sentence2idx
        self.whole_sentences = whole_sentences
        self.splittable_sentences = splittable_sentences
        self._zero_vector = np.zeros((self.embeddings.shape[1]), dtype=np.float32)
        assert type(self.whole_sentences) == set
    
    def __len__(self):
        return len(self._d)

    def __getitem__(self, i):
        try:
            item = self._d[i]
            input_text = item['input_text']
            output_text = item['output_text']
            assert input_text.startswith('NLI1: ') # sanity check
            sep_idx = input_text.index(' #H: ')
            premise = input_text[6:sep_idx]
            if premise in self.whole_sentences:
                premise_sentences = [premise.strip()]
            else:
                assert premise in self.splittable_sentences, f'Premise not found: {premise}'
                premise_sentences = sent_tokenize(premise)

            hypothesis = input_text[sep_idx+5:].strip()
            h_emb = self.embeddings[self.sentence2idx[hypothesis]]

            h_emb, p_most_sim_emb, p_least_sim_emb, p_max_emb, p_avg_emb = _get_nli_embedding_inputs_for_mlp(
                self.embeddings, self.sentence2idx, premise_sentences, hypothesis
            )
            
            label = _LABEL_TO_INDEX[output_text]
            return {
                'h_emb': h_emb,
                'p_most_sim_emb': p_most_sim_emb,
                'p_least_sim_emb': p_least_sim_emb,
                'p_max_emb': p_max_emb,
                'p_avg_emb': p_avg_emb,
                'l': label,
            }
        except Exception as e:
            print(f'Error in EmbeddingNLIDatasetWrapper.__getitem__: {e}')
            print(f'Index: {i}')
            print(f'Item: {item}')
            print(f'Premise: {premise}')
            print(f'Hypothesis: {hypothesis}')
            print(f'Premise sentences: {premise_sentences}')
            raise e

class EmbeddingNLITrainer:

    def __init__(self,
                train_mode=False,
                dev_mode=False,
                test_mode=False,
                batch_size=32,
                num_workers=4,
                collate_batch_fn=None,
                integrated_nli_jsonl_filepath=None,
                use_sentence2facts_for_nli=False,
                sentence_to_facts_input_output_jsonl_filepaths=None,
                use_anli=False,
                use_multinli=False,
                use_snli=False,
                use_report_nli=False,
                raw_report_nli_input_output_train_jsonl_filepaths=None,
                report_nli_input_output_train_jsonl_filepaths=None,
                report_nli_input_output_val_jsonl_filepaths=None,
                use_report_nli_entailment_dataset=False,
                use_report_nli_paraphrases_dataset=False,
                integrated_report_facts_jsonl_filepath=None,
                paraphrased_inputs_jsonl_filepaths=None,
                verbose=False,
                fact_embedding_model_name=None,
                fact_embedding_model_checkpoint_folder_path=None,
                fact_embedding_batch_size=32,
                fact_embedding_num_workers=4,
                ):
        
        assert sum([train_mode, dev_mode, test_mode]) == 1 # only one mode must be True
        if train_mode:
            if use_sentence2facts_for_nli:
                _, _, s2f_input_output_for_nli = _prepare_sentence_to_facts_data(
                    sentence_to_facts_input_output_jsonl_filepaths, collect_input_output_for_nli=True,
                )
                s2f_aux = s2f_input_output_for_nli
            else:
                s2f_aux = None
            medical_sentences = set()
            general_sentences = set()
            whole_sentences = set()
            splittable_sentences = set()
            if use_report_nli_paraphrases_dataset:
                assert paraphrased_inputs_jsonl_filepaths is not None
                input_to_paraphrases = _compute_input2paraphrases(paraphrased_inputs_jsonl_filepaths, verbose=verbose)
            else:
                input_to_paraphrases = None
            train_dataset, dev_dataset = _prepare_nli_data(integrated_nli_jsonl_filepath, s2f_aux,
                          use_anli, use_multinli, use_snli, use_report_nli,
                          raw_report_nli_input_output_train_jsonl_filepaths,
                          report_nli_input_output_train_jsonl_filepaths,
                          report_nli_input_output_val_jsonl_filepaths,
                          use_report_nli_entailment_dataset=use_report_nli_entailment_dataset,
                          use_report_nli_paraphrases_dataset=use_report_nli_paraphrases_dataset,
                          integrated_report_facts_jsonl_filepath=integrated_report_facts_jsonl_filepath,
                          verbose=verbose, nli1_only_on_train=True, nli1_only_on_val=True,
                          medical_sentences=medical_sentences, general_sentences=general_sentences,
                          input_to_paraphrases=input_to_paraphrases,
                          whole_sentences=whole_sentences, splittable_sentences=splittable_sentences)
            print(f'Number of general sentences: {len(general_sentences)}')
            print(f'Number of medical sentences: {len(medical_sentences)}')
            print(f'Number of whole sentences: {len(whole_sentences)}')
            print(f'Number of splittable sentences: {len(splittable_sentences)}')
            len1 = len(medical_sentences.union(general_sentences))
            len2 = len(whole_sentences.union(splittable_sentences))
            assert len1 == len2, f'{len1} != {len2}' # sanity check
            
            # Tokenize sentences
            unique_sentences = set()
            for s in whole_sentences:
                unique_sentences.add(s.strip())
            sentences_list = sentence_tokenize_texts_in_parallel(splittable_sentences)
            for sentences in sentences_list:
                for s in sentences:
                    unique_sentences.add(s)
            unique_sentences = list(unique_sentences)
            sentence2idx = { s: i for i, s in enumerate(unique_sentences) }
            print(f'Number of unique sentences (after sent_tokenize): {len(unique_sentences)}')
            
            # Print examples of train_dataset and dev_dataset
            print_bold('Example train texts:')
            for _ in range(3):
                _i = random.randint(0, len(train_dataset)-1)
                print(train_dataset[_i])

            print_bold('Examples dev texts:')
            for _ in range(3):
                _i = random.randint(0, len(dev_dataset)-1)
                print(dev_dataset[_i])
            
            embedding_extractor = CachedTextEmbeddingExtractor(
                model_name=fact_embedding_model_name,
                model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
                batch_size=fact_embedding_batch_size,
                num_workers=fact_embedding_num_workers,
            )
            embeddings = embedding_extractor.compute_text_embeddings(unique_sentences)
            print(f'embeddings.shape: {embeddings.shape}')
            
            train_dataset = EmbeddingNLIDatasetWrapper(train_dataset, embeddings, sentence2idx, whole_sentences, splittable_sentences)
            dev_dataset = EmbeddingNLIDatasetWrapper(dev_dataset, embeddings, sentence2idx, whole_sentences, splittable_sentences)

            # Print examples of train_dataset and dev_dataset (after wrapping)
            print_bold('Example train text (after wrapping):')
            _i = random.randint(0, len(train_dataset)-1)
            print(train_dataset[_i])

            print_bold('Example dev text (after wrapping):')
            _i = random.randint(0, len(dev_dataset)-1)
            print(dev_dataset[_i])

            self.train_dataset = train_dataset
            self.dev_dataset = dev_dataset
            self.train_dataloader = DataLoader(
                self.train_dataset,
                # self.dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
            self.dev_dataloader = DataLoader(
                self.dev_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch_fn,
                pin_memory=True,
            )
        elif dev_mode:
            raise NotImplementedError('Dev mode not implemented yet')
        elif test_mode:
            raise NotImplementedError('Test mode not implemented yet')
        else:
            raise ValueError('Invalid mode')
        
    @property
    def name(self):
        return 'NLI(Embedding-based)'