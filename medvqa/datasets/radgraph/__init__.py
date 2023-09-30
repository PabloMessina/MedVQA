from dotenv import load_dotenv
load_dotenv()

import os
import re
from nltk.tokenize import sent_tokenize
from medvqa.utils.hashing import hash_string

RADGRAPH_MODEL_CHECKPOINT_PATH = os.environ['RADGRAPH_MODEL_CHECKPOINT_PATH']
DYGIE_PACKAGE_PARENT_FOLDER =  os.environ['DYGIE_PACKAGE_PARENT_FOLDER']
RADGRAPH_DATA_DIR_PATH = os.environ['RADGRAPH_DATA_DIR_PATH']
RADGRAPH_CHEXPERT_GRAPHS_JSON_PATH = os.path.join(RADGRAPH_DATA_DIR_PATH, 'CheXpert_graphs.json')
RADGRAPH_MIMICCXR_GRAPHS_JSON_PATH = os.path.join(RADGRAPH_DATA_DIR_PATH, 'MIMIC-CXR_graphs.json')
RADGRAPH_TRAIN_GRAPH_JSON_PATH = os.path.join(RADGRAPH_DATA_DIR_PATH, 'train.json')
RADGRAPH_DEV_GRAPH_JSON_PATH = os.path.join(RADGRAPH_DATA_DIR_PATH, 'dev.json')
RADGRAPH_TEST_GRAPH_JSON_PATH = os.path.join(RADGRAPH_DATA_DIR_PATH, 'test.json')
RADGRAPH_CONLLFORMAT_TYPES_JSON_PATH = os.environ['RADGRAPH_CONLLFORMAT_TYPES_JSON_PATH']
RADGRAPH_CONLLFORMAT_TRAIN_JSON_PATH = os.environ['RADGRAPH_CONLLFORMAT_TRAIN_JSON_PATH']
RADGRAPH_CONLLFORMAT_DEV_JSON_PATH = os.environ['RADGRAPH_CONLLFORMAT_DEV_JSON_PATH']
RADGRAPH_CONLLFORMAT_TEST_JSON_PATH = os.environ['RADGRAPH_CONLLFORMAT_TEST_JSON_PATH']
RADGRAPH_CONLLFORMAT_ENTITY_TYPE_COUNT = 5
RADGRAPH_CONLLFORMAT_RELATION_TYPE_COUNT = 3 # 4

_ONLY_UNDERSCORES_REGEX = re.compile(r'^_+$')

def _token_idx_to_sentence_idx(token_idx, sentence_offsets):
    for i in range(len(sentence_offsets)):
        if token_idx < sentence_offsets[i]:
            return i-1
    return len(sentence_offsets) - 1

def _extract_entities_and_relations_per_sentence(text, entities, max_sentences_per_entity=2, max_sentences_per_relation=1,
                                                 clean_text=True, entity_radius=3):
    sentences = sent_tokenize(text)
    ns = len(sentences)
    sentence_tokens = [s.split() for s in sentences]
    sentence_offsets = [0] * ns
    for i in range(1, ns):
        sentence_offsets[i] = sentence_offsets[i-1] + len(sentence_tokens[i-1])

    # Clean text
    if clean_text:
        n_tokens = sum([len(s) for s in sentence_tokens])
        counts = [0] * n_tokens
        for e in entities.values():
            s_idx = max(e['start_ix'] - entity_radius, 0)
            e_idx = min(e['end_ix'] + entity_radius, n_tokens-1) + 1
            counts[s_idx] += 1
            if e_idx < n_tokens:
                counts[e_idx] -= 1
        for i in range(1, n_tokens):
            counts[i] += counts[i-1]
        i = 0
        for s in range(ns):
            for t in range(len(sentence_tokens[s])):
                if counts[i] == 0 or _ONLY_UNDERSCORES_REGEX.match(sentence_tokens[s][t]):
                    sentence_tokens[s][t] = ''
                i += 1
            sentence_tokens[s] = [t for t in sentence_tokens[s] if t != '']
            sentences[s] = ' '.join(sentence_tokens[s])

    pairs_to_ent_rels = {}
    
    # Add entities
    e_pairs = {}
    for k, e in entities.items():
        start_sent_idx = _token_idx_to_sentence_idx(e['start_ix'], sentence_offsets)
        end_sent_idx = _token_idx_to_sentence_idx(e['end_ix'], sentence_offsets)
        p = (start_sent_idx, end_sent_idx)
        assert p[0] <= p[1]
        if p[1] - p[0] >= max_sentences_per_entity:
            continue # skip entities spanning more than max_sentences_per_entity
        e_string = f'{e["tokens"].lower()}|{e["label"]}'
        assert start_sent_idx <= end_sent_idx
        try:
            pairs_to_ent_rels[p].append(e_string)
        except KeyError:
            pairs_to_ent_rels[p] = [e_string]
        e_pairs[k] = (e_string, p)
    
    # Add relations
    for k, e in entities.items():
        try:
            ep1 = e_pairs[k]
        except KeyError:
            continue
        for r in e['relations']:
            try:
                ep2 = e_pairs[r[1]]
            except KeyError:
                continue
            start_sent_idx = min(ep1[1][0], ep2[1][0])
            end_sent_idx = max(ep1[1][1], ep2[1][1])
            p = (start_sent_idx, end_sent_idx)
            if p[1] - p[0] >= max_sentences_per_relation:
                continue # skip relations spanning more than max_sentences_per_relation
            r_string_1 = f'{ep1[0]}|{r[0]}|{ep2[0]}' # e1|rel|e2
            r_string_2 = f'{ep1[0]}|{ep2[0]}' # e1|e2
            try:
                pairs_to_ent_rels[p].append(r_string_1)
                pairs_to_ent_rels[p].append(r_string_2)
            except KeyError:
                pairs_to_ent_rels[p] = [r_string_1, r_string_2]

    # Sort pairs by length
    if len(pairs_to_ent_rels) > 1:
        pairs = list(pairs_to_ent_rels.keys())
        pairs.sort(key=lambda x: x[1]-x[0], reverse=True)
        len_last = pairs[-1][1] - pairs[-1][0]
        for i in range(len(pairs)):
            len_i  = pairs[i][1] - pairs[i][0]
            if len_i == len_last:
                break
            for j in range(i+1, len(pairs)):
                len_j = pairs[j][1] - pairs[j][0]
                if len_i > len_j and pairs[i][0] <= pairs[j][0] and pairs[i][1] >= pairs[j][1]:
                    # pairs[i] contains pairs[j] -> include all entities and relations
                    # from pairs[j] in pairs[i]
                    pairs_to_ent_rels[pairs[i]].extend(pairs_to_ent_rels[pairs[j]])

    return sentences, pairs_to_ent_rels

def _print_entities_and_relations_per_sentence(sentences, pairs_to_ent_rels):
    # Print entities and relations
    for p, ent_rels in pairs_to_ent_rels.items():
        print('-'*80)
        print(f'[{p[0]}-{p[1]}]')
        for i in range(p[0], p[1]+1):
            print(f'\t{sentences[i]}')
        print('Entities and relations:')
        for er in ent_rels:
            print(f'\t{er}')

def print_entities_and_relations_per_sentence(data):
    
    text = data['text']
    if 'entities' in data:
        entities = data['entities']
        sentences, pairs_to_ent_rels = _extract_entities_and_relations_per_sentence(text, entities)
        _print_entities_and_relations_per_sentence(sentences, pairs_to_ent_rels)
    else:
        for key in ('labeler_1', 'labeler_2'):
            assert key in data
            entities = data[key]['entities']
            sentences, pairs_to_ent_rels = _extract_entities_and_relations_per_sentence(text, entities)
            print(f'Labeler: {key}')
            _print_entities_and_relations_per_sentence(sentences, pairs_to_ent_rels)

def _append_entities_and_relations_per_sentence(sentences, pairs_to_ent_rels, output, hash2string, skip_empty_sentences=True):
    for p, ent_rels in pairs_to_ent_rels.items():
        assert 0 <= p[1] - p[0] <= 1
        if p[0] == p[1]:
            s = sentences[p[0]]
        else:
            s = ' '.join(sentences[p[0]:p[1]+1])
        if s.endswith(' .'):
            s = s[:-2] # remove trailing period
        label_set = set()
        for x in ent_rels:
            h = hash_string(x)
            if h in hash2string:
                assert hash2string[h] == x
            else:
                hash2string[h] = x
            label_set.add(h)
        if skip_empty_sentences and len(label_set) == 0:
            continue
        if len(label_set) > 0:
            assert len(s) > 0
        output.append((s, label_set))

def compute_label_set_per_sentence(data, hash2string):
    output = []
    text = data['text']
    if 'entities' in data:
        entities = data['entities']
        sentences, pairs_to_ent_rels = _extract_entities_and_relations_per_sentence(text, entities)
        _append_entities_and_relations_per_sentence(sentences, pairs_to_ent_rels, output, hash2string)
    else:
        for key in ('labeler_1', 'labeler_2'):
            assert key in data
            entities = data[key]['entities']
            sentences, pairs_to_ent_rels = _extract_entities_and_relations_per_sentence(text, entities)
            _append_entities_and_relations_per_sentence(sentences, pairs_to_ent_rels, output, hash2string)
    return output
