import numpy as np

from medvqa.utils.constants import (
    CHEXPERT_CXR14_SYNONYMS,
    CHEXPERT_DATASET_ID,
    CHEXPERT_LABELS,
    CHEXPERT_VINBIG_SYNONYMS,
    CXR14_DATASET_ID,
    CXR14_LABELS,
    CXR14_VINBIG_SYNONYMS,
    VINBIG_DATASET_ID,
    VINBIG_LABELS,
)
from medvqa.utils.data_structures import UnionFind
from medvqa.utils.files_utils import get_cached_json_file

def deduplicate_indices(indices, report_ids):
    seen = set()
    indices_ = []
    for i in indices:
        if report_ids[i] not in seen:
            seen.add(report_ids[i])
            indices_.append(i)
    return indices_

def get_merged_findings(use_chexpert=True, use_cxr14=True, use_vinbig=True):
    assert use_chexpert or use_cxr14 or use_vinbig
    
    n = 0
    original_labels = []
    offsets = {}
    if use_chexpert:
        offsets[CHEXPERT_DATASET_ID] = n
        original_labels += CHEXPERT_LABELS
        n += len(CHEXPERT_LABELS)
    if use_cxr14:
        offsets[CXR14_DATASET_ID] = n
        original_labels += CXR14_LABELS
        n += len(CXR14_LABELS)
    if use_vinbig:
        offsets[VINBIG_DATASET_ID] = n
        original_labels += VINBIG_LABELS
        n += len(VINBIG_LABELS)   

    uf = UnionFind(n)
    
    def _merge_findings(use1, use2, synonyms, id1, id2, labels1, labels2):
        if use1 and use2:
            for a, b in synonyms:
                assert a in labels1
                assert b in labels2
                ai = offsets[id1] + labels1.index(a)
                bi = offsets[id2] + labels2.index(b)
                uf.unionSet(ai, bi)

    uses = [use_chexpert, use_cxr14, use_vinbig]
    ids = [CHEXPERT_DATASET_ID, CXR14_DATASET_ID, VINBIG_DATASET_ID]
    labels = [CHEXPERT_LABELS, CXR14_LABELS, VINBIG_LABELS]
    synonyms = [CHEXPERT_CXR14_SYNONYMS, CHEXPERT_VINBIG_SYNONYMS, CXR14_VINBIG_SYNONYMS]
    for i in range(len(uses)):
        for j in range(i+1, len(uses)):
            _merge_findings(uses[i], uses[j], synonyms[i+j-1], ids[i], ids[j], labels[i], labels[j])
    
    count = 0
    tmp = dict()
    labels_remapping = dict()
    final_labels = []
    for i in range(len(uses)):
        offset = offsets[ids[i]]        
        new_labels = [None] * len(labels[i])
        for j in range(len(labels[i])):
            try:
                new_labels[j] = tmp[uf.findSet(offset + j)]
            except KeyError:
                new_labels[j] = tmp[uf.findSet(offset + j)] = count
                final_labels.append(original_labels[uf.findSet(offset + j)])
                count += 1
        labels_remapping[ids[i]] = new_labels    
    
    return labels_remapping, final_labels

def adapt_label_matrix_as_merged_findings(label_matrix, n_findings, new_labels):
    print(f'Adapting label matrix of shape = {label_matrix.shape} ...')
    assert len(new_labels) < n_findings
    assert label_matrix.shape[1] == len(new_labels)
    n = label_matrix.shape[0]
    m = len(new_labels)
    new_matrix = np.zeros((n, n_findings), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if label_matrix[i][j] == 1:
                new_matrix[i][new_labels[j]] = 1
    print('   new_matrix.shape =', new_matrix.shape)
    print('   Done!')
    return new_matrix
        

def get_tfidf_matrix_from_qa_pairs(qa_adapted_reports_path, max_features=1000):

    from tqdm import tqdm
    qa_adapted_reports = get_cached_json_file(qa_adapted_reports_path)
    texts = [[] for _ in range(len(qa_adapted_reports['questions']))]
    for r in tqdm(qa_adapted_reports['reports']):
        s = r['sentences']
        for k, v in r['qa'].items():
            qid = int(k)
            for i in v:
                texts[qid].append(s[i])
    for i in tqdm(range(len(texts))):
        texts[i] = ' '.join(texts[i])

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    return {
        'questions': qa_adapted_reports['questions'],
        'tfidf_matrix': vectorizer.fit_transform(texts).toarray(),
    }

# Plot a hierarchical clustering dendrogram of questions and corresponding tfidf vectors
def plot_tfidf_dendrogram(tfidf_matrix, questions, figsize=(10, 20)):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.metrics.pairwise import cosine_similarity    

    dist = 1 - cosine_similarity(tfidf_matrix)
    linkage_matrix = linkage(dist, 'ward')
    fig, ax = plt.subplots(figsize=figsize)
    ax = dendrogram(linkage_matrix, orientation="right", labels=questions, leaf_font_size=12)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tight_layout()
    plt.show()