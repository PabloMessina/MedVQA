from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_vectors_by_consine_similarity(vectors, query_vector):
    """
    Rank vectors by cosine similarity to query vector
    :param vectors: a 2D array of vectors
    :param query_vector: query vector
    :return: an array of indices of vectors sorted by cosine similarity to query vector
    """
    similarities = cosine_similarity(query_vector.reshape(1, -1), vectors)[0]
    return np.argsort(similarities)[::-1]

def rank_vectors_by_dot_product(vectors, query_vector):
    """
    Rank vectors by dot product with query vector
    :param vectors: a 2D array of vectors
    :param query_vector: query vector
    :return: an array of indices of vectors sorted by dot product with query vector
    """
    similarities = np.dot(query_vector, vectors.T)
    return np.argsort(similarities)[::-1]

def dot_product_triplets_accuracy(vectors, A, P, N):
    """
    Compute accuracy of dot product triplets
    :param vectors: a 2D array of vectors
    :param A: indices of anchor vectors
    :param P: indices of positive vectors
    :param N: indices of negative vectors
    :return: accuracy of dot product triplets
    """
    A = vectors[A]
    P = vectors[P]
    N = vectors[N]
    AP = np.sum(A * P, axis=1)
    AN = np.sum(A * N, axis=1)
    return np.mean(AP > AN)
