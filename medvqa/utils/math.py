from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_vectors_by_cosine_similarity(vectors, query_vector):
    similarities = cosine_similarity(query_vector.reshape(1, -1), vectors)[0]
    return np.argsort(similarities)[::-1]

def rank_vectors_by_dot_product(vectors, query_vector):
    similarities = np.dot(query_vector, vectors.T)
    return np.argsort(similarities)[::-1]

def rank_vectors_by_euclidean_distance(vectors, query_vector):
    distances2 = np.sum((vectors - query_vector) ** 2, axis=1)
    return np.argsort(distances2)

def dot_product_triplets_accuracy(vectors, A, P, N):
    A = vectors[A]
    P = vectors[P]
    N = vectors[N]
    AP = np.sum(A * P, axis=1)
    AN = np.sum(A * N, axis=1)
    return np.mean(AP > AN)

def cosine_similarity_triplets_accuracy(vectors, A, P, N):
    A = vectors[A]
    P = vectors[P]
    N = vectors[N]
    AP = cosine_similarity(A, P)
    AN = cosine_similarity(A, N)
    return np.mean(AP > AN)

def euclidean_distance_triplets_accuracy(vectors, A, P, N):
    A = vectors[A]
    P = vectors[P]
    N = vectors[N]
    AP = np.sum((A - P) ** 2, axis=1)
    AN = np.sum((A - N) ** 2, axis=1)
    return np.mean(AP < AN)
