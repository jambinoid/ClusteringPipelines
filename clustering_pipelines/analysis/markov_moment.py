from typing import Literal
from numpy.typing import ArrayLike
import numpy as np


# Presented this way to be used in matrix operations
_CRITEREA_COEFFICIENTS = {
    "quadratic": {
        3: np.array([[  2,  -7],
                     [ -7,   5]]),  # / 39
        4: np.array([[ 19,   6, -32],
                     [  6, -11, -23],
                     [-32, -23,  41]]),  # / 245,
        5: np.array([[ 19,  25,   6, -38],
                     [ 25,  10, -15, -50],
                     [  6, -15, -21, -12],
                     [-38, -50, -12,  76]]),  # / 435
    },
    "exponential": {
        3: np.array([[ 0.044302, -0.165955],
                     [-0.165955,  0.121650]]),
        4: np.array([[ 0.065630,  0.059300, -0.149700],
                     [ 0.059300, -0.049250, -0.172550],
                     [-0.149700, -0.172550,  0.226500]]),
        5: np.array([[ 0.005560,  0.071400,  0.078600, -0.073700],
                     [ 0.071400,  0.048300, -0.014475, -0.185000],
                     [ 0.078600, -0.014475, -0.095700, -0.144500],
                     [-0.073700, -0.185000, -0.144500,  0.309600]])
    }
}


def markov_moment_of_stop(
    features: np.ndarray,
    criterion: Literal["quadratic", "exponential"],
    n_nodes: Literal[3, 4, 5],
    q: float = 0.0
) -> int:
    """
    Calculates Markov moment of stop for agglomerative clustering dendrogram.

    Args:
        features (array): 1d-array of features calculated the stepwise
            dendrogram of agglomerative clustering.
        criterion (str): name of approximation-estimation criterion to use,
            must be one of the "quadratic", "exponential".
        n_nodes (int): amount of nodes to use in chosen
            aproximation-estimation criterion.
        q (float): trend coefficient to use in approximation-estimation
            criterion. Defaults to 0.

    Returns:
        int, index for stepwise dendrogram of Markov moment of stop.

    """
    features = features.copy()
    assert len(features.shape) == 1

    if criterion not in _CRITEREA_COEFFICIENTS:
        raise ValueError(
            f"`criterion` must be one of [" +
            ", ".join(k for k in _CRITEREA_COEFFICIENTS) +
            f"], got '{criterion}' instead."
        )
    elif n_nodes not in _CRITEREA_COEFFICIENTS[criterion]:
        raise ValueError(
            f"`n_nodes` must be one of [" +
            ", ".join(k for k in _CRITEREA_COEFFICIENTS[criterion]) +
            f"], got {n_nodes} instead."
        )
    coeffs = _CRITEREA_COEFFICIENTS[criterion][n_nodes]

    n_features = len(features)
    features += q * np.arange(1, n_features + 1)
    features = np.concatenate([
        features[np.newaxis, n_nodes-i:n_features-i+1]
        for i in range(n_nodes-1, 0, -1)
    ], axis=0) - features[:1-n_nodes]
    features = np.einsum("i...,j...", features, features)
    stopping_moment = np.argwhere(np.sum(coeffs * features, axis=(1, 2)) > 0)
    if stopping_moment.size:
        return stopping_moment[0, 0] + n_nodes - 2
    else:
        return n_features - 1


def _calculate_reversed_coeffs(
    a: ArrayLike,
    f: ArrayLike
) -> tuple[float, float, float]:
    """
    Args:
        a (array-like): 2d-array of coefficients of nodes from
            aprroximation-estimation criterion formula in terms of nodes
            in the following format:

                a = [[a_11, a_12, ..., a_1N],
                     [a_21, a_22, ..., a_2N],
                                  ...
                     [a_N1, a_N2, ..., a_NN]]
             
        f (array-like): array of normalized nodes for given step in
            the following format:
            
                f = [y_1, y_2, ..., y_N]

    Return:
        Three float numbers which are the coefficients of a quadratic
        aprroximation-estimation criterion formula in terms of
        trend coefficient.
    
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)

    assert len(a.shape) == 2
    assert a.shape[0] == a.shape[1] == len(f)

    i = np.arange(1, len(f) + 1)

    A = np.einsum("i,j", i, i)
    B = np.einsum("i,j", i, f) + np.einsum("i,j", f, i)
    C = np.einsum("i,j", f, f)
    return np.sum(a * A), np.sum(a * B), np.sum(a * C)


def get_stability_segments(
    features: np.ndarray,
    criterion: Literal["quadratic", "exponential"],
    n_nodes: Literal[3, 4, 5],
) -> tuple[list[int], list[tuple[float, float]]]:
    """
    Calculates staibility segments for Markov moment of stop for
    agglomerative clustering dendrogram.

    Args:
        features (array): 1d-array of features calculated the stepwise
            dendrogram of agglomerative clustering.
        criterion (str): name of approximation-estimation criterion to use,
            must be one of the "quadratic", "exponential".
        n_nodes (int): amount of nodes to use in chosen
            aproximation-estimation criterion.

    Returns:
        list of integers which means number of clusters for the given segment. 
        list of tuples, each tuple is an stability segment for corresponding
            number of clusters in following format: [q_lower, q_higher).
            That means, that second value of trend coefficient will provide
            further moment of stop. In the other word, `q_higher` for
            i-th moment of stop is equal to the `q_lower` of (i+1)-th
            moment of stop.

    """
    if criterion not in _CRITEREA_COEFFICIENTS:
        raise ValueError(
            f"`criterion` must be one of [" +
            ", ".join(k for k in _CRITEREA_COEFFICIENTS) +
            f"], got '{criterion}' instead."
        )
    elif n_nodes not in _CRITEREA_COEFFICIENTS[criterion]:
        raise ValueError(
            f"`n_nodes` must be one of [" +
            ", ".join(k for k in _CRITEREA_COEFFICIENTS[criterion]) +
            f"], got {n_nodes} instead."
        )
    coeffs = _CRITEREA_COEFFICIENTS[criterion][n_nodes]

    n_clusters = list()
    segments = list()
    prev_bigger_root = 0
    for step in range(n_nodes-1, len(features)):
        # Calculate discriminant
        A, B, C = _calculate_reversed_coeffs(
            coeffs, features[step-n_nodes+2:step+1]-features[step-n_nodes+1])
        D = (B * B - 4 * A * C)

        if D > 0:
            sqrt_d = np.sqrt(D)
            root1, root2 = (-B + sqrt_d) / (2 * A), (-B - sqrt_d) / (2 * A)
            lower_root, bigger_root = sorted((root1, root2))
            if bigger_root > prev_bigger_root:
                n_clusters.append(len(features) - step + 1)
                if lower_root < prev_bigger_root:
                    segments.append((prev_bigger_root, bigger_root))
                else:
                    segments.append((lower_root, bigger_root))
                prev_bigger_root = bigger_root
    
    return n_clusters, segments
