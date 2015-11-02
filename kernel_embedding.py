"""Computation of graph embeddings and graph kernels.

Author : Sandro Vega-Pons, Emanuele Olivetti
"""

import numpy as np
import networkx as nx
from gk_weisfeiler_lehman import GK_WL
from gk_shortest_path import GK_SP


def DCE_embedding(X, th=0.0):
    """
    Direct connection label embedding.
    """
    return np.where(X > th, X, 0.0)


def DR_embedding(X, th=0.0, K=1):
    """Dissimilarity representation based embedding.

    From: Richiardi, J.; Van De Ville, D.; Riesen, K.; Bunke, H.:
    Vector Space Embedding of Undirected Graphs with Fixed-cardinality
    Vertex Sequences for Classification, Proceddings of 20th
    International Conference on Pattern Recognition (ICPR),
    pp.902,905, 23-26, Aug. 2010.

    Parameters:
    ----------
    X: ndarray of dimensions (n, d), where n is the number of samples
       and d is the number lenght of the vector obtained after
       unfolding the upper triangular matrix of the adjancency matrix
       of each graph.  Dataset.
    th: float
       Threshold to be applied in the edge weights. Edges with weights
       below the given threshold are removed.
    K: int
       A value to be used when

    Return:
    ------
    X: ndarray
       Dataset embedded into a vector space

    """
    # Application of threshold and changing from similarities to
    # dissimilarities the weight edges values.
    X = np.where(X > th, np.max(X) - X, 0.0)
    XX = np.zeros((X.shape[0], X.shape[0]))
    for t, v in enumerate(X):
        for q, u in enumerate(X):
            aux = 0
            for i in range(len(v)):
                if v[i] == 0 or u[i] == 0:
                    aux += K
                else:
                    aux += np.abs(u[i] - v[i])

            XX[t, q] = aux

    return XX


def WL_K_embedding(X, th=0.):
    """Computation of Weisfeiler-Lehman graph kernel. The kernel matrix is
    used as an embedding.

    Parameters:
    ----------
    X: ndarray of dimensions (n, d), where n is the number of samples
       and d is the lenght of the vector obtained after unfolding the
       upper triangular matrix of the adjancency matrix of each graph.
       Dataset.

    th: float
       Threshold to be applied in the edge weights. Edges with weights
       below the given threshold are removed.

    Return:
    ------
    X: ndarray
       Dataset embedded into a vector space

    """

    dim = int(np.sqrt(X.shape[1]*2)+1)
    graphs = []
    for t, v in enumerate(X):
        # Compute adjacency matrix
        mat = np.zeros((dim, dim))
        cont = 0
        for i in range(dim-1):
            for j in range(i+1, dim):
                mat[i, j] = v[cont]
                mat[j, i] = v[cont]
                cont += 1

        # Applying the threshold and keeping binary edges
        adj_mat = np.where(mat > th, 1.0, 0)
        g = nx.from_numpy_matrix(adj_mat)
        graphs.append(g)

    gk_wl = GK_WL()
    XX = gk_wl.compare_list(graphs, node_label=False)
    return XX


def SP_K_embedding(X, th=0.):
    """Computation of Shortest_Path graph kernel. The kernel matrix is
    used as an embedding.

    Parameters:
    ----------
    X: ndarray of dimensions (n, d), where n is the number of samples
       and d is the lenght of the vector obtained after unfolding the
       upper triangular matrix of the adjancency matrix of each graph.
       Dataset.
    th: float
       Threshold to be applied in the edge weights. Edges with weights
       below the given threshold are removed.

    Return:
    ------
    X: ndarray
       Dataset embedded into a vector space

    """

    dim = int(np.sqrt(X.shape[1] * 2) + 1)
    graphs = []
    for t, v in enumerate(X):
        # Compute adjacency matrix
        mat = np.zeros((dim, dim))
        cont = 0
        for i in range(dim-1):
            for j in range(i+1, dim):
                mat[i, j] = v[cont]
                mat[j, i] = v[cont]
                cont += 1

        # Applying the threshold and keeping binary edges
        adj_mat = np.where(mat > th, 1.0, 0)
        g = nx.from_numpy_matrix(adj_mat)
        graphs.append(g)

    gk_sp = GK_SP()
    XX = gk_sp.compare_list(graphs)
    return XX
