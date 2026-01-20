from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import sys
import matplotlib.pyplot as plt
import importlib
import os
import importlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import KernelCenterer
import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import KernelPCA
from geomloss import SamplesLoss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import KernelPCA
import numpy as np
from scipy.spatial.distance import cdist
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from scipy.spatial.distance import cdist
import muon as mu
import numpy as np
import anndata as ad
import scanpy as sc
import numpy as np
import torch
from torch import optim

try:
    from geomloss import SamplesLoss  # pip install geomloss
except Exception as e:
    raise ImportError("This code requires 'geomloss'. Install via `pip install geomloss`.") from e
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph

def build_knn_distance_matrix(
    K,
    k=20,
    use_rkhs_weights=False,      # square kernel only: True => geodesics sum RKHS distances; False => hop counts
    normalize=True,
    disconnected="global_max",   # 'global_max' | 'nan' | None
    metric="correlation",        # for data-matrix case (SCOT default: correlation)
    mode="connectivity",         # 'connectivity' or 'distance' (SCOT supports both)
):
    """
    If K is square (n,n): treat as PSD kernel, build kNN on RKHS distances.
    If K is rectangular (n,d): treat as data matrix (cells x features), build kNN like SCOT.

    Returns
    -------
    SP : (n,n) ndarray
        All-pairs shortest-path distances (normalized if requested).
    A  : csr_matrix
        kNN adjacency (as produced by sklearn), interpreted as undirected (union) via directed=False in dijkstra.
    """
    X = np.asarray(K)
    n = X.shape[0]

    # -------------------------------
    # Case 1) Square: kernel matrix
    # -------------------------------
    if X.ndim == 2 and X.shape[1] == n:
        Kmat = X

        # Kernel -> RKHS distances: d^2_ij = K_ii + K_jj - 2 K_ij
        diag = np.clip(np.diag(Kmat), a_min=0.0, a_max=None)
        D2 = diag[:, None] + diag[None, :] - 2.0 * Kmat
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2, dtype=float)
        np.fill_diagonal(D, 0.0)

        # Binary kNN adjacency from precomputed distances
        A = kneighbors_graph(
            D, n_neighbors=k, mode="connectivity",
            metric="precomputed", include_self=False
        )

        # Graph for Dijkstra
        if use_rkhs_weights:
            G = A.multiply(D)   # edge weights are RKHS distances on existing edges
        else:
            G = A               # hop counts

        SP = dijkstra(csgraph=csr_matrix(G), directed=False, return_predecessors=False)

    # -----------------------------------------
    # Case 2) Rectangular: data matrix (SCOT)
    # -----------------------------------------
    else:
        # SCOT builds kNN on the data directly; graph can be connectivity or distance
        # include_self=True for connectivity, False for distance (matches SCOT)
        include_self = True if mode == "connectivity" else False

        A = kneighbors_graph(
            X, n_neighbors=k, mode=mode, metric=metric, include_self=include_self
        )

        # SCOT then computes shortest paths on this graph, treating it as undirected
        SP = dijkstra(csgraph=csr_matrix(A), directed=False, return_predecessors=False)

        # SCOT-style: cap disconnected distances by per-graph max then normalize by global max
        finite = np.isfinite(SP)
        if not finite.all():
            cap = np.nanmax(SP[finite])
            SP[~finite] = cap

    # --- Handle disconnected pairs (your policy) ---
    finite = np.isfinite(SP)
    if not finite.all():
        if disconnected == "global_max":
            cap = np.nanmax(SP[finite])
            SP[~finite] = cap
        elif disconnected == "nan":
            SP[~finite] = np.nan
        elif disconnected is None:
            pass
        else:
            raise ValueError("disconnected must be one of {'global_max','nan',None}")

    # --- Normalize to [0,1] if requested ---
    if normalize:
        finite = np.isfinite(SP)
        if finite.any():
            m = np.nanmax(SP[finite])
            if m > 0:
                SP = SP / m

    return SP, A




# def build_knn_distance_matrix(data, k=20, mode="connectivity", metric="correlation"):
#     """
#     Builds a kNN-based distance matrix for the given data by:
#       1. Constructing the kNN graph using kneighbors_graph.
#       2. Computing the all-pairs shortest paths via Dijkstra.
#       3. Handling unconnected components (infinite distances).
#       4. Normalizing distances by the global max distance.

    # Parameters
    # ----------
    # data : np.ndarray
    #     2D array of shape (n_samples, n_features).
    # k : int
    #     Number of neighbors to use for kNN graph.
    # mode : {'connectivity', 'distance'}
    #     - 'connectivity' returns a binary adjacency matrix.
    #     - 'distance' returns a weighted adjacency matrix of distances.
    # metric : str
    #     Distance metric for neighbors. Often 'euclidean' or 'correlation',
    #     but can be any valid metric recognized by `kneighbors_graph`.

    # Returns
    # -------
    # distance_matrix : np.ndarray
    #     A normalized (0 to 1) all-pairs shortest-path distance matrix
    #     of shape (n_samples, n_samples).
    # """
    # # If mode="connectivity", we allow self-loops (include_self=True)
    # # so each point has a 0 distance to itself.
    # # If mode="distance", usually we skip self-loops (include_self=False).
    # if mode == "connectivity":
    #     include_self = True
    # else:
    #     include_self = False

    # # 1. Construct the kNN graph (sparse matrix)
    # adjacency = kneighbors_graph(
    #     data,
    #     n_neighbors=k,
    #     mode=mode,
    #     metric=metric,
    #     include_self=include_self
    # )

    # # 2. Compute shortest paths via Dijkstra
    # shortest_path = dijkstra(csgraph=csr_matrix(adjacency), directed=False, return_predecessors=False)

    # # 3. Handle unconnected nodes (infinite distances)
    # max_val = np.nanmax(shortest_path[shortest_path != np.inf])
    # shortest_path[shortest_path > max_val] = max_val

    # # 4. Normalize so the maximum distance becomes 1
    # if shortest_path.max() > 0:  # Avoid dividing by zero if everything is zero
    #     shortest_path /= shortest_path.max()

    # return shortest_path,adjacency

def zscore_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer
import numpy as np

def compute_centered_rbf_kernel(
    X: np.ndarray
   
):
    """
    Compute RBF kernel + centered kernel for a feature matrix.

    Parameters
    ----------
    X : np.ndarray
        (n_cells, n_features), e.g. PCA embeddings.

    gamma : float, optional
        If provided, uses this gamma directly for the RBF kernel.

    gamma_fun : callable, optional
        Function to compute gamma as gamma_fun(X).
        If both gamma and gamma_fun are None, raises an error.

    Returns
    -------
    dict with keys:
        "gamma" : float
        "K_original" : np.ndarray, raw RBF kernel
        "K_centered" : np.ndarray, centered RBF kernel
    """
   
    
    gamma = calculate_gamma(X)
    # Compute RBF kernel
    K = rbf_kernel(X, gamma=gamma)

    # Center the kernel
    centerer = KernelCenterer().fit(K)
    K_center = centerer.transform(K)

    return {
        "gamma": gamma,
        "K_original": K,
        "K_centered": K_center,
        "Ori_Feature":X,
    }

def visualize_by_domain(mapped_K1, mapped_K2):
    """
    Visualize the data using PCA, color-coded by domain.

    Parameters:
        mapped_K1 (np.ndarray): Mapped kernel matrix for domain X.
        mapped_K2 (np.ndarray): Mapped kernel matrix for domain Y.
    """
    pca = PCA(n_components=2)
    combined_data = np.vstack([mapped_K1, mapped_K2])
    domain_labels = np.array(['X'] * len(mapped_K1) + ['Y'] * len(mapped_K2))

    # Perform PCA
    pca_result = pca.fit_transform(combined_data)

    # Plotting
    plt.figure(figsize=(6, 4))
    for i, label in enumerate(np.unique(domain_labels)):
        idx = domain_labels == label
        plt.scatter(pca_result[idx, 0], pca_result[idx, 1], label=f'Domain {label}', alpha=0.75, edgecolors='w')
    plt.title('PCA Visualization by Domain')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
def visualize_by_cell_type(mapped_K1, mapped_K2, cellTypes_X, cellTypes_y):
    """
    Visualize the data using PCA, color-coded by cell type using discrete categories.

    Parameters:
        mapped_K1 (np.ndarray): Mapped kernel matrix for domain X.
        mapped_K2 (np.ndarray): Mapped kernel matrix for domain Y.
        cellTypes_X (np.ndarray): Cell types for domain X.
        cellTypes_y (np.ndarray): Cell types for domain Y.
    """
    pca = PCA(n_components=2)
    combined_data = np.vstack([mapped_K1, mapped_K2])
    combined_labels = np.concatenate([cellTypes_X, cellTypes_y])

    # Find unique labels and assign colors

    # Perform PCA
    pca_result = pca.fit_transform(combined_data)
    unique_cell_types = np.unique(combined_labels)
    # Plotting
    for cell_type in unique_cell_types:
        indices = np.where(combined_labels == cell_type)[0]
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], alpha=0.7, label=f'Cell Type {int(cell_type)}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Map of Kernel Matrix by Cell Type')
    plt.legend()
    plt.show()
def visualize_by_cell_type_1(mapped_K1, mapped_K2, cellTypes_X, cellTypes_y):
    """
    Visualize the data using PCA, color-coded by cell type using discrete categories.

    Parameters:
        mapped_K1 (np.ndarray): Mapped kernel matrix for domain X.
        mapped_K2 (np.ndarray): Mapped kernel matrix for domain Y.
        cellTypes_X (np.ndarray): Cell types for domain X.
        cellTypes_y (np.ndarray): Cell types for domain Y.
    """
    pca = PCA(n_components=2)
    combined_data = np.vstack([mapped_K1, mapped_K2])
    combined_labels = np.concatenate([cellTypes_X, cellTypes_y])

    # Find unique labels and assign colors

    # Perform PCA
    pca_result = pca.fit_transform(combined_data)
    unique_cell_types = np.unique(combined_labels)
    # Plotting
    for cell_type in unique_cell_types:
        indices = np.where(combined_labels == cell_type)[0]
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], alpha=0.7, label=f'Cell Type {(cell_type)}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Map of Kernel Matrix by Cell Type')
    plt.legend()
    plt.show()

def plot_pca_map(kernel_matrix, cell_types):
    # Perform PCA on the kernel matrix
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(kernel_matrix)

    # Plot the first two principal components with colors based on cell types
    plt.figure(figsize=(8, 6))
    unique_cell_types = np.unique(cell_types)
    for cell_type in unique_cell_types:
        indices = np.where(cell_types == cell_type)[0]
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], alpha=0.7, label=f'Cell Type {int(cell_type)}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Map of Kernel Matrix by Cell Type')
    plt.legend()
    plt.show()

def plot_pca_map_1(kernel_matrix, cell_types):
    # Perform PCA on the kernel matrix
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(kernel_matrix)

    # Plot the first two principal components with colors based on cell types
    plt.figure(figsize=(8, 6))
    unique_cell_types = np.unique(cell_types)
    for cell_type in unique_cell_types:
        indices = np.where(cell_types == cell_type)[0]
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], alpha=0.7, label=f'Cell Type {(cell_type)}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Map of Kernel Matrix by Cell Type')
    plt.legend()
    plt.show()

# def generate_weight_matrix(S, p):
#     n = S.shape[0]
#     W = torch.zeros_like(S)

#     # For each row in the similarity matrix, find the indices of the p nearest neighbors (highest similarities)
#     for i in range(n):
#         # Get indices of the p largest similarities (excluding the diagonal)
#         nearest_indices = torch.argsort(S[i, :], descending=True)[:p+1]  # +1 because the largest will be itself
#         W[i, nearest_indices] = S[i, nearest_indices]

#     # Make W symmetric
#     W = torch.max(W, W.T)

#     return W
def generate_laplacian_matrix(A_torch: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    Generate a Laplacian matrix from a PyTorch adjacency tensor.

    If normalized=True, return the normalized Laplacian:
        L_norm = I - D^{-1/2} * A_sym * D^{-1/2}
    Otherwise, return the unnormalized Laplacian:
        L = D - A_sym

    Parameters
    ----------
    A_torch : torch.Tensor
        A square adjacency matrix (n x n). If it's not symmetric,
        we symmetrize it internally.
    normalized : bool
        Whether to use the normalized Laplacian formula.

    Returns
    -------
    L_torch : torch.Tensor
        A Laplacian matrix (n x n) in PyTorch.
    """
    # 1. Symmetrize adjacency to ensure an undirected graph
    A_sym = 0.5 * (A_torch + A_torch.T)

    # 2. Compute the degree vector (sum of rows)
    degrees = torch.sum(A_sym, dim=1)

    # 3. Construct unnormalized Laplacian if normalized=False
    if not normalized:
        D = torch.diag(degrees)
        L_unnorm = D - A_sym
        return L_unnorm

    # 4. Construct normalized Laplacian if normalized=True
    #    L_norm = I - D^{-1/2} * A_sym * D^{-1/2}
    #    We must handle zero degrees to avoid division by zero.
    deg_sqrt = torch.sqrt(degrees)
    deg_inv_sqrt = 1.0 / deg_sqrt
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  # handle zero-degree nodes

    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    I = torch.eye(A_sym.shape[0], dtype=A_sym.dtype, device=A_sym.device)

    # Normalized Laplacian
    L_norm = I - (D_inv_sqrt @ A_sym @ D_inv_sqrt)
    return L_norm
def generate_weight_matrix(S, p):
    """
    Generate a weighted k-NN adjacency matrix from the similarity matrix S.

    Args:
        S (torch.Tensor): n x n similarity matrix (larger means more similar).
        p (int): number of nearest neighbors to keep for each row.

    Returns:
        W (torch.Tensor): n x n adjacency matrix with original similarity values
                          for the top-p neighbors, and 0 otherwise.
    """
    n = S.shape[0]
    W = torch.zeros_like(S)

    for i in range(n):
        # Clone the row so we can modify it safely
        row = S[i, :].clone()
        # Exclude diagonal (self-similarity)
        row[i] = float("-inf")  # ensures it won't be chosen among top-p
        # Get the top-p similarities in this row
        values, indices = torch.topk(row, k=p)
        # Assign these similarity values to W
        W[i, indices] = 1

    # Make W symmetric by taking the max of W and W^T
    # (you could also average them if you prefer)
    W = torch.max(W, W.T)

    return W

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def plot_pca_map(kernel_matrix, cell_types):
    # Perform PCA on the kernel matrix
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(kernel_matrix)

    # Plot the first two principal components with colors based on cell types
    plt.figure(figsize=(8, 6))
    unique_cell_types = np.unique(cell_types)
    for cell_type in unique_cell_types:
        indices = np.where(cell_types == cell_type)[0]
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], alpha=0.7, label=f'Cell Type {int(cell_type)}')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Map of Kernel Matrix by Cell Type')
    plt.legend()
    plt.show()
# def transfer_accuracy(domain1, domain2, type1, type2, n):
# 	"""
# 	Metric from UnionCom: "Label Transfer Accuracy"
# 	"""
# 	knn = KNeighborsClassifier(n_neighbors=n)
# 	knn.fit(domain2, type2)
# 	type1_predict = knn.predict(domain1)
# 	# np.savetxt("type1_predict.txt", type1_predict)
# 	count = 0
# 	for label1, label2 in zip(type1_predict, type1):
# 		if label1 == label2:
# 			count += 1
# 	return count / len(type1)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def transfer_accuracy(domain1, domain2, type1, type2, n=5):
    """
    Label transfer accuracy (UnionCom-style) that handles label-set mismatch.

    Parameters
    ----------
    domain1 : np.ndarray  # query embeddings
    domain2 : np.ndarray  # reference embeddings
    type1   : array-like  # true labels for domain1
    type2   : array-like  # labels for domain2
    n       : int         # number of neighbors

    Returns
    -------
    accuracy : float
        Fraction of correctly predicted labels among overlapping classes.
    """
    # ensure numpy arrays
    type1 = np.asarray(type1)
    type2 = np.asarray(type2)

    # find overlapping labels
    shared_labels = np.intersect1d(np.unique(type1), np.unique(type2))
    if len(shared_labels) == 0:
        return np.nan  # or 0.0, but nan signals 'no overlap'

    # mask both domains to shared classes only
    mask1 = np.isin(type1, shared_labels)
    mask2 = np.isin(type2, shared_labels)

    domain1_sub = domain1[mask1]
    domain2_sub = domain2[mask2]
    type1_sub = type1[mask1]
    type2_sub = type2[mask2]

    # train KNN on domain2, predict domain1
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(domain2_sub, type2_sub)
    pred = knn.predict(domain1_sub)

    # accuracy within shared labels
    return np.mean(pred == type1_sub)

def rbf_kernel(X, gamma=None):

    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    return np.exp(-gamma * pairwise_sq_dists)

def calculate_gamma(data_normalized):
    # Calculate the pairwise Euclidean distances
    dists = euclidean_distances(data_normalized, data_normalized)

    # Calculate the median of the distances
    median_dist = np.median(dists)

    # Calculate and return gamma
    gamma = 1 / (2 * median_dist ** 2)
    return gamma

def set_seed(seed: int = 50) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_torch(
    x: np.ndarray | torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device != device:
            t = t.to(device)
        if requires_grad != t.requires_grad:
            t = t.detach().requires_grad_(requires_grad)
        return t
    t = torch.tensor(x, dtype=dtype or torch.float32, device=device, requires_grad=requires_grad)
    return t


def _maybe_dense(a) -> np.ndarray:
    """Converts scipy.sparse matrix or torch tensor to dense numpy array, otherwise returns np.array(a)."""
    try:
        import scipy.sparse as sp
        if sp.issparse(a):
            return a.toarray()
    except Exception:
        pass
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# -----------------------------#
# Graph Laplacian
# -----------------------------#

def generate_laplacian_matrix(A_torch: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    Generate (normalized) Laplacian from an adjacency/affinity matrix (dense torch tensor).

    If normalized:
        L = I - D^{-1/2} A_sym D^{-1/2}
    else:
        L = D - A_sym
    """
    # 1) Symmetrize (undirected)
    A_sym = 0.5 * (A_torch + A_torch.T)

    # 2) Degree vector
    degrees = torch.sum(A_sym, dim=1)

    if not normalized:
        D = torch.diag(degrees)
        return D - A_sym

    # Normalized Laplacian
    # Safe inverse sqrt: 0 for isolated nodes
    deg_inv_sqrt = torch.where(
        degrees > 0, degrees.pow(-0.5), torch.zeros_like(degrees)
    )
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    I = torch.eye(A_sym.shape[0], dtype=A_sym.dtype, device=A_sym.device)
    return I - (D_inv_sqrt @ A_sym @ D_inv_sqrt)



def foscttm_label_aware(
    X1, X2, y1, y2, metric="euclidean", return_details=True
):
    """
    Label-aware domain-averaged FOSCTTM.
    - Drops labels that are not shared by both domains.
    - For each shared label c, pairs the first min(n1_c, n2_c) cells
      in their existing order: (i_k^c in domain1) <-> (j_k^c in domain2).
    - Computes FOSCTTM on the union of all such pairs, using *only*
      cells from the shared-label subset as the 'other-domain' pool.

    Parameters
    ----------
    X1, X2 : array-like, shape (n1, d), (n2, d)
        Embeddings for domain 1 and domain 2.
    y1, y2 : array-like, shape (n1,), (n2,)
        Labels for domain 1 and domain 2.
    metric : str
        Distance metric for cdist (e.g., 'euclidean', 'cosine').
    return_details : bool
        If True, returns dict with per-pair scores and indices.

    Returns
    -------
    If return_details:
        {
          'per_pair_foscttm': list of per-pair domain-averaged FOSCTTM,
          'mean_foscttm': float,
          'n_pairs': int,
          'shared_labels': np.ndarray,
          'idx1_used': np.ndarray,   # indices in X1 used
          'idx2_used': np.ndarray    # indices in X2 used (paired order)
        }
    Else:
        mean_foscttm : float
    """
    X1 = np.asarray(X1); X2 = np.asarray(X2)
    y1 = np.asarray(y1); y2 = np.asarray(y2)

    # 1) Keep only shared labels
    shared = np.intersect1d(np.unique(y1), np.unique(y2))
    if shared.size == 0:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    # 2) Build paired indices by label, preserving existing order
    idx1_list, idx2_list = [], []
    for c in shared:
        i1 = np.flatnonzero(y1 == c)
        i2 = np.flatnonzero(y2 == c)
        m = min(len(i1), len(i2))
        if m == 0:
            continue
        idx1_list.append(i1[:m])
        idx2_list.append(i2[:m])

    if not idx1_list:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    idx1_used = np.concatenate(idx1_list)
    idx2_used = np.concatenate(idx2_list)

    # 3) Extract pooled subsets and compute cross-domain distances
    X1s = X1[idx1_used]               # shape (Np, d)
    X2s = X2[idx2_used]               # shape (Np, d)
    D12 = cdist(X1s, X2s, metric=metric)  # (Np x Np)
    D21 = D12.T                           # symmetric metric assumed

    # 4) FOSCTTM per pair:
    #    fraction of other-domain samples closer than the true match
    #    (strictly '<' as in the usual definition)
    Np = len(idx1_used)
    per_pair = []
    for k in range(Np):
        frac1 = (D12[k] < D12[k, k]).sum() / Np
        frac2 = (D21[k] < D21[k, k]).sum() / Np
        per_pair.append(0.5 * (frac1 + frac2))

    mean_score = float(np.mean(per_pair)) if per_pair else np.nan

    if return_details:
        return {
            "per_pair_foscttm": per_pair,
            "mean_foscttm": mean_score,
            "n_pairs": Np,
            "shared_labels": shared,
            "idx1_used": idx1_used,
            "idx2_used": idx2_used,
        }
    else:
        return mean_score




"""
Alignment module

This module implements a sinkhorn-OT alignment between two domains given
precomputed kernels K1 and K2. All hyperparameters are routed through
`AlignmentConfig` and passed into `run_alignment`. A grid wrapper
(`run_alignment_grid`) enumerates multiple configs cleanly.

You should provide these externally (do NOT define inside run_alignment):
- build_knn_distance_matrix(X, k, mode, metric) -> (dist, adjacency)
- accuracy_fn(mapped_X, mapped_Y, labels_X, labels_Y, k_for_accuracy) -> float or array
"""

import math
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import optim

try:
    from geomloss import SamplesLoss  # pip install geomloss
except Exception as e:
    raise ImportError("This code requires 'geomloss'. Install via `pip install geomloss`.") from e


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int = 50) -> None:
    """Set NumPy and PyTorch (CPU/GPU) RNG seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_torch(
    x: np.ndarray | torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert arraylike to torch tensor with target device/dtype/grad."""
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device != device:
            t = t.to(device)
        if requires_grad != t.requires_grad:
            t = t.detach().requires_grad_(requires_grad)
        return t
    return torch.tensor(x, dtype=dtype or torch.float32, device=device, requires_grad=requires_grad)


def _maybe_dense(a) -> np.ndarray:
    """Convert scipy.sparse or torch.Tensor to dense numpy; else np.asarray."""
    try:
        import scipy.sparse as sp
        if sp.issparse(a):
            return a.toarray()
    except Exception:
        pass
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# ---------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------

def generate_laplacian_matrix(A_torch: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    Compute (normalized) graph Laplacian on a dense torch adjacency/affinity.

    If `normalized`:
        L = I - D^{-1/2} A_sym D^{-1/2}
    else:
        L = D - A_sym
    """
    # Symmetrize
    A_sym = 0.5 * (A_torch + A_torch.T)

    # Degree
    degrees = torch.sum(A_sym, dim=1)

    if not normalized:
        D = torch.diag(degrees)
        return D - A_sym

    # Safe D^{-1/2}
    deg_inv_sqrt = torch.zeros_like(degrees)
    positive = degrees > 0
    deg_inv_sqrt[positive] = degrees[positive].pow(-0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    I = torch.eye(A_sym.shape[0], dtype=A_sym.dtype, device=A_sym.device)
    return I - (D_inv_sqrt @ A_sym @ D_inv_sqrt)


# ---------------------------------------------------------------------
# Initialization via Kernel PCA
# ---------------------------------------------------------------------

def initialize_vector(K: torch.Tensor, p: int, device: torch.device) -> torch.nn.Parameter:
    """
    Initialize embedding vectors via Kernel PCA on a precomputed kernel matrix.

    Parameters
    ----------
    K : torch.Tensor
        Kernel matrix (n x n), symmetric PSD preferred.
    p : int
        Number of components.
    device : torch.device
        Target device.

    Returns
    -------
    torch.nn.Parameter
        Parameter of shape (n, p) with eigenvectors scaled by 1/sqrt(lambda).
    """
    import numpy as np
    from sklearn.decomposition import KernelPCA

    K_cpu = K.detach().cpu().numpy()

    pca = KernelPCA(n_components=p, kernel="precomputed")
    _ = pca.fit_transform(K_cpu)

    eigenvectors_ = getattr(pca, "eigenvectors_", None)
    eigenvalues_  = getattr(pca, "eigenvalues_", None)

    # Fallback for sklearn variants
    if eigenvectors_ is None and hasattr(pca, "alphas_"):
        eigenvectors_ = pca.alphas_
    if eigenvalues_ is None and hasattr(pca, "lambdas_"):
        eigenvalues_ = pca.lambdas_

    if eigenvectors_ is None or eigenvalues_ is None:
        raise AttributeError(
            "KernelPCA object lacks eigenvectors_/eigenvalues_. Check sklearn version."
        )

    eigenvalues_ = np.array(eigenvalues_, dtype=np.float64)
    scaled = np.zeros_like(eigenvectors_, dtype=np.float64)
    non_zeros = np.flatnonzero(eigenvalues_)
    if non_zeros.size > 0:
        scaled[:, non_zeros] = eigenvectors_[:, non_zeros] / np.sqrt(eigenvalues_[non_zeros])

    scaled = torch.tensor(scaled, dtype=torch.float64, device=device)
    return torch.nn.Parameter(scaled)


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

@dataclass
class AlignmentConfig:
    # Embedding / model
    p: int = 8

    # Regularization weights
    lambda_topo: float = 1e-2    # weight for orthogonality penalties
    lambda_reg: float = 1e-4     # weight for Laplacian (graph) smoothness

    # Training
    iterations: int = 2000
    lr: float = 1e-5
    patience: int = 10
    print_every: int = 100

    # Sinkhorn / OT
    reach: float = 5.0
    scaling: float = 0.8
    blur: float = 0.01

    # Graph/Laplacian (used only if L1/L2 not passed)
    laplacian_normalized: bool = False
    k_for_graph: Optional[int] = None
    graph_metric: str = "correlation"
    graph_mode: str = "connectivity"

    # Evaluation (optional)
    k_for_accuracy: int = 5

    # Misc
    dtype: torch.dtype = torch.float64
    seed: int = 50
    device: Optional[str] = None  # "cuda" | "cpu" | None -> auto

    # Initialization (optional)
    init_K1: Optional[torch.Tensor] = None
    init_K2: Optional[torch.Tensor] = None


@dataclass
class GridConfig:
    """
    Parameter grid enumerations + shared training knobs for all runs.
    All run-time hyperparameters are mirrored into AlignmentConfig before each run.
    """
    p_values: Sequence[int] = (8,)
    k_values: Sequence[int] = (5,)
    lambda_topo_values: Sequence[float] = (1e-2,)
    lambda_reg_values: Sequence[float] = (1e-4,)
    reach_values: Sequence[float] = (5.0,)

    # Shared (not enumerated) knobs used to build each AlignmentConfig
    iterations: int = 2000
    lr: float = 1e-5
    patience: int = 10
    print_every: int = 100
    dtype: torch.dtype = torch.float64
    seed: int = 50
    scaling: float = 0.8
    blur: float = 0.01
    laplacian_normalized: bool = False
    graph_metric: str = "correlation"
    graph_mode: str = "connectivity"
    k_for_accuracy: int = 5
    device: Optional[str] = None  # "cuda" | "cpu" | None -> auto

    # Optional persistence
    save_path: Optional[str] = "alignment_tuning_results.pkl"


# ---------------------------------------------------------------------
# Metrics: label-aware FOSCTTM
# ---------------------------------------------------------------------

from scipy.spatial.distance import cdist

def foscttm_label_aware(
    X1, X2, y1, y2, metric: str = "euclidean", return_details: bool = True
):
    """
    Label-aware domain-averaged FOSCTTM.

    Keeps only shared labels; pairs cells within each shared label in order,
    and computes per-pair FOSCTTM using only the shared-label pool.

    Returns a dict if return_details else a scalar mean.
    """
    X1 = np.asarray(X1); X2 = np.asarray(X2)
    y1 = np.asarray(y1); y2 = np.asarray(y2)

    shared = np.intersect1d(np.unique(y1), np.unique(y2))
    if shared.size == 0:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    idx1_list, idx2_list = [], []
    for c in shared:
        i1 = np.flatnonzero(y1 == c)
        i2 = np.flatnonzero(y2 == c)
        m = min(len(i1), len(i2))
        if m:  # pair first m
            idx1_list.append(i1[:m])
            idx2_list.append(i2[:m])

    if not idx1_list:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    idx1_used = np.concatenate(idx1_list)
    idx2_used = np.concatenate(idx2_list)

    X1s = X1[idx1_used]
    X2s = X2[idx2_used]
    D12 = cdist(X1s, X2s, metric=metric)
    D21 = D12.T

    Np = len(idx1_used)
    per_pair = []
    for k in range(Np):
        frac1 = (D12[k] < D12[k, k]).sum() / Np
        frac2 = (D21[k] < D21[k, k]).sum() / Np
        per_pair.append(0.5 * (frac1 + frac2))

    mean_score = float(np.mean(per_pair)) if per_pair else np.nan
    if return_details:
        return {
            "per_pair_foscttm": per_pair,
            "mean_foscttm": mean_score,
            "n_pairs": Np,
            "shared_labels": shared,
            "idx1_used": idx1_used,
            "idx2_used": idx2_used,
        }
    return mean_score

"""
Alignment module

This module implements a sinkhorn-OT alignment between two domains given
precomputed kernels K1 and K2. All hyperparameters are routed through
`AlignmentConfig` and passed into `run_alignment`. A grid wrapper
(`run_alignment_grid`) enumerates multiple configs cleanly.

You should provide these externally (do NOT define inside run_alignment):
- build_knn_distance_matrix(X, k, mode, metric) -> (dist, adjacency)
- accuracy_fn(mapped_X, mapped_Y, labels_X, labels_Y, k_for_accuracy) -> float or array
"""

import math
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import optim

try:
    from geomloss import SamplesLoss  # pip install geomloss
except Exception as e:
    raise ImportError("This code requires 'geomloss'. Install via `pip install geomloss`.") from e


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int = 50) -> None:
    """Set NumPy and PyTorch (CPU/GPU) RNG seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_torch(
    x: np.ndarray | torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert arraylike to torch tensor with target device/dtype/grad."""
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device != device:
            t = t.to(device)
        if requires_grad != t.requires_grad:
            t = t.detach().requires_grad_(requires_grad)
        return t
    return torch.tensor(x, dtype=dtype or torch.float32, device=device, requires_grad=requires_grad)


def _maybe_dense(a) -> np.ndarray:
    """Convert scipy.sparse or torch.Tensor to dense numpy; else np.asarray."""
    try:
        import scipy.sparse as sp
        if sp.issparse(a):
            return a.toarray()
    except Exception:
        pass
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# ---------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------

def generate_laplacian_matrix(A_torch: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    Compute (normalized) graph Laplacian on a dense torch adjacency/affinity.

    If `normalized`:
        L = I - D^{-1/2} A_sym D^{-1/2}
    else:
        L = D - A_sym
    """
    # Symmetrize
    A_sym = 0.5 * (A_torch + A_torch.T)

    # Degree
    degrees = torch.sum(A_sym, dim=1)

    if not normalized:
        D = torch.diag(degrees)
        return D - A_sym

    # Safe D^{-1/2}
    deg_inv_sqrt = torch.zeros_like(degrees)
    positive = degrees > 0
    deg_inv_sqrt[positive] = degrees[positive].pow(-0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    I = torch.eye(A_sym.shape[0], dtype=A_sym.dtype, device=A_sym.device)
    return I - (D_inv_sqrt @ A_sym @ D_inv_sqrt)


# ---------------------------------------------------------------------
# Initialization via Kernel PCA
# ---------------------------------------------------------------------

def initialize_vector(K: torch.Tensor, p: int, device: torch.device) -> torch.nn.Parameter:
    """
    Initialize embedding vectors via Kernel PCA on a precomputed kernel matrix.

    Parameters
    ----------
    K : torch.Tensor
        Kernel matrix (n x n), symmetric PSD preferred.
    p : int
        Number of components.
    device : torch.device
        Target device.

    Returns
    -------
    torch.nn.Parameter
        Parameter of shape (n, p) with eigenvectors scaled by 1/sqrt(lambda).
    """
    import numpy as np
    from sklearn.decomposition import KernelPCA

    K_cpu = K.detach().cpu().numpy()

    pca = KernelPCA(n_components=p, kernel="precomputed")
    _ = pca.fit_transform(K_cpu)

    eigenvectors_ = getattr(pca, "eigenvectors_", None)
    eigenvalues_  = getattr(pca, "eigenvalues_", None)

    # Fallback for sklearn variants
    if eigenvectors_ is None and hasattr(pca, "alphas_"):
        eigenvectors_ = pca.alphas_
    if eigenvalues_ is None and hasattr(pca, "lambdas_"):
        eigenvalues_ = pca.lambdas_

    if eigenvectors_ is None or eigenvalues_ is None:
        raise AttributeError(
            "KernelPCA object lacks eigenvectors_/eigenvalues_. Check sklearn version."
        )

    eigenvalues_ = np.array(eigenvalues_, dtype=np.float64)
    scaled = np.zeros_like(eigenvectors_, dtype=np.float64)
    non_zeros = np.flatnonzero(eigenvalues_)
    if non_zeros.size > 0:
        scaled[:, non_zeros] = eigenvectors_[:, non_zeros] / np.sqrt(eigenvalues_[non_zeros])

    scaled = torch.tensor(scaled, dtype=torch.float64, device=device)
    return torch.nn.Parameter(scaled)


# ---------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------

@dataclass
class AlignmentConfig:
    # Embedding / model
    p: int = 8

    # Regularization weights
    lambda_topo: float = 1e-2    # weight for orthogonality penalties
    lambda_reg: float = 1e-4     # weight for Laplacian (graph) smoothness

    # Training
    iterations: int = 2000
    lr: float = 1e-5
    patience: int = 10
    print_every: int = 100

    # Sinkhorn / OT
    reach: float = 5.0
    scaling: float = 0.8
    blur: float = 0.01

    # Graph/Laplacian (used only if L1/L2 not passed)
    laplacian_normalized: bool = False
    k_for_graph: Optional[int] = None
    graph_metric: str = "correlation"
    graph_mode: str = "connectivity"

    # Evaluation (optional)
    k_for_accuracy: int = 5

    # Misc
    dtype: torch.dtype = torch.float64
    seed: int = 50
    device: Optional[str] = None  # "cuda" | "cpu" | None -> auto

    # Initialization (optional)
    init_K1: Optional[torch.Tensor] = None
    init_K2: Optional[torch.Tensor] = None


@dataclass
class GridConfig:
    """
    Parameter grid enumerations + shared training knobs for all runs.
    All run-time hyperparameters are mirrored into AlignmentConfig before each run.
    """
    p_values: Sequence[int] = (8,)
    k_values: Sequence[int] = (5,)
    lambda_topo_values: Sequence[float] = (1e-2,)
    lambda_reg_values: Sequence[float] = (1e-4,)
    reach_values: Sequence[float] = (5.0,)

    # Shared (not enumerated) knobs used to build each AlignmentConfig
    iterations: int = 2000
    lr: float = 1e-5
    patience: int = 10
    print_every: int = 100
    dtype: torch.dtype = torch.float64
    seed: int = 50
    scaling: float = 0.8
    blur: float = 0.01
    laplacian_normalized: bool = False
    graph_metric: str = "correlation"
    graph_mode: str = "connectivity"
    k_for_accuracy: int = 5
    device: Optional[str] = None  # "cuda" | "cpu" | None -> auto

    # Optional persistence
    save_path: Optional[str] = "alignment_tuning_results.pkl"


# ---------------------------------------------------------------------
# Metrics: label-aware FOSCTTM
# ---------------------------------------------------------------------

from scipy.spatial.distance import cdist

def foscttm_label_aware(
    X1, X2, y1, y2, metric: str = "euclidean", return_details: bool = True
):
    """
    Label-aware domain-averaged FOSCTTM.

    Keeps only shared labels; pairs cells within each shared label in order,
    and computes per-pair FOSCTTM using only the shared-label pool.

    Returns a dict if return_details else a scalar mean.
    """
    X1 = np.asarray(X1); X2 = np.asarray(X2)
    y1 = np.asarray(y1); y2 = np.asarray(y2)

    shared = np.intersect1d(np.unique(y1), np.unique(y2))
    if shared.size == 0:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    idx1_list, idx2_list = [], []
    for c in shared:
        i1 = np.flatnonzero(y1 == c)
        i2 = np.flatnonzero(y2 == c)
        m = min(len(i1), len(i2))
        if m:  # pair first m
            idx1_list.append(i1[:m])
            idx2_list.append(i2[:m])

    if not idx1_list:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    idx1_used = np.concatenate(idx1_list)
    idx2_used = np.concatenate(idx2_list)

    X1s = X1[idx1_used]
    X2s = X2[idx2_used]
    D12 = cdist(X1s, X2s, metric=metric)
    D21 = D12.T

    Np = len(idx1_used)
    per_pair = []
    for k in range(Np):
        frac1 = (D12[k] < D12[k, k]).sum() / Np
        frac2 = (D21[k] < D21[k, k]).sum() / Np
        per_pair.append(0.5 * (frac1 + frac2))

    mean_score = float(np.mean(per_pair)) if per_pair else np.nan
    if return_details:
        return {
            "per_pair_foscttm": per_pair,
            "mean_foscttm": mean_score,
            "n_pairs": Np,
            "shared_labels": shared,
            "idx1_used": idx1_used,
            "idx2_used": idx2_used,
        }
    return mean_score


# ---------------------------------------------------------------------
# Core single-run optimizer
# ---------------------------------------------------------------------

@dataclass
class AlignmentConfig:
    p: int = 8
    lambda_topo: float = 1e-2
    lambda_reg: float = 1e-4
    iterations: int = 2000
    lr: float = 1e-5
    reach: float = 5.0
    scaling: float = 0.8          # geomloss internal scaling
    blur: float = 0.01            # geomloss blur
    patience: int = 10
    print_every: int = 100
    dtype: torch.dtype = torch.float64
    seed: int = 50
    stop_when_lr_below: float = 1e-6

    # If None, K1/K2 are used for initialization
    init_K1: Optional[torch.Tensor] = None
    init_K2: Optional[torch.Tensor] = None


@dataclass
class GridConfig:
    p_values: Sequence[int] = (8,)
    k_values: Sequence[int] = (5,)
    lambda_topo_values: Sequence[float] = (1e-2,)
    lambda_reg_values: Sequence[float] = (1e-4,)
    reach_values: Sequence[float] = (5.0,)
    # Training knobs shared across runs
    iterations: int = 2000
    lr: float = 1e-5
    patience: int = 10
    print_every: int = 100
    dtype: torch.dtype = torch.float64
    seed: int = 50
    scaling: float = 0.8
    blur: float = 0.01
    # Optional file to persist results
    save_path: Optional[str] = "alignment_tuning_results.pkl"

import numpy as np
from scipy.spatial.distance import cdist

def foscttm_label_aware(
    X1, X2, y1, y2, metric="euclidean", return_details=True
):
    """
    Label-aware domain-averaged FOSCTTM.
    - Drops labels that are not shared by both domains.
    - For each shared label c, pairs the first min(n1_c, n2_c) cells
      in their existing order: (i_k^c in domain1) <-> (j_k^c in domain2).
    - Computes FOSCTTM on the union of all such pairs, using *only*
      cells from the shared-label subset as the 'other-domain' pool.

    Parameters
    ----------
    X1, X2 : array-like, shape (n1, d), (n2, d)
        Embeddings for domain 1 and domain 2.
    y1, y2 : array-like, shape (n1,), (n2,)
        Labels for domain 1 and domain 2.
    metric : str
        Distance metric for cdist (e.g., 'euclidean', 'cosine').
    return_details : bool
        If True, returns dict with per-pair scores and indices.

    Returns
    -------
    If return_details:
        {
          'per_pair_foscttm': list of per-pair domain-averaged FOSCTTM,
          'mean_foscttm': float,
          'n_pairs': int,
          'shared_labels': np.ndarray,
          'idx1_used': np.ndarray,   # indices in X1 used
          'idx2_used': np.ndarray    # indices in X2 used (paired order)
        }
    Else:
        mean_foscttm : float
    """
    X1 = np.asarray(X1); X2 = np.asarray(X2)
    y1 = np.asarray(y1); y2 = np.asarray(y2)

    # 1) Keep only shared labels
    shared = np.intersect1d(np.unique(y1), np.unique(y2))
    if shared.size == 0:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    # 2) Build paired indices by label, preserving existing order
    idx1_list, idx2_list = [], []
    for c in shared:
        i1 = np.flatnonzero(y1 == c)
        i2 = np.flatnonzero(y2 == c)
        m = min(len(i1), len(i2))
        if m == 0:
            continue
        idx1_list.append(i1[:m])
        idx2_list.append(i2[:m])

    if not idx1_list:
        out = dict(per_pair_foscttm=[], mean_foscttm=np.nan, n_pairs=0,
                   shared_labels=shared, idx1_used=np.array([], int), idx2_used=np.array([], int))
        return out if return_details else np.nan

    idx1_used = np.concatenate(idx1_list)
    idx2_used = np.concatenate(idx2_list)

    # 3) Extract pooled subsets and compute cross-domain distances
    X1s = X1[idx1_used]               # shape (Np, d)
    X2s = X2[idx2_used]               # shape (Np, d)
    D12 = cdist(X1s, X2s, metric=metric)  # (Np x Np)
    D21 = D12.T                           # symmetric metric assumed

    # 4) FOSCTTM per pair:
    #    fraction of other-domain samples closer than the true match
    #    (strictly '<' as in the usual definition)
    Np = len(idx1_used)
    per_pair = []
    for k in range(Np):
        frac1 = (D12[k] < D12[k, k]).sum() / Np
        frac2 = (D21[k] < D21[k, k]).sum() / Np
        per_pair.append(0.5 * (frac1 + frac2))

    mean_score = float(np.mean(per_pair)) if per_pair else np.nan

    if return_details:
        return {
            "per_pair_foscttm": per_pair,
            "mean_foscttm": mean_score,
            "n_pairs": Np,
            "shared_labels": shared,
            "idx1_used": idx1_used,
            "idx2_used": idx2_used,
        }
    else:
        return mean_score
# -----------------------------#
# Core single-run optimizer
# -----------------------------#

def run_alignment(
    K1: np.ndarray | torch.Tensor,
    K2: np.ndarray | torch.Tensor,
    *,
    # Graph/Laplacian options
    L1: Optional[np.ndarray | torch.Tensor] = None,
    L2: Optional[np.ndarray | torch.Tensor] = None,
    A1: Optional[np.ndarray | torch.Tensor] = None,
    A2: Optional[np.ndarray | torch.Tensor] = None,

    X_features: Optional[np.ndarray] = None,
    Y_features: Optional[np.ndarray] = None,
    k_for_graph: Optional[int] = 5,
    graph_metric: str = "correlation",
    graph_mode: str = "connectivity",

    # Evaluation hooks (you said you have these; keep them pluggable)
    labels_X: Optional[Sequence[Any]] = None,
    labels_Y: Optional[Sequence[Any]] = None,
  
    k_for_accuracy: int = 5,

    # Training config
    config: AlignmentConfig = AlignmentConfig(),

    # Device
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run a SINGLE alignment with given hyperparameters.

    You can provide either:
      - L1/L2 directly (graph Laplacians), OR
      - A1/A2 (adjacency/affinity) and we will build L1/L2, OR
      - X_features / Y_features with `build_knn_distance_matrix` and `k_for_graph`
        to construct adjacency -> Laplacian.

    Returns a dict with embeddings, losses, and metrics.
    """
    set_seed(config.seed)

    # Choose device/dtype
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = config.dtype

    # Convert kernels to torch (no grads for kernels)
    K1_t = _to_torch(K1, device=device, dtype=dtype, requires_grad=False)
    K2_t = _to_torch(K2, device=device, dtype=dtype, requires_grad=False)

    # --- Build/obtain Laplacians ---
    if L1 is None or L2 is None:
        if A1 is None or A2 is None:
            if X_features is not None and Y_features is not None and k_for_graph is not None:
                # User-provided function should return (dist_matrix, adjacency)
                dist1, adj1 = build_knn_distance_matrix(
                    X_features, k=k_for_graph, mode=graph_mode, metric=graph_metric
                )
                dist2, adj2 = build_knn_distance_matrix(
                    Y_features, k=k_for_graph, mode=graph_mode, metric=graph_metric
                )
                A1 = _maybe_dense(adj1)
                A2 = _maybe_dense(adj2)
            else:
                # Fall back: treat K as an affinity and build Laplacian directly
                A1 = K1_t.detach().clone().cpu().numpy()
                A2 = K2_t.detach().clone().cpu().numpy()

        A1_t = _to_torch(_maybe_dense(A1), device=device, dtype=dtype, requires_grad=False)
        A2_t = _to_torch(_maybe_dense(A2), device=device, dtype=dtype, requires_grad=False)
        L1_t = generate_laplacian_matrix(A1_t, normalized=False)
        L2_t = generate_laplacian_matrix(A2_t, normalized=False)
    else:
        L1_t = _to_torch(L1, device=device, dtype=dtype, requires_grad=False)
        L2_t = _to_torch(L2, device=device, dtype=dtype, requires_grad=False)

    # --- Initialization (Kernel PCA on centered/precomputed kernels) ---
    initK1 = config.init_K1 if config.init_K1 is not None else K1_t
    initK2 = config.init_K2 if config.init_K2 is not None else K2_t
    alpha = initialize_vector(initK1, config.p, device=device)
    beta  = initialize_vector(initK2, config.p, device=device)


    # --- Optimizer / scheduler ---
    optimizer = optim.Adam([alpha, beta], lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    ot_loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=config.blur,
        reach=config.reach,
        scaling=config.scaling
    )

    # --- Early stopping bookkeeping ---
    best: Dict[str, Any] = {
        "loss": math.inf,
        "iter": -1,
        "alpha": None,
        "beta": None,
        "mapped_X": None,
        "mapped_Y": None,
        "breakdown": None,
    }
    no_improve = 0

    # --- Training loop ---
    for it in range(1, config.iterations + 1):
        optimizer.zero_grad()

        mapped_X = K1_t @ alpha  # (N1, p)
        mapped_Y = K2_t @ beta   # (N2, p)

        # OT in embedding space (note: this is NOT GW; your earlier comment said GW)
        loss_ot = ot_loss_fn(mapped_X, mapped_Y)

        # Orthogonality penalties
        I = torch.eye(config.p, dtype=dtype, device=device)
        pen_alpha = torch.norm(alpha.T @ K1_t @ alpha - I, p='fro') ** 2
        pen_beta  = torch.norm(beta.T  @ K2_t @ beta  - I, p='fro') ** 2
        loss_ortho = pen_alpha + pen_beta

        # Graph Laplacian regularization
        loss_graph = torch.trace(mapped_X.T @ L1_t @ mapped_X) + torch.trace(mapped_Y.T @ L2_t @ mapped_Y)

        # Total
        total = loss_ot + config.lambda_topo * loss_ortho + config.lambda_reg * loss_graph
        total.backward()
        optimizer.step()

        scheduler.step(total.item())

        # Track best
        tval = float(total.item())
        if tval < best["loss"]:
            best["loss"] = tval
            best["iter"] = it
            best["alpha"] = alpha.detach().clone()
            best["beta"]  = beta.detach().clone()
            best["mapped_X"] = mapped_X.detach().cpu().numpy()
            best["mapped_Y"] = mapped_Y.detach().cpu().numpy()
            best["breakdown"] = {
                "ot": float(loss_ot.item()),
                "ortho": float(loss_ortho.item()),
                "graph": float(loss_graph.item()),
            }
            no_improve = 0
        else:
            no_improve += 1

        # Console log
        if it % config.print_every == 0 or it == 1:
            with torch.no_grad():
                lr_now = optimizer.param_groups[0]["lr"]
            msg = (
                f"[Iter {it:5d}] total={tval:.6f} | "
                f"OT={float(loss_ot.item()):.6f} | "
                f"Ortho={float(loss_ortho.item()):.6f} | "
                f"Graph={float(loss_graph.item()):.6f} | "
                f"lr={lr_now:.2e}"
            )
            # Optional accuracy (if hooks provided)
            if labels_X is not None and labels_Y is not None:
                acc = transfer_accuracy(
                    best["mapped_X"], best["mapped_Y"], labels_X, labels_Y, k_for_accuracy
                )
                if isinstance(acc, (np.ndarray, list, tuple)):
                    msg += f" | Acc={np.mean(acc):.4f}"
                else:
                    msg += f" | Acc={float(acc):.4f}"
            print(msg)

        # Early stopping
        current_lr = optimizer.param_groups[0]["lr"]

        if (
            it >=100
            and no_improve >= config.patience
        ) or (
            current_lr <= config.stop_when_lr_below
            and it >=100
        ):
            print(
                f"Early stopping at iter={it} "
                f"(best@{best['iter']} total={best['loss']:.6f})."
            )
            break

      

    # Final evaluation (optional)
    final_metrics: Dict[str, Any] = {}
    if labels_X is not None and labels_Y is not None:
        acc1 = transfer_accuracy(
            best["mapped_X"], best["mapped_Y"], labels_X, labels_Y, k_for_accuracy
        )
        acc2 = transfer_accuracy(
            best["mapped_Y"], best["mapped_X"], labels_Y, labels_X, k_for_accuracy
        )
        acc=(acc1+acc2)/2
        final_metrics["accuracy"] = float(np.mean(acc)) if np.ndim(acc) else float(acc)
        final_metrics["accuracy_raw"] = acc
        final_metrics['foscttm']=foscttm_label_aware(
    best["mapped_X"], best["mapped_Y"], labels_X, labels_Y, metric="euclidean", return_details=False
)


    # Package result
    result = {
        "mapped_X": best["mapped_X"],
        "mapped_Y": best["mapped_Y"],
        "alpha": best["alpha"],
        "beta": best["beta"],
        "final_loss": best["loss"],
        "loss_breakdown": best["breakdown"],
        "best_iter": best["iter"],
        "config": asdict(config),
        "metrics": final_metrics,
    }
    return result


# -----------------------------#
# Grid search wrapper (multi-run)
# -----------------------------#

def run_alignment_grid(
    K1: np.ndarray | torch.Tensor,
    K2: np.ndarray | torch.Tensor,
    *,
    # graph construction (optional)
    X_features: Optional[np.ndarray] = None,
    Y_features: Optional[np.ndarray] = None,
    graph_metric: str = "correlation",
    graph_mode: str = "connectivity",

    # labels + eval
    labels_X: Optional[Sequence[Any]] = None,
    labels_Y: Optional[Sequence[Any]] = None,

    k_for_accuracy: int = 5,

    # grid + training configs
    grid: GridConfig = GridConfig(),

    # device
    device: Optional[torch.device] = None,
) -> Dict[Tuple[int, int, float, float, float], Dict[str, Any]]:
    """
    Loop over p, k, lambda_topo, lambda_reg, reach.

    Returns:
      results[(p, k, lambda_topo, lambda_reg, reach)] = result_dict (same format as run_alignment)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[Tuple[int, int, float, float, float], Dict[str, Any]] = {}

    for p in grid.p_values:
        for k in grid.k_values:
            # Build graph once per k
            A1 = A2 = None
            if build_knn_distance_matrix is not None and X_features is not None and Y_features is not None:
                _, adj1 = build_knn_distance_matrix(X_features, k=k, mode=graph_mode, metric=graph_metric)
                _, adj2 = build_knn_distance_matrix(Y_features, k=k, mode=graph_mode, metric=graph_metric)
                A1 = _maybe_dense(adj1)
                A2 = _maybe_dense(adj2)

            for lambda_topo in grid.lambda_topo_values:
                for lambda_reg in grid.lambda_reg_values:
                    # if lambda_topo <= lambda_reg:
                    #     # Mirror your condition "only consider cases where lambda_topo > lambda_reg"
                    #     continue
                    for reach in grid.reach_values:
                        print(
                            f"\n>>> Hyperparams: p={p}, k={k}, lambda_topo={lambda_topo}, "
                            f"lambda_reg={lambda_reg}, reach={reach}"
                        )
                        cfg = AlignmentConfig(
                            p=p,
                            lambda_topo=lambda_topo,
                            lambda_reg=lambda_reg,
                            iterations=grid.iterations,
                            lr=grid.lr,
                            reach=reach,
                            scaling=grid.scaling,
                            blur=grid.blur,
                            patience=grid.patience,
                            print_every=grid.print_every,
                            dtype=grid.dtype,
                            seed=grid.seed,
                        )

                        res = run_alignment(
                            K1, K2,
                            A1=A1, A2=A2,
                          
                            X_features=X_features, Y_features=Y_features,
                            k_for_graph=k,
                            graph_metric=graph_metric,
                            graph_mode=graph_mode,
                            labels_X=labels_X, labels_Y=labels_Y,
                  
                            k_for_accuracy=k_for_accuracy,
                            config=cfg,
                            device=device
                        )

                        key = (p, k, lambda_topo, lambda_reg, reach)
                        results[key] = res

    # Optional save
    if grid.save_path:
        with open(grid.save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nGrid search complete. Saved {len(results)} runs to {grid.save_path}")

    return results

import numpy as np
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------------------------------------
# 1. Helper functions (prune_small_clusters & reconstruction_error_pruned)
#    (Use your existing definitions or import them if they are in a separate module)
# -----------------------------------------------------------

def prune_small_clusters(original_data, row_labels, col_labels, min_size=20):
    """
    Removes (prunes) all row or column indices whose assigned cluster
    has fewer than `min_size` members. Returns:
        pruned_data, pruned_row_labels, pruned_col_labels,
        row_keep_mask, col_keep_mask, row_removed_count, col_removed_count
    """
    # Count row-cluster frequencies
    row_cluster_counts = Counter(row_labels)
    # Keep only rows whose cluster size >= min_size
    row_keep_mask = np.array([row_cluster_counts[lab] >= min_size for lab in row_labels], dtype=bool)

    # Count col-cluster frequencies
    col_cluster_counts = Counter(col_labels)
    # Keep only columns whose cluster size >= min_size
    col_keep_mask = np.array([col_cluster_counts[lab] >= min_size for lab in col_labels], dtype=bool)

    # Prune the matrix
    pruned_data = original_data[row_keep_mask][:, col_keep_mask]
    pruned_row_labels = row_labels[row_keep_mask]
    pruned_col_labels = col_labels[col_keep_mask]

    row_removed_count = np.sum(~row_keep_mask)
    col_removed_count = np.sum(~col_keep_mask)

    return (pruned_data,
            pruned_row_labels,
            pruned_col_labels,
            row_keep_mask,
            col_keep_mask,
            row_removed_count,
            col_removed_count)

def reconstruction_error_pruned(data_pruned, row_labels_pruned, col_labels_pruned):
    """
    Builds an approximate matrix from the row/column cluster assignments
    and returns the Frobenius norm of the difference, using ONLY the pruned
    subset of rows/columns.
    """
    row_clusters = np.unique(row_labels_pruned)
    col_clusters = np.unique(col_labels_pruned)

    # Construct the approximate matrix
    approx = np.zeros_like(data_pruned, dtype=np.float64)
    for r_clust in row_clusters:
        r_mask = (row_labels_pruned == r_clust)
        for c_clust in col_clusters:
            c_mask = (col_labels_pruned == c_clust)
            block = data_pruned[r_mask][:, c_mask]
            if block.size == 0:
                continue
            block_mean = np.mean(block)
            approx[np.ix_(r_mask, c_mask)] = block_mean

    diff = data_pruned - approx
    err = np.linalg.norm(diff, 'fro')
    return err

# -----------------------------------------------------------
# 2. Hyperparameter search: number of clusters
# -----------------------------------------------------------
# from coclust.coclustering import CoclustSpecMod

def hyperparameter_search_cluster_count(ot_plan_array, min_size=20, cluster_range=range(2, 21)):
    """
    Loop over a range of cluster numbers, fit coclustering, prune small clusters,
    compute reconstruction error, and store the results.

    Parameters
    ----------
    ot_plan_array : np.ndarray
        The OT plan (or any data matrix) to cocluster.
    min_size : int
        Minimum cluster size for pruning.
    cluster_range : range
        Range of cluster sizes to evaluate, e.g. range(2, 21).

    Returns
    -------
    cluster_values : list
        The list of cluster numbers (x-axis).
    errors : list
        The reconstruction error corresponding to each cluster number (y-axis).
    """
    errors = []
    cluster_values = []

    for k in cluster_range:
        # 1. Fit CoclustSpecMod
        print(f"Fitting CoclustSpecMod with {k} clusters...")
        model = CoclustSpecMod(n_clusters=k,random_state=0)
        model.fit(ot_plan_array)

        # 2. Prune small clusters
        predicted_row_labels = np.array(model.row_labels_)
        predicted_col_labels = np.array(model.column_labels_)
        (pruned_data,
         pruned_row_labels,
         pruned_col_labels,
         row_keep_mask,
         col_keep_mask,
         row_removed_count,
         col_removed_count) = prune_small_clusters(
             ot_plan_array,
             predicted_row_labels,
             predicted_col_labels,
             min_size=min_size
         )

        # If everything (or nearly everything) got removed, we might skip or set error to inf
        if pruned_data.size == 0:
            err = np.inf
        else:
            # 3. Compute reconstruction error on pruned data
            err = reconstruction_error_pruned(pruned_data, pruned_row_labels, pruned_col_labels)

        # Store results
        cluster_values.append(k)
        errors.append(err)

    return cluster_values, errors

def load_results(path_mappedK1="mappedK1.npy",
                 path_mappedK2="mappedK2.npy",
                 path_otplan="ot_plan.npy",
                 path_alpha="alpha.npy",
                 path_beta="beta.npy"):
    """
    Load the mapped embeddings, OT plan, alpha, and beta from disk in .npy format.

    Parameters
    ----------
    path_mappedK1 : str
        Filename for mapped K1.
    path_mappedK2 : str
        Filename for mapped K2.
    path_otplan : str
        Filename for the OT plan.
    path_alpha : str
        Filename for alpha.
    path_beta : str
        Filename for beta.

    Returns
    -------
    mapped_K1 : np.ndarray
        Loaded embeddings for domain 1.
    mapped_K2 : np.ndarray
        Loaded embeddings for domain 2.
    ot_plan : np.ndarray
        Loaded transport plan matrix.
    alpha : np.ndarray
        Loaded alpha array.
    beta : np.ndarray
        Loaded beta array.
    """
    mapped_K1 = np.load(path_mappedK1)
    mapped_K2 = np.load(path_mappedK2)
    ot_plan = np.load(path_otplan)
    alpha = np.load(path_alpha)
    beta = np.load(path_beta)

    return mapped_K1, mapped_K2, ot_plan, alpha, beta
# from coclust.coclustering import CoclustSpecMod

def hyperparameter_search_cluster_count(ot_plan_array, min_size=20, cluster_range=range(2, 21)):
    """
    Loop over a range of cluster numbers, fit coclustering, prune small clusters,
    compute reconstruction error, and store the results.

    Parameters
    ----------
    ot_plan_array : np.ndarray
        The OT plan (or any data matrix) to cocluster.
    min_size : int
        Minimum cluster size for pruning.
    cluster_range : range
        Range of cluster sizes to evaluate, e.g. range(2, 21).

    Returns
    -------
    cluster_values : list
        The list of cluster numbers (x-axis).
    errors : list
        The reconstruction error corresponding to each cluster number (y-axis).
    """
    errors = []
    cluster_values = []

    for k in cluster_range:
        # 1. Fit CoclustSpecMod
        print(f"Fitting CoclustSpecMod with {k} clusters...")
        model = CoclustSpecMod(n_clusters=k,random_state=0)
        model.fit(ot_plan_array)

        # 2. Prune small clusters
        predicted_row_labels = np.array(model.row_labels_)
        predicted_col_labels = np.array(model.column_labels_)
        (pruned_data,
         pruned_row_labels,
         pruned_col_labels,
         row_keep_mask,
         col_keep_mask,
         row_removed_count,
         col_removed_count) = prune_small_clusters(
             ot_plan_array,
             predicted_row_labels,
             predicted_col_labels,
             min_size=min_size
         )

        # If everything (or nearly everything) got removed, we might skip or set error to inf
        if pruned_data.size == 0:
            err = np.inf
        else:
            # 3. Compute reconstruction error on pruned data
            err = reconstruction_error_pruned(pruned_data, pruned_row_labels, pruned_col_labels)

        # Store results
        cluster_values.append(k)
        errors.append(err)

    return cluster_values, errors


import numpy as np
import pandas as pd
import scanpy as sc
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
# from coclust.coclustering import CoclustSpecMod
import scipy.sparse as sp
import warnings

# ---------- helpers ----------
def _to_array(X, dtype=np.float64):
    if isinstance(X, np.matrix):
        return np.array(X, dtype=dtype)
    if sp.issparse(X):
        return X.toarray().astype(dtype, copy=False)
    return np.asarray(X, dtype=dtype)

def cluster_purity(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    clusters = np.unique(pred_labels)
    total = len(pred_labels)
    correct = 0
    for c in clusters:
        idx = np.where(pred_labels == c)[0]
        if idx.size == 0:
            continue
        most_common = Counter(true_labels[idx]).most_common(1)[0][1]
        correct += most_common
    return correct / total if total > 0 else np.nan

# ---------- main (no prune) ----------
def run_coclust_full_pipeline(
    ot_plan,
    mapped_K1,               # (R x d1) embedding for row entities
    mapped_K2,               # (C x d2) embedding for col entities
    n_clusters=14,
    metric_labels_true=None, # length R+C, rows first then cols
    cluster_grid=None,       # e.g. range(8, 21) to select best k by ARI then NMI
    neighbors_k=15,
    do_umap=True,
    random_state=0,
    suppress_future_warnings=True,
    verbose=True,
):
    """
    End-to-end CoclustSpecMod without pruning.
    Returns:
      dict with adata, joint_labels, metrics, chosen n_clusters, and optional grid summary.
    """
    # Optional: silence the sklearn deprecation warning you saw
    if suppress_future_warnings:
        warnings.filterwarnings(
            "ignore",
            message="'force_all_finite' was renamed to 'ensure_all_finite'"
        )
        warnings.filterwarnings("ignore", category=FutureWarning)

    # Ensure arrays
    ot_plan  = _to_array(ot_plan)
    mapped_K1 = _to_array(mapped_K1)
    mapped_K2 = _to_array(mapped_K2)

    R, C = mapped_K1.shape[0], mapped_K2.shape[0]
    if verbose: print(f"[INFO] Rows={R}, Cols={C}, Embedding dims: K1={mapped_K1.shape[1]}, K2={mapped_K2.shape[1]}")

    # Optional cluster sweep (requires ground truth to score)
    results_grid = None
    if cluster_grid is not None:
        if metric_labels_true is None:
            raise ValueError("cluster_grid provided but metric_labels_true is None. Provide labels to score.")
        grid_rows = []
        for k in cluster_grid:
            if verbose: print(f"Fitting CoclustSpecMod with {k} clusters...")
            mk = CoclustSpecMod(n_clusters=k, random_state=random_state)
            mk.fit(ot_plan)
            # full joint labels, no pruning
            joint_k = np.concatenate([np.array(mk.row_labels_), np.array(mk.column_labels_)], axis=0)
            ari_k = adjusted_rand_score(metric_labels_true, joint_k)
            nmi_k = normalized_mutual_info_score(metric_labels_true, joint_k)
            grid_rows.append({"n_clusters": k, "ARI": ari_k, "NMI": nmi_k})
        results_grid = pd.DataFrame(grid_rows).sort_values(["ARI", "NMI", "n_clusters"], ascending=[False, False, True]).reset_index(drop=True)
        n_clusters = int(results_grid.loc[0, "n_clusters"])
        if verbose:
            print("\nGrid search summary (top 10):")
            print(results_grid.head(10).to_string(index=False))
            print(f"\nSelected n_clusters = {n_clusters} (best by ARI, tie-break NMI)")

    # Final fit with chosen n_clusters
    if verbose: print(f"\n[FINAL] Fitting CoclustSpecMod with n_clusters={n_clusters}")
    model = CoclustSpecMod(n_clusters=n_clusters, random_state=random_state)
    model.fit(ot_plan)

    # Full labels (no pruning)
    row_labels = np.array(model.row_labels_)
    col_labels = np.array(model.column_labels_)
    joint_labels = np.concatenate([row_labels, col_labels], axis=0)

    # Full joint embedding (rows first, then cols)
    full_concat_embedding = np.concatenate([mapped_K1, mapped_K2], axis=0)

    # Build AnnData & neighbors/UMAP
    adata = sc.AnnData(X=full_concat_embedding)
    adata.obs["bicluster"] = pd.Categorical(joint_labels)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=neighbors_k)
    if do_umap:
        sc.tl.umap(adata, random_state=random_state)
        sc.pl.umap(adata, color="bicluster", size=30, alpha=1.0, title=f"UMAP (CoclustSpecMod, k={n_clusters}, no prune)")

    # Metrics on ALL points
    ari = nmi = purity = sil = None
    if metric_labels_true is not None:
        if len(metric_labels_true) != (R + C):
            raise ValueError("metric_labels_true must have length R + C (rows first, then cols).")
        ari = adjusted_rand_score(metric_labels_true, joint_labels)
        nmi = normalized_mutual_info_score(metric_labels_true, joint_labels)
        purity = cluster_purity(metric_labels_true, joint_labels)
        # Silhouette only valid if >1 cluster
        sil = silhouette_score(full_concat_embedding, joint_labels) if len(np.unique(joint_labels)) > 1 else np.nan
        if verbose:
            print(f"\n[Metrics | full set]")
            print(f"  ARI:        {ari:.3f}")
            print(f"  NMI:        {nmi:.3f}")
            print(f"  Purity:     {purity:.3f}")
            print(f"  Silhouette: {sil if np.isfinite(sil) else np.nan:.3f}")

    return {
        "adata": adata,
        "joint_labels": joint_labels,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "ari": ari,
        "nmi": nmi,
        "purity": purity,
        "silhouette": sil,
        "n_clusters_used": n_clusters,
        "results_grid": results_grid
    }
import numpy as np
import pandas as pd
import scanpy as sc
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.stats import zscore
import warnings

def cluster_purity(true_labels, pred_labels):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    clusters = np.unique(pred_labels)
    total = len(pred_labels)
    correct = 0
    for c in clusters:
        idx = np.where(pred_labels == c)[0]
        if idx.size == 0:
            continue
        true_sub = true_labels[idx]
        most_common = Counter(true_sub).most_common(1)[0][1]
        correct += most_common
    return correct / total if total > 0 else np.nan

def grid_cluster_louvain(
    adata_combined,
    true_labels,
    resolutions=(0.1, 0.3, 0.4, 0.5, 0.6, 1.0, 1.5),
    use_rep="X_integrated",
    neighbors_k=15,
    key_added="louvain_clusters",
    plot_umap=True,
    random_state=0,
):
    """
    Run Louvain across 'resolutions', compute ARI/NMI/Purity/Silhouette,
    aggregate to a composite score (mean z-score), pick best, plot UMAP, and return results.

    Returns
    -------
    results_df : pd.DataFrame
        One row per resolution with metrics and composite score.
    best_res : float
        Resolution with highest composite score.
    best_labels : pd.Series
        Cluster labels at best resolution (adata.obs[key_added]).
    """

    # --- Preconditions: neighbors + UMAP if we want to plot ---
    if use_rep not in adata_combined.obsm:
        raise ValueError(f"`use_rep='{use_rep}'` not found in adata_combined.obsm")

    if "neighbors" not in adata_combined.uns:
        sc.pp.neighbors(adata_combined, use_rep=use_rep, n_neighbors=neighbors_k, random_state=random_state)

    if plot_umap and "X_umap" not in adata_combined.obsm:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.tl.umap(adata_combined, random_state=random_state)

    # --- Sweep resolutions ---
    rows = []
    for res in resolutions:
        print(f"\nRunning Louvain with resolution={res} ...")
        sc.tl.louvain(adata_combined, resolution=res, key_added=key_added, random_state=random_state)
        pred = adata_combined.obs[key_added].astype(str).to_numpy()

        # External metrics
        ari = adjusted_rand_score(true_labels, pred)
        nmi = normalized_mutual_info_score(true_labels, pred)
        pur = cluster_purity(true_labels, pred)

        # Internal metric
        X = adata_combined.obsm[use_rep]
        sil = silhouette_score(X, pred) if len(np.unique(pred)) > 1 else np.nan

        n_clusters = len(np.unique(pred))
        print(f"  -> Clusters: {n_clusters} | ARI {ari:.3f} | NMI {nmi:.3f} | Purity {pur:.3f} | Silhouette {sil if np.isfinite(sil) else np.nan:.3f}")

        rows.append({
            "resolution": res,
            "n_clusters": n_clusters,
            "ARI": ari,
            "NMI": nmi,
            "Purity": pur,
            "Silhouette": sil
        })

    results_df = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)

    # --- Composite score: mean z-score across available metrics ---
    # Handle constant metrics (std=0) safely: zscore(..., nan_policy="omit") is not in scipy<=1.11, so do manual guard
    def safe_z(x):
        x = np.asarray(x, dtype=float)
        if np.all(~np.isfinite(x)) or np.nanstd(x) == 0:
            return np.full_like(x, np.nan, dtype=float)
        return (x - np.nanmean(x)) / np.nanstd(x)

    for col in ["ARI", "NMI", "Purity", "Silhouette"]:
        results_df[f"z_{col}"] = safe_z(results_df[col].values)

    results_df["Composite"] = results_df[[f"z_{m}" for m in ["ARI", "NMI", "Purity", "Silhouette"]]].mean(axis=1, skipna=True)

    # Pick best by Composite (tie-breaker: higher ARI, then fewer clusters)
    results_df = results_df.sort_values(by=["Composite", "ARI", "NMI"], ascending=[False, False, False]).reset_index(drop=True)
    best_res = results_df.loc[0, "resolution"]

    # --- Refit at best resolution and plot ---
    sc.tl.louvain(adata_combined, resolution=best_res, key_added=key_added, random_state=random_state)
    best_labels = adata_combined.obs[key_added].copy()

    if plot_umap:
        sc.pl.umap(adata_combined, color=key_added, size=30, alpha=1.0)

    # Sort results back by resolution for readability
    results_df = results_df.sort_values("resolution").reset_index(drop=True)

    # Print summary
    print("\nSummary of all resolutions:")
    for _, r in results_df.iterrows():
        print(
            f"Resolution={r['resolution']}, Clusters={int(r['n_clusters'])}, "
            f"ARI={r['ARI']:.3f}, NMI={r['NMI']:.3f}, Purity={r['Purity']:.3f}, "
            f"Silhouette={r['Silhouette']:.3f}, Composite={r['Composite']:.3f}"
        )
    print(f"\nBest resolution by Composite score: {best_res}")

    return results_df, best_res, best_labels

import numpy as np

def interpret_embedding_top_genes(
    adata,
    alpha,                 # shape (N, p): per-cell coefficients for each latent dim
    K_centered,            # shape (N, N): centered kernel over cells
    gamma,                 # scalar for the RBF-like gradient scaling
    pca_key="X_pca",       # adata.obsm key for PCA coordinates (N, D_pca)
    pcs_key="PCs",         # adata.varm key for loadings (n_vars, D_pca)
    top_k=20,              # how many genes to report per dimension
    print_top=True,        # whether to print a nice top-k list
):
    """
    End-to-end:
      1) Compute partial derivatives in PCA space for each latent dim using alpha, K_centered, gamma
      2) Chain rule back to original genes via adata.varm[pcs_key]
      3) Aggregate (mean absolute) per dimension
      4) Return (avg_abs_sens_genes, top_genes_by_dim)

    Returns
    -------
    avg_abs_sens_genes : np.ndarray, shape (p, n_genes)
        Mean absolute sensitivity per dimension per gene.
    top_genes_by_dim : dict[int, list[str]]
        dim -> top_k gene names (descending by score).
    """

    # ---- 0) Pull PCA coords and PC loadings with basic checks
    if pca_key not in adata.obsm:
        raise KeyError(f"{pca_key!r} not found in adata.obsm")
    if pcs_key not in adata.varm:
        raise KeyError(f"{pcs_key!r} not found in adata.varm")

    X_pca = adata.obsm[pca_key]          # (N, D_pca)
    PCs   = adata.varm[pcs_key]          # (n_genes, D_pca)
    gene_names = np.asarray(adata.var_names)

    N, D_pca = X_pca.shape
    n_genes, D_pca_check = PCs.shape
    if D_pca != D_pca_check:
        raise ValueError(f"PCA dims mismatch: X_pca has {D_pca}, PCs has {D_pca_check}")
    if K_centered.shape != (N, N):
        raise ValueError(f"K_centered must be (N,N), got {K_centered.shape}")
    if alpha.shape[0] != N:
        raise ValueError(f"alpha first dim must be N={N}, got {alpha.shape[0]}")

    p = alpha.shape[1]  # number of latent dimensions

    # ---- 1) Vectorized partials in PCA space (shape: N x p x D_pca)
    # For each dim d:
    #   partial_d = -2*gamma * [ X_pca * sum_j K_centered[j,i]*alpha[j,d]  - (K_centered^T * diag(alpha[:,d])) X_pca ]
    sensitivities_pca = np.zeros((N, p, D_pca), dtype=float)
    for d in range(p):
        # Weighted kernel along dim d
        w = alpha[:, d].reshape(-1, 1)            # (N,1)
        WeightedK = K_centered * w                # (N,N)
        sumWK   = WeightedK.sum(axis=0)           # (N,)
        sumWKX  = WeightedK.T @ X_pca             # (N, D_pca)
        partial = (X_pca * sumWK[:, None]) - sumWKX
        sensitivities_pca[:, d, :] = (-2.0 * gamma) * partial

    # ---- 2) Chain rule back to genes
    # sensitivities_pca: (N, p, D_pca) -> reshape (N*p, D_pca)
    sens_p_flat = sensitivities_pca.reshape(-1, D_pca)          # (N*p, D_pca)
    # PCs^T: (D_pca, n_genes)
    sens_g_flat = sens_p_flat @ PCs.T                           # (N*p, n_genes)
    sensitivities_genes = sens_g_flat.reshape(N, p, n_genes)    # (N, p, n_genes)

    # ---- 3) Aggregate mean absolute per dim
    avg_abs_sens_genes = np.mean(np.abs(sensitivities_genes), axis=0)  # (p, n_genes)

    # ---- 4) Top-k per dimension
    top_genes_by_dim = {}
    k = min(top_k, n_genes)
    for d in range(p):
        order = np.argsort(-avg_abs_sens_genes[d, :])[:k]
        top_genes = gene_names[order].tolist()
        top_genes_by_dim[d] = top_genes
        if print_top:
            print(f"\nDimension {d} — top {k} genes:")
            for r, gi in enumerate(order, 1):
                print(f"  {r:2d}. {gene_names[gi]:15s}  score={avg_abs_sens_genes[d, gi]:.6g}")

    return avg_abs_sens_genes, top_genes_by_dim


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_top_genes_by_dimension_sequential(
    avg_abs_sens_genes,
    gene_names,
    top_k=5
):
    """
    For each dimension d:
      1) Identify the top_k genes.
      2) Append those genes (in that dimension's order) to a master list,
         preserving the dimension grouping in consecutive rows.

    Rows in the final heatmap:
      D1: top_k genes
      D2: top_k genes
      ...
      Dp: top_k genes

    Columns are always the p dimensions (D1, D2, ..., Dp).

    Modifications:
      - Place x-axis ticks on top.
      - Draw ONE red bounding box per dimension around its top_k cells
        (i.e., rows) in the dimension's column.
    """
    # avg_abs_sens_genes shape = (p, n_genes)
    #   p = number of dimensions
    #   n_genes = number of genes
    p, n_genes = avg_abs_sens_genes.shape

    # Build the matrix rows and row labels
    rows = []
    row_labels = []

    # The order in which the rows are appended is:
    #   dimension 0's top_k, dimension 1's top_k, etc.
    # So dimension d's rows occupy [d*top_k : d*top_k + top_k]
    for d in range(p):
        # Sort genes by importance in dimension d (descending)
        order = np.argsort(avg_abs_sens_genes[d, :])[::-1]
        top_idx = order[:top_k]

        for idx in top_idx:
            row_data = avg_abs_sens_genes[:, idx]  # shape = (p,)
            rows.append(row_data)
            row_labels.append(f"D{d+1}_{gene_names[idx]}")

    # Stack all rows into one 2D array => (p * top_k) x p
    data_matrix = np.vstack(rows)

    # Column labels
    col_labels = [f"D {i+1}" for i in range(p)]

    # Create the heatmap
    plt.figure(
        figsize=(max(9, p * 0.4), max(8, (p * top_k) * 0.3))
    )
    ax = sns.heatmap(
        data_matrix,
        cmap="Blues",
        yticklabels=row_labels,
        xticklabels=col_labels,
        cbar_kws={"label": "Importance"}
    )

    # 1) Move x-axis (dimension) labels and ticks to the top
    ax.xaxis.set_label_position("top")  # Move x-axis label to the top
    ax.xaxis.tick_top()                # Move tick labels to the top
    # Optional: set an x-axis label on top if you like
    ax.set_xlabel("Dimension", fontsize=14)

    # 2) We typically invert the y-axis so that row 0 appears at the top
    #    in *matplotlib's* coordinate system.
    #    By default, sns.heatmap puts row 0 at the top visually,
    #    but that is actually y=0 at the top in display.
    #    For consistent rectangle coordinates, let's invert the axis:
    ax.invert_yaxis()

    # 3) Draw ONE red bounding box per dimension
    #    around the top_k cells in that dimension's column.
    #
    # Dimension d's rows go from rowStart to rowStart + top_k - 1
    # in the data_matrix (0-based).
    # We'll define one rectangle with:
    #   x = d
    #   y = rowStart
    #   width = 1
    #   height = top_k
    #
    # Because we've inverted the y-axis, row=0 is near the top,
    # and row increases going downward in standard coordinates.
    for d in range(p):
        row_start = d * top_k
        rect = Rectangle(
            (d, row_start),  # (x, y) bottom-left corner in data coords
            1,               # width
            top_k,           # height
            fill=False,
            edgecolor='red',
            linewidth=2.0
        )
        ax.add_patch(rect)

    # Tweak rotations, font sizes as needed
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Adjust colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Importance", rotation=270, labelpad=15)

    # Set a y-axis label if desired
    ax.set_ylabel("Top genes by dimension", fontsize=14)

    plt.tight_layout()
    plt.show()

    # If you need to save:
    # plt.savefig("top_genes_sequential_heatmap_with_single_box_per_dim.pdf",
    #             dpi=300, bbox_inches='tight')
def run_pathway_analysis_top_genes(top_genes_by_dim):
    """
    For each dimension in top_genes_by_dim, run a pathway/GO enrichment
    (g:Profiler), and return a dictionary of dimension->DataFrame.
    """
    from gprofiler import GProfiler
    import pandas as pd

    gp = GProfiler(return_dataframe=True)

    enrichment_dict = {}  # <-- Store each dim's enrichment results here

    for dim, gene_list in top_genes_by_dim.items():
        print(f"\n--- Pathway Analysis for Dimension {dim} ---")
        if not gene_list:
            print("  (No genes to analyze.)")
            continue

        results_df = gp.profile(
            organism='hsapiens',      # or 'mmusculus' for mouse, etc.
            query=gene_list,
            sources=['GO:BP','GO:MF','GO:CC','KEGG','REAC'],
            user_threshold=0.05,      # significance threshold for adjusted p-values
            all_results=True ,
            no_evidences=False# get everything, not just significant
        )

        if not results_df.empty:
            # Sort by p-value or adjusted p-value
            if "p_value" in results_df.columns:
                results_df = results_df.sort_values("p_value", ascending=True)
            elif "p_value_adj" in results_df.columns:
                results_df = results_df.sort_values("p_value_adj", ascending=True)
            else:
                print("  No 'p_value' or 'p_value_adj' column found.")

            # Print top 5 hits
            print(results_df.head(5))
        else:
            print("  No results returned for these genes.")

        # Store in the dictionary
        enrichment_dict[dim] = results_df

    return enrichment_dict
def write_peaks_to_bed(peaks_list, bed_filename):
    """
    Convert a list of 'chrX:start-end' strings into a standard BED file.
    """
    with open(bed_filename, 'w') as f:
        for interval_str in peaks_list:
            match = re.match(r"(chr[\w]+):(\d+)-(\d+)", interval_str)
            if match:
                chrom = match.group(1)
                start = match.group(2)
                end   = match.group(3)
                f.write(f"{chrom}\t{start}\t{end}\n")
import re
import subprocess

REFERENCE_GENOME = "genome_data/hg38.fa"  # Path to your full hg38 genome (unzipped)
BEDTOOLS_PATH = "/usr/bin/bedtools"       # Adjust if bedtools is installed elsewhere


def bed_to_fasta(bed_file, fa_file, genome_fa):
    """
    Convert BED -> FASTA using bedtools getfasta.
    """
    cmd = [
        BEDTOOLS_PATH, "getfasta",
        "-fi", genome_fa,
        "-bed", bed_file,
        "-fo", fa_file
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr)
        print("STDOUT:", proc.stdout)
        raise RuntimeError(f"bedtools getfasta failed (returncode={proc.returncode})")
    else:
        print(f"Success! Created {fa_file} from {bed_file}.")

def run_gimme_motifs_with_genome(fa_file, out_dir, genome_fa, keep_original_size=False):
    """
    Run gimme motifs on a FASTA foreground, but still specify -g genome.fa
    so that GC-based z-score normalization and background are done properly.

    If keep_original_size=True, pass '--size 0' so it won't forcibly center
    or clip to 200 bp. By default, gimme motifs uses size=200.
    """
    cmd = ["gimme", "motifs", fa_file, out_dir, "-g", genome_fa]

    if keep_original_size:
        # Avoid the default 200-bp resizing
        cmd += ["--size", "0"]

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr)
        print("STDOUT:", proc.stdout)
        raise RuntimeError(f"gimmeMotifs failed (returncode={proc.returncode})")
    else:
        print(f"Success! gimme motifs results are in: {out_dir}/report.html")
import re

def write_all_peaks_to_bed(adata, output_bed="all_atac_peaks.bed"):
    """
    Extracts all peak intervals from adata.var["interval"] and writes them to a BED file.
    Assumes each entry is like: chr6:44058491-44059286.
    """
    intervals = adata.var["interval"].tolist()

    with open(output_bed, "w") as f:
        for interval_str in intervals:
            # Parse the "chr:start-end" format
            match = re.match(r"(chr[\w]+):(\d+)-(\d+)", interval_str)
            if match:
                chrom = match.group(1)
                start = match.group(2)
                end   = match.group(3)
                # Write a line in BED format
                f.write(f"{chrom}\t{start}\t{end}\n")
            else:
                # If it doesn't match, handle or skip
                # (Could print a warning or raise an error)
                pass
