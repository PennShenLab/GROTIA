# An Interpretable Graph-Regularized Optimal Transport Framework for Diagonal Single-Cell Integrative Analysis

This repository holds the official source codes of the **GROTIA** package for the paper [An Interpretable Graph-Regularized Optimal Transport Framework for Diagonal Single-Cell Integrative Analysis]()

### ✨ Abstract
Recent advancements in single-cell omics technologies have enabled detailed characterization of cellular processes. However, coassay sequencing technologies remain limited, resulting in un-paired single-cell omics datasets with differing feature dimensions. Here, we present GROTIA (Graph-Regularized Optimal Transport Framework for Diagonal Single-Cell Integrative Analysis), a computational method to align multi-omics datasets without requiring any prior correspondence information. GROTIA achieves global alignment through optimal transport while preserving local relationships via graph regularization. Additionally, our approach provides interpretability by deriving domain-specific feature importance from partial derivatives, highlighting key biological markers. We demonstrate GROTIA’s superior performance on four simulated and four real-world datasets, surpassing state-of-the-art unsupervised alignment methods and confirming the biological significance of the top features identified in each domain.

## ✨ About
| Capability | What it gives you |
|------------|------------------|
| **Interpretable embeddings** | Learns a shared latent space for RNA **and** ATAC; every gene/peak receives an importance score so you know *why* cells align. |
| **OT-based co-clustering** | Converts the optimal-transport plan into a cross-modal affinity matrix for robust cell clustering and trajectory analysis after integration. |

---

### ✨ Data
Simulation Datasets could be found at the [here](https://drive.google.com/drive/folders/1qc0U6GRJvn5BmjcYIarGQe1nlUSYmWqn?usp=drive_link).
Real-world Datasets could be downloaded at [here](https://drive.google.com/drive/folders/1DGE3PFmwwyKUYHn2xYdCWh-lwIwGODvB?usp=drive_link).


### ✨ Usage
The implementation is based on Python. To check each dataset, simply run the notebook.

GROTIA_Replicate.ipynb: End-to-end instructions to reproduce the integrative analysis pipeline and run each integration method.

GROT_Co_Clustering_Guide.ipynb: Guide to the downstream co-clustering analysis workflow.

GROT_Interpretable_Embedding_Guide.ipynb: Tutorial for generating and interpreting the interpretable embeddings.

### ✨ Example for multi-omics integration

Step 1: Before running alignment or optimization, each modality must be converted into a kernel representation.

In practice, this typically corresponds to using `adata.obsm['X_pca']` (or another low-dimensional representation) as input for each modality.

```python
# Compute centered RBF kernels for both modalities
# X_normalized and y_normalized should already be normalized feature matrices

res_x = compute_centered_rbf_kernel(
    X_normalized
)

res_y = compute_centered_rbf_kernel(
    y_normalized
)

```

Step 2: Define Hyperparameters and Grid Configuration This step defines the hyperparameters for the alignment method. Please refer to the paper for detailed explanations of each parameter.

Key parameters: `p`: latent dimensionality, `k`: number of neighbors for kNN graph construction, `lambda_topo`: strength of the orthogonality (topological) constraint, `lambda_reg`: graph regularization weight, `reach_values`: reach parameter for unbalanced optimal transport

```python
grid = GridConfig(
    p_values=[5],                      # latent dimension
    k_values=[5],                      # kNN size
    lambda_topo_values=[1, 1e-1, 1e-2, 1e-3],
    lambda_reg_values=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    reach_values=[0.1, 1.0, 5.0],       # unbalanced OT reach
    iterations=8000,
    lr=1e-3,
    patience=10,
    print_every=500,
    dtype=torch.float64,
    seed=50,
    save_path="alignment_tuning_results.pkl",
)
```

Step 3: Run Alignment Grid Search This step runs the full alignment procedure over the specified hyperparameter grid.

Inputs:`K_centered`: centered kernels used for kernel projection, `X_features`, `Y_features`: original kernels used to construct graph Laplacians, `labels_X`, `labels_Y`: optional cell-type labels (used only for evaluation)

The result contains alignment scores and optimization statistics
for each hyperparameter configuration.
```python
results = run_alignment_grid(
    res_x["K_centered"],
    res_y["K_centered"],
    X_features=res_x["K_original"],
    Y_features=res_y["K_original"],
    labels_X=cellTypes_X,
    labels_Y=cellTypes_y,
    grid=grid,
    device=device_str,
)
```

### ✨ License
The MIT license is issued for this project.

### ✨ Maintainers

- [Zexuan Wang](mailto:zxwang@sas.upenn.edu) 
- [Qipeng Zhan](mailto:qipengz@sas.upenn.edu) 
