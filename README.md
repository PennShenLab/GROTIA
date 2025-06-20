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
The implementation is based on Python. To check each dataset, simply run the notebook under GROT/Tutorial folder.


### License
The MIT license is issued for this project.

### Contacts

- [Zexuan Wang](mailto:zxwang@sas.upenn.edu) 
- [Li Shen](mailto:li.shen@pennmedicine.upenn.edu) 
