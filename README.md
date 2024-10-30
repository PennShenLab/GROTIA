# Graph-Regularized Optimal Transport for Single-Cell Data Integration

This repository holds the official source codes of the **GROT** package for the paper [Graph-Regularized Optimal Transport for Single-Cell Data Integration]()

```
@article {QOT: Efficient Computation of Sample Level Distance Matrix from Single-Cell Omics Data through Quantized Optimal Transport
Zexuan Wang, Qipeng Zhan, Shu Yang, Shizhuo Mu, Jiong Chen, Sumita Garai, Patryk Orzechowski, Joost Wagenaar, Li Shen
bioRxiv 2024.02.06.578032; doi: https://doi.org/10.1101/2024.02.06.578032
}
```

### Abstract
Recent advancements in single-cell omics technologies have enabled comprehensive analyses of cellular processes. However, due to the lack of co-assay sequencing technologies, single-cell omics datasets often comprise unpaired observations with varying feature dimensions. In this study, we present Graph-Regularized Optimal Transport for Single-Cell Data Integration (GROT), a computational method that aligns different multi-omics datasets without requiring correspondence information. GROT achieves global alignment through optimal transport while preserving local structure via graph regularization. We demonstrate its effectiveness on four simulated datasets and two real-world datasets, achieving superior performance compared to current state-of-the-art unsupervised alignment methods.
### Data
Simulation Datasets could be found at the github folder under Simulation Datasets.
Real-world Datasets could be downloaded at github folder under Real-Word Datasets.


### Usage
The implementation is based on Python. To check each dataset, simply run the notebook under GROT/Tutorial folder.

### Contacts

- [Zexuan Wang](mailto:zxwang@sas.upenn.edu) 
- [Li Shen](mailto:li.shen@pennmedicine.upenn.edu) 
