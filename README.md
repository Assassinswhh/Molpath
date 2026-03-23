## Molpath

This repository is the official implementation of the paper  
"**Chain-aware graph neural network for molecular property prediction**".

---

## Introduction

A novel chain-aware graph neural network model, wherein messages pass along shortest paths with different preferences indicated by attention weights.
<img width="1001" alt="model1" src="https://github.com/user-attachments/assets/5ef5b33f-0838-463b-9677-6da0ecd2fca2">

MolPath is a molecular property prediction framework that explicitly models chain structures in molecular graphs to alleviate feature squashing in conventional GNNs. By combining shortest-path-based chain representation learning with IRDC and attentive pooling, it captures long-range dependencies more effectively and achieves strong performance on real-world datasets.
---

## Environment

It is recommended to use **Anaconda** or **Miniconda** to manage the environment.

### Requirements

- python >= 3.9.0
- pytorch >= 2.0
- torch_geometric >= 2.3.0
- rdkit >= 2022.09.1
- numpy == 1.23.5
- pandas == 1.5.3
- scipy == 1.11.4

---

## Datasets

we used eight benchmark datasets from MoleculeNet。

### Tasks

- **Classification Tasks**: evaluated using ROC-AUC  
- **Regression Tasks**: evaluated using RMSE

All datasets will be automatically downloaded and processed during the first run.


