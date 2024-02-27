# ICML 2024 Submission - Graph Multi-Similarity Learning for Molecular Property Prediction

This repository contains the code and supplementary materials for the paper submitted to ICML 2024 titled "Graph Multi-Similarity Learning for Molecular Property Prediction". 

## Abstract

Enhancing accurate molecular property predic-
tion relies on effective and proficient represen-
tation learning. It is crucial to incorporate di-
verse molecular relationships characterized by
multi-similarity (Wang et al., 2019) between
molecules. However, current molecular repre-
sentation learning methods fall short in exploring
multi-similarity and often underestimate the com-
plexity of relationships between molecules. Ad-
ditionally, previous multi-similarity approaches
require the specification of positive and negative
pairs to attribute distinct pre-defined weights to
different relative similarities, which can introduce
potential bias. In this work, we introduce Graph
Multi-Similarity Learning for Molecular Property
Prediction (GraphMSL) framework, along with a
novel approach to formulate a generalized multi-
similarity metric without the need to define pos-
itive and negative pairs. In each of the chemical
modality spaces (e.g., molecular depiction image,
fingerprint, NMR, and SMILES) under consid-
eration, we first define a self-similarity metric
(i.e., similarity between an anchor molecule and
another molecule), and then transform it into a
generalized multi-similarity metric for the anchor
through a pair weighting function. GraphMSL val-
idates the efficacy of the multi-similarity metric
across MoleculeNet datasets. Furthermore, these
metrics of all modalities are integrated into a mul-
timodal multi-similarity metric, which showcases
the potential to improve the performance. More-
over, the focus of the model can be redirected or
customized by altering the fusion function. Last
but not least, GraphMSL proves effective in drug
discovery evaluations through post-hoc analyses
of the learnt representations.

## Prerequisites

[Include any software dependencies or hardware requirements needed to run your code. For example:]

- Python 3.7 or higher
- PyTorch 1.12 or higher
- Pytorch
- rdkit 2023.3.2
- torch-cluster 1.6.1
- torch-geometric  2.3.1
- torch-geometric-temporal  0.0.7
- torch-scatter  2.1.1
- torch-sparse  0.6.17
- torch-spline-conv  1.2.2
- torchaudio  0.12.1
- torchmetrics  0.11.4
- GPU with CUDA support (recommended)

## Installation
You can install the required package by running the following command
```bash
pip install -r requirements.txt
```

## Pretraining
1. Specify the data file and the directory for fixed encoders for different modalities in loss function. that we are going to use in Multi-Similarity Calculation in "pretrained/Utils/KMGCLConfig.py".
2. run the following command to pretrain the graph encoder that we are going to use in the downstream tasks later.
   ```bash
   python3 pretrained/main_chemprop.py --graphMetric [metric selection] --alpha []
   ```
    a. In particular, we currently have the following choices for metric selection at the graph level: image, NMR, smiles, fingerprint, fusion_image, fusion_nmr, fusion_fingerprint, fusion_smiles, and fusion_average. The corresponding weight for each modality in each selection is shown in the table below. We will make the weights tunable by adding regularization in the next version.
    | Fused multimodal | $w_{SM}$ | $w_{N}$ | $w_{M}$ | $w_{F}$ |
    |-------------------|----------|---------|---------|---------|
    | smiles            | 1.00     | 0.00    | 0.00    | 0.00    |
    | nmr               | 0.00     | 1.00    | 0.00    | 0.00    |
    | image             | 0.00     | 0.00    | 1.00    | 0.00    |
    | fingerprint       | 0.00     | 0.00    | 0.00    | 1.00    |
    | fusion_smiles     | 0.70     | 0.10    | 0.10    | 0.10    |
    | fusion_nmr        | 0.10     | 0.70    | 0.10    | 0.10    |
    | fusion_image      | 0.10     | 0.10    | 0.70    | 0.10    |
    | fusion_fingerprint| 0.10     | 0.10    | 0.10    | 0.70    |
    | fusion_average    | 0.25     | 0.25    | 0.25    | 0.25    |
  
   b. For alpha, if it is 0 then only loss of graph-level is applied; if it is 1, then only the loss of node-level is applied. If not specified, then both level will be included.

3. Extract the encoder weights only. This will be initialized weight for the graph encoder when fintuning downstream tasks. 


## Fine tuning in downstreams 
1. Run the following command to do the finetuning for the downstream task:
   ```bash
   python3 finetune_updated.py --data_path [directory of the data csv file] --dataset_type [classification/regression] --save_dir [directory to save the results] --epochs [number of epochs to train] --batch_size [number of samples in each batech] --split_type [how to split the data. scaffold_balanced or random] --num_folds [number of folds for cross validation] --encoder_path [directory for the pretrained encoder]
   ```