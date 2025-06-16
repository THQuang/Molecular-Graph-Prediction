# Molecular Property Prediction Models: HIGNN and TrimNet

This repository contains two state-of-the-art models for molecular property prediction:
1. **HIGNN (Hierarchical and Interactive Graph Neural Networks)**
2. **TrimNet (Triplet Message Neural Network)**

## Overview

### HIGNN
HIGNN is a hierarchical and interactive informative graph neural networks framework for predicting molecular properties. It utilizes co-representation learning of molecular graphs and chemically synthesizable BRICS fragments. The model features a plug-and-play feature-wise attention block to adaptively recalibrate atomic features after the message passing phase.

### TrimNet
TrimNet is a lightweight message passing neural network designed for multiple molecular property predictions. It can accurately complete various molecular property prediction tasks with significant parameter reduction, including quantum properties, bioactivity, physiology, and compound-protein interaction (CPI) prediction tasks.

## Installation

### Prerequisites
- Python 3.7.10 or higher
- CUDA-compatible GPU (recommended)

### Environment Setup

1. Clone this repository:
```bash
git clone <https://github.com/THQuang/Molecular-Graph-Prediction.git>
```

2. Install dependencies for HIGNN:
```bash
cd HIGNN
pip install -r requirements.txt
```

3. Install dependencies for TrimNet:
```bash
cd TrimNet
pip install -r requirements.txt
```

## Running the Models

### Running HIGNN

1. Navigate to the HIGNN source directory:
```bash
cd HIGNN/clear\ source
```

2. Run the model on a dataset (e.g., BACE):
```bash
python run.py --dataset bace
```

Available datasets for HIGNN:
- BACE
- HIV
- MUV
- Tox21
- ToxCast
- BBBP
- ClinTox
- SIDER
- FreeSolv
- ESOL
- Lipo

### Running TrimNet

1. Navigate to the TrimNet source directory:
```bash
cd TrimNet/source
```

2. Run the model on a dataset (e.g., BACE):
```bash
python run.py --dataset bace
```


## Citation

If you use these models in your research, please cite the respective papers:

### HIGNN
```bibtex
@article{hignn,
    title={HiGNN: Hierarchical and Interactive Graph Neural Networks for Molecular Property Prediction},
    journal={Journal of Chemical Information and Modeling},
    year={2022},
    doi={10.1021/acs.jcim.2c01099}
}
```

### TrimNet
```bibtex
@article{10.1093/bib/bbaa266,
    author = {Li, Pengyong and Li, Yuquan and Hsieh, Chang-Yu and Zhang, Shengyu and Liu, Xianggen and Liu, Huanxiang and Song, Sen and Yao, Xiaojun},
    title = "{TrimNet: learning molecular representation from triplet messages for biomedicine}",
    journal = {Briefings in Bioinformatics},
    year = {2020},
    month = {11},
    doi = {10.1093/bib/bbaa266}
}
```