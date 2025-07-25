## Introduction
HiGNN is a well-designed hierarchical and interactive informative graph neural networks framework for predicting molecular property by utilizing a co-representation learning of molecular graphs and chemically synthesizable BRICS fragments. Meanwhile, a plug-and-play feature-wise attention block was first designed in HiGNN architecture to adaptively recalibrate atomic features after message passing phase. [HiGNN](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01099) has been accepted for publication in [Journal of Chemical Information and Modeling](https://pubs.acs.org/journal/jcisd8/).
![overview](https://github.com/idrugLab/hignn/blob/main/hignn.png)
Fig.1 The overview of HiGNN

## Requirements
This project is developed using python 3.7.10, and mainly requires the following libraries.
```txt
rdkit==2021.03.1
scikit_learn==1.1.1
torch==1.7.1+cu101
torch_geometric==1.7.1
torch_scatter==2.0.7
```
To install [requirements](https://github.com/idrugLab/hignn/blob/main/requirements.txt):
```txt
pip install -r requirements.txt
```

### Usage example
**For drug dataset**
```sh
cd ./HIGNN/clear source
python run.py --dataset bace

```
Table 1 Predictive performance results of HiGNN on the drug discovery-related benchmark datasets.

| Dataset | Split Type | Metric  | Chemprop | GCN   | GAT   | Attentive FP | HRGCN+ | XGBoost | HiGNN   |
|---------|------------|---------|----------|-------|-------|--------------|--------|---------|---------|
| BACE    | random     | ROC-AUC | **0.898**  | **0.898** | 0.886 | 0.876        | 0.891  | 0.889   | 0.890   |
|         | scaffold   | ROC-AUC | 0.857    |       |       |              |        |         | **0.882**   |
| HIV     | random     | ROC-AUC | 0.827    | **0.834** | 0.826 | 0.822        | 0.824  | 0.816   | 0.816   |
|         | scaffold   | ROC-AUC | 0.794    |       |       |              |        |         | **0.802**   |
| MUV     | random     | PRC-AUC | 0.053    | 0.061 | 0.057 | 0.038        | 0.082  | 0.068   | **0.186**   |
| Tox21   | random     | ROC-AUC | 0.854    | 0.836 | 0.835 | 0.852        | 0.848  | 0.836   | **0.856**   |
| ToxCast | random     | ROC-AUC | 0.764    | 0.770 | 0.768 | **0.794**        | 0.793  | 0.774   | 0.781   |
| BBBP    | random     | ROC-AUC | 0.917    | 0.903 | 0.898 | 0.887        | 0.926  | 0.926   | **0.932**   |
|         | scaffold   | ROC-AUC | 0.886    |       |       |              |        |         | **0.927**   |
| ClinTox  | random | ROC-AUC | 0.897  | 0.895 | 0.888 | 0.904  | 0.899  | 0.911  | **0.930**   |
| SIDER    | random | ROC-AUC | **0.658**  | 0.634 | 0.627 | 0.623  | 0.641  | 0.642  | 0.651   |
| FreeSolv | random | RMSE    | 1.009  | 1.149 | 1.304 | 1.091  | 0.926  | 1.025  | **0.915**   |
| ESOL     | random | RMSE    | 0.587  | 0.708 | 0.658 | 0.587  | 0.563  | 0.582  | **0.532**   |
| Lipo     | random | RMSE    | 0.563  | 0.664 | 0.683 | 0.553  | 0.603  | 0.574  | **0.549**   |

## Acknowledgments
The code was partly built based on [chemprop](https://github.com/chemprop/chemprop), [TrimNet](https://github.com/yvquanli/trimnet) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer). Thanks a lot for their open source codes!

