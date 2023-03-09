# ClusterX
A Novel Representation Learning-Based Deep Clustering Framework for Accurate Visual Inspection in Virtual Screening

## Introduction
Molecular clustering analysis has been developed to facilitate visual inspection in the process of structure-based virtual screening. However, traditional methods based on molecular fingerprints or molecular descriptors limit the accuracy of selecting active hit compounds, which may be attributed to the lack of representations of receptor structural and protein–ligand interaction during the clustering. Here, a novel deep clustering framework named ClusterX is proposed to learn molecular representations of protein-ligand complexes and clustering the ligands. In ClusterX, the graph was used to represent the protein–ligand complex, and the joint optimization can be used efficiently for learning the cluster-friendly features. This framework can provide a unique tool for cluster analysis and we hope that it will help computational medicinal chemists to make visual decisions.

## Requirements
The project mainly uses two python packages, ***RDKit***, ***dgllife*** and ***openselfsup***. RDKit is used to read the structure of the protein-ligand complex, dgllife for building molecular graphs and openselfsup is used to quickly build a deep clustering model. For more details on the specific version numbers of required packages, see Requirements.txt. 

## Getting Started
After installing the required dependencies, you can quickly start with the following command：
```
git clone https://github.com/ChenSikang/ClusterX.git
```

### 1. graph construct
The training data (ClusterX dataset) needs to be constructed through ***Featurizer.py***, and the molecular graph is output as a numpy file. The same steps are used for the data to be clusteredBefore running, please make sure that the dataset exists in the data_dir path. 
```
python Featurizer.py --data_dir 'your_data_path'
```

### 2. pre-training
We provide the pre-trained network parameters (pretrain_ClusterX.pkl), you can also use ClusterX dataset or your own training set to retrain a pre-trained model
```
python pretrain.py --data_dir 'your_train_data_path'
```

### 3.clustering(training)
After training, the clustering information will be output
```
python train.py --data_dir 'your_clustering_analysis_data_path' --pretrain_path 'your_pretrain_model_path'
```
Running following command will use model to predict the ***example_klifs_data.npy*** protein-ligand complex file in KLIFS database.

Before running, please make sure that the model file exists in the current path.
```
python train.py --data_dir './example_klifs_data.npy' --pretrain_path './pretrain_ClusterX.pkl'
```

### 4.visual inspection
ClusterX does not have the function of visualizing the binding conformation of the protein-ligand complex. You can import the initial protein-ligand complex file into ***Schrödinger***, ***discovery studio***，***MOE*** and other visualization software combined with the clustering information provided by ClusterX for visual inspection.

## Citation
If you use ClusterX in your work, please cite the following paper:

Sikang Chen, Jian Gao, Jiexuan Chen, Yufeng Xie, Zheyuan Shen, Lei Xu, Jinxin Che*, Jian Wu*, Xiaowu Dong* (2023) ClusterX: A Novel Representation Learning-Based Deep Clustering Framework for Accurate Visual Inspection in Virtual Screening. Briefings in Bioinformatics.

BibTeX entry:
```
@article{ClusterX,
  title={ClusterX: A Novel Representation Learning-Based Deep Clustering Framework for Accurate Visual Inspection in Virtual Screening.},
  author={Sikang Chen, Jian Gao, Jiexuan Chen, Yufeng Xie, Zheyuan Shen, Lei Xu, Jinxin Che, Jian Wu, Xiaowu Dong},
  journal={Briefings in Bioinformatics},
  year={2023}
}
```
