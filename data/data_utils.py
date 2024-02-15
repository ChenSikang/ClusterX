import torch
import os.path as osp
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader, DataListLoader, InMemoryDataset
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
# import schrodinger
from sklearn.cluster import KMeans, AgglomerativeClustering


class DockingDataset(Dataset):
    def __init__(self, data_file):
        super(DockingDataset, self).__init__()
        self.data_file = data_file
        self.data_dict = {}  # will use this to store data once it has been computed if cache_data is True
        # self.data_list = []  # will use this to store ids for data
        self.data_dict = np.load(data_file, allow_pickle=True).item()
        self.ligands = list(self.data_dict.keys())
        self.labels = [-1 for _ in range(len(self.ligands))]

    def __len__(self):
        return len(self.ligands)

    def __getitem__(self, item):
        (target, pose) = self.ligands[item]
        print((target, pose))
        ligand_graph = self.data_dict[(target, pose)]['ligand_graph']
        protein_graph = self.data_dict[(target, pose)]['protein_graph']
        complex_graph = self.data_dict[(target, pose)]['complex_graph']
        
        print(ligand_graph.adjacency_matrix())
        # ligand_data = Data(x=ligand_graph.ndata['h'], e=ligand_graph.edata['e'], adj=ligand_graph.adjacency_matrix().to_dense())
        # protein_data = Data(x=protein_graph.ndata['h'], e=protein_graph.edata['e'], adj=protein_graph.adjacency_matrix().to_dense())
        
        ligand_edge_index, ligand_edge_attr = ligand_graph.adjacency_matrix().indices, ligand_graph.adjacency_matrix().values
        protein_edge_index, protein_edge_attr = protein_graph.adjacency_matrix().indices, protein_graph.adjacency_matrix().values
        data = Data(x1=ligand_graph.ndata['h'], x2=protein_graph.ndata['h'], 
        e1=ligand_graph.edata['e'], e2=protein_graph.edata['e'], 
        adj1=ligand_graph.adjacency_matrix().to_dense(), adj2=protein_graph.adjacency_matrix().to_dense(),
        edge_index1=ligand_edge_index, edge_index2=protein_edge_index, 
        edge_attr1=ligand_edge_attr, edge_attr2=protein_edge_attr)
        
        return data

    def assign_labels(self, labels):
        assert len(self.labels) == len(labels), \
            "Inconsistent lenght of asigned labels, \
            {} vs {}".format(len(self.labels), len(labels))
        self.labels = labels[:]

if __name__ == '__main__':
    data_dir = ''
    dataset = DockingDataset('')
