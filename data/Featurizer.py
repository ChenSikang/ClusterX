 import os
import warnings

import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem
from prody import *

import dgl
import dgl.backend as F
from dgl import graph, heterograph, batch

from dgl.data.utils import save_graphs, load_graphs
from dgllife.utils.mol_to_graph import k_nearest_neighbors, mol_to_bigraph
from dgllife.utils.featurizers import one_hot_encoding, BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer, \
    atom_type_one_hot, atom_degree_one_hot, atom_total_degree_one_hot, atom_total_num_H_one_hot, atom_hybridization_one_hot, \
    atom_formal_charge, atom_num_radical_electrons, atom_formal_charge_one_hot, atom_implicit_valence_one_hot, atom_explicit_valence_one_hot, atom_is_aromatic, \
    bond_type_one_hot, bond_is_in_ring, bond_is_conjugated, bond_stereo_one_hot

# from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
#     atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
#     one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
#     bond_is_in_ring, bond_stereo_one_hot
# from dgl.data.chem import BaseBondFeaturizer

from torchani import SpeciesConverter, AEVComputer
import multiprocessing
from functools import partial
from itertools import repeat
import argparse
from glob import glob
from tqdm import tqdm
# warnings.filterwarnings('ignore')


def int_to_one_hot(a, bins):
    """Convert integer encodings on a vector to a matrix of one-hot encoding"""
    n = len(a)
    b = np.zeros((n, len(bins)))
    b[np.arange(n), a] = 1
    return b


def filter_out_hydrogens(mol):
    """Get indices for non-hydrogen atoms."""
    indices_left = []
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        # Hydrogen atoms have an atomic number of 1.
        if atomic_num != 1:
            indices_left.append(i)
    return indices_left


def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
            [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                     encode_unknown=True)])})


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_

# claculate the 3D info for each directed edge


def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


def graphs_from_mol(m1, m2, dis_threshold=5, add_self_loop=False, add_3D=False):
    """
    :param m1: ligand molecule
    :param m2: pocket molecule
    :param add_self_loop: Whether to add self loops in DGLGraphs. Default to False.
    :return: 
    complex: graphs contain m1, m2 and complex
    """
    # the distance threshold to determine the interaction between ligand atoms and protein atoms
    dis_threshold = dis_threshold
    # small molecule
    new_order1 = rdmolfiles.CanonicalRankAtoms(m1)
    mol1 = rdmolops.RenumberAtoms(m1, new_order1)

    # pocket
    new_order2 = rdmolfiles.CanonicalRankAtoms(m2)
    mol2 = rdmolops.RenumberAtoms(m2, new_order2)

    # construct graphs
    g = dgl.DGLGraph()   # complex
    g1 = dgl.DGLGraph()  # small molecule
    g2 = dgl.DGLGraph()  # pocket

    # add nodes
    num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
    num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
    num_atoms = num_atoms_m1 + num_atoms_m2
    g.add_nodes(num_atoms)
    g1.add_nodes(num_atoms_m1)
    g2.add_nodes(num_atoms_m2)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)
        nodes1 = g1.nodes()
        g1.add_edges(nodes1, nodes1)
        nodes2 = g2.nodes()
        g2.add_edges(nodes2, nodes2)

    # add edges, ligand molecule
    num_bonds1 = mol1.GetNumBonds()
    src1 = []
    dst1 = []
    for i in range(num_bonds1):
        bond1 = mol1.GetBondWithIdx(i)
        u = bond1.GetBeginAtomIdx()
        v = bond1.GetEndAtomIdx()
        src1.append(u)
        dst1.append(v)
    src_ls1 = np.concatenate([src1, dst1])
    dst_ls1 = np.concatenate([dst1, src1])
    g.add_edges(src_ls1, dst_ls1)
    g1.add_edges(src_ls1, dst_ls1)

    # add edges, pocket
    num_bonds2 = mol2.GetNumBonds()
    src2 = []
    dst2 = []
    for i in range(num_bonds2):
        bond2 = mol2.GetBondWithIdx(i)
        u = bond2.GetBeginAtomIdx()
        v = bond2.GetEndAtomIdx()
        src2.append(u)
        dst2.append(v)
    src_ls2 = np.concatenate([src2, dst2])
    dst_ls2 = np.concatenate([dst2, src2])
    g.add_edges(src_ls2, dst_ls2)
    g2.add_edges(src_ls2, dst_ls2)

    # add interaction edges, only consider the euclidean distance within dis_threshold
    g3 = dgl.DGLGraph()
    g3.add_nodes(num_atoms)
    dis_matrix = distance_matrix(mol1.GetConformers(
    )[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    node_idx = np.where(dis_matrix < dis_threshold)
    src_ls3 = np.concatenate([node_idx[0], node_idx[1] + num_atoms_m1])
    dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1, node_idx[0]])
    g3.add_edges(src_ls3, dst_ls3)

    # assign atom features
    # 'h', features of atoms
    g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size(
        'h'), dtype=torch.float)  # init 'h'
    g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
    g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

    g1.ndata['h'] = AtomFeaturizer(mol1)['h']
    g2.ndata['h'] = AtomFeaturizer(mol2)['h']

    # assign edge features
    # 'd', distance between ligand atoms
    dis_matrix_L = distance_matrix(mol1.GetConformers(
    )[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
    g1_d = torch.tensor(
        dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

    # 'd', distance between pocket atoms
    dis_matrix_P = distance_matrix(mol2.GetConformers(
    )[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
    g2_d = torch.tensor(
        dis_matrix_P[src_ls2, dst_ls2], dtype=torch.float).view(-1, 1)

    # 'd', distance between ligand atoms and pocket atoms
    inter_dis = np.concatenate(
        [dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
    g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

    # efeats1
    g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size(
        'e'), dtype=torch.float)  # init 'e'
    efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat(
        [efeats1[::2], efeats1[::2]])
    g1.edata['e'] = torch.cat([efeats1[::2], efeats1[::2]])

    # efeats2
    efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
    g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat(
        [efeats2[::2], efeats2[::2]])
    g2.edata['e'] = torch.cat([efeats2[::2], efeats2[::2]])

    # 'e'
    g_d = torch.cat([g1_d, g2_d])
    g.edata['e'] = torch.cat([g.edata['e'], g_d * 0.1], dim=-1)
    g1.edata['e'] = torch.cat([g1.edata['e'], g1_d * 0.1], dim=-1)
    g2.edata['e'] = torch.cat([g2.edata['e'], g2_d * 0.1], dim=-1)
    g3.edata['e'] = g3_d * 0.1

    if add_3D:
        g.ndata['pos'] = torch.zeros(
            [g.number_of_nodes(), 3], dtype=torch.float)
        g.ndata['pos'][:num_atoms_m1] = torch.tensor(
            mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g.ndata['pos'][-num_atoms_m2:] = torch.tensor(
            mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
        g1.ndata['pos'] = torch.tensor(
            mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g2.ndata['pos'] = torch.tensor(
            mol2.GetConformers()[0].GetPositions(), dtype=torch.float)

        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            # the source node id and destination id of an edge
            tmp = [src_node, dst_nodes[i]]
            neighbors = g.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
        g.ndata.pop('pos')

        # calculate the 3D info for g1
        src_nodes, dst_nodes = g1.find_edges(range(g1.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            # the source node id and destination id of an edge
            tmp = [src_node, dst_nodes[i]]
            neighbors = g1.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g1), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g1.edata['e'] = torch.cat([g1.edata['e'], D3_info_th], dim=-1)

        # calculate the 3D info for g2
        src_nodes, dst_nodes = g2.find_edges(range(g2.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            # the source node id and destination id of an edge
            tmp = [src_node, dst_nodes[i]]
            neighbors = g2.predecessors(src_node).tolist()
            neighbors.remove(dst_nodes[i])
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g2), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g2.edata['e'] = torch.cat([g2.edata['e'], D3_info_th], dim=-1)
        g1.ndata.pop('pos')
        g2.ndata.pop('pos')

    return g, g1, g2, g3


def PN_graph_construction_and_featurization(ligand_mol,
                                            protein_mol,
                                            max_num_neighbors=4,
                                            distance_bins=[1.5, 2.5, 3.5, 4.5]):
    """Graph construction and featurization for `PotentialNet for Molecular Property Prediction
     <https://pubs.acs.org/doi/10.1021/acscentsci.8b00507>`__.
    Parameters
    ----------
    distance_bins : list of float
        Distance bins to determine the edge types.
        Edges of the first edge type are added between pairs of atoms whose distances are less than `distance_bins[0]`.
        The length matches the number of edge types to be constructed.
        Default `[1.5, 2.5, 3.5, 4.5]`.
    strip_hydrogens : bool
        Whether to exclude hydrogen atoms. Default to False.
    Returns
    -------
    complex_bigraph : DGLGraph
        Bigraph with the ligand and the protein (pocket) combined and canonical features extracted.
        The atom features are stored as DGLGraph.ndata['h'].
        The edge types are stored as DGLGraph.edata['e'].
        The bigraphs of the ligand and the protein are batched together as one complex graph.
    complex_knn_graph : DGLGraph
        K-nearest-neighbor graph with the ligand and the protein (pocket) combined and edge features extracted based on distances.
        The edge types are stored as DGLGraph.edata['e'].
        The knn graphs of the ligand and the protein are batched together as one complex graph.
    """

    ligand_coordinates = ligand_mol.GetConformers()[0].GetPositions()
    protein_coordinates = protein_mol.GetConformers()[0].GetPositions()

    # Node featurizer for stage 1
    atoms = ['H', 'N', 'O', 'C', 'P', 'S', 'F', 'Br', 'Cl', 'I', 'Fe',
             'Zn', 'Mg', 'Na', 'Mn', 'Ca', 'Co', 'Ni', 'Se', 'Cu', 'Cd', 'Hg', 'K']
    atom_total_degrees = list(range(5))
    atom_formal_charges = [-1, 0, 1]
    atom_implicit_valence = list(range(4))
    atom_explicit_valence = list(range(8))
    atom_concat_featurizer = ConcatFeaturizer([partial(atom_type_one_hot, allowable_set=atoms),
                                               partial(
                                                   atom_total_degree_one_hot, allowable_set=atom_total_degrees),
                                               partial(
                                                   atom_formal_charge_one_hot, allowable_set=atom_formal_charges),
                                               atom_is_aromatic,
                                               partial(
                                                   atom_implicit_valence_one_hot, allowable_set=atom_implicit_valence),
                                               partial(atom_explicit_valence_one_hot, allowable_set=atom_explicit_valence)])
    PN_atom_featurizer = BaseAtomFeaturizer({'h': atom_concat_featurizer})

    # Bond featurizer for stage 1
    bond_concat_featurizer = ConcatFeaturizer(
        [bond_type_one_hot, bond_is_in_ring])
    PN_bond_featurizer = BaseBondFeaturizer({'e': bond_concat_featurizer})

    # construct graphs for stage 1
    ligand_bigraph = mol_to_bigraph(ligand_mol, add_self_loop=False,
                                    node_featurizer=PN_atom_featurizer,
                                    edge_featurizer=PN_bond_featurizer,
                                    canonical_atom_order=False)  # Keep the original atomic order)
    protein_bigraph = mol_to_bigraph(protein_mol, add_self_loop=False,
                                     node_featurizer=PN_atom_featurizer,
                                     edge_featurizer=PN_bond_featurizer,
                                     canonical_atom_order=False)
    complex_bigraph = batch([ligand_bigraph, protein_bigraph])

    # Construct knn graphs for stage 2
    complex_coordinates = np.concatenate(
        [ligand_coordinates, protein_coordinates])
    complex_srcs, complex_dsts, complex_dists = k_nearest_neighbors(
        complex_coordinates, distance_bins[-1], max_num_neighbors)
    complex_srcs = np.array(complex_srcs)
    complex_dsts = np.array(complex_dsts)
    complex_dists = np.array(complex_dists)

    complex_knn_graph = graph(
        (complex_srcs, complex_dsts), num_nodes=len(complex_coordinates))
    d_features = np.digitize(complex_dists, bins=distance_bins, right=True)
    d_one_hot = int_to_one_hot(d_features, distance_bins)

    # add bond types and bonds (from bigraph) to stage 2
    u, v = complex_bigraph.edges()
    complex_knn_graph.add_edges(u.to(F.int64), v.to(F.int64))
    n_d, f_d = d_one_hot.shape
    n_e, f_e = complex_bigraph.edata['e'].shape
    complex_knn_graph.edata['e'] = F.zerocopy_from_numpy(
        np.block([
            [d_one_hot, np.zeros((n_d, f_e))],
            [np.zeros((n_e, f_d)), np.array(complex_bigraph.edata['e'])]
        ]).astype(np.long)
    )
    return ligand_bigraph, complex_bigraph, complex_knn_graph

# extract complex graph features and make .npy
#export: [(target, lig_idx)] = dict(complex_graph = complex_graph, ligand_graph = ligand_graph, protein_graph = protein_graph, interact_graph = interact_graph)


def construct_PN_graph():
    targets = pd.read_excel('./dude.xlsx')['target']
    labels = []
    for target in targets:
        labels.append(target)
    print(f'共{len(labels)}个靶点:{labels}')
    final_output = dict()
    failure_dict = {"target": [], "error_num": []}
    weight = dict()

    for target in tqdm(targets):
        nums = 0
        label = labels.index(target)
        target = target.lower()
        print(target)

        try:
            docking_pocket_file = './cluster_dataset_docking-poses/{}/protein_cut.pdb'.format(
                target)
            pose_pocket = Chem.MolFromPDBFile(docking_pocket_file)

            ligand_pose_file = './cluster_dataset_docking-poses/{}/ligand_docking_pose.sdf'.format(
                target)
            pose_ligands = Chem.SDMolSupplier(ligand_pose_file)
            # ligand = Chem.MolFromMolFile(ligand_file)

            lig_idx = 0
            error_num = 0
            for pose_ligand in tqdm(pose_ligands):
                try:
                    ligand_bigraph, complex_bigraph, complex_knn_graph = PN_graph_construction_and_featurization(
                        pose_ligand, pose_pocket)
                    lig_idx += 1
                    final_output[(target, lig_idx)] = dict(
                        ligand_bigraph=ligand_bigraph, complex_graph=complex_bigraph, complex_knn_graph=complex_knn_graph, label=label)
                    nums += 1
                except:
                    error_num += 1
                    print(f'{target}有{error_num}个配体构建失败')
                    failure_dict["target"].append(
                        target), failure_dict["error_num"].append(error_num)
                    continue
        except:
            error_num += 1
            failure_dict["target"].append(
                target), failure_dict["error_num"].append('Bad input protein file')
            continue

        weight[label] = nums
    print(weight)
    np.save(f'./output/pn_data.npy', final_output)

    failure_df = pd.DataFrame(failure_dict)
    failure_df.to_csv('./failure.csv', index=False)

def construct_PN_graph_test(target, distance=5):
    final_output = dict()
    failure_dict = {"target": [], "error_num": []}
    IDs = list(pd.read_csv(f'/home/sikang/cluster-AE/dataset/test_set/{target}/activity.csv')['ChemDiv Reference'])
    scores = list(pd.read_csv(f'/home/sikang/cluster-AE/dataset/test_set/{target}/activity.csv')['Docking Score'])

    print(target)
    error_num = 0
    pdb_file = '/home/sikang/cluster-AE/dataset/test_set/{}/{}.pdb'.format(
        target,target)
    structure = parsePDB(pdb_file)
    protein = structure.select('protein')

    ligand_pose_file = '/home/sikang/cluster-AE/dataset/test_set/{}/ligand_docking_pose.sdf'.format(
        target)
    pose_ligands = Chem.SDMolSupplier(ligand_pose_file)
    # ligand = Chem.MolFromMolFile(ligand_file)

    lig_idx = 0
    for pose_ligand in tqdm(pose_ligands):
        ID = pose_ligand.GetProp('IDNUMBER')
        index = IDs.index(ID)
        docking_score = scores[index]
        selected = protein.select(f'same residue as within {distance} of ligand', ligand=pose_ligand.GetConformer().GetPositions())
        writePDB('/home/sikang/cluster-AE/dataset/test_set/{}/pro_cut/{}_{}.pdb'.format(target,target,ID), selected)
        pose_pocket = Chem.MolFromPDBFile('/home/sikang/cluster-AE/dataset/test_set/{}/pro_cut/{}_{}.pdb'.format(target,target,ID), sanitize=True)

        if pose_pocket:
            ligand_bigraph, complex_bigraph, complex_knn_graph = PN_graph_construction_and_featurization(
                pose_ligand, pose_pocket)
            lig_idx += 1
            final_output[(target, lig_idx)] = dict(
                ligand_bigraph=ligand_bigraph, complex_graph=complex_bigraph, complex_knn_graph=complex_knn_graph, ligand_id = ID, docking_score=docking_score)

        else:
            print(f'pocket file read error for {target}-{ID}')
        # error_num += 1
        # print(f'{target}有{error_num}个配体构建失败')
        # failure_dict["target"].append(
        #     target), failure_dict["error_num"].append(error_num)

    np.save(f'./output/pn_testset_{target}.npy', final_output)

    failure_df = pd.DataFrame(failure_dict)
    failure_df.to_csv(f'./failure_{target}.csv', index=False)

if __name__ == '__main__':
    # construct_IGN_graph()
    construct_PN_graph_test('notume')
