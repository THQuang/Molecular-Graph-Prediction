import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from utils import onehot_encoding_unk
from utils import onehot_encoding


def load_dataset_random(path, dataset, seed, tasks=None):
    pyg_dataset = MultiDataset(root=path, dataset=dataset, tasks=tasks)
    df = pd.read_csv(os.path.join(path, "raw/{}.csv".format(dataset)))
    smilesList = df.smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            canonical_smiles_list.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                 isomericSmiles=True)
            )
            remained_smiles.append(smiles)
        except Exception:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))

    df = df[df["smiles"].isin(remained_smiles)].reset_index()
    if (
        dataset == "sider"
        or dataset == "clintox"
        or dataset == "tox21"
        or dataset == "ecoli"
        or dataset == "AID1706_binarized_sars"
    ):
        train_size = int(0.8 * len(pyg_dataset))
        val_size = int(0.1 * len(pyg_dataset))
        pyg_dataset = pyg_dataset.shuffle()
        trn, val, test = (
            pyg_dataset[:train_size],
            pyg_dataset[train_size: (train_size + val_size)],
            pyg_dataset[(train_size + val_size):],
        )
        weights = []
        for i, task in enumerate(tasks):
            negative_df = df[df[task] == 0][["smiles", task]]
            positive_df = df[df[task] == 1][["smiles", task]]
            neg_len = len(negative_df)
            pos_len = len(positive_df)
            weights.append(
                [(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len]
            )
        trn.weights = weights

    else:
        train_size = int(0.8 * len(pyg_dataset))
        val_size = int(0.1 * len(pyg_dataset))
        # test_size = len(pyg_dataset) - train_size - val_size
        pyg_dataset = pyg_dataset.shuffle()
        trn, val, test = (
            pyg_dataset[:train_size],
            pyg_dataset[train_size:(train_size + val_size)],
            pyg_dataset[(train_size + val_size):],
        )
        trn.weights = "regression task has no class weights!"
  
    return trn, val, test


def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        # if atom.GetDegree()>5:
        #     print(Chem.MolToSmiles(mol))
        #     print(atom.GetSymbol())
        results = (
            onehot_encoding_unk(
                atom.GetSymbol(),
                [
                    "B",
                    "C",
                    "N",
                    "O",
                    "F",
                    "Si",
                    "P",
                    "S",
                    "Cl",
                    "As",
                    "Se",
                    "Br",
                    "Te",
                    "I",
                    "At",
                    "other",
                ],
            )
            + onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # noqa: E501
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + onehot_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                    "other",
                ],
            )
            + [atom.GetIsAromatic()]
        )
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + onehot_encoding_unk(
                atom.GetTotalNumHs(), [0, 1, 2, 3, 4]
            )
        if use_chirality:
            try:
                results = (
                    results
                    + onehot_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"])
                    + [atom.HasProp("_ChiralityPossible")]
                )
            #
            except Exception:
                results = results + [0, 0] + [atom.HasProp("_ChiralityPossible")]  # noqa: E501
        feat.append(results)

    return np.array(feat)


def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE,
                        bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE,
                        bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing(),
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
                        )
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)


class MultiDataset(InMemoryDataset):
    def __init__(
        self, root, dataset, tasks, transform=None, pre_transform=None,
        pre_filter=None
    ):
        self.tasks = tasks
        self.dataset = dataset

        self.weights = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform,
                                           pre_filter)
        self.data, self.slices = self.process()
        # os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["{}.csv".format(self.dataset)]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.dataset)]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smilesList = df.smiles.values
        print("number of all smiles: ", len(smilesList))
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                canonical_smiles_list.append(
                    Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                     isomericSmiles=True)
                )
                remained_smiles.append(smiles)
            except Exception:
                print("not successfully processed smiles: ", smiles)
                pass
        print("number of successfully processed smiles: ",
              len(remained_smiles))

        df = df[df["smiles"].isin(remained_smiles)].reset_index()
        target = df[self.tasks].values
        smilesList = df.smiles.values
        data_list = []

        for i, smi in enumerate(tqdm(smilesList)):
            mol = MolFromSmiles(smi)
            data = self.mol2graph(mol)

            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 6
                data.y = torch.LongTensor([label])
                if (
                    self.dataset == "esol"
                    or self.dataset == "freesolv"
                    or self.dataset == "lipophilicity"
                ):
                    data.y = torch.FloatTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        return data, slices

    def mol2graph(self, mol):
        if mol is None:
            return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        # pos = torch.FloatTensor(geom)
        data = Data(
            x=torch.FloatTensor(node_attr),
            # pos=pos,
            edge_index=torch.LongTensor(edge_index).t(),
            edge_attr=torch.FloatTensor(edge_attr),
            y=None,  # None as a placeholder
        )
        return data

