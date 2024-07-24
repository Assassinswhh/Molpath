from utils import *

import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from typing import Callable, List, Optional
from tqdm import tqdm
import argparse
import time
import numpy as np
import networkx as nx

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.datasets import ZINC
from torch_geometric.data import InMemoryDataset
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, download_url, extract_zip
import pickle
import hashlib
import os.path as osp
import pickle
import shutil
import pandas as pd
from collections import defaultdict
from mol import smiles2graph
# from dig.threedgraph.dataset import QM93D
from sklearn.utils import shuffle
import sys
from torch_scatter import scatter
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
from rdkit import Chem



HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class PathTransform(object):
    def __init__(self, path_type, cutoff):
        self.cutoff = cutoff
        self.path_type = path_type

    def __call__(self, data):
        if self.path_type is None and self.cutoff is None:
            data.pp_time = 0
            return data

        setattr(data, f"path_2", data.edge_index.T.flip(1))
        setattr(
            data,
            f"edge_indices_2",
            get_edge_indices(
                data.x.size(0), data.edge_index, data.edge_index.T.flip(1)
            ),
        )

        if self.cutoff == 2 and self.path_type is None:
            data.pp_time = 0
            return ModifData(**data.stores[0])

        t0 = time.time()
        G = ig.Graph.from_networkx(to_networkx(data, to_undirected=True))
        if self.path_type == "all_simple_paths":
            setattr(
                data,
                f"sp_dists_2",
                torch.cat(
                    [torch.ones(data.num_edges, 1), torch.zeros(data.num_edges, 1)],
                    dim=1,
                ).long(),
            )
        graph_info = fast_generate_paths2(
            G, self.cutoff, self.path_type, undirected=True
        )

        cnt = 0
        for jj in range(1, self.cutoff - 1):
            paths = torch.LongTensor(graph_info[0][jj]).view(-1, jj + 2)
            setattr(data, f"path_{jj+2}", paths.flip(1))
            setattr(
                data,
                f"edge_indices_{jj+2}",
                get_edge_indices(data.x.size(0), data.edge_index, paths.flip(1)),
            )
            if self.path_type == "all_simple_paths":
                if len(paths) > 0:
                    setattr(
                        data,
                        f"sp_dists_{jj+2}",
                        torch.Tensor(graph_info[2][jj]).long().flip(1),
                    )
                else:
                    setattr(data, f"sp_dists_{jj+2}", torch.empty(0, jj + 2).long())
                    cnt += 1
        data.max_cutoff = self.cutoff
        data.cnt = cnt
        data.pp_time = time.time() - t0
        return ModifData(**data.stores[0])


def get_edge_indices(size, edge_index_n, paths):
    index_tensor = torch.zeros(size, size, dtype=torch.long, device=paths.device)
    index_tensor[edge_index_n[0], edge_index_n[1]] = torch.arange(
        edge_index_n.size(1), dtype=torch.long, device=paths.device
    )
    indices = []
    for i in range(paths.size(1) - 1):
        indices.append(index_tensor[paths[:, i], paths[:, i + 1]].unsqueeze(1))

    return torch.cat(indices, -1)


class ZincDataset(InMemoryDataset):
    """This is ZINC from the Benchmarking GNNs paper. This is a graph regression task."""

    def __init__(
        self,
        root,
        path_type="shortest_path",
        cutoff=3,
        transform=None,
        pre_filter=None,
        pre_transform=None,
        subset=True,
        n_jobs=2,
    ):
        self.name = "ZINC"
        self._subset = subset
        self._n_jobs = n_jobs
        self.path_type = path_type
        self.cutoff = cutoff
        self.task_type = "regression"
        self.num_node_type = 28
        self.num_edge_type = 4
        self.num_tasks = 1
        self.eval_metric = "mae"
        super(ZincDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices, idx = self.load_dataset()
        self.train_ids = idx[0]
        self.val_ids = idx[1]
        self.test_ids = idx[2]

    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def processed_file_names(self):
        name = self.name
        return [f"{name}.pt", f"{name}_idx.pt"]

    def download(self):
        # Instantiating this will download and process the graph dataset.
        ZINC(self.raw_dir, subset=self._subset)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        train_data = ZINC(self.raw_dir, subset=self._subset, split="train")
        val_data = ZINC(self.raw_dir, subset=self._subset, split="val")
        test_data = ZINC(self.raw_dir, subset=self._subset, split="test")

        data_list = []
        idx = []
        start = 0
        t0 = time.time()
        train_data = [self.convert(data) for data in train_data]
        data_list += train_data
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        val_data = [self.convert(data) for data in val_data]
        data_list += val_data
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        test_data = [self.convert(data) for data in test_data]
        data_list += test_data
        idx.append(list(range(start, len(data_list))))

        self.preprocessing_time = time.time() - t0
        path = self.processed_paths[0]
        print(f"Saving processed dataset in {path}....")
        torch.save(self.collate(data_list), path)

        path = self.processed_paths[1]
        print(f"Saving idx in {path}....")
        torch.save(idx, path)

    def convert(self, data):

        if self.path_type is None and self.cutoff is None:
            return data

        data.x = data.x.squeeze(1)
        setattr(data, f"path_2", data.edge_index.T.flip(1))
        setattr(
            data,
            f"edge_indices_2",
            get_edge_indices(
                data.x.size(0), data.edge_index, data.edge_index.T.flip(1)
            ),
        )

        if self.cutoff == 2 and self.path_type is None:
            return ModifData(**data.stores[0])

        else:
            G = ig.Graph.from_networkx(to_networkx(data, to_undirected=True))
            if self.path_type == "all_simple_paths":
                setattr(
                    data,
                    f"sp_dists_2",
                    torch.cat(
                        [torch.ones(data.num_edges, 1), torch.zeros(data.num_edges, 1)],
                        dim=1,
                    ).long(),
                )
            graph_info = fast_generate_paths2(
                G, self.cutoff, self.path_type, undirected=True
            )

            cnt = 0
            for jj in range(1, self.cutoff - 1):
                paths = torch.LongTensor(graph_info[0][jj]).view(-1, jj + 2)
                setattr(data, f"path_{jj+2}", paths.flip(1))
                setattr(
                    data,
                    f"edge_indices_{jj+2}",
                    get_edge_indices(data.x.size(0), data.edge_index, paths.flip(1)),
                )
                if self.path_type == "all_simple_paths":
                    if len(paths) > 0:
                        setattr(
                            data,
                            f"sp_dists_{jj+2}",
                            torch.LongTensor(graph_info[2][jj]).flip(1),
                        )
                    else:
                        setattr(data, f"sp_dists_{jj+2}", torch.empty(0, jj + 2).long())
                        cnt += 1

            data.max_cutoff = self.cutoff
            data.cnt = cnt
            return ModifData(**data.stores[0])

    def get_idx_split(self):
        return {"train": self.train_ids, "valid": self.val_ids, "test": self.test_ids}


class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "peptides-functional")
        self.task_type = "classification"
        self.num_tasks = 10
        self.eval_metric = "ap"
        self.root = root

        self.url = "https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1"
        self.version = (
            "701eb743e899f4d793f0e13c8fa5a1b4"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.preprocessing_time = sum(self.data.pp_time).item()

    @property
    def raw_file_names(self):
        return "peptide_multi_class_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "peptide_multi_class_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([eval(data_df["labels"].iloc[i])])


            data_list.append(data)

        t0 = time.time()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root, "splits_random_stratified_peptide.pickle")
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict



class PeptidesStructuralDataset(InMemoryDataset):
    def __init__(
        self,
        root="datasets",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
    ):
        """
        PyG dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.
        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.
        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.task_type = "regression"
        self.num_tasks = 11
        self.eval_metric = "mae"
        self.folder = osp.join(root, "peptides-structural")

        self.url = "https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1"
        self.version = (
            "9786061a34298a0684150f2e4ff13f47"  # MD5 hash of the intended dataset file
        )
        self.url_stratified_split = "https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "peptide_structure_dataset.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "peptide_structure_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]
        target_names = [
            "Inertia_mass_a",
            "Inertia_mass_b",
            "Inertia_mass_c",
            "Inertia_valence_a",
            "Inertia_valence_b",
            "Inertia_valence_c",
            "length_a",
            "length_b",
            "length_c",
            "Spherocity",
            "Plane_best_fit",
        ]
        # Normalize to zero mean and unit standard deviation.
        data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][target_names]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([y])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """Get dataset splits.
        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(
            self.root, "splits_random_stratified_peptide_structure.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


class Qm9dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = 'mae'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)
        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue


            # conf = mol.GetConformer()
            # pos = conf.GetPositions()
            # pos = torch.tensor(pos, dtype=torch.float)

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))

            # z = torch.tensor(atomic_number, dtype=torch.long)

            num_bond_features = 3  # bond type, bond stereo, is_conjugated
            if len(mol.GetBonds()) > 0:  # mol has bonds
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    m = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((m, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, m))
                    edge_features_list.append(edge_feature)

                # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
                edge_index = torch.tensor(edges_list, dtype=torch.int64).T

                # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
                edge_attr = torch.tensor(edge_features_list, dtype=torch.int64)
            else:  # mol has no bonds
                edge_index = torch.tensor((2, 0), dtype=torch.int64)
                edge_attr = torch.tensor((0, num_bond_features), dtype=torch.int64)


            x = torch.tensor(atom_features_list, dtype=torch.int64)


            y = target[i].unsqueeze(0)
            name = mol.GetProp('_Name')

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=name, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class BBBPDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "BBBP")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "BBBP.csv"

    @property
    def processed_file_names(self):
        return "BBBP_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "BBBP.csv")
        )
        smiles_list = data_df["smiles"]
        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]

            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([(data_df["p_np"].iloc[i])])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)
        data.y = data.y.view(len(data.y), 1)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class ClinToxDataset(InMemoryDataset):
    def  __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "ClinTox")
        self.task_type = "classification"
        self.num_tasks = 2
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "clintox.csv.gz"

    @property
    def processed_file_names(self):
        return "clintox_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "clintox.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["FDA_APPROVED",
                 "CT_TOX"]


        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class SiderDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Sider")
        self.task_type = "classification"
        self.num_tasks = 27
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Sider.csv.gz"

    @property
    def processed_file_names(self):
        return "sider_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "sider.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["Hepatobiliary disorders",
                 "Metabolism and nutrition disorders",
                 "Product issues",
                 "Eye disorders",
                 "Investigations",
                 "Musculoskeletal and connective tissue disorders",
                 "Gastrointestinal disorders",
                 "Social circumstances",
                 "Immune system disorders",
                 "Reproductive system and breast disorders",
                 "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                 "General disorders and administration site conditions",
                 "Endocrine disorders",
                 "Surgical and medical procedures",
                 "Vascular disorders",
                 "Blood and lymphatic system disorders",
                 "Skin and subcutaneous tissue disorders",
                 "Congenital, familial and genetic disorders",
                 "Infections and infestations",
                 "Respiratory, thoracic and mediastinal disorders",
                 "Psychiatric disorders",
                 "Renal and urinary disorders",
                 "Pregnancy, puerperium and perinatal conditions",
                 "Ear and labyrinth disorders",
                 "Cardiac disorders",
                 "Nervous system disorders",
                 "Injury, poisoning and procedural complications",
                 ]


        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class Tox21Dataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Tox21")
        self.task_type = "classification"
        self.num_tasks = 12
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "tox21.csv.gz"

    @property
    def processed_file_names(self):
        return "tox21_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "tox21.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["NR-AR",
                 "NR-AR-LBD",
                 "NR-AhR",
                 "NR-Aromatase",
                 "NR-ER",
                 "NR-ER-LBD",
                 "NR-PPAR-gamma",
                 "SR-ARE",
                 "SR-ATAD5",
                 "SR-HSE",
                 "SR-MMP",
                 "SR-p53"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class HIVDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "HIV")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "HIV.csv"

    @property
    def processed_file_names(self):
        return "HIV_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "HIV.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["HIV_active"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class BaceDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Bace")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "bace.csv"

    @property
    def processed_file_names(self):
        return "bace_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "bace.csv")
        )
        smiles_list = data_df["mol"]
        lable = ["Class"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class EsolDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Esol")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "delaney-processed.csv"

    @property
    def processed_file_names(self):
        return "Esol_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "delaney-processed.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["measured log solubility in mols per litre"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class FreeSolvDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "FreeSolv")
        self.task_type = "mes_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "SAMPL.csv"

    @property
    def processed_file_names(self):
        return "FreeSolv_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "SAMPL.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["expt"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class LipopDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Lipop")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Lipophilicity.csv"

    @property
    def processed_file_names(self):
        return "Lipop_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "Lipophilicity.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["exp"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


def get_dataset(dataset, cutoff, path_type, output_dir="./"):
    print(f"Preprocessing {dataset} - {path_type}".upper())
    if not os.path.exists(os.path.join(output_dir, "dataset")):
        os.makedirs(os.path.join(output_dir, "dataset"))
    root = os.path.join(
        output_dir, "dataset", dataset + "_" + str(path_type) + "_cutoff_" + str(cutoff)
    )

    if dataset in ["ogbg-molhiv", "ogbg-molpcba"]:
        data = PygGraphPropPredDataset(
            name=dataset, pre_transform=PathTransform(path_type, cutoff), root=root
        )
        data.preprocessing_time = sum([i.pp_time for i in data]).item()

    elif dataset == "ZINC":
        data = ZincDataset(root=root, path_type=path_type, cutoff=cutoff)
    elif dataset == "peptides-functional":
        data = PeptidesFunctionalDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "peptides-structural":
        data = PeptidesStructuralDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "Qm9":
        data = Qm9dataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "BBBP":
        data = BBBPDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "ClinTox":
        data = ClinToxDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "Tox21":
        data = Tox21Dataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "Sider":
        data = SiderDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "HIV":
        data = HIVDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "Esol":
        data = EsolDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "Freesolv":
        data = FreeSolvDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "Lipop":
        data = LipopDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    elif dataset == "Bace":
        data = BaceDataset(
            root=root, pre_transform=PathTransform(path_type, cutoff)
        )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--min_cutoff", type=int, default=3)
    parser.add_argument(
        "--max_cutoff", type=int, default=5, help="Max length of shortest paths"
    )
    args = parser.parse_args()

    for cutoff in range(args.min_cutoff, args.max_cutoff + 1):
        # for dataset in ["ogbg-molhiv", "ogbg-molpcba", "ZINC", "peptides-functional", "peptides-structural"] :
        for dataset in [args.dataset]:

            for path_type in [
                "shortest_path",
                "all_shortest_paths",
                "all_simple_paths",
            ]:

                data = get_dataset(dataset, cutoff, path_type)
                print(data.preprocessing_time)
