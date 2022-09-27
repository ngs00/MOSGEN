import json
import torch
from sklearn.preprocessing import scale
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import Batch
from util.chem import *


MARKOV_ORDER = 3


class SpectrumGraphDataset:
    def __init__(self, dataset, comp_names, class_names):
        self.dataset = dataset
        self.comp_names = comp_names
        self.class_names = class_names
        self.labels = [d.y.item() for d in dataset]
        self.n_classes = len(self.class_names)
        self.dim_node_feats = self.dataset[0].x.shape[1]
        self.dim_edge_feats = self.dataset[0].edge_attr.shape[1]
        self.data_dist = dict()
        self.data_db = dict()

        self.__init_data_dist()
        self.__init_comp_db()

    def __init_data_dist(self):
        for d in self.dataset:
            label = d.y.item()

            if label not in self.data_dist.keys():
                self.data_dist[label] = list()

            self.data_dist[label].append(d)

    def __init_comp_db(self):
        for i in range(0, len(self.dataset)):
            label = self.dataset[i].y.item()
            mol_weight = self.dataset[i].query_value.item()

            if label not in self.data_db.keys():
                self.data_db[label] = list()

            self.data_db[label].append([mol_weight, self.comp_names[i]])

    def get_batch(self, n_per_class):
        balanced_batch = list()

        for key in self.data_dist.keys():
            if len(self.data_dist[key]) < n_per_class:
                balanced_batch += self.data_dist[key]
            else:
                idx_rand = numpy.random.permutation(len(self.data_dist[key]))
                for i in range(0, n_per_class):
                    balanced_batch.append(self.data_dist[key][idx_rand[i]])

        return Batch.from_data_list(balanced_batch)

    def get_k_fold(self, k, random_seed=None):
        if random_seed is not None:
            numpy.random.seed(random_seed)

        idx_rand = numpy.random.permutation(len(self.dataset))
        dataset = [self.dataset[idx] for idx in idx_rand]
        comp_names = [self.comp_names[idx] for idx in idx_rand]
        sf = int(len(self.dataset) / k)
        kfolds = list()

        for i in range(0, k):
            if i == k - 1:
                dataset_train = dataset[:(k-1)*sf]
                dataset_test = dataset[(k-1)*sf:]
                comp_names_train = comp_names[:(k-1)*sf]
                comp_names_test = comp_names[(k-1)*sf:]
            else:
                dataset_train = dataset[:i*sf] + dataset[(i+1)*sf:]
                dataset_test = dataset[i*sf:(i+1)*sf]
                comp_names_train = comp_names[:i*sf] + comp_names[(i+1)*sf:]
                comp_names_test = comp_names[i*sf:(i+1)*sf]

            dataset_train = SpectrumGraphDataset(dataset_train, comp_names_train, self.class_names)
            dataset_test = SpectrumGraphDataset(dataset_test, comp_names_test, self.class_names)
            kfolds.append([dataset_train, dataset_test])

        return kfolds

    def search(self, label, mw, k):
        comp_ids = list()
        classes = list()

        for i in range(0, len(label)):
            comp_mws = numpy.array([c[0] for c in self.data_db[label[i]]])
            comp_idx = numpy.argsort(numpy.abs(comp_mws - mw[i]))[:k]
            comp_ids.append([self.data_db[label[i]][idx][1] for idx in comp_idx])
            classes.append(label[i])

        return comp_ids


def load_dataset(path_dataset):
    data_file = open(path_dataset, 'r')
    data = json.load(data_file)
    data_keys = list(data.keys())
    class_dict = get_class_dict(data)
    dataset = list()
    comp_names = list()

    for i in tqdm(range(0, len(data_keys))):
        ir_data = data[data_keys[i]]['ir_data']
        mat_class = data[data_keys[i]]['mat_class']

        if mat_class == 'Unclassified materials (UC)':
            continue

        label = class_dict[mat_class]
        label = torch.tensor(label, dtype=torch.long).view(1, 1)
        g = get_spectrum_graph(data[data_keys[i]]['elements'], ir_data, label)
        dataset.append(g)
        comp_names.append(data[data_keys[i]]['names'][0])

    return SpectrumGraphDataset(dataset, comp_names, list(class_dict.keys()))


def load_dataset_fg(path_dataset, smarts_fg):
    data_file = open(path_dataset, 'r')
    data = json.load(data_file)
    data_keys = list(data.keys())
    fg = Chem.MolFromSmarts(smarts_fg)
    dataset = list()
    comp_names = list()

    n_neg = 0
    n_pos = 0
    for i in tqdm(range(0, len(data_keys))):
        if 'smiles' in data[data_keys[i]]:
            mol = Chem.MolFromSmiles(data[data_keys[i]]['smiles'])

            if mol is not None:
                matches = mol.GetSubstructMatches(fg)
                label = 1 if len(matches) > 0 else 0
                label = torch.tensor(label, dtype=torch.float).view(1, 1)
                ir_data = data[data_keys[i]]['ir_data']
                g = get_spectrum_graph(data[data_keys[i]]['elements'], ir_data, label)
                dataset.append(g)
                comp_names.append(data[data_keys[i]]['names'][0])

                if label == 0:
                    n_neg += 1
                else:
                    n_pos += 1

    return SpectrumGraphDataset(dataset, comp_names, ['false', 'true'])


def get_class_dict(data):
    class_dict = dict()
    n_classes = 0

    for d in data:
        mat_class = data[d]['mat_class']

        if mat_class == 'Unclassified materials (UC)':
            continue

        if mat_class not in class_dict.keys():
            class_dict[mat_class] = n_classes
            n_classes += 1

    return class_dict


def get_spectrum_graph(elems, ir_data, label=None):
    wn = torch.tensor(scale(numpy.array([float(key) for key in ir_data.keys()])), dtype=torch.float).view(-1, 1)
    asb = torch.tensor(scale(numpy.array(list(ir_data.values()))), dtype=torch.float).view(-1, 1)
    atomic_vec = torch.zeros(5)
    edges = list()
    edge_feats = list()

    for e in elems:
        if e == 'H':
            atomic_vec[0] = 1
        elif e == 'C':
            atomic_vec[1] = 1
        elif e == 'N':
            atomic_vec[2] = 1
        elif e == 'O':
            atomic_vec[3] = 1
        elif e == 'S':
            atomic_vec[4] = 1
    atomic_vec = atomic_vec.repeat(wn.shape[0], 1)

    # # Integration with EA
    # form_dict = {
    #     'H': 0,
    #     'C': 0,
    #     'N': 0,
    #     'O': 0,
    #     'S': 0,
    # }
    # for e in elems:
    #     if e in ['H', 'C', 'N', 'O', 'S']:
    #         form_dict[e] += 1
    # for e in elems:
    #     if e == 'H':
    #         atomic_vec[0] = form_dict[e]
    #     elif e == 'C':
    #         atomic_vec[1] = form_dict[e]
    #     elif e == 'N':
    #         atomic_vec[2] = form_dict[e]
    #     elif e == 'O':
    #         atomic_vec[3] = form_dict[e]
    #     elif e == 'S':
    #         atomic_vec[4] = form_dict[e]
    # atomic_vec = atomic_vec.repeat(wn.shape[0], 1)

    for k in range(0, MARKOV_ORDER):
        order = k + 1
        for i in range(0, wn.shape[0] - order):
            edges.append([i, i + order])
            edge_feats.append([wn[i + order] - wn[i], asb[i + order] - asb[i]])

    node_feats = torch.hstack([wn, asb, atomic_vec])
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_feats = torch.tensor(edge_feats, dtype=torch.float)

    if label is None:
        return Data(x=node_feats, edge_index=edges, edge_attr=edge_feats)
    else:
        mol_weight = torch.tensor(get_mol_weight(elems), dtype=torch.float).view(1, 1)
        return Data(x=node_feats, edge_index=edges, edge_attr=edge_feats, y=label, query_value=mol_weight)
