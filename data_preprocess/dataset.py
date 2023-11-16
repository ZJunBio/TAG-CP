import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit import Chem
#source and target file.
network_file = pd.read_csv("../data/one_graph/network_file_0self.csv", header = 0)
smile_64 = pd.read_csv("../data/jaaks_druginfo_ttid.csv", header = 0)
#drug_ttid and smiles mapping
dict_smile = {network_file.iloc[i,2]:network_file.iloc[i,4] for i in network_file.index}
for i in pd.unique(network_file["drug_name"]):
        dict_smile[i] = smile_64[smile_64["ttid"] == i]["canonicalsmiles"].values[0]
#node list
nodes = list(dict_smile)
node_index = {v:i for i,v in enumerate(nodes)}
node_mols = [Chem.MolFromSmiles(dict_smile[i]) for i in nodes]
#node features
node_feature = []
for m in node_mols:
    macc_arr = np.zeros((1,))
    maccskeys = MACCSkeys.GenMACCSKeys(m)
    DataStructs.ConvertToNumpyArray(maccskeys, macc_arr)
    node_feature.append(macc_arr)
node_feature = np.array(node_feature)
#edge index
source = [node_index[i] for i in network_file["drug_name"]]
temp_ = source * 1
target = [node_index[i] for i in network_file["shared_by"]]
source.extend(target)
target.extend(temp_)
edge_index = torch.tensor([source, target])
#edge feature
#node labels (42 types of inhibitors)
target_index = {v:i for i,v in enumerate(pd.unique(network_file["targets"]))}
labels = np.zeros((1374,42))
for c, d in enumerate(nodes):
    temp_ = network_file[network_file["drug_name"] == d]["targets"]
    if len(temp_) != 0:
        target_id = [target_index[i] for i in pd.unique(temp_)]
        labels[c, np.array(target_id)] += 1
    else:
        temp_ = network_file[network_file["shared_by"] == d]["targets"]
        target_id = [target_index[i] for i in pd.unique(temp_)]
        labels[c, np.array(target_id)] += 1
########################
#CREATE DATASET
########################
data = Data(x = node_feature, y = labels, edge_index = edge_index)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset, download_url


class GraDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    #def download(self):
    #    # Download to `self.raw_dir`.
    #    download_url(url, self.raw_dir)
    #    ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
