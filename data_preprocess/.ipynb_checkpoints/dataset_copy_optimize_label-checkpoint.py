import torch
import numpy as np
import pandas as pd
import random
from torch_geometric.data import Data
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit import Chem
import torch
from torch_geometric.data import InMemoryDataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def idx_mask(l, idx):
    mask = torch.zeros(l)
    mask[idx] = 1
    return torch.as_tensor(mask, dtype=torch.bool)

def process_data(args):
    network_file = pd.read_csv(args[0], header = 0)
    smile_64 = pd.read_csv(args[1], header = 0)

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
    edge_index = torch.tensor([source, target], dtype = torch.long)
    #edge feature
    #node labels (42 types of inhibitors)
    target_index = {v:i for i,v in enumerate(pd.unique(network_file["targets"]))}
    labels = np.zeros((1362,42))
    for c, d in enumerate(nodes):
        temp_ = network_file[network_file["drug_name"] == d]["targets"]
        if len(temp_) != 0:
            target_id = [target_index[i] for i in pd.unique(temp_)]
            labels[c, np.array(target_id)] += 1
        else:
            temp_ = network_file[network_file["shared_by"] == d]["targets"]
            target_id = [target_index[i] for i in pd.unique(temp_)]
            labels[c, np.array(target_id)] += 1
    #CREATE DATA object
    data = Data(x = torch.as_tensor(node_feature, dtype = torch.float32), 
                y = torch.as_tensor(labels, dtype = torch.float32), edge_index = edge_index)
    #generate train/validation/test mask
    l = labels.shape[0]
    shuffle_idx = random.sample(population = list(range(l)), k = l)
    train_idx = shuffle_idx[0:int(0.8 * l)]
    val_idx = shuffle_idx[int(0.8 * l) : int(0.9 * l)]
    test_idx = shuffle_idx[int(0.9 * l):]
    train_idx = idx_mask(l, train_idx)
    val_mask = idx_mask(l, val_idx)
    test_mask = idx_mask(l, test_idx)

    data.train_idx = train_idx
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

#args = ("../data/one_graph/network_file_0self.csv", "../data/jaaks_druginfo_ttid.csv")    
#data = process_data(args) 
    
'''
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

    def process(self, data = data):
        # Read data into huge `Data` list.
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

dataset = GraDataset("../data/one_graph/dataset/")
'''

####################
#create model
####################
'''
embed_dim = 128
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, embed_dim)
        self.conv2 = GCNConv(embed_dim, data.y.shape[1])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.binary_cross_entropy(out[data.train_idx], data.y[data.train_idx])
    if epoch % 10 == 0:
        print(f'Loss:{loss}')
    loss.backward()
    optimizer.step()
    
model.eval()
pred = model(data)
from sklearn.metrics import accuracy_score
#correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
#acc = int(correct) / int(data.test_mask.sum())
acc = accuracy_score(pred[data.test_mask].detach().numpy(), 
                     data.y[data.test_mask].detach().numpy())

pred_label = np.zeros(data.y[data.test_mask].size())
pred = pred[data.test_mask].detach().numpy()
for i, v in enumerate(data.y[data.test_mask].detach().numpy()):
    sum_ = int(v.sum())
    if sum_ == 1:
        l = np.argsort(pred[i])[-1]
        pred_label[i, l] = 1
    else:
        l = np.argsort(pred[i])[-sum_:]
        pred_label[i, l] = 1
acc = accuracy_score(pred_label, data.y[data.test_mask].detach().numpy())
        



print(f'Accuracy: {acc:.4f}')
'''