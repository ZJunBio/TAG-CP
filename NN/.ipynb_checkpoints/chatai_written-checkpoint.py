import torch
from torch_geometric.data import Data, DataLoader

# 假设 x1 和 x2 分别表示两个样本，节点数分别为 10 和 20
x1 = torch.randn(10, 50)   # 第一个样本的输入特征维度为 50
x2 = torch.randn(20, 50)   # 第二个样本的输入特征维度为 50
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # 假设边信息已知

# 将这两个样本在第 0 维进行合并，并按照节点数量排序
x = torch.cat([x1.unsqueeze(0), x2.unsqueeze(0)], dim=0)
batch_size = [len(x1), len(x2)]
_, perm = torch.sort(torch.tensor(batch_size), descending=True)
x = x[perm]
batch_size = [batch_size[i] for i in perm]

# 构建 PyG 数据对象
data = Data(x=x, edge_index=edge_index)

# 通过 `batch` 属性指定批次信息
batch = []
for i, size in enumerate(batch_size):
    batch += [i] * size
data.batch = torch.tensor(batch)

# 使用 DataLoader 将数据和模型组合起来，形成一个 batch
batch_size = max(batch_size)  # 批次中节点数最大为 20
loader = DataLoader([data], batch_size=batch_size)

# 使用 AttentiveFP 模型进行前向传播
model = nn.models.AttentiveFP(50, 2)
out = model(loader.dataset)

# out.shape 应该是 (2, 2)，表示输出结果的大小为 2 x 2

######################################################
import torch
from torch_geometric.data import Data, DataLoader

# 假设 x1 和 x2 分别表示两个样本，节点数分别为 10 和 20
x1 = torch.randn(10, 50)   # 第一个样本的输入特征维度为 50
x2 = torch.randn(20, 50)   # 第二个样本的输入特征维度为 50
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # 假设边信息已知

# 将这两个样本在第 0 维进行合并，并按照节点数量排序
subgraphs = [Data(x=x1, edge_index=edge_index), Data(x=x2, edge_index=edge_index)]
batch_size = [len(subgraph.x) for subgraph in subgraphs]
perm = torch.argsort(torch.tensor(batch_size))
subgraphs = [subgraphs[i] for i in perm]
batch_size = [batch_size[i] for i in perm]
x = torch.cat([subgraph.x for subgraph in subgraphs], dim=0)

# 通过 `batch` 属性指定批次信息
batch = []
num_nodes = 0
for i, size in enumerate(batch_size):
    batch += [i] * size
    num_nodes += size
data = Data(x=x, edge_index=edge_index)
data.batch = torch.tensor(batch)

# 使用 DataLoader 将数据和模型组合起来，形成一个 batch
loader = DataLoader([data], batch_size=num_nodes)

# 使用 AttentiveFP 模型进行前向传播
model = nn.models.AttentiveFP(50, 2)
out = model(loader.dataset)

# out.shape 应该是 (2, 2)，表示输出结果的大小为 2 x 2
