import numpy as np
import pandas as pd
from scipy import stats
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt

#prepare the heatmap data of pearson correlation for R.
nodes_repre = np.load("../data/one_graph/repr_nodes_1362.npy", allow_pickle = True)
args = ("../data/one_graph/network_file_0self.csv", "../data/jaaks_druginfo_ttid.csv")

nodes, node_mols = mols_gen(args=args)
'''
pe = repr_sim(self_ = True, rep = nodes_repre[0:100])
print(pe.shape)

def xy(nodes = nodes):
    x = []
    y = []
    for i, m in enumerate(nodes):
        for j, n in enumerate(nodes):
            if j >= i:
                x.append(m)
                y.append(n)
                
    return x, y

x, y = xy(nodes[0:100])
print(len(x), len(y))

df = pd.DataFrame({"x":x, "y":y, "value":pe})
df.to_csv("../data/one_graph/heatmap_data_graph_similarity.csv", header = True, index = False)
'''
#heatmap
def heatmapdata(nodes):
    arr = np.zeros(shape = (len(nodes), len(nodes)))
    for i, m in enumerate(nodes):
        for j, n in enumerate(nodes):
            res = stats.pearsonr(m,n)
            arr[i,j] = res.statistic
    return arr

matrix = heatmapdata(nodes = nodes_repre)
print(matrix.shape)
matrix = pd.DataFrame(matrix, index = nodes, columns = nodes)
matrix.to_csv("../data/one_graph/snsheatmap_data_graph_similarity.csv", header = True, index = False)

mask = np.triu(np.ones_like(matrix, dtype=bool))

#fig = sns.heatmap(matrix, mask = mask, xticklabels = False, yticklabels = False)
fig = sns.heatmap(matrix, xticklabels = False, yticklabels = False)
heatmap = fig.get_figure()
heatmap.savefig("../figures/sns_heatmap_graphre_simi_nomask.pdf", dpi = 600)
#heatmap.savefig("../figures/sns_heatmap_graphre_simi.pdf", dpi = 600)

#cluster heatmap
fig2 = sns.clustermap(matrix, xticklabels = False, yticklabels = False)
fig2.savefig("../figures/sns_clustermap_graphre_simi_nomask_600dpi.pdf", dpi = 600)
fig2.data2d.to_csv("../data/one_graph/clustermap_nomask_data2d.csv", header = True, index = False)
#data for ggplot barplot of compounds' targets.
cluster_data2d = pd.read_csv("../data/one_graph/clustermap_nomask_data2d.csv", header = 0)
cluster_list = cluster_data2d.columns
network_file = pd.read_csv("../data/one_graph/network_file_0self.csv", header = 0)
compounds = []
targets = []
for i in cluster_list:
    temp_df = network_file[network_file["drug_name"] == i]
    if temp_df.shape[0] != 0:
        targets_ = pd.unique(temp_df["targets"])
        length = len(targets_)
        compounds.extend([i] * length)
        targets.extend(targets_)
    else:
        temp_df = network_file[network_file["shared_by"] == i]
        targets_ = pd.unique(temp_df["targets"])
        length = len(targets_)
        compounds.extend([i] * length)
        targets.extend(targets_)
print(len(compounds), len(targets))
df = pd.DataFrame({"compounds":compounds, "targets":targets})
df.to_csv("../data/one_graph/bardata_1362_targets.csv", header = True, index = False)

