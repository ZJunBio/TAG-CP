from sklearn import preprocessing
from utils import drug_sim, cellsim_cal
import pandas as pd
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from scipy.spatial import distance_matrix

args = ("../data/one_graph/network_file_0self.csv", "../data/jaaks_druginfo_ttid.csv")
dc_syn = pd.read_csv("../data/drugs/synergy_pair_dnf.csv", header = 0)
dc_anta = pd.read_csv("../data/drugs/anta_pair_dnf.csv", header = 0)
nodes_repre = np.load("../data/one_graph/repr_nodes_1362_2layers.npy", allow_pickle = True)

#List molecules according to GNN training order.
def mols_gen(args = args):
    network_file = pd.read_csv(args[0], header = 0)
    smile_64 = pd.read_csv(args[1], header = 0)

    #drug_ttid and smiles mapping
    dict_smile = {network_file.iloc[i,2]:network_file.iloc[i,4] for i in network_file.index}
    print(f"Extended compounds: {len(dict_smile)}")
    compounds = pd.unique(network_file["drug_name"])
    print(f"Compounds in jaak's dataset: {len(compounds)}")
    ttid_name = []
    for i in pd.unique(network_file["drug_name"]):
        df = smile_64[smile_64["ttid"] == i]
        dict_smile[i] = df["canonicalsmiles"].values[0]
        ttid_name.append(df["drug_name"].values[0])
    print(f"Compounds name in jaak's dataset: {len(ttid_name)}")
    #node list
    nodes = list(dict_smile)
    node_index = {v:i for i,v in enumerate(nodes)}
    node_mols = [Chem.MolFromSmiles(dict_smile[i]) for i in nodes]
    #node features
    return nodes, compounds, ttid_name, node_mols

nodes, compounds, ttid_name, node_mols = mols_gen()

c_ = set(ttid_name)
syn_anchor = [True if i in c_ else False for i in dc_syn["ANCHOR_NAME"]]
syn_library = [True if i in c_ else False for i in dc_syn["LIBRARY_NAME"]]
dc_syn_ = dc_syn[np.logical_and(syn_anchor, syn_library)]

anta_anchor = [True if i in c_ else False for i in dc_anta["ANCHOR_NAME"]]
anta_library = [True if i in c_ else False for i in dc_anta["LIBRARY_NAME"]]
dc_anta_ = dc_anta[np.logical_and(anta_anchor, anta_library)]

cells_set = set(pd.unique(dc_syn_["SIDM"])) | set(pd.unique(dc_anta_["SIDM"]))
sanger_models = pd.read_csv("D:/Datasets/CellModelPassports/rnaseq_tpm_20220624.csv", 
                            header = 0, index_col = 0, skiprows = [1,2,3,4])
sanger_models = sanger_models.loc[:, list(cells_set)]
sanger_models.reset_index(drop = True, inplace = True)

sanger_models.to_csv("../data/one_graph/sanger_models.csv", header = True, index = False)
dc_syn_.to_csv("synergy_pair_graph.csv", header = True, index = False)
dc_anta_.to_csv("anta_pair_graph.csv", header = True, index = False)

df = pd.concat((dc_syn_.loc[:,["ANCHOR_NAME", "LIBRARY_NAME", "SIDM","Synergy"]], 
                dc_anta_.loc[:,["ANCHOR_NAME", "LIBRARY_NAME", "SIDM","Synergy"]]), 
               ignore_index = True)
#df.to_csv("../data/drugs/train_combinations.csv", header = True, index = False)
nodes_index = [i for i, v in enumerate(nodes) if v in compounds]
ttid_index = [v for i, v in enumerate(nodes) if v in compounds]
temp_dict = {compounds[i]:ttid_name[i] for i in range(len(compounds))}

graph_re = nodes_repre[nodes_index,]
#normalize graph representation using L2 normalization at sample level
graph_re = preprocessing.normalize(graph_re, norm = "l2", axis = 1)
comp_feature_dict = {temp_dict[v]:graph_re[i,] for i, v in enumerate(ttid_index)}

'''
drug combination similarities.
'''
sim_matrix = drug_sim(df, comp_feature_dict)
sim_matrix.to_csv("../data/one_graph/skernel_716dcsimlarity_2lwayers128.csv", header = True, index = True)

'''
cell line similarities.
'''
sanger_models.dropna(axis = 0, inplace = True, ignore_index = True)
index = [True if sanger_models.iloc[i,:].sum() != 0 else False for i in range(sanger_models.shape[0])]
sanger_models = sanger_models.loc[index,:]
sanger_models.reset_index(drop = True, inplace = True)
temp = preprocessing.normalize(sanger_models, norm = "l2", axis = 1)
temp = pd.DataFrame(temp, columns = sanger_models.columns) 

cell_sim = cellsim_cal(temp)
cell_sim.to_csv("../data/one_graph/gkernel_125cellsimlarity.csv", header = True, index = True)
