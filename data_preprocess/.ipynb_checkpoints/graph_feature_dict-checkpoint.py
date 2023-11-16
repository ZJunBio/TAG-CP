import pandas as pd
import numpy as np
from sklearn import preprocessing
from mols_gen import mols_gen
import pickle

args = ("../data/one_graph/network_file_0self.csv", "../data/jaaks_druginfo_ttid.csv")
nodes_repre = np.load("../data/one_graph/repr_nodes_1362_2layers.npy", allow_pickle = True)
nodes, cids, smiles = mols_gen(args)

#normalize graph representation using L2 normalization at sample level
nodes_repre = preprocessing.normalize(nodes_repre, norm = "l2", axis = 1)
comp_feature_dict = {v:nodes_repre[i,] for i, v in enumerate(cids)}

with open("../data/drugs/graph_re.pickle", 'wb') as file:
    pickle.dump(comp_feature_dict, file)