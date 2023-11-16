import pandas as pd
import os
import re
import json

synoms = "D:/Datasets/TTD/TTDdatabase/P1-04-Drug_synonyms.txt"
synoms = pd.read_table(synoms, skiprows = list(range(22)), index_col = None,header = None)
synoms.dropna(axis = 0, inplace = True)
synoms.reset_index(drop = True, inplace = True) 

#make mapping of name/synonyms:ttid
ttids = pd.unique(synoms[0])
dict_ = {}
for i in ttids:
    df = synoms[synoms[0] == i].reset_index(drop = True)
    for s in df[2].to_list():
        dict_[s] = i

#save above dictionary
os.mkdir("../data/json")
with open("../data/json/name_ttid.json", 'w') as file:
    json.dump(obj = dict_, fp = file)

##################################22aw
#1362 drug information of drugs in graph
##################################
args = ("../data/one_graph/network_file_0self.csv", "../data/jaaks_druginfo_ttid.csv")
def mols_gen(args = args):
    network_file = pd.read_csv(args[0], header = 0)
    smile_64 = pd.read_csv(args[1], header = 0)

    #drug_ttid and cids mapping
    dict_cid = {network_file.iloc[i,2]:network_file.iloc[i,3] for i in network_file.index}
    dict_smiles = {network_file.iloc[i,2]:network_file.iloc[i,4] for i in network_file.index}
    print(f"Extended compounds: {len(dict_cid)}")
    
    for i in pd.unique(network_file["drug_name"]):
        df = smile_64[smile_64["ttid"] == i]
        dict_cid[i] = df["cid"].values[0]
        dict_smiles[i] = df["canonicalsmiles"].values[0]
    #node list
    nodes = list(dict_cid)
    cids = [dict_cid[i] for i in nodes]
    smiles = [dict_smiles[i] for i in nodes]
    return nodes, cids, smiles

nodes, cids, smiles = mols_gen()
df = pd.DataFrame({"graph_drug":nodes, "cid":cids, "smiles":smiles}, dtype = 'int')
df.to_csv("../data/drugs/graph_drug_info.csv", header = True, index = False)
