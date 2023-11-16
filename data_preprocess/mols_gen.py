import pandas as pd

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