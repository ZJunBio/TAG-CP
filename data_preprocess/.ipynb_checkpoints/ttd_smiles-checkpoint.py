import numpy as np
import pandas as pd
import re

network_file = pd.read_csv("../data/one_graph/TTD_drugs_sharing_targets.csv", header = 0)
network_file = network_file.loc[:,['drug_name', 'targets', 'shared_by', 'shared_by_cid']]
smiles = pd.read_csv("../data/one_graph/TTD_drugsmiles.csv", header = 0)
smiles_dict = {smiles.iloc[i,0].strip():smiles.iloc[i,1].strip() for i in smiles.index}
smiles_ = []
for i in network_file["shared_by"]:
    if re.search(';',i) is None:
        try:
            smiles_.append(smiles_dict[i])
        except KeyError:
            smiles_.append("zero")
    else:
        smiles_.append("zero")
network_file["shared_by_smiles"] = smiles_
network_file.to_csv("../data/one_graph/TTD_sharing_targets_fromttd.csv", header = True, index = False)