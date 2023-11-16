import pandas as pd
import numpy as np
import random
'''
Generate neighbors and label for each node in each graph.
'''
drug_info = pd.read_csv("../data/jaaks_druginfo.csv", header = 0)
pathway_info = pd.read_table("../data/source_target_pathway0self.txt", header = 0)
pathways = pd.unique(drug_info["target_pathway"])
cids = pd.read_csv("../data/jaaks64drug_cids.csv", header = 0)
cids = {cids.iloc[i]["compound"]:cids.iloc[i]["CID"] for i in cids.index}
neighbor_smiles = pd.DataFrame(columns = ["cid", "isosmiles", "canonicalsmiles"])
for p in pathways:
    dataframe = pathway_info[pathway_info["pathway"] == p].reset_index(drop = True)
    drugs = drug_info[drug_info["target_pathway"] == p]
    drugs = drugs["drug_name"].to_list()
    source = []
    target = []
    coef = []
    pathway = []
    #neighbor_smiles = pd.DataFrame(columns = ["cid", "isosmiles", "canonicalsmiles"])
    for d in drugs:
        cid = cids[d]
        path = f'PubChem_compound_structure_by_cid_similarity_CID{cid} structure.csv'
        neighbors = pd.read_csv("../data/"+path)
        num = len(neighbors)
        if num > 10:
            index = random.sample(list(neighbors.index), 10)
            neighbor_nodes = neighbors.iloc[index]
            source.extend(neighbor_nodes["cid"].to_list())
            target.extend([d] * 10)
            coef.extend([2] * 10)
            pathway.extend([p] * 10)
            neighbor_smiles = pd.concat([neighbor_smiles, 
                                         neighbor_nodes.loc[:,["cid", "isosmiles", "canonicalsmiles"]]], 
                                       ignore_index = True)
        elif num < 10:
            neighbor_nodes = neighbors
            source.extend(neighbor_nodes["cid"].to_list())
            target.extend([d] * num)
            coef.extend([2] * num)
            pathway.extend([p] * num)
            neighbor_smiles = pd.concat([neighbor_smiles, 
                                         neighbor_nodes.loc[:,["cid", "isosmiles", "canonicalsmiles"]]], 
                                       ignore_index = True)
    neighbor_smiles.reset_index(drop = True, inplace = True)
    temp = pd.DataFrame({"source":source, "target":target, "coeff":coef, "pathway":pathway})
    dataframe = pd.concat([dataframe, temp], ignore_index = True)
    if re.search("/", p) or re.search(";", p):
        p = re.sub("/", "_", p)
        p = re.sub(";", "_", p)
        dataframe.to_csv("../data/"+f"source_target_pathway_{p}.csv", header = True, index = False)
    else:
        dataframe.to_csv("../data/"+f"source_target_pathway_{p}.csv", header = True, index = False)
neighbor_smiles.to_csv("../data/jaaks_neighbor_nodes_smiles.csv", header = True, index = False)

'''
calculate sigaling pathways similarity between compounds using signaturizer
'''
from signaturizer import Signaturizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy import stats
import pandas as pd
import numpy as np
##heatmap validation of signaturizer with cosine similarity.
cpds = pd.read_csv("../data/jaaks_druginfo.csv", header = 0)
sign = Signaturizer('C3')
#sign = Signaturizer('B1')
smile_list = cpds["canonicalsmiles"].to_list()
results = sign.predict(smile_list)
results.signature.shape #(64, 128)
matrix = np.zeros((len(cpds), len(cpds)))
indexes = cpds["drug_name"].to_list()

for i in range(len(cpds)):
    for m in range(len(cpds)):
        s = cosine_similarity(results.signature[i].reshape(1,-1), results.signature[m].reshape(1, -1))
        matrix[i, m] = s[0][0]
simi_df = pd.DataFrame(matrix, columns = indexes, index = indexes)
simi_df.to_csv("../data/cosine_similarity_64cpds_mechanism.csv", header = True, index = True)
##################################################################
##heatmap validation of signaturizer with Minkowski distance.
for i in range(len(cpds)):
    for m in range(len(cpds)):
        s = distance.minkowski(results.signature[i], results.signature[m], p = 2)
        matrix[i, m] = s
simi_df = pd.DataFrame(matrix, columns = indexes, index = indexes)
simi_df.to_csv("../data/minkowski_similarity_64cpds_mechanism.csv", header = True, index = True)
##################################################################
####heatmap validation of signaturizer with Pearson correlation.
for i in range(len(cpds)):
    for m in range(len(cpds)):
        s = stats.pearsonr(results.signature[i], results.signature[m])
        matrix[i, m] = s.statistic
simi_df = pd.DataFrame(matrix, columns = indexes, index = indexes)
simi_df.to_csv("../data/pearsonr_similarity_64cpds_sigpath.csv", header = True, index = True)

'''
Filter the neighbor compounds of similar signaling pathways by pearson relationship.  
'''

import pandas as pd
import numpy as np
from signaturizer import Signaturizer
from scipy import stats
import random
import re

drug_info = pd.read_csv("../data/jaaks_druginfo.csv", header = 0)
pathway_info = pd.read_table("../data/source_target_pathway0self.txt", header = 0)
pathways = pd.unique(drug_info["target_pathway"])
cids = pd.read_csv("../data/jaaks64drug_cids.csv", header = 0)
cids = {cids.iloc[i]["compound"]:cids.iloc[i]["CID"] for i in cids.index}
smiles_64 = {drug_info["drug_name"][i]:drug_info["canonicalsmiles"][i] for i in drug_info.index}
neighbor_smiles = pd.DataFrame(columns = ["cid", "isosmiles", "canonicalsmiles"])

for p in pathways:
    dataframe = pathway_info[pathway_info["pathway"] == p].reset_index(drop = True)
    drugs = drug_info[drug_info["target_pathway"] == p]
    smiles_ = drugs["canonicalsmiles"].to_list()
    drugs = drugs["drug_name"].to_list()
    source = []
    target = []
    coef = []
    pathway = []
    similarities = []
    sign = Signaturizer('C3')
    #similarities between original sources and targets.
    if len(dataframe) > 0:
        for i in dataframe.index:
            s = [smiles_64[dataframe["source"][i]], smiles_64[dataframe["target"][i]]]
            sv = sign.predict(s)
            similarities.append(stats.pearsonr(sv.signature[0], sv.signature[-1]).statistic)
    print("original similarities",len(similarities))
    #extend neighbor nodes based on signature similarities for each original node
    for i,d in enumerate(drugs):
            sims = []
            cid = cids[d]
            path = f'PubChem_compound_structure_by_cid_similarity_CID{cid} structure.csv'
            neighbors = pd.read_csv("../data/"+path)
            smiles = neighbors["canonicalsmiles"].to_list()
            smiles.append(smiles_[i])
            vecs = sign.predict(smiles)
            print(vecs.signature.shape)
            #similarities between extend nodes and original nodes.
            for n in range(vecs.signature.shape[0] - 1):
                sims.append(stats.pearsonr(vecs.signature[n], vecs.signature[-1]).statistic)
            index = [i for i, d in enumerate(sims) if d > 0.5]
            
            if len(index) > 10:
                index = random.sample(index, 10)
                #print(len(np.array(sims)[index]))
                similarities.extend(np.array(sims)[index])
                neighbor_nodes = neighbors.iloc[index,:]
                source.extend(neighbor_nodes["cid"].to_list())
                target.extend([d] * len(index))
                coef.extend([2] * len(index))
                pathway.extend([p] * len(index))
                neighbor_smiles = pd.concat([neighbor_smiles, 
                                             neighbor_nodes.loc[:,["cid", "isosmiles", "canonicalsmiles"]]], 
                                            ignore_index = True)
            elif len(index) < 10 and len(index) >= 1:
                neighbor_nodes = neighbors.iloc[index,:]
                #print(len(sims))
                similarities.extend(np.array(sims)[index])
                #similarities.extend(sims)
                source.extend(neighbor_nodes["cid"].to_list())
                target.extend([d] * len(index))
                coef.extend([2] * len(index))
                pathway.extend([p] * len(index))
                neighbor_smiles = pd.concat([neighbor_smiles, 
                                             neighbor_nodes.loc[:,["cid", "isosmiles", "canonicalsmiles"]]], 
                                            ignore_index = True)
            elif len(index) == 0:
                continue
    neighbor_smiles.reset_index(drop = True, inplace = True)
    temp = pd.DataFrame({"source":source, "target":target, "coeff":coef, "pathway":pathway})
    dataframe = pd.concat([dataframe, temp], ignore_index = True)
    dataframe["similarity"] = similarities
    if re.search("/", p) or re.search(";", p):
        p = re.sub("/", "_", p)
        p = re.sub(";", "_", p)
        dataframe.to_csv("../data/"+f"source_target_pathway_{p}.csv", header = True, index = False)
    else:
        dataframe.to_csv("../data/"+f"source_target_pathway_{p}.csv", header = True, index = False)
neighbor_smiles.to_csv("../data/jaaks_neighbor_nodes_smiles.csv", header = True, index = False)
