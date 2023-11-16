#GET DRUG ID IN TTD DATABASE OF 64 compounds.
import re
import pandas as pd

drugs = pd.read_csv("../data/jaaks_druginfo.csv", header = 0, index_col = 0)
with open("D:/Datasets/TTD/TTDdatabase/P1-03-TTD_crossmatching.txt", 'r') as file:
    lines = file.readlines()
TTID = []
TTDmatch = {}
for l  in lines:
    if re.search("PUBCHCID", l):
        ttid = l.split("PUBCHCID")[0].strip()
        cid = l.split("PUBCHCID")[-1].strip()
        #TTDmatch[cid] = ttid
        TTDmatch[ttid] = cid
        
for h, i in enumerate(drugs["cid"]):
    try:
        TTID.append(TTDmatch[str(i)])
    except KeyError:
        print(drugs["drug_name"][h])
        TTID.append("zero")
drugs["ttid"] = TTID
drugs.to_csv("../data/jaaks_druginfo_ttid.csv", header = True, index = False)
##Get target name and id of these compounds:
dt_mapping = pd.read_csv("D:/Datasets/TTD/TTDdatabase/P1-07-Drug-TargetMapping.csv", 
                         header = 0)
#get mapping target id first (each map to 0, 1, or a list)
target_id = []
target_str = []
for i in TTID:
    if i == 'zero':
        target_id.append("zero")
        target_str.append("zero")
    else:
        targets = dt_mapping[dt_mapping["DrugID"] == i]
        ls = targets["TargetID"].to_list()
        if len(targets) == 1:
            target_id.append(ls[0])
            target_str.append(ls[0])
        else:
            target_id.append(ls)
            a = ""
            for s in ls[0:len(ls)-1]:
                a = a + s + ';'
            a = a + ls[-1]
            target_str.append(a)

drugs["tttargets_id"] = target_str
with open("D:/Datasets/TTD/TTDdatabase/P1-01-TTD_target_download.txt", 'r') as file:
    lines = file.readlines()
target_match = {}
for l  in lines:
    if re.search("TARGNAME", l):
        ttid = l.split("TARGNAME")[0].strip()
        ttname = l.split("TARGNAME")[-1].strip()
        target_match[ttid] = ttname
target_names = []
for i in target_id:
    if i == 'zero':
        target_names.append('zero')
    else:
        if type(i) is list:
            a = ""
            for h in i[0: len(i) - 1]:
                a = a + target_match[h] + ';'
            a = a + target_match[i[-1]]
            target_names.append(a)
        else:
            target_names.append(target_match[i])
drugs["tttargets_name"] = target_names
drugs.to_csv("../data/jaaks_druginfo_ttid.csv", header = True, index = False)

#get compounds share this targets.
d = []
t = []
shared_by = []
for h, i in enumerate(target_id):
    if type(i) is list:
        num = len(i)
        for m in i:
            temp = dt_mapping[dt_mapping["TargetID"] == m]
            d_list = temp["DrugID"].to_list()
            shared_by.extend(d_list)
            d.extend([TTID[h]] * len(temp))
            t.extend([m] * len(temp))
    elif i == 'zero':
        continue
    elif type(i) is str:
        temp = dt_mapping[dt_mapping["TargetID"] == i]
        d_list = temp["DrugID"].to_list()
        shared_by.extend(d_list)
        d.extend([TTID[h]] * len(temp))
        t.extend([i] * len(temp))
#     
shared_by_cid = []
no_cid = []
for i in shared_by:
    try:
        shared_by_cid.append(TTDmatch[i])
    except KeyError:
        no_cid.append(i)
        #1961 compounds no pubchem cid information.
        shared_by_cid.append(0)
smiles = pd.read_csv("../data/TTD_drugsmiles.csv", header = 0)
TTD_drugs_sharing_targets = pd.DataFrame({"drug_name":d, "targets":t, 
                                          "shared_by":shared_by, 
                                          "shared_by_cid":shared_by_cid})
TTD_drugs_sharing_targets.to_csv("../data/TTD_drugs_sharing_targets.csv", header = True)

#smiles_dict = {smiles.iloc[i,0].strip():smiles.iloc[i,1].strip() for i in smiles.index}
pubchem_data = pd.read_csv("../data/one_graph/PubChem_ttd_2937.csv", header = 0)
data = pd.read_csv("../data/one_graph/TTD_drugs_sharing_targets.csv", header = 0)
smiles_dict = {pubchem_data["cid"][i]:pubchem_data["canonicalsmiles"][i] for i in pubchem_data.index}
smiles_ = []

for i in data["shared_by_cid"]:
    if i != '0'and re.search(';',i) is None:
        try:
            smiles_.append(smiles_dict[int(i)])
        except KeyError:
            smiles_.append("zero")
    else:
        smiles_.append("zero")
data["shared_by_smiles"] = smiles_
#data.to_csv("../data/one_graph/TTD_drugs_sharing_targets.csv", header = True)
#data[data["shared_by_smiles"] != "zero"].shape
#(2921,6)
##target-pathway mapping.
data = pd.read_csv("../data/one_graph/TTD_drugs_sharing_targets.csv", header = 0)
tlist = pd.unique(data["targets"]).tolist()
with open("D:/Datasets/TTD/TTDdatabase/P1-01-TTD_target_download.txt", 'r') as file:
    lines = file.readlines()
target_match = {}
for l  in lines:
    if re.search("TARGNAME", l):
        ttid = l.split("TARGNAME")[0].strip()
        ttname = l.split("TARGNAME")[-1].strip()
        target_match[ttid] = ttname
target_name = [target_match[i] for i in tlist]
id_name = pd.DataFrame({"target_id":tlist, "target_name":target_name})
id_name.to_csv("../data/one_graph/target_id_name.csv", header = 0)
####KEGG target_pathway mapping.
with open("D:/Datasets/TTD/PathwayInformation/P4-01-Target-KEGGpathway_all.txt", 'r') as file:
    lines = file.readlines()
kegg_pathways = []
for i in tlist:
    for l in lines:
        if re.match(i,l):
            kegg_pathways.append(l)
with open('../data/one_graph/target_kegg_pathway.txt', 'w') as file:
    file.writelines(kegg_pathways)
####wiki target_pathway mapping.
with open("D:/Datasets/TTD/PathwayInformation/P4-06-Target-wikipathway_all.txt", 'r') as file:
    lines = file.readlines()
wiki_pathways = []
for i in tlist:
    for l in lines:
        if re.match(i,l):
            wiki_pathways.append(l)
with open('../data/one_graph/target_wiki_pathway.txt', 'w') as file:
    file.writelines(wiki_pathways)