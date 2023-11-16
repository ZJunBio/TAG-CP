import pandas as pd
'''
data = pd.read_csv("../data/jaaks_druginfo.csv", header = 0)
drugs = []
targets = []
marker = []
for i, v in enumerate(data["target_type"], start = 1):
    if len(v.split(';')) == 1:
        drugs.append(data["drug_name"][i-1])
        targets.append(data["target_type"][i-1])
        marker.append(str(i))
    elif len(v.split(';')) > 1:
        types = v.split(';')
        num = len(types)
        drugs.extend([data["drug_name"][i-1]] * num)
        targets.extend(types)
        marker.extend([str(i)] * num)
print(len(drugs), len(targets), len(marker))
hp.to_csv("../data/drugs_info4heatmap.csv", header = True)
'''
data = pd.read_csv("../data/jaaks_druginfo.csv", header = 0)
target_info = pd.read_csv("../data/drugs_info4heatmap.csv", header = 0)
len(pd.unique(target_info["target_type"]))
matrix = np.zeros((91,64))
cpdname_index = {v:i for i, v in enumerate(data["drug_name"])}
target_list = pd.unique(target_info["target_type"])

for i,v in enumerate(target_list, start = 1):
    cpds = target_info[target_info["target_type"] == v]["drug_name"]
    cpd_cand = [cpdname_index[c] for c in cpds]
    matrix[i-1,cpd_cand] = i
matrix = pd.DataFrame(matrix, columns = data["drug_name"].to_list(), index = target_list)
matrix.to_csv("../data/target_drug4heatmap.csv", header = True, index = True)

#######
###Make target bar, pathway, cancer type with phase condition and drug name.
unique_words = ['BCL', 'CDK', 'CHEK', 'PLK', 'HDAC', 'BRD', 'ROCK', 'ERBB', 'ERK', 
                'MEK', 'AURK', 'JAK', 'AKT', 'PIK3C', 'MTORC', 'FGFR', 'GSK3']
import re
target_info = pd.read_csv("../data/drugs_info4heatmap.csv", header = 0)
targets = []
for i, v in enumerate(target_info["target_type"]):
    m = 0
    for w in unique_words:
        if re.match(w, v) or re.match(' ' + w, v):
            targets.append(w.strip() + " inhibitor")
            m = 1
    if m == 0:
        targets.append(v.strip())
len(set(targets)) #56
target_classified = pd.DataFrame({"drug_name":target_info["drug_name"].to_list(), "target_type":targets})
import numpy as np
target_classified = target_classified[np.logical_not(target_classified.duplicated(keep = 'first'))]
target_classified.reset_index(drop = True, inplace=True)
target_classified.to_csv("../data/target_classified.csv", header = True, index = False)

#########
###make heatmap of cancer type.
import pandas as pd
import re
data = pd.read_csv("../data/jaaks_druginfo.csv", header = 0)
drugs = []
cancers = []
status = []
for i, v in enumerate(data["Disease"], start = 1):
    types = v.split(';')
    if len(types) == 1:
        l = v.split('(')
        if re.match('phase', l[-1]) or re.match('discontinued', l[-1]) or re.match('investigative', l[-1]):
            drugs.append(data["drug_name"][i-1])
            if len(l) > 2:
                cancers.append(l[0].strip() + "(" + l[1])
            else:
                cancers.append(l[0].strip())
            status.append(l[-1])
        else:
            drugs.append(data["drug_name"][i-1])
            cancers.append(v.strip())
            status.append("approved")
    elif len(types) > 1:
        num = len(types)
        drugs.extend([data["drug_name"][i-1]] * num)
        for i in types:
            l = i.split('(')
            if re.match('phase', l[-1]) or re.match('discontinued', l[-1]) or re.match('investigative', l[-1]):
                if len(l) > 2:
                    cancers.append(l[0].strip() + "(" + l[1])
                else:
                    cancers.append(l[0].strip())
                status.append(l[-1])
            else:
                cancers.append(i.strip())
                status.append("approved")
cancer_info = pd.DataFrame({'drug_name':drugs, 'cancer_type': cancers, 'status':status})
cancer_info.to_csv("../data/jaaks_cancerinfo.csv", index = False, header = True)

######
########
import numpy as np
matrix = np.zeros((27, 64))
matrix = pd.DataFrame(matrix, columns = pd.unique(cancer_info["drug_name"]), index = pd.unique(cancer_info["cancer_type"]))
for i in cancer_info.index:
    s = cancer_info.iloc[i]
    matrix.loc[s['cancer_type'], s['drug_name']] = s['status']
matrix.replace({"phase 1": 1,"phase 2": 2, "phase 3": 3,  "discontinued": 4, "investigative": 5, "approved": 6}, inplace=True)
matrix.to_csv("../data/cancer_info_matrix.csv", header = True, index = True)