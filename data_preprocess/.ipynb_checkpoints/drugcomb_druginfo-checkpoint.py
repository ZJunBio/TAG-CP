import pandas as pd
import requests

#Get compounds info-dict from url. 
url = "https://api.drugcomb.org/drugs"
r = requests.get(url)
response = r.json()

#make these information to dataframe format.
drug_name = []
drug_cid = []
drug_smiles = []

for i in response[1:]:
    drug_name.append(i["dname"])
    drug_cid.append(i["cid"])
    drug_smiles.append(i["smiles"])
info = pd.DataFrame({"drug_name":drug_name, "drug_cid":drug_cid, "drug_smiles":drug_smiles},
                    dtype = "int32")
info.to_csv("../data/drugs/drugcomb_info.csv", header = True, index = False)
