import json
import pandas as pd

path = "../data/cells/sdim_symbol.csv"
data = pd.read_csv(path, header = 0, index_col = None, dtype = 'str')

dict_sidm2symbol = {i:data.loc[data.index[0], i] for i in data.columns}
dict_symbol2sidm = {data.loc[data.index[0], i]:i for i in data.columns}



with open("../data/cells/SIDM2Symbol.json", 'w') as file:
    json.dump(dict_sidm2symbol, file)

with open("../data/cells/Symbol2SIDM.json", 'w') as file:
    json.dump(dict_symbol2sidm, file)