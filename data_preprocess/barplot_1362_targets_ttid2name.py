import pandas as pd
import os
os.chdir("D:/python_work/Interpretation_DDC/")
os.listdir()

df = pd.read_csv("data/one_graph/target_id_name.csv", header = None, index_col = 0)
df[0:5]

id_name = {}
for i in df.index:
  id_ = df.iloc[i, 0]
  n_ = df.iloc[i, 1]
  name = n_.split("(")[-1]
  name = name.split(")")[0]
  id_name[id_] = name

graph1 = pd.read_csv("data/one_graph/bardata_1362_targets.csv", header = 0)
graph2 = pd.read_csv("data/one_graph/bardata_1362_targets_1stlayer.csv", header = 0)
macc = pd.read_csv("data/one_graph/bardata_1362_targets_macc.csv", header = 0)

for i in graph1.index:
  tid = graph1.iloc[i,1]
  graph1.iloc[i,1] = id_name[tid] 

graph1.to_csv("data/one_graph/bardata_1362_targets_v2.csv", header = True, index = False)

for i in graph2.index:
  tid = graph2.iloc[i,1]
  graph2.iloc[i,1] = id_name[tid] 

graph2.to_csv("data/one_graph/bardata_1362_targets_1stlayer_v2.csv", header = True, index = False)

for i in macc.index:
  tid = macc.iloc[i,1]
  macc.iloc[i,1] = id_name[tid] 

macc.to_csv("data/one_graph/bardata_1362_targets_macc_v2.csv", header = True, index = False)
