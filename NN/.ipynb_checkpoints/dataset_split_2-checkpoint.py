#train, validation, test datasets.
import pandas as pd
import numpy as np
import random
import os
random.seed(2023)

syn = pd.read_csv("../data/one_graph/synergy_pair_graph.csv", header = 0)
anta = pd.read_csv("../data/one_graph/anta_pair_graph.csv", header = 0)

len_syn = syn.shape[0]
len_anta = anta.shape[0]

syn_simplified = syn.loc[:,['ANCHOR_NAME','LIBRARY_NAME', 'SIDM', 'Synergy']]
anta_simplified = anta.loc[:,['ANCHOR_NAME','LIBRARY_NAME', 'SIDM', 'Synergy']]

'''
In this splitting, we will balance the counts of synergism and antagonism samples.
'''

syn_simplified.reset_index(drop = True, inplace = True)
#anta_index = random.sample(list(anta_simplified.index), 3 * len_syn)
#anta_simplified = anta_simplified.iloc[anta_index, :]
anta_simplified.reset_index(drop = True, inplace = True)
len_anta = anta_simplified.shape[0]

train_num, test_num = (0.9, 0.1)

train_syn = random.sample(list(range(len_syn)), int(train_num * len_syn))
test_syn = list(set(list(range(len_syn))) - set(train_syn)) 

train_anta = random.sample(list(range(len_anta)), int(train_num * len_anta))
test_anta = list(set(list(range(len_anta))) - set(train_anta)) 

train = pd.concat([syn_simplified.iloc[train_syn,:], anta_simplified.iloc[train_anta,:]],  
                  ignore_index = True)
test = pd.concat([syn_simplified.iloc[test_syn,:], anta_simplified.iloc[test_anta,:]],  
                  ignore_index = True)
print(f"Train dataset has shape of {train.shape}\nTest dataset has shape of {test.shape}")

os.mkdir("../data/datasets")

train.to_csv("../data/datasets/train_2split.csv", header = True, index = False)
test.to_csv("../data/datasets/test_2_split.csv", header = True, index = False)
