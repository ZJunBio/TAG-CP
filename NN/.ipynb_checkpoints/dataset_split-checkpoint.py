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

syn_simplified.reset_index(drop = True, inplace = True)
anta_simplified.reset_index(drop = True, inplace = True)

train_num, val_num = (0.8, 0.1)

train_syn = random.sample(list(range(len_syn)), int(train_num * len_syn))
val_syn = random.sample(list(range(len_syn)), int(val_num * len_syn))
test_syn = random.sample(list(range(len_syn)), 
                         (len_syn - int(train_num * len_syn) - int(val_num * len_syn)))

train_anta = random.sample(list(range(len_anta)), int(train_num * len_anta))
val_anta = random.sample(list(range(len_anta)), int(val_num * len_anta))
test_anta = random.sample(list(range(len_anta)), 
                         (len_anta - int(train_num * len_anta) - int(val_num * len_anta)))

train = pd.concat([syn_simplified.iloc[train_syn,:], anta_simplified.iloc[train_anta,:]],  
                  ignore_index = True)
val = pd.concat([syn_simplified.iloc[val_syn,:], anta_simplified.iloc[val_anta,:]],  
                  ignore_index = True)
test = pd.concat([syn_simplified.iloc[test_syn,:], anta_simplified.iloc[test_anta,:]],  
                  ignore_index = True)
print(f"Train dataset has shape of {train.shape} \nValidation dataset has shape of {val.shape} \nTest dataset has shape of {test.shape}")

os.mkdir("../data/datasets")

train.to_csv("../data/datasets/train.csv", header = True, index = False)
val.to_csv("../data/datasets/validation.csv", header = True, index = False)
test.to_csv("../data/datasets/test.csv", header = True, index = False)