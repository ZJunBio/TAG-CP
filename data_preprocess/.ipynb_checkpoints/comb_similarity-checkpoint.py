import random
import pandas as pd
import numpy as np
import sys
import pickle
from drugs_sim import drug_sim
random.seed(2023)
# For combinations consists of drugs that both included in graph training.
# Reserve up to 716 drug combinations to keep same dimension.

path = "../data/drugs/train_combinations.csv"
jaaks_path = "../data/jaaks64drug_cids.csv"
feature_path = "../data/drugs/graph_re.pickle"
up_value = 716
#preprocess replicated combinations.
def unique_comb(df):
    '''
    df: 2 columns
    '''
    uni_index = df.duplicated().to_list()
    uni_index = np.logical_not(uni_index)
    df = df.loc[uni_index,:].reset_index(drop = True)
    
    return df
#read train_combination.
def comb_known(path = path, jaaks_path = jaaks_path):
    comb = pd.read_csv(path, header = None, index_col = None)
    j = pd.read_csv(jaaks_path, header = 0, index_col = None)
    nameid_dict = {j.iloc[:,0][i]:j.iloc[:,1][i] for i in j.index}
    comb_1 = [nameid_dict[i] for i in comb[0]]
    comb_2 = [nameid_dict[i] for i in comb[1]]
    
    comb = pd.DataFrame({"drug_row":comb_1, "drug_col":comb_2})
    
    return comb

#cross match the unique test combination and get differrentiate number.
def cross_match(df, df_k):
    '''
    df: test combinations
    df_k: combinations used in train
    '''
    df_list = [set([df.iloc[i,0], df.iloc[i,1]]) for i in df.index]
    df_k_list = [set([df_k.iloc[i,0], df_k.iloc[i,1]]) for i in df_k.index]
    
    #remove training combinations overlap with test data.
    df_overlap = [1 if s in df_list else 0 for s in df_k_list]
    df_index = [False if s in df_list else True for s in df_k_list]
    df_k = df_k.loc[df_index,:].reset_index(drop = True)
    
    num = df.shape[0] - sum(df_overlap)
    #sample combinations equal to un-overlap combs. in test.
    #print(f"population: {list(df.index)}, sample: {df.shape[0] - num}")
    index_ = random.sample(list(df_k.index), df_k.shape[0] - num)
    df_k = df_k.iloc[index_,:].reset_index(drop = True)
    
    #concat 2 dataframe.
    df = pd.concat([df, df_k], ignore_index = True)
    
    return df

#load representation of 1362 compounds in graph.
def load_feature(path):
    with open(path, 'rb') as file:
        dict_ = pickle.load(file)
        
    return dict_

def comb_sim(df):
    '''
    df: 2 columns ['drug_row', 'drug_col']
    '''
    df = unique_comb(df)
    print(df)
    
    if df.shape[0] > up_value:
        raise ValueError(f"{df.shape[0]} over max combination number")
    #read train combinations and convert name to cid.
    comb_k = comb_known()
    #cross match the unique test combination.
    df = cross_match(df, comb_k)
    
    #load representation of 1362 drugs.
    feature_dict = load_feature(feature_path)
    
    #calculate similarities.
    sim_matrix = drug_sim(df, feature_dict)
    
    return sim_matrix