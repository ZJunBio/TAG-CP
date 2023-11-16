#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 16:53:59 2022

@author: jun
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix

#Concat the vector of drug pair and cell line.
def concat_vec(df, drug_sim, cell_sim):
    array = np.empty((df.shape[0], drug_sim.shape[0] + cell_sim.shape[0]), dtype = np.float32)
    for i in range(len(df.index)):
        try:
            dname = df["COMBINATION"][i]
            cname = df["CELLNAME"][i]
        except KeyError:
            dname = df["COMBINATION"].iat[i]
            cname = df["CELLNAME"].iat[i]
        array[i] = np.hstack((drug_sim[dname].values, cell_sim[cname].values))
    return array

#Extracting part of dataframes uses given indexes.
def extraction(list_f, df_f, header = 0, index_col = 0):
    with open(list_f, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    df = pd.read_csv(df_f, header = header, index_col = index_col)
    df = df[labels]
    
    return df


#Scaling original data.
def scale(df = None, scale_axis = 0, drop_axis = 1, index_col = 0, header = 0):
    if type(df) == str:
        exp = pd.read_csv(df, index_col = index_col, header = header)
    else:
        exp = df
    
    exp = exp.dropna(axis=drop_axis) #0, rows; 1, columns  
    
    exp = pd.DataFrame(preprocessing.normalize(exp.to_numpy(), axis = scale_axis, norm = "l2"), 
                      index = exp.index, columns = exp.columns)
    return exp

#Calculating the exponential of all elements in the euclidean matrix.
def gaussian_kernel(df = None, sigma = 0.01, dtype = np.float32, to_numpy = True, dframe = True):
    #df is L2 norm matrix.
    if to_numpy:
        matrix = np.exp(-sigma * (df.to_numpy(dtype = dtype)))
        print(matrix[0:5,0:5])
    else:
        matrix = np.exp(-sigma * df.astype(dtype = dtype))
        print(matrix[0:5,0:5])
        
    if dframe:
        return pd.DataFrame(matrix, index = df.index, columns = df.columns, dtype = dtype)
    else:
        return matrix

#Calculating drugs similarity.
def drugsim_cal(com_path, reset_index = False, scale_axis = 0, int_index = True, desc = "../cell_drug/descrip_105.csv"):
    com = pd.read_csv(com_path, header = 0, index_col = 0)
    if reset_index:
        com.reset_index(drop = True)
    index = pd.unique(com["COMBINATION"])
    sim_matrix = pd.DataFrame(index = index, columns = index)
    #Generating drug-descriptors dictionary for fast fetching vectors and calculating.
    desc = pd.read_csv(desc, header = 0, index_col = 0)
    desc = desc.dropna(axis=1)
    desc_scaled = pd.DataFrame(preprocessing.normalize(desc.to_numpy(), axis = 0, norm = "l2"),
                               columns=desc.columns, index = desc.index)
    drug_dict = {}
    for i in range(len(desc_scaled.index)):
        drug_dict[desc_scaled.index[i]] = desc_scaled.iloc[i,:].to_numpy()
    
    num = len(index)
    #Filling half of symmetric matrix element by element.
    for i in range(num):  
        if int_index:
            X = list(map(int, index[i].split("+")))
        else:
            X = index[i].split("+")
        for s in range(i, num):
            #calculating simlarity
            if int_index:
                Y = list(map(int, index[s].split("+")))
            else:
                Y = index[s].split("+")
            sim1 = euclidean(drug_dict[X[0]], drug_dict[Y[0]])
            sim2 = euclidean(drug_dict[X[1]], drug_dict[Y[1]])
            sim3 = euclidean(drug_dict[X[0]], drug_dict[Y[1]])
            sim4 = euclidean(drug_dict[X[1]], drug_dict[Y[0]])
            s1 = 0.5 * (sim1 + sim2)
            s2 = 0.5 * (sim3 + sim4)
            sim = max(s1, s2)
            sim_matrix.iat[i,s] = sim
    #Completing the  whole symmetric matrix.
    
    for i in range(num):
        for s in range(i):
            sim_matrix.iat[i, s] = sim_matrix.iat[s, i]
        sim_matrix.iat[i, i] = 0
    print("Simlarity of drug combinations:")
    print(sim_matrix.iloc[0:5, 0:5])
    return sim_matrix


#Calculating cells similarity.
def cellsim_cal(matrix_path,index_col= None, transp = True, sigma = 0.01):
    if type(matrix_path) == str:
        matrix = pd.read_csv(matrix_path, index_col = index_col, header=0)
    else:
        matrix = matrix_path
    #matrix = matrix.dropna(axis=drop_axis) #0, rows; 1, columns   
    if transp:
        dist_d = distance_matrix(matrix.T.to_numpy(), matrix.T.to_numpy(), p = 2)
    else:
        dist_d = distance_matrix(matrix.to_numpy(), matrix.to_numpy(), p = 2)
    
    dist_d = gaussian_kernel(df = dist_d, sigma = sigma, to_numpy = False, dframe = False)

    dist_d = pd.DataFrame(dist_d, index = matrix.columns, columns = matrix.columns, dtype = np.float32)

    return dist_d

#Random sampling subset of samples with labels.
#def rsample()

#Writing matrix to csv farmat.
def mat2csv(mat, cnames, f_path):
    #require pandas package
    #import pandas as pd
    if type(cnames) is not list:
        print("Please give a list in a list type. ")
        return 
    frame = pd.DataFrame(mat, columns = cnames)
    frame.to_csv(f_path, header = True, index = False, quotechar=' ')

# Given precision and recall value with different thresholds, function computes the best f_1 score
def f1_score_best(precision, recall, thresholds):
    '''
    fb_score = (1+b^2) * precision * recall / (b^2 * precision + recall)
    f1_score = 2 * precision * recall / (precision + recall)
    '''
    f1_score = np.divide((2 * np.multiply(precision, recall)),  (precision + recall))
    best_index = np.argmax(f1_score)
    best_score = f1_score[best_index]
    best_threshold = thresholds[best_index]
    best_precision = precision[best_index]
    best_recall = recall[best_index]
    return best_score, best_threshold, best_precision, best_recall

#Random sampling n samples from all samples uses indexs.
def rsample(train_set, label, sample_num, seed = 2022):
    
    import random
    random.seed(seed)
    index = sorted(random.sample(range(train_set.shape[0]), sample_num))
    train_subset = train_set[index]
    trainlab_subset = label[index]
    return train_subset, trainlab_subset  