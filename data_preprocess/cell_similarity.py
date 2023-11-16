import json
import pandas as pd
from sklearn import preprocessing
import sys
import random
from utils import cellsim_cal
sys.path.append("../NN/")

#For input of test cell lines, caculate similarities with cells.
#preprocess the TPM file.
tpm_path = "../data/cells/rnaseq_tpm_20220624_symbol.csv"
cell_train_path = "../data/cells/cells_train.txt"
jsons = ["../data/cells/SIDM2Symbol.json", "../data/cells/Symbol2SIDM.json"]
#convert SIDM to Symbol or convert Symbol to SIDM
def id_convert(cells, json_file = jsons, id_target = "symbol"):
    '''
    id_target: "symbol" or "sidm"
    '''
    if id_target == "symbol":
        json_file = json_file[0]
    else:
        json_file = json_file[1]
    #load dictionary.
    with open(json_file, 'r') as file:
        json_d = json.load(file)
    
    cells = [json_d[c] for c in cells]
    
    return cells
    
def tpm_pre(path = tpm_path):
    tpm = pd.read_csv(path, header = 0, index_col = 0)
    tpm.dropna(axis = 0, inplace = True, ignore_index = True)
    
    return tpm

def tpm_select(tpm, cells):
    #location required tpm.
    tpm = tpm.loc[:, cells]
    index = [True if tpm.iloc[i,:].sum() != 0 else False for i in range(tpm.shape[0])]
    tpm = tpm.loc[index,:].reset_index(drop = True)
    #name
    cell_name = tpm.columns
    #normalize
    tpm = preprocessing.normalize(tpm, norm = "l2", axis = 1)
    tpm = pd.DataFrame(tpm, columns = cell_name) 
    
    return tpm

def cal_cell_similarity(cell_list, train_cells, symbol = "symbol"):
    '''
    symbol: "sidm", "symbol"
    '''
    #read TPM expression file.
    tpm = tpm_pre()
    
    #read cells used in train dataset.
    with open(train_cells) as file:
        cell_k = file.readlines()
    cell_k = [c.strip() for c in cell_k]
    
    #convert symbols of cell line.
    if symbol == "symbol":
        cell_k = id_convert(cell_k, id_target = symbol)
    elif symbol == "sidm":
        cell_list = id_convert(cell_list, id_target = symbol)
    
    cell_list = list(set(cell_list))
    uni_cell = set(cell_list) - set(cell_k)
    #format a dataframe.
    cell_k = random.sample(cell_k, k = len(cell_k) - len(uni_cell)) #keep dimension of cell similarity.
    cell_k.extend(list(uni_cell))
    df = tpm_select(tpm, cell_k)
    
    #calculation and generate similarity matrix of cell lines.
    cell_sim = cellsim_cal(df)
    
    return cell_sim