import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing

sys.path.append("../data_preprocess/")
sys.path.append("../NN/")

from cell_similarity import cal_cell_similarity
from comb_similarity import comb_sim
from concatenate import concat_vec
import tensorflow as tf

test_path = "breast_test.csv"
model_path = "../models/DNN/BNrelu256be_3SGDes150Fold.h5"
cell_train = "../data/cells/cells_train.txt"

def df_split(df):
    #split cells and drug combinations.
    combs = df.iloc[:,[0,1]]
    cells = df.iloc[:,2].to_list()
    
    return combs, cells

def predict(test_path = test_path, model_path = model_path, cell_train = cell_train):
    test = pd.read_csv(test_path)
    combs, cells = df_split(test)
    
    #read training cells.
    #cell_k = read_cells()
    
    #calculate similarities.
    cell_feature = cal_cell_similarity(cells, cell_train)
    comb_feature = comb_sim(combs)
    print(comb_feature.columns)
    
    #concatenate feature.
    input_ = concat_vec(test, comb_feature, cell_feature)
    
    #load model.
    model = tf.keras.models.load_model(model_path)
    
    #predict
    result = model.predict(input_)
    
    #handle results.
    return result
    comb_feature = comb_sim(combs)
    
    #concatenate feature.
    input_ = concat_vec(test, comb_feature, cell_feature)
    input_ = preprocessing.normalize(input_)
    input_ = tf.Tensor(input_)
    
    #load model.
    model = tf.keras.models.load_model(model_path)
    
    #predict
    result = model.predict(input_)
    
    #handle results.
    return result