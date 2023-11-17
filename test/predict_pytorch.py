import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing

sys.path.append("../data_preprocess/")
sys.path.append("../NN/")

from cell_similarity import cal_cell_similarity
from comb_similarity import comb_sim
from concatenate import concat_vec
import torch
from torch import nn
import torch.nn.functional as F

#An input csv file of with test combinations.
test_path = sys.argv[1]
#An output csv file which used to save predictions.
output_path = sys.argv[2]
#The model path.
model_path = "../models/DNN/pytorch_es5.pt"
#Cell list for training process.
cell_train = "../data/cells/cells_train.txt"

# Define model
class DNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.linear1 = nn.Linear(params.dim1, params.dim2)
        self.bn1 = nn.BatchNorm1d(params.dim2)
        self.linear2 = nn.Linear(params.dim2, params.dim3)
        self.bn2 = nn.BatchNorm1d(params.dim3)
        self.linear3 = nn.Linear(params.dim3, params.dim5)
        self.bn3 = nn.BatchNorm1d(params.dim5)
        self.linear4 = nn.Linear(params.dim5, params.dim6)
        self.bn4 = nn.BatchNorm1d(params.dim6)
        self.linear5 = nn.Linear(params.dim6, 1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        return x

def df_split(df):
    #split cells and drug combinations.
    combs = df.iloc[:,[0,1]]
    cells = df.iloc[:,2].to_list()
    
    return combs, cells

def output(test_path, pre, output_path):
    test = pd.read_csv(test_path)
    #threshold for defining synergy
    threshold = 0.138837
    lab_p = [1 if i > threshold else 0 for i in pre]
    
    test["pre_prob"] = pre
    test["pre_label"] =lab_p
    
    test.to_csv(output_path)
   
    
#The main predicting program.
def predict(test_path = test_path, model_path = model_path, cell_train = cell_train):
    test = pd.read_csv(test_path)
    combs, cells = df_split(test)
    
    #read training cells.
    #cell_k = read_cells()
    
    #calculate the feature of cell lines and drug combinations.
    print("Calculating the feature of cell lines......")
    cell_feature = cal_cell_similarity(cells, cell_train)
    print("Done!")
    print("Calculating the feature of drug combinations......")
    comb_feature = comb_sim(combs)
    print("Done!")
    
    #concatenate feature of DDCs.
    input_ = concat_vec(test, comb_feature, cell_feature)
    input_ = preprocessing.normalize(input_)
    #convert numpy.ndarray to torch tensor.
    input_ = torch.tensor(input_)
    #load model.
    model = torch.load(model_path)
    model.eval()
    
    #predict
    result = model(input_)
    
    #handle results.
    return result

result = predict()
#Save result and put them into the user specified csv file.
result = result.detach().numpy()
output(test_path, result, output_path)
