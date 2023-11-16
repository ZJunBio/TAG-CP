import numpy as np
import pandas as pd
from scipy import stats
import torch
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.metrics import accuracy_score
from sklearn import metrics
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
import torch
import random

#set up random seeds.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#####################
#validation function
#####################

def val_acc(model, data):
    model.eval()
    _, pred = model(data)
    loss = F.binary_cross_entropy(pred[data.val_mask], data.y[data.val_mask])
    pred_label = np.zeros(data.y[data.val_mask].size())
    pred = pred[data.val_mask].detach().numpy()
    for i, v in enumerate(data.y[data.val_mask].detach().numpy()):
        sum_ = int(v.sum())
        if sum_ == 1:
            l = np.argsort(pred[i])[-1]
            pred_label[i, l] = 1
        else:
            l = np.argsort(pred[i])[-sum_:]
            pred_label[i, l] = 1
    acc = accuracy_score(pred_label, data.y[data.val_mask].detach().numpy())
    
    return acc, loss

#####################
#validation function
#####################

def train_acc(out, data):
    pred_label = np.zeros(data.y[data.train_idx].size())
    pred = out[data.train_idx].detach().numpy()
    for i, v in enumerate(data.y[data.train_idx].detach().numpy()):
        sum_ = int(v.sum())
        if sum_ == 1:
            l = np.argsort(pred[i])[-1]
            pred_label[i, l] = 1
        else:
            l = np.argsort(pred[i])[-sum_:]
            pred_label[i, l] = 1
    acc = accuracy_score(pred_label, data.y[data.train_idx].detach().numpy())
    
    return acc

#####################
#early stopping
#####################

class EarlyStopping():
    
    def __init__(self, save_path, patience = 20, delta = 0):
        self.patience = patience
        self.delta = delta
        self.earlystop = False
        self.counter = 0
        self.best_loss = None
        self.save_path = save_path
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.earlystop = True

#################################
#pearson calculation of 2 vectors
#################################
def pearson_cal(arr1, arr2):
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        res = stats.pearsonr(arr1, arr2)
        return res.statistic, res.pvalue
    else:
        print("arr1 and arr2 are not numpy array format.")
        
#####################################
#TanimotoSimilarity of 2 fingerprints
#####################################
def fp_sim(fp_list):
    sim_list = []
    for i, m in enumerate(fp_list):
        for j, n in enumerate(fp_list):
            if j > i:
                sim = DataStructs.TanimotoSimilarity(m, n)
                sim_list.append(sim)
    return np.array(sim_list)

#####################################
#DiceSimilarity of 2 fingerprints
#####################################
def fp_dicesim(fp_list):
    sim_list = []
    for i, m in enumerate(fp_list):
        for j, n in enumerate(fp_list):
            if j > i:
                sim = DataStructs.DiceSimilarity(m, n)
                sim_list.append(sim)
    return np.array(sim_list)

#####################################
#pearson correlation of n compounds
#####################################

def repr_sim(rep, self_ = False):
    sim_list = []
    if self_:
        for i, m in enumerate(rep):
            for j, n in enumerate(rep):
                if j >= i:
                    sim, _ = pearson_cal(m, n)
                    sim_list.append(sim)
    else:
        for i, m in enumerate(rep):
            for j, n in enumerate(rep):
                if j > i:
                    sim, _ = pearson_cal(m, n)
                    sim_list.append(sim)
    return np.array(sim_list)

#####################################
#Generate the list of
#####################################

def mols_gen(args):
    network_file = pd.read_csv(args[0], header = 0)
    smile_64 = pd.read_csv(args[1], header = 0)

    #drug_ttid and smiles mapping
    dict_smile = {network_file.iloc[i,2]:network_file.iloc[i,4] for i in network_file.index}
    for i in pd.unique(network_file["drug_name"]):
            dict_smile[i] = smile_64[smile_64["ttid"] == i]["canonicalsmiles"].values[0]
    #node list
    nodes = list(dict_smile)
    node_index = {v:i for i,v in enumerate(nodes)}
    node_mols = [Chem.MolFromSmiles(dict_smile[i]) for i in nodes]
    #node features
    return nodes, node_mols


#Calculating the exponential of all elements in the euclidean matrix.
def g_kernel(x,y, sigma = 0.1):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        value = euclidean(x, y)
        g = np.exp(-sigma * value)
    else:
        print("x and y must be numpy.ndarray type.")
    return g

#Calculating the exponential of all elements in the euclidean matrix.
def gaussian_kernel(df = None, sigma = 0.01, dtype = np.float32, to_numpy = True, dframe = True):
    #df is L2 norm matrix.
    if to_numpy:
        matrix = np.exp(-sigma * (df.to_numpy(dtype = dtype)))
    else:
        matrix = np.exp(-sigma * df.astype(dtype = dtype))
        
    if dframe:
        return pd.DataFrame(matrix, index = df.index, columns = df.columns, dtype = dtype)
    else:
        return matrix

#################################
#Calculating S-Kernel
#################################

def drug_sim(com, feature_dict):
    # read dataframe
    if isinstance(com, str):
        com = pd.read_csv(com, header = 0, index_col = 0)
    elif isinstance(com, pd.DataFrame):
        com = com
    # get only unique combinations
    com.drop_duplicates(["ANCHOR_NAME","LIBRARY_NAME"], inplace = True, ignore_index = True)
    # generate indexs for similarity matrix
    index = [com["ANCHOR_NAME"][i] + "+" + com["LIBRARY_NAME"][i] for i in com.index]
    #create empty matrix
    sim_matrix = pd.DataFrame(index = index, columns = index)
    #Generating drug-descriptors dictionary for fast fetching vectors and calculating.
    
    num = len(index)
    #Filling half of symmetric matrix element by element.
    for i in range(num): 
        anchor_x = com["ANCHOR_NAME"][i]
        library_x = com["LIBRARY_NAME"][i]
        for s in range(i, num):
            #calculating simlarity
            anchor_y = com["ANCHOR_NAME"][s]
            library_y = com["LIBRARY_NAME"][s]

            sim1 = g_kernel(feature_dict[anchor_x], feature_dict[anchor_y], sigma = 0.1)
            sim2 = g_kernel(feature_dict[library_x], feature_dict[library_y], sigma = 0.1)
            sim3 = g_kernel(feature_dict[anchor_x], feature_dict[library_y], sigma = 0.1)
            sim4 = g_kernel(feature_dict[library_x], feature_dict[anchor_y], sigma = 0.1)
            s1 = 0.5 * (sim1 + sim2)
            s2 = 0.5 * (sim3 + sim4)
            sim = max(s1, s2)
            sim_matrix.iat[i,s] = sim
    #Completing the  whole symmetric matrix.
    
    for i in range(num):
        for s in range(i):
            sim_matrix.iat[i, s] = sim_matrix.iat[s, i]
    return sim_matrix

##############################
#Calculating cells similarity.
##############################
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

#Concat the vector of drug pair and cell line.
def concat_vec(df, drug_sim, cell_sim):
    array = np.empty((df.shape[0], drug_sim.shape[0] + cell_sim.shape[0]), dtype = np.float32)
    for i in df.index:
        dname = df['ANCHOR_NAME'][i] + "+" + df['LIBRARY_NAME'][i]
        cname = df["SIDM"][i]
        
        array[i] = np.hstack((drug_sim[dname].values, cell_sim[cname].values))
    return array

## Metrics for binary classification
def metric_auc(pred, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label = 1)
    auc = metrics.auc(fpr, tpr)
    return auc
    