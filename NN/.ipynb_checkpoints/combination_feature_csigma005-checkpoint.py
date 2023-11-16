#calculate drug combination-cell line feature
import pandas as pd
import numpy as np
from utils import concat_vec

train = pd.read_csv("../data/datasets/train.csv", header = 0)
val = pd.read_csv("../data/datasets/validation.csv", header = 0)
test = pd.read_csv("../data/datasets/test.csv", header = 0)

dc_similarity = pd.read_csv("../data/one_graph/skernel_716dcsimlarity.csv", header = 0, index_col=0)
ce_similarity = pd.read_csv("../data/one_graph/gkernel_125cellsimlarity_sigmal005.csv", header=0, index_col=0)

train_vec = concat_vec(train, dc_similarity, ce_similarity)
val_vec = concat_vec(val, dc_similarity, ce_similarity)
test_vec = concat_vec(test, dc_similarity, ce_similarity)

print(f"Train tensor has shape {train_vec.shape} \nValidation tensor has shape {val_vec.shape} \nTest tensor has shape {test_vec.shape}")

np.save("../data/datasets/train_tensor_csigma005.npy", train_vec, allow_pickle = True)
np.save("../data/datasets/val_tensor_csigma005.npy", val_vec, allow_pickle = True)
np.save("../data/datasets/test_tensor_csigma005.npy", test_vec, allow_pickle = True)

train_label = train["Synergy"].astype('int32').to_numpy()
val_label = val["Synergy"].astype('int32').to_numpy()
test_label = test["Synergy"].astype('int32').to_numpy()

np.save("../data/datasets/train_label_csima005.npy", train_label, allow_pickle = True)
np.save("../data/datasets/val_label_csima005.npy", val_label, allow_pickle = True)
np.save("../data/datasets/test_label_csima005.npy", test_label, allow_pickle = True)
