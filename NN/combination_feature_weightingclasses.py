#calculate drug combination-cell line feature
import pandas as pd
import numpy as np
from utils import concat_vec

train = pd.read_csv("../data/datasets/train_weightingclass.csv", header = 0)
test = pd.read_csv("../data/datasets/test_weightingclass.csv", header = 0)

dc_similarity = pd.read_csv("../data/one_graph/skernel_716dcsimlarity_2lwayers128.csv", header = 0, index_col=0)
ce_similarity = pd.read_csv("../data/one_graph/gkernel_125cellsimlarity_sigmal005.csv", header=0, index_col=0)

train_vec = concat_vec(train, dc_similarity, ce_similarity)
#val_vec = concat_vec(val, dc_similarity, ce_similarity)
test_vec = concat_vec(test, dc_similarity, ce_similarity)

print(f"Train tensor has shape {train_vec.shape} \nTest tensor has shape {test_vec.shape}")

#np.save("../data/datasets/train_tensor_wc_csigma005.npy", train_vec, allow_pickle = True)
#np.save("../data/datasets/test_tensor_wc_csigma005.npy", test_vec, allow_pickle = True)

np.save("../data/datasets/train_tensor_wc_2layer128.npy", train_vec, allow_pickle = True)
np.save("../data/datasets/test_tensor_wc_2layer128.npy", test_vec, allow_pickle = True)

train_label = train["Synergy"].astype('int32').to_numpy()
test_label = test["Synergy"].astype('int32').to_numpy()

#np.save("../data/datasets/train_label_wc_csima005.npy", train_label, allow_pickle = True)
#np.save("../data/datasets/test_label_wc_csima005.npy", test_label, allow_pickle = True)

np.save("../data/datasets/train_label_wc_2layer128.npy", train_label, allow_pickle = True)
np.save("../data/datasets/test_label_wc_2layer128.npy", test_label, allow_pickle = True)
