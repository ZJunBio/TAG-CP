import pandas as pd
from scipy import stats
import numpy as np

sig = pd.read_csv("../data/jaaks_druginfo.csv", header = 0)
pearson = pd.read_csv("../data/pearsonr_similarity_64cpds_sigpath.csv", header = 0, index_col = 0)
sig_dict = {i:sig[sig["target_pathway"] == i]["drug_name"].to_list() for i in pd.unique(sig["target_pathway"])}

for k,v in sig_dict.items():
    print(f"FOR {k} pathway:")
    sample_1 = pearson.loc[sig_dict[k], sig_dict[k]].values.flatten()
    sample_2_index = list(set(pearson.columns) - set(sig_dict[k]))
    sample_2 = pearson.loc[sample_2_index, sample_2_index].values.flatten()
    results = stats.ttest_ind(sample_1,sample_2,equal_var=False,alternative='greater')
    print(f'Sample1: mean is {np.mean(sample_1)} with std {np.std(sample_1)}')
    print(f'Sample2: mean is {np.mean(sample_2)} with std {np.std(sample_2)}')
    print(f'The statistic value is {results. statistic} with pvalue {results.pvalue}')
    
