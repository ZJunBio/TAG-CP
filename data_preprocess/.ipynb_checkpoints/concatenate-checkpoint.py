import pandas as pd
import numpy as np

#Concat the vector of drug pair and cell line.
def concat_vec(df, drug_sim, cell_sim):
    array = np.empty((df.shape[0], drug_sim.shape[0] + cell_sim.shape[0]), dtype = np.float32)
    for i in df.index:
        dname = str(df['drug_row'][i]) + "+" + str(df['drug_col'][i])
        cname = df["cell_line_name"][i]
        array[i] = np.hstack((drug_sim[dname].values, cell_sim[cname].values))
    return array