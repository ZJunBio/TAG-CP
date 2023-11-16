import pandas as pd
temp = pd.read_csv("data/cells/rnaseq_tpm_part0.csv", header = 0, index_col = 0)
df = pd.DataFrame(columns = temp.columns)
for i in range(6):
    temp = pd.read_csv("data/cells/rnaseq_tpm_part" + str(i) + ".csv", header = 0, index_col = 0)
    df = pd.concat([df, temp])
df.to_csv("data/cells/rnaseq_tpm_20220624_symbol_.csv", header = True,index = True)
