import pandas as pd
import numpy as np
import sys
sys.path.append("../NN/")
from utils import g_kernel


#calculate similarities betwteen 2 combinations.
def drug_sim(com, feature_dict, sig = 0.15):
    '''
    com: dataframe of combinations containing 2 columns.
    '''
    # generate indexs for similarity matrix
    index = [str(com["drug_row"][i]) + "+" + str(com["drug_col"][i]) for i in com.index]
    #create empty matrix
    sim_matrix = pd.DataFrame(index = index, columns = index)
    #Generating drug-descriptors dictionary for fast fetching vectors and calculating.
    
    num = len(index)
    #Filling half of symmetric matrix element by element.
    for i in range(num): 
        anchor_x = com["drug_row"][i]
        library_x = com["drug_col"][i]
        for s in range(i, num):
            #calculating simlarity
            anchor_y = com["drug_row"][s]
            library_y = com["drug_col"][s]

            sim1 = g_kernel(feature_dict[anchor_x], feature_dict[anchor_y], sigma = sig)
            sim2 = g_kernel(feature_dict[library_x], feature_dict[library_y], sigma = sig)
            sim3 = g_kernel(feature_dict[anchor_x], feature_dict[library_y], sigma = sig)
            sim4 = g_kernel(feature_dict[library_x], feature_dict[anchor_y], sigma = sig)
            s1 = 0.5 * (sim1 + sim2)
            s2 = 0.5 * (sim3 + sim4)
            sim = max(s1, s2)
            sim_matrix.iat[i,s] = sim
    #Completing the  whole symmetric matrix.
    
    for i in range(num):
        for s in range(i):
            sim_matrix.iat[i, s] = sim_matrix.iat[s, i]
    return sim_matrix