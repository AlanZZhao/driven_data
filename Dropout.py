import pandas as pd
import numpy as np

# Dropout function: if number of non-null columns is below threshold, return
# 'True' for each of these columns, and return as a vector (that can be used to
# subset the training and testing dfs)

def Dropcols(df, threshold = 0.02):
    TFvec = np.zeros(len(df.columns), dtype = bool)
    for i in range(len(df.columns)):
        percent_null = sum(df.iloc[:,i].isnull())/len(df.iloc[:,i])
        if 1- percent_null > threshold:
            TFvec[i] = False
        else:
            TFvec[i] = True
    return(TFvec)


