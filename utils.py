import numpy as np
import pandas as pd

max_memory_demand = 3
file_name = 'test1.xlsx'
'''
return np an numpy array of shape (K, H) with the release dates
for the forward pass (first batch).
Use profiling values
'''
def get_fwd_release_delays(K,H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_release_delays', header=None)
    return df.values.tolist()
    #return [[3,1],[3,4],[2,2]]


'''
return np an numpy array of shape (K, H) with the processing time
for the forward pass on the compute nodes.
Use profiling values
'''
def get_fwd_proc_compute_node(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_proc_compute_node', header=None)
    machines = df.values.tolist()
    #machines = [2,4]
    total = []
    for i in range(K):
        total += [machines[0]]
    return total

'''
return np an numpy array of shape (H, 1) with the processing time
for the forward pass on the data owner for the last layers.
Use profiling values
'''
def get_fwd_end_local(K):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_end_local', header=None)
    return df.values.tolist()[0]
    #return [1,1,4]


'''
return np an numpy array of shape (H, H) with the transmission time
from the compute nodes to the data owner sending the activations from the second cut layer.
Use profiling values
'''
def get_trans_back(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_trans_back', header=None)
    return df.values.tolist()
    #return [[1,1],[1,1],[1,1]]


'''
return np an numpy array of shape(H,1) with the memory capacity of each 
helper node
'''
def get_memory_characteristics(H):
    df = pd.read_excel(io=file_name, sheet_name='get_memory_characteristics', header=None)
    return df.values.tolist()[0]
    #return [10,10]
