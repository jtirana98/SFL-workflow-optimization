import numpy as np
import pandas as pd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

max_memory_demand = 3
file_name = 'test3.xlsx'
'''
return np an numpy array of shape (K, H) with the release dates
for the forward pass (first batch).
Use profiling values
'''
def get_fwd_release_delays(K,H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_release_delays', header=None)
    return df.values.tolist()


def get_bwd_release_delays(K,H):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_release_delays', header=None)
    return df.values.tolist()


'''
return np an numpy array of shape (K, H) with the processing time
for the forward pass on the compute nodes.
Use profiling values
'''
def get_fwd_proc_compute_node(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_proc_compute_node', header=None)
    temp = df.values.tolist()

    machines = []
    for i in  range(len(temp)):
        machines.append(temp[i][0])

    total = []
    for i in range(K):
        total += [machines]
    return total

def get_bwd_proc_compute_node(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_proc_compute_node', header=None)
    temp = df.values.tolist()

    machines = []
    for i in  range(len(temp)):
        machines.append(temp[i][0])

    total = []
    for i in range(K):
        total += [machines]
    return total

'''
return np an numpy array of shape (H, 1) with the processing time
for the forward pass on the data owner for the last layers.
Use profiling values
'''
def get_fwd_end_local(K):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_end_local', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in  range(len(temp)):
        df_list.append(temp[i][0])

    return df_list

def get_bwd_end_local(K):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_end_local', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in  range(len(temp)):
        df_list.append(temp[i][0])

    return df_list


'''
return np an numpy array of shape (H, H) with the transmission time
from the compute nodes to the data owner sending the activations from the second cut layer.
Use profiling values
'''
def get_trans_back(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_trans_back', header=None)
    return df.values.tolist()

def get_grad_trans_back(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_grad_trans_back', header=None)
    return df.values.tolist()

'''
return np an numpy array of shape(H,1) with the memory capacity of each 
helper node
'''
def get_memory_characteristics(H):
    df = pd.read_excel(io=file_name, sheet_name='get_memory_characteristics', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in  range(len(temp)):
        df_list.append(temp[i][0])

    return df_list
    #return [10,10]
