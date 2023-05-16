import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

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
file_name = 'test4.xlsx'
'''
return np an numpy array of shape (K, H) with the release dates
for the forward pass (first batch).
Use profiling values
'''

def mysum(mylist):
    sum = 0

    for s in mylist:
        sum += s

    return sum

def get_fwd_release_delays(K,H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_release_delays', header=None)

    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list


def get_bwd_release_delays(K,H):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_release_delays', header=None)

    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list


'''
return np an numpy array of shape (K, H) with the processing time
for the forward pass on the compute nodes.
Use profiling values
'''
def get_fwd_proc_compute_node(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_proc_compute_node', header=None)
    temp = df.values.tolist()

    machines = []
    for i in  range(H):#range(len(temp)):
        machines.append(temp[i][0])

    total = []
    for i in range(K):
        total += [machines]
    return total

def get_bwd_proc_compute_node(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_proc_compute_node', header=None)
    temp = df.values.tolist()

    machines = []
    for i in  range(H):#range(len(temp)):
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
    for i in  range(K):#range(len(temp)):
        df_list.append(temp[i][0])

    return df_list

def get_bwd_end_local(K):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_end_local', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in range(K):#range(len(temp)):
        df_list.append(temp[i][0])

    return df_list


'''
return np an numpy array of shape (H, H) with the transmission time
from the compute nodes to the data owner sending the activations from the second cut layer.
Use profiling values
'''
def get_trans_back(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_trans_back', header=None)
    
    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list

def get_grad_trans_back(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_grad_trans_back', header=None)

    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list

'''
return np an numpy array of shape(H,1) with the memory capacity of each 
helper node
'''
def get_memory_characteristics(H, K=10):
    '''
    df = pd.read_excel(io=file_name, sheet_name='get_memory_characteristics', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in  range(len(temp)):
        df_list.append(temp[i][0])

    return df_list
    #return [10,10]
    '''
    df_list = []
    
    total_demand = max_memory_demand * K
    if file_name == 'fully_symmetric.xlsx' or file_name == 'symmetric_machines.xlsx': # all the same
        minimum_avail = int(K/H)
        if K % H == 0:
            df_list = [minimum_avail*max_memory_demand for i in range(H)]
        else:
            df_list = [minimum_avail*max_memory_demand for i in range(H-1)]
            df_list.append((minimum_avail+1)*max_memory_demand)
    else: # choose from a distribution but in the end all data owners should be served
        end = int(K/H)
        first_round = False
        while True:
            for i in range(H):
                data_owners = random.randint(1,end)
                if first_round:
                    df_list[i] = df_list[i] + data_owners
                    if mysum(df_list) >= K:
                        for i in range(H):
                            df_list[i] = df_list[i] * max_memory_demand
                        return df_list
                else:
                    df_list.append(data_owners)
            
            first_round = True
            if mysum(df_list) < K:
                end = K - mysum(df_list)
            else:  
                break
        
        for i in range(H):
            df_list[i] = df_list[i] * max_memory_demand
    return df_list


def plot_approach(w_start, w_approach, constraints_1, constraints_2=[], maxC=[], violation=[]):
    fig,ax = plt.subplots()
    ax2=ax.twinx()

    x_ticks = [i+1 for i in range(len(w_approach))]
    print(x_ticks)
    ax.plot(x_ticks, [w_start for i in range(len(x_ticks))], linewidth = 2, marker='o', markersize=12, color="green", label = "Optimal value")
    ax.plot(x_ticks, w_approach, linestyle='dashed', linewidth = 2, marker='o', markersize=12, color="orange", label = "W-approx.")
    
    if len(maxC) != 0:
        ax.plot(x_ticks, maxC, linestyle='dashed', color="black", linewidth = 2, marker='o', markersize=12, label = "max - C")
        ax2.plot(x_ticks, violation, linestyle = 'None', color="red", marker='+', markersize=12, label = "Violation (T/F)")


    ax2.plot(x_ticks, constraints_1, linestyle = 'None', color="magenta", marker='*', markersize=12, label = "constraint-1-violation (%)")
    if len(constraints_2) != 0:
        ax2.plot(x_ticks, constraints_2, linestyle = 'None', color="magenta", marker='o', markersize=12, label = "constraint-2-violation (%)")

    
    ax2.set_ylabel("Violations")
    plt.xlim(0.9,len(w_approach))
    #ax.legend(loc=0)
    #ax2.legend(loc=2)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.show()