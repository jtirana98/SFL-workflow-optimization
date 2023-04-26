import numpy as np
import time

import utils

import warnings
warnings.filterwarnings("ignore")

def check_memory(capacity, load):
    return ((load*utils.max_memory_demand) <= capacity)

def check(i, mylist):
    for v in mylist:
        if v == i:
            return True
    return False

def check_balance(distributions_):
    larger = -1
    smaller = -1

    for i in range(len(distributions_)):
        if i == 0:
            smaller = distributions_[i]
            larger = distributions_[i]
        else:
            if smaller > distributions_[i]:
                smaller = distributions_[i]
            
            if larger < distributions_[i]:
                larger = distributions_[i]
    return (larger - smaller) > 2

def main():

    K = 5 # number of data owners
    H = 2 # number of compute nodes
    utils.file_name = 'test1.xlsx'

    memory_capacity = np.array(utils.get_memory_characteristics(H, K))

    # forward-propagation parameters
    release_date_fwd = np.array(utils.get_fwd_release_delays(K,H))
    proc_fwd = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local_fwd = np.array(utils.get_fwd_end_local(K))
    trans_back_activations = np.array(utils.get_trans_back(K, H))

    # back-propagation parameters
    release_date_back = np.array(utils.get_bwd_release_delays(K,H))
    proc_bck = np.array(utils.get_bwd_proc_compute_node(K,H))
    proc_local_back = np.array(utils.get_bwd_end_local(K))
    trans_back_gradients = np.array(utils.get_grad_trans_back(K,H))


    f_temp = []
    f_temp_faster_machine = []
    distribution = [0 for i in range(H)]
    for i in range(K):
        exclude = []
        find = True
        flag = True
        while find:
            my_machine = -1
            smallest = -1
            for j in range(H):
                if my_machine == -1 and (check(j, exclude) == False):
                    my_machine = j
                    smallest = release_date_fwd[i][j] + proc_fwd[i][j] + trans_back_activations[i][j]
                    if flag:
                        f_temp_faster_machine.append((i,smallest, j))
                        flag = False
                    else:
                        f_temp_faster_machine[-1] = (i,smallest, j)
                else:
                    if (smallest > release_date_fwd[i][j]) and (check(j, exclude) == False):
                        my_machine = j
                        smallest = release_date_fwd[i][j]
                        f_temp_faster_machine[-1] = (i,smallest, j)
            
            # check memory
            index_m = f_temp_faster_machine[-1][2]
            distribution[index_m] += 1
            if ((check_memory(memory_capacity[index_m], distribution[index_m]) == False )or (check_balance(distribution) and len(exclude) - 1 < H)): #search for another device
                distribution[index_m] =  distribution[index_m] - 1
                exclude.append(index_m)
            else:
                f_temp.append(smallest)
                find = False

    print(distribution)
    for j in range(H):
        my_devices = []
        for tok in f_temp_faster_machine:
            if tok[2] == j:
                my_devices.append((tok[0], tok[1]))

        order = 1
        while len(my_devices) > 0:
            smallest = -1
            for k in range(len(my_devices)):
                if k == 0:
                    smallest = k
                else:
                    if my_devices[smallest][1] > my_devices[k][1]:
                        smallest = k
            index = my_devices[smallest][0]
            f_temp[index] = f_temp[index] +(order*proc_fwd[index][j]) + proc_local_fwd[index] + release_date_back[index][j]
            order += 1
            my_devices.pop(smallest)

    for i in range(K):
        print(f_temp[i])

    print("-------------")
    for j in range(H):
        my_devices = []
        for tok in f_temp_faster_machine:
            if tok[2] == j:
                my_devices.append((tok[0], tok[1]))

        order = 1
        while len(my_devices) > 0:
            smallest = -1
            for k in range(len(my_devices)):
                index_s = my_devices[smallest][0]
                index_k = my_devices[smallest][0]
                if k == 0:
                    smallest = k
                else:
                    if f_temp[index_s] > f_temp[index_k]:
                        smallest = k
            index = my_devices[smallest][0]
            f_temp[index] = f_temp[index] +(order*proc_bck[index][j]) + proc_local_back[index] + trans_back_gradients[index][j]
            order += 1
            my_devices.pop(smallest)
        
    for i in range(K):
        print(f_temp[i])

if __name__ == '__main__':
    main()