import numpy as np
import cvxpy as cp
import time

import utils

import warnings
warnings.filterwarnings("ignore")



def main():

# Problem input

    K = 20 # number of data owners
    H = 5 # number of compute nodes
    utils.file_name = 'fully_symmetric.xlsx'

    # fully_symmetric
    # fully_heterogeneous
    # symmetric_machines
    # symmetric_data_owners

    release_date = cp.Parameter((K,H), value=np.array(utils.get_fwd_release_delays(K, H)))
    # memory_capacity = cp.Parameter(H, value=np.array(utils.get_memory_characteristics(H, K)))
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    proc = cp.Parameter((K,H), value=np.array(utils.get_fwd_proc_compute_node(K, H)))
    proc_local = cp.Parameter(K, value=np.array(utils.get_fwd_end_local(K)))
    trans_back = cp.Parameter((K,H), value=np.array(utils.get_trans_back(K, H)))



    T = np.max(release_date.value) + K*np.max(proc[0,:].value) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    print(f" Memory: {memory_capacity}")



    y = cp.Variable((K,H), boolean=True) # auxiliary variable
    f = cp.Variable(K, integer=True) # finish time
    w = cp.Variable(integer=True) # completion time

    # Dual variables (parameters)

    lala = cp.Parameter((K,H), value=np.ones((K,H)))
    mama = {}
    for i in range(H):
        #mama[i] = cp.Parameter((T,K), value=np.ones((T,K)))
        mama[i] = cp.Parameter((T,K), value=np.random.randint(0,5, size=(T, K)))
    #mama[0]=np.ones((T,K))*20



    # Define constraints
    constraints1 = []
    # constraints o to restrict the values of f and  w
    constraints1 += [f <= 26]
    constraints1 += [f >= 0]
    constraints1 += [w <= 29]
    constraints1 += [w >= 0]


    # C3: each job is assigned to one and only machine
    constraints1 += [y @ ones_H == ones_K]

    # C4: memory constraint
    constraints1 += [(y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape)]

    # w and completion time

    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back[i,:] * y[i,:]))

    for i in range(K): # for each job/data owner
        constraints1 += [w >= f[i] + proc_local[i] + trans[i]]

    print(mama[1].shape)
    term_mu_f = 0
    for j in range(H):
        term_mu_f += cp.sum(mama[j]@f)
        # term_mu_f += cp.sum(mama[j])


    obj1 = cp.Minimize(w + cp.sum(cp.multiply(cp.multiply(lala, y), proc.value)) - term_mu_f)
    ### ATTENTION, make sure we don't need two cp.sum here!!!
    # wrap the formula to a Problem
    prob1 = cp.Problem(obj1, constraints1)
    start = time.time()
    prob1.solve(solver=cp.GUROBI, verbose=True)
    end = time.time()
    print("TIME: ", end - start)

    print("status:", prob1.status)
    print("optimal value", prob1.value)
    print(f.value)

if __name__ == '__main__':
    main()
