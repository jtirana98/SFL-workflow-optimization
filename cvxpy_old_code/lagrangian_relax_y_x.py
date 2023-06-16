import numpy as np
import cvxpy as cp
import time

import utils

import warnings
warnings.filterwarnings("ignore")


def main():

    MAX_ITER = 20

    # Function that gives the step
    # it is a function of the number of iterations (u):

    def step_func(u):
        return 1/np.sqrt(u+2)

    K = 5 # number of data owners
    H = 2 # number of compute nodes
    utils.file_name = 'fully_symmetric.xlsx'

    # fully_symmetric
    # fully_heterogeneous
    # symmetric_machines
    # symmetric_data_owners

    release_date = np.array(utils.get_fwd_release_delays(K,H))
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local = np.array(utils.get_fwd_end_local(K))
    trans_back = np.array(utils.get_trans_back(K, H))
    f = cp.Parameter((K))
    mu = cp.Parameter((K,H), nonneg=True) # Dual Variables
    subgradient = cp.Parameter((K,H))


    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    print(f" Memory: {memory_capacity}")

    # Define variables
    x = {}
    for i in range(H):
        x[i] = cp.Variable((K,T), boolean=True)

    y = cp.Variable((K,H), boolean=True) # auxiliary variable

    # Define constraints
    constraints1 = []
    constraints2 = []

    # P1: all jobs interval are assigned to one only machine
    constraints1 += [y @ ones_H == ones_K]

    # P1: memory constraint
    constraints1 += [ (y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape) ]

    #P2: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    constraints2 += [x[i][j,t] == 0]

    # P2:  machine processes only a single job at each interval
    for j in range(H): #for all devices
        constraints2 += [x[j].T @ ones_K <= ones_T]

    # P2: processing time of each job
    for i in range(K): #for all jobs
        subsum = []
        for j in range(H):
            subsum += [cp.sum(x[j][i, :])/proc[i, j]]
        totalsum = cp.sum(subsum)
        constraints2 += [totalsum == 1]

    # P2: Define finish times
    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*x[j][i,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f = cp.hstack(f_values)

    # Define the two problems (primal)

    # P1 first (Y)
    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back[i,:] * y[i,:]))


    obj1 = cp.Minimize(cp.max(cp.hstack(trans)) - cp.multiply(T, cp.sum(cp.sum(cp.multiply(mu,y)))))

    # wrap the formula to a Problem
    prob1 = cp.Problem(obj1, constraints1)

    
    # P2 next (x,f)

    musum = 0
    for j in range(H):
        for i in range(K):
            musum += cp.multiply(cp.sum(x[j][i,:]),mu[i,j])

    obj2 = cp.Minimize(cp.max( f + proc_local ) +musum)


    opt_iter=[] # list that gives the optimal objective f. value at each iteration

    for iter in range(MAX_ITER):
        if iter == 0:
            mu.value = np.ones((K,H))
            subgradient.value = np.ones((K,H))
        
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(mu.value)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        prob1.solve(solver=cp.GUROBI, verbose=True)

        print("status:", prob1.status)
        print("optimal value", prob1.value)


        prob2 = cp.Problem(obj2, constraints2)
        prob2.solve(solver=cp.GUROBI, verbose=True)

        print("status:", prob2.status)
        print("optimal value", prob2.value)

        trans2 = []
        lala = []
        for i in range(K): # for each job/data owner
            # trans2.append(sum(trans_back[i,:] * y.value[i,:]))
            lala.append(f.value[i] + proc_local[i] + sum(trans_back[i,:] * y.value[i,:]))


        initial_prob = max(lala)
        opt_iter.append(initial_prob)
        print("-------------------------------------------", initial_prob, iter)
        step = step_func(iter)
        print("step size:", step)

        for j in range(H):
            for i in range(K):
                # print(sum(x[j][i,:].value))
                subgradient.value[i,j] = sum(x[j][i,:].value) - T*y.value[i,j]
                mu.value[i,j] = max(0, mu.value[i,j] + step*subgradient.value[i,j])

        print("--------Machine allocation--------")

        for i in range(H):
            for k in range(T):
                at_least = 0
                for j in range(K):
                    if(np.rint(x[i][j,k].value) <= 0):
                        continue
                    else:
                        print(f'{j+1}', end='\t')
                        at_least = 1
                        break
                if(at_least == 0):
                    print(f'0', end='\t')
            print('')


    print("optimal value per iter.", opt_iter)

if __name__ == '__main__':
    main()
