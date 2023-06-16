import numpy as np
import cvxpy as cp
import time

import utils
import gurobipy
import warnings
warnings.filterwarnings("ignore")
from numpy import linalg as LA

import random


def main():

    MAX_ITER = 30
    rho = 66

    K = 30 # number of data owners
    H = 5 # number of compute nodes
    # utils.file_name = 'fully_heterogeneous.xlsx'
    np.random.seed(4)
    # for seed 4, (50,5), the optimal value is 41
    # for seed 4, (30,5), the optimal value is 27
    # release_date = np.array(utils.get_fwd_release_delays(K,H))
    release_date = np.random.randint(3, size=(K,H))
    # release_date = (2*np.ones((K,H))).astype(int)
    #print(release_date)
    #memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    memory_capacity = np.array([200, 21, 100, 100, 100])
    # proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    memory_demand = 3
    # proc = np.array([[9, 10, 11], [16, 10, 9], [6, 8, 6], [8, 9, 11], [12, 9, 6], [8, 6, 11]])
    proc = np.random.randint(3,10, size=(K,H))
    proc[:, 1] = 2
    # proc = (5*np.ones((K,H))).astype(int)
    # proc_local = np.array(utils.get_fwd_end_local(K))
    proc_local = np.random.randint(2,8, size=(K))
    # trans_back = np.array(utils.get_trans_back(K, H))
    # trans_back = np.array([[7, 6, 1], [2, 6, 10], [5,5,5], [2, 2, 6], [9,8, 7], [6,1,4]])
    trans_back = np.random.randint(3, size=(K,H))


    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1)).astype(int)
    ones_K = np.ones((K,1)).astype(int)
    ones_T = np.ones((T,1)).astype(int)

    print(f" Memory: {memory_capacity}")

    # Define variables and parameters (for the fixed variables)

    f = cp.Parameter((K))


    x = {}
    x_par = {}
    x_cont = {}
    x_cont_par = {}
    for i in range(H):
        x[i] = cp.Variable((K,T), boolean=True)
        x_par[i] = cp.Parameter((K,T))
        x_cont[i] = cp.Variable((K,T), nonneg=True)  # the continuous version
        x_cont_par[i] = cp.Parameter((K,T))

    y = cp.Variable((K,H), boolean=True)
    y_par = cp.Parameter((K,H), boolean=True)
    y_cont = cp.Variable((K,H), nonneg=True)  # the continuous version
    y_cont_par = cp.Parameter((K,H))


    # Dual variables (as parameters)
    mu = cp.Parameter((K,H))  # Dual Variables related to y
    lam = {}
    for i in range(H):
        lam[i] = cp.Parameter((K,T))  # Dual Variables related to x (lambda)

    nu = cp.Parameter((K))  # dual variables related to f


    # Initialize rho and parameters for 1st iteration!
     # or ...?  (rho is the penalty parameter, needs to be chosen very carefully)

    for i in range(H):
        x_par[i].value = np.zeros((K, T)).astype(int)
        lam[i].value = np.zeros((K,T)).astype(int)
        # x_par[i].value = np.random.randint(2, size=(K, T))


    y_par.value = np.zeros((K, H)).astype(int)


    mu.value = np.zeros((K, H)).astype(int) # or ones???
    nu.value = np.zeros((K)).astype(int)
    # mu.value = np.random.randint(2, size=(K, H))

    # Define constraints
    constraints_cont = []  # for the continuous subproblem
    constraints_int = []  # for the integer subproblem

    # Range for the continuous variables: [0,1]
    constraints_cont += [y_cont <= 1]
    for j in range(H):
        constraints_cont += [x_cont[j] <= 1]

    #  job cannot be assigned to a time interval before the release time
    for i in range(H):  # for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    constraints_int += [ x[i][j,t] == 0 ]
                    constraints_cont += [ x_cont[i][j,t] == 0 ]


    #  all jobs interval are assigned to one only machine
    constraints_int += [ y @ ones_H == ones_K ]
    constraints_cont += [ y_cont @ ones_H == ones_K ]


    #  memory constraint
    constraints_int += [ (y.T * memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape) ]
    constraints_cont += [ (y_cont.T * memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape) ]

    #  machine processes only a single job at each interval
    for j in range(H): #for all devices
        constraints_int += [ x[j].T @ ones_K <= ones_T ]
        constraints_cont += [ x_cont[j].T @ ones_K <= ones_T ]



    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H):  #for all machines
        constraints_int += [cp.sum(x[j],axis=1) == cp.multiply(y,proc).T[j,:]]
        constraints_cont += [cp.sum(x_cont[j],axis=1) == cp.multiply(y_cont,proc).T[j,:]]

    # x-subpr: processing time of each job

    #C8: the completition time for each data owner -- NOTE: Removed it from constraints
    # constraints_int += [f_int >=0]

    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*x_cont[j][i,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f = cp.hstack(f_values)
    """
    for j in range(H):
        for i in range(K):
            for t in range(T):
                constraints_cont += [f[i] >= (t+1)*x_cont[j][i,t]]

    for j in range(H):
        for i in range(K):
            for t in range(T):
                constraints_int += [f_int[i] >= (t+1)*x[j][i,t]]
    """
    env = gurobipy.Env()
    env.setParam('MIPGap', 0.10) # in seconds



    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back[i,:] * y_cont[i,:]))

    rhosum = []
    for j in range(H):
        rhosum.append(cp.sum(cp.abs(x_par[j] - x_cont[j] + lam[j])))


    obj_cont = cp.Minimize(cp.max(f + proc_local + cp.hstack(trans))
                + (rho/2)*cp.sum(rhosum)
                + (rho/2)*cp.sum(cp.abs(y_cont - y_par + mu)))

    prob_cont = cp.Problem(obj_cont, constraints_cont)

    rhosum_int = []
    for j in range(H):
        rhosum_int.append(cp.sum(cp.abs(x_cont_par[j] - x[j] + lam[j])))

    obj_int = cp.Minimize((rho/2)*cp.sum(rhosum_int)
            + (rho/2)*cp.sum(cp.abs(y_cont_par - y + mu)))

    prob_int = cp.Problem(obj_int, constraints_int)


    # Now let's start the iterations
    opt_iter=[] # list that gives the optimal objective f. value at each iteration

    for iter in range(MAX_ITER):
        if iter < 4:
            prob_cont.solve(solver=cp.GUROBI, verbose=False, env=env)
        else:
            prob_cont.solve(solver=cp.GUROBI, verbose=False)
        print("optimal value of continuous:", prob_cont.value, prob_cont.status, "at iteration:", iter)


        y_cont_par.value = np.copy(y_cont.value)
        for j in range(H):
            x_cont_par[j].value = np.copy(x_cont[j].value)
        #print(y.value)


        prob_int.solve(solver=cp.GUROBI, verbose=False, env=env)

        print("optimal value of integer:", prob_int.value, "at iteration:", iter)
        # print("And the time for the x-subproblem is...", prob_x.solver_stats.solve_time, "sec.")
        # print("&&&&&&&&&&&&&&&&&")

        for j in range(H):
            x_par[j].value = np.copy(x[j].value)
            # print(j, x[j].value)
        y_par = np.copy(y.value)
        # f_int_par = np.copy(f_int.value)
        # Update dual variabes:

        temp_mu = np.copy(mu.value + np.abs(y_cont.value - y.value))

        mu.value = temp_mu

        temp_lam = {}

        for j in range(H):
            temp_lam[j] = np.copy(lam[j].value + np.abs(x_cont[j].value - x[j].value))
            lam[j].value = temp_lam[j]

        #temp_nu = np.copy(nu.value + np.abs(f.value - f_int.value))

        #nu = temp_nu

        # print(mu.value)

        # Now calculate the value of (original) objective function at this iter
        g_values = []
        for i in range(K): #for all jobs
            g_interm = []
            for j in range(H): #for all machines
                for t in range(T): #for all timeslots
                    g_interm += [(t+1)*np.abs(x[j].value[i,t])]
            g_values += [max(g_interm)]

        trans2 = []
        lala = []
        for i in range(K): # for each job/data owner
            # trans2.append(sum(trans_back[i,:] * y.value[i,:]))
            lala.append(g_values[i] + proc_local[i] + sum(trans_back[i,:] * y.value[i,:]))
        print("#### finish times:", f.value)
        print(g_values)
        # f.values = g_values
        initial_prob = max(lala)
        opt_iter.append(initial_prob)
        print("----------------------------------------------------------------", initial_prob, iter)


        # print(mu.value)



    #for j in range(H):
        #print(j, x[j].value)

if __name__ == '__main__':
    main()
