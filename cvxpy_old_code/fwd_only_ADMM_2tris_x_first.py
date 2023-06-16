import numpy as np
import cvxpy as cp
import time

import utils
import gurobipy
import warnings
warnings.filterwarnings("ignore")

import random
from numpy import linalg as LA


def main():

    MAX_ITER = 4
    rho = 0.6

    K = 10 # number of data owners
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
    memory_capacity = np.array([10, 10, 100, 100, 100])
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

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    print(f" Memory: {memory_capacity}")

    # Define variables and parameters (for the fixed variables)

    f = cp.Parameter(K, integer=True)
    f_par = cp.Parameter(K, integer=True)

    x = {}
    x_par = {}
    for i in range(H):
        x[i] = cp.Variable((K,T), boolean=True)
        x_par[i] = cp.Parameter((K,T))

    y = cp.Variable((K,H), boolean=True)
    y_par = cp.Parameter((K,H), boolean=True)


    # Dual variables (as parameters)
    mu = cp.Parameter((K,H))  # Dual Variables


    # Initialize rho and parameters for 1st iteration!


    #for i in range(H):
        #x_par[i].value = np.zeros((K, T))
        # x_par[i].value = np.random.randint(2, size=(K, T))


    y_par.value = np.zeros((K, H))
    f_par.value = np.zeros((K))

    mu.value = np.zeros((K, H))  # or ones???
    # mu.value = np.random.randint(2, size=(K, H))

    # Define constraints
    constraints_y = []  # for the y-subproblem
    constraints_x = []  # for the x-subproblem


    # x-subpr: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        constraints_x += [x[i] >= 0]
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    constraints_x += [ x[i][j,t] == 0 ]


    # y-subpr: all jobs interval are assigned to one only machine
    constraints_y += [ y @ ones_H == ones_K ]


    # y_subpr: memory constraint
    constraints_y += [ (y.T * memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape) ]

    # x_subpr: machine processes only a single job at each interval
    for j in range(H): #for all devices
        constraints_x += [ x[j].T @ ones_K <= ones_T ]

    #for i in range(K):
        #constraints_x += [f[i] >= min(proc[:, j])]

    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    #  for j in range(H): #for all machines
        #  constraints += [cp.sum(x[j],axis=1) == cp.multiply(y,proc).T[j,:]]

    # x-subpr: processing time of each job

    # C5: job should be processed entirely once

    for i in range(K):
        # print(min(proc[i,:]))
        pprr = []
        for j in range(H):
            pprr += [cp.sum(x[j][i, :])]
        constraints_x += [cp.sum(pprr) >= min(proc[i, :])]

    """

    for i in range(K): #for all jobs
        sub_sum = []
        for j in range(H):
            sub_sum += [cp.sum(x[j][i, :])/proc[i, j]]
        sum_ = cp.sum(cp.hstack(sub_sum))
        constraints_x += [sum_ == 1]


    for i in range(K):
        pprr = []
        for j in range(H):
            pprr += [cp.sum(x[j][i, :])]
            constraints_x += [cp.sum(pprr) >= min(release_date[i, :]) + min(proc[i, :])]


    # C5: job should be processed entirely once
    for i in range(K): #for all jobs
        sub_sum = []
        for j in range(H):
            sub_sum += [cp.sum(x[j][i, :])/ proc[i, j]]
        sum_ = cp.sum(cp.hstack(sub_sum))
        constraints_x += [sum_ == 1]
    """
    #C8: the completition time for each data owner -- NOTE: Removed it from constraints

    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*x[j][i,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f = cp.hstack(f_values)

    # env = gurobipy.Env()
    # env.setParam('MIPGap', 0.10) # in seconds


    # Define Augmented Lagrangian
    # First the augm. lagr. for the y-subproblem (for fixed x)

    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back[i,:] * y[i,:]))

    rhosum = []
    for j in range(H):
        for i in range(K):
            #rhosum.append((cp.sum(x_par[j][i,:]) - y[i,j]*proc[i,j] + mu[i,j])**2)
            rhosum.append(cp.abs(cp.sum(x_par[j][i,:]) - y[i,j]*proc[i,j] + mu[i,j]))


    obj_y = cp.Minimize(cp.max(f_par + proc_local + cp.hstack(trans)) + (rho/2)*cp.sum(rhosum))

    prob_y = cp.Problem(obj_y, constraints_y)

    # Now the x-subproblem
    trans_fixed_y = []
    for i in range(K): # for each job/data owner
        trans_fixed_y.append(cp.sum(trans_back[i,:] * y_par[i, :]))


    rhosum_fixed_y = []
    for j in range(H):
        for i in range(K):
            #rhosum_fixed_y.append((cp.sum(x[j][i,:]) - y_par[i,j]*proc[i,j] + mu[i,j])**2)
            rhosum_fixed_y.append(cp.abs(cp.sum(x[j][i,:]) - y_par[i,j]*proc[i,j] + mu[i,j]))

    obj_x = cp.Minimize(cp.max(f + proc_local + cp.hstack(trans_fixed_y)) + (rho/2)*cp.sum(rhosum_fixed_y))

    prob_x = cp.Problem(obj_x, constraints_x)


    # Now let's start the iterations
    opt_iter=[] # list that gives the optimal objective f. value at each iteration
    # print("!!!!!!!!!!!!!!!!!!!!!", len(constraints_x),len(constraints_y))
    for iter in range(MAX_ITER):
        prob_x.solve(solver=cp.GUROBI, verbose=False)

        print("optimal value of x-subroblem:", prob_x.value, "at iteration:", iter)
        # print("And the time for the x-subproblem is...", prob_x.solver_stats.solve_time, "sec.")
        # print("&&&&&&&&&&&&&&&&&")

        for j in range(H):
            x_par[j].value = np.copy(np.abs(np.rint(x[j].value)))
            # print(j, x[j].value)
            # print(j, x[j].value)

        f_par.value = np.copy(f.value)
        print("finishhhhh", f.value)

        rhosum_fixed_y2 = []
        for j in range(H):
            for i in range(K):
                #rhosum_fixed_y.append((cp.sum(x[j][i,:]) - y_par[i,j]*proc[i,j] + mu[i,j])**2)
                rhosum_fixed_y2.append(np.abs(np.sum(x[j].value[i,:]) - y_par.value[i,j]*proc[i,j] + mu.value[i,j]))
        print((rho/2)*sum(rhosum_fixed_y2))
        prob_y.solve(solver=cp.GUROBI, verbose=False)
        print("optimal value of y-subroblem:", prob_y.value, prob_y.status, "at iteration:", iter)
        aaa = LA.norm((y_par.value-y.value), 'fro')**2  # this helps keep track of the changes in y from iter to iter
        y_par.value = np.copy(y.value)
        print(y.value)
        # print(np.sum(x_par[0].value[:, 0:20]))


        #prob_x.solve(solver=cp.GUROBI, verbose=False, env=env)

        # Update dual variabes:
        temp_mu = np.zeros((K,H))  # just making sure I don't make a mistake
        for j in range(H):
            for i in range(K):
                temp_mu[i,j] = np.copy(mu.value[i,j] + (np.sum(np.abs(x[j].value[i,:]))-(y.value[i,j]*proc[i,j])))

        print(np.sum(mu.value), np.sum(temp_mu))
        mu.value = np.copy(temp_mu)
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
            lala.append(f.value[i] + proc_local[i] + sum(trans_back[i,:] * y.value[i,:]))
        print("#### finish times:", f.value)
        # print(g_values)
        # f.values = g_values
        initial_prob = max(lala)
        opt_iter.append(initial_prob)



        print("----------------------------------------------------------------", initial_prob, aaa,  np.rint(iter))


        # print(mu.value)



    #for j in range(H):
        #print(j, x[j].value)
    print(opt_iter)

if __name__ == '__main__':
    main()
