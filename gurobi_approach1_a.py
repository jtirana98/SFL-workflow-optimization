import numpy as np
import math

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum


import utils
import time

import warnings
warnings.filterwarnings("ignore")

def main():
    K = 20 # number of data owners
    H = 2 # number of compute nodes
    utils.file_name = 'fully_symmetric.xlsx'

    release_date = np.array(utils.get_fwd_release_delays(K,H))
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local = np.array(utils.get_fwd_end_local(K))
    trans_back = np.array(utils.get_trans_back(K, H))

    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m1 = gp.Model("relax_approach_1_p1")
    m2 = gp.Model("relax_approach_1_p2")


    # define variables - problem 1
    
    y = m1.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    f = m1.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    w = m1.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m1.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")

    # define variables - problem 2
    x = m2.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")

    # dual variables

    lala = np.zeros((K,H)) # lamda variable
    mama = np.zeros((H,T,K)) # m variable
    #mama = np.random.normal(0,4, size=(H,T,K))

    # Define constraints for problem 1
    print(f"max-f: {T - np.min(trans_back[0,:]) - np.min(proc_local)} min-f: {np.min(release_date) + np.min(proc[0,:])}")
    print(f"min-w: {np.min(release_date) + np.min(proc[0,:]) + np.min(trans_back[0,:]) + np.min(proc_local)}")

    m1.addConstr(f <= T - np.min(trans_back[0,:]) - np.min(proc_local))
    m1.addConstr(f >=  np.min(release_date) + np.min(proc[0,:]))
    m1.addConstr(w <= T)
    m1.addConstr(w >= np.min(release_date) + np.min(proc[0,:]) + np.min(trans_back[0,:]) + np.min(proc_local))
    
    # C3: each job is assigned to one and only machine
    m1.addConstr( y @ ones_H == ones_K )

    # C4: memory constraint
    m1.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))

    # completition time definition
    m1.addConstrs(comp[i] == qsum(trans_back[i,:] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
    max_constr = m1.addConstr(w == gp.max_(comp[i] for i in range(K)))

    # Define constraints for problem 2

    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    m2.addConstr(x[i,j,t] == 0)
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m2.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    
    # Iterative algorithm
    step = 0
    alpha = 1
    bhta = 1
    while step<100:
       
        m1.setObjective(w + qsum(lala[i,j] * y[i,j] * proc[i,j] for i in range(K) for j in range(H)) - qsum(qsum(mama[j]@f) for j in range(H)), GRB.MINIMIZE)    
        
        m2.setObjective(qsum(x[i,j,t]*(mama[i,t,j]*(t+1) - lala[j,i]) for i in range(H) for j in range(K) for t in range(T)), GRB.MINIMIZE)

    
        print(f"{utils.bcolors.OKBLUE}-------------{step}------------{utils.bcolors.ENDC}")
        # solve P1:
        start = time.time()
        m1.optimize()
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P1 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj1: {m1.ObjVal}{utils.bcolors.ENDC}')

        print("-------------------------------")
        start = time.time()
        m2.optimize()
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P2 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj2: {m2.ObjVal}{utils.bcolors.ENDC}')

        # update dual variables
        for i in range(H):
            for j in range(K):
                lala[j,i] = max(lala[j,i] + alpha*(np.rint(y[j,i].X)*proc[j,i] - sum([np.rint(x[i,j,k].X) for k in range(T)])), 0)
                
                for t in range(T):
                    mama[i,t,j] = max(mama[i,t,j] + bhta*(np.rint(x[i,j,t].X)*(t+1) - np.rint(f[j].X)), 0)
        step = step + 1
        alpha = 1/math.sqrt(step+1)
        bhta = 1/math.sqrt(step+1)

        print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE: {w[0].X}{utils.bcolors.ENDC}')

    #for v in m2.getVars():
    #    print('%s %g' % (v.VarName, v.X))
      
if __name__ == '__main__':
    main()