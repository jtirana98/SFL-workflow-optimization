import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum

import utils
import time

import warnings
warnings.filterwarnings("ignore")

K = 50 # number of data owners
H = 2 # number of compute nodes
utils.file_name = 'fully_symmetric.xlsx'

def run(release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, memory_capacity, 
        release_date_back, proc_bck, proc_local_back, trans_back_gradients, filename=''):
    
    f_y = open(f'test_{K}_{H}', "w")

    start = time.time()

    T = np.max(release_date_fwd) + K*np.max(proc_fwd[0,:]) + np.max(release_date_back) + K*np.max(proc_bck[0,:]) \
                        + np.max(proc_local_fwd) + np.max(proc_local_back)\
                        + np.max(np.max(trans_back_activations)) + np.max(np.max(trans_back_gradients))
    print(f"T = {T}")
    print(f" Memory: {memory_capacity}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("fwd_back")

    # define variables
    x = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    z = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="z")
    
    # auxilary variables
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")
    
    # define constraints FORWARD

    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date_fwd[j,i]:
                    m.addConstr(x[i,j,t] == 0)

    # C3: all jobs interval are assigned to one only machine
    m.addConstr( y @ ones_H == ones_K )
    
    # C4: memory constraint
    m.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    
    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc_fwd[i,j])
    '''
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*x[j,i,t] for t in range(T))
    '''
    
    # define constraints BACK
    
    # backprop job cannot be assigned to a time interval before the backprops release time

    for i in range(K): #for all jobs
        for j in range(H):
            temp = proc_local_fwd[i] + trans_back_activations[i,j] + release_date_back[i,j]

            for t in range(temp):
                m.addConstr(z[j,i,t] == 0)

            for t in range(T):
                if t + temp >= T:
                    break
                
                m.addConstr(z[j,i,t+temp] <= qsum(x[j,i,l] for l in range(t))/proc_fwd[i,j])
    
   
    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        for j in range(H):
            m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc_bck[i, j] == y[i,j])
    
    # C11: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T + z[j,:,:].T @ ones_K <= ones_T )
 
        
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*z[j,i,t] for t in range(T))


    # Define objective function
    
    m.addConstrs(comp[i] == qsum(trans_back_activations[i,:] * y[i,:]) + f[i] + proc_local_back[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    end = time.time()
    build_time = end-start
    print(f'{utils.bcolors.OKBLUE}build took: {end-start}{utils.bcolors.ENDC}')

    start = time.time()
    # Optimize model
    m.optimize()
    end = time.time()
   
    print(f'{utils.bcolors.OKBLUE}optimize took: {m.Runtime}{utils.bcolors.ENDC}')
    print(f'{utils.bcolors.OKBLUE}TOTAL TIME: {(end-start) + build_time}{utils.bcolors.ENDC}')
    print(f'{utils.bcolors.OKBLUE}Objective is: {m.ObjVal}{utils.bcolors.ENDC}')

    f_y.write(f'Optima solution\n')
    f_y.write(f'{np.rint(y.X)}\n')
    f_y.close()
    
    return(m.ObjVal)

if __name__ == '__main__':
    run()