import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum

import utils
import time

import warnings
warnings.filterwarnings("ignore")

def main():
    K = 5 # number of data owners
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

    m = gp.Model("fwd_only")

    # define variables
    x = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")

    '''
    x = m.addMVar(shape = (H,K,T), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")
    y = m.addMVar(shape=(K,H), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")
    f = m.addMVar(shape=(K), lb=0, vtype=GRB.CONTINUOUS, name="f")
    maxobj = m.addMVar(shape=(1),lb=0, vtype=GRB.CONTINUOUS, name="maxobj")
    comp = m.addMVar(shape=(K),lb=0, vtype=GRB.CONTINUOUS, name="comp")
    '''
    start = time.time()
    # define constraints
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
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
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc[i,j])

    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*x[j,i,t] for t in range(T))

    # Define objective function
    

    m.addConstrs(comp[i] == qsum(trans_back[i,:] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    end = time.time()
    time1 = end-start
    print(f'problem formulation: {time1}')


    start = time.time()
    # Optimize model
    m.optimize()
    end = time.time()
    print(f'problem solver: {end-start}')
    print(f'TOTAL: {(end-start) + time1}')

    
    #print('%s %g' % (v.VarName, v.X))
    print('Obj: %g' % m.ObjVal)

    # Checking if constraints are satisfied
    print("checking if constraints are satisfied")

    
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(K): #for all jobs
        my_machine = -1
        for j in range(H):
            if y[i,j].X == 1:
                my_machine = j
                break
        for k in range(release_date[i,my_machine]):
            if x[my_machine,i,k].X == 1:
                print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                return

    # C3: all jobs interval are assigned to one only machine
    for i in range(K): #for all jobs
        if np.sum([y[i,j].X for j in range(H)]) != 1:
            print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
            return

    # C4: memory constraint
    for j in range(H): #for all devices
        if np.sum([y[i,j].X for j in range(H)])*utils.max_memory_demand > memory_capacity[j]:
            print(f"{utils.bcolors.FAIL}Constraint 4 is violated{utils.bcolors.ENDC}")
            return

    
    # C5: job should be processed entirely once
    for i in range(K):
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].X)  == 1:
                my_machine = j
                break
        
        sum = 0
        for k in range(T):
            sum += np.rint(x[my_machine,i,k].X)
        if sum != proc[i, my_machine]:
            print(f"{utils.bcolors.FAIL}Constraint 5 is violated{utils.bcolors.ENDC}")
            #return

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in range(K):
                temp += np.rint(x[j,key,t].X)
            if temp > 1:
                print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                return

    #C8: the completition time for each data owner
    '''
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].X) == 1:
                my_machine = j
                break
        print("NEW")
        last_zero = -1
        for k in range(T):
            print(f'{x[my_machine,i,k].X} {k}')
            if np.rint(x[my_machine,i,k].X) >= 1:
                last_zero = k+1
        fmax = last_zero
        if fmax != f[i].X:
            print(fmax)
            print(f[i].X)
            print(f"{utils.bcolors.FAIL}Constraint 8 is violated{utils.bcolors.ENDC}")
            return
    '''

    print(f"{utils.bcolors.OKGREEN}All constraints are satisfied{utils.bcolors.ENDC}")

    print("--------Machine allocation--------")

    for i in range(H):
        for k in range(T):
            at_least = 0
            for j in range(K):
                if(np.rint(x[i,j,k].X) <= 0):
                    continue
                else:
                    print(f'{j+1}', end='\t')
                    at_least = 1
                    break
            if(at_least == 0):
                print(f'0', end='\t')
        print('')

    print("--------Completition time--------")
    '''
    for i in range(K):
        C = np.rint(f[i].X)
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        C += proc_local[i] + trans_back[i,my_machine]
        print(f'C{i+1}: {C}')
    '''
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].X) == 1:
                my_machine = j
                break
        last_zero = -1
        for k in range(T):
            if np.rint(x[my_machine,i,k].X) >= 1:
                last_zero = k+1
        fmax = last_zero
        C = fmax + proc_local[i] + trans_back[i,my_machine]
        print(f'C{i+1}: {C}')


if __name__ == '__main__':
    main()