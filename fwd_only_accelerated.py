import numpy as np
import cvxpy as cp
import time

import utils

import warnings
warnings.filterwarnings("ignore")



def main():
    K = 100 # number of data owners
    H = 5 # number of compute nodes
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
    constraints = []
    '''
    for j in range(H): #for all devices
        constraints += [ x[j]<= 1 ]

   
    #constraints += [ y <= 1 ]
    ''''
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    constraints += [ x[i][j,t] == 0 ]
    
    # C3: all jobs interval are assigned to one only machine
    constraints += [ y @ ones_H == ones_K ]
    
    # C4: memory constraint
    constraints += [ (y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape) ]
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        constraints += [ x[j].T @ ones_K <= ones_T ]
    
    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        constraints += [cp.sum(x[j],axis=1) == cp.multiply(y,proc).T[j,:]]
    
    #C8: the completition time for each data owner -- NOTE: Removed it from constraints
    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*x[j][i,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f = cp.hstack(f_values)
    
    # Define objective function
    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back[i,:] * y[i,:]))


    obj = cp.Minimize(cp.max( f + proc_local + cp.hstack(trans)))

    # wrap the formula to a Problem
    prob = cp.Problem(obj, constraints)
    start = time.time()
    prob.solve(solver=cp.GUROBI, verbose=True)
    end = time.time()
    print("TIME: ", end - start)

    print("status:", prob.status)
    print("optimal value", prob.value)
    
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(K): #for all jobs
        my_machine = -1
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        for k in range(release_date[i,my_machine]):
            if np.rint(x[j][i,k].value) == 1:
                print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                return

    # C2: define auxiliary variable
    # C3: all jobs interval are assigned to one only machine
    for i in range(K): #for all jobs
        if np.sum(np.rint(y[i,:].value)) != 1:
            print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
            return

    # C4: memory constraint
    for j in range(H): #for all devices
        if np.sum(np.rint(y[:,j].value))*utils.max_memory_demand > memory_capacity[j]:
            print(f"{utils.bcolors.FAIL}Constraint 4 is violated{utils.bcolors.ENDC}")
            return

    # C5: job should be processed entirely once
    for i in range(K):
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        sum = 0
        for k in range(T):
            sum += np.rint(x[my_machine][i,k].value)
        if sum != proc[i, my_machine]:
            print(f"{utils.bcolors.FAIL}Constraint 5 is violated{utils.bcolors.ENDC}")
            return

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for i in range(K):
                temp += np.rint(x[j][i,t].value)
            if temp > 1:
                print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                return

    #C8: the completition time for each data owner
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        
        last_zero = -1
        for k in range(T):
            if np.rint(x[my_machine][i,k].value) >= 1:
                last_zero = k+1
        fmax = last_zero
        if fmax != f[i].value:
            print(f"{utils.bcolors.FAIL}Constraint 8 is violated{utils.bcolors.ENDC}")
            return

    print(f"{utils.bcolors.OKGREEN}All constraints are satisfied{utils.bcolors.ENDC}")
    
    print('X:')
    for i in range(len(list(x.keys()))):
        print(f'Data ownwer/job {i+1}:')
        print(x[i].value)

    print("--------------------------------")
    print("optimal device allocation (Y)")
    print(y.value)

    #print("--------------------------------")
    #print("optimal end time")
    #print(f.value)

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

    print("--------Completition time--------")

    for i in range(K):
        C = np.rint(f[i].value)
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        C += proc_local[i] + trans_back[i,my_machine]
        print(f'C{i+1}: {C}')
    return (x,y)

if __name__ == '__main__':
    main()