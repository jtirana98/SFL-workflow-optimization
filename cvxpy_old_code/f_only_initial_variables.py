import numpy as np
import cvxpy as cp
import time

import utils

import warnings
warnings.filterwarnings("ignore")


def run():
    start = time.time()
    utils.file_name = 'test3.xlsx'
    K = 50 # number of data owners
    H = 5 # number of compute nodes
    release_date = np.array(utils.get_fwd_release_delays(K,H)) # release date - shape (K,H)
    #release_date = np.array([[3,2],[3,4],[2,2]])
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    #memory_capacity = np.array([10,10])
    proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    #proc=np.array([[2,3], [1,2], [3,4]])

    proc_local = np.array(utils.get_fwd_end_local(K))
    #proc_local=np.array([1,1,1])
    #trans_back=np.array([[1,1],[1,1],[1,1]])
    trans_back = np.array(utils.get_trans_back(K, H))


    proc_param = cp.Parameter((K, H))
    trans_back_pp = cp.Parameter((K, H))
    #f = cp.Parameter((K))
    #f.value = np.zeros(K)

    trans_back_pp.value  = np.array(trans_back)
    proc_param.value = np.array(proc)

    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    # Define variables
    x = {}
    for i in range(K):
        x[i] = cp.Variable((H,T), boolean=True)

    y = cp.Variable((K,H), boolean=True) # auxiliary variable
    f = cp.Variable(K, integer=True) # completition time

    # Define constraints
    constraints = []

    # C1: job cannot be assigned to a time interval before the release time
    for i in range(K): #for all jobs
        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                if t < release_date[i,j]: # <= ?? -- minor
                    constraints += [x[i][j,t] == 0]

    # C2: define auxiliary variable
    for i in range(K): #for all jobs
        for j in range(H): #for all devices
            constraints += [cp.sum(x[i][j,:]) <= T*y[i,j]]

    # C3: all jobs interval are assigned to one only machine
    for i in range(K): #for all jobs
        constraints += [cp.sum(y[i,:]) == 1]

    # C4: memory constraint
    for j in range(H): #for all devices
        constraints += [cp.sum(y[:,j])*utils.max_memory_demand <= memory_capacity[j]]

    # C5: job should be processed entirely once
    for i in range(K): #for all jobs
        sub_sum = []
        for j in range(H):
            sub_sum += [cp.sum(x[i][ j, :])/ proc_param[i, j]]
        sum_ = cp.sum(cp.hstack(sub_sum))
        constraints += [sum_ == 1]

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += x[key][j,t]
            constraints += [temp <= 1]

    #C8: the completition time for each data owner -- NOTE: Removed it from constraints
    #S1
    '''
    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*x[i][j,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f = cp.hstack(f_values)
    
    '''
    #S3
    for i in range(K): #for all jobs
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                constraints += [f[i] >= (t+1)*x[i][j,t]]
    '''
    f_temp = []
    for i in range(K):
        r_min = np.min(release_date[i][:])
        p_min = np.min(proc[i][:])
        f_temp += [r_min + p_min]

    '''

    # initialize with FIFO order

    '''
    f.value = np.array([28, 34, 25, 41, 24, 19, 33, 29, 37, 38, 32, 37, 30, 33, 21, 41, 40, 37,
                        38, 38, 19, 27, 36, 35, 27, 30, 34, 31, 40, 12, 25, 41, 24, 36, 22, 32,
                        37, 38, 27, 30, 35, 26, 39, 39, 15, 39, 17, 40, 20, 15])
    '''
    print(f_array)
    f.value = f_array
    # Define objective function

    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back_pp[i,:] * y[i,:]))


    obj = cp.Minimize(cp.max( f + proc_local + cp.hstack(trans)))

    # wrap the formula to a Problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.GUROBI, warm_start= True, verbose=True)
    '''
    prob.solve(solver=cp.MOSEK, verbose=True,
            mosek_params={
                    'MSK_IPAR_NUM_THREADS': 8,
                    },
                #save_file = 'dump_dump.ptf',
                )
    '''

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
            if np.rint(x[i][j,k].value) == 1:
                print("Constraint 1 is violated")
                return

    # C2: define auxiliary variable
    # C3: all jobs interval are assigned to one only machine
    for i in range(K): #for all jobs
        if np.sum(np.rint(y[i,:].value)) != 1:
            print("Constraint 3 is violated")
            return

    # C4: memory constraint
    for j in range(H): #for all devices
        if np.sum(np.rint(y[:,j].value))*utils.max_memory_demand > memory_capacity[j]:
            print("Constraint 4 is violated")
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
            sum += np.rint(x[i][my_machine,k].value)
        if sum != proc_param[i, my_machine].value:
            print("Constraint 5 is violated")
            return

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += np.rint(x[key][j,t].value)
            if temp > 1:
                print("Constraint 6 is violated")
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
            if np.rint(x[i][my_machine,k].value) >= 1:
                last_zero = k+1
        fmax = last_zero
        if fmax != f[i].value:
            #f[i].value = fmax
            print(f"Constraint 8 is violated for client {i+1}")
            #return
    end = time.time()
    print('All constraints are satisfied')

    print("--------Machine allocation--------")

    for i in range(H):
        for k in range(T):
            at_least = 0
            for j in range(K):
                if(np.rint(x[j][i,k].value) <= 0):
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
        #C = np.rint(f[i].value)
        
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        
        last_zero = -1
        for k in range(T):
            if np.rint(x[i][my_machine,k].value) >= 1:
                last_zero = k+1
        fmax = last_zero


        #C += proc_local[i] + trans_back[i,my_machine]
        C = proc_local[i] + trans_back[i,my_machine] + fmax
        print(f'C{i+1}: {C}')

    
    return (end - start)

def main():
    #for i in range(10):
    print(run())

if __name__ == '__main__':
    main()