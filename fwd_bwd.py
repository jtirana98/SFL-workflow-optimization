import numpy as np
import cvxpy as cp

import utils

import warnings
warnings.filterwarnings("ignore")



def main():
    K = 3 # number of data owners
    H = 2 # number of compute nodes

    proc_param_fwd = cp.Parameter((K, H))
    proc_param_bwd = cp.Parameter((K, H))
    trans_back_pp = cp.Parameter((K, H))
    C_fwd = cp.Parameter(K)
    f_fwd = cp.Parameter((K))
    f_bwd = cp.Parameter((K))

    
    memory_capacity = np.array(utils.get_memory_characteristics(H))

    # forward-propagation parameters
    release_date_fwd = np.array(utils.get_fwd_release_delays(K,H))
    proc_fwd = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_param_fwd.value = np.array(proc_fwd)
    proc_local_fwd = np.array(utils.get_fwd_end_local(K))
    trans_back_activations = np.array(utils.get_trans_back(K, H))
    trans_back_pp.value  = np.array(trans_back_activations)

    # back-propagation parameters
    release_date_back = np.array(utils.get_bwd_release_delays(K,H))
    proc_bck = np.array(utils.get_bwd_proc_compute_node(K,H))
    proc_param_bwd.value = np.array(proc_bck)
    proc_local_back = np.array(utils.get_bwd_end_local(K))
    trans_back_gradients = np.array(utils.get_grad_trans_back(K,H))

    
    T = np.max(release_date_back) + K*np.max(proc_bck[0,:]) # time intervals
    print(f"T = {T}")

    # Define variables
    x = {}
    for i in range(K):
        x[i] = cp.Variable((H,T), boolean=True)

    y = cp.Variable((K,H), boolean=True) # auxiliary variable
    
    z = {}
    for i in range(K):
        z[i] = cp.Variable((H,T), boolean=True)

    # Define constraints
    constraints = []

    # C1: job cannot be assigned to a time interval before the release time
    for i in range(K): #for all jobs
        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                if t < release_date_fwd[i,j]: # <= ?? -- minor
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
            sub_sum += [cp.sum(x[i][ j, :])/ proc_param_fwd[i, j]]
        sum_ = cp.sum(cp.hstack(sub_sum))
        constraints += [sum_ == 1]

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += x[key][j,t]
            constraints += [temp <= 1]

    #C8: the completition time for each data owner
    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*x[i][j,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f_fwd = cp.hstack(f_values)

    
    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back_pp[i,:] * y[i,:]))


    C_fwd =  f_fwd + cp.hstack(proc_local_fwd) + cp.hstack(trans)

    
    # C9: backprop job cannot be assigned to a time interval before the backprops release time
    for i in range(K): #for all jobs
        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                if t < (release_date_back[i,j]+ C_fwd[i]):
                    constraints += [z[i][j,t] == 0]
    
    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        sub_sum = []
        for j in range(H):
            sub_sum += [cp.sum(z[i][ j, :])/ proc_param_bwd[i, j]]
            sum_ = cp.sum(cp.hstack(sub_sum))
            constraints += [sum_ == y[i,j]]

    # C11: compute node will run only one job at each slot either fwd or bwd NOTE: can we ignore C6?
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += x[key][j,t] + z[key][j,t]
            constraints += [temp <= 1]

    #C12: the completition time for each data owner
    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*z[i][j,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f_bwd = cp.hstack(f_values)

    # Define objective function
    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back_gradients[i,:] * y[i,:]))

    obj = cp.Minimize(cp.max( f_bwd + proc_local_back + cp.hstack(trans)))

    # wrap the formula to a Problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.GUROBI, verbose=True,
                options={'Threads': 8},)

    print("status:", prob.status)
    print("optimal value", prob.value)


    # check - C1
    for i in range(K): #for all jobs
        my_machine = -1
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        for k in range(release_date_fwd[i,my_machine]):
            if np.rint(x[i][j,k].value) == 1:
                print(f"{utils.bcolors.WARNING}Constraint 1 is violated{utils.bcolors.ENDC}")
                return

    
    # check - C3
    for i in range(K): #for all jobs
        if np.sum(np.rint(y[i,:].value)) != 1:
            print(f"{utils.bcolors.WARNING}Constraint 3 is violated{utils.bcolors.ENDC}")
            return

    # check - C4
    for j in range(H): #for all devices
        if np.sum(np.rint(y[:,j].value))*utils.max_memory_demand > memory_capacity[j]:
            print(f"{utils.bcolors.WARNING}Constraint 4 is violated{utils.bcolors.ENDC}")
            return

    # check - C5
    for i in range(K):
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        sum = 0
        for k in range(T):
            sum += np.rint(x[i][my_machine,k].value)
        if sum != proc_param_fwd[i, my_machine].value:
            print(f"{utils.bcolors.WARNING}Constraint 5 is violated{utils.bcolors.ENDC}")
            return

    # check - C6
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += np.rint(x[key][j,t].value)
            if temp > 1:
                print(f"{utils.bcolors.WARNING}Constraint 6 is violated{utils.bcolors.ENDC}")
                return

    # check - C8
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
        if fmax != f_fwd[i].value:
            print(f"{utils.bcolors.WARNING}Constraint 8 is violated{utils.bcolors.ENDC}")
            return
        
     # check - C9
    for i in range(K): #for all jobs
        my_machine = -1
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break

        for j in range(H):
            if j == my_machine:
                continue

            for k in range(K):
                if np.rint(z[i][j,k].value) == 1:
                    print(f"{utils.bcolors.WARNING}Constraint 9 is violated - backpropagation assigned to different machine from fwd{utils.bcolors.ENDC}")
                    return

        for k in range(release_date_back[i,my_machine]):
            if np.rint(z[i][j,k].value) == 1:
                print(f"{utils.bcolors.WARNING}Constraint 9 is violated{utils.bcolors.ENDC}")
                return
    
    # check - C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K):
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        sum = 0
        for k in range(T):
            sum += np.rint(z[i][my_machine,k].value)
        if sum != proc_param_bwd[i, my_machine].value:
            print(f"{utils.bcolors.WARNING}Constraint 10 is violated{utils.bcolors.ENDC}")
            return

    # check - C11
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += np.rint(x[key][j,t].value + z[key][j,t].value)
            if temp > 1:
                print(f"{utils.bcolors.WARNING}Constraint 11 is violated{utils.bcolors.ENDC}")
                return

    #C12: the completition time for each data owner
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        
        last_zero = -1
        for k in range(T):
            if np.rint(z[i][my_machine,k].value) >= 1:
                last_zero = k+1
        fmax = last_zero
        if fmax != f_fwd[i].value:
            print(f"{utils.bcolors.WARNING}Constraint 12 is violated{utils.bcolors.ENDC}")
            return
    
    print(f"{utils.bcolors.OKGREEN}All constraints are satisfied{utils.bcolors.ENDC}")


    print("release date - shape (K,H)\n", release_date_fwd)
    print("memory capacity\n", memory_capacity)
    print("proc. times\n", proc_fwd)
    print("send back\n", trans_back_activations)
    print("fwd last local\n", proc_local_fwd)
    print("--------------------------------")
    print("optimal time allocation:")

    for i in range(len(list(x.keys()))):
        print(f'Data ownwer/job {i+1}:')
        print(np.rint(x[i].value))

    print("--------------------------------")
    print("optimal device allocation")
    print(np.rint(y.value))

    print("--------------------------------")
    print("optimal end time")
    print(f.value)


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

    print("--------Machine allocation--------")

    for i in range(K):
        C = np.rint(f_fwd[i].value)
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        C += proc_local_fwd[i] + trans_back_activations[i,my_machine]
        print(f'C{i+1}: {C}')

if __name__ == '__main__':
    main()