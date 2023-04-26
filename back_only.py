import numpy as np
import cvxpy as cp
import time

import utils
import fwd_only

import warnings
warnings.filterwarnings("ignore")



def main():
    K = 5 # number of data owners
    H = 2 # number of compute nodes
    utils.file_name = 'test1.xlsx'

    C_fwd = cp.Parameter((K), integer=True)
    #f_fwd = cp.Parameter((K))
    f_bwd = cp.Parameter((K))
    #f_bwd = np.zeros(K)
    #f_bwd = cp.Variable((K), integer=True)
    
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))

    # forward-propagation parameters
    release_date_fwd = np.array(utils.get_fwd_release_delays(K,H))
    proc_fwd = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local_fwd = np.array(utils.get_fwd_end_local(K))
    trans_back_activations = np.array(utils.get_trans_back(K, H))

    # back-propagation parameters
    release_date_back = np.array(utils.get_bwd_release_delays(K,H))
    proc_bck = np.array(utils.get_bwd_proc_compute_node(K,H))
    proc_local_back = np.array(utils.get_bwd_end_local(K))
    trans_back_gradients = np.array(utils.get_grad_trans_back(K,H))

    
    T = np.max(release_date_fwd) + K*np.max(proc_fwd[0,:]) + np.max(release_date_back) + K*np.max(proc_bck[0,:]) \
                        + np.max(proc_local_back) + np.max(proc_local_back) +np.max(proc_fwd) \
                        + np.max(np.max(trans_back_activations)) + np.max(np.max(trans_back_gradients))
    print(f"T = {T}")
    
    x_prev, y_prev = fwd_only.main()
    # Define variables
    x = {}
    for i in range(K):
        x[i] = cp.Parameter((H,T), boolean=True)
        x_temp = np.zeros((H,T))
        x_prev_array = np.abs(np.rint(x_prev[i].value))

        for j in range(H):
            for k in range(x_prev_array.shape[1]):
                x_temp[j,k] = x_prev_array[j,k]
        
        x[i].value = x_temp

    

    y = cp.Parameter((K,H), boolean=True) # auxiliary variable

    y.value = np.abs(np.rint(y_prev.value))
    
    z = {}
    for i in range(K):
        z[i] = cp.Variable((H,T), boolean=True)

    # Define constraints
    constraints = []
    
    # C9: backprop job cannot be assigned to a time interval before the backprops release time
    for i in range(K): #for all jobs
        for j in range(H):
            temp = proc_local_fwd[i] + trans_back_activations[i,j] + release_date_back[i,j]

            for t in range(temp):
                constraints += [z[i][j,t] == 0]

            for t in range(T):
                if t + temp >= T:
                    break
                
                constraints += [z[i][j,t+temp] <= cp.sum(x[i][j,:t])/proc_fwd[i,j]]
           

    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        for j in range(H):
            constraints += [cp.sum(z[i][ j, :])/ proc_bck[i, j] == y[i,j]]
     
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
                #constraints += [f_bwd[i] >= (t)*z[i][j,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f_bwd = cp.hstack(f_values)

    # Define objective function
    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back_gradients[i,:] * y[i,:]))

    obj = cp.Minimize(cp.max( f_bwd + proc_local_back + cp.hstack(trans)))

    # wrap the formula to a Problem
    prob = cp.Problem(obj, constraints)
    start = time.time()
    prob.solve(solver=cp.GUROBI, verbose=True,
                options={'Threads': 16},)
    end = time.time()
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("Time: ", (end - start))
    '''
    # check - C1
    for i in range(K): #for all jobs
        my_machine = -1
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        for k in range(release_date_fwd[i,my_machine]):
            if np.rint(x[i][j,k].value) == 1:
                print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                return

    
    # check - C3
    for i in range(K): #for all jobs
        if np.sum(np.rint(y[i,:].value)) != 1:
            print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
            return

    # check - C4
    for j in range(H): #for all devices
        if np.sum(np.rint(y[:,j].value))*utils.max_memory_demand > memory_capacity[j]:
            print(f"{utils.bcolors.FAIL}Constraint 4 is violated{utils.bcolors.ENDC}")
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
        if sum != proc_fwd[i, my_machine]:
            print(f"{utils.bcolors.FAIL}Constraint 5 is violated{utils.bcolors.ENDC}")
            return

    # check - C6
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += np.rint(x[key][j,t].value)
            if temp > 1:
                print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
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
            print(f"{utils.bcolors.FAIL}Constraint 8 is violated{utils.bcolors.ENDC}")
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
                    print(f"{utils.bcolors.FAIL}Constraint 9 is violated - backpropagation assigned to different machine from fwd{utils.bcolors.ENDC}")
                    return

        for t in range(T):
            if t < release_date_back[i,my_machine] + np.rint(f_fwd[i].value) + trans_back_activations[i, my_machine] + proc_local_fwd[i]:
                if np.rint(z[i][j,t].value) == 1:
                    print(f'{i} {j} {t}')
                    print(f"{utils.bcolors.FAIL}Constraint 9 is violated{utils.bcolors.ENDC}")
                    return
            else:
                break
    '''
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
        if sum != proc_bck[i, my_machine]:
            print(f"{utils.bcolors.FAIL}Constraint 10 is violated{utils.bcolors.ENDC}")
            return

    # check - C11
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in x:
                temp += np.rint(x[key][j,t].value + z[key][j,t].value)
            if temp > 1:
                print(f"{utils.bcolors.FAIL}Constraint 11 is violated{utils.bcolors.ENDC}")
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
        if fmax != f_bwd[i].value:
            print(f"{utils.bcolors.FAIL}Constraint 12 is violated{utils.bcolors.ENDC}")
            #return
    
    print(f"{utils.bcolors.OKGREEN}All constraints are satisfied{utils.bcolors.ENDC}")

    f_out = open("all_out.txt", "w")



    f_out.write(f"release date - shape (K,H)\n, {release_date_fwd}")
    f_out.write(f"memory capacity\n {memory_capacity}",)
    f_out.write(f"proc. times\n, {proc_fwd}")
    f_out.write(f"send back\n, {trans_back_activations}")
    f_out.write(f"fwd last local\n, {proc_local_fwd}")
    f_out.write("--------------------------------")
    f_out.write("optimal time allocation:")

    for i in range(len(list(x.keys()))):
        print(f'Data ownwer/job {i+1}:\n {np.rint(z[i].value)}')


    f_out.write("--------------------------------")
    f_out.write(f"optimal device allocation\n {np.rint(y.value)}")


    f_out.write("--------------------------------")
    f_out.write("optimal forward finish time")
    print("finish time:")
    print(f_bwd.value)
    f_out.write("optimal bacward finish time")
    #f_out.write(f_bwd.value)

   
    f_out.close()


    print("--------Machine TIMELINE--------")

    for i in range(H):
        for k in range(T):
            at_least = 0
            for j in range(K):
                if(np.rint(x[j][i,k].value) <= 0 and np.rint(z[j][i,k].value) <= 0):
                    continue
                else:
                    if np.rint(x[j][i,k].value) > 0:
                        print(f'{j+1}', end='\t')
                    if np.rint(z[j][i,k].value) > 0:
                        print(f'{j+1}\'', end='\t')
                    at_least = 1
                    break
            if(at_least == 0):
                print(f'0', end='\t')
        print('')

    print("--------End allocation--------")

    for i in range(K):
        C = np.rint(f_bwd[i].value)
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].value) == 1:
                my_machine = j
                break
        C += proc_local_back[i] + trans_back_gradients[i,my_machine]
        print(f'C{i+1}: {C}')

if __name__ == '__main__':
    main()