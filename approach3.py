import numpy as np
import cvxpy as cp
import time
import gurobipy

import utils

import warnings
warnings.filterwarnings("ignore")



def main():
    # Problem input

    K = 50 # number of data owners
    H = 2 # number of compute nodes
    utils.file_name = 'fully_heterogeneous.xlsx'

    # fully_symmetric
    # fully_heterogeneous
    # symmetric_machines
    # symmetric_data_owners

    release_date = cp.Parameter((K,H), value=np.array(utils.get_fwd_release_delays(K, H)))
    # memory_capacity = cp.Parameter(H, value=np.array(utils.get_memory_characteristics(H, K)))
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    proc = cp.Parameter((K,H), value=np.array(utils.get_fwd_proc_compute_node(K, H)))
    proc_local = cp.Parameter(K, value=np.array(utils.get_fwd_end_local(K)))
    trans_back = cp.Parameter((K,H), value=np.array(utils.get_trans_back(K, H)))

    memory_capacity = np.array([90,75])

    T = np.max(release_date.value) + K*np.max(proc[0,:].value) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    print(f" Memory: {memory_capacity}")

    

    # problem 1
    y = cp.Variable((K,H), boolean=True) # auxiliary variable
    f = cp.Variable(K, integer=True) # finish time
    w = cp.Variable(integer=True) # completion time

    s = {}
    for i in range(H):
        s[i] = cp.Variable((K,T), boolean=True)

    x_ = {}
    for i in range(H):
        x_[i] = cp.Parameter((K,T), value=np.zeros((K,T)))


    # problem 2
    x = {}
    for i in range(H):
        x[i] = cp.Variable((K,T), boolean=True)

    y_ = cp.Parameter((K,H), value=np.zeros((K,H))) # auxiliary variable
    f_ = cp.Parameter(K, value=np.zeros((K))) # finish time
    w_ = cp.Parameter(value=0) # completion time

    s_ = {}
    for i in range(H):
        s_[i] = cp.Parameter((K,T), value=np.zeros((K,T)))
    
    # Dual variables (parameters)
    lala = cp.Parameter((K,H), value=np.ones((K,H)))
    mama = {}
    for i in range(H):
        mama[i] = cp.Parameter((T,K), value=np.ones((T,K)))
    
    rho = 3

    # Define constraints problem 1

    constraints1 = []
    # constraints o to restrict the values of f and  w
    constraints1 += [f <= T]
    constraints1 += [f >=  np.min(release_date.value) + np.min(proc[0,:].value)]
    constraints1 += [w <= T + np.max(trans_back.value) + np.max(proc_local.value)]
    constraints1 += [w >= np.min(release_date.value) + np.min(proc[0,:].value) + np.min(trans_back[0,:].value) + np.min(proc_local.value)]


    # C3: each job is assigned to one and only machine
    constraints1 += [y @ ones_H == ones_K]

    # C4: memory constraint
    constraints1 += [(y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape)]

    # w and completion time
    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back[i,:] * y[i,:]))

    for i in range(K): # for each job/data owner
        constraints1 += [w >= f[i] + proc_local[i] + trans[i]]
   
    

    # Define constraints problem 2
    constraints2 = []
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i].value:
                    constraints2 += [ x[i][j,t] == 0 ]


    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        constraints2 += [ x[j].T @ ones_K <= ones_T ]

    # C5: job should be processed entirely once
    for i in range(K): #for all jobs
        sub_sum = []
        for j in range(H):
            sub_sum += [cp.sum(x[j][ i, :])/ proc[i, j]]
        sum_ = cp.sum(cp.hstack(sub_sum))
        constraints2 += [sum_ == 1]


    # Iterative algorithm
    step = 0
    alpha = 0
    bhta = 0

    violations = []
    ws = []
    obj1 = []
    obj2 = []
    max_ = 0
    while step<10:
        print(f"{utils.bcolors.OKBLUE}-------------{step}------------{utils.bcolors.ENDC}")

        y.value = np.zeros(y.shape)
        f.value = np.zeros(f.shape)
        w.value = np.zeros(w.shape)
        for i in range(H):
            s[i].value = np.zeros((K,T))


        # objective - 1
        term_lu_x = np.zeros((K,H))
        for i in range(K):
            for j in range(H):
                for t in range(T):
                    term_lu_x[i,j] += x_[j][i,t].value

        term_lu_x_param = cp.Parameter((K,H), value=term_lu_x)   

        term_mu = 0
        for i in range(H):
            for j in range(K):
                for t in range(T):
                    term_mu += f[j] - s[i][j,t] + (t+1)*x_[i][j,t]

        term_mu_abs = 0
        for i in range(H):
            for j in range(K):
                for t in range(T):
                    term_mu_abs += (rho/2)*cp.abs(f[j] - s[i][j,t] + (t+1)*x_[i][j,t])


        obj1_ = cp.Minimize(  w   
                        + cp.sum(cp.multiply(lala,y*T) - cp.multiply(lala,term_lu_x_param)) 
                        + term_mu 
                        + (rho)*cp.sum(cp.abs(cp.multiply(lala,y*T) - cp.multiply(lala,term_lu_x_param)))
                        + term_mu_abs)



    

        # wrap the formula to a Problem
        prob1 = cp.Problem(obj1_, constraints1)

        start = time.time()
        prob1.solve(solver=cp.GUROBI, verbose=False)
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P1 took: {prob1.solver_stats.solve_time}{utils.bcolors.ENDC}')

        #print("status:", prob1.status)
        #print("optimal value", prob1.value)

        # pass results to second problem
        y_.value =  np.abs(np.rint(y.value))
        f_.value =  np.abs((np.rint(f.value)))
        w_.value =  np.abs(np.rint(w.value))
        
        for i in range(H):
            s_[i].value =  np.abs(np.rint(s[i].value))

        for i in range(H):
            x[i].value = np.zeros((K,T))


        # objective - 2
    
        term_lu_x_param = cp.vstack(cp.sum(x[i], 1) for i in range(H)).T

        term_mu = 0
        for i in range(H):
            for j in range(K):
                for t in range(T):
                    term_mu += f_[j] - s_[i][j,t] + (t+1)*x[i][j,t]

        term_mu_abs = 0
        for i in range(H):
            for j in range(K):
                for t in range(T):
                    term_mu_abs += (rho/2)*cp.abs(f_[j] - s_[i][j,t] + (t+1)*x[i][j,t])

        obj2_ = cp.Minimize(  w_   
                        + cp.sum(cp.multiply(lala,y_*T) - cp.multiply(lala,term_lu_x_param)) 
                        + term_mu 
                        + (rho)*cp.sum(cp.abs(cp.multiply(lala,y_*T) - cp.multiply(lala,term_lu_x_param)))
                        + term_mu_abs)

        if step<3:
            env = gurobipy.Env()
            env.setParam('MIPGap', 0.05) # in seconds
        else:
            env = gurobipy.Env()
            env.setParam('MIPGap', 0.0001) # in seconds

        prob2 = cp.Problem(obj2_, constraints2)
        start = time.time()
        prob2.solve(solver=cp.GUROBI, verbose=True, env=env)
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P2 took: {prob2.solver_stats.solve_time}{utils.bcolors.ENDC}')

        obj1 += [prob1.value]
        obj2 += [prob2.value]

        # pass results to first problem
        for i in range(H):
            x_[i].value =  np.abs(np.rint(x[i].value))

        # update dual variables
        step = step + 1

        
        lala_ = np.zeros((K,H))
        mama_ = np.zeros((T,K))
        for i in range(H):
            for j in range(K):
                lala_[j,i] = lala[j,i].value + rho*(sum([x[i][j,k].value for k in range(T)]) - y[j,i].value*T)
                for t in range(T):
                    mama_[t,j] = mama[i][t,j].value + rho*(f[j].value - s[i][j,t].value + x[i][j,t].value*(t+1))
            mama[i].value = mama_
        lala.value = lala_

        print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE: {w.value}{utils.bcolors.ENDC}')
        ws.append(w.value)
        print(f'{utils.bcolors.OKBLUE}Checking constraints{utils.bcolors.ENDC}')

        counter = 0
        print(f'{utils.bcolors.OKBLUE}C8{utils.bcolors.ENDC}')
        for i in range(H):
            for j in range(K):
                for t in range(T):
                    if np.rint(f[j].value) < np.rint(x[i][j,t].value)*(t+1):
                        #print(f"{utils.bcolors.FAIL}Constraint 8 is violated expected larger than: {x[i,j,t].X*(t+1)} got:{f[i].X} {utils.bcolors.ENDC}")
                        counter += 1
        
        print(f'{utils.bcolors.OKBLUE}C1{utils.bcolors.ENDC}')
        for i in range(H):
            for j in range(K):
                temp = 0
                for t in range(T):
                    temp += np.rint(x[i][j,t].value)
                
                if temp < np.rint(y[j,i].value)*T:
                    #print(f"{utils.bcolors.FAIL}Constraint 1 is violated expected larger than: {y[j,i].X*proc[j,i]} got:{temp} {utils.bcolors.ENDC}")
                    counter += 1
        
        print(f'{utils.bcolors.OKBLUE}constraints violated: {counter}{utils.bcolors.ENDC}')
        violations += [counter]
        print("--------Machine allocation--------")

        for i in range(H):
            for k in range(T):
                at_least = 0
                for j in range(K):
                    if(np.rint(x[i][j,k].value) <= 0):
                        continue
                    else:
                        print(f'{j+1}', end='\t')
                        at_least += 1
                        break
                if(at_least == 0):
                    print(f'0', end='\t')
            print('')

        print("--------Completition time--------")
        cs = []
        reserved = [0 for i in range(H)]
        for i in range(K): #for all jobs
            my_machine = 0
            my_super_machine = 0
            last_zero = -1
            for my_machine in range(H):
                for k in range(T):
                    if np.rint(x[my_machine][i,k].value) >= 1:
                        if last_zero < k+1:
                            last_zero = k+1
                            my_super_machine = my_machine
            fmax = last_zero
            C = fmax + proc_local[i].value + trans_back[i,my_machine].value
            cs.append(C)
            print(f'C{i+1}: {C} - {my_super_machine} {y[i,:].value}')
            reserved[my_super_machine] += 1
                     
        print(f'max is: {max(cs)}')
        max_ = max(cs)
        print("check other constraints")
        for i in range(K): #for all jobs
            my_machine = -1
            for j in range(H):
                if y[i,j].value == 1:
                    my_machine = j
                    break
            for k in range(release_date[i,my_machine].value):
                if x[my_machine][i,k].value == 1:
                    print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")

        for i in range(K): #for all jobs
            if np.sum([y[i,j].value for j in range(H)]) != 1:
                print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")

        for j in range(H): #for all devices
            if np.sum([y[i,j].value for j in range(H)])*utils.max_memory_demand > memory_capacity[j]:
                print(f"{utils.bcolors.FAIL}Constraint 4 is violated{utils.bcolors.ENDC}")

            occupied = reserved[j]*utils.max_memory_demand
            if occupied > memory_capacity[j]:
                print(f"{utils.bcolors.FAIL}Constraint 4 is violated for machine {j}{utils.bcolors.ENDC}")

        
        for i in range(K):
            my_machine = 0
            #for j in range(H):
            #    if np.rint(y[i,j].X)  == 1:
            #        my_machine = j
            #        break
            at_least = 0
            for my_machine in range(H):
                sum_ = 0
                for k in range(T):
                    sum_ += np.rint(x[my_machine][i,k].value)
                if sum_ != 0 and sum_ != proc[i, my_machine].value :
                    print(f"{utils.bcolors.FAIL}Constraint 5 is violated {i+1}{utils.bcolors.ENDC}")
                else:
                    if sum_ != 0:
                        at_least += 1
            if at_least == 0:
                print(f"{utils.bcolors.FAIL}Constraint 5 is violated job not assigned {i+1}{utils.bcolors.ENDC}")
            if at_least > 1:
                print(f"{utils.bcolors.FAIL}Constraint 5 is violated job assigned more tmes {i+1}{utils.bcolors.ENDC}")
            

        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                temp = 0
                for key in range(K):
                    temp += np.rint(x[j][key,t].value)
                if temp > 1:
                    print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")


    print(f'optimal:  {ws}')
    print(f'violations: {violations}')


if __name__ == '__main__':
    main()
