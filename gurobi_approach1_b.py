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
    K = 10 # number of data owners
    H = 2 # number of compute nodes
    utils.file_name = 'fully_heterogeneous.xlsx'

    release_date = np.array(utils.get_fwd_release_delays(K,H))
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local = np.array(utils.get_fwd_end_local(K))
    trans_back = np.array(utils.get_trans_back(K, H))
    memory_capacity = np.array([9,21])
    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m1 = gp.Model("relax_approach_1_p1")
    m2 = gp.Model("relax_approach_1_p2")


    # define variables - problem 1
    
    y = m1.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")

    # define variables - problem 2
    x = m2.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    f = m2.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    w = m2.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m2.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")

    # dual variables

    lala = np.ones((K,H)) # lamda variable
    mama = np.ones((K)) # m variable
    #lala = np.random.normal(0,4, size=(K,H))
    #mama = np.random.normal(0,4, size=(H))

    # Define constraints for problem 1
    print(f"max-f: {T} min-f: {np.min(release_date) + np.min(proc[0,:])}")
    print(f"min-w: {np.min(release_date) + np.min(proc[0,:]) + np.min(trans_back[0,:]) + np.min(proc_local)} max-w: {T + np.max(trans_back) + np.max(proc_local)}")

        
    # C3: each job is assigned to one and only machine
    m1.addConstr( y @ ones_H == ones_K )

    # C4: memory constraint
    m1.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))


    # Define constraints for problem 2
    
    m2.addConstr(f <= T)
    m2.addConstr(f >=  np.min(release_date) + np.min(proc[0,:]))
    m2.addConstr(w <= T + np.max(trans_back) + np.max(proc_local))
    m2.addConstr(w >= np.min(release_date) + np.min(proc[0,:]) + np.min(trans_back[0,:]) + np.min(proc_local))

    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    m2.addConstr(x[i,j,t] == 0)
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m2.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    for j in range(H): #for all machines
        for i in range(K):
            m2.addConstrs( f[i] >= (t+1)*x[j,i,t] for t in range(T))


    # forgoten constraint
    # C5: job should be processed entirely once
    for i in range(K):
        m2.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc[i,j] for j in range(H)) == 1)

    # Iterative algorithm
    step = 0
    alpha = 0
    bhta = 0

    violations = []
    ws = []
    obj1 = []
    obj2 = []
    cool = True
    while step<3:
        m1.setObjective(qsum(-lala[i,j] * y[i,j] * T for i in range(K) for j in range(H)) + qsum(mama[i] * qsum(y[i,j] * trans_back[i,j] for j in range(H)) for i in range(K) ), GRB.MINIMIZE)    
        m1.update()
        m2.setObjective(w + qsum(qsum(x[i,j,t] for t in range(T))*lala[j,i] for i in range(H) for j in range(K)) + qsum(mama[i]*(proc_local[i] + f[i] - w) for i in range(K)), GRB.MINIMIZE)
        m2.update()
          
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
        
        obj1 += [m1.ObjVal]
        obj2 += [m2.ObjVal]
        
        # update dual variables
        for j in range(K):
            mama[j] = max(mama[i] + bhta*(proc_local[i] + sum(y[j,k].X*trans_back[i,k] for k in range(H)) + f[j].X - w.X), 0)
            for i in range(H):
                lala[j,i] = max(lala[j,i] + alpha*(-y[j,i].X*T + sum([x[i,j,k].X for k in range(T)])), 0)
                    
                    
        step = step + 1
        alpha = 1/math.sqrt(step)
        bhta = 1/math.sqrt(step)

        print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE: {w.X}{utils.bcolors.ENDC}')
        ws.append(w[0].X)
        print(f'{utils.bcolors.OKBLUE}Checking constraints{utils.bcolors.ENDC}')

        counter = 0
        
        
        for j in range(K):
            comp_ = np.rint(f[j].X) + sum(np.rint(y[j,i].X)*trans_back[j,i] for i in range(H)) + proc_local[j]
            if np.rint(w.X) < comp_:
                #print(f"{utils.bcolors.FAIL}Constraint  is violated expected larger than: {np.rint(w.X)} got:{comp_} {utils.bcolors.ENDC}")
                counter += 1
        
        print(f'{utils.bcolors.OKBLUE}C1{utils.bcolors.ENDC}')
        for i in range(H):
            for j in range(K):
                temp = 0
                for t in range(T):
                    temp += np.rint(x[i,j,t].X)
                
                if temp < np.rint(y[j,i].X)*T:
                    #print(f"{utils.bcolors.FAIL}Constraint 1 is violated expected larger than: {y[j,i].X*proc[j,i]} got:{temp} {utils.bcolors.ENDC}")
                    counter += 1
        
        print(f'{utils.bcolors.OKBLUE}constraints violated: {counter}{utils.bcolors.ENDC}')
        violations += [counter]
        print("--------Machine allocation--------")

        for i in range(H):
            for k in range(T):
                at_least = 0
                for j in range(K):
                    if(np.rint(x[i,j,k].X) <= 0):
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
        for i in range(K): #for all jobs
            my_machine = 0
            my_super_machine = 0
            last_zero = -1
            for my_machine in range(H):
                for k in range(T):
                    if np.rint(x[my_machine,i,k].X) >= 1:
                        if last_zero < k+1:
                            last_zero = k+1
                            my_super_machine = my_machine
            fmax = last_zero
            C = fmax + proc_local[i] + trans_back[i,my_machine]
            cs.append(C)
            print(f'C{i+1}: {C} - {my_super_machine} {y[i,:].X}')

        print(f'max is: {max(cs)}')
        print("check other constraints")
        for i in range(K): #for all jobs
            my_machine = -1
            for j in range(H):
                if y[i,j].X == 1:
                    my_machine = j
                    break
            for k in range(release_date[i,my_machine]):
                if x[my_machine,i,k].X == 1:
                    print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")

        for i in range(K): #for all jobs
            if np.sum([y[i,j].X for j in range(H)]) != 1:
                print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")

        for j in range(H): #for all devices
            if np.sum([y[i,j].X for j in range(H)])*utils.max_memory_demand > memory_capacity[j]:
                print(f"{utils.bcolors.FAIL}Constraint 4 is violated{utils.bcolors.ENDC}")

        for i in range(K):
            my_machine = 0
            #for j in range(H):
            #    if np.rint(y[i,j].X)  == 1:
            #        my_machine = j
            #        break
            for my_machine in range(H):
                sum_ = 0
                for k in range(T):
                    sum_ += np.rint(x[my_machine,i,k].X)
                if sum_ != 0 and sum_ != proc[i, my_machine] :
                    print(f"{utils.bcolors.FAIL}Constraint 5 is violated {i+1}{utils.bcolors.ENDC}")

        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                temp = 0
                for key in range(K):
                    temp += np.rint(x[j,key,t].X)
                if temp > 1:
                    print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")

    print(f'optimal:  {ws}')
    print(f'violations: {violations}')
    #for v in m2.getVars():
    #    print('%s %g' % (v.VarName, v.X))
      
if __name__ == '__main__':
    main()