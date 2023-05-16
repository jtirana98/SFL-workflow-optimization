import numpy as np
import math

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum


import utils
import time

import warnings
warnings.filterwarnings("ignore")

K = 10 # number of data owners
H = 2 # number of compute nodes
utils.file_name = 'fully_symmetric.xlsx'

def run(filename='', testcase='fully_symmetric'):
    #fully_heterogeneous
    #fully_symmetric

    if testcase == 'fully_symmetric':
        utils.file_name = 'fully_symmetric.xlsx'
    else:
        utils.file_name = 'fully_heterogeneous.xlsx'

    release_date = np.array(utils.get_fwd_release_delays(K,H))
    proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local = np.array(utils.get_fwd_end_local(K))
    trans_back = np.array(utils.get_trans_back(K, H))

    if utils.file_name == 'fully_symmetric.xlsx':
        memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    else:
        if K == 50:
            memory_capacity = np.array([30,120])
        else:
            memory_capacity = np.array([105,195])
    
    if filename != '':
        f_log = open(filename, "a")
        f_log.write("Approach:\n")
    else:
        f_log = open('mytest', "w")

    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m1 = gp.Model("relax_approach_1_p1")
    m2 = gp.Model("relax_approach_1_p2")

    rho = 3
    # define variables - problem 1
    
    y = m1.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    f = m1.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    w = m1.addMVar(shape=(1),vtype=GRB.INTEGER, name="w")
    comp = m1.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")
    s = m1.addMVar(shape=(H,K,T), vtype=GRB.BINARY, name="s")

    x_ = np.zeros((H,K,T))

    # define variables - problem 2
    x = m2.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y_ = np.zeros((K,H))
    f_ = np.zeros((K))
    w_ = np.zeros((1))
    s_ = np.zeros((H,K,T))

    # dual variables

    lala = np.ones((K,H)) # lamda variable
    mama = np.ones((H,T,K)) # m variable
    
    # Define constraints for problem 1
    print(f"max-f: {T - np.min(trans_back[0,:]) - np.min(proc_local)} min-f: {np.min(release_date) + np.min(proc[0,:])}")
    print(f"min-w: {np.min(release_date) + np.min(proc[0,:]) + np.min(trans_back[0,:]) + np.min(proc_local)}")
    

    m1.addConstr(f <= T)
    m1.addConstr(f >=  np.min(release_date) + np.min(proc[0,:]))
    m1.addConstr(w <= T + np.max(trans_back) + np.max(proc_local))
    m1.addConstr(w >= np.min(release_date) + np.min(proc[0,:]) + np.min(trans_back[0,:]) + np.min(proc_local))
    
    # C3: each job is assigned to one and only machine
    m1.addConstr( y @ ones_H == ones_K )

    # C4: memory constraint
    m1.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))

    # completition time definition
    m1.addConstrs(comp[i] == qsum(trans_back[i,:] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
    max_constr = m1.addConstrs(w >= comp[i] for i in range(K))

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
    

    # forgoten constraint
    # C5: job should be processed entirely once
    for i in range(K):
        m2.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc[i,j] for j in range(H)) == 1)

    # Iterative algorithm
    step = 0
    alpha = 0
    bhta = 0

    violations_1 = []
    violations_2 = []
    max_c = []
    accepted = []
    ws = []
    obj1 = []
    obj2 = []
    max_ = T
    add = False
    aux_abs = m1.addMVar(shape=1, vtype=GRB.BINARY, name="aux_abs")

    while step<5:
        print(f"{utils.bcolors.OKBLUE}-------------{step}------------{utils.bcolors.ENDC}")
        f_log.write(f"-------------{step}------------\n")
        
        
        
        #m1.addConstr(aux_abs == gp.norm((qsum(x_[:,:,t] for t in range(K))) - proc.T*y.T,1))         
        m1.setObjective(w + qsum(qsum(mama[j,t,i]*(f[i] - s[j,i,t] - x_[j,i,t]*(t+1)) \
                          + lala[i,j]*x_[j,i,t] for t in range(T)) - lala[i,j] * y[i,j] * proc[i,j]\
                          for i in range(K) for j in range(H)) \
                          #+ (rho/2)*qsum((qsum(x_[j,i,t] for t in range(K)) - proc[i,j]*y[i,j]) for i in range(K) for j in range(H)) \
                          + (rho/2)*qsum(gp.abs_((qsum(x_[j,i,t] for t in range(K)))) for i in range(K) for j in range(H)) \
                          + (rho/2)*qsum((f[i] - s[j,i,t] - x_[j,i,t]*(t+1)) for t in range(T) for i in range(K) for j in range(H))
                          , GRB.MINIMIZE)
        
        if add:
            m1.addConstr(f <= max_)
        
        m1.reset()
        m1.update()

        # solve P1:
        start = time.time()
        m1.optimize()
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P1 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj1: {m1.ObjVal}{utils.bcolors.ENDC}')
        
        max_new = 0
        for i in range(H):
            max_rel = 0
            sum_ = 0
            for j in range(K):
                if abs(np.rint(y[j,i].X)) == 1:
                    sum_ += 1
                    if max_rel < release_date[j,i]:
                        max_rel = release_date[j,i]
            temp = sum_*proc[0,i] + max_rel
            if temp > max_new:
                max_new = temp
                add = True
        if max_new < max_:
            max_ = max_new
            add = True
        else:
            add = False


        # pass results to second problem
        y_ = np.array(y.X)
        f_ = np.array(f.X)
        w_ = np.array(w.X)
        s_ = np.array(s.X)

        m2.setObjective(w_ + qsum(qsum(mama[j,t,i]*(f_[i] - s_[j,i,t] - x[j,i,t]*(t+1)) \
                           + lala[i,j]*x[j,i,t] for t in range(T)) - lala[i,j] * y_[i,j] * proc[i,j]\
                           for i in range(K) for j in range(H)) \
                           + (rho/2)*qsum((qsum(x[j,i,t] for t in range(K)) - proc[i,j]*y_[i,j]) for i in range(K) for j in range(H)) \
                           + (rho/2)*qsum((f_[i] - s_[j,i,t] - x[j,i,t]*(t+1)) for t in range(T) for i in range(K) for j in range(H))
                           , GRB.MINIMIZE) 
        m2.reset()
        m2.update()

        if step<3:
            m2.setParam('MIPGap', 0.12) # 5%
        else:
            m2.setParam('MIPGap', 0.0001)
        
        start = time.time()
        m2.optimize()
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P2 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj2: {m2.ObjVal}{utils.bcolors.ENDC}')
        
        obj1 += [m1.ObjVal]
        obj2 += [m2.ObjVal]

        # pass results to first problem
        x_ = np.array(x.X)


        # update dual variables
        for i in range(H):
            for j in range(K):
                lala[j,i] = lala[j,i] + rho*(sum([x[i,j,k].X for k in range(T)]) - y[j,i].X*proc[j,i])
                for t in range(T):
                    mama[i,t,j] = mama[i,t,j] + rho*(f[j].X - s[i,j,t].X - x[i,j,t].X*(t+1))
                    
        step = step + 1
        alpha = 1/math.sqrt(step+1)
        bhta = 1/math.sqrt(step+1)

        print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE: {w.X}{utils.bcolors.ENDC}')
        f_log.write((f'OPTIMAL VALUE: {w.X}\n'))
        ws.append(w[0].X)
        #print(f'{utils.bcolors.OKBLUE}Checking constraints{utils.bcolors.ENDC}')

        counter = 0
        total_counter = 0
        print(f'{utils.bcolors.OKBLUE}C8{utils.bcolors.ENDC}')
        for i in range(H):
            for j in range(K):
                for t in range(T):
                    if np.rint(f[j].X) < np.rint(x[i,j,t].X)*(t+1):
                        #print(f"{utils.bcolors.FAIL}Constraint 8 is violated expected larger than: {x[i,j,t].X*(t+1)} got:{f[i].X} {utils.bcolors.ENDC}")
                        counter += 1
                    total_counter += 1
        
        print(f"Violations 1 --- {counter} from {total_counter}")
        violations_1 += [(counter/total_counter)*100]

        counter = 0
        total_counter = 0
        #print(f'{utils.bcolors.OKBLUE}C1{utils.bcolors.ENDC}')
        for i in range(H):
            for j in range(K):
                temp = 0
                for t in range(T):
                    temp += np.rint(x[i,j,t].X)
                
                if temp < np.rint(y[j,i].X)*T:
                    #print(f"{utils.bcolors.FAIL}Constraint 1 is violated expected larger than: {y[j,i].X*proc[j,i]} got:{temp} {utils.bcolors.ENDC}")
                    counter += 1
                total_counter += 1
        
        #print(f'{utils.bcolors.OKBLUE}constraints violated: {counter}{utils.bcolors.ENDC}')
        violations_2 += [(counter/total_counter)*100]

        print(f"Violations 2 --- {counter} from {total_counter}")
        f_log.write("--------Machine allocation--------\n")

        for i in range(H):
            for k in range(T):
                at_least = 0
                for j in range(K):
                    if(np.rint(x[i,j,k].X) <= 0):
                        continue
                    else:
                        #print(f'{j+1}', end='\t')
                        f_log.write(f'{j+1}\t')
                        at_least += 1
                        break
                if(at_least == 0):
                    #print(f'0', end='\t')
                    f_log.write(f'0\t')
            #print('')
            f_log.write('\n')

        f_log.write("--------Completition time--------\n")
        cs = []
        reserved = [0 for i in range(H)]
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
            #print(f'C{i+1}: {C} - {my_machine}')
            f_log.write(f'C{i+1}: {C} - {my_super_machine} {y[i,:].X} - {f[i].X}\n')
            reserved[my_super_machine] += 1
                     
        f_log.write(f'max is: {max(cs)}\n')
        print(f'{utils.bcolors.OKBLUE}max is: {max(cs)}{utils.bcolors.ENDC}')
        #max_ = max(cs)         
        max_c.append(max(cs))
        #print(f'max is: {max(cs)}')
        f_log.write("check other constraints\n")
        violated = False
        for i in range(K): #for all jobs
            my_machine = -1
            for j in range(H):
                if y[i,j].X == 1:
                    my_machine = j
                    break
            for k in range(release_date[i,my_machine]):
                if x[my_machine,i,k].X == 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                    violated = True

        for i in range(K): #for all jobs
            if np.sum([y[i,j].X for j in range(H)]) != 1:
                #print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
                violated = True

        for j in range(H): #for all devices
            if np.sum([y[i,j].X for j in range(H)])*utils.max_memory_demand > memory_capacity[j]:
                #print(f"{utils.bcolors.FAIL}Constraint 4 is violated{utils.bcolors.ENDC}")
                violated = True

            occupied = reserved[j]*utils.max_memory_demand
            if occupied > memory_capacity[j]:
                #print(f"{utils.bcolors.FAIL}Constraint 4 is violated for machine {j}{utils.bcolors.ENDC}")
                violated = True
        
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
                    sum_ += np.rint(x[my_machine,i,k].X)
                if sum_ != 0 and sum_ != proc[i, my_machine] :
                    #print(f"{utils.bcolors.FAIL}Constraint 5 is violated {i+1}{utils.bcolors.ENDC}")
                    violated = True
                else:
                    if sum_ != 0:
                        at_least += 1
            if at_least == 0:
                #print(f"{utils.bcolors.FAIL}Constraint 5 is violated job not assigned {i+1}{utils.bcolors.ENDC}")
                violated = True
            if at_least > 1:
                #print(f"{utils.bcolors.FAIL}Constraint 5 is violated job assigned more tmes {i+1}{utils.bcolors.ENDC}")
                violated = True

        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                temp = 0
                for key in range(K):
                    temp += np.rint(x[j,key,t].X)
                if temp > 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                    violated = True

        if violated:
            f_log.write('VIOLATED\n')
            print(f'{utils.bcolors.OKBLUE}VIOLATED{utils.bcolors.ENDC}')
            #add = False
            accepted.append(0)
        else:
            f_log.write('OK\n')
            print(f'{utils.bcolors.OKBLUE}OK{utils.bcolors.ENDC}')
            #add = False
            accepted.append(100)
        
    f_log.close()
    ##print(f'optimal:  {ws}')
    #print(f'violations: {violations}')
    #for v in m2.getVars():
    #    print('%s %g' % (v.VarName, v.X))
    return (ws, violations_1, violations_2, max_c, accepted)
      
if __name__ == '__main__':
    run()