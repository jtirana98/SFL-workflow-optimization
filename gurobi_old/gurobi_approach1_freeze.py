import numpy as np
import math

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import time

import utils
import gurobi_for_subproblems as sub


import warnings
warnings.filterwarnings("ignore")

K = 10 # number of data owners
H = 5 # number of compute nodes
utils.file_name = 'fully_symmetric.xlsx'

def run(filename='', testcase='fully_symmetric'):
    #fully_heterogeneous
    #fully_symmetric
    start = time.time()
    if testcase == 'fully_symmetric':
        utils.file_name = 'fully_symmetric.xlsx'
    else:
        utils.file_name = 'fully_heterogeneous.xlsx'
        

    release_date = np.array(utils.get_fwd_release_delays(K,H))
    proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local = np.array(utils.get_fwd_end_local(K))
    trans_back = np.array(utils.get_trans_back(K, H))
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    if utils.file_name != 'fully_symmetric.xlsx':
        if H == 2:
            if K == 50:
                memory_capacity = np.array([120,120])
            else:
                memory_capacity = np.array([270,270])

        if H == 5:
            if K == 50:
                memory_capacity = np.array([21, 150,  150, 21, 150])
            if K == 100:
                memory_capacity = np.array([21, 300, 300,  21,  300])
    
    if filename != '':
        f_ = open(filename, "a")
        f_.write("Approach:\n")
    else:
        f_ = open('mytest', "w")

    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")
    print(memory_capacity)
    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m1 = gp.Model("relax_approach_1_p1")
    m2 = gp.Model("relax_approach_1_p2")

    
    # define variables - problem 1
    
    y = m1.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    f = m1.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    w = m1.addMVar(shape=(1),vtype=GRB.INTEGER, name="w")
    comp = m1.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")

    # define variables - problem 2
    x = m2.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")

    # dual variables

    lala = np.zeros((K,H)) # lamda variable
    mama = np.ones((H,T,K)) # m variable
    #lala = np.random.normal(0,4, size=(K,H))
    #mama = np.random.normal(0,4, size=(H,T,K))

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
    violations_3 = []
    max_c = []
    accepted = []
    ws = []
    obj1 = []
    obj2 = []
    max_ = T
    add = False
    y_history = {}
    w_prev = -10
    stable = 0
    do_it = True
    time_stamps = []
    first_problem = 0
    while step<10:
        
        if step >=1:
            start = time.time()

        print(f"{utils.bcolors.OKBLUE}-------------{step}------------{utils.bcolors.ENDC}")
        m1.setObjective(w + qsum(lala[i,j] * y[i,j] * proc[i,j] - qsum(f[i]*mama[j,t,i] for t in range(T)) for i in range(K) for j in range(H)) , GRB.MINIMIZE)    
        
        
        if step >= 1 and add:
            m1.addConstr(f <= max_)

        m1.update()
        
        
        if step == 0:
            end = time.time()
            time_stamps.append(end-start)
            start = time.time()

        #print(f"{utils.bcolors.OKBLUE}-------------{step}------------{utils.bcolors.ENDC}")
        f_.write(f"-------------{step}------------\n")
        
        # solve P1:
        
        m1.optimize()
        end = time.time()
        #time_stamps.append(end-start)
        first_problem = end-start

        

        if w[0].X == w_prev:
            stable += 1
            print(f'IT IS THE SAME {stable}')
        else:
            print(f'NOT THE SAME {w_prev}')
            stable = 0

        w_prev = int(w[0].X)


        print(f'{utils.bcolors.OKBLUE}P1 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj1: {m1.ObjVal}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}W: {w[0].X}{utils.bcolors.ENDC}')

        print("-------------------------------")

        ws.append(w[0].X)
        if stable >= 3:
            print(f'{utils.bcolors.OKBLUE}Calling the subproblems{utils.bcolors.ENDC}')
            # Call subproblem
            do_it = False
            x_ = np.zeros((H,K,T))
            print(w_prev)
            
            y_ = y_history[w_prev]
            all_time = []
            for i in range(H):
                Kx = list(np.transpose(np.argwhere(y_[:,i]==1))[0]) # finds which data owners are assigned to the machine i
                print(Kx)
                if len(Kx) == 0:
                    continue
                
                procx = np.copy(proc[Kx, i])  # this is a row-vector
                
                release_datex = np.copy(release_date[Kx, i])
                proc_localx = np.copy(proc_local[Kx])
                trans_backx = np.copy(trans_back[Kx, i])
                Tx = np.max(release_datex) + len(Kx)*np.max(procx)  # to constrain the T
                start_sub = time.time()
                x__ = sub.for_each_machine(len(Kx), release_datex, proc[Kx,i], proc_localx, trans_backx, memory_capacity[i], Tx)
                end_sub = time.time()
                all_time.append(end_sub-start_sub)
                
                jj = 0
                for j in Kx:
                    for t in range(Tx):
                        x_[i,j,t] = x__[0,jj,t]
                    jj += 1
            print(f'Parallel machines longest time {max(all_time)}')
            time_stamps.append(max(all_time))


            f_.write("--------Machine allocation From subproblems--------\n")

            for i in range(H):
                for k in range(T):
                    at_least = 0
                    for j in range(K):
                        if(np.rint(x_[i,j,k]) <= 0):
                            continue
                        else:
                            #print(f'{j+1}', end='\t')
                            f_.write(f'{j+1}\t')
                            at_least += 1
                            break
                    if(at_least == 0):
                        #print(f'0', end='\t')
                        f_.write(f'0\t')
                #print('')
                f_.write('\n')

            f_.write("--------Completition time from sub problems--------\n")
            cs = []
            reserved = [0 for i in range(H)]
            for i in range(K): #for all jobs
                my_machine = 0
                my_super_machine = 0
                last_zero = -1
                for my_machine in range(H):
                    for k in range(T):
                        if np.rint(x_[my_machine,i,k]) >= 1:
                            if last_zero < k+1:
                                last_zero = k+1
                                my_super_machine = my_machine
                fmax = last_zero
                C = fmax + proc_local[i] + trans_back[i,my_machine]
                cs.append(C)
                #print(f'C{i+1}: {C} - {my_machine}')
                f_.write(f'C{i+1}: {C} - {my_super_machine}\n')
                reserved[my_super_machine] += 1
            
            f_.write(f'max is: {max(cs)}\n')
            #max_ = max(cs)         
            max_c.append(max(cs))
            print(f'FINAL max is: {max(cs)}')
            f_.write("check other constraints\n")
            violated = False
            for i in range(K): #for all jobs
                my_machine = -1
                for j in range(H):
                    if y_[i,j] == 1:
                        my_machine = j
                        break
                for k in range(release_date[i,my_machine]):
                    if x_[my_machine,i,k] == 1:
                        #print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                        violated = True

            for i in range(K): #for all jobs
                if np.sum([y_[i,j] for j in range(H)]) != 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
                    violated = True

            for j in range(H): #for all devices
                if np.sum([y_[i,j] for j in range(H)])*utils.max_memory_demand > memory_capacity[j]:
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
                        sum_ += np.rint(x_[my_machine,i,k])
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
                        temp += np.rint(x_[j,key,t])
                    if temp > 1:
                        #print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                        violated = True
            
            if violated:
                f_.write('VIOLATED\n')
                #add = False
                accepted.append(0)
            else:
                f_.write('OK\n')
                #add = False
                accepted.append(1)
            
            violations_1.append(0)
            violations_2.append(0)
            violations_3.append(0)

            print('TIME PERIODS')
            total = 0
            for t in time_stamps:
                print(f'{t}', end='\t')
                total += t
            print(f'\nTotal time {total}')

            print('Violations 1')
            for v in violations_3:
                print(f'{v}', end='\t')
            print('')

            print('Violations 2')
            for v in violations_2:
                print(f'{v}', end='\t')
            print('')

            return (ws, violations_1, violations_2, violations_3, max_c, accepted)
        
        if do_it:
            max_new = 0
            maxCC = 0
            temp_f = [0 for i in range(K)]
            for i in range(H):
                max_rel = 0
                sum_ = 0
                list_K = []
                for j in range(K):
                    if abs(np.rint(y[j,i].X)) == 1:
                        sum_ += 1
                        list_K.append(j)
                        if max_rel < release_date[j,i]:
                            max_rel = release_date[j,i]
                            
                temp = sum_*proc[0,i] + max_rel
                
                for j in list_K:
                    temp_f[j] = temp

                if temp > max_new:
                    max_new = temp

                    temp_C = 0
                    for j in range(K):
                        if temp_C < temp_f[j] + y[j,i].X*trans_back[j,i] + proc_local[j]:
                            temp_C = temp_f[j] + y[j,i].X*trans_back[j,i] + proc_local[j]
                    
                    if maxCC < temp_C:
                        maxCC = temp_C
            
            if max_new < max_:
                max_ = max_new
                add = True

                # store the history
                y_temp = np.copy(np.array(y.X))
                y_history[maxCC] = y_temp
                print(f'ADDING NEW ENTRY {maxCC}')
                print(maxCC)
            else:
                add = False
            
            
            if step<3:
                m2.setParam('MIPGap', 0.532) # 5%
            else:
                m2.setParam('MIPGap', 0.256)



            start = time.time()

           # if step == 2:
            #    for i in range(K):
            #        m2.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc[i,j] for j in range(H)) == 1)


            m2.setObjective(qsum(x[i,j,t]*(mama[i,t,j]*(t+1) - lala[j,i]) for i in range(H) for j in range(K) for t in range(T)), GRB.MINIMIZE)
            
            #if step == 0:
            
            #else:
            #    m2.setParam('MIPGap', 0.267)
            #m2.setParam('MIPGap', 0.123) # 5%
            m2.update()


            
            m2.optimize()
            end = time.time()
            
            second_problem = end-start
            if second_problem > first_problem:
                time_stamps.append(second_problem)
            else:
                time_stamps.append(first_problem)
            
            print(f'{utils.bcolors.OKBLUE}P2 took: {end-start}{utils.bcolors.ENDC}')
            print(f'{utils.bcolors.OKBLUE}Obj2: {m2.ObjVal}{utils.bcolors.ENDC}')
            
            obj1 += [m1.ObjVal]
            obj2 += [m2.ObjVal]
            
            #print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE: {w.X}{utils.bcolors.ENDC}')
            f_.write((f'OPTIMAL VALUE: {w.X}\n'))

            
            #print(f'{utils.bcolors.OKBLUE}Checking constraints{utils.bcolors.ENDC}')

            counter = 0
            total_counter = 0
            #print(f'{utils.bcolors.OKBLUE}C8{utils.bcolors.ENDC}')
            for j in range(K):
                for i in range(H):
                    for t in range(T):
                        if abs(np.int(f[j].X)) < abs(np.int(x[i,j,t].X))*(t+1):
                            #print(f"{utils.bcolors.FAIL}Constraint 8 is violated expected larger than: {x[i,j,t].X*(t+1)} got:{f[i].X} {utils.bcolors.ENDC}")
                            counter += 1
                        #else:
                        #print(f"{utils.bcolors.OKBLUE}OKK expected larger than: {abs(np.rint(x[i,j,t].X))*(t+1)} got:{f[i].X} {utils.bcolors.ENDC}")
                        total_counter += 1
            print(f"Violations 1 --- {counter} from {total_counter}")
            violations_1 += [(counter/total_counter)*100]
            #print(f'{utils.bcolors.OKBLUE}C1{utils.bcolors.ENDC}')

            counter = 0
            total_counter = 0
            for j in range(K):
                max_tempp = 0
                for i in range(H):
                    for t in range(T):
                        if max_tempp <  abs(np.int(x[i,j,t].X))*(t+1):
                            max_tempp = abs(np.int(x[i,j,t].X))*(t+1)
                        
                if abs(np.int(f[j].X)) < max_tempp:
                    counter += 1
                
                total_counter += 1

            print(f"Violations 1-max --- {counter} from {total_counter}")
            print(f"pososto: {(counter/total_counter)*100}")
            violations_3 += [(counter/total_counter)*100]
            
            counter = 0
            total_counter = 0
            for i in range(H):
                for j in range(K):
                    temp = 0
                    for t in range(T):
                        temp += np.rint(x[i,j,t].X)
                    
                    if temp < abs(np.rint(y[j,i].X))*proc[j,i]:
                        #print(f"{utils.bcolors.FAIL}Constraint 1 is violated expected larger than: {y[j,i].X*proc[j,i]} got:{temp} {utils.bcolors.ENDC}")
                        counter += 1
                    total_counter += 1

            violations_2 += [(counter/total_counter)*100]
            #print(f'{utils.bcolors.OKBLUE}constraints violated: {counter}{utils.bcolors.ENDC}')
            print(f"Violations 2 --- {counter} from {total_counter}")


            # update dual variables
            alpha = violations_2[-1]/100  #violations_2[-1]/100 #1/math.sqrt(step+1)
            bhta = violations_3[-1]/100 #violations_3[-1]/100
            
            for i in range(H):
                for j in range(K):
                    lala[j,i] = max(lala[j,i] + alpha*(abs(y[j,i].X)*proc[j,i] - sum([abs(x[i,j,k].X) for k in range(T)])), 0)
                    for t in range(T):
                        mama[i,t,j] = max(mama[i,t,j] + bhta*(abs(x[i,j,t].X)*(t+1) - abs(f[j].X)), 0)
            
            step = step + 1
        
            f_.write("--------Machine allocation--------\n")

            for i in range(H):
                for k in range(T):
                    at_least = 0
                    for j in range(K):
                        if(np.rint(x[i,j,k].X) <= 0):
                            continue
                        else:
                            #print(f'{j+1}', end='\t')
                            f_.write(f'{j+1}\t')
                            at_least += 1
                            break
                    if(at_least == 0):
                        #print(f'0', end='\t')
                        f_.write(f'0\t')
                #print('')
                f_.write('\n')

            f_.write("--------Completition time--------\n")
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
                f_.write(f'C{i+1}: {C} - {my_super_machine} {y[i,:].X} - {f[i].X}\n')
                reserved[my_super_machine] += 1
            
            f_.write(f'max is: {max(cs)}\n')
            #max_ = max(cs)         
            max_c.append(max(cs))
            #print(f'max is: {max(cs)}')
            f_.write("check other constraints\n")
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
                f_.write('VIOLATED\n')
                #add = False
                accepted.append(0)
            else:
                f_.write('OK\n')
                #add = False
                accepted.append(1)
        
    f_.close()
    ##print(f'optimal:  {ws}')
    #print(f'violations: {violations}')
    #for v in m2.getVars():
    #    print('%s %g' % (v.VarName, v.X))
    return (ws, violations_1, violations_2, violations_3, max_c, accepted)
if __name__ == '__main__':
    run()