import numpy as np
from numpy import linalg as LA
import math
import random
import time
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum

import gurobi_for_subproblems as sub
import utils

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


K = 50 # number of data owners
H = 2 # number of compute nodes
utils.file_name = 'fully_symmetric.xlsx'


def run(release_date, proc, proc_local, trans_back, memory_capacity, memory_demand, 
        release_date_back=[], proc_back=[], proc_local_back=[], trans_back_gradients=[], 
        back_flag=False, filename=''):
    
    stable = 0
    T = np.max(release_date) + K*np.max(proc) # time intervals
    print(f"T = {T}")
    print(f" Memory: {memory_capacity}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    MAX_ITER = 10
    rho = 0.9
    DT = np.empty((0, 6))

    if filename != '':
        f_ = open(filename, "a")
        f_.write("Approach:\n")
    else:
        f_ = open('mytest', "w")


    m1 = gp.Model("xsubproblem")
    m2 = gp.Model("ysubproblem")


    # define variables -x subproblem
    x = m1.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    f = m1.addMVar(shape=(K), name="f")
    comp = m1.addMVar(shape=(K), name="comp") # completion time
    w = m1.addMVar(shape=(1), name="w")  # min of compl. times

    # define variables -y subproblem
    y = m2.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    comp_x_fixed = m2.addMVar(shape=(K), name="comp_x_fixed") # completion time
    w_x_fixed = m2.addMVar(shape=(1), name="w_x_fixed")  # min of compl. times
    
    
    # Auxiliary
    contr1_add_1 = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY,name="contr1_add_1")
    contr1_abs_1 = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY,name="contr1_abs_1")  # auxiliary for abs value

    contr2_add_1 = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY, name="contr2_add_1")
    contr2_abs_1 = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY,name="contr2_abs_1")   # auxiliary for abs value
    
    
    # "Parameters"
    x_par = np.zeros((H,K,T))
    y_par = np.zeros((K,H))
    if back_flag:
        T_back = np.max(release_date) + K*np.max(proc[0,:]) + np.max(release_date_back) + K*np.max(proc_back[0,:]) \
                        + np.max(proc_local) + np.max(proc_local_back) \
                        + np.max(np.max(trans_back)) + np.max(np.max(trans_back_gradients))

        z_par = np.zeros((H,K,T_back))

    # dual variables
    mu = np.zeros((K,H)) # dual variable
    
    
    
    

    start = time.time()
    # C3: each job is assigned to one and only machine
    m2.addConstr(y @ ones_H == ones_K)

    # C4: memory constraint
    m2.addConstr(memory_demand @ y<= memory_capacity)
    end = time.time()
    first_build = end-start

    start = time.time()
    # completition time definition
    m1.addConstrs(f[i] >= (t+1)*x[j, i, t] for i in range(K) for j in range(H) for t in range(T))

    #m2.addConstrs(comp_x_fixed[i] == qsum(trans_back[i,:]*y[i,:]) + f_par[i] + proc_local[i] for i in range(K))
    #max_constr_x_fixed = m2.addConstrs(w_x_fixed >= comp_x_fixed[i] for i in range(K))

    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    m1.addConstr(x[i,j,t] == 0)

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m1.addConstr( x[j,:,:].T @ ones_K <= ones_T )


    # forgoten constraint
    # C5: job should be processed entirely once
    for i in range(K):
        m1.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc[i,j] for j in range(H)) == 1)
        #m1.addConstr(qsum(qsum(x[j,i,t] for t in range(T)) for j in range(H)) >= min(proc[i,:]))
    end = time.time()
    
    second_build = end-start
    time_stamps = []

    if second_build > first_build:
        time_stamps.append(second_build)
    else:
        time_stamps.append(first_build)

    # Iterative algorithm
    my_ds2 = []
    my_ds = []
    obj_per_iter =[]
    violations = []
    
    flag = -1
    for iter in range(MAX_ITER):
        if flag == -2:
            break
        start = time.time()
        if iter >= 1:
            for d in my_ds:
                m1.remove(d)
            my_ds = []

        for i in range(K):
            for j in range(H):
                if iter >= 1:
                    c = m1.getConstrByName(f'const1add-{i}-{j}')
                    m1.remove(c)

                m1.addConstr(contr1_add_1[i,j] == qsum(x[j,i,t] for t in range(T)) - y_par[i,j]*proc[i,j] + mu[i,j], name=f'const1add-{i}-{j}')
                my_ds.append(m1.addConstr(contr1_abs_1[i,j] == gp.abs_(contr1_add_1[i,j]), name=f'const1ab-{i}-{j}'))

        for i in range(K):
            if iter >= 1:
                f11 = m1.getConstrByName(f'const1f-{i}')
                m1.remove(f11)
                

            m1.addConstr(comp[i] == np.sum(np.multiply(trans_back[i,:], y_par[i,:])) + f[i] + proc_local[i], name=f'const1f-{i}')
            my_ds.append(m1.addConstr(w >= comp[i], name=f'const1fw-{i}'))
            #m1.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc[i,j] for j in range(H)) == 1, name=f'constpij-{i}')

        #if iter >= 2:
            #m1.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc[i,j] for j in range(H)) == 1)

        m1.setObjective(w + (rho/2)*qsum(contr1_abs_1[i,j] for i in range(K) for j in range(H)), GRB.MINIMIZE)


        # m1.reset()
        m1.update()

        
        """
        if iter < 5:
            m1.setParam('MIPGap', 0.20)
        elif iter < 6:
            m1.setParam('MIPGap', 0.15)
        else:
            m1.setParam('MIPGap', 0.10)
        """
        if iter < 7:
            m1.setParam('MIPGap', 0.20)
        else:
            m1.setParam('MIPGap', 0.15)


        # solve P1:
        m1.optimize()
        end = time.time()
        time_stamps.append(end-start)
        print(f'{utils.bcolors.OKBLUE}P1 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj1: {m1.ObjVal}{utils.bcolors.ENDC}', iter)
        
        x_par = np.copy(np.array(x.X))
        # f_par = np.copy(np.array(f.X)) # f are variables and not calculated correctly
        np.copy(np.array(w.X))
        
        # Solve P2
        start = time.time()
        # Now calculate the value of (original) objective function at this iter
        g_values = []
        for i in range(K): #for all jobs
            g_interm = []
            for j in range(H): #for all machines
                for t in range(T): #for all timeslots
                    g_interm += [(t+1)*x[j,i,t].X]
            g_values += [max(g_interm)]

        f_par = np.copy(g_values)

        if iter >= 1:
            for dd in my_ds2:
                m2.remove(dd)
            my_ds2 = []

    
        ll = np.sum(x_par, axis=2)

        for i in range(K):
            for j in range(H):
                if iter >= 1:
                    c2 = m2.getConstrByName(f'const2add-{i}-{j}')
                    m2.remove(c2)

                m2.addConstr(contr2_add_1[i,j] == ll[j,i] - y[i,j]*proc[i,j] + mu[i,j], name=f'const2add-{i}-{j}')
                my_ds2.append(m2.addConstr(contr2_abs_1[i,j] == gp.abs_(contr2_add_1[i,j]), name=f'const2ab-{i}-{j}'))


        for i in range(K):
            if iter >= 1:
                f2 = m2.getConstrByName(f'const2f-{i}')
                m2.remove(f2)
                

            m2.addConstr(comp_x_fixed[i] == qsum(trans_back[i,:]*y[i,:]) + f_par[i] + proc_local[i], name=f'const2f-{i}')
            my_ds2.append(m2.addConstr(w_x_fixed >= comp_x_fixed[i], name=f'const2fw-{i}'))
        """
        if iter < 3:
            m2.setParam('MIPGap', 0.15) # 5%
        else:
            m2.setParam('MIPGap', 0.0001)
        """
        m2.setObjective(w_x_fixed + (rho/2)*qsum(contr2_abs_1[i,j] for i in range(K) for j in range(H)), GRB.MINIMIZE)

        #m2.setObjective((rho/2)*qsum(contr2_abs_1[i,j] for i in range(K) for j in range(H)), GRB.MINIMIZE)

        m2.update()
        m2.optimize()
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P2 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj2: {m2.ObjVal}{utils.bcolors.ENDC}', iter)
        time_stamps.append(end-start)
        
        #print(obj_per_iter)
        #print(y.X)

        aaa = LA.norm((np.array(y.X)-np.copy(y_par)), 'fro')**2
        changes_y = 0
        for j in range(K):
            # print(np.abs(np.rint(np.array(y.X)[j, :])) != np.abs(np.rint(y_par[j, :])))
            if np.any(np.abs(np.rint(np.array(y.X)[j, :])) != np.abs(np.rint(y_par[j, :]))):
                changes_y += 1

        y_par = np.copy(np.array(y.X))

        if changes_y <= 0:
            stable += 1
        else:
            stable = 0

        # call sub-problems
        if iter == 1:#stable >3 or iter == MAX_ITER - 1:
            flag = -2
            print(f'{utils.bcolors.OKBLUE}Calling the subproblems {iter}{utils.bcolors.ENDC}')
            # Call subproblem
            do_it = False
            x_ = np.zeros((H,K,T))
            x_par = np.zeros((H,K,T))
            y_ = y_par
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
                x__ = sub.for_each_machine(len(Kx), release_datex, procx, proc_localx, trans_backx, memory_capacity[i], Tx)
                end_sub = time.time()

                machine_time = end_sub-start_sub

                if back_flag:
                    f_temp = np.zeros((len(Kx)))

                    for kk in range(len(Kx)):
                        for t in range(Tx):
                            if f_temp[kk] < (t+1)*x__[0,kk,t]:
                                f_temp[kk] = (t+1)*x__[0,kk,t]
                    
                    min_f = min(f_temp)
                    
                    procz = np.copy(proc_back[Kx, i])  # this is a row-vector
                    release_datez = np.copy(release_date_back[Kx, i])
                    proc_localz = np.copy(proc_local_back[Kx])
                    trans_backz = np.copy(trans_back_gradients[Kx, i])

                    for kk in range(len(Kx)):
                        release_datez[kk] += (f_temp[kk] - min_f) + proc_localx[kk] + trans_backx[kk]
                    
                    Tz = np.max(release_datez) + len(Kx)*np.max(procz)  # to constrain the T

                    start_sub = time.time()
                    z__ = sub.for_each_machine(len(Kx), release_datez, procz, proc_localz, trans_backz, memory_capacity[i], Tz)
                    end_sub = time.time()

                    machine_time += end_sub - start_sub

                    jj = 0
                    for j in Kx:
                        for t in range(Tz):
                            
                            if int(min_f)+t >= T_back:
                                break

                            z_par[i,j,int(min_f)+t] = z__[0,jj,t]
                        jj += 1
                
                all_time.append(machine_time)
                
                jj = 0
                for j in Kx:
                    for t in range(Tx):
                        x_par[i,j,t] = x__[0,jj,t]
                    jj += 1
                    

            print(f'Parallel machines longest time {max(all_time)}')
            time_stamps.append(max(all_time))

        # END OF IF

        # update dual variables
        temp_mu = np.zeros((K,H))  # just making sure I don't make a mistake
        for j in range(H):
            for i in range(K):
                temp_sum = []
                for t in range(T):
                    temp_sum += [x[j,i,t].X]
                temp_mu[i,j] = np.copy(mu[i,j] + (sum(temp_sum)-(y[i,j].X*proc[i,j])))

        # diff_mu = LA.norm((mu-temp_mu), 'fro')**2

        mu = np.copy(temp_mu)

        # Calculate original objective function:

        calc_obj = np.zeros(K)
        temptemp = np.multiply(y.X, trans_back)
        for i in range(K):
            calc_obj[i] = g_values[i] + np.sum(temptemp[i,:]) + proc_local[i]
        # print(w_x_fixed.X)
        
        if flag != -2:
            obj_per_iter += [max(calc_obj)]

        violated_constraints = 0
        total_constraints = 0
        for j in range(H):
            for i in range(K):
                if np.all(np.abs(np.rint(ll[j,i])) != proc[i,j]*np.abs(np.rint(y_par[i,j]))):
                    violated_constraints += 1
                    #print(np.abs(np.rint(ll[j,i])), proc[i,j]*np.abs(np.rint(y_par[i,j])))
                total_constraints += 1
        
        violations.append((violated_constraints/total_constraints)*100)

        primal_residual = LA.norm((ll.T - np.multiply(y_par, proc)), 'fro')**2
    
        print("-----------------------------------------------------objective equals to:", iter, max(calc_obj), changes_y, aaa, violated_constraints, primal_residual)
        #DT = np.append(DT, np.array([[iter, max(calc_obj), changes_y, aaa, violated_constraints, primal_residual]]), axis=0)

        
        f_.write("--------Machine allocation--------\n")

        for i in range(H):
            for k in range(T):
                at_least = 0
                for j in range(K):
                    if(np.rint(x_par[i,j,k]) <= 0):
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
        cs_back = []
        reserved = [0 for i in range(H)]
        f_m = np.zeros(K)
        for i in range(K): #for all jobs
            my_machine = 0
            my_super_machine = 0
            last_zero = -1
            for my_machine in range(H):
                for k in range(T):
                    if np.rint(x_par[my_machine,i,k]) >= 1:
                        if last_zero < k+1:
                            last_zero = k+1
                            my_super_machine = my_machine
            fmax = last_zero
            f_m[i] = fmax
            C = fmax + proc_local[i] + trans_back[i,my_machine]
            cs.append(C)
            #print(f'C{i+1}: {C} - {my_machine}')
            f_.write(f'C{i+1}: {C} - {my_super_machine} {y[i,:].X} - {f[i].X}\n')
            reserved[my_super_machine] += 1
        
        f_.write(f'max is: {max(cs)}\n')
        print(f'max is: {max(cs)}')

        if back_flag and flag == -2:
            for i in range(K): #for all jobs
                my_machine = 0
                my_super_machine = 0
                last_zero = -1
                for my_machine in range(H):
                    for k in range(T_back):
                        if np.rint(z_par[my_machine,i,k]) >= 1:
                            if last_zero < k+1:
                                last_zero = k+1
                                my_super_machine = my_machine
                fmax = last_zero
                C = fmax + proc_local_back[i] + trans_back_gradients[i,my_machine]
                cs_back.append(C)

            print(f'BACK max is: {max(cs_back)}')
       
        if flag == -2:
            if back_flag:
                obj_per_iter += [max(cs_back)]
            else:
                obj_per_iter += [max(cs)]

        f_.write("check other constraints\n")
        violated = False
        for i in range(K): #for all jobs
            my_machine = -1
            for j in range(H):
                if y[i,j].X == 1:
                    my_machine = j
                    break
            for k in range(release_date[i,my_machine]):
                if x_par[my_machine,i,k] == 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                    violated = True
            
            if back_flag:
                for k in range(int(release_date_back[i,my_machine] + trans_back[i,my_machine] + f_m[i])):
                    if z_par[my_machine,i,k] == 1:
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
                    sum_ += np.rint(x_par[my_machine,i,k])
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
            
            if back_flag:
                sum_z = 0
                for k in range(T_back):
                    sum_z += np.rint(z_par[my_machine,i,k])
                if sum_z != 0 and sum_ != proc_back[i, my_machine] :
                    #print(f"{utils.bcolors.FAIL}Constraint 5 is violated {i+1}{utils.bcolors.ENDC}")
                    violated = True
            

        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                temp = 0
                for key in range(K):
                    temp += np.rint(x_par[j,key,t])
                if temp > 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                    violated = True
            
            if back_flag:
                for t in range(T_back): #for all timeslots
                    temp = 0
                    for key in range(K):
                        temp += np.rint(z_par[j,key,t])
                    if temp > 1:
                        #print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                        violated = True
        
        if violated:
            f_.write('VIOLATED\n')
        else:
            f_.write('OK\n')
    
    f_.close()

    total_time = 0
    for t in time_stamps:
        print(t)
        total_time += t

    print(f"{utils.bcolors.FAIL}Total time {total_time}{utils.bcolors.ENDC}")

    return(violations, obj_per_iter)
    


