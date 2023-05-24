import numpy as np
import math

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum

from numpy import linalg as LA


import utils
import time

import warnings
warnings.filterwarnings("ignore")

K = 50 # number of data owners
H = 2 # number of compute nodes
utils.file_name = 'fully_symmetric.xlsx'

def run(filename='', testcase='fully_heterogeneous'):
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

    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    if utils.file_name != 'fully_symmetric.xlsx':
        if H == 2:
            if K == 50:
                memory_capacity = np.array([30,120])
            else:
                memory_capacity = np.array([105,195])

        if H == 5:
            if K == 50:
                memory_capacity = np.array([63, 48,  9, 21, 24])
    
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

    m1 = gp.Model("cont_prob")
    m2 = gp.Model("int_prob")

    rho = 1
    
    # define variables - problem 1

    x_cont = m1.addMVar(shape = (H,K,T), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x_cont")
    y_cont = m1.addMVar(shape=(K,H), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y_cont")
    f_cont = m1.addMVar(shape=(K), lb=0, vtype=GRB.CONTINUOUS, name="f_cont")
    w_cont = m1.addMVar(shape=(1),lb=0, vtype=GRB.CONTINUOUS, name="w_cont")

    contr1_add_x = m1.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_x")
    contr1_add_y = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_y")
    contr1_add_f = m1.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_f")
    contr1_add_w = m1.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_w")
    contr1_abs_x = m1.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_w")
    contr1_abs_y = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_y")
    contr1_abs_f = m1.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_f")
    contr1_abs_w = m1.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_w")

    # define variables - problem 2

    x_int = m2.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x_int")
    y_int = m2.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y_int")
    f_int = m2.addMVar(shape=(K), vtype=GRB.INTEGER, name="f_int")
    w_int = m2.addMVar(shape=(1),vtype=GRB.INTEGER, name="w_int")

    contr2_add_x = m2.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_x")
    contr2_add_y = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_y")
    contr2_add_f = m2.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_f")
    contr2_add_w = m2.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_w")
    contr2_abs_x = m2.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_x")
    contr2_abs_y = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_y")
    contr2_abs_f = m2.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_f")
    contr2_abs_w = m2.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_w")
    

    x_int_par = np.zeros((H,K,T))
    x_cont_par = np.zeros((H,K,T))
    
    y_int_par = np.zeros((K,H))
    y_cont_par = np.zeros((K,H))

    f_int_par = np.zeros(K)
    f_cont_par = np.zeros(K)

    w_int_par = 0
    w_cont_par = 0

    # dual variables

    x_dual = np.zeros((H,K,T))
    y_dual = np.zeros((K,H))
    f_dual = np.zeros((K))
    w_dual = 0
    
    
    # K1 - C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    m1.addConstr(x_cont[i,j,t] == 0)
    
    # K2 - C3: all jobs interval are assigned to one only machine
    m1.addConstr( y_cont @ ones_H == ones_K )
    
    # K3 - C4: memory constraint
    m1.addConstr((y_cont.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    # K4 - C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m1.addConstr( x_cont[j,:,:].T @ ones_K <= ones_T )
    
    # K5 - C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m1.addConstr( qsum(x_cont[j,i,:]) == y_cont[i,j]*proc[i,j])
    '''
    # K6
    for j in range(H): #for all machines
        for i in range(K):
            m1.addConstrs( f_cont[i] >= (t+1)*x_cont[j,i,t] for t in range(T))
    
    # K7
    for i in range(K):
         m1.addConstrs( w_cont >= qsum(trans_back[i,:] * y_cont[i,:]) + f_cont[i] + proc_local[i] for i in range(K))
    '''
    
    # SECOND PROM CONSTRAINTS
    
    # K1 - C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    m2.addConstr(x_int[i,j,t] == 0)
    
    # K2 - C3: all jobs interval are assigned to one only machine
    m2.addConstr( y_int @ ones_H == ones_K )
    
    # K3 - C4: memory constraint
    m2.addConstr((y_int.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    # K4 - C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m2.addConstr( x_int[j,:,:].T @ ones_K <= ones_T )
    
    # K5 - C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m2.addConstr( qsum(x_int[j,i,:]) == y_int[i,j]*proc[i,j])
    
    # K6 
    for j in range(H): #for all machines
        for i in range(K):
            m2.addConstrs( f_int[i] >= (t+1)*x_int[j,i,t] for t in range(T))
    # K7
    for i in range(K):
         m2.addConstrs( w_int >= qsum(trans_back[i,:] * y_int[i,:]) + f_int[i] + proc_local[i] for i in range(K))
    
    
    max_c = []
    ws = []
    my_ds1 = []
    my_ds2 = []
    step = 0
    while step<20:
        print(f"{utils.bcolors.OKBLUE}-------------{step}------------{utils.bcolors.ENDC}")
        f_log.write(f"-------------{step}------------\n")

        if step>=1:
            for d in my_ds1:
                m1.remove(d)
            my_ds1 = []

            for d in my_ds2:
                m2.remove(d)
            my_ds2 = []
        
        # objective function
        my_ds1.append(m1.addConstr(contr1_add_w == w_cont - w_int_par + w_dual))
        my_ds1.append(m1.addConstr(contr1_abs_w == gp.abs_(contr1_add_w)))
        for j in range(K):
            my_ds1.append(m1.addConstr(contr1_add_f[j] == f_cont[j] - f_int_par[j] + f_dual[j]))
            my_ds1.append(m1.addConstr(contr1_abs_f[j] == gp.abs_(contr1_add_f[j])))
            for i in range(H):
                my_ds1.append(m1.addConstr(contr1_add_y[j,i] == y_cont[j,i] - y_int_par[j,i] + y_dual[j,i]))
                my_ds1.append(m1.addConstr(contr1_abs_y[j,i] == gp.abs_(contr1_add_y[j,i])))
                for t in range(T):
                    my_ds1.append(m1.addConstr(contr1_add_x[i,j,t] == x_cont[i,j,t] - x_int_par[i,j,t] + x_dual[i,j,t]))
                    my_ds1.append(m1.addConstr(contr1_abs_x[i,j,t] == gp.abs_(contr1_add_x[i,j,t])))
        
        m1.setObjective(w_cont +  \
                          + (rho/2)*qsum(contr1_abs_w)  \
                          + (rho/2)*qsum(contr1_abs_f[j] for j in range(K))  \
                          + (rho/2)*qsum(contr1_abs_y[j,i] for j in range(K) for i in range(H))  \
                          + (rho/2)*qsum(contr1_abs_x[i,j,t] for t in range(T) for j in range(K) for i in range(H))
                          , GRB.MINIMIZE)
        
        m1.reset()
        m1.update()
        #m1.setParam('MIPGap', 0.20) # 5%
        start = time.time()
        m1.optimize()
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P1 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj1: {m1.ObjVal}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE} {w_cont.X} {utils.bcolors.ENDC}')

        
        # pass variables
        x_cont_par = np.copy(np.array(x_cont.X))
        y_cont_par = np.copy(np.array(y_cont.X))
        f_cont_par = np.copy(np.array(f_cont.X))
        w_cont_par = w_cont.X
        print('--------------- SHOW 1 ---------------')
        print(f_cont.X)
        print(y_cont_par)

        # objective function
        my_ds2.append(m2.addConstr(contr2_add_w == w_cont_par - w_int + w_dual))
        my_ds2.append(m2.addConstr(contr2_abs_w == gp.abs_(contr2_add_w)))
        for j in range(K):
            my_ds2.append(m2.addConstr(contr2_add_f[j] == f_cont_par[j] - f_int[j] + f_dual[j]))
            my_ds2.append(m2.addConstr(contr2_abs_f[j] == gp.abs_(contr2_add_f[j])))
            for i in range(H):
                my_ds2.append(m2.addConstr(contr2_add_y[j,i] == y_cont_par[j,i] - y_int[j,i] + y_dual[j,i]))
                my_ds2.append(m2.addConstr(contr2_abs_y[j,i] == gp.abs_(contr2_add_y[j,i])))
                for t in range(T):
                    my_ds2.append(m2.addConstr(contr2_add_x[i,j,t] == x_cont_par[i,j,t] - x_int[i,j,t] + x_dual[i,j,t]))
                    my_ds2.append(m2.addConstr(contr2_abs_x[i,j,t] == gp.abs_(contr2_add_x[i,j,t])))

        m2.setObjective(#w_int +  \
                          (rho/2)*qsum(contr2_abs_w)  \
                          + (rho/2)*qsum(contr2_abs_f[i] for i in range(K))  \
                          + (rho/2)*qsum(contr2_abs_y[i,j] for i in range(K) for j in range(H))  \
                          + (rho/2)*qsum(contr2_abs_x[j,i,t] for t in range(T) for i in range(K) for j in range(H))
                          , GRB.MINIMIZE)

        #m2.setParam('MIPGap', 0.30) # 5%
        '''
        if step == 3:
            # K6 
            for j in range(H): #for all machines
                for i in range(K):
                    m2.addConstrs( f_int[i] >= (t+1)*x_int[j,i,t] for t in range(T))
            # K7
            for i in range(K):
                m2.addConstrs( w_int >= qsum(trans_back[i,:] * y_int[i,:]) + f_int[i] + proc_local[i] for i in range(K))
            #m2.setParam('MIPGap', 0.30) 
        '''
        m2.reset()
        m2.update()

        start = time.time()
        m2.optimize()
        end = time.time()
        print(f'{utils.bcolors.OKBLUE}P2 took: {end-start}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}Obj2: {m2.ObjVal}{utils.bcolors.ENDC}')

        # pass variables
        x_int_par = np.copy(np.array(x_int.X))
        y_int_par = np.copy(np.array(y_int.X))
        f_int_par = np.copy(np.array(f_int.X))
        w_int_par = w_int.X

        print('--------------- SHOW 2 ---------------')
        print(f_int.X)
        print(y_int.X)

        # update dual variables
        w_dual = w_dual + np.abs(w_cont.X - w_int.X)
        for j in range(K):
            f_dual[j] = f_dual[j] + np.abs(f_cont[j].X - f_int[j].X) 
            for i in range(H):
                y_dual[j,i] = y_dual[j,i] + np.abs(y_cont[j,i].X - y_int[j,i].X)
                for t in range(T):
                    x_dual[i,j,t] = x_dual[i,j,t] + np.abs(x_cont[i,j,t].X - x_int[i,j,t].X)

        print('--------------- DUAL VARIABLES ---------------')
        print(w_dual)
        print(f_dual)
        print(y_dual)

        step = step + 1

        print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE: {w_int.X}{utils.bcolors.ENDC}')
        f_log.write((f'OPTIMAL VALUE: {w_int.X}\n'))
        ws.append(w_int[0].X)

        # count violations?

        

        f_log.write("--------Machine allocation--------\n")
        for i in range(H):
            for k in range(T):
                at_least = 0
                for j in range(K):
                    if(np.rint(x_int[i,j,k].X) <= 0):
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
                    if np.rint(x_int[my_machine,i,k].X) >= 1:
                        if last_zero < k+1:
                            last_zero = k+1
                            my_super_machine = my_machine
            fmax = last_zero
            #f_int_par[i] = last_zero
            C = fmax + proc_local[i] + trans_back[i,my_machine]
            cs.append(C)
            #print(f'C{i+1}: {C} - {my_machine}')
            f_log.write(f'C{i+1}: {C} - {my_super_machine} {y_int[i,:].X} - {f_int[i].X}\n')
            reserved[my_super_machine] += 1
        
        #max_c.append(max(cs))
        w_int_par =  max(cs)      
        f_log.write(f'max is: {max(cs)}\n')
        print(f'{utils.bcolors.OKBLUE}max is: {max(cs)}{utils.bcolors.ENDC}')

        f_log.write("check other constraints\n")
        violated = False
        for i in range(K): #for all jobs
            my_machine = -1
            for j in range(H):
                if y_int[i,j].X == 1:
                    my_machine = j
                    break
            for k in range(release_date[i,my_machine]):
                if x_int[my_machine,i,k].X == 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                    violated = True

        for i in range(K): #for all jobs
            if np.sum([y_int[i,j].X for j in range(H)]) != 1:
                #print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
                violated = True

        for j in range(H): #for all devices
            if np.sum([y_int[i,j].X for j in range(H)])*utils.max_memory_demand > memory_capacity[j]:
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
                    sum_ += np.rint(x_int[my_machine,i,k].X)
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
                    temp += np.rint(x_int[j,key,t].X)
                if temp > 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                    violated = True

        if violated:
            f_log.write('VIOLATED\n')
            print(f'{utils.bcolors.OKBLUE}VIOLATED{utils.bcolors.ENDC}')
        else:
            f_log.write('OK\n')
            print(f'{utils.bcolors.OKBLUE}OK{utils.bcolors.ENDC}')


    f_log.close()
    print(max_c)
    return #(ws, violations_1, violations_2, max_c, accepted)

if __name__ == '__main__':
    run()