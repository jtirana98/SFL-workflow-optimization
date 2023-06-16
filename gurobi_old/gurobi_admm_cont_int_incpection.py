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
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

K = 50 # number of data owners
H = 2 # number of compute nodes
utils.file_name = 'fully_symmetric.xlsx'

def run(filename='', testcase='fully_heterogeneous'):
    #fully_heterogeneous
    #fully_symmetric
    rho = 2
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
                memory_capacity = np.array([3,195])

        if H == 5:
            if K == 50:
                memory_capacity = np.array([63, 48,  9, 21, 24])
    '''
    if filename != '':
        f_log = open(filename, "a")
        f_log.write("Approach:\n")
    else:
        f_log = open('mytest', "w")
    '''
    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m1 = gp.Model("cont_prob")
    # m2 = gp.Model("int_prob")


    # define variables - problem 1

    x_cont = m1.addMVar(shape = (H,K,T), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x_cont")
    y_cont = m1.addMVar(shape=(K,H), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y_cont")
    f_cont = m1.addMVar(shape=(K), lb=0, vtype=GRB.CONTINUOUS, name="f_cont")
    w_cont = m1.addMVar(shape=(1),lb=0, vtype=GRB.CONTINUOUS, name="w_cont")

    contr1_add_x = m1.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_x")
    contr1_add_y = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_y")
    # contr1_add_f = m1.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_f")
    # contr1_add_w = m1.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_w")
    contr1_abs_x = m1.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_w")
    contr1_abs_y = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_y")
    # contr1_abs_f = m1.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_f")
    # contr1_abs_w = m1.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_w")

    # define variables - problem 2

    # x_int = m2.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x_int")
    # y_int = m2.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y_int")
    # f_int = m2.addMVar(shape=(K), vtype=GRB.INTEGER, name="f_int")
    # w_int = m2.addMVar(shape=(1),vtype=GRB.INTEGER, name="w_int")

    # contr2_add_x = m2.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_x")
    # contr2_add_y = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_y")
    # contr2_add_f = m2.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_f")
    # contr2_add_w = m2.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_add_w")
    # contr2_abs_x = m2.addMVar(shape=(H,K,T), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_x")
    # contr2_abs_y = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_y")
    # contr2_abs_f = m2.addMVar(shape=(K), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_f")
    # contr2_abs_w = m2.addMVar(shape=(1), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="contr1_abs_w")


    x_int_par = np.zeros((H,K,T))
    x_cont_par = np.zeros((H,K,T))

    y_int_par = np.zeros((K,H))
    y_cont_par = np.zeros((K,H))

    #f_int_par = np.zeros(K)
    #f_cont_par = np.zeros(K)

    #w_int_par = 0
    #w_cont_par = 0

    # dual variables

    x_dual = np.zeros((H,K,T))
    y_dual = np.zeros((K,H))
    #f_dual = np.zeros((K))
    #w_dual = 0


    # K1 - C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    m1.addConstr(x_cont[i,j,t] == 0)

    # K2 - C3: all jobs interval are assigned to one only machine
    m1.addConstr( y_cont @ ones_H == ones_K )

    # K3 - C4: memory constraint
    m1.addConstr((y_cont.T *utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))

    # K4 - C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m1.addConstr( x_cont[j,:,:].T @ ones_K <= ones_T )

    # K5 - C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m1.addConstr( qsum(x_cont[j,i,:]) == y_cont[i,j]*proc[i,j])

    # K6
    for j in range(H): #for all machines
        for i in range(K):
            m1.addConstrs( f_cont[i] >= (t+1)*x_cont[j,i,t] for t in range(T))

    # K7
    for i in range(K):
         m1.addConstrs( w_cont >= qsum(trans_back[i,:] * y_cont[i,:]) + f_cont[i] + proc_local[i] for i in range(K))



    max_c = []
    ws = []
    my_ds1 = []
    my_ds2 = []
    step = 0
    violations_1 = []
    violations_2 = []
    while step < 15:
        print(f"{utils.bcolors.OKBLUE}-------------{step}------------{utils.bcolors.ENDC}")
        #f_log.write(f"-------------{step}------------\n")

        if step >= 1:
            for d in my_ds1:
                m1.remove(d)
            my_ds1 = []


        # objective function
        for j in range(K):
            for i in range(H):
                my_ds1.append(m1.addConstr(contr1_add_y[j,i] == y_cont[j,i] - y_int_par[j,i] + y_dual[j,i]))
                my_ds1.append(m1.addConstr(contr1_abs_y[j,i] == gp.abs_(contr1_add_y[j,i])))
                for t in range(T):
                    my_ds1.append(m1.addConstr(contr1_add_x[i,j,t] == x_cont[i,j,t] - x_int_par[i,j,t] + x_dual[i,j,t]))
                    my_ds1.append(m1.addConstr(contr1_abs_x[i,j,t] == gp.abs_(contr1_add_x[i,j,t])))

        m1.setObjective(w_cont +  \
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
        # f_cont_par = np.copy(np.array(f_cont.X))
        # w_cont_par = w_cont.X

        allocated_jobs = np.zeros(H)
        # find the x int and y int
        x_int_par[:,:,:] = 0
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        for i in range(K):
            temp_y_dual = y_cont_par[i, :] + y_dual[i, :]
            print(temp_y_dual)
            aa = np.argsort(-temp_y_dual)[0]
            ii = 0
            print(f'---prin {aa}')
            while((allocated_jobs[aa]+1)*utils.max_memory_demand > memory_capacity[aa]):
                aa = np.argsort(-temp_y_dual)[ii+1]
                ii += 1
            y_int_par[i, :] = 0
            y_int_par[i, aa] = 1
            print(f'--- {aa}')
            allocated_jobs[aa] += 1
            for j in range(H):
                for t in range(T):
                    if (j == aa):
                        if t <= release_date[i,j]:
                            continue
                        temp_x_dual = x_cont_par[j,i,t:] + x_dual[j,i,t:]
                        
                        aa__ = np.argsort(-temp_x_dual)
                        #print(aa__)

                        old_jj = 0
                        for jj in range(proc[i,j]):
                            Kx = list(np.transpose(np.argwhere(x_int_par[j,:,t+aa__[jj]]==1))) 
                            
                            if len(Kx[0]) == 0:
                                continue
                            
                            
                            ind = proc[i,j]
                            
                            for ii in range(len(aa__)- old_jj-proc[i,j]):
                                Kx = list(np.transpose(np.argwhere(x_int_par[j,:,t+aa__[old_jj+proc[i,j]+ii]]==1))) 
                                
                                if len(Kx[0]) == 0:
                                    aa__[jj] = aa__[old_jj+proc[i,j]+ii]
                                    old_jj = ii + old_jj + 1
                                    
                                    break
                                
                                ind += 1

                        jj = 0
                        for jj in range(proc[i,j]):
                            x_int_par[j,i,t+aa__[jj]] = 1
                        break

                        '''
                        if np.abs(x_cont_par[j,i,t] + x_dual[j,i,t] - 0) > np.abs(x_cont_par[j,i,t] + x_dual[j,i,t] - 1):
                            x_int_par[j,i,t] = 1
                        elif np.abs(x_cont_par[j,i,t] + x_dual[j,i,t] - 0) < np.abs(x_cont_par[j,i,t] + x_dual[j,i,t] - 1):
                            x_int_par[j,i,t] = 0
                        else:
                            x_int_par[j,i,t] = np.random.randint(2)
                        '''
                    else:
                        x_int_par[j,i,t] = 0


        # print(f'{utils.bcolors.OKBLUE}Obj2: {m2.ObjVal}{utils.bcolors.ENDC}')




        print('--------------- SHOW 2 ---------------')


        # update dual variables

        for j in range(K):
            for i in range(H):
                y_dual[j,i] = y_dual[j,i] + y_cont[j,i].X - y_int_par[j,i]
                for t in range(T):
                    x_dual[i,j,t] = x_dual[i,j,t] + x_cont[i,j,t].X - x_int_par[i,j,t]

        print('--------------- DUAL VARIABLES ---------------')

        print(y_dual)

        step = step + 1

        g_values = []
        for i in range(K): #for all jobs
            g_interm = []
            for j in range(H): #for all machines
                for t in range(T): #for all timeslots
                    g_interm += [(t+1)*x_int_par[j,i,t]]
            g_values += [max(g_interm)]

        calc_obj = np.zeros(K)
        temptemp = np.multiply(y_int_par, trans_back)
        for i in range(K):
            calc_obj[i] = g_values[i] + np.sum(temptemp[i,:]) + proc_local[i]
        # print(w_x_fixed.X)

        # obj_per_iter += [max(calc_obj)]
        #print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE: {w_int.X}{utils.bcolors.ENDC}')
        print(f'{utils.bcolors.OKBLUE}OPTIMAL VALUE (based on x, y only): {max(calc_obj)}{utils.bcolors.ENDC}')
        #print(f'{utils.bcolors.OKBLUE}changes in y: {changes_y}{utils.bcolors.ENDC}')

        #f_log.write((f'OPTIMAL VALUE: {w_int.X}\n'))
        #ws.append(w_int[0].X)
        # count violations?

        total_constraints_1 = 0
        total_constraints_2 = 0

        counter_1 = 0
        counter_2 = 0

        for i in range(K):
            for j in range(H):
                if y_int_par[i, j] != y_cont_par[i, j]:
                    counter_2 += 1
                total_constraints_2 += 1
                for t in range(T):
                    if x_int_par[j, i, t] != x_cont_par[j, i, t]:
                        counter_1 += 1
                total_constraints_1 += 1
        
        violations_1 += [(counter_1/total_constraints_1)*100]
        violations_2 += [(counter_2/total_constraints_2)*100]

        print("----------------------------------------------------- objective equals to:", step, max(calc_obj))
        #DT = np.append(DT, np.array([[step, w_int_par, max(calc_obj), changes_y, violated_constraints]]), axis=0)


        #np.savetxt('data_for_plot_contintrho20.txt', DT, delimiter=' ', newline='\n')
        

        #f_log.write("--------Machine allocation--------\n")
        print("--------Machine allocation--------\n")
        for i in range(H):
            for k in range(T):
                at_least = 0
                for j in range(K):
                    if(np.rint(x_int_par[i,j,k]) <= 0):
                        continue
                    else:
                        print(f'{j+1}', end='\t')
                        #f_log.write(f'{j+1}\t')
                        at_least += 1
                        break
                if(at_least == 0):
                    print(f'0', end='\t')
                    #f_log.write(f'0\t')
            print('')
            #f_log.write('\n')

        #f_log.write("--------Completition time--------\n")
        print("--------Completition time--------\n")
        cs = []
        reserved = [0 for i in range(H)]
        for i in range(K): #for all jobs
            my_machine = 0
            my_super_machine = -1
            last_zero = -1
            for my_machine in range(H):
                for k in range(T):
                    if np.rint(x_int_par[my_machine,i,k]) >= 1:
                        if last_zero < k+1:
                            last_zero = k+1
                            my_super_machine = my_machine
            fmax = last_zero
            #f_int_par[i] = last_zero
            C = fmax + proc_local[i] + trans_back[i,my_machine]
            cs.append(C)
            #print(f'C{i+1}: {C} - {my_machine}')
            #f_log.write(f'C{i+1}: {C} - {my_super_machine} {y_int_par[i,:].X}\n')
            print(f'C{i+1}: {C} - {my_super_machine} {y_int_par[i,:]}')
            if my_super_machine != -1:
                reserved[my_super_machine] += 1

        #a.append(max(cs))
        w_int_par =  max(cs)
        #f_log.write(f'max is: {max(cs)}\n')
        print(f'max is: {max(cs)}\n')
        print(f'{utils.bcolors.OKBLUE}max is: {max(cs)}{utils.bcolors.ENDC}')

        #f_log.write("check other constraints\n")
        violated = False
        for i in range(K): #for all jobs
            my_machine = -1
            for j in range(H):
                if y_int_par[i,j] == 1:
                    my_machine = j
                    break
            for k in range(release_date[i,my_machine]):
                if x_int_par[my_machine,i,k] == 1:
                    print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                    violated = True

        for i in range(K): #for all jobs
            if np.sum([y_int_par[i,j] for j in range(H)]) != 1:
                print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
                violated = True

        for j in range(H): #for all devices
            if np.sum([y_int_par[i,j] for j in range(H)])*utils.max_memory_demand > memory_capacity[j]:
                print(f"{utils.bcolors.FAIL}Constraint 4 is violated{utils.bcolors.ENDC}")
                violated = True

            occupied = reserved[j]*utils.max_memory_demand
            if occupied > memory_capacity[j]:
                print(f"{utils.bcolors.FAIL}Constraint 4 is violated for machine {j}{utils.bcolors.ENDC}")
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
                    sum_ += np.rint(x_int_par[my_machine,i,k])
                if sum_ != 0 and sum_ != proc[i, my_machine] :
                    print(f"{utils.bcolors.FAIL}Constraint 5 is violated {i+1}{utils.bcolors.ENDC}")
                    violated = True
                else:
                    if sum_ != 0:
                        at_least += 1
            if at_least == 0:
                print(f"{utils.bcolors.FAIL}Constraint 5 is violated job not assigned {i+1}{utils.bcolors.ENDC}")
                violated = True
            if at_least > 1:
                print(f"{utils.bcolors.FAIL}Constraint 5 is violated job assigned more times {i+1}{utils.bcolors.ENDC}")
                violated = True

        for j in range(H): #for all devices
            for t in range(T): #for all timeslots
                temp = 0
                for key in range(K):
                    temp += np.rint(x_int_par[j,key,t])
                if temp > 1:
                    #print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                    violated = True

        if violated:
            #f_log.write('VIOLATED\n')
            print(f'{utils.bcolors.OKBLUE}VIOLATED{utils.bcolors.ENDC}')
        else:
            #f_log.write('OK\n')
            print(f'{utils.bcolors.OKBLUE}OK{utils.bcolors.ENDC}')


    #f_log.close()
    #print(max_c)
    return (violations_1, violations_2, max_c)

if __name__ == '__main__':
    run()
