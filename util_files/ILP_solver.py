import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import warnings

import utils 
warnings.filterwarnings("ignore")


def run(K, H, T, release_date, proc, proc_local, trans_back, memory_capacity, memory_demand, filename=''):
    T = np.max(release_date) + K*np.max(proc) # time intervals
    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("optimal_solution")
    
    # define variables
    
    x = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")

    # define constraints
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    m.addConstr(x[i,j,t] == 0)

    # C3: all jobs interval are assigned to one only machine
    m.addConstr( y @ ones_H == ones_K )
    
    # C4: memory constraint
    m.addConstr(memory_demand @ y <= memory_capacity)
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc[i,j])

    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*x[j,i,t] for t in range(T))

    # Define objective function
    

    m.addConstrs(comp[i] == qsum(trans_back[i,:] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
       
    
    m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
      
    m.update()
      
    # Optimize model
    m.optimize()

    # Checking if constraints are satisfied
   
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(K): #for all jobs
        my_machine = -1
        for j in range(H):
            if y[i,j].X == 1:
                my_machine = j
                break
        for k in range(release_date[i,my_machine]):
            if x[my_machine,i,k].X == 1:
                print(f"{utils.bcolors.FAIL}Constraint 1 is violated{utils.bcolors.ENDC}")
                return

    # C3: all jobs interval are assigned to one only machine
    for i in range(K): #for all jobs
        if np.sum([y[i,j].X for j in range(H)]) != 1:
            print(f"{utils.bcolors.FAIL}Constraint 3 is violated{utils.bcolors.ENDC}")
            return
    
    # C5: job should be processed entirely once
    for i in range(K):
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].X)  == 1:
                my_machine = j
                break
        
        sum = 0
        for k in range(T):
            sum += np.rint(x[my_machine,i,k].X)
        if sum != proc[i, my_machine]:
            print(f"{utils.bcolors.FAIL}Constraint 5 is violated{utils.bcolors.ENDC}")
            return

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            temp = 0
            for key in range(K):
                temp += np.rint(x[j,key,t].X)
            if temp > 1:
                print(f"{utils.bcolors.FAIL}Constraint 6 is violated{utils.bcolors.ENDC}")
                return

    if filename != '':
        f_ = open(filename, "a")
        f_.write("Original:\n")
    else:
        f_ = open("my_test", "w")
        
    '''
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
                    at_least = 1
                    break
            if(at_least == 0):
                #print(f'0', end='\t')
                f_.write(f'0\t')
        #print('')
        f_.write('\n')
    
    g_var = []
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H):
            if np.rint(y[i,j].X) == 1:
                my_machine = j
                break
        last_zero = -1
        for k in range(T):
            if np.rint(x[my_machine,i,k].X) >= 1:
                last_zero = k+1
        fmax = last_zero
        g_var.append(fmax)
        C = fmax + proc_local[i] + trans_back[i,my_machine]
        #print(f'C{i+1}: {C} - {my_machine}')
    '''

    return(m.ObjVal)

if __name__ == '__main__':
    run()