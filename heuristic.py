import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import time

import utils
import gurobi_final_approach

import warnings
warnings.filterwarnings("ignore")

K = 10
H = 2

class arrival_date:
    def __init__(self, value, back, job):
        self.value = value
        self.back = back
        self.job = job
    

def check_memory(capacity, load, memory_demand):
    #print(f'mem: {load} {load*memory_demand} {capacity}')
    return ((load*memory_demand) <= capacity)

def check(i, mylist):
    for v in mylist:
        if v == i:
            return True
    return False

def check_balance(distributions_):
    larger = -1
    smaller = -1

    for i in range(len(distributions_)):
        if i == 0:
            smaller = distributions_[i]
            larger = distributions_[i]
        else:
            if smaller > distributions_[i]:
                smaller = distributions_[i]
            
            if larger < distributions_[i]:
                larger = distributions_[i]
    return (larger - smaller)

def balance_run(release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, 
         memory_capacity, memory_demand, y=[], y_ready=False):
    #print(memory_capacity)
    #print(memory_demand)
    # random seed 
    #random.seed(42)

    f_temp = np.zeros(K)
    f_temp_faster_machine = []

    # machine selection
    if not y_ready:
        y = np.zeros((K,H))
        
        distribution = [0 for i in range(H)] #how many devices on machine
        #print(distribution)
        for i in range(K):
            fit = []
            for j in range(H):
                #print(j)
                if check_memory(memory_capacity[j], distribution[j]+1, memory_demand):
                    fit.append(j)
                
            if len(fit) == 1:
                distribution[fit[0]] += 1
                y[i,fit[0]] = 1
            else:
                best_load = (-1,-1)
                for j in range(len(fit)):
                    distribution[fit[j]] += 1
                    load = check_balance(distribution)
                    distribution[fit[j]] -= 1
                    if j == 0 or (load < best_load[1]):
                        best_load = (fit[j], load)
                
                distribution[best_load[0]] += 1
                y[i,best_load[0]] = 1  
            #print(distribution)       

                 
        #print('good')
    print(y)
    if not y_ready:
        print(distribution)

    fifo(release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, y) 

    

def gap_run(release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients,
         release_date_proc, release_date_back_proc,
         memory_capacity, memory_demand, y=[], y_ready=False):
    
    start = time.time()
    m = gp.Model("gap")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    f = np.array((K,H))

    y = m.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    sumobj = m.addMVar(shape=(1), name="sumobj")
    
    m.addConstr( y @ ones_H == ones_K )
    m.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    f = (release_date_proc+proc_fwd+trans_back_activations+release_date_back_proc+proc_bck+trans_back_gradients)
    m.addConstr(sumobj == qsum(qsum(f[i,:]*y[i,:] for i in range(K))))

    m.setObjective(sumobj, GRB.MINIMIZE)
    m.optimize()
    end = time.time()

    print(f'Time for gap :  {end-start}')

    print('distribution')
    for machine in range(H):
        my_jobs = list(np.transpose(np.argwhere(y.X[:,machine]==1))[0])
        print(len(my_jobs), end='\t')
    print(' ')

    print(y.X)
    fifo(release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, np.absolute(np.rint(y.X)))
    


def fifo(release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, y):
    
    
    f_temp = np.zeros(K)
    f_temp_faster_machine = []

    # Estimated Completition time
    for machine in range(H):
        #print('----------------------------------------')
        my_jobs = list(np.transpose(np.argwhere(y[:,machine]==1))[0])
        machine_time = 0
        arival_jobs = []
        for j in my_jobs:
            new_job = arrival_date(release_date_fwd[j,machine], False, j)
            arival_jobs.append(new_job)
        first = True
        while len(arival_jobs) > 0:
            #print('rr')
            faster = -1
            for i in range(len(arival_jobs)):
                #print(arival_jobs[i].value)
                if (i == 0) or (arival_jobs[faster].value > arival_jobs[i].value):
                    #print(f'is {i}')
                    faster = i

            next_task = arival_jobs[faster]
            arival_jobs.pop(faster)

            if next_task.back:
                #print('b')
                if machine_time <= next_task.value:
                    #print('smaller')
                    machine_time = next_task.value

                #next_task.value += (machine_time-next_task.value)
                f_temp[next_task.job] = machine_time + proc_bck[next_task.job, machine] +\
                                        trans_back_gradients[next_task.job, machine] +\
                                        proc_local_back[next_task.job]
                
                machine_time +=  proc_bck[next_task.job, machine]
            else:
                #print('f')
                if machine_time <= next_task.value:
                    #print('smaller f')
                    machine_time = next_task.value

                #next_task.value += (machine_time-next_task.value)
                next_task.value = machine_time + proc_fwd[next_task.job, machine] +\
                                   trans_back_activations[next_task.job, machine] + \
                                   proc_local_fwd[next_task.job] +\
                                   release_date_back[next_task.job, machine]
                
                machine_time +=  proc_fwd[next_task.job, machine]
                
                
                next_task.back = True
                arival_jobs.append(next_task)
    print('----------------------------------------')
    for i in range(K):
        print(f_temp[i])
    print(f'completion time: {np.max(f_temp)}')


if __name__ == '__main__':
    balance_run()