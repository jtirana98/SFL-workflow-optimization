import numpy as np
import random
import utils

class arrival_date:
    def __init__(self, value, back, job):
        self.value = value
        self.back = back
        self.job = job

def check_memory(capacity, load):
    #print(f'mem: {load} {load*memory_demand} {capacity}')
    return ((load) <= capacity)

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


def run(K, H, release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, 
         memory_capacity, memory_demand, y=[]):
    
    # random seed 
    random.seed(42)

    f_temp = np.zeros(K)
    f_temp_faster_machine = []

    # machine selection
    y = np.zeros((K,H))
    
    distribution = [0 for i in range(H)] #how many devices on machine
    load_ = [0 for i in range(H)]
    print(distribution)
    for i in range(K):
        fit = []
        for j in range(H):
            if check_memory(memory_capacity[j], load_[j]+memory_demand[i]):
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
            load_[best_load[0]] += memory_demand[i]
            y[i,best_load[0]] = 1  


    f_temp_slower = utils.fifo(K, H, release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, y)

    return f_temp_slower