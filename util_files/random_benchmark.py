import numpy as np
import random
import utils

def check_memory(capacity, load):
    #print(f'mem: {load} {load*memory_demand} {capacity}')
    return ((load) <= capacity)

def run(K, H, release_date_fwd, proc_fwd, 
            proc_local_fwd, trans_back_activations, 
            memory_capacity, memory_demand,
            release_date_back, proc_bck, 
            proc_local_back, trans_back_gradients, y=[]):
    
    # random seed 
    random.seed(42)

    distribution = [0 for i in range(H)]
    load_ = [0 for i in range(H)]
    y = np.zeros((K,H))
    done = []

    while len(done) < K:
        while True:
            i = random.randint(0, K-1)
            if i not in done:
                done.append(i)
                break
        fit = []
        for j in range(H):
            if check_memory(memory_capacity[j], load_[j]+memory_demand[i]):
                fit.append(j)
            
        if len(fit) == 1:
            distribution[fit[0]] += 1
            y[i,fit[0]] = 1
            load_[fit[0]] += memory_demand[i]
        else:
            my_machine = random.randint(0, len(fit)-1)
            y[i,fit[my_machine]] = 1
            load_[fit[my_machine]] += memory_demand[i]
            distribution[fit[my_machine]] += 1

        
    f_temp_slower = utils.fifo(K, H, release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, y)

    return f_temp_slower