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


def run_hybrid(K, H, release_date_fwd, proc_fwd, 
            proc_local_fwd, trans_back_activations, 
            memory_capacity, memory_demand,
            release_date_back, proc_bck, 
            proc_local_back, trans_back_gradients, y=[]):
    
    # random seed 
    random.seed(42)
    H_prime = K + H
    #distribution = [0 for i in range(H)]
    load_ = [0 for i in range(H_prime)]
    y = np.zeros((K,H_prime))
    done = []

    #while len(done) < K:
        # while True:
        #     i = random.randint(0, K-1)
        #     if i not in done:
        #         done.append(i)
        #         break
    for i in range(K):
        fit = []
        for j in range(H):
            if check_memory(memory_capacity[j], load_[j]+memory_demand[i]):
                fit.append(j)
        
        if check_memory(memory_capacity[H+i], load_[H+i]+memory_demand[i]):
            fit.append(H+i)
        
        print(fit)
        if len(fit) == 1:
            #distribution[fit[0]] += 1
            y[i,fit[0]] = 1
            load_[fit[0]] += memory_demand[i]
        else:
            my_machine = random.choice(fit)
            y[i,my_machine] = 1
            load_[my_machine] += memory_demand[i]
            #distribution[fit[my_machine]] += 1
            print(my_machine)

    print('--------------------- RANDOM ------------------------')
    print(y)
    f_temp_slower = utils.fifo(K, H+K, release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, y)

    return f_temp_slower