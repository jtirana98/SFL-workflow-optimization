import numpy as np
import time
import sys


sys.path.insert(0,'../util_files')

import ILP_hybrid as ilp_hybrid
import ADMM_hybrid as admm_hybrid
import heuristic_FCFS as fcfs_sol
import utils as utils


if __name__ == '__main__':
    logs = 'test1.txt'
    K = 30
    H = 5

    scenario = 1 # low heterogeneity 

    splitting_points = '2,30'

    points = list(splitting_points.split(','))
    point_a = int(points[0])
    point_b = int(points[1])

    # create the parameter table

    model_type = 'resnet101'
    dataset = 'cifar10'
    if model_type == 'resnet101':
        if dataset == 'cifar10':
            filename = '../real_data/resnet101_CIFAR.xlsx'
        elif dataset == 'mnist':
            filename = '../real_data/resnet101_MNIST.xlsx'
    elif model_type == 'vgg19':
        if dataset == 'cifar10':
            filename = '../real_data/vgg19_CIFAR.xlsx'
        elif dataset == 'mnist':
            filename = '../real_data/vgg19_MNIST.xlsx'

    # get the scenario of the system
            
     # get the scenario of the system
            
    (release_date, proc, 
    proc_local, trans_back, 
    memory_capacity, memory_demand, 
    release_date_back, proc_bck, 
    proc_local_back, trans_back_gradients) = utils.create_scenario_hybrid(filename, point_a, point_b, 
                                                                                K, H, 6000, scenario)

    # Define the time horizon
    T_hybrid = np.max(release_date[1]) + int(K/H)*np.max(proc[1][0,0:H]) + np.max([proc[1][k,H+k] for k in range(K)])  \
                        + np.max(release_date_back[1]) + int(K/H)*np.max(proc_bck[1][0,0:H]) + np.max([proc_bck[1][k,H+k] for k in range(K)])  \
                        + np.max(proc_local[1]) + np.max(proc_local_back[1])\
                        + np.max(np.max(trans_back[1])) + np.max(np.max(trans_back_gradients[1])) 

    T = int(T_hybrid)
    print(T)

    w_hybrid_admm = ([0,0], -1)
    print('---------------------- ADMM -----------------------------------')
    (w_hybrid_admm, _, y_admm, x_par, z_par, cs_back) = admm_hybrid.run(K, H, T_hybrid, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), memory_demand[1].astype(int),
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    
    (y_fcfs, w_fcfs) = fcfs_sol.run_hybrid(K, H, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), [memory_demand[1].astype(int) for i in range(K)],
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    
    clients_0 = []
    clients_1 = []
    for j in range(2):
        print(f'helper {j}')
        for k in range(30):
            if y_admm[k,j] == 1:
                print(f'client {k}')
                if j == 0:
                    clients_0.append(k)
                else:
                    clients_1.append(k)
    
    # helper 0: 16- 20 28-
    # helper 1: 7- 11 12
    # helper 2: 5
    # helper 3: 9
    # helper 4: 10

    print(f"{utils.bcolors.OKGREEN}The hybrid-makespan for the admm is {w_hybrid_admm[-1]}{utils.bcolors.ENDC}")
    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is  {w_fcfs}{utils.bcolors.ENDC}")  


    print('------------- make change --> delay client --------------')

    # adaptive phase ADMM
    release_date[1][9,2] = 15
    release_date[1][7,1] = 5
    release_date[1][5,0] = 7
    release_date[1][16,0] = 5
    release_date[1][28,0] = 6

    h = 0.5

    # for helper 0:
    start_client = [-1 for _ in range(len(clients_0))]
    end_client = [-1 for _ in range(len(clients_0))]

    start_client_z = [-1 for _ in range(len(clients_0))]
    end_client_z = [-1 for _ in range(len(clients_0))]


    T_back = z_par.shape[3]
    T_fwd = x_par.shape[3]

    print(f'T forward is {T_fwd} the backward is {T_back}')
    print(T_back)
    my_machine = 0

    budget_x = []
    budget_z = []

    for client in clients_0:
        budget_x.append(proc[1][client, my_machine])
        budget_z.append(proc_bck[1][client, my_machine])

    for k in range(max(T_fwd, T_back)):
        kathisterimenos = False
        for client in range(len(clients_0)):
            if k < T_fwd and np.rint(x_par[my_machine,clients_0[client],k]) >= 1:
                if release_date[1][clients_0[client], my_machine] <= k: # have not been delayed
                    if start_client[client] == -1:
                        start_client[client] = k 
                    if budget_x[client] == 0.5:
                        kathisterimenos = True # we have space to shift
                        budget_x[client] = 0
                        end_client[client] = k-0.5
                    else:
                        budget_x[client] -= 1
                        if  budget_x[client] <= 0:
                            end_client[client] = k
                else:
                    kathisterimenos = True
            elif np.rint(z_par[my_machine,clients_0[client],k]) >= 1:
                has_arrived = end_client[1][client] + trans_back[1][clients_0[client],my_machine]  \
                            + proc_local[1][clients_0[client]] + release_date_back[1][clients_0[client], my_machine]
                if end_client[client] != -1 and has_arrived <= k: # have not been delayed
                    if start_client_z[client] == -1:
                        start_client_z = k
                    if budget_z[client] == 0.5:
                        kathisterimenos = True # we have space to shift
                        budget_z[client] = 0
                        end_client_z[client] = k-0.5
                    else:
                        budget_z[client] -= 1
                        if  budget_z[client] <= 0:
                            end_client_z[client] = k
                else:
                    kathisterimenos = True
        
        if kathisterimenos: # exoume eleutero slot
            for k_n in range(k_n, max(T_fwd, T_back)):
                for client_l in range(len(clients_0)):
                    if k_n < T_fwd and np.rint(x_par[my_machine,clients_0[client_l],k_n]) >= 1:
                        if release_date[1][clients_0[client_l], my_machine] <= k: # have not been delayed
                            if start_client[client_l] == -1:
                                start_client[client_l] = k + 0.5
                            budget_x[client_l] -= 0.5  
                            if  budget_x[client_l] <= 0:
                                end_client[client_l] = k-0.5
                    elif  np.rint(z_par[my_machine,clients_0[client_l],k_n]) >= 1:
                        has_arrived = end_client[1][client_l] + trans_back[1][clients_0[client_l],my_machine]  \
                            + proc_local[1][clients_0[client_l]] + release_date_back[1][clients_0[client_l], my_machine]
                        if end_client[client_l] != -1 and has_arrived <= k: # have not been delayed
                            if start_client_z[client_l] == -1:
                                start_client_z[client_l] = k + 0.5
                            budget_z[client_l] -= 0.5
                            if  budget_z[client_l] <= 0:
                                end_client_z[client_l] = k-0.5
                        


    # sublirwsi perissevoumwn
    for x_incomplet in range(len(clients_0)):
        if budget_z[]

    # compute new completition time for machine
    completition_0 = 0
    f_client0 = [0 for _ in range(len(clients_0))]
    for client in range(len(clients_0)):
        f_client0[client] = end_client_z[client] + proc_local_back[1][clients_0[client]] + trans_back_gradients[1][clients_0[client],my_machine]
    
    
    for client in range(clients_0):
        cs_back[clients_0[client]] = f_client0[client] 

    Completition_ADMM_LAZE = max(cs_back)

    # ignoring for FCFS
    f_temp_slower = utils.fifo(K, H+K, release_date[1].astype(int), proc[1].astype(int), proc_local[1].astype(int), 
                               trans_back[1].astype(int), release_date_back[1].astype(int),  proc_bck[1].astype(int), 
                               proc_local_back[1].astype(int), trans_back_gradients[1].astype(int), y_fcfs)

    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is  {f_temp_slower} in lazy phase {utils.bcolors.ENDC}") 


    print('------------- Recomputing next round --------------')

    (y_fcfs, w_fcfs) = fcfs_sol.run_hybrid(K, H, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), [memory_demand[1].astype(int) for i in range(K)],
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    
    (w_hybrid_admm, _, y_admm, _, _, _) = admm_hybrid.run(K, H, T_hybrid, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), memory_demand[1].astype(int),
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    
    for j in range(5):
        print(f'helper {j}')
        for k in range(30):
            if y_admm[k,j] == 1:
                print(f'client {k}')
    
    
    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is next round  {w_fcfs}{utils.bcolors.ENDC}")  
    print(f"{utils.bcolors.OKGREEN}The hybrid-makespan for the admm in next round is {w_hybrid_admm[-1]}{utils.bcolors.ENDC}")

