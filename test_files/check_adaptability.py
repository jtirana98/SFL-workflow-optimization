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
    K = 50
    H = 5

    scenario = 1 # low heterogeneity 

    splitting_points = '20,30'

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
        
    print(f' FINISH TIMES BEFORE {cs_back}')
    print(f"{utils.bcolors.OKGREEN}The hybrid-makespan for the admm is {w_hybrid_admm[-1]}{utils.bcolors.ENDC}")
    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is  {w_fcfs}{utils.bcolors.ENDC}")  
    
    print('------------- make change --> delay client --------------')
    
    release_date[1][9,0] = release_date[1][9,0]*2
    release_date_back[1][9,0] = release_date_back[1][9,0]*2

    release_date[1][13,0] = release_date[1][13,0]*2
    release_date_back[1][13,0] = release_date_back[1][13,0]*2

    release_date[1][16,0] = release_date[1][16,0]*2
    release_date_back[1][16,0] = release_date_back[1][16,0]*2

    release_date[1][34,0] = release_date[1][34,0]*2
    release_date_back[1][34,0] = release_date_back[1][34,0]*2

    # release_date[1][39,0] = release_date[1][39,0]*2
    # release_date_back[1][39,0] = release_date_back[1][39,0]*2

    # release_date[1][47,0] = release_date[1][47,0]*2
    # release_date_back[1][47,0] = release_date_back[1][47,0]*2

    release_date[1][15,1] = release_date[1][15,1]*2
    release_date_back[1][15,1] = release_date_back[1][15,1]*2

    release_date[1][22,1] = release_date[1][22,1]*2
    release_date_back[1][22,1] = release_date_back[1][22,1]*2

    # release_date[1][33,1] = release_date[1][33,1]*2
    # release_date_back[1][33,1] = release_date_back[1][33,1]*2

    release_date[1][41,1] = release_date[1][41,1]*2
    release_date_back[1][41,1] = release_date_back[1][41,1]*2

    release_date[1][8,0] = release_date[1][8,0]*2
    release_date_back[1][8,0] = release_date_back[1][8,0]*2
    
    release_date[1][8,1] = release_date[1][8,1]*2
    release_date_back[1][8,1] = release_date_back[1][8,1]*2

    release_date[1][8,2] = release_date[1][8,2]*2
    release_date_back[1][8,2] = release_date_back[1][8,2]*2
    
    release_date[1][8,3] = release_date[1][8,3]*2
    release_date_back[1][8,3] = release_date_back[1][8,3]*2

    release_date[1][8,4] = release_date[1][8,4]*2
    release_date_back[1][8,4] = release_date_back[1][8,4]*2

    release_date[1][0,0] = release_date[1][0,0]*8
    release_date_back[1][0,0] = release_date_back[1][0,0]*8

    # release_date[1][21,0] = release_date[1][21,0]*10
    # release_date_back[1][21,0] = release_date_back[1][21,0]*10

    machines = [i for i in range(H)]
    clients = []

    for j in range(H):
        print(f'helper {j}')
        client_h = []
        for k in range(K):
            if y_admm[k,j] == 1:
                print(f'client {k}')
                client_h.append(k)
        clients.append(client_h)     

    h = 0.5
    T_back = z_par.shape[2]
    T_fwd = x_par.shape[2]

    # adaptive phase ADMM
    print(f'T forward is {T_fwd} the backward is {T_back}')
    print(T_back)


    print(f' FINISH TIMES BEFORE {cs_back}')
    for my_machine in machines:
        budget_x = []
        budget_z = []
        
        print(f'The machine: {my_machine}')
        print(clients[my_machine])
        if len(clients[my_machine]) == 0:
            continue

        for client in clients[my_machine]:
            budget_x.append(proc[1][client, my_machine])
            budget_z.append(proc_bck[1][client, my_machine])
        
        # print(budget_x)
        start_client = [-1 for _ in range(len(clients[my_machine]))]
        end_client = [-1 for _ in range(len(clients[my_machine]))]

        start_client_z = [-1 for _ in range(len(clients[my_machine]))]
        end_client_z = [-1 for _ in range(len(clients[my_machine]))]

        last_t = -1
        for k in range(max(T_fwd, T_back)):
            for client in range(len(clients[my_machine])):
                kathisterimenos = False
                if k < T_fwd and np.rint(x_par[my_machine,clients[my_machine][client],k]) >= 1:
                    # print(f'at time {k} client {clients[my_machine][client]} should be here fwd')
                    if release_date[1][clients[my_machine][client], my_machine] <= k and budget_x[client] > 0: # have not been delayed
                        # print('IT IS!')
                        last_t = k
                        if start_client[client] == -1:
                            start_client[client] = k 
                        if budget_x[client] == 0.5:
                            last_t -= 0.5
                            # print('finish 0.5')
                            kathisterimenos = True # we have space to shift
                            budget_x[client] = 0
                            end_client[client] = k-0.5
                        else:
                            budget_x[client] -= 1
                            if  budget_x[client] <= 0:
                                end_client[client] = k
                    else:
                        # print(f'NOOO! {budget_x[client]}')
                        kathisterimenos = True
                elif np.rint(z_par[my_machine,clients[my_machine][client],k]) >= 1:
                    print(f'at time {k} client {clients[my_machine][client]} should be here back')
                    has_arrived = end_client[client] + trans_back[1][clients[my_machine][client],my_machine]  \
                                + proc_local[1][clients[my_machine][client]] + release_date_back[1][clients[my_machine][client], my_machine]
                    if end_client[client] != -1 and has_arrived <= k  and budget_z[client] > 0: # have not been delayed
                        # print('IT IS!')
                        last_t = k
                        if start_client_z[client] == -1:
                            start_client_z[client] = k
                        if budget_z[client] == 0.5:
                            last_t -= 0.5
                            kathisterimenos = True # we have space to shift
                            budget_z[client] = 0
                            # print('finish 0.5')
                            end_client_z[client] = k-0.5
                        else:
                            budget_z[client] -= 1
                            if  budget_z[client] <= 0:
                                end_client_z[client] = k
                    else:
                        # print(f'NOOO! {budget_z[client]}')
                        kathisterimenos = True
                
                if kathisterimenos: # exoume eleutero slot
                    # print('KATHISTERIMENA')
                    found = False
                    for k_n in range(k, max(T_fwd, T_back)):
                        if found:
                            break
                        for client_l in range(len(clients[my_machine])):
                            if k_n < T_fwd and np.rint(x_par[my_machine,clients[my_machine][client_l],k_n]) >= 1 and budget_x[client_l] > 0:
                                if release_date[1][clients[my_machine][client_l], my_machine] <= k: 
                                    # print(f'found client {clients[my_machine][client_l]} insted fwd')
                                    last_t = k
                                    if start_client[client_l] == -1:
                                        start_client[client_l] = k
                                    budget_x[client_l] -= 0.5
                                    if  budget_x[client_l] <= 0:
                                        end_client[client_l] = k
                                        # print(f'teleiwse')
                                    found = True
                                    break
                            elif  np.rint(z_par[my_machine,clients[my_machine][client_l],k_n]) >= 1:
                                has_arrived = end_client[client_l] + trans_back[1][clients[my_machine][client_l],my_machine]  \
                                    + proc_local[1][clients[my_machine][client_l]] + release_date_back[1][clients[my_machine][client_l], my_machine]
                                if end_client[client_l] != -1 and has_arrived <= k and budget_z[client_l] > 0:
                                    last_t = k
                                    print(f'found client {clients[my_machine][client_l]} insted bwd')
                                    if start_client_z[client_l] == -1:
                                        start_client_z[client_l] = k
                                    budget_z[client_l] -= 0.5
                                    if  budget_z[client_l] <= 0:
                                        end_client_z[client_l] = k
                                        print(f'teleiwse')
                                    found = True
                                    break
                            
        # sublirwsi perissevoumwn
        for x_incomplete in range(len(clients[my_machine])):
            print(f'hehhe incomplete {clients[my_machine][x_incomplete]} {last_t}')
            if budget_x[x_incomplete] > 0:
                end_client[x_incomplete] = last_t + budget_x[x_incomplete]
                last_t = last_t + budget_x[x_incomplete]
                budget_x[x_incomplete] = 0
                print(f'fwd {end_client[x_incomplete]}')

            if budget_z[x_incomplete] > 0:
                end_client_z[x_incomplete] = last_t + budget_z[x_incomplete] + trans_back[1][clients[my_machine][x_incomplete],my_machine]  \
                                + proc_local[1][clients[my_machine][x_incomplete]] + release_date_back[1][clients[my_machine][x_incomplete], my_machine]
                last_t = last_t + budget_z[x_incomplete]
                budget_z[x_incomplete] = 0
                print(f'back {end_client_z[x_incomplete]}')

            

        # compute new completition time for machine
        f_client0 = [0 for _ in range(len(clients[my_machine]))]
        for client in range(len(clients[my_machine])):
            f_client0[client] = end_client_z[client] + proc_local_back[1][clients[my_machine][client]] + trans_back_gradients[1][clients[my_machine][client],my_machine]
    
    
        for client in range(len(clients[my_machine])):
            cs_back[clients[my_machine][client]] = f_client0[client] 
    
    print(f' FINISH TIMES AFTER {cs_back}')
    Completition_ADMM_LAZY = max(cs_back)

     # ignoring for FCFS
    f_temp_slower = utils.fifo(K, H+K, release_date[1].astype(int), proc[1].astype(int), proc_local[1].astype(int), 
                               trans_back[1].astype(int), release_date_back[1].astype(int),  proc_bck[1].astype(int), 
                               proc_local_back[1].astype(int), trans_back_gradients[1].astype(int), y_fcfs)


    print(f"{utils.bcolors.OKGREEN}The makespan for ADMM is  {Completition_ADMM_LAZY} in lazy phase {utils.bcolors.ENDC}") 
    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is  {f_temp_slower} in lazy phase {utils.bcolors.ENDC}") 
    start_time = time.time()
    temp_sc = admm_hybrid.run_scheduling(K, H, T_hybrid, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), memory_demand[1].astype(int),
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int), y_admm)
    end_time = time.time()
    print(f"{utils.bcolors.OKGREEN}The makespan for ADMM is  {temp_sc} with fixed scheduling {utils.bcolors.ENDC}") 
    print(f"{utils.bcolors.OKGREEN}The computation of the alg2 lasted {(end_time-start_time)} {utils.bcolors.ENDC}") 
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
    
    
    print(f"{utils.bcolors.OKGREEN}The hybrid-makespan for the admm in next round is {w_hybrid_admm[-1]}{utils.bcolors.ENDC}")
    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is next round  {w_fcfs}{utils.bcolors.ENDC}")  
