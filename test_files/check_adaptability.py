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
    w_hybrid_admm = admm_hybrid.run(K, H, T_hybrid, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), memory_demand[1].astype(int),
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    
    w_fcfs = fcfs_sol.run_hybrid(K, H, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), [memory_demand[1].astype(int) for i in range(K)],
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    

    
    print(f"{utils.bcolors.OKGREEN}The hybrid-makespan for the admm is {w_hybrid_admm[0][-1]}{utils.bcolors.ENDC}")
    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is  {w_fcfs}{utils.bcolors.ENDC}")  