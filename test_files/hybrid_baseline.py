import argparse
import numpy as np
import pandas as pd
import random
import math
import time
import sys


sys.path.insert(0,'../util_files')

import heuristic_FCFS as fcfs_sol
import random_benchmark as random_sol
import ADMM_hybrid as admm_hybrid
import utils as utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--clients', '-K', type=int, default=50, help='the number of clients')
    parser.add_argument('--helpers', '-H', type=int, default=2, help='the number of helpers')
    parser.add_argument('--splitting_points', '-S', type=str, default='5,30', help='give an input in the format of s1,s2')
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='select model resnet101/vgg19')
    parser.add_argument('--scenario', '-s', type=int, default=1, help='scenario 1 for low heterogeneity or 2 for high')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10', help='dataset, options cifar10/mnist')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()


    K = args.clients
    H = args.helpers

    scenario = args.scenario

    splitting_points = args.splitting_points

    points = list(splitting_points.split(','))
    point_a = int(points[0])
    point_b = int(points[1])

    # create the parameter table

    model_type = args.model
    dataset = args.dataset
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
            
    (release_date, proc, 
    proc_local, trans_back, 
    memory_capacity, memory_demand, 
    release_date_back, proc_bck, 
    proc_local_back, trans_back_gradients) = utils.create_scenario_hybrid(filename, point_a, point_b, 
                                                                                K, H, 3000, args.scenario)
    
    # Define the time horizon
    T_hybrid = np.max(release_date[1]) + int(K/H)*np.max(proc[1][0,0:H]) + np.max([proc[1][k,H+k] for k in range(K)])  \
                        + np.max(release_date_back[1]) + int(K/H)*np.max(proc_bck[1][0,0:H]) + np.max([proc_bck[1][k,H+k] for k in range(K)])  \
                        + np.max(proc_local[1]) + np.max(proc_local_back[1])\
                        + np.max(np.max(trans_back[1])) + np.max(np.max(trans_back_gradients[1])) 
    print(T_hybrid)
    w_hybrid_admm = ([-1], -1)
    w_hybrid_admm = admm_hybrid.run(K, H, T_hybrid, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), memory_demand[1].astype(int),
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int)) 

    
    start_fcfs = time.time()
    w_fcfs = fcfs_sol.run_hybrid(K, H, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), [memory_demand[1].astype(int) for i in range(K)],
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    end_fcfs = time.time()
    duration_fcfs = end_fcfs - start_fcfs

    start_random = time.time()
    w_random = -1
    w_random = random_sol.run_hybrid(K, H, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), [memory_demand[1].astype(int) for i in range(K)],
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    end_random = time.time()
    duration_random = end_random - start_random

    start_random2 = time.time()
    w_random2 = -1
    w_random2 = random_sol.run_hybrid2(K, H, release_date[1].astype(int), proc[1].astype(int), 
                                            proc_local[1].astype(int), trans_back[1].astype(int), 
                                            memory_capacity[1].astype(int), [memory_demand[1].astype(int) for i in range(K)],
                                            release_date_back[1].astype(int), proc_bck[1].astype(int), 
                                            proc_local_back[1].astype(int), trans_back_gradients[1].astype(int))
    end_random2 = time.time()
    duration_random2 = end_random2 - start_random2
    
    print(memory_capacity)
    print(f"{utils.bcolors.OKGREEN}for the ADMM solution is {w_hybrid_admm[0][-1]}{utils.bcolors.ENDC}")
    print(f"{utils.bcolors.OKGREEN}The makespan for FCFS is  {w_fcfs}{utils.bcolors.ENDC}")    
    print(f"{utils.bcolors.OKGREEN}for the benchmark {w_random}{utils.bcolors.ENDC}")
    print(f"{utils.bcolors.OKGREEN}for the benchmark2 {w_random2}{utils.bcolors.ENDC}")
    print(f"For the ADMM solution {w_hybrid_admm[1]} sec")
    
    #print(f"{utils.bcolors.OKGREEN}For the FCFS we needed {duration_fcfs} sec, for the ADMM solution {w_hybrid_admm[1]} sec, and for the benchmark we need {duration_random}{utils.bcolors.ENDC}")
