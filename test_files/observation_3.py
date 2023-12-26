import argparse
import numpy as np
import pandas as pd
import random
import math
import time

import util_files.ADMM_solution as admm_sol
import util_files.heuristic_FCFS as fcfs_sol
import util_files.random_benchmark as random_sol
import util_files.utils as utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--clients', '-K', type=int, default=50, help='the number of clients')
    parser.add_argument('--helpers', '-H', type=int, default=2, help='the number of helpers')
    parser.add_argument('--splitting_points', '-S', type=str, default='3,33', help='give an input in the form of s1,s2')
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='select model resnet101/vgg19')
    parser.add_argument('--scenario', '-s', type=int, default=1, help='scenario 1 for low heterogeneity and 2 for high')
    parser.add_argument('--dataset', '-d', type=int, default=1, help='dataset, options cifar10/mnist')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    f = open(args.log, "w")
    f.write(f"Experiment for {args.data_owners} data ownners and {args.compute_nodes} compute nodes.\n")
    f.close()

    K = args.data_owners
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
            filename = 'real_data/resnet101_CIFAR.xlsx'
        elif dataset == 'mnist':
            filename = 'real_data/resnet101_MNIST.xlsx'
    elif model_type == 'vgg19':
        if dataset == 'cifar10':
            filename = 'real_data/vgg19_CIFAR.xlsx'
        elif dataset == 'mnist':
            filename = 'real_data/vgg19_MNIST.xlsx'

    # get the scerio of the system
            
    (release_date, proc, 
    proc_local, trans_back, 
    memory_capacity, memory_demand_, 
    release_date_back, proc_bck, 
    proc_local_back, trans_back_gradients) = utils.create_scenario(point_a, point_b, 
                                                                   K, H, 
                                                                   scenario,100)
    
    # Define the time horizon
    T = np.max(release_date) + K*np.max(proc[0,:]) + np.max(release_date_back) + K*np.max(proc_bck[0,:]) \
                        + np.max(proc_local) + np.max(proc_local_back)\
                        + np.max(np.max(trans_back)) + np.max(np.max(trans_back_gradients))    

    start_fcfs = time.time()
    w_fcfs = fcfs_sol.run(K, H, T, release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), 
                                            release_date_back.astype(int), proc_bck.astype(int), 
                                            proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                            args.log)
    end_fcfs = time.time()
    duration_fcfs = end_fcfs - start_fcfs

    start_random = time.time()
    w_random = random_sol.run(K, H, T, release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), 
                                            release_date_back.astype(int), proc_bck.astype(int), 
                                            proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                            args.log)
    end_random = time.time()
    duration_random = end_random - start_random

    start_admm = time.time()
    w_admm = admm_sol.run(K, H, T, release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), 
                                            release_date_back.astype(int), proc_bck.astype(int), 
                                            proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                            args.log)
    end_admm = time.time()
    duration_admm = end_admm - start_admm

    print(f"The makespan for FCFS is  {w_fcfs}, for the ADMM solution is {w_admm}, and for the benchmark {w_random}")
    print(f"For the FCFS we needed {duration_fcfs} sec, for the ADMM solution {duration_admm} sec, and for the benchmark we need {duration_random}")
