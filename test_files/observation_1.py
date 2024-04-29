import argparse
import numpy as np
import pandas as pd
import random
import math
import time
import sys


sys.path.insert(0,'../util_files')

import ADMM_solution as admm_sol
import ILP_solver as ilp_sol
import utils as utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--clients', '-K', type=int, default=50, help='the number of clients')
    parser.add_argument('--helpers', '-H', type=int, default=2, help='the number of helpers')
    parser.add_argument('--splitting_points', '-S', type=str, default='3,35', help='give an input in the format of s1,s2')
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
    proc_local_back, trans_back_gradients) = utils.create_scenario(filename, point_a, point_b, 
                                                                   K, H, 
                                                                   scenario,50)
    
    # Define the time horizon
    T = np.max(release_date) + K*np.max(proc[0,:]) + np.max(release_date_back) + K*np.max(proc_bck[0,:]) \
                        + np.max(proc_local) + np.max(proc_local_back)\
                        + np.max(np.max(trans_back)) + np.max(np.max(trans_back_gradients))    

    T = int(T)
    print(f'The time horizon {T}')
    start_ilp = time.time()
    w_star = -1
    w_star = ilp_sol.run(K, H, T, release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), 
                                            release_date_back.astype(int), proc_bck.astype(int), 
                                            proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                            args.log)

    end_ilp = time.time()

    duration_ilp = end_ilp - start_ilp

    w_admm, duration_admm  = admm_sol.run(K, H, T, release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), memory_demand.astype(int),
                                            release_date_back.astype(int), proc_bck.astype(int), 
                                            proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                            args.log)

    print(f"{utils.bcolors.OKGREEN}The optimal makespan is {w_star}, whereas the ADMM solution is {w_admm[-1]}{utils.bcolors.ENDC}")
    print(f"{utils.bcolors.OKGREEN}For the optimal solution we needed {duration_ilp} sec, while for the ADMM solution {duration_admm} sec{utils.bcolors.ENDC}")
    #utils.plot_approach(w_star, w_star)
