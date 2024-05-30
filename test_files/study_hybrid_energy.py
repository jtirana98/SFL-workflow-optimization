import argparse
import numpy as np
import pandas as pd
import random
import math
import time
import sys


sys.path.insert(0,'../util_files')

#import ADMM_solution as admm_sol
import ILP_hybrid as ilp_hybrid
import ILP_solver as ilp_sol
import ADMM_hybrid as admm_hybrid
import utils as utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--clients', '-K', type=int, default=50, help='the number of clients')
    parser.add_argument('--helpers', '-H', type=int, default=2, help='the number of helpers')
    parser.add_argument('--splitting_points', '-S', type=str, default='2,33', help='give an input in the format of s1,s2')
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='select model resnet101/vgg19')
    parser.add_argument('--alpha', '-a', type=float, default=1, help='alpha value for the multiobjective')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10', help='dataset, options cifar10/mnist')
    parser.add_argument('--scenario', '-s', type=int, default=1, help='scenario 1 for low heterogeneity or 2 for high')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    K = args.clients
    H = args.helpers

    #K = 6
    #H = 2

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
    proc_local_back, trans_back_gradients,
    P_comp, P_transf, P_receive,
    max_slot, network_bwd, ksi) = utils.create_scenario_hybrid_energy(filename, point_a, point_b, K, H, 500)

    # Define the time horizon (hybrid)
    T_hybrid = np.max(release_date) + max(K*np.max(proc[0,0:H]), np.max([proc[k,H+k] for k in range(K)]))  \
                        + np.max(release_date_back) \
                        + max(K*np.max(proc_bck[0,0:H]), np.max([proc_bck[k,H+k] for k in range(K)])) \
                        + np.max(proc_local)  \
                        + np.max(proc_local_back) \
                        + np.max(np.max(trans_back)) \
                        + np.max(np.max(trans_back_gradients))

    T_hybrid = int(T_hybrid)
    print(f'time horizon {T_hybrid} alpha is {args.alpha}')
    
    start_hybrid_optimal = time.time()
    w_hybrid = 2
    print('---------------------- HYBRID -----------------------------------')
    w_hybrid = ilp_hybrid.run_energy(K, H, T_hybrid, release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), 
                                            release_date_back.astype(int), proc_bck.astype(int), 
                                            proc_local_back.astype(int), trans_back_gradients.astype(int),
                                            P_comp, P_transf, P_receive, max_slot, network_bwd, ksi, args.alpha)
    
    print(f"{utils.bcolors.OKGREEN}The hybrid ilp makespan is {w_hybrid}{utils.bcolors.ENDC}")