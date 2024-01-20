import argparse
import numpy as np
import pandas as pd
import random
import math
import time
import sys


sys.path.insert(0,'../util_files')

import ADMM_solution as admm_sol
import utils as utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--splitting_points', '-S', type=str, default='3,33', help='give an input in the format of s1,s2')
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='select model resnet101/vgg19')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10', help='dataset, options cifar10/mnist')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    K = 100
    H = [1, 2, 5, 10, 20, 25]

    scenario = 1

    splitting_points = args.splitting_points
    slot_duration = 100

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

    
    w_makespans = []
    duration = []

    for h in H:
        # get the scenario of the system
        (release_date, proc, 
        proc_local, trans_back, 
        memory_capacity, memory_demand, 
        release_date_back, proc_bck, 
        proc_local_back, trans_back_gradients) = utils.create_scenario(filename, point_a, point_b, 
                                                                    K, h, 
                                                                    scenario, 100)
        
        # Define the time horizon
        T = np.max(release_date) + K*np.max(proc[0,:]) + np.max(release_date_back) + K*np.max(proc_bck[0,:]) \
                            + np.max(proc_local) + np.max(proc_local_back)\
                            + np.max(np.max(trans_back)) + np.max(np.max(trans_back_gradients))    


        w_admm, duration_admm = admm_sol.run(K, h, T, release_date.astype(int), proc.astype(int), 
                                                proc_local.astype(int), trans_back.astype(int), 
                                                memory_capacity.astype(int), memory_demand.astype(int), 
                                                release_date_back.astype(int), proc_bck.astype(int), 
                                                proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                                args.log)

        duration.append(duration_admm)
        w_makespans.append(w_admm)

    print(f"The results:")
    i = 0
    for h in H:
        print(f'The makespan for {h} helpers is {w_makespans[i][-1]} and the computing time is {duration[i]}')
        i += 1
