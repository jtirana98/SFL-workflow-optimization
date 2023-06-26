import argparse
import numpy as np
import pandas as pd
import random
import math
import time

import gurobi_solver
import gurobi_fwd_back
import gurobi_final_approach
import utils

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--testcase', type=str, default='fully_symmetric', help='fully_symmetric or fully_heterogeneous')
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--data_owners', '-K', type=int, default=50, help='the number of data owners')
    parser.add_argument('--compute_nodes', '-H', type=int, default=2, help='the number of compute nodes')
    parser.add_argument('--splitting_points', '-S', type=str, default='3,35', help='give an input in the form of s1,s2')
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='select model resnet101/vgg19')
    parser.add_argument('--repeat', '-r', type=str, default='', help='avoid generating input include file')
    parser.add_argument('--back', '-b', type=int, default=0, help='include backpropagation')
    #parser.add_argument('--approach', type=str, default='approach3', help='select one of the approaches: approach1a/approach1aFreeze/approach2/approach3')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    f = open(args.log, "w")
    f.write(f"Experiment for {args.data_owners} data ownners and {args.compute_nodes} compute nodes.\n")
    f.close()

    K = args.data_owners
    H = args.compute_nodes

    back_flag = False
    if args.back == 1:
        back_flag = True
    
    
    splitting_points = args.splitting_points

    points = list(splitting_points.split(','))
    point_a = int(points[0])
    point_b = int(points[1])

    print(f'I have your splitting points after {point_a} and {point_b}, including')
    
    if args.repeat == '':
        # create the parameter table
        model_type = args.model

        if model_type == 'resnet101':
            filename = 'real_data/resnet101_CIFAR.xlsx'
        elif model_type == 'vgg19':
            filename = 'real_data/vgg19.xlsx'


        df_vm = pd.read_excel(io=filename, sheet_name='VM', header=None)
        df_laptop = pd.read_excel(io=filename, sheet_name='laptop', header=None)
        df_d1 = pd.read_excel(io=filename, sheet_name='d1', header=None)
        df_d2 = pd.read_excel(io=filename, sheet_name='d2', header=None)
        df_jetson_cpu = pd.read_excel(io=filename, sheet_name='jetson-cpu', header=None)
        df_jetson_gpu = pd.read_excel(io=filename, sheet_name='jetson-gpu', header=None)
        df_memory = pd.read_excel(io=filename, sheet_name='memory', header=None)
        

        # processing time on vms
        vm_data = df_vm.values.tolist()

        vm_proc_fwd = 0
        vm_proc_back = 0
        for i in range(point_a, point_b):
            vm_proc_fwd += vm_data[i][0]
            vm_proc_back += vm_data[i][1] + vm_data[i][2]

        # processing time on my laptop
        laptop_data = df_laptop.values.tolist()
        
        laptop_proc_fwd = 0
        laptop_proc_back = 0
        for i in range(point_a, point_b):
            laptop_proc_fwd += laptop_data[i][0]
            laptop_proc_back += laptop_data[i][1] + laptop_data[i][2]

        
        # processing time on d1
        d1_data = df_d1.values.tolist()

        d1_proc_fwd_first = 0
        d1_proc_fwd_last = 0
        d1_proc_back_first = 0
        d1_proc_back_last = 0
        for i in range(0, point_a):
            d1_proc_fwd_first += d1_data[i][0]
            d1_proc_back_first += d1_data[i][1] + d1_data[i][2]
        
        for i in range(point_b, len(d1_data)):
            d1_proc_fwd_last += d1_data[i][0]
            d1_proc_back_last += d1_data[i][1] + d1_data[i][2]

        # processing time on jetson-cpu
        jetson_cpu_data = df_jetson_cpu.values.tolist()

        jetson_cpu_proc_fwd_first = 0
        jetson_cpu_proc_fwd_last = 0
        jetson_cpu_proc_back_first = 0
        jetson_cpu_proc_back_last = 0
        for i in range(0, point_a):
            jetson_cpu_proc_fwd_first += jetson_cpu_data[i][0]
            jetson_cpu_proc_back_first += jetson_cpu_data[i][1] + jetson_cpu_data[i][2]
        
        for i in range(point_b, len(jetson_cpu_data)):
            jetson_cpu_proc_fwd_last += jetson_cpu_data[i][0]
            jetson_cpu_proc_back_last += jetson_cpu_data[i][1] + jetson_cpu_data[i][2]

        # processing time on jetson-gpu
        jetson_gpu_data = df_jetson_gpu.values.tolist()

        jetson_gpu_proc_fwd_first = 0
        jetson_gpu_proc_fwd_last = 0
        jetson_gpu_proc_back_first = 0
        jetson_gpu_proc_back_last = 0
        for i in range(0, point_a):
            jetson_gpu_proc_fwd_first += jetson_gpu_data[i][0]
            jetson_gpu_proc_back_first += jetson_gpu_data[i][1] + jetson_gpu_data[i][2]
        
        for i in range(point_b, len(jetson_gpu_data)):
            jetson_gpu_proc_fwd_last += jetson_gpu_data[i][0]
            jetson_gpu_proc_back_last += jetson_gpu_data[i][1] + jetson_gpu_data[i][2]

        # processing time on d2
        d2_data = df_d2.values.tolist()

        d2_proc_fwd_first = 0
        d2_proc_fwd_last = 0
        d2_proc_back_first = 0
        d2_proc_back_last = 0
        for i in range(0, point_a):
            d2_proc_fwd_first += d2_data[i][0]
            d2_proc_back_first += d2_data[i][1] + d2_data[i][2]

        for i in range(point_b, len(d2_data)):
            d2_proc_fwd_last += d2_data[i][0]
            d2_proc_back_last += d2_data[i][1] + d2_data[i][2]


        memory_data = df_memory.values.tolist()
        
        # travel data
        activations_to_cn = memory_data[point_a-1][0]
        activations_to_do = memory_data[point_b-1][0]

        # store data
        store_data_owner = 0
        for i in range(0, point_a):
            store_data_owner += memory_data[i][0] + memory_data[i][1]
            
        for i in range(point_b, len(d1_data)):
            store_data_owner += memory_data[i][0] + memory_data[i][1]

        store_compute_node = 0
        for i in range(point_a, point_b):
            store_compute_node += memory_data[i][0] + memory_data[i][1]
        store_compute_node = (store_compute_node/1024) # prefer GB

        my_net = lambda data,bandwidth : ((data*0.000008)/bandwidth)*1000
        network_connections = [ lambda a : ((a*0.000008)/8)*1000, # 8 Mbits/sec
                                lambda a : (a*0.0000008)*1000, # 10 Mbits/sec
                                lambda a : ((a*0.000000008)/7.13)*1000, # 7.13 Gbits/sec
                                lambda a : ((a*0.000008)/2)*1000 # 2 Mbits/sec
                              ]
        
        # random seed 
        random.seed(42)

        # randomly select the network connections
        network_type = np.zeros((K,H))
        '''
        network_type = np.zeros((K,H))
        for j in range(K):
            for i in range(H):
                network_type[j,i] = random.randint(0,len(network_connections)-1)
        '''

        '''
        class-0        <= 4 Mbps      --> 30%
        class-1        >4 and <= 10   --> 42%
        class-2        >10 and <= 15  --> 12%
        class-3        > 15 and <= 20 --> 28%
        '''

        class0 = []
        class1 = []
        class2 = []
        class3 = []

        total_connections = K*H

        num_class0 = int((total_connections*30)/100)
        num_class1 = int((total_connections*42)/100)
        num_class2 = int((total_connections*12)/100)
        num_class3 = int((total_connections*28)/100)

        num_class2 += K*H - (num_class0+num_class1+num_class2+num_class3)

        completed = []

        for i in range(num_class0):
            while True:
                net_line = int(random.randint(0,total_connections-1))

                if not (net_line in completed):
                    break
            
            completed.append(net_line)
            network_type[int(net_line/H),int(net_line%H)] = random.randint(1,4)

        for i in range(num_class1):
            
            while True:
                net_line = int(random.randint(0,total_connections-1))

                if not (net_line in completed):
                    break
            
            completed.append(net_line)
            network_type[int(net_line/H),int(net_line%H)] = random.randint(5,10)
        
        for i in range(num_class2):
            
            while True:
                net_line = int(random.randint(0,total_connections-1))

                if not (net_line in completed):
                    break
            
            completed.append(net_line)
            network_type[int(net_line/H),int(net_line%H)] = random.randint(11,15)
        
        for i in range(num_class3):
            
            while True:
                net_line = int(random.randint(0,total_connections-1))

                if not (net_line in completed):
                    break
            
            completed.append(net_line)
            network_type[int(net_line/H),int(net_line%H)] = random.randint(16,20)

        print(len(set(completed)))

        for i in range(K):
            for j in range(H):
                print(network_type[i,j], end='\t')
            print('')

        # compute node device type 
        # we have:
        # 0 for vm
        # 1 for laptop
        machine_devices = np.zeros((H))
        for i in range(H):
            machine_devices[i] = random.randint(0,1)

        # data owner device type 
        # we have:
        # 0 for d1
        # 1 for d2
        # 2 for jetson gpu
        # 3 for jetson cpu
        do_devices = np.zeros((K))
        for i in range(K):
            do_devices[i] = random.randint(0,3)

        # forward parameters
        release_date = np.zeros((K,H))
        proc = np.zeros((K,H))
        proc_local = np.zeros((K))
        trans_back = np.zeros((K, H))

        # back-propagation parameters
        release_date_back = np.zeros((K,H))
        proc_bck = np.zeros((K,H))
        proc_local_back =np.zeros((K))
        trans_back_gradients = np.zeros((K, H))

        for j in range(K):
            for i in range(H):
                indx = int(network_type[j,i])
                
                release_date[j,i] = my_net(activations_to_cn, indx)
                trans_back[j,i] = my_net(activations_to_do, indx)

                release_date_back[j,i] = my_net(activations_to_do, indx)
                trans_back_gradients[j,i] = my_net(activations_to_cn, indx)
                
                if int(do_devices[j]) == 0:
                    release_date[j,i] +=  d1_proc_fwd_first
                    release_date_back[j,i] += d1_proc_back_last
                elif int(do_devices[j]) == 1:
                    release_date[j,i] +=  d2_proc_fwd_first
                    release_date_back[j,i] += d2_proc_back_last
                elif int(do_devices[j]) == 3:
                    release_date[j,i] +=  jetson_cpu_proc_fwd_first
                    release_date_back[j,i] += jetson_cpu_proc_back_last
                elif int(do_devices[j]) == 2:
                    release_date[j,i] +=  jetson_gpu_proc_fwd_first
                    release_date_back[j,i] += jetson_gpu_proc_back_last
                    
                if i == 0:
                    if int(do_devices[j]) == 0:
                        proc_local[j] =  d1_proc_fwd_last
                        proc_local_back[j] =  d1_proc_back_first
                    elif int(do_devices[j]) == 1:
                        proc_local[j] =  d2_proc_fwd_last
                        proc_local_back[j] =  d2_proc_back_first
                    elif int(do_devices[j]) == 2:
                        proc_local[j] =  jetson_cpu_proc_fwd_last
                        proc_local_back[j] =  jetson_cpu_proc_back_first
                    elif int(do_devices[j]) == 3:
                        proc_local[j] =  jetson_gpu_proc_fwd_last
                        proc_local_back[j] =  jetson_gpu_proc_back_first
                    
                
                if int(machine_devices[i]) == 0:
                    proc[j,i] =  vm_proc_fwd
                    proc_bck[j,i] =  vm_proc_back
                elif int(machine_devices[i]) == 1:
                    proc[j,i] = laptop_proc_fwd
                    proc_bck[j,i] = laptop_proc_back

                        
        memory_demand = np.ones((K)) * store_compute_node
        for i in range(K):
            memory_demand[i] = int(math.ceil(memory_demand[i]))

        print(f'Memory --- {memory_demand}')
        utils.max_memory_demand = int(max(memory_demand))
        
        memory_capacity = np.array(utils.get_memory_characteristics(H, K))
        for i in range(len(memory_capacity)):
            memory_capacity[i] = int(memory_capacity[i])

        unique_friends = []
        unique_friends_back = []
        
        print('proc')
        for i in range(H):
            print(f'{proc[0,i]}--{int(machine_devices[i])}', end='\t')
            print(f'{proc_bck[0,i]}--{int(machine_devices[i])}', end='\t')
            print('~~~~~')
            unique_friends.append(int(np.rint(proc[0,i])))
            unique_friends_back.append(int(np.rint(proc_bck[0,i])))
        #print('')

        #print('release date')
        for j in range(K):
            for i in range(H):
                #print(f'{release_date[j,i]}--{int(network_type[j,i])}--{int(do_devices[j])}', end='\t')
                unique_friends.append(int(np.rint(release_date[j,i])))
                unique_friends_back.append(int(np.rint(release_date_back[j,i])))
            #print('')

        #print('proc local: ')
        for j in range(K):
            #print(f'{proc_local[j]}', end='\t')
            unique_friends.append(int(np.rint(proc_local[j])))
            unique_friends_back.append(int(np.rint(proc_local_back[j])))
        #print('')

        #print('trans back: ')
        for j in range(K):
            for i in range(H):
                #print(f'{trans_back[j,i]}', end='\t')
                unique_friends.append(int(np.rint(trans_back[j,i])))
                unique_friends_back.append(int(np.rint(trans_back_gradients[j,i])))
            #print('')
        
        print('\n----------------------     max values:  -------------------\n')
        max_value = max(max(unique_friends), max(unique_friends_back))
        max_value_back = max(unique_friends_back)
        print('-------------------')
        print(unique_friends)
        print(math.gcd(*unique_friends))
        print(f'max is {max_value}')
        print(max_value_back)
        print(min(unique_friends))
        print('\n----------------------     max values:  -------------------\n')
        
        # Re-difine parameters
        max_slot = 500
        max_slot_back = 50
        for j in range(K):
            for i in range(H):
                
                release_date[j,i] = np.rint((release_date[j,i]*max_slot)/max_value).astype(int)
                trans_back[j,i] = np.rint((trans_back[j,i]*max_slot)/max_value).astype(int)
                
                #release_date_back[j,i] = np.rint((release_date_back[j,i]*max_slot_back)/max_value_back).astype(int)
                #trans_back_gradients[j,i] = np.rint((trans_back_gradients[j,i]*max_slot_back)/max_value_back).astype(int)
                release_date_back[j,i] = np.rint((release_date_back[j,i]*max_slot_back)/max_value).astype(int)
                trans_back_gradients[j,i] = np.rint((trans_back_gradients[j,i]*max_slot_back)/max_value).astype(int)

                if i == 0:
                    proc_local[j] = np.rint((proc_local[j]*max_slot)/max_value).astype(int)
                    #proc_local_back[j] = np.rint((proc_local_back[j]*max_slot_back)/max_value_back).astype(int)
                    proc_local_back[j] = np.rint((proc_local_back[j]*max_slot_back)/max_value).astype(int)

                proc[j,i] =  np.rint((proc[j,i]*max_slot)/max_value).astype(int)
                
                if proc[j,i] == 0:
                        proc[j,i] = 1

                #proc_bck[j,i] =  np.rint((proc_bck[j,i]*max_slot_back)/max_value_back).astype(int)
                proc_bck[j,i] =  np.rint((proc_bck[j,i]*max_slot_back)/max_value).astype(int)

                if proc_bck[j,i] == 0:
                        proc_bck[j,i] = 1


        
        print('                                             NEW')
        print('proc')
        for i in range(H):
            print(f'{proc[0,i]}', end='\t')
            print(f'{proc_bck[0,i]} --- {machine_devices[i]}', end='\t')
            print('~~~~~~')
        print('')
        '''
        print('release date')
        for j in range(K):
            for i in range(H):
                #print(f'{release_date[j,i]}', end='\t')
                print(f'{release_date_back[j,i]}', end='\t')
            print('')

        print('proc local: ')
        for j in range(K):
            #print(f'{proc_local[j]}', end='\t')
            print(f'{proc_local_back[j]}', end='\t')
        print('')

        print('trans back: ')
        for j in range(K):
            for i in range(H):
                #print(f'{trans_back[j,i]}', end='\t')
                print(f'{trans_back_gradients[j,i]}', end='\t')
            print('')
        '''
        # call the original solver for comparison
        if back_flag:
            print('Calling the original approach. -- BACK')
            gurobi_fwd_back.K = args.data_owners
            gurobi_fwd_back.H = args.compute_nodes
            w_start = 118
        
            w_start = gurobi_fwd_back.run(release_date.astype(int), proc.astype(int), 
                                          proc_local.astype(int), trans_back.astype(int), 
                                          memory_capacity.astype(int), 
                                          release_date_back.astype(int), proc_bck.astype(int), 
                                          proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                          args.log)
            

        else:
            print('Calling the original approach.')
            gurobi_solver.K = args.data_owners
            gurobi_solver.H = args.compute_nodes
            w_start = gurobi_solver.run(release_date.astype(int), proc.astype(int), 
                                        proc_local.astype(int), trans_back.astype(int), 
                                        memory_capacity.astype(int), 
                                        args.log)
      
    # TODO: save parameters into file

    else: # get parameters from file   
        pass

    print('')
    print('                                                 Using ADDM approach')
    
    # call the approach to solve the problem
    violations = []
    gurobi_final_approach.K = args.data_owners
    gurobi_final_approach.H = args.compute_nodes

        
    start = time.time()
    violations, ws = gurobi_final_approach.run(release_date.astype(int), proc.astype(int), 
                                               proc_local.astype(int), trans_back.astype(int), 
                                               memory_capacity.astype(int), memory_demand.astype(int), 
                                               release_date_back.astype(int), proc_bck.astype(int), 
                                               proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                               back_flag, args.log)
    

    end = time.time()
    print(f'final time {end-start}')
    utils.plot_approach(w_start, ws, violations)


