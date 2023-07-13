import argparse
import numpy as np
import pandas as pd
import random
import math
import time

import gurobi_solver
import gurobi_fwd_back
import gurobi_final_approach
import guro_fwd_back
import heuristic
import utils

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--testcase', type=str, default='fully_symmetric', help='fully_symmetric or fully_heterogeneous')
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--data_owners', '-K', type=int, default=50, help='the number of data owners')
    parser.add_argument('--compute_nodes', '-H', type=int, default=2, help='the number of compute nodes')
    parser.add_argument('--splitting_points', '-S', type=str, default='3,23', help='give an input in the form of s1,s2')
    parser.add_argument('--model', '-m', type=str, default='vgg19', help='select model resnet101/vgg19')
    parser.add_argument('--repeat', '-r', type=str, default='', help='avoid generating input include file')
    parser.add_argument('--back', '-b', type=int, default=0, help='0: only fwd, 1: include backpropagation')
    parser.add_argument('--fifo', '-f', type=int, default=0, help='run fifo with load balancer')
    parser.add_argument('--gap', '-g', type=int, default=0, help='run fifo with gap')
    parser.add_argument('--scenario', '-s', type=int, default=1, help='scenario')
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
    
    fifo_flag = False
    if args.fifo == 1:
        fifo_flag = True
    
    gap_flag = False
    if args.gap == 1:
        gap_flag = True
    
    scenario = args.scenario
    
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
            filename = 'real_data/vgg19_CIFAR.xlsx'


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

        max_proc_fwd = int(max([vm_proc_fwd, laptop_proc_fwd]))
        min_proc_fwd = int(min([vm_proc_fwd, laptop_proc_fwd]))

        max_proc_back = int(max([vm_proc_back, laptop_proc_back]))
        min_proc_back = int(min([vm_proc_back, laptop_proc_back]))
        
        # processing time on d1
        d1_data = df_d1.values.tolist()
        d1_proc_fwd_first = 0
        d1_proc_fwd_last = 0
        d1_proc_back_first = 0
        d1_proc_back_last = 0
        '''
        point_a_d1 = 3
        point_b_d1 = 33
        point_a = point_a_d1
        point_b = point_b_d1
        '''
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
        #point_a_jet = 2
        #point_b_jet = 31
        #point_a = point_a_jet
        #point_b = point_b_jet
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
        '''
        point_a_d2 = 3
        point_b_d2 = 33
        point_a = point_a_d1
        point_b = point_b_d1
        '''
        for i in range(0, point_a):
            d2_proc_fwd_first += d2_data[i][0]
            d2_proc_back_first += d2_data[i][1] + d2_data[i][2]

        for i in range(point_b, len(d2_data)):
            d2_proc_fwd_last += d2_data[i][0]
            d2_proc_back_last += d2_data[i][1] + d2_data[i][2]


        print('devices time fwd first:')
        print(f'd1: {d1_proc_fwd_first}')
        print(f'd2: {d2_proc_fwd_first}')
        print(f'jetson-cpu: {jetson_cpu_proc_fwd_first}')
        print(f'gpu1: {jetson_gpu_proc_fwd_first}')
        max_fwd_first = int(max([d1_proc_fwd_first, d2_proc_fwd_first, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_first]))
        min_fwd_first = int(min([d1_proc_fwd_first, d2_proc_fwd_first, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_first]))

        print('devices time fwd last:')
        print(f'd1: {d1_proc_fwd_last}')
        print(f'd2: {d2_proc_fwd_last}')
        print(f'jetson-cpu: {jetson_cpu_proc_fwd_last}')
        print(f'gpu1: {jetson_gpu_proc_fwd_last}')
        max_fwd_last = int(max([d1_proc_fwd_last, d2_proc_fwd_last, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_last]))
        min_fwd_last = int(min([d1_proc_fwd_last, d2_proc_fwd_last, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_last]))

        print('devices time back first:')
        print(f'd1: {d1_proc_back_first}')
        print(f'd2: {d2_proc_back_first}')
        print(f'jetson-cpu: {jetson_cpu_proc_back_first}')
        print(f'gpu1: {jetson_gpu_proc_back_first}')
        max_back_first = int(max([d1_proc_back_first, d2_proc_back_first, jetson_cpu_proc_back_first, jetson_gpu_proc_back_first]))
        min_back_first = int(min([d1_proc_back_first, d2_proc_back_first, jetson_cpu_proc_back_first, jetson_gpu_proc_back_first]))

        print('devices time back last:')
        print(f'd1: {d1_proc_back_last}')
        print(f'd2: {d2_proc_back_last}')
        print(f'jetson-cpu: {jetson_cpu_proc_back_last}')
        print(f'gpu1: {jetson_gpu_proc_back_last}')
        max_back_last = int(max([d1_proc_back_last, d2_proc_back_last, jetson_cpu_proc_back_first, jetson_gpu_proc_back_last]))
        min_back_last = int(min([d1_proc_back_last, d2_proc_back_last, jetson_cpu_proc_back_first, jetson_gpu_proc_back_last]))

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
        store_compute_node = (store_compute_node/1024) # prefer MB

        my_net = lambda data,bandwidth : ((data*0.0008)/bandwidth)*1000
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

        #print(len(set(completed)))

        #for i in range(K):
         #   for j in range(H):
         #       print(network_type[i,j], end='\t')
         #   print('')

        # compute node device type 
        # we have:
        # 0 for vm
        # 1 for laptop

        
        mine_machine = [0,0,0,0,0,0,0,1,1,1]
        
        print('MACHINES')
        machine_devices = np.zeros((H))
        for i in range(H):
            machine_devices[i] = random.randint(0,1)
            print(f'{machine_devices[i]}', end='\t')
            
            #machine_devices[i] = 1
            '''
            if machine_devices[i] == 1:
                for j in range(K):
                    network_type[j,i] = 1
            else:
                for j in range(K):
                    network_type[j,i] = 20
            '''
        print(' ')

        # data owner device type 
        # we have:
        # 0 for d1
        # 1 for d2
        # 2 for jetson gpu
        # 3 for jetson cpu
        do_devices = np.zeros((K))
        for i in range(K):
            do_devices[i] = random.randint(0,1)
            #do_devices[i] = 0
        #do_devices[2] = 1
        # forward parameters
        release_date = np.zeros((K,H))
        release_date_proc = np.zeros((K,H))
        proc = np.zeros((K,H))
        proc_local = np.zeros((K))
        trans_back = np.zeros((K, H))

        # back-propagation parameters
        release_date_back = np.zeros((K,H))
        release_date_back_proc = np.zeros((K,H))
        proc_bck = np.zeros((K,H))
        proc_local_back =np.zeros((K))
        trans_back_gradients = np.zeros((K, H))

        '''
        for i in range(H):
            if machine_devices[i] == 1:
                for j in range(K):
                    network_type[j,i] = int(random.randint(1,5))
            else:
                for j in range(K):
                    network_type[j,i] = int(random.randint(15,20))
        '''

        if scenario == 2:
            release_date_ = np.zeros((K))
            release_date_back_ = np.zeros((K))
            proc_local_ = np.zeros((K))
            proc_local_back_ = np.zeros((K))
            
            proc_fwd_ = np.zeros((H))
            proc_back_ = np.zeros((H))
        
            for j in range(K):
                release_date_[j] =  random.randint(min_fwd_first, max_fwd_first)
                release_date_back_[j] = random.randint(min_back_last, max_back_last)
                proc_local_[j] =  random.randint(min_fwd_last, max_fwd_last)
                proc_local_back_[j] =  random.randint(min_back_first, max_back_first)
            

            release_date_.sort()
            release_date_back_.sort()
            proc_local_.sort()
            proc_local_back_.sort()

            for j in range(K):
                do_devices[j] = int(random.randint(0, K-1))
                

            for i in range(H):
                proc_fwd_[i] =  random.randint(min_proc_fwd, max_proc_fwd)
                proc_back_[i] = random.randint(min_proc_back, max_proc_back)

            proc_fwd_.sort()
            proc_back_.sort()

            for i in range(H):
                machine_devices[i] = int(random.randint(0, H-1))


        for j in range(K):
            for i in range(H):
                indx = int(network_type[j,i])
                
                release_date[j,i] = my_net(activations_to_cn, indx)
                trans_back[j,i] = my_net(activations_to_do, indx)

                release_date_back[j,i] = my_net(activations_to_do, indx)
                trans_back_gradients[j,i] = my_net(activations_to_cn, indx)
                
                if scenario == 1:
                    if int(do_devices[j]) == 0:
                        release_date[j,i] +=  d1_proc_fwd_first
                        release_date_proc[j,i] = d1_proc_fwd_first
                        release_date_back[j,i] += d1_proc_back_last

                        release_date_proc[j,i] = d1_proc_fwd_first
                        release_date_back_proc[j,i] = d1_proc_back_last
                    elif int(do_devices[j]) == 1:
                        release_date[j,i] +=  d2_proc_fwd_first
                        release_date_back[j,i] += d2_proc_back_last

                        release_date_proc[j,i] = d2_proc_fwd_first
                        release_date_back_proc[j,i] = d2_proc_back_last
                    elif int(do_devices[j]) == 3:
                        release_date[j,i] +=  jetson_cpu_proc_fwd_first
                        release_date_back[j,i] += jetson_cpu_proc_back_last

                        release_date_proc[j,i] = jetson_cpu_proc_fwd_first
                        release_date_back_proc[j,i] = jetson_cpu_proc_back_last
                    elif int(do_devices[j]) == 2:
                        release_date[j,i] +=  jetson_gpu_proc_fwd_first
                        release_date_back[j,i] += jetson_gpu_proc_back_last

                        release_date_proc[j,i] = jetson_gpu_proc_fwd_first
                        release_date_back_proc[j,i] = jetson_gpu_proc_back_last

                if scenario == 2:
                    release_date[j,i] +=  release_date_[int(do_devices[j])]
                    release_date_back[j,i] += release_date_back_[int(do_devices[j])]

                    release_date_proc[j,i] = release_date_[int(do_devices[j])]
                    release_date_back_proc[j,i] = release_date_[int(do_devices[j])]

                if i == 0:
                    if scenario == 1:
                        if int(do_devices[j]) == 0:
                            proc_local[j] =  d1_proc_fwd_last
                            proc_local_back[j] =  d1_proc_back_first
                        elif int(do_devices[j]) == 1:
                            proc_local[j] =  d2_proc_fwd_last
                            proc_local_back[j] =  d2_proc_back_first
                        elif int(do_devices[j]) == 2:
                            proc_local[j] =  jetson_gpu_proc_fwd_last
                            proc_local_back[j] =  jetson_gpu_proc_back_first
                        elif int(do_devices[j]) == 3:
                            proc_local[j] =  jetson_cpu_proc_fwd_last
                            proc_local_back[j] =  jetson_cpu_proc_back_first
                            

                    if scenario == 2:
                        proc_local[j] =  proc_local_[int(do_devices[j])]
                        proc_local_back[j] =  proc_local_back[int(do_devices[j])]

   
                if scenario == 1:
                    if int(machine_devices[i]) == 0:
                        proc[j,i] =  vm_proc_fwd
                        proc_bck[j,i] =  vm_proc_back
                    elif int(machine_devices[i]) == 1:
                        proc[j,i] = laptop_proc_fwd
                        proc_bck[j,i] = laptop_proc_back

                if scenario == 2:
                    proc[j,i] =  proc_fwd_[int(machine_devices[i])]
                    proc_bck[j,i] = proc_back_[int(machine_devices[i])]

                        
         
                        
        memory_demand = np.ones((K)) * store_compute_node
        for i in range(K):
            memory_demand[i] = int(math.ceil(memory_demand[i]))

        #print(f'Memory --- {memory_demand}')
        utils.max_memory_demand = int(max(memory_demand))
        print("MEMORY CAPACITY")
        memory_capacity = np.array(utils.get_memory_characteristics(H, K))
        for i in range(len(memory_capacity)):
            memory_capacity[i] = int(memory_capacity[i])
            print(f'{int(memory_capacity[i])/int(utils.max_memory_demand)}', end=',\t')
            #memory_capacity[i] = int(max(memory_demand))*K
        print(' ')
        
        #array_mine = [1.0,	1.0,	14.0,	18.0,	1.0,	1.0,	1.0,    1.0,	1.0,	1.0]
        
        array_mine = [1, 1, 1, 1, 1, 1, 1, 4, 4, 5]
        '''
        for i in range(H):
            #memory_capacity[i] = int(array_mine[i]*utils.max_memory_demand)
            #memory_capacity[i] = int(array_mine[i]*utils.max_memory_demand)
            memory_capacity[i] = int(max(memory_demand))*K
            print(f'{int(memory_capacity[i])/int(utils.max_memory_demand)}', end=',\t')
            
        print(' ')
        '''
        unique_friends = []
        
        #print('proc')
        for i in range(H):
         #   print(f'{proc[0,i]}--{int(machine_devices[i])}', end='\t')
         #   print(f'{proc_bck[0,i]}--{int(machine_devices[i])}', end='\t')
         #   print('~~~~~')
            unique_friends.append(int(np.rint(proc[0,i])))
            unique_friends.append(int(np.rint(proc_bck[0,i])))
        #print('')

        #print('release date')
        for j in range(K):
            for i in range(H):
         #       print(f'{release_date[j,i]}--{int(network_type[j,i])}--{int(do_devices[j])}', end='\t')
                unique_friends.append(int(np.rint(release_date[j,i])))
                unique_friends.append(int(np.rint(release_date_back[j,i])))
          #  print('')

       # print('release date back')
        for j in range(K):
            for i in range(H):
                print(f'{release_date_back[j,i]}', end='\t')
        #    print('')

        #print('proc local: ')
        for j in range(K):
         #   print(f'{proc_local[j]}', end='\t')
            unique_friends.append(int(np.rint(proc_local[j])))
            unique_friends.append(int(np.rint(proc_local_back[j])))
        #print('')

        #print('trans back: ')
        for j in range(K):
            for i in range(H):
         #       print(f'{trans_back[j,i]}', end='\t')
                unique_friends.append(int(np.rint(trans_back[j,i])))
                unique_friends.append(int(np.rint(trans_back_gradients[j,i])))
          #  print('')
        
        print('\n----------------------     max values:  -------------------\n')
        max_value = max(unique_friends)
        print('-------------------')
        #print(unique_friends)
        print('')
        print(f'max is {max_value}')
        print('\n----------------------     max values:  -------------------\n')
        
        # Re-difine parameters
        max_slot = 50
        max_slot_back = max_slot


        for j in range(K):
            for i in range(H):
                
                release_date[j,i] = np.ceil((release_date[j,i]*max_slot)/max_value).astype(int)
                trans_back[j,i] = np.ceil((trans_back[j,i]*max_slot)/max_value).astype(int)
                
                #release_date_back[j,i] = np.rint((release_date_back[j,i]*max_slot_back)/max_value_back).astype(int)
                #trans_back_gradients[j,i] = np.rint((trans_back_gradients[j,i]*max_slot_back)/max_value_back).astype(int)
                release_date_back[j,i] = np.ceil((release_date_back[j,i]*max_slot)/max_value).astype(int)
                trans_back_gradients[j,i] = np.ceil((trans_back_gradients[j,i]*max_slot)/max_value).astype(int)

                if i == 0:
                    proc_local[j] = np.ceil((proc_local[j]*max_slot)/max_value).astype(int)
                    #proc_local_back[j] = np.rint((proc_local_back[j]*max_slot_back)/max_value_back).astype(int)
                    proc_local_back[j] = np.ceil((proc_local_back[j]*max_slot)/max_value).astype(int)

                proc[j,i] =  np.ceil((proc[j,i]*max_slot)/max_value).astype(int)
                
                if proc[j,i] == 0:
                        proc[j,i] = 1

                #proc_bck[j,i] =  np.rint((proc_bck[j,i]*max_slot_back)/max_value_back).astype(int)
                proc_bck[j,i] =  np.ceil((proc_bck[j,i]*max_slot)/max_value).astype(int)

                if proc_bck[j,i] == 0:
                        proc_bck[j,i] = 1


        
        print('                                             NEW')
        print('proc')
        for i in range(H):
            print(f'{proc[0,i]}', end='\t')
            print(f'{proc_bck[0,i]} --- {machine_devices[i]}', end='\t')
            print('~~~~~~')
        print('')
        
        print('release date')
        for j in range(K):
            for i in range(H):
                print(f'{release_date[j,i]} -- {do_devices[j]}', end='\t')
            print('')

        print('release date back')
        for j in range(K):
            for i in range(H):
                print(f'back: {release_date_back[j,i]}', end='\t')
            print('')

        print('proc local: ')
        for j in range(K):
            print(f'{proc_local[j]}', end='\t')
        print('')

        print('proc local back: ')
        for j in range(K):
            print(f'back: {proc_local_back[j]}', end='\t')
        print('')

        print('trans back: ')
        for j in range(K):
            for i in range(H):
                print(f'{trans_back[j,i]}', end='\t')
            print('')

        print('trans back back: ')
        for j in range(K):
            for i in range(H):
                print(f'back: {trans_back_gradients[j,i]}', end='\t')
            print('')
        
        # call the original solver for comparison
        if fifo_flag:
            print('Calling fifo')
            heuristic.K = args.data_owners
            heuristic.H = args.compute_nodes
            gurobi_final_approach.K = args.data_owners
            gurobi_final_approach.H = args.compute_nodes

            if back_flag:
                
                y = gurobi_final_approach.run(release_date.astype(int), proc.astype(int), 
                                                proc_local.astype(int), trans_back.astype(int), 
                                                memory_capacity.astype(int), memory_demand.astype(int), 
                                                release_date_back.astype(int), proc_bck.astype(int), 
                                                proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                                back_flag, args.log, fifo_flag)
                
                w_start = heuristic.balance_run(release_date, proc, proc_local, trans_back, 
                                release_date_back, proc_bck, proc_local_back, trans_back_gradients, 
                                           memory_capacity.astype(int), memory_demand[0].astype(int), y, True)
        
            else:
                w_start = heuristic.balance_run(release_date, proc, proc_local, trans_back, 
                                release_date_back, proc_bck, proc_local_back, trans_back_gradients, 
                                           memory_capacity.astype(int), memory_demand[0].astype(int))
        elif gap_flag:

            print('Calling GAP')
            heuristic.K = args.data_owners
            heuristic.H = args.compute_nodes
        
            w_start = heuristic.gap_run(release_date, proc, proc_local, trans_back, 
                                release_date_back, proc_bck, proc_local_back, trans_back_gradients,
                                release_date_proc, release_date_back_proc,
                                memory_capacity.astype(int), memory_demand[0].astype(int))    
        else:
            if back_flag:
                
                print('Calling the original approach. -- BACK')
                gurobi_fwd_back.K = args.data_owners
                gurobi_fwd_back.H = args.compute_nodes
                w_start = 127
                '''
                w_start = gurobi_fwd_back.run(release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), 
                                            release_date_back.astype(int), proc_bck.astype(int), 
                                            proc_local_back.astype(int), trans_back_gradients.astype(int), 
                                            args.log)
                '''
                
            else:
                print('Calling the original approach.')
                gurobi_solver.K = args.data_owners
                gurobi_solver.H = args.compute_nodes
                w_start = 59
                
                w_start = gurobi_solver.run(release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), 
                                            args.log)
                
    # TODO: save parameters into file

    else: # get parameters from file   
        pass

    print('')
    
    if (not fifo_flag) and (not gap_flag):
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


