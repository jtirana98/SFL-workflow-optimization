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
import gurobi_for_subproblems as sub

def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--testcase', type=str, default='fully_symmetric', help='fully_symmetric or fully_heterogeneous')
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--data_owners', '-K', type=int, default=50, help='the number of data owners')
    parser.add_argument('--compute_nodes', '-H', type=int, default=2, help='the number of compute nodes')
    parser.add_argument('--splitting_points', '-S', type=str, default='3,33', help='give an input in the form of s1,s2')
    parser.add_argument('--model', '-m', type=str, default='resnet101', help='select model resnet101/vgg19')
    parser.add_argument('--repeat', '-e', type=str, default='', help='avoid generating input include file')
    parser.add_argument('--back', '-b', type=int, default=0, help='0: only fwd, 1: include backpropagation')
    parser.add_argument('--fifo', '-f', type=int, default=0, help='run fifo with load balancer')
    parser.add_argument('--gap', '-g', type=int, default=0, help='run fifo with gap')
    parser.add_argument('--random', '-r', type=int, default=0, help='run fifo with raandom')
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

    random_flag = False
    if args.random == 1:
        random_flag = True
    
    scenario = args.scenario
    
    splitting_points = args.splitting_points

    points = list(splitting_points.split(','))
    point_a = int(points[0])
    point_b = int(points[1])

    point_a = np.ones(K).astype(int)
    point_b = np.ones(K).astype(int)

    
    

    print(f'I have your splitting points after {point_a} and {point_b}, including')
    
    if args.repeat == '':
        # create the parameter table
        model_type = args.model

        if model_type == 'resnet101':
            filename = 'real_data/resnet101_CIFAR.xlsx'
        elif model_type == 'vgg19':
            filename = 'real_data/vgg19_CIFAR.xlsx'

        if model_type == 'resnet101': 
            for i in range(int(K/2)):
                point_a[i] = 9
                point_b[i] = 33

            for i in range(int(K/2), K):
                point_a[i] = 17
                point_b[i] = 33
        
        else:    
            for i in range(int(K/2)):
                point_a[i] = 9
                point_b[i] = 23

            for i in range(int(K/2), K):
                point_a[i] = 5
                point_b[i] = 20


        df_vm = pd.read_excel(io=filename, sheet_name='VM', header=None)
        df_laptop = pd.read_excel(io=filename, sheet_name='laptop', header=None)
        df_d1 = pd.read_excel(io=filename, sheet_name='d1', header=None)
        df_d2 = pd.read_excel(io=filename, sheet_name='d2', header=None)
        df_jetson_cpu = pd.read_excel(io=filename, sheet_name='jetson-cpu', header=None)
        df_jetson_gpu = pd.read_excel(io=filename, sheet_name='jetson-gpu', header=None)
        df_memory = pd.read_excel(io=filename, sheet_name='memory', header=None)
        

        # processing time on vms
        vm_data = df_vm.values.tolist()

        vm_proc_fwd = np.zeros(K)
        vm_proc_back = np.zeros(K)
        for k in range(K):
            for i in range(point_a[k], point_b[k]):
                vm_proc_fwd[k] += vm_data[i][0]
                vm_proc_back[k] += vm_data[i][1] + vm_data[i][2]

        # processing time on my laptop
        laptop_data = df_laptop.values.tolist()
        
        laptop_proc_fwd = np.zeros(K)
        laptop_proc_back = np.zeros(K)
        for k in range(K):
            for i in range(point_a[k], point_b[k]):
                laptop_proc_fwd[k] += laptop_data[i][0]
                laptop_proc_back[k] += laptop_data[i][1] + laptop_data[i][2]


        # processing time on d1
        d1_data = df_d1.values.tolist()
        
        d1_proc_fwd_first = np.zeros(K)
        d1_proc_fwd_last = np.zeros(K)
        d1_proc_back_first = np.zeros(K)
        d1_proc_back_last = np.zeros(K)

        for k in range(K):
            for i in range(0, point_a[k]):
                d1_proc_fwd_first[k] += d1_data[i][0]
                d1_proc_back_first[k] += d1_data[i][1] + d1_data[i][2]
            
            for i in range(point_b[k], len(d1_data)):
                d1_proc_fwd_last[k] += d1_data[i][0]
                d1_proc_back_last[k] += d1_data[i][1] + d1_data[i][2]
        '''
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
        '''
        # processing time on jetson-gpu
        jetson_gpu_data = df_jetson_gpu.values.tolist()

        jetson_gpu_proc_fwd_first = np.zeros(K)
        jetson_gpu_proc_fwd_last = np.zeros(K)
        jetson_gpu_proc_back_first = np.zeros(K)
        jetson_gpu_proc_back_last = np.zeros(K)
        
        for k in range(K):
            for i in range(0, point_a[k]):
                jetson_gpu_proc_fwd_first[k] += jetson_gpu_data[i][0]
                jetson_gpu_proc_back_first[k] += jetson_gpu_data[i][1] + jetson_gpu_data[i][2]
            
            for i in range(point_b[k], len(jetson_gpu_data)):
                jetson_gpu_proc_fwd_last[k] += jetson_gpu_data[i][0]
                jetson_gpu_proc_back_last[k] += jetson_gpu_data[i][1] + jetson_gpu_data[i][2]

        '''
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


        '''

        memory_data = df_memory.values.tolist()
        activations_to_cn = np.zeros(K)
        activations_to_do = np.zeros(K)

        # travel data
        for k in range(K):
            activations_to_cn[k] = memory_data[point_a[k]-1][0]
            activations_to_do[k] = memory_data[point_b[k]-1][0]

        # store data
        '''
        store_data_owner = 0
        for i in range(0, point_a):
            store_data_owner += memory_data[i][0] + memory_data[i][1]
            
        for i in range(point_b, len(d1_data)):
            store_data_owner += memory_data[i][0] + memory_data[i][1]
        '''

        store_compute_node = np.zeros(K)

        print('MEMORY DEMANDS ON CN')
        for k in range(K):
            for i in range(point_a[k], point_b[k]):
                #print(f'{memory_data[i][0]}  {memory_data[i][1]}')
                store_compute_node[k] += memory_data[i][0] + memory_data[i][1]
            store_compute_node[k] = (store_compute_node[k]/1024) # prefer MB
            #print('--------------------------------')
        print(store_compute_node)

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

        for i in range(K):
            for j in range(H):
                network_type[i][j] = 7
        
      
        
        for i in range(K-H+1, K):
            network_type[i][H-1] = 1
        
               
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
        '''
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
            machine_devices[i] = 1
            print(f'{machine_devices[i]}', end='\t')
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
            if do_devices[i] == 1:
                do_devices[i] = 2
            #do_devices[i] = 0
        #do_devices[2] = 1

        for i in range(K-H+1, K):
            do_devices[i] = 0

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

        for j in range(K):
            for i in range(H):
                indx = int(network_type[j,i])
                
                release_date[j,i] = my_net(activations_to_cn[j], indx)
                trans_back[j,i] = my_net(activations_to_do[j], indx)

                release_date_back[j,i] = my_net(activations_to_do[j], indx)
                trans_back_gradients[j,i] = my_net(activations_to_cn[j], indx)
                
                if do_devices[j] == 2:
                    release_date[j,i] +=  jetson_gpu_proc_fwd_first[j]
                    release_date_back[j,i] += jetson_gpu_proc_back_last[j]

                    release_date_proc[j,i] = jetson_gpu_proc_fwd_first[j]
                    release_date_back_proc[j,i] = jetson_gpu_proc_back_last[j]
                else:
                    release_date[j,i] +=  d1_proc_fwd_first[j]
                    release_date_back[j,i] += d1_proc_back_last[j]

                    release_date_proc[j,i] = d1_proc_fwd_first[j]
                    release_date_back_proc[j,i] = d1_proc_back_last[j]
                        
                if i == 0:
                    if do_devices[j] == 2:
                        proc_local[j] =  jetson_gpu_proc_fwd_last[j]
                        proc_local_back[j] =  jetson_gpu_proc_back_first[j]
                    else:
                        proc_local[j] =  d1_proc_fwd_last[j]
                        proc_local_back[j] =  d1_proc_back_first[j]
                
                
                proc[j,i] =  vm_proc_fwd[j]
                proc_bck[j,i] =  vm_proc_back[j]
                
                if i == H-1:
                    proc[j,i] =  vm_proc_fwd[j]*5
                    proc_bck[j,i] =  vm_proc_back[j]*5
         
                        
        memory_demand =  store_compute_node
        '''
        for i in range(K):
            memory_demand[i] = int(math.ceil(memory_demand[i]))
        '''
        #print(f'Memory --- {memory_demand}')
        utils.max_memory_demand = int(max(memory_demand))
        print("MEMORY CAPACITY")
        #memory_capacity = np.array(utils.get_memory_characteristics(H, K))

        
        if model_type == 'resnet101': 
            if H == 5:
                memory_capacity = np.array([224,224,224,224,K*171])
            else:
                memory_capacity = np.array([224,224,224,224,224,224,224,224,224,K*171])
        else:    
            if H == 5:
                memory_capacity = np.array([537,537,537,537,K*526])
            else:
                memory_capacity = np.array([537,537,537,537,537,537,537,537,537,K*526])

        

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
        
        
        for i in range(K-H+1, K):
            for j in range(H-1):
                release_date[i][j] += 10
                release_date_back[i][j] += 10
                proc_local[i] += 10
                proc_local_back[i] += 10
            

        
        
        print('                                             NEW')
        print('proc')
        for i in range(H):
            print(f'{proc[0,i]}', end='\t')
            print(f'{proc_bck[0,i]} --- {machine_devices[i]}', end='\t')
            print('~~~~~~')
        print('')

        print('proc FAST')
        for i in range(H):
            print(f'{proc[10,i]}', end='\t')
            print(f'{proc_bck[10,i]} --- {machine_devices[i]}', end='\t')
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
                                           memory_capacity, memory_demand)
        elif gap_flag:

            print('Calling GAP')
            heuristic.K = args.data_owners
            heuristic.H = args.compute_nodes
        
            w_start = heuristic.gap_run(release_date, proc, proc_local, trans_back, 
                                release_date_back, proc_bck, proc_local_back, trans_back_gradients,
                                release_date_proc, release_date_back_proc,
                                memory_capacity.astype(int), memory_demand.astype(int))    
        
        elif random_flag:
            print('Calling RANDOM')
            heuristic.K = args.data_owners
            heuristic.H = args.compute_nodes
    
            w_start = heuristic.random_run(release_date, proc, proc_local, trans_back, 
                                release_date_back, proc_bck, proc_local_back, trans_back_gradients, 
                                           memory_capacity.astype(int), memory_demand.astype(int))
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
                #       UNCOMMENT FOR    FWD ONLY OPTIMAL AND BACK SUB  AND -b 0
                (y,x,f_t) = gurobi_solver.run(release_date.astype(int), proc.astype(int), 
                                            proc_local.astype(int), trans_back.astype(int), 
                                            memory_capacity.astype(int), memory_demand.astype(int),
                                            args.log)
                
                print('------------------------------ALLOCATION - Y OPTIMAL-------------------------------------')
                print(y)

                T_back = np.max(release_date.astype(int)) + K*np.max(proc[0,:].astype(int)) + np.max(release_date_back.astype(int)) + K*np.max(proc_bck[0,:].astype(int)) \
                        + np.max(proc_local.astype(int)) + np.max(proc_local_back.astype(int)) \
                        + np.max(np.max(trans_back.astype(int))) + np.max(np.max(trans_back_gradients.astype(int)))

                z_par = np.zeros((H,K,T_back))

                Tx = np.max(release_date.astype(int)) + K*np.max(proc.astype(int))
                for i in range(H):
                    Kx = list(np.transpose(np.argwhere(y[:,i]==1))[0])
                    if len(Kx) == 0:
                        continue
                    print(f'For machine {i+1} I have allocatedd: {Kx}')
                    procx = np.copy(proc[Kx, i])  # this is a row-vector
                    release_datex = np.copy(release_date[Kx, i])
                    proc_localx = np.copy(proc_local[Kx])
                    trans_backx = np.copy(trans_back[Kx, i])
                    
                    
                    f_temp = np.zeros((len(Kx)))
                    jj = 0
                    for j in Kx:
                        f_temp[jj] = f_t[j]
                        print(f_temp[jj])
                        jj += 1
                        

                    '''
                    print('finish times')
                    print(Tx)
                    for kk in range(len(Kx)):
                        for t in range(Tx):
                            if f_temp[kk] < (t+1)*x[i,kk,t]:
                                f_temp[kk] = (t+1)*x[i,kk,t]
                        print(f_temp[kk])
                    '''
                    #min_f = min(f_temp)
                    
                    procz = np.copy(proc_bck[Kx, i])  # this is a row-vector
                    release_datez = np.copy(release_date_back[Kx, i])
                    proc_localz = np.copy(proc_local_back[Kx])
                    trans_backz = np.copy(trans_back_gradients[Kx, i])

                    for kk in range(len(Kx)):
                        #release_datez[kk] += (f_temp[kk] - min_f) + proc_localx[kk] + trans_backx[kk]
                        release_datez[kk] += f_temp[kk] + proc_localx[kk] + trans_backx[kk]
                    
                    Tz = np.max(release_datez.astype(int)) + len(Kx)*np.max(procz.astype(int))  # to constrain the T
                    x__extend = np.zeros((1,len(Kx),Tz))
   
                    print(f'---- Tx {Tx} ---- Tz {Tz}')
                    
                    jj = 0
                    for j in Kx:
                        for t in range(min(Tx,Tz)):
                            x__extend[0,jj,t] = x[i,j,t]
                        jj += 1
                    
                    start_sub = time.time()
                    #z__ = sub.for_each_machine(len(Kx), release_datez, procz, proc_localz, trans_backz, memory_capacity[i], Tz)
                    z__ = sub.for_each_machine_back(len(Kx), release_datez.astype(int), procz.astype(int), proc_localz.astype(int), trans_backz.astype(int), memory_capacity[i].astype(int), Tz, x__extend)

                    jj = 0
                    for j in Kx:
                        for t in range(Tz):
                            z_par[i,j,t] = z__[0,jj,t]
                        jj += 1

                cs_back = []
                for i in range(K): #for all jobs
                    my_machine = 0
                    my_super_machine = 0
                    last_zero = -1
                    for my_machine in range(H):
                        for k in range(T_back):
                            if np.rint(z_par[my_machine,i,k]) >= 1:
                                if last_zero < k+1:
                                    last_zero = k+1
                                    my_super_machine = my_machine
                    fmax = last_zero
                    C = fmax + proc_local_back[i] + trans_back_gradients[i,my_super_machine]
                    cs_back.append(C)

                print(f'{utils.bcolors.FAIL}BACK max is: {max(cs_back)}{utils.bcolors.ENDC}')
             #           FWD ONLY OPTIMAL AND BACK SUB  END
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


