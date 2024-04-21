import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

max_memory_demand = 3
file_name = 'test4.xlsx'

class arrival_date:
    def __init__(self, value, back, job):
        self.value = value
        self.back = back
        self.job = job

def my_net(data,bandwidth):
    delay = ((data*0.0008)/bandwidth)*1000

    if bandwidth == 1:
        delay = delay*10
    
    return delay

def create_scenario(filename, point_a, point_b, K, H, scenario, max_slot):
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

    max_proc_fwd = int(max([vm_proc_fwd]))
    min_proc_fwd = int(min([vm_proc_fwd]))

    max_proc_back = int(max([vm_proc_back, laptop_proc_back]))
    min_proc_back = int(min([vm_proc_back, laptop_proc_back]))
    
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

    max_fwd_first = int(max([d1_proc_fwd_first, d2_proc_fwd_first, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_first]))
    min_fwd_first = int(min([d1_proc_fwd_first, d2_proc_fwd_first, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_first]))

    max_fwd_last = int(max([d1_proc_fwd_last, d2_proc_fwd_last, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_last]))
    min_fwd_last = int(min([d1_proc_fwd_last, d2_proc_fwd_last, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_last]))

    max_back_first = int(max([d1_proc_back_first, d2_proc_back_first, jetson_cpu_proc_back_first, jetson_gpu_proc_back_first]))
    min_back_first = int(min([d1_proc_back_first, d2_proc_back_first, jetson_cpu_proc_back_first, jetson_gpu_proc_back_first]))

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
    original_state = np.random.get_state()
    random.seed(42)

    # randomly select the network connections using the Atari stats
    network_type = np.zeros((K,H))

    '''
    We consider the following cases:

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

    # helper device type 
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
    proc_local_back = np.zeros((K))
    trans_back_gradients = np.zeros((K, H))

    # Construct scenario

    if scenario == 2: # heterogeneous
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

    memory_demand_ = np.ones((K)) * store_compute_node
    for i in range(K):
        memory_demand_[i] = int(math.ceil(memory_demand_[i]))
    
    global max_memory_demand
    max_memory_demand = int(max(memory_demand_))

    memory_capacity = np.array(get_memory_characteristics(H, K))
    for i in range(H):
        memory_capacity[i] = int(max(memory_demand_))*K

    unique_values = []
    for j in range(K):
        for i in range(H):
            unique_values.append(int(np.rint(release_date[j,i])))
            unique_values.append(int(np.rint(release_date_back[j,i])))

    for j in range(K):
        unique_values.append(int(np.rint(proc_local[j])))
        unique_values.append(int(np.rint(proc_local_back[j])))


    for j in range(K):
        for i in range(H):
            unique_values.append(int(np.rint(trans_back[j,i])))
            unique_values.append(int(np.rint(trans_back_gradients[j,i])))

    max_value = max(unique_values)

    # Re-difine parameters as splots
    max_slot_back = max_slot
 
    for j in range(K):
        for i in range(H):
            
            release_date[j,i] = np.ceil((release_date[j,i]*max_slot)/max_value).astype(int)
            trans_back[j,i] = np.ceil((trans_back[j,i]*max_slot)/max_value).astype(int)
            
            release_date_back[j,i] = np.ceil((release_date_back[j,i]*max_slot)/max_value).astype(int)
            trans_back_gradients[j,i] = np.ceil((trans_back_gradients[j,i]*max_slot)/max_value).astype(int)

            if i == 0:
                proc_local[j] = np.ceil((proc_local[j]*max_slot)/max_value).astype(int)
                proc_local_back[j] = np.ceil((proc_local_back[j]*max_slot)/max_value).astype(int)

            proc[j,i] =  np.ceil((proc[j,i]*max_slot)/max_value).astype(int)
            
            if proc[j,i] == 0:
                    proc[j,i] = 1

            proc_bck[j,i] =  np.ceil((proc_bck[j,i]*max_slot)/max_value).astype(int)

            if proc_bck[j,i] == 0:
                    proc_bck[j,i] = 1
    
    return (release_date, proc, 
    proc_local, trans_back, 
    memory_capacity, memory_demand_, 
    release_date_back, proc_bck, 
    proc_local_back, trans_back_gradients)

def create_scenario_hybrid(filename, point_a, point_b, K, H, 
                                 max_slot, slow_client, slow_networks, scenario):
    

    df_vm = pd.read_excel(io=filename, sheet_name='VM', header=None)
    df_laptop = pd.read_excel(io=filename, sheet_name='laptop', header=None)
    df_d1 = pd.read_excel(io=filename, sheet_name='d1', header=None)
    df_jetson_cpu = pd.read_excel(io=filename, sheet_name='jetson-cpu', header=None)
    df_jetson_gpu = pd.read_excel(io=filename, sheet_name='jetson-gpu', header=None)
    df_memory = pd.read_excel(io=filename, sheet_name='memory', header=None)
    
    H_prime = H + K
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

    max_proc_fwd = int(max([vm_proc_fwd, laptop_proc_fwd/100]))
    min_proc_fwd = int(min([vm_proc_fwd, laptop_proc_fwd/100]))

    max_proc_back = int(max([vm_proc_back, laptop_proc_back/100]))
    min_proc_back = int(min([vm_proc_back, laptop_proc_back/100]))
    
    # processing time on d1
    d1_data = df_d1.values.tolist()
    d1_proc_fwd_first = 0
    d1_proc_fwd_medium = 0
    d1_proc_fwd_last = 0
    d1_proc_back_first = 0
    d1_proc_back_medium = 0
    d1_proc_back_last = 0


    for i in range(0, point_a):
            d1_proc_fwd_first += d1_data[i][0]
            d1_proc_back_first += d1_data[i][1] + d1_data[i][2]

    for i in range(point_a, point_b):
        d1_proc_fwd_medium += d1_data[i][0]
        d1_proc_back_medium += d1_data[i][1] + d1_data[i][2]
        
    for i in range(point_b, len(d1_data)):
        d1_proc_fwd_last += d1_data[i][0]
        d1_proc_back_last += d1_data[i][1] + d1_data[i][2]

    # processing time on jetson-cpu
    jetson_cpu_data = df_jetson_cpu.values.tolist()

    jetson_cpu_proc_fwd_first = 0
    jetson_cpu_proc_fwd_medium = 0
    jetson_cpu_proc_fwd_last = 0
    jetson_cpu_proc_back_first = 0
    jetson_cpu_proc_back_medium = 0
    jetson_cpu_proc_back_last = 0
    
    for i in range(0, point_a):
        jetson_cpu_proc_fwd_first += jetson_cpu_data[i][0]
        jetson_cpu_proc_back_first += jetson_cpu_data[i][1] + jetson_cpu_data[i][2]
    
    for i in range(point_a, point_b):
        jetson_cpu_proc_fwd_medium += jetson_cpu_data[i][0]
        jetson_cpu_proc_back_medium += jetson_cpu_data[i][1] + jetson_cpu_data[i][2]

    
    for i in range(point_b, len(jetson_cpu_data)):
        jetson_cpu_proc_fwd_last += jetson_cpu_data[i][0]
        jetson_cpu_proc_back_last += jetson_cpu_data[i][1] + jetson_cpu_data[i][2]

    # processing time on jetson-gpu
    jetson_gpu_data = df_jetson_gpu.values.tolist()

    jetson_gpu_proc_fwd_first = 0
    jetson_gpu_proc_fwd_medium = 0
    jetson_gpu_proc_fwd_last = 0
    jetson_gpu_proc_back_first = 0
    jetson_gpu_proc_back_medium = 0
    jetson_gpu_proc_back_last = 0

    for i in range(0, point_a):
        jetson_gpu_proc_fwd_first += jetson_gpu_data[i][0]
        jetson_gpu_proc_back_first += jetson_gpu_data[i][1] + jetson_gpu_data[i][2]

    for i in range(point_a, point_b):
        jetson_gpu_proc_fwd_medium += jetson_gpu_data[i][0]
        jetson_gpu_proc_back_medium += jetson_gpu_data[i][1] + jetson_gpu_data[i][2]
    
    for i in range(point_b, len(jetson_gpu_data)):
        jetson_gpu_proc_fwd_last += jetson_gpu_data[i][0]
        jetson_gpu_proc_back_last += jetson_gpu_data[i][1] + jetson_gpu_data[i][2]


    max_fwd_first = int(max([d1_proc_fwd_first, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_first]))
    min_fwd_first = int(min([d1_proc_fwd_first, jetson_cpu_proc_fwd_first, jetson_gpu_proc_fwd_first]))

    max_fwd_last = int(max([d1_proc_fwd_last, jetson_cpu_proc_fwd_last, jetson_gpu_proc_fwd_last]))
    min_fwd_last = int(min([d1_proc_fwd_last, jetson_cpu_proc_fwd_last, jetson_gpu_proc_fwd_last]))
    
    max_fwd_medium = int(max([d1_proc_fwd_medium, jetson_cpu_proc_fwd_medium, jetson_gpu_proc_fwd_medium]))
    min_fwd_medium = int(min([d1_proc_fwd_medium, jetson_cpu_proc_fwd_medium, jetson_gpu_proc_fwd_medium]))

    max_back_medium = int(max([d1_proc_back_medium, jetson_cpu_proc_back_medium, jetson_gpu_proc_back_medium]))
    min_back_medium = int(min([d1_proc_back_medium, jetson_cpu_proc_back_medium, jetson_gpu_proc_back_medium]))

    max_back_first = int(max([d1_proc_back_first, jetson_cpu_proc_back_first, jetson_gpu_proc_back_first]))
    min_back_first = int(min([d1_proc_back_first, jetson_cpu_proc_back_first, jetson_gpu_proc_back_first]))

    max_back_last = int(max([d1_proc_back_last, jetson_cpu_proc_back_first, jetson_gpu_proc_back_last]))
    min_back_last = int(min([d1_proc_back_last,jetson_cpu_proc_back_first, jetson_gpu_proc_back_last]))

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
    original_state = np.random.get_state()
    random.seed(42)

    # randomly select the network connections using the Atari stats
    network_type = np.zeros((K,H))

    '''
    We consider the following cases:

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

    # helper device type 
    # we have:
    # 0 for vm
    # 1 for laptop
    
    machine_devices = np.zeros((H_prime))
    for i in range(H):
        machine_devices[i] = 1#random.randint(0,1)


    # data owner device type 
    # we have:
    # 0 for d1
    # 1 for jetson cpu
    # 2 for jetson gpu
    
    do_devices = np.zeros((K))
    for i in range(K):
        do_devices[i] = random.randint(0,1)
        #do_devices[i] = 0
    #do_devices[2] = 1
    
    release_date = [np.zeros((K,H)), np.zeros((K,H_prime))]
    proc = [np.zeros((K,H)), np.zeros((K,H_prime))] 
    proc_local = [np.zeros((K)), np.zeros((K))]
    trans_back = [np.zeros((K, H)), np.zeros((K, H_prime))] 
    release_date_back = [np.zeros((K,H)), np.zeros((K,H_prime))]
    proc_bck = [np.zeros((K,H)), np.zeros((K,H_prime))] 
    proc_local_back = [np.zeros((K)), np.zeros((K))]
    trans_back_gradients = [np.zeros((K, H)), np.zeros((K, H_prime))]


    # Construct scenario

    if scenario == 2: # heterogeneous
        release_date_ = np.zeros((K))
        release_date_back_ = np.zeros((K))
        proc_local_ = np.zeros((K))
        proc_local_back_ = np.zeros((K))
        
        proc_fwd_ = np.zeros((H_prime))
        proc_back_ = np.zeros((H_prime))
    
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
            

        for i in range(H_prime):
            if i < H:
                proc_fwd_[i] =  random.randint(min_proc_fwd, max_proc_fwd)
                proc_back_[i] = random.randint(min_proc_back, max_proc_back)
            else:
                proc_fwd_[i] =  random.randint(min_fwd_medium, max_fwd_medium)
                proc_back_[i] = random.randint(min_back_medium, max_back_medium)

        #proc_fwd_.sort()
        #proc_back_.sort()

        for i in range(H_prime):
            if i < H:
                machine_devices[i] = int(random.randint(0, H-1))
            else:
                machine_devices[i] = int(random.randint(H, H_prime-1))

    for j in range(K):
        for i in range(H_prime):
            if i < H:
                indx = int(network_type[j,i])
                
                for k in range(2):
                    slow = 1
                    if indx <= 1:
                        slow = 5
                    release_date[k][j,i] = my_net(activations_to_cn, indx)#*slow
                    trans_back[k][j,i] = my_net(activations_to_do, indx)#*slow

                    release_date_back[k][j,i] = my_net(activations_to_do, indx)#*slow
                    trans_back_gradients[k][j,i] = my_net(activations_to_cn, indx)#*slow
            
            if scenario == 1:
                if int(do_devices[j]) == 0:
                    if i < H:
                        release_date[0][j,i] +=  d1_proc_fwd_first
                        release_date_back[0][j,i] += d1_proc_back_last
                    
                    release_date[1][j,i] +=  d1_proc_fwd_first
                    release_date_back[1][j,i] += d1_proc_back_last

                elif int(do_devices[j]) == 1:
                    if i < H:
                        release_date[0][j,i] +=  jetson_cpu_proc_fwd_first
                        release_date_back[0][j,i] += jetson_cpu_proc_back_last
                    
                    release_date[1][j,i] +=  jetson_cpu_proc_fwd_first
                    release_date_back[1][j,i] += jetson_cpu_proc_back_last
                
                elif int(do_devices[j]) == 2:
                    if i < H:    
                        release_date[0][j,i] +=  jetson_gpu_proc_fwd_first
                        release_date_back[0][j,i] += jetson_gpu_proc_back_last
                    
                    release_date[1][j,i] +=  jetson_gpu_proc_fwd_first
                    release_date_back[1][j,i] += jetson_gpu_proc_back_last

            if scenario == 2:
                if i < H:
                    release_date[0][j,i] +=  release_date_[int(do_devices[j])]
                    release_date_back[0][j,i] += release_date_back_[int(do_devices[j])]

                
            if i == 0:
                if scenario == 1:
                    for k in range(2):
                        if int(do_devices[j]) == 0:
                            proc_local[k][j] =  d1_proc_fwd_last
                            proc_local_back[k][j] =  d1_proc_back_first
                        elif int(do_devices[j]) == 1:
                            proc_local[k][j] =  jetson_gpu_proc_fwd_last
                            proc_local_back[k][j] =  jetson_gpu_proc_back_first
                        elif int(do_devices[j]) == 3:
                            proc_local[k][j] =  jetson_cpu_proc_fwd_last
                            proc_local_back[k][j] =  jetson_cpu_proc_back_first
                        

                if scenario == 2:
                    for k in range(2):
                        proc_local[k][j] =  proc_local_[int(do_devices[j])]
                        proc_local_back[k][j] =  proc_local_back_[int(do_devices[j])]


            if scenario == 1:
                if i < H:
                    for k in range(2):
                        if int(machine_devices[i]) == 0:
                            proc[k][j,i] =  vm_proc_fwd
                            proc_bck[k][j,i] =  vm_proc_back
                        elif int(machine_devices[i]) == 1:
                            proc[k][j,i] = laptop_proc_fwd
                            proc_bck[k][j,i] = laptop_proc_back
                else:
                    if j != i - H:
                        continue
                    if int(do_devices[j]) == 0:
                        proc[1][j,i] =  d1_proc_fwd_medium
                        proc_bck[1][j,i] =  d1_proc_back_medium
                    elif int(do_devices[j]) == 1:
                        proc[1][j,i] =  jetson_gpu_proc_fwd_medium
                        proc_bck[1][j,i] =  jetson_gpu_proc_back_medium
                    elif int(do_devices[j]) == 3:
                        proc[1][j,i] =  jetson_cpu_proc_fwd_medium
                        proc_bck[1][j,i] =  jetson_cpu_proc_back_medium

            if scenario == 2:
                if i < H:
                    for k in range(2):
                        proc[k][j,i] =  proc_fwd_[int(machine_devices[i])]
                        proc_bck[k][j,i] = proc_back_[int(machine_devices[i])]
                else:
                    proc[1][j,i] =  proc_fwd_[int(machine_devices[i])]
                    proc_bck[1][j,i] = proc_back_[int(machine_devices[i])]

    memory_demand_ = np.ones((K)) * store_compute_node
    for i in range(K):
        memory_demand_[i] = int(math.ceil(memory_demand_[i]))
    
    global max_memory_demand
    max_memory_demand = int(max(memory_demand_))

    memory_capacity = [np.zeros((H)), np.zeros((H_prime))]
    for i in range(H_prime):
        if i < H:
            memory_capacity[0][i] = int(max(memory_demand_))*K
        memory_capacity[1][i] = int(max(memory_demand_))*K
    
    '''
    unique_values = []
    for j in range(K):
        for i in range(H):
            unique_values.append(int(np.rint(release_date[j,i])))
            unique_values.append(int(np.rint(release_date_back[j,i])))

    for j in range(K):
        unique_values.append(int(np.rint(proc_local[j])))
        unique_values.append(int(np.rint(proc_local_back[j])))

    
    
    for j in range(K):
        for i in range(H):
            unique_values.append(int(np.rint(trans_back[j,i])))
            unique_values.append(int(np.rint(trans_back_gradients[j,i])))

    max_value = max(unique_values)

    # Re-difine parameters as splots
    max_slot_back = max_slot
 
    for j in range(K):
        for i in range(H):
            
            release_date[j,i] = np.ceil((release_date[j,i]*max_slot)/max_value).astype(int)
            trans_back[j,i] = np.ceil((trans_back[j,i]*max_slot)/max_value).astype(int)
            
            release_date_back[j,i] = np.ceil((release_date_back[j,i]*max_slot)/max_value).astype(int)
            trans_back_gradients[j,i] = np.ceil((trans_back_gradients[j,i]*max_slot)/max_value).astype(int)

            if i == 0:
                proc_local[j] = np.ceil((proc_local[j]*max_slot)/max_value).astype(int)
                proc_local_back[j] = np.ceil((proc_local_back[j]*max_slot)/max_value).astype(int)

            proc[j,i] =  np.ceil((proc[j,i]*max_slot)/max_value).astype(int)
            
            if proc[j,i] == 0:
                    proc[j,i] = 1

            proc_bck[j,i] =  np.ceil((proc_bck[j,i]*max_slot)/max_value).astype(int)

            if proc_bck[j,i] == 0:
                    proc_bck[j,i] = 1
    '''

    max_slot = 500
    max_slot_back = 500 #v 2000#r 1500
    max_slot_maxx = 500#v 5000# r-4000
    for j in range(K):
        for i in range(H_prime):
            if i < H:
                release_date[0][j,i] = np.ceil(release_date[0][j,i]/max_slot).astype(int)
                trans_back[0][j,i] = np.ceil(trans_back[0][j,i]/max_slot).astype(int)
                
                release_date_back[0][j,i] = np.ceil(release_date_back[0][j,i]/max_slot_maxx).astype(int)
                trans_back_gradients[0][j,i] = np.ceil(trans_back_gradients[0][j,i]/max_slot).astype(int)
            
                release_date[1][j,i] = np.ceil(release_date[1][j,i]/max_slot).astype(int)
                trans_back[1][j,i] = np.ceil(trans_back[1][j,i]/max_slot).astype(int)
                
                release_date_back[1][j,i] = np.ceil(release_date_back[1][j,i]/max_slot_maxx).astype(int)
                trans_back_gradients[1][j,i] = np.ceil(trans_back_gradients[1][j,i]/max_slot).astype(int)
            else:
                release_date[1][j,i] = np.ceil(release_date[1][j,i]/max_slot).astype(int)
                trans_back[1][j,i] = np.ceil(trans_back[1][j,i]/max_slot).astype(int)
                
                release_date_back[1][j,i] = np.ceil(release_date_back[1][j,i]/max_slot_maxx).astype(int)
                trans_back_gradients[1][j,i] = np.ceil(trans_back_gradients[1][j,i]/max_slot).astype(int)

            if i == 0:
                proc_local[0][j] = np.ceil(proc_local[0][j]/max_slot).astype(int)
                proc_local_back[0][j] = np.ceil(proc_local_back[0][j]/max_slot_back).astype(int)

                proc_local[1][j] = np.ceil(proc_local[1][j]/max_slot).astype(int)
                proc_local_back[1][j] = np.ceil(proc_local_back[1][j]/max_slot_back).astype(int)

            if i < H:
                proc[0][j,i] =  np.ceil(proc[0][j,i]/max_slot).astype(int)
            
                if proc[0][j,i] == 0:
                        proc[0][j,i] = 1

                proc_bck[0][j,i] =  np.ceil(proc_bck[0][j,i]/max_slot_back).astype(int)
                if proc_bck[0][j,i] == 0:
                        proc_bck[0][j,i] = 1

                proc[1][j,i] =  np.ceil(proc[1][j,i]/max_slot).astype(int)
            
                if proc[1][j,i] == 0:
                        proc[1][j,i] = 1

                proc_bck[1][j,i] =  np.ceil(proc_bck[1][j,i]/max_slot_back).astype(int)

                if proc_bck[1][j,i] == 0:
                        proc_bck[1][j,i] = 1
            else:
                
                proc[1][j,i] =  np.ceil(proc[1][j,i]/max_slot).astype(int)
                
                if proc[1][j,i] == 0:
                   proc[1][j,i] = 1


                
                proc_bck[1][j,i] =  np.ceil(proc_bck[1][j,i]/max_slot_back).astype(int)

                if proc_bck[1][j,i] == 0:
                    proc_bck[1][j,i] = 1
    
    
    print('release date')
    print(release_date[1])
    print('proc')
    print(proc[1])
    print('proc local')
    print(proc_local[1])
    print('trans back')
    print(trans_back[1])
    print('release data back')
    print(release_date_back[1])
    print('proc back')
    print(proc_bck[1])
    print('proc loack back')
    print(proc_local_back[1])
    print('trans back')
    print(trans_back_gradients[1])
    print('------')

    
    
    return (release_date, proc, 
    proc_local, trans_back, 
    memory_capacity, memory_demand_, 
    release_date_back, proc_bck, 
    proc_local_back, trans_back_gradients)

def create_scenario_hybrid_typeA(filename, point_a, point_b, K, H, 
                                 max_slot, slow_client, slow_networks):
    
    df_vm = pd.read_excel(io=filename, sheet_name='VM', header=None)
    df_d1 = pd.read_excel(io=filename, sheet_name='d1', header=None)
    df_memory = pd.read_excel(io=filename, sheet_name='memory', header=None)

    H_prime = H + K
    slow_factor = 2
    d1_data = df_d1.values.tolist()
    # processing time on vms
    vm_data = df_vm.values.tolist()
    vm_proc_fwd = 0
    vm_proc_back = 0

    d1_p2_middle_fwd = 0
    d1_p2_middle_back = 0
    for i in range(point_a, point_b):
        vm_proc_fwd += vm_data[i][0]
        vm_proc_back += vm_data[i][1] + vm_data[i][2]

        d1_p2_middle_fwd += d1_data[i][0]
        d1_p2_middle_back += d1_data[i][1] + d1_data[i][2]
    
    # processing time on d1
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

    #my_net = lambda data,bandwidth : ((data*0.0008)/bandwidth)*1000
    network_connections = [ lambda a : ((a*0.000008)/8)*1000, # 8 Mbits/sec
                            lambda a : (a*0.0000008)*1000, # 10 Mbits/sec
                            lambda a : ((a*0.000000008)/7.13)*1000, # 7.13 Gbits/sec
                            lambda a : ((a*0.000008)/2)*1000 # 2 Mbits/sec
                            ]
    
    # random seed 
    original_state = np.random.get_state()
    random.seed(42)

    # randomly select the network connections using the Atari stats
    network_type = np.zeros((K,H))

    '''
    We consider the following cases:

    class-0        <= 4 Mbps             --> SLOW
    class-3        > 15 and <= 20        --> FAST
    '''

    class0 = []
    class3 = []

    total_connections = K*H

    num_class0 = int((total_connections*slow_networks)/100)
    num_class3 = K*H - num_class0

    completed = []
    for i in range(num_class0):
        while True:
            net_line = int(random.randint(0,total_connections-1))

            if not (net_line in completed):
                break
        
        completed.append(net_line)
        network_type[int(net_line/H),int(net_line%H)] = 1 #random.randint(1,4)

    for i in range(num_class3):
        
        while True:
            net_line = int(random.randint(0,total_connections-1))

            if not (net_line in completed):
                break
        
        completed.append(net_line)
        network_type[int(net_line/H),int(net_line%H)] = 20 #random.randint(16,20)

    print('Network types:')
    print(network_type)

    do_devices = np.zeros((K))
    num_slow = int((K*slow_client)/100)
    num_fast = K - num_slow

    completed = []
    for i in range(num_slow):
        while True:
            client_id = int(random.randint(0,K-1))

            if not (client_id in completed):
                break
        
        completed.append(client_id)
        do_devices[client_id] = 0

    for i in range(num_fast):
        while True:
            client_id = int(random.randint(0,K-1))

            if not (client_id in completed):
                break
        
        completed.append(client_id)
        do_devices[client_id] = 1
    
    print('slow devices:')
    print(do_devices)

    release_date = [np.zeros((K,H)), np.zeros((K,H_prime))]
    proc = [np.zeros((K,H)), np.zeros((K,H_prime))] 
    proc_local = [np.zeros((K)), np.zeros((K))]
    trans_back = [np.zeros((K, H)), np.zeros((K, H_prime))] 
    release_date_back = [np.zeros((K,H)), np.zeros((K,H_prime))]
    proc_bck = [np.zeros((K,H)), np.zeros((K,H_prime))] 
    proc_local_back = [np.zeros((K)), np.zeros((K))]
    trans_back_gradients = [np.zeros((K, H)), np.zeros((K, H_prime))]

    # Construct scenario
    for j in range(K):
        for i in range(H_prime):
            if i < H:
                indx = int(network_type[j,i])
                
                release_date[0][j,i] = my_net(activations_to_cn, indx)
                trans_back[0][j,i] = my_net(activations_to_do, indx)

                release_date_back[0][j,i] = my_net(activations_to_do, indx)
                trans_back_gradients[0][j,i] = my_net(activations_to_cn, indx)

                release_date[1][j,i] = my_net(activations_to_cn, indx)
                trans_back[1][j,i] = my_net(activations_to_do, indx)

                release_date_back[1][j,i] = my_net(activations_to_do, indx)
                trans_back_gradients[1][j,i] = my_net(activations_to_cn, indx)
            
            if int(do_devices[j]) == 0: #slow
                if i < H:
                    release_date[0][j,i] +=  d1_proc_fwd_first*slow_factor
                    release_date_back[0][j,i] += d1_proc_back_last*slow_factor

                release_date[1][j,i] +=  d1_proc_fwd_first*slow_factor
                release_date_back[1][j,i] += d1_proc_back_last*slow_factor
            else:
                if i < H:
                    release_date[0][j,i] +=  d1_proc_fwd_first
                    release_date_back[0][j,i] += d1_proc_back_last

                release_date[1][j,i] +=  d1_proc_fwd_first
                release_date_back[1][j,i] += d1_proc_back_last

            if i == 0:
                if int(do_devices[j]) == 0:
                    proc_local[0][j] =  d1_proc_fwd_last*slow_factor
                    proc_local_back[0][j] =  d1_proc_back_first*slow_factor

                    proc_local[1][j] =  d1_proc_fwd_last*slow_factor
                    proc_local_back[1][j] =  d1_proc_back_first*slow_factor
                else:
                    proc_local[0][j] =  d1_proc_fwd_last
                    proc_local_back[0][j] =  d1_proc_back_first

                    proc_local[1][j] =  d1_proc_fwd_last
                    proc_local_back[1][j] =  d1_proc_back_first
            
            if i < H:                           
                proc[0][j,i] =  vm_proc_fwd
                proc_bck[0][j,i] =  vm_proc_back

                proc[1][j,i] =  vm_proc_fwd
                proc_bck[1][j,i] =  vm_proc_back
            else:
                if int(do_devices[j]) == 0:
                    proc[1][j,i] =  d1_p2_middle_fwd*slow_factor
                    proc_bck[1][j,i] =  d1_p2_middle_back*slow_factor
                else:
                    proc[1][j,i] =  d1_p2_middle_fwd
                    proc_bck[1][j,i] =  d1_p2_middle_back

    memory_demand_ = np.ones((K)) * store_compute_node
    for i in range(K):
        memory_demand_[i] = int(math.ceil(memory_demand_[i]))
    
    global max_memory_demand
    max_memory_demand = int(max(memory_demand_))
    memory_capacity = [np.zeros((H)), np.zeros((H_prime))]
    for i in range(H_prime):
        if i < H:
            memory_capacity[0][i] = int(max(memory_demand_))*K
        memory_capacity[1][i] = int(max(memory_demand_))*K
    
    max_slot = 500
    max_slot_back = 1500
    max_slot_maxx = 4000
    for j in range(K):
        for i in range(H_prime):
            if i < H:
                release_date[0][j,i] = np.ceil(release_date[0][j,i]/max_slot).astype(int)
                trans_back[0][j,i] = np.ceil(trans_back[0][j,i]/max_slot).astype(int)
                
                release_date_back[0][j,i] = np.ceil(release_date_back[0][j,i]/max_slot_maxx).astype(int)
                trans_back_gradients[0][j,i] = np.ceil(trans_back_gradients[0][j,i]/max_slot).astype(int)
            
                release_date[1][j,i] = np.ceil(release_date[1][j,i]/max_slot).astype(int)
                trans_back[1][j,i] = np.ceil(trans_back[1][j,i]/max_slot).astype(int)
                
                release_date_back[1][j,i] = np.ceil(release_date_back[1][j,i]/max_slot_maxx).astype(int)
                trans_back_gradients[1][j,i] = np.ceil(trans_back_gradients[1][j,i]/max_slot).astype(int)
            else:
                release_date[1][j,i] = np.ceil(release_date[1][j,i]/max_slot).astype(int)
                trans_back[1][j,i] = np.ceil(trans_back[1][j,i]/max_slot).astype(int)
                
                release_date_back[1][j,i] = np.ceil(release_date_back[1][j,i]/max_slot_maxx).astype(int)
                trans_back_gradients[1][j,i] = np.ceil(trans_back_gradients[1][j,i]/max_slot).astype(int)

            if i == 0:
                proc_local[0][j] = np.ceil(proc_local[0][j]/max_slot).astype(int)
                proc_local_back[0][j] = np.ceil(proc_local_back[0][j]/max_slot_back).astype(int)

                proc_local[1][j] = np.ceil(proc_local[1][j]/max_slot).astype(int)
                proc_local_back[1][j] = np.ceil(proc_local_back[1][j]/max_slot_back).astype(int)

            if i < H:
                proc[0][j,i] =  np.ceil(proc[0][j,i]/max_slot).astype(int)
            
                if proc[0][j,i] == 0:
                        proc[0][j,i] = 1

                proc_bck[0][j,i] =  np.ceil(proc_bck[0][j,i]/max_slot_back).astype(int)
                if proc_bck[0][j,i] == 0:
                        proc_bck[0][j,i] = 1

                proc[1][j,i] =  np.ceil(proc[1][j,i]/max_slot).astype(int)
            
                if proc[1][j,i] == 0:
                        proc[1][j,i] = 1

                proc_bck[1][j,i] =  np.ceil(proc_bck[1][j,i]/max_slot_back).astype(int)

                if proc_bck[1][j,i] == 0:
                        proc_bck[1][j,i] = 1
            else:
                
                proc[1][j,i] =  np.ceil(proc[1][j,i]/max_slot).astype(int)
                
                if proc[1][j,i] == 0:
                   proc[1][j,i] = 1*(do_devices[j]/max_slot).astype(int)


                
                proc_bck[1][j,i] =  np.ceil(proc_bck[1][j,i]/max_slot_back).astype(int)

                if proc_bck[1][j,i] == 0:
                    proc_bck[1][j,i] = 1
    
    
    print('release date')
    print(release_date[1])
    print('proc')
    print(proc[1])
    print('proc local')
    print(proc_local[1])
    print('trans back')
    print(trans_back[1])
    print('release data back')
    print(release_date_back[1])
    print('proc back')
    print(proc_bck[1])
    print('proc loack back')
    print(proc_local_back[1])
    print('trans back')
    print(trans_back_gradients[1])
    print('------')
    return (release_date, proc, 
    proc_local, trans_back, 
    memory_capacity, memory_demand_, 
    release_date_back, proc_bck, 
    proc_local_back, trans_back_gradients)

'''
return np an numpy array of shape (K, H) with the release dates
for the forward pass (first batch).
Use profiling values
'''
def mysum(mylist):
    sum = 0

    for s in mylist:
        sum += s

    return sum

def get_fwd_release_delays(K,H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_release_delays', header=None)

    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list

'''
return np an numpy array of shape (H, 1) with the processing time
for the forward pass on the data owner for the last layers.
Use profiling values
'''
def get_fwd_end_local(K):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_end_local', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in  range(K):#range(len(temp)):
        df_list.append(temp[i][0])

    return df_list

def get_bwd_end_local(K):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_end_local', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in range(K):#range(len(temp)):
        df_list.append(temp[i][0])

    return df_list

def get_bwd_release_delays(K,H):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_release_delays', header=None)

    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list


'''
return np an numpy array of shape (K, H) with the processing time
for the forward pass on the compute nodes.
Use profiling values
'''
def get_fwd_proc_helper(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_fwd_proc_helper', header=None)
    temp = df.values.tolist()

    machines = []
    for i in  range(H):#range(len(temp)):
        machines.append(temp[i][0])

    total = []
    for i in range(K):
        total += [machines]
    return total

def get_bwd_proc_helper(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_bwd_proc_helper', header=None)
    temp = df.values.tolist()

    machines = []
    for i in  range(H):#range(len(temp)):
        machines.append(temp[i][0])

    total = []
    for i in range(K):
        total += [machines]
    return total


'''
return np an numpy array of shape (H, H) with the transmission time
from the compute nodes to the data owner sending the activations from the second cut layer.
Use profiling values
'''
def get_trans_back(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_trans_back', header=None)
    
    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list

def get_grad_trans_back(K, H):
    df = pd.read_excel(io=file_name, sheet_name='get_grad_trans_back', header=None)

    all_data = df.values.tolist()
    return_list = []

    for i in range(K):
        row_ = []
        for j in range(H):
            row_.append(all_data[i][j])
        return_list.append(row_)

    return return_list

'''
return np an numpy array of shape(H,1) with the memory capacity of each 
helper node
'''
def get_memory_characteristics(H, K=10):
    # random seed 
    random.seed(42)
    '''
    df = pd.read_excel(io=file_name, sheet_name='get_memory_characteristics', header=None)
    temp = df.values.tolist()

    df_list = []
    for i in  range(len(temp)):
        df_list.append(temp[i][0])

    return df_list
    #return [10,10]
    '''
    df_list = []
    
    total_demand = max_memory_demand * K
    if False: # all the same
        minimum_avail = int(K/H)
        if K % H == 0:
            df_list = [minimum_avail*max_memory_demand for i in range(H)]
        else:
            df_list = [minimum_avail*max_memory_demand for i in range(H-1)]
            df_list.append((minimum_avail+(K-H*minimum_avail))*max_memory_demand)
    else: # choose from a distribution but in the end all data owners should be served
        end = int(K/H)
        first_round = False
        while True:
            for i in range(H):
                data_owners = random.randint(1,end)
                if first_round:
                    df_list[i] = df_list[i] + data_owners
                    if mysum(df_list) >= K:
                        for i in range(H):
                            df_list[i] = df_list[i] * max_memory_demand
                        return df_list
                else:
                    df_list.append(data_owners)
            
            first_round = True
            if mysum(df_list) < K:
                end = K - mysum(df_list)
            else:  
                break
        for i in range(H):
            df_list[i] = df_list[i] * max_memory_demand
    return df_list


def plot_approach(w_1, w_2):
    fig, ax1 = plt.subplots(1)
    x_ticks = [i+1 for i in range(len(w_2))]
    #print(w_approach)
    
    ax1.plot(x_ticks, [w_1 for i in range(len(x_ticks))], linewidth = 2, marker='o', markersize=2, color="green", label = "Optimal value")
    
    if len(w_2):
        ax1.plot(x_ticks, w_2, linestyle='dashed', linewidth = 2, marker='o', markersize=5, color="orange", label = "W-approx.")
    
    ax1.set_ylabel("w value")
    plt.xlim(0.9,len(w_2))

    ax1.legend()
    plt.show()

def fifo(K, H, release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations, 
         release_date_back, proc_bck, proc_local_back, trans_back_gradients, y):
    
    
    f_temp = np.zeros(K)
    f_temp_faster_machine = []

    # Estimated Completition time
    for machine in range(H):
        my_jobs = list(np.transpose(np.argwhere(y[:,machine]==1))[0])
        machine_time = 0
        arival_jobs = []
        for j in my_jobs:
            new_job = arrival_date(release_date_fwd[j,machine], False, j)
            arival_jobs.append(new_job)
        first = True
        while len(arival_jobs) > 0:
            #print('rr')
            faster = -1
            for i in range(len(arival_jobs)):
                #print(arival_jobs[i].value)
                if (i == 0) or (arival_jobs[faster].value > arival_jobs[i].value):
                    faster = i

            next_task = arival_jobs[faster]
            arival_jobs.pop(faster)

            if next_task.back:
                if machine_time <= next_task.value:
                    machine_time = next_task.value

                f_temp[next_task.job] = machine_time + proc_bck[next_task.job, machine] +\
                                        trans_back_gradients[next_task.job, machine] +\
                                        proc_local_back[next_task.job]
                
                machine_time +=  proc_bck[next_task.job, machine]
            else:
                if machine_time <= next_task.value:
                    machine_time = next_task.value

                next_task.value = machine_time + proc_fwd[next_task.job, machine] +\
                                   trans_back_activations[next_task.job, machine] + \
                                   proc_local_fwd[next_task.job] +\
                                   release_date_back[next_task.job, machine]
                
                machine_time +=  proc_fwd[next_task.job, machine]
                
                
                next_task.back = True
                arival_jobs.append(next_task)
        
        return max(f_temp)