import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import warnings

import utils 
warnings.filterwarnings("ignore")

def run(K, H, T, release_date_fwd, proc_fwd, 
            proc_local_fwd, trans_back_activations, 
            memory_capacity, 
            release_date_back, proc_bck, 
            proc_local_back, trans_back_gradients):
    
    H_prime = H + K
    ones_H = np.ones((H_prime,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("hybrid_solution")
    
    # define variables
    x = m.addMVar(shape = (H_prime,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H_prime), vtype=GRB.BINARY, name="y")
    z = m.addMVar(shape = (H_prime,K,T), vtype=GRB.BINARY, name="z")
    
    # auxilary variables
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")

    
    for j in range(K):
        for i in range(H_prime):
            if i < H:
                continue
            if i != H + j:
                m.addConstr(y[j,i] == 0)

    # define constraints FORWARD

    # C1: A job cannot be assigned to a time interval before the release time
    for i in range(H_prime): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date_fwd[j,i]:
                    m.addConstr(x[i,j,t] == 0)

    # C3: all job intervals are assigned to one machine
    m.addConstr( y @ ones_H == ones_K )
    
    # C4: memory constraint 
    m.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    # C6: machine processes only a single job at each interval
    for j in range(H_prime): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    
    # C9: new constraint - the merge of C2 and C3 (job should be processed all once and only in one machine)
    for j in range(H_prime): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc_fwd[i,j])

     # define constraints BACK
    
    # The backprop job cannot be assigned to a time interval before the backdrops release time

    for i in range(K): #for all jobs
        for j in range(H_prime):
            temp = proc_local_fwd[i] + trans_back_activations[i,j] + release_date_back[i,j]
            temp = int(temp)
            for t in range(temp):
                m.addConstr(z[j,i,t] == 0)

            for t in range(T):
                if t + temp >= T:
                    break
                
                m.addConstr(z[j,i,t+temp] <= qsum(x[j,i,l] for l in range(t))/proc_fwd[i,j])
    
   
    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        for j in range(H_prime):
            m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc_bck[i, j] == y[i,j])
    
    # C11: machine processes only a single job at each interval
    for j in range(H_prime): #for all devices
        m.addConstr( x[j,:,:].T + z[j,:,:].T @ ones_K <= ones_T )
 
        
    for j in range(H_prime): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*z[j,i,t] for t in range(T))


    # Define the objective function
    
    m.addConstrs(comp[i] == qsum(trans_back_gradients[i,:] * y[i,:]) + f[i] + proc_local_back[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    m.optimize()

    print(f'Allocation policy: {np.rint(y.X)}')
    print('----- Scheduling policy -----')    

    for i in range(H_prime):
        if i < H:
            print(f'In machine-{i}', end='\t')
        else:
            print(f'In client-{i - H+1}', end='\t')
        for k in range(T):
            at_least = 0
            for j in range(K):
                #print(np.rint(x[i,j,k].X))
                if(np.rint(x[i,j,k].X) <= 0 and np.rint(z[i,j,k].X) <= 0):
                    #print('shit1')
                    continue
                else:
                    if np.rint(x[i,j,k].X) > 0:
                        print(f'{(j+1)}', end='\t')
                        #print('shit2')
                        at_least = 1
                        break
                    else:
                        print(f'{(j+1)}\'', end='\t')
                        #print('shit2')
                        at_least = 1
                        break
            if(at_least == 0):
                print(f'0', end='\t')
        print('')
    
    g_var = []
    print("--------Completition time--------")
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H_prime):
            if np.rint(y[i,j].X) == 1:
                my_machine = j
                break
        last_zero = -1
        for k in range(T):
            if np.rint(z[my_machine,i,k].X) >= 1:
                last_zero = k+1
        fmax = last_zero
        g_var.append(fmax)
        C = fmax + proc_local_back[i] + trans_back_gradients[i,my_machine]
        if my_machine >= H:
            print(f'C{i+1}: {C} - C-{my_machine - H+1}')
        else:
            print(f'C{i+1}: {C} - M-{my_machine}')
    print(f'objective function: {m.ObjVal}')

    return(m.ObjVal)


def run_energy(K, H, T, release_date_fwd, proc_fwd, 
                proc_local_fwd, trans_back_activations, 
                memory_capacity, 
                release_date_back, proc_bck, 
                proc_local_back, trans_back_gradients, 
                P_comp, P_transf, P_receive,
                max_slot, network_bwd, ksi, alpha):

    H_prime = H + K
    ones_H = np.ones((H_prime,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("hybrid_solution")
    
    # define variables
    x = m.addMVar(shape = (H_prime,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H_prime), vtype=GRB.BINARY, name="y")
    z = m.addMVar(shape = (H_prime,K,T), vtype=GRB.BINARY, name="z")
    
    # auxilary variables
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")
    eng_comp = m.addMVar(shape=(K), vtype=GRB.CONTINUOUS, name="eng_comp")
    eng_transf = m.addMVar(shape=(K), vtype=GRB.CONTINUOUS, name="eng_transf")
    eng_total = m.addMVar(shape=(1), vtype=GRB.CONTINUOUS, name="eng_total")
    
    for j in range(K):
        for i in range(H_prime):
            if i < H:
                continue
            if i != H + j:
                m.addConstr(y[j,i] == 0)

    # define constraints FORWARD

    # C1: A job cannot be assigned to a time interval before the release time
    for i in range(H_prime): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date_fwd[j,i]:
                    m.addConstr(x[i,j,t] == 0)

    # C3: all job intervals are assigned to one machine
    m.addConstr( y @ ones_H == ones_K )
    
    # C4: memory constraint 
    m.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    # C6: machine processes only a single job at each interval
    for j in range(H_prime): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    
    # C9: new constraint - the merge of C2 and C3 (job should be processed all once and only in one machine)
    for j in range(H_prime): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc_fwd[i,j])

     # define constraints BACK
    
    # The backprop job cannot be assigned to a time interval before the backdrops release time

    for i in range(K): #for all jobs
        for j in range(H_prime):
            temp = proc_local_fwd[i] + trans_back_activations[i,j] + release_date_back[i,j]
            temp = int(temp)
            for t in range(temp):
                m.addConstr(z[j,i,t] == 0)

            for t in range(T):
                if t + temp >= T:
                    break
                
                m.addConstr(z[j,i,t+temp] <= qsum(x[j,i,l] for l in range(t))/proc_fwd[i,j])
    
   
    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        for j in range(H_prime):
            m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc_bck[i, j] == y[i,j])
    
    # C11: machine processes only a single job at each interval
    for j in range(H_prime): #for all devices
        m.addConstr( x[j,:,:].T + z[j,:,:].T @ ones_K <= ones_T )
 
        
    for j in range(H_prime): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*z[j,i,t] for t in range(T))


    # Define the objective function
    
    m.addConstrs(comp[i] == qsum(trans_back_gradients[i,:] * y[i,:]) + f[i] + proc_local_back[i] for i in range(K))
       
    
    m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))

    for i in range(K):
        m.addConstr(eng_comp[i] == (max_slot/1000)*P_comp[i]*(release_date_fwd[i,H+i]+release_date_back[i,H+i] \
                                            + proc_local_fwd[i] + proc_local_back[i] \
                                            + qsum(y[i,j]*(proc_fwd[i,j] + proc_bck[i,j]) for j in range(H,H_prime))))
        
        m.addConstr(eng_transf[i] == qsum((P_transf[i,j]+P_receive[i,j])*float(1/network_bwd[i,j])*y[i,j] for j in range(H))*(ksi[0] + ksi[1])*0.0008)
    
    m.addConstr(eng_total == qsum(eng_comp+eng_transf)) # make it Joule from mJ
    
    # approach a: with propotional values
    max_eng = 0
    for i in range(K):
        max_eng += float(max_slot/1000)*P_comp[i] * (release_date_fwd[i,H+i]+release_date_back[i,H+i] \
                            + proc_local_fwd[i] + proc_local_back[i]) \
                            + max(float(max_slot/1000)*P_comp[i] * (proc_fwd[i,H+i] + proc_bck[i,H+i]),
                            + ((ksi[0] + ksi[1])*0.0008)*max([(P_transf[i,j]+P_receive[i,j])*float(1/network_bwd[i,j]) for j in range(H)]))
        
    #max_eng = max_eng
    

    # approach b: with log 
    # log_object = m.addMVar(shape=(1), vtype=GRB.CONTINUOUS, name="log_object")
    # m.addGenConstrLogA(eng_total, log_object, 10.0, "log10", "FuncPieces=-1 FuncPieceError=1e-5")
    # m.setObjective(alpha*maxobj + (1-alpha)*log_object, GRB.MINIMIZE)
    
    makespan_obj = m.addMVar(shape=(1), vtype=GRB.CONTINUOUS, name="makespan_obj")
    energy_obj = m.addMVar(shape=(1), vtype=GRB.CONTINUOUS, name="energy_obj")

    m.addConstr(makespan_obj == maxobj/T)
    m.addConstr(energy_obj == eng_total/max_eng)
    m.setObjective(alpha*makespan_obj \
                   + (1-alpha)*energy_obj, GRB.MINIMIZE)
    
    m.optimize()
    '''
    print(f'Allocation policy: {np.rint(y.X)}')
    print('----- Scheduling policy -----')    

    for i in range(H_prime):
        if i < H:
            print(f'In machine-{i}', end='\t')
        else:
            print(f'In client-{i - H+1}', end='\t')
        for k in range(T):
            at_least = 0
            for j in range(K):
                #print(np.rint(x[i,j,k].X))
                if(np.rint(x[i,j,k].X) <= 0 and np.rint(z[i,j,k].X) <= 0):
                    #print('shit1')
                    continue
                else:
                    if np.rint(x[i,j,k].X) > 0:
                        print(f'{(j+1)}', end='\t')
                        #print('shit2')
                        at_least = 1
                        break
                    else:
                        print(f'{(j+1)}\'', end='\t')
                        #print('shit2')
                        at_least = 1
                        break
            if(at_least == 0):
                print(f'0', end='\t')
        print('')
    
    g_var = []
    print("--------Completition time--------")
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H_prime):
            if np.rint(y[i,j].X) == 1:
                my_machine = j
                break
        last_zero = -1
        for k in range(T):
            if np.rint(z[my_machine,i,k].X) >= 1:
                last_zero = k+1
        fmax = last_zero
        g_var.append(fmax)
        C = fmax + proc_local_back[i] + trans_back_gradients[i,my_machine]
        if my_machine >= H:
            print(f'C{i+1}: {C} - C-{my_machine - H+1}')
        else:
            print(f'C{i+1}: {C} - M-{my_machine}')
    print(f'objective function: {m.ObjVal}')
    '''
    print('print output:')
    print(f' Computing: {eng_comp.X}')
    print(f' Transport: {eng_transf.X}')
    print(f'TOTAL is:   {eng_total.X}')
    print(f'The maximum value of the energy consumption is: {max_eng}')
    
    print('------------ normilized: -------------')
    print(makespan_obj.X)
    print(energy_obj.X)

    return(maxobj.X[0])


def run_second(K, H, T, release_date_fwd, proc_fwd, 
            proc_local_fwd, trans_back_activations, 
            memory_capacity, 
            release_date_back, proc_bck, 
            proc_local_back, trans_back_gradients):
    
    H_prime = H + K
    ones_H_prime = np.ones((H_prime,1))
    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("hybrid_solution")
    
    # define variables
    x = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H_prime), vtype=GRB.BINARY, name="y")
    z = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="z")
    
    # auxilary variables
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    f_offload = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f_offload")
    f_device = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f_device")
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")

    
    for j in range(K):
        for i in range(H_prime):
            if i < H:
                continue
            if i != H + j:
                m.addConstr(y[j,i] == 0)

    # define constraints FORWARD

    # C1: A job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date_fwd[j,i]:
                    m.addConstr(x[i,j,t] == 0)

    # C3: all job intervals are assigned to one machine
    m.addConstr( y @ ones_H_prime == ones_K )
    
    # C4: memory constraint 
    m.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H_prime.shape))
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    
    # C9: new constraint - the merge of C2 and C3 (job should be processed all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc_fwd[i,j])

     # define constraints BACK
    
    # The backprop job cannot be assigned to a time interval before the backdrops release time

    for i in range(K): #for all jobs
        for j in range(H):
            temp = proc_local_fwd[i] + trans_back_activations[i,j] + release_date_back[i,j]
            temp = int(temp)
            for t in range(temp):
                m.addConstr(z[j,i,t] == 0)

            for t in range(T):
                if t + temp >= T:
                    break
                
                m.addConstr(z[j,i,t+temp] <= qsum(x[j,i,l] for l in range(t))/proc_fwd[i,j])
    
   
    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        for j in range(H):
            m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc_bck[i, j] == y[i,j])
    
    # C11: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T + z[j,:,:].T @ ones_K <= ones_T )
 
        
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f_offload[i] >= (t+1)*z[j,i,t] for t in range(T))

    #for all devices
    for i in range(K):
        m.addConstr( f_device[i] >= y[i, H+i]*(release_date_fwd[i,H+i] + release_date_back[i,H+i] + 
                                                proc_fwd[i,H+i] + proc_bck[i,H+i] + 
                                                proc_local_fwd[i]))

        m.addConstr( f[i] >= f_offload[i] +f_device[i])


    # Define the objective function
    
    m.addConstrs(comp[i] == qsum(trans_back_gradients[i,:] * y[i,:]) + f[i] + proc_local_back[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    m.optimize()

    print(f'Allocation policy: {np.rint(y.X)}')
    print('----- Scheduling policy -----')    
    '''
    for i in range(H_prime):
        if i < H:
            print(f'In machine-{i}', end='\t')
        else:
            print(f'In client-{i - H+1}', end='\t')
        for k in range(T):
            at_least = 0
            for j in range(K):
                #print(np.rint(x[i,j,k].X))
                if(np.rint(x[i,j,k].X) <= 0 and np.rint(z[i,j,k].X) <= 0):
                    #print('shit1')
                    continue
                else:
                    if np.rint(x[i,j,k].X) > 0:
                        print(f'{(j+1)}', end='\t')
                        #print('shit2')
                        at_least = 1
                        break
                    else:
                        print(f'{(j+1)}\'', end='\t')
                        #print('shit2')
                        at_least = 1
                        break
            if(at_least == 0):
                print(f'0', end='\t')
        print('')
    '''
    g_var = []
    print("--------Completition time--------")
    for i in range(K): #for all jobs
        my_machine = 0
        for j in range(H_prime):
            if np.rint(y[i,j].X) == 1:
                my_machine = j
                break
        last_zero = -1
        if my_machine < H:
            for k in range(T):
                if np.rint(z[my_machine,i,k].X) >= 1:
                    last_zero = k+1
            fmax = last_zero
        else:
            fmax = release_date_fwd[i,H+i] + release_date_back[i,H+i] + proc_fwd[i,H+i] + proc_bck[i,H+i] + proc_local_fwd[i]
        g_var.append(fmax)
        C = fmax + proc_local_back[i] + trans_back_gradients[i,my_machine]
        if my_machine >= H:
            print(f'C{i+1}: {C} - C-{my_machine - H+1}')
        else:
            print(f'C{i+1}: {C} - M-{my_machine}')
    print(f'objective function: {m.ObjVal}')

    return(m.ObjVal)