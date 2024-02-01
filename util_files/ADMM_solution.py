import numpy as np
from numpy import linalg as LA
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import warnings
import time

import utils 
warnings.filterwarnings("ignore")

def feasibility_check(K, release_date, proc, proc_local, trans_back, memory_capacity, T):
    H = 1

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("fwd_only")

    # define variables
    #print(f" Memory: {memory_capacity}")
    #print(f"T: {T}")
    x = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")


    # define constraints
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j]:
                    m.addConstr(x[i,j,t] == 0)

    # C3: all jobs interval are assigned to one only machine
    m.addConstr( y @ ones_H == ones_K )
        
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    # C9: new constraint - the merge of C2 and C3 (job should be processed all in once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc[i])

    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*x[j,i,t] for t in range(T))

    # Define the objective function

    m.addConstrs(comp[i] == qsum(trans_back[i] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    
    m.update()
    # Optimize model
    m.optimize()
    return(x.X)

def backward_for_each_machine(K, release_date, proc, proc_local, trans_back, memory_capacity, T, x):
    H = 1

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("fwd_only")

    # define variables
    #print(f" Memory: {memory_capacity}")
    #print(f"T: {T}")
    
    z = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")


    # define constraints
    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j]:
                    m.addConstr(z[i,j,t] == 0)    
    
    m.addConstr( y @ ones_H == ones_K )

    # C10: backprop job should be processed entirely once and in the same machine as the fwd
    for i in range(K): #for all jobs
        for j in range(H):
            m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc[i] == y[i,j])
    
    # C11: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T + z[j,:,:].T @ ones_K <= ones_T )
 
        
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*z[j,i,t] for t in range(T) )

    # Define the objective function
    m.addConstrs(comp[i] == qsum(trans_back[i] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    
    m.update()
    m.optimize()
   
    return(z.X)

def algo2(K, release_date, proc, proc_local, trans_back, memory_capacity, T, x):
    H = 1 
    z = np.zeros(((K,T)))

    # step - 1 order clients by releasing date
    order = np.argsort(release_date)
    for i in range(K):
        client = order[i]
        start = release_date[client]
        startt = start
        for t in range(start, T):
            clear = True
            for j in range(K):
              if z[j][t] != 0 or (t <= x.shape[2] and x[0][j][t] != 0):
                  clear = False
                  break 
            if clear:
                break
            else:
                startt += 1 

        alloced = 0
        while alloced < proc[client]:
            clear = True
            for j in range(K):
              if (t <= x.shape[2] and x[0][j][startt] != 0):
                  clear = False
                  startt += 1
                  break 
            if clear:
                z[client][startt] = 1
                alloced += 1
                startt = startt + 1

    # step - 2 define blocks
    
    '''
    for each block-i, we store a list of two values bita_i = [[slots_i], [clients_i]]
    where,
    slots_i = [s(bita_i), e(bita_i)]
    clients_i : list of clients inside that slot
    '''
    
    B = {}
    start = 0
    new_set = False
    iter = 0
    next = ([0,0], [])
    for t in range(T):
        if new_set == False:
            sum_z = 0
            for c in range(K):
                sum_z += z[c,t]
            if sum_z > 0:
                new_set = True
                iter += 1
                next[0][0] = t
                for j in range(K):
                    if z[j,t] != 0:
                        next[1].append(j+1)
                        break
        else:
            is_end = True
            sum_x = 0
            sum_z = 0
            for c in range(K):
                sum_x += x[0][c][t]
                sum_z += z[c,t]

            if (sum_z > 0) or (sum_z == 0 and (t <= x.shape[2] and sum_x > 0)):
                is_end = False
                for j in range(K):
                    if z[j,t] != 0:
                        if (j+1) not in next[1]:
                            next[1].append(j+1)
                        break

            if is_end:    
                next[0][1] = t
                B.update({iter:next})
                next = ([0,0], [])
                new_set = False

    
    for bita in list(B.keys()):
        if len(B[bita][1]) == 1: # no need to reschedule
            continue
        
        sub_blocks = [B[bita]]
        block_i = 0
        while len(sub_blocks) > 0: 
            bloc = sub_blocks.pop(block_i)
            if len(bloc[1]) == 1:
                if len(sub_blocks) != 0:
                        block_i = (block_i + 1)%len(sub_blocks)
                continue
            
            # step - 3 find l-client
            l = -1
            min = T+1000000
            for client in bloc[1]:
                temp = bloc[0][1]+proc_local[int(client)-1]+trans_back[int(client)-1]
                if temp < min:
                    min = temp
                    l = client

            # step 4 - reschedule
            # l tasks should allocated in slots where no other client has been released

            # find start of l
            start_l = -1
            for i in range(bloc[0][0], bloc[0][1]):
                if z[l-1,i] != 0:
                    start_l = i
                    break
            
            allocated_slots = 0
            
            slot_i = start_l
            at_least_one = False
            while slot_i <= bloc[0][1]:
                no_change = True
                for jj in range(K):
                    give_priority = -1
                    client_c = order[jj]
                    if client_c + 1 == l:
                        continue
                    
                    if release_date[client_c] <= slot_i and (client_c+1 in bloc[1]):
                        is_allocated = False
                        ii = bloc[0][0]
                        while ii <= slot_i:
                            if z[client_c-1,ii] != 0:
                                is_allocated = True
                            ii += 1
                        
                        if is_allocated:
                            continue
                        give_priority = client_c+1
                    
                    
                    if give_priority != -1:
                        alloced = 0
                        while alloced < proc[give_priority-1]:
                            clear = True
                            for j in range(K):
                                if x[0][j][slot_i] != 0:
                                    clear = False
                                    slot_i += 1
                                    break 
                            if clear:
                                z[give_priority-1][slot_i] = 1
                                alloced += 1
                                slot_i += 1
                        no_change = False
                        at_least_one = True
                        break
                    
                if no_change:
                    if allocated_slots == 0:
                        start_l = slot_i
                    while True:
                        clear = True
                        for j in range(K):
                            if x[0][j][slot_i] != 0:
                                clear = False
                                slot_i += 1
                                break 
                        if clear:
                            z[l-1][slot_i] = 1
                            allocated_slots += 1
                            break
                    if allocated_slots == proc[int(l)-1]:
                        break
                        

            # if cannot reschedule:
            if not at_least_one:
                if len(sub_blocks) != 0:
                    block_i = (block_i + 1)%len(sub_blocks)
                continue

            # step 5 - update subset
            slot_i = bloc[0][0]
            
            new_block_f = True
            while slot_i < bloc[0][1]:
                if new_block_f:
                    new_block = [[slot_i,-1],[]]
                    new_block_f = False
                
                if z[int(l)-1,slot_i] != 0:
                    if len(new_block[1]) > 0:
                        new_block[0][1] = slot_i
                        sub_blocks.append(new_block)
                    new_block_f = True
                    slot_i += 1
                    continue
                else:
                    for c in range(K):
                        if z[c,slot_i] != 0 and (c+1) not in new_block[1]:
                            new_block[1].append(c+1)
                
                new_block[0][1] = slot_i
                if slot_i == bloc[0][1]-1:
                    sub_blocks.append(new_block)
                slot_i += 1
            if len(sub_blocks) != 0:
                block_i = (block_i + 1)%len(sub_blocks)
    return(z)


def run(K, H, T_all, release_date_fwd, proc_fwd, 
            proc_local_fwd, trans_back_activations, 
            memory_capacity, memory_demand,
            release_date_back, proc_bck, 
            proc_local_back, trans_back_gradients, filename=''):
    
    stable = 0
    T = np.max(release_date_fwd) + K*np.max(proc_fwd) # time intervals
    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    MAX_ITER = 10
    rho = 350

    m1 = gp.Model("xsubproblem") # forward job assigment problem
    m2 = gp.Model("ysubproblem") # allocation problem

    # define variables -x subproblem
    x = m1.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    f = m1.addMVar(shape=(K), name="f")
    comp = m1.addMVar(shape=(K), name="comp") # completion time
    w = m1.addMVar(shape=(1), name="w")  # min of compl. times

    # define variables for y subproblem
    y = m2.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    comp_x_fixed = m2.addMVar(shape=(K), name="comp_x_fixed") # completion time
    w_x_fixed = m2.addMVar(shape=(1), name="w_x_fixed")  # min of compl. times
    
    # Auxiliary
    contr1_add_1 = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY,name="contr1_add_1")
    contr1_abs_1 = m1.addMVar(shape=(K,H), lb=-GRB.INFINITY,name="contr1_abs_1")  # auxiliary for abs value

    contr2_add_1 = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY, name="contr2_add_1")
    contr2_abs_1 = m2.addMVar(shape=(K,H), lb=-GRB.INFINITY,name="contr2_abs_1")   # auxiliary for abs value
    
    # Parameters
    x_par = np.zeros((H,K,T))
    y_par = np.zeros((K,H))

    T_back = np.max(release_date_fwd) + K*np.max(proc_fwd[0][:]) + np.max(release_date_back) + K*np.max(proc_bck[0,:]) \
                        + np.max(proc_local_fwd) + np.max(proc_local_back) \
                        + np.max(np.max(trans_back_activations)) + np.max(np.max(trans_back_gradients))

    z_par = np.zeros((H,K,T_back))

    # Dual variables
    mu = np.zeros((K,H)) # dual variable


    #Define constraints
    start = time.time()
    # C3: each job is assigned to one and only machine
    m2.addConstr(y @ ones_H == ones_K)
    end = time.time()
    first_build = end-start

    # C4: memory constraint
    m2.addConstr(memory_demand @ y<= memory_capacity)

    start = time.time()
    # completition time definition
    m1.addConstrs(f[i] >= (t+1)*x[j, i, t] for i in range(K) for j in range(H) for t in range(T))

    # C1: A job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date_fwd[j,i]:
                    m1.addConstr(x[i,j,t] == 0)

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m1.addConstr( x[j,:,:].T @ ones_K <= ones_T )

    # C5: A job should be processed entirely once
    for i in range(K):
        m1.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc_fwd[i,j] for j in range(H)) == 1)

    end = time.time()
    
    second_build = end-start
    time_stamps = []
    time_stamps_nobuild = []

    # Theoritically these builds take place in parallel
    if second_build > first_build:
        time_stamps.append(second_build)
        time_stamps_nobuild.append(second_build)
    else:
        time_stamps.append(first_build)
        time_stamps_nobuild.append(first_build)

    # Iterative algorithm - Algorithm 1
    my_ds2 = []
    my_ds = []
    obj_per_iter =[]
    violations = []

    flag_exit = False
    for iter in range(MAX_ITER):
        if flag_exit: # in the previous iteration we called the back-propagation problem
            break # exit;
        start = time.time()
        
        # Redefine error constraints P1 
        if iter >= 1:
            for d in my_ds:
                m1.remove(d)
            my_ds = []
        
        for i in range(K):
            for j in range(H):
                if iter >= 1:
                    c = m1.getConstrByName(f'const1add-{i}-{j}')
                    m1.remove(c)

                m1.addConstr(contr1_add_1[i,j] == qsum(x[j,i,t] for t in range(T)) - y_par[i,j]*proc_fwd[i,j] + mu[i,j], name=f'const1add-{i}-{j}')
                my_ds.append(m1.addConstr(contr1_abs_1[i,j] == gp.abs_(contr1_add_1[i,j]), name=f'const1ab-{i}-{j}'))

        for i in range(K):
            if iter >= 1:
                f11 = m1.getConstrByName(f'const1f-{i}')
                m1.remove(f11)
                

            m1.addConstr(comp[i] == np.sum(np.multiply(trans_back_activations[i,:], y_par[i,:])) + f[i] + proc_local_fwd[i], name=f'const1f-{i}')
            my_ds.append(m1.addConstr(w >= comp[i], name=f'const1fw-{i}'))
           
        m1.setObjective(w + (rho/2)*qsum(contr1_abs_1[i,j] for i in range(K) for j in range(H)), GRB.MINIMIZE)
        m1.update()

        # solve P1:
        start_opt = time.time()
        m1.optimize()
        end = time.time()
        time_stamps.append(end-start)
        time_stamps_nobuild.append(end-start_opt)

        x_par = np.copy(np.array(x.X))
        np.copy(np.array(w.X))

        # Solve P2
        start = time.time()


        # Now calculate the value of the (original) objective function at this iteration
        g_values = []
        for i in range(K): #for all jobs
            g_interm = []
            for j in range(H): #for all machines
                for t in range(T): #for all timeslots
                    g_interm += [(t+1)*x[j,i,t].X]
            g_values += [max(g_interm)]

        f_par = np.copy(g_values)

        # Redefine error constraints P2
        if iter >= 1:
            for dd in my_ds2:
                m2.remove(dd)
            my_ds2 = []

        ll = np.sum(x_par, axis=2)

        for i in range(K):
            for j in range(H):
                if iter >= 1:
                    c2 = m2.getConstrByName(f'const2add-{i}-{j}')
                    m2.remove(c2)

                m2.addConstr(contr2_add_1[i,j] == ll[j,i] - y[i,j]*proc_fwd[i,j] + mu[i,j], name=f'const2add-{i}-{j}')
                my_ds2.append(m2.addConstr(contr2_abs_1[i,j] == gp.abs_(contr2_add_1[i,j]), name=f'const2ab-{i}-{j}'))

        for i in range(K):
            if iter >= 1:
                f2 = m2.getConstrByName(f'const2f-{i}')
                m2.remove(f2)

            m2.addConstr(comp_x_fixed[i] == qsum(trans_back_activations[i,:]*y[i,:]) + f_par[i] + proc_local_fwd[i], name=f'const2f-{i}')
            my_ds2.append(m2.addConstr(w_x_fixed >= comp_x_fixed[i], name=f'const2fw-{i}'))

        m2.setObjective(w_x_fixed + (rho/2)*qsum(contr2_abs_1[i,j] for i in range(K) for j in range(H)), GRB.MINIMIZE)

        m2.update()
        start_opt = time.time()
        m2.optimize()
        end = time.time()
        time_stamps.append(end-start)

        changes_y = 0
        for j in range(K):
            if np.any(np.abs(np.rint(np.array(y.X)[j, :])) != np.abs(np.rint(y_par[j, :]))):
                changes_y += 1

        y_par = np.copy(np.array(y.X))

        # update dual variables
        temp_mu = np.zeros((K,H))  # just making sure I don't make a mistake
        for j in range(H):
            for i in range(K):
                temp_sum = []
                for t in range(T):
                    temp_sum += [x[j,i,t].X]
                temp_mu[i,j] = np.copy(mu[i,j] + (sum(temp_sum)-(y[i,j].X*proc_fwd[i,j])))

        mu = np.copy(temp_mu)

        # Calculate original objective function:

        calc_obj = np.zeros(K)
        temptemp = np.multiply(y.X, trans_back_activations)
        for i in range(K):
            calc_obj[i] = g_values[i] + np.sum(temptemp[i,:]) + proc_local_fwd[i]

        obj_per_iter += [max(calc_obj)]
        violated_constraints = 0
        total_constraints = 0
        for j in range(H):
            for i in range(K):
                if np.all(np.abs(np.rint(ll[j,i])) != proc_fwd[i,j]*np.abs(np.rint(y_par[i,j]))):
                    violated_constraints += 1
                total_constraints += 1
        
        violations.append((violated_constraints/total_constraints)*100)

        primal_residual = LA.norm((ll.T - np.multiply(y_par, proc_fwd)), 'fro')**2

        if changes_y <= 0:
            stable += 1
        else:
            stable = 0
        

        # Call Algorithm-2 to compute z variables
        if iter == 2: 
            flag_exit = True # mark it that we reached the end

        x_par = np.zeros((H,K,T))
        y_ = y_par
        all_time = []

        for i in range(H): # theoritically this can be done in parallel
            Kx = list(np.transpose(np.argwhere(y_[:,i]==1))[0]) # finds which data owners are assigned to the machine i
            if len(Kx) == 0:
                continue
            
            procx = np.copy(proc_fwd[Kx, i])  # this is a row-vector
            release_datex = np.copy(release_date_fwd[Kx, i])
            proc_localx = np.copy(proc_local_fwd[Kx])
            trans_backx = np.copy(trans_back_activations[Kx, i])
            Tx = np.max(release_datex) + len(Kx)*np.max(procx)  # to constrain the T
            start_sub = time.time()
            x__ = feasibility_check(len(Kx), release_datex, procx, proc_localx, trans_backx, memory_capacity[i], Tx)
            end_sub = time.time()

            machine_time = end_sub-start_sub

            jj = 0
            for j in Kx:
                for t in range(Tx):
                    x_par[i,j,t] = x__[0,jj,t]
                jj += 1
            
            f_temp = np.zeros((len(Kx)))
            for kk in range(len(Kx)):
                for t in range(Tx):
                    if f_temp[kk] < (t+1)*x__[0,kk,t]:
                        f_temp[kk] = (t+1)*x__[0,kk,t]
            
            procz = np.copy(proc_bck[Kx, i])  # this is a row-vector
            release_datez = np.copy(release_date_back[Kx, i])
            proc_localz = np.copy(proc_local_back[Kx])
            trans_backz = np.copy(trans_back_gradients[Kx, i])

            for kk in range(len(Kx)):
                release_datez[kk] += f_temp[kk] + proc_localx[kk] + trans_backx[kk]
            
            Tz = np.max(release_datez) + len(Kx)*np.max(procz)  # to constrain the T
            x__extend = np.zeros((1,len(Kx),Tz))
            
            for jj in range(len(Kx)):
                for t in range(min(Tx,Tz)):
                    x__extend[0,jj,t] = x__[0,jj,t]
            
            start_sub = time.time()

            # SELECT ALGORITHM -- P_b ----- SELECT ONLY ONE

            # USE SOLVER UNCOMMENT BELOW
            z__ = backward_for_each_machine(len(Kx), release_datez, procz, proc_localz, trans_backz, memory_capacity[i], Tz, x__extend)
            
            # OR USE OUR IMPLEMENTATION:
            z__ = np.expand_dims(algo2(len(Kx), release_datez, procz, proc_localz, trans_backz, memory_capacity[i], Tz, x__extend), axis=0)
            
            # ----- SELECT ONLY ONE

            end_sub = time.time()
            machine_time += end_sub - start_sub

            jj = 0
            for j in Kx:
                for t in range(Tz):
                    z_par[i,j,t] = z__[0,jj,t]
                    #z_par[i,j,t] = z__[jj,t]
                jj += 1

            all_time.append(machine_time)
        
        time_stamps.append(max(all_time))
        cs = []
        cs_back = []
        reserved = [0 for i in range(H)]
        f_m = np.zeros(K)

        for i in range(K): #for all jobs
            my_machine = 0
            my_super_machine = 0
            last_zero = -1
            for my_machine in range(H):
                for k in range(T):
                    if np.rint(x_par[my_machine,i,k]) >= 1:
                        if last_zero < k+1:
                            last_zero = k+1
                            my_super_machine = my_machine
            fmax = last_zero
            f_m[i] = fmax
            C = fmax + proc_local_fwd[i] + trans_back_activations[i,my_super_machine]
            cs.append(C)
            reserved[my_super_machine] += 1

        if flag_exit: # we are at the end, and have solved backward() as well
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

            #print(f'{utils.bcolors.FAIL}BACK max is: {max(cs_back)}{utils.bcolors.ENDC}')
       
        if flag_exit:
            obj_per_iter += [max(cs_back)]
   
    total_time = 0
    for t in time_stamps:
        total_time += t
    
    return (obj_per_iter, total_time)

