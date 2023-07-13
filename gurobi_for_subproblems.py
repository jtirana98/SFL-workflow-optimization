import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum

import utils
import time

import warnings
warnings.filterwarnings("ignore")


def for_each_machine(K, release_date, proc, proc_local, trans_back, memory_capacity, T):
    H = 1

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("fwd_only")

    # define variables
    print(f" Memory: {memory_capacity}")
    print(f"T: {T}")
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
    
    # C4: memory constraint
    #m.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )
    
    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc[i])

    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*x[j,i,t] for t in range(T))

    # Define objective function

    m.addConstrs(comp[i] == qsum(trans_back[i] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    
    m.update()
    # Optimize model
    m.optimize()
   

    return(x.X)

def for_each_machine_fwd_back(K, release_date_fwd, proc_fwd, proc_local_fwd, trans_back_activations,
                            release_date_back, proc_bck, proc_local_back, trans_back_gradients, memory_capacity, T):
    H = 1

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("fwd_only")

    # define variables
    print(f" Memory: {memory_capacity}")
    print(f"T: {T}")
    
    x = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
    y = m.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
    z = m.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="z")

    f = m.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
    
    maxobj = m.addMVar(shape=(1),vtype=GRB.INTEGER, name="maxobj")
    comp = m.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp")


   # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date_fwd[j]:
                    m.addConstr(x[i,j,t] == 0)   
    
    # C3: all jobs interval are assigned to one only machine
    m.addConstr( y @ ones_H == ones_K )

    # C4: memory constraint
    m.addConstr((y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))
    
    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T @ ones_K <= ones_T )

    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstr( qsum(x[j,i,:]) == y[i,j]*proc_fwd[j])

    for i in range(K): #for all jobs
        for j in range(H):
            temp = proc_local_fwd[i] + trans_back_activations[i] + release_date_back[j]

            for t in range(temp):
                m.addConstr(z[j,i,t] == 0)

            for t in range(T):
                if t + temp >= T:
                    break
                
                m.addConstr(z[j,i,t+temp] <= qsum(x[j,i,l] for l in range(t))/proc_fwd[i])

    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        for j in range(H):
            m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc_bck[i] == y[i,j])
    
    # C11: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T + z[j,:,:].T @ ones_K <= ones_T )
 
        
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*z[j,i,t] for t in range(T))


    # Define objective function
    
    m.addConstrs(comp[i] == qsum(trans_back_gradients[i] * y[i,:]) + f[i] + proc_local_back[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))


    m.update()
    m.optimize()
   
    return(x.X,z.X)


def for_each_machine_back(K, release_date, proc, proc_local, trans_back, memory_capacity, T, x):
    H = 1

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    m = gp.Model("fwd_only")

    # define variables
    print(f" Memory: {memory_capacity}")
    print(f"T: {T}")
    
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

    # C10: backprop job should be processed entirely once and in the same machine as fwd
    for i in range(K): #for all jobs
        for j in range(H):
            m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc[i] == y[i,j])
    
    # C11: machine processes only a single job at each interval
    for j in range(H): #for all devices
        m.addConstr( x[j,:,:].T + z[j,:,:].T @ ones_K <= ones_T )
 
        
    for j in range(H): #for all machines
        for i in range(K):
            m.addConstrs( f[i] >= (t+1)*z[j,i,t] for t in range(T) )

    # Define objective function

    m.addConstrs(comp[i] == qsum(trans_back[i] * y[i,:]) + f[i] + proc_local[i] for i in range(K))
       
    
    max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))
    
    m.setObjective(maxobj, GRB.MINIMIZE)
    
    m.update()
    m.optimize()
   
    return(z.X)


def main():
    K = 50 # number of data owners
    H = 5 # number of compute nodes
    # utils.file_name = 'fully_heterogeneous.xlsx'
    np.random.seed(9)
    # release_date = np.array(utils.get_fwd_release_delays(K,H))
    release_date = np.random.randint(3, size=(K,H))
    # release_date = (2*np.ones((K,H))).astype(int)
    #memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    memory_capacity = np.array([10, 10, 100, 100, 100])
    # memory_capacity = np.array([500, 500])
    # proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    memory_demand = 3
    # proc = np.array([[9, 10, 11], [16, 10, 9], [6, 8, 6], [8, 9, 11], [12, 9, 6], [8, 6, 11]])
    proc = np.random.randint(3,10, size=(K,H))
    proc[:, 1] = 2
    # proc_local = np.array(utils.get_fwd_end_local(K))
    proc_local = np.random.randint(2,8, size=(K))
    # proc = (5*np.ones((K,H))).astype(int)
    # trans_back = np.array(utils.get_trans_back(K, H))
    # trans_back = np.array([[7, 6, 1], [2, 6, 10], [5,5,5], [2, 2, 6], [9,8, 7], [6,1,4]])
    trans_back = np.random.randint(3, size=(K,H))
    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")
    print(f" Memory: {memory_capacity}")


    y_given = np.empty((K,H))
    with open('ymatrix1b.txt', 'r') as f1:
        lines = f1.readlines()
        ii = 0
        for line in lines:
            y_given[ii, :] = np.array([float(k) for k in line.split()])
            ii += 1

    y_given = np.copy(np.rint(y_given))
    print(y_given)
    print(np.sum(y_given, axis=1))
    max_compl_time_per_machine = []
    for j in range(H):
        #for each machine I solve a subproblem
        Kx = list(np.transpose(np.argwhere(y_given[:,j]==1))[0])  # finds which data owners are assigned to the machine j
        print("!!!!------assigned jobs at machine", j, "are the following:", Kx)
        # now we take the submatrices for the input parameters
        procx = np.copy(proc[Kx, j])  # this is a row-vector
        release_datex = np.copy(release_date[Kx, j])
        proc_localx = np.copy(proc_local[Kx])
        trans_backx = np.copy(trans_back[Kx, j])
        Tx = np.max(release_datex) + len(Kx)*np.max(procx)  # to constrain the T


        m = gp.Model("x_subproblem")

    # define variables
        Kx = len(Kx)
        print(Kx)
        x = m.addMVar(shape = (Kx,Tx), vtype=GRB.BINARY, name="x")
        f = m.addMVar(shape=(Kx),  name="f")
        maxobj = m.addMVar(shape=(1), name="maxobj")
        comp = m.addMVar(shape=(Kx), name="comp")



        start = time.time()
        # define constraints
        # C1: job cannot be assigned to a time interval before the release time

        for i in range(Kx): #for all jobs
            for t in range(Tx): #for all timeslots
                if t < release_datex[i]:
                    m.addConstr(x[i, t] == 0)




    # C6: machine processes only a single job at each interval
        ones_K = np.ones((1, Kx))
        ones_T = np.ones((1, Tx))
        m.addConstr(ones_K @ x <= ones_T)

    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
        for i in range(Kx):
            m.addConstr( qsum(x[i,:]) == procx[i])

        for i in range(Kx):
            m.addConstrs( f[i] >= (t+1)*x[i,t] for t in range(Tx))

        # Define objective function
        m.addConstrs(comp[i] == trans_backx[i] + f[i] + proc_localx[i] for i in range(Kx))


        max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(Kx)))

        m.setObjective(maxobj, GRB.MINIMIZE)
        end = time.time()
        time1 = end-start
        #print(f'problem formulation: {time1}')


        start = time.time()
        # Optimize model
        m.optimize()
        end = time.time()
        #print(f'problem solver: {end-start}')
        #print(f'TOTAL: {(end-start) + time1}')


    #print('%s %g' % (v.VarName, v.X))
        print('----------------------------------Obj: %g' % m.ObjVal)

        max_compl_time_per_machine.append(m.ObjVal)
        # print(x.X)
        m.reset()

    print("completion time per machine:", max_compl_time_per_machine)
if __name__ == '__main__':
    main()
