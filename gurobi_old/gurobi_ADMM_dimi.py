import numpy as np
import math

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum

import random
from numpy import linalg as LA
import utils
import time

import warnings
warnings.filterwarnings("ignore")

MAX_ITER = 20
rho = 0.6

K = 7 # number of data owners
H = 5 # number of compute nodes
# utils.file_name = 'fully_heterogeneous.xlsx'
np.random.seed(4)
# release_date = np.array(utils.get_fwd_release_delays(K,H))
release_date = np.random.randint(3, size=(K,H))
# release_date = (2*np.ones((K,H))).astype(int)
#memory_capacity = np.array(utils.get_memory_characteristics(H, K))
memory_capacity = np.array([10, 10, 100, 100, 100])
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

# print("local proc")  # [3 7 7 6 4 5 5 6 2 6]


T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
print(f"T = {T}")



ones_H = np.ones((H,1))
ones_K = np.ones((K,1))
ones_T = np.ones((T,1))

m1 = gp.Model("xsubproblem")
m2 = gp.Model("ysubproblem")


# define variables -x subproblem
x = m1.addMVar(shape = (H,K,T), vtype=GRB.BINARY, name="x")
f = m1.addMVar(shape=(K), vtype=GRB.INTEGER, name="f")
comp = m1.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp") # completion time
w = m1.addMVar(shape=(1),vtype=GRB.INTEGER, name="w")  # min of compl. times

# define variables -y subproblem
y = m2.addMVar(shape=(K,H), vtype=GRB.BINARY, name="y")
comp_x_fixed = m2.addMVar(shape=(K),vtype=GRB.INTEGER, name="comp_x_fixed") # completion time
w_x_fixed = m2.addMVar(shape=(1),vtype=GRB.INTEGER, name="w_x_fixed")  # min of compl. times
# Auxiliary
contr1_add_1 = m1.addMVar(shape=(K,H), name="contr1_add_1")
contr1_abs_1 = m1.addMVar(shape=(K,H), name="contr1_abs_1")  # auxiliary for abs value

contr2_add_1 = m2.addMVar(shape=(K,H), name="contr2_add_1")
contr2_abs_1 = m2.addMVar(shape=(K,H), name="contr2_abs_1")   # auxiliary for abs value

# "Parameters"
# x_par = np.zeros((H,K,T))
y_par = np.zeros((K,H))
# f_par = np.zeros((K))
w_par = np.zeros((1))
# dual variables
mu = np.ones((K,H)) # dual variable

# C3: each job is assigned to one and only machine
m2.addConstr(y @ ones_H == ones_K)
# m2.addConstrs(qsum(y[i,:]) == 1 for i in range(K))

# C4: memory constraint
m2.addConstr((y.T * memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape))

# completition time definition
m1.addConstrs(f[i] >= (t+1)*x[j, i, t] for i in range(K) for j in range(H) for t in range(T))

m1.addConstrs(comp[i] == np.sum(np.multiply(trans_back[i,:], y_par[i,:])) + f[i] + proc_local[i] for i in range(K))
max_constr = m1.addConstrs(w >= comp[i] for i in range(K))


#m2.addConstrs(comp_x_fixed[i] == qsum(trans_back[i,:]*y[i,:]) + f_par[i] + proc_local[i] for i in range(K))
#max_constr_x_fixed = m2.addConstrs(w_x_fixed >= comp_x_fixed[i] for i in range(K))

# C1: job cannot be assigned to a time interval before the release time
for i in range(H): #for all devices
    for j in range(K): #for all jobs
        for t in range(T): #for all timeslots
            if t < release_date[j,i]:
                m1.addConstr(x[i,j,t] == 0)

# C6: machine processes only a single job at each interval
for j in range(H): #for all devices
    m1.addConstr( x[j,:,:].T @ ones_K <= ones_T )


# forgoten constraint
# C5: job should be processed entirely once
for i in range(K):
    # m1.addConstr(qsum(qsum(x[j,i,t] for t in range(T))/proc[i,j] for j in range(H)) == 1)
    m1.addConstr(qsum(qsum(x[j,i,t] for t in range(T)) for j in range(H)) >= min(proc[i,:]))
# Iterative algorithm


violations_1 = []
violations_2 = []
max_c = []
accepted = []
ws = []
obj1 = []
obj2 = []
max_ = T
my_ds2 = []
my_ds = []
#add = False
# aux_abs = m1.addMVar(shape=1, vtype=GRB.BINARY, name="aux_abs")

for iter in range(MAX_ITER):

    if iter >= 1:
        for d in my_ds:
            m1.remove(d)
        my_ds = []

    for i in range(K):
        for j in range(H):
            if iter >= 1:
                c = m1.getConstrByName(f'const1add-{i}-{j}')
                m1.remove(c)

            m1.addConstr(contr1_add_1[i,j] == qsum(x[j,i,t] for t in range(T)) - y_par[i,j]*proc[i,j] + mu[i,j], name=f'const1add-{i}-{j}')
            my_ds.append(m1.addConstr(contr1_abs_1[i,j] == gp.abs_(contr1_add_1[i,j]), name=f'const1ab-{i}-{j}'))


    m1.setObjective(w + (rho/2)*qsum(contr1_abs_1[i,j] for i in range(K) for j in range(H)), GRB.MINIMIZE)

    m1.update()

    print('---------------------------------------------------')
    print(len(m1.getConstrs()))
    print('---------------------------------------------------')

    """
    if iter < 3:
        m1.setParam('MIPGap', 0.15) # 5%
    else:
        m1.setParam('MIPGap', 0.0001)
    """
    # solve P1:
    start = time.time()
    m1.optimize()
    end = time.time()
    print(f'{utils.bcolors.OKBLUE}P1 took: {end-start}{utils.bcolors.ENDC}')
    print(f'{utils.bcolors.OKBLUE}Obj1: {m1.ObjVal}{utils.bcolors.ENDC}')
    print(x.X, w.X)
    # [4. 2. 1. 2. 4. 4. 3. 2. 5. 3.]

    # pass results to second problem
    x_par = np.copy(np.array(x.X))
    # f_par = np.copy(np.array(f.X)) # f are variables and not calculated correctly
    w_par = np.copy(np.array(w.X))

    # Now calculate the value of (original) objective function at this iter
    g_values = []
    for i in range(K): #for all jobs
        g_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                g_interm += [(t+1)*x[j,i,t].X]
        g_values += [max(g_interm)]

    f_par = np.copy(g_values)
    # f_par = 90*np.ones((K)) ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # f_par = np.array([4, 2, 1, 2, 4, 4, 3, 2, 5, 3])
    calc_obj = np.zeros(K)
    for i in range(K):
        calc_obj[i] = g_values[i] + np.sum(np.multiply(y_par, trans_back)) + proc_local[i]
    print(calc_obj)
    
    if iter >= 1:
        for dd in my_ds2:
            m2.remove(dd)
        my_ds2 = []
    
    if iter == 0:
        x_par = np.ones((H,K,T))


    ll = np.sum(x_par, axis=2)

    # ll = np.array([[1, 2, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 2, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 2, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 2, 1], [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]])

    #print(g_values)
    #print("9999999999999999", ll)
    print(rho/2*np.sum(contr1_abs_1.X))
    
    for i in range(K):
        for j in range(H):
            if iter >= 1:
                c2 = m2.getConstrByName(f'const2add-{i}-{j}')
                m2.remove(c2)

            m2.addConstr(contr2_add_1[i,j] == ll[j,i] - y[i,j]*proc[i,j] + mu[i,j], name=f'const2add-{i}-{j}')
            my_ds2.append(m2.addConstr(contr2_abs_1[i,j] == gp.abs_(contr2_add_1[i,j]), name=f'const2ab-{i}-{j}'))
    

    for i in range(K):
        if iter >= 1:
            f2 = m2.getConstrByName(f'const2f-{i}')
            m2.remove(f2)
            #fw2 = m2.getConstrByName(f'const2fw-{i}')
            #m2.remove(fw2)

        m2.addConstr(comp_x_fixed[i] == qsum(trans_back[i,:]*y[i,:]) + f_par[i] + proc_local[i], name=f'const2f-{i}')
        my_ds2.append(m2.addConstr(w_x_fixed >= comp_x_fixed[i], name=f'const2fw-{i}'))



    m2.setObjective(w_par + (rho/2)*qsum(contr2_abs_1[i,j] for i in range(K) for j in range(H)), GRB.MINIMIZE)
    


    m2.update()

    print('-------------------!!!!!!!!!!!--------------------------------')
    print(len(m2.getConstrs()))
    print('---------------------------------------------------')

    start = time.time()
    m2.optimize()
    end = time.time()
    print(f'{utils.bcolors.OKBLUE}P2 took: {end-start}{utils.bcolors.ENDC}')
    # print(m2.computeIIS())
    print(f'{utils.bcolors.OKBLUE}Obj2: {m2.ObjVal}{utils.bcolors.ENDC}')
    print(rho/2*np.sum(contr2_abs_1.X))
    obj1 += [m1.ObjVal]
    obj2 += [m2.ObjVal]

    # pass results to first problem
    y_par = np.copy(np.array(y.X))


    # update dual variables
    temp_mu = np.zeros((K,H))  # just making sure I don't make a mistake
    for j in range(H):
        for i in range(K):
            temp_sum = []
            for t in range(T):
                temp_sum += [x[j,i,t].X]
            temp_mu[i,j] = np.copy(mu[i,j] + (sum(temp_sum)-(y[i,j].X*proc[i,j])))

    mu = np.copy(temp_mu)

    # Calculate original objective function:
    calc_obj = np.zeros(K)
    for i in range(K):
        calc_obj[i] = g_values[i] + np.sum(np.multiply(y.X, trans_back)) + proc_local[i]
    print("---------------------objective equals to:", max(calc_obj), iter)
