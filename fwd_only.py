import numpy as np
import cvxpy as cp

import utils

T = 2 # time intervals
K = 2 # number of data owners
H = 2 # number of compute nodes


# Define variables
#x = cp.Variable((K,H, T), boolean=True) DOES NOT SUPPORT 3D
# job allocation to time and machine

#n = K*H  # total number of elements
#x_2d = cp.Variable((n, T), boolean=True)
#x = np.reshape(x_2d, (K, H, T))

x = {}
for i in range(K):
  x[i] = cp.Variable((H,T), boolean=True)

y = cp.Variable((K,H), boolean=True) # auxiliary variable
f = cp.Variable(K, integer=True) # completition time


release_date = np.array(utils.get_fwd_release_delays(K,H)) # release date - shape (K,H) 
memory_capacity = np.array(utils.get_memory_characteristics(H))
proc = np.array(utils.get_fwd_proc_compute_node(K, H))
proc_local = np.array(utils.get_fwd_end_local(H))
trans_back = np.array(utils.get_trans_back(K, H))



# Define constraints
constraints = []

# C1: job cannot be assigned to a time interval before the release time
for i in range(K): #for all jobs
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            if t < release_date[i,j]: # <= ??
                constraints += [x[i][j,t] == 0]



# C2: define auxiliary variable
for i in range(K): #for all jobs
    for j in range(H): #for all devices
        constraints += [cp.sum(x[i][j,:]) <= T*y[i,j]]

# C3: all jobs interval are assigned to one only machine
for i in range(K): #for all jobs
    constraints += [cp.sum(y[i,:]) == 1]

# C4: memory constraint
for j in range(H): #for all devices
    constraints += [cp.sum(y[:,j])*utils.max_memory_demand <= memory_capacity[j]] 


# C5: job should be processed entirely once
for i in range(K): #for all jobs
    constraints += [cp.sum(x[i][j,t]/proc[i,j]) == 1 for j in range(H) for t in range(T)]

# C6: machine processes only a single job at each interval
for j in range(H): #for all devices
    for t in range(T): #for all timeslots
        constraints += [cp.sum(x[i][j,t]) <= 1 for i in range(K)]


#C7: at each time interval a job is processed in at most once  # NOTE: maybe we can ommit that TO BE DISCUSSED
for i in range(K): #for all jobs
    for t in range(T): #for all timeslots
        constraints += [cp.sum(x[i][:,t]) <= 1]


#C8: the completition time for each data owner
for i in range(K): #for all jobs
    for j in range(H): #for all machines
        for t in range(T): #for all timeslots
            constraints += [f[i] >= (t+1)*x[i][j,t]]

# OR:
for i in range(K): #for all jobs
    constraints += [f[i] >= cp.max((t+1)*x[i][j,t]) for j in range(H) for t in range(T)]

# Define objective function

C = np.ones(K) # complete time for each data owner

trans = []
for i in range(K): # for each job/data owner
    trans.append(cp.sum(y[i,j]*trans_back[i,j]) for j in range(H))
    #C[i] = cp.sum(f[i] + np.array(trans)[0] + proc_local[i])

obj = cp.Minimize((cp.maximum(f+ np.array(trans) + proc_local)))

'''
# wrap the formula to a Problem
prob = cp.Problem(obj, constraints)

# solve
prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal time allocation", x.value)
print("optimal device allocation", y.value)
'''


