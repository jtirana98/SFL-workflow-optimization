import numpy as np
import cvxpy as cp

import utils

T = 10 # time intervals
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

proc_local = np.array(utils.get_fwd_end_local(K))
trans_back = np.array(utils.get_trans_back(K, H))


proc_param = cp.Parameter((K, H))
#f = cp.Parameter((K))
trans_back_param = cp.Parameter((K))
trans_back_pp = cp.Parameter((K, H))

#f = np.array(np.zeros(K))
trans_back_pp.value  = np.array(trans_back)
trans_back_param.value  = np.array(np.zeros(K))
proc_param.value = np.array(proc)
# Define constraints
constraints = []

# C1: job cannot be assigned to a time interval before the release time
for i in range(K): #for all jobs
    for j in range(H): #for all devices
        for t in range(T): #for all timeslots
            if t < release_date[i,j]: # <= ?? -- minor
                constraints += [x[i][j,t] == 0]

#constraints += [x[0][0][6] == 1]
#constraints += [x[1][1][6] == 1]

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

#for i in range(K): #for all jobs
#    for j in range(H):
#        constraints += [cp.sum(x[i][j,:]/proc[i,j]) == 1]
# C5: job should be processed entirely once  #NOTE: I am not sure about this one
for i in range(K): #for all jobs
    #sum_ = cp.sum(cp.sum(x[i][ :, :], axis=1)[:-1] / proc_param[i, :])
    sub_sum = []
    for j in range(H):
        sub_sum += [cp.sum(x[i][ j, :] )/ proc_param[i, j]]
    
    sum_ = cp.sum(cp.hstack(sub_sum))
    constraints += [sum_ == 1]

# C6: machine processes only a single job at each interval
for j in range(H): #for all devices
    for t in range(T): #for all timeslots
        temp = 0
        for key in x:
            temp += x[key]
        constraints += [temp <= 1]

#C7: at each time interval a job is processed in at most once  # NOTE: maybe we can ommit that TO BE DISCUSSED
#for i in range(K): #for all jobs
#    for t in range(T): #for all timeslots
#        constraints += [cp.sum(x[i][:,t]) <= 1]


#C8: the completition time for each data owner
for i in range(K): #for all jobs
    for j in range(H): #for all machines
        for t in range(T): #for all timeslots
            constraints += [f[i] >= (t+1)*x[i][j,t]]
            #print(constraints[-1])

# Define objective function

#C = cp.Parameter(K) # complete time for each data owner

trans = []
for i in range(K): # for each job/data owner
    trans.append(cp.sum(trans_back_pp[i,:] * y[i,:]))



trans_back_param.value = cp.sum([trans_back_pp[i,:] * y[i,:] for i in range(K)]).value

obj = cp.Minimize(cp.max( f + proc_local + cp.hstack(trans)))

# wrap the formula to a Problem
prob = cp.Problem(obj, constraints)
#prob.solve(solver=cp.GUROBI, verbose=True)
prob.solve(solver=cp.MOSEK, verbose=True, 
           mosek_params={
                'MSK_IPAR_NUM_THREADS': 2,
                },
            save_file = 'dump_dump.ptf',)
#prob.params.StartNumber = 10
# solve
prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal time allocation", x[0].value)
print("optimal time allocation", x[1].value)
print("optimal device allocation", y.value)
print("optimal end time", f.value)