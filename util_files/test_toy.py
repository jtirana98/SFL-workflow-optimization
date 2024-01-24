import numpy as np
from numpy import linalg as LA
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum
import warnings
import time

import utils 
warnings.filterwarnings("ignore")

H = 1



m = gp.Model("fwd_only")

num_clients = 5
K = num_clients
release_date = np.array([0, 2, 3, 1, 9]) 
proc = np.array([1, 2, 3, 2, 1])
proc_local = np.array([5, 3, 8, 1, 1])


release_date = np.array([1, 2, 3, 1, 4]) 
proc = np.array([1, 2, 4, 2, 1])
proc_local = np.array([5, 3, 1, 1, 2])


memory_capacity = 100 

T = np.max(release_date) + num_clients*np.max(proc)

ones_H = np.ones((H,1))
ones_K = np.ones((K,1))
ones_T = np.ones((T,1))

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

# C10: backprop job should be processed entirely once and in the same machine as the fwd
for i in range(K): #for all jobs
    for j in range(H):
        m.addConstr(qsum(z[j, i, t] for t in range(T))/ proc[i] == y[i,j])

# C11: machine processes only a single job at each interval
for j in range(H): #for all devices
    m.addConstr(z[j,:,:].T @ ones_K <= ones_T )

    
for j in range(H): #for all machines
    for i in range(K):
        m.addConstrs( f[i] >= (t+1)*z[j,i,t] for t in range(T) )

# Define the objective function
m.addConstrs(comp[i] == f[i] + proc_local[i] for i in range(K))
    

max_constr = m.addConstr(maxobj == gp.max_(comp[i] for i in range(K)))

m.setObjective(maxobj, GRB.MINIMIZE)

m.update()
m.optimize()


print(f'slots:', end='\t')
            
for i in range(T):
    print(f'{i}', end='\t')
print('')


print(f'alloc:', end='\t')
for i in range(T):
    client = 0
    for j in range(num_clients):
        if z.X[0][j][i] > 0:
            client = j+1
    
    print(f'{client}', end='\t')
print('')

print(f'THE MAKESPAN IS {m.objVal}')