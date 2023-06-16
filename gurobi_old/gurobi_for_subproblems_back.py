import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum as qsum

import utils
import time

import warnings
warnings.filterwarnings("ignore")

#np.savetxt('data_for_plot.txt', y.X, delimiter=' ', newline='\n')

def main():
    K = 50 # number of data owners
    H = 5 # number of compute nodes
    # utils.file_name = 'fully_heterogeneous.xlsx'
    np.random.seed(9)


    proc_back_server_part = np.random.randint(3,10, size=(K,H))
    # proc[:, 1] = 2
    proc_local_back_last_layers = np.random.randint(2,8, size=(K))
    proc_local_back_first_layers = np.random.randint(2,8, size=(K))
    trans_back = np.random.randint(3, size=(K,H))

    # Important: T=0 translates to min(compl. time) of the forward problem


    y_given = np.empty((K,H))
    with open('y_matrix_test.txt', 'r') as f1:
        lines = f1.readlines()
        ii = 0
        for line in lines:
            y_given[ii, :] = np.array([float(k) for k in line.split()])
            ii += 1


    compl_given = np.empty((K))
    with open('completion_times_file.txt', 'r') as f2:
        lines = f2.readlines()
        ii = 0
        for line in lines:
            compl_given = np.array([float(k) for k in line.split()])

    release_date_back = np.zeros((K,H))
    # the release_date_back is the sum of proc. time for last layers (gradients) locally
    # and the transmission time to the compute node
    for i in range(K):
        release_date_back[i,:] = proc_local_back_last_layers[i].astype(int) + trans_back[i,:].astype(int)
        #print(np.rint(release_date_back[i,:]).astype(int))


    #print(f"T = {T}")

    y_given = np.copy(np.rint(y_given))
    print(y_given)
    print(compl_given)
    print(release_date_back)

    compl_given = np.copy(np.rint(compl_given))

    max_compl_time_per_machine = []
    max_compl_time_per_machine_2 = []
    for j in range(H):
        #for each machine I solve a subproblem
        Kx = list(np.transpose(np.argwhere(y_given[:,j]==1))[0])  # finds which data owners are assigned to the machine j
        print("!!!!------assigned jobs at machine", j, "are the following:", Kx)
        # now we take the submatrices for the input parameters
        find_min_compl_time = min(compl_given[Kx])
        procx = np.copy(proc_back_server_part[Kx, j])  # this is a row-vector
        release_datex = np.copy(release_date_back[Kx, j])
        proc_localx = np.copy(proc_local_back_first_layers[Kx])
        trans_backx = np.copy(trans_back[Kx, j])
        Tx = np.max(release_datex) + len(Kx)*np.max(procx)  # to constrain the T

        Tx=Tx.astype(int)


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

        # Optimize model
        m.optimize()



    #print('%s %g' % (v.VarName, v.X))
        print('----------------------------------Obj: %g' % m.ObjVal)

        max_compl_time_per_machine.append(m.ObjVal) # this is in terms of local time horizon for bwd only
        max_compl_time_per_machine_2.append(m.ObjVal + find_min_compl_time) # this is in terms of local time horizon for fwd+bwd

        # print(x.X)
        m.reset()


    print("completion time per machine (for backward only):", max_compl_time_per_machine)
    print("This translates to a TOTAL completion time per machine(fwd+bwd):", max_compl_time_per_machine_2)
if __name__ == '__main__':
    main()
