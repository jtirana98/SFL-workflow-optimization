import numpy as np
import cvxpy as cp
import time

import utils

import warnings
warnings.filterwarnings("ignore")



def main():

    MAX_ITER = 20

    K = 20 # number of data owners
    H = 3 # number of compute nodes
    utils.file_name = 'fully_heterogeneous.xlsx'

    # fully_symmetric
    # fully_heterogeneous
    # symmetric_machines
    # symmetric_data_owners

    # the numbers flip1 and flip2 are used at the flipping step (step 9)
    flip1 = np.rint(K*H/2)
    flip2 = 2*np.rint(3*K*H/2)

    release_date = np.array(utils.get_fwd_release_delays(K,H))
    memory_capacity = np.array(utils.get_memory_characteristics(H, K))
    proc = np.array(utils.get_fwd_proc_compute_node(K, H))
    proc_local = np.array(utils.get_fwd_end_local(K))
    trans_back = np.array(utils.get_trans_back(K, H))
    f = cp.Parameter((K))


    T = np.max(release_date) + K*np.max(proc[0,:]) # time intervals
    print(f"T = {T}")

    ones_H = np.ones((H,1))
    ones_K = np.ones((K,1))
    ones_T = np.ones((T,1))

    print(f" Memory: {memory_capacity}")

    # Define variables (and associated parameters)
    x = {}
    x_integ = {}
    x_star = {}
    x_star2 = {}  # auxiliary parameters for flipping action
    x_integ2 = {}
    for i in range(H):
        x[i] = cp.Variable((K,T), nonneg=True)
        x_integ[i] = cp.Parameter((K,T), nonneg=True)
        x_star[i] = cp.Parameter((K,T), nonneg=True)
        x_integ2[i] = cp.Parameter((K,T+1), nonneg=True)  # T+1 here!
        x_star2[i] = cp.Parameter((K,T+1), nonneg=True)

    y = cp.Variable((K,H), nonneg=True) # auxiliary variable
    y_integ = cp.Parameter((K,H), nonneg=True)
    y_star = cp.Parameter((K,H), nonneg=True)


    # Define constraints
    constraints = []

    # range of variables:
    constraints += [y <= 1]

    for i in range(H):
        constraints += [x[i] <= 1]

    # C1: job cannot be assigned to a time interval before the release time
    for i in range(H): #for all devices
        for j in range(K): #for all jobs
            for t in range(T): #for all timeslots
                if t < release_date[j,i]:
                    constraints += [ x[i][j,t] == 0 ]

    # C3: all jobs interval are assigned to one only machine
    constraints += [ y @ ones_H == ones_K ]

    # C4: memory constraint
    constraints += [ (y.T * utils.max_memory_demand) @ ones_K <= memory_capacity.reshape(ones_H.shape) ]

    # C6: machine processes only a single job at each interval
    for j in range(H): #for all devices
        constraints += [ x[j].T @ ones_K <= ones_T ]

    # C9: new constraint - the merge of C2 and C3 (job should be process all once and only in one machine)
    for j in range(H): #for all machines
        constraints += [cp.sum(x[j],axis=1) == cp.multiply(y,proc).T[j,:]]

    #C8: the completition time for each data owner -- NOTE: Removed it from constraints
    f_values = []
    for i in range(K): #for all jobs
        f_interm = []
        for j in range(H): #for all machines
            for t in range(T): #for all timeslots
                f_interm += [(t+1)*x[j][i,t]]
        f_values += [cp.max(cp.hstack(f_interm))]

    f = cp.hstack(f_values)

    # Define objective function
    trans = []
    for i in range(K): # for each job/data owner
        trans.append(cp.sum(trans_back[i,:] * y[i,:]))


    obj = cp.Minimize(cp.max( f + proc_local + cp.hstack(trans)))

    # wrap the formula to a Problem
    prob = cp.Problem(obj, constraints)

    prob.solve(solver=cp.GUROBI, verbose=False)

    print("The solution of the CONTINUOUS problem is:", prob.value, prob.status)

    # initialize x_star, y_star with the solution of the continuous problem
    # and check if they are integer... if yes, no need to go further

    y_star.value = y.value
    y_integ.value = np.rint(y_star.value)
    for i in range(H):
        x_star[i].value = x[i].value
        x_integ[i].value = np.rint(x_star[i].value)

    condlist = []

    for i in range(H): # I write it in 2 steps to avoid building a huge list
        # the loop will stop once a comparison does not give True
        ttemp = (y_integ[:,i].value == y_star[:,i].value)
        ttemp = np.reshape(ttemp, K).tolist()
        if all(ttemp):
            condlist.append(True)
        else:
            condlist.append(False)
            break
        ttemp2 = (x_integ[i].value == x_star[i].value)
        ttemp2 = np.reshape(ttemp2, (K*H)).tolist()
        if all(ttemp2):
            condlist.append(True)
        else:
            condlist.append(False)
            break


    if all(condlist):
        print("------The continuous problem gives integer solution------")
        return y.value  # or whatever...

    # Now, we built the second problem of minimizing the L1 norm
    delta_obj = cp.norm(y - y_integ, 1)
    for i in range(H):
        delta_obj += cp.norm(x[i] - x_integ[i], 1)

    obj2 = cp.Minimize(delta_obj)
    prob2 = cp.Problem(obj2, constraints)

    for iter in range(MAX_ITER):
        prob2.solve(solver=cp.GUROBI, verbose=False)
        # define x_star, y_star *BUT* not the x,y_integ yet!!
        # and check if we have an integer solution
        y_star.value = y.value
        print("Delta is equal to:", prob2.value, "at iteration:", iter)

        for i in range(H):
            x_star[i].value = x[i].value

        condlist = []  # this is to check is the solution obtained is integer
        condlist2 = []  # this is to check cond. of step 7 (is the rounding of the
        # solution equal to x_integ??)

        for i in range(H):
            ttemp = (np.rint(y_star[:,i].value) == y_star[:,i].value)
            ttemp = np.reshape(ttemp, K).tolist()
            ttemp2 = (np.rint(y_star[:,i].value) == y_integ[:,i].value)
            ttemp2 = np.reshape(ttemp2, K).tolist()
            if all(ttemp):
                condlist.append(True)
            else:
                condlist.append(False)
            if all(ttemp2):
                condlist2.append(True)
            else:
                condlist2.append(False)
            ttemp3 = (np.rint(x_star[i].value) == x_star[i].value)
            ttemp3 = np.reshape(ttemp3, (K,T)).tolist()
            ttemp4 = (np.rint(x_star[i].value) == x_integ[i].value)
            ttemp4 = np.reshape(ttemp4, (K,T)).tolist()
            if all(ttemp3):
                condlist.append(True)
            else:
                condlist.append(False)
            if all(ttemp4):
                condlist2.append(True)
            else:
                condlist2.append(False)


        if all(condlist):
            print("------------Done at iteration: ", iter)
            fff = []
            for i in range(K): #for all jobs
                f_interm = []
                for j in range(H): #for all machines
                    for t in range(T): #for all timeslots
                        f_interm += [(t+1)*x_integ[j].value[i,t]]
                fff += [max(f_interm)]




            for i in range(K): # for each job/data owner
                for j in range(H):
                    if y_star.value[i,j] == 1:
                        fff[i] = fff[i] + proc_local[i] + trans_back[i,j]



            objvalue_integ = max(fff)
            print("objective function value for INTEGER problem is:", objvalue_integ)
            return y.value  # or whatever...

        if all(condlist2):
            print("-----------calling the flipping function", iter)
            indic = []
            flipping_number = np.random.randint(flip1, flip2)
            print("flipping number:", flipping_number)
            corresponding_values = np.empty((H, flipping_number))
            # for ease, I y into x and, at the end, I will reshape
            for i in range(H):
                x_star2[i].value = np.hstack((x_star[i].value[:,:], np.reshape(y_star.value[:, i], (K, 1))))
                x_integ2[i].value = np.hstack((x_integ[i].value[:, :], np.reshape(y_integ.value[:, i], (K, 1))))
                shh = (x_star2[0].value).shape
                matr = np.abs(x_star2[i].value-x_integ2[i].value)
                te = (-matr).argsort(axis=None)[:flipping_number]
                # find the first (flipping_number)-indices of sorting
                # |x_Star-x_integ| in decreasing order
                te2 = np.vstack(np.unravel_index(te, shh)).T  # unflattened indices
                indd = [tuple(p) for p in te2]  # make a list of tuples
                indic.append(indd)
                # indices is a list of lists, first list corresponds to indices
                # of i=0, second list to i=2, and so on...
                for p in range(flipping_number):
                    corresponding_values[i,p] = matr[indd[p]] # keep an array of the
                    # corresponding values of |x_Star-x_integ| for these indices
                # Now I sort all the sublists (of different i's)

            corresponding_values = np.array(corresponding_values)
            cc = (-corresponding_values).argsort(axis=None)[:flipping_number]  # sort again
            cc2 = np.vstack(np.unravel_index(cc, corresponding_values.shape)).T  # unflattened indices
            indd2 = [tuple(p) for p in cc2]  # make a list of tuples
            # indd2[p][0] gives me the i index of x[i][j,t]
            # indd2[p][1] gives me the order inside the sublists (of different i's)
            # I want to find the first flipping_number of them & flip them

            for p in range(flipping_number):
                find_ind1 = indd2[p][0]  # gives me the i index of x
                find_ind2 = indd2[p][1]  # gives me the order inside the sublists (of different i's)
                find_ind3 = indic[find_ind1][find_ind2]  # this should be the tuple for x[i][:,:]
                # print("----find1,2,3:", find_ind1, find_ind2, find_ind3)
                x_integ2[find_ind1].value[find_ind3] = np.abs(x_integ2[find_ind1].value[find_ind3]-1)
                # this will flip from 1 to 0 and from 0 to 1
            # let's reshape back to the original shapes
            for i in range(H):
                y_integ.value[:, i] = x_integ2[i].value[:, -1]
                x_integ[i].value = x_integ2[i].value[:, :-1]
        else:
            print("-----------calling the rounding function", iter)
            # (x_integ, y_integ) = round_func(x_star.value, y_star.value, H)
            y_integ.value = np.rint(y_star.value)
            for i in range(H):
                x_integ[i].value = np.rint(x_star[i].value)

        #  I want now to find the objective function value of the integer problems
        #  Comment the following code when counting execution time!!

        fff = []
        for i in range(K): #for all jobs
            f_interm = []
            for j in range(H): #for all machines
                for t in range(T): #for all timeslots
                    f_interm += [(t+1)*x_integ[j].value[i,t]]
            fff += [max(f_interm)]




        for i in range(K): # for each job/data owner
            for j in range(H):
                if y_star.value[i,j] == 1:
                    fff[i] = fff[i] + proc_local[i] + trans_back[i,j]



        objvalue_integ = max(fff)
        print("objective function value for INTEGER problem is:", objvalue_integ)




if __name__ == '__main__':
    main()
