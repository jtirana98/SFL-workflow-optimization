'''
This is implementation of Algorithm-2 for one helper. Note this contains values from toy examples.
'''

import numpy as np

def print_res(alloc, duration, num_clients):
    print(f'slots:', end='\t')
            
    for i in range(duration):
        print(f'{i}', end='\t')
    print('')


    print(f'alloc:', end='\t')
    for i in range(duration):
        client = 0
        for j in range(num_clients):
            if alloc[j][i] > 0:
                client = alloc[j][i]
        
        print(f'{client}', end='\t')
    print('')

def main():
    num_clients = 5
    arriv = np.array([0, 2, 3, 1, 9]) # arrival time at helper 
    proc = np.array([1, 2, 3, 2, 1]) # process time at helper
    ll = np.array([5, 3, 8, 1, 1]) # transmission an proc at the client
    duration = sum(arriv) + sum(proc)

    allocation = np.zeros((num_clients,duration))
    # step - 1 order clients by releasing date
    order = np.argsort(arriv)
    print(order)
    for i in range(num_clients):
        client = order[i]
        start = arriv[client]
        clear = True
        reserved = 0
        
        for i in range(start, duration):
            clear = True
            for j in range(num_clients):
                if allocation[j][i] == j+1:
                    clear = False
                    start = start + 1
            if clear == True:
                break
        
        for j in range(proc[client]):
            allocation[client][start+j] = client+1


    print('Initial allocation')
    print_res(allocation, duration, num_clients)

    # step - 2 define blocks
    '''
    for each block I store a list of two values bita_i = [[slots_i], [clients_i]]
    slots_i = [s(bita_i), e(bita_i)]
    clients_i : list of clients inside that slot
    '''
    B = {}
    start = 0
    new_set = False
    iter = 0
    next = ([0,0], [])
    for i in range(duration):
        if new_set == False:
            for j in range(num_clients):
                if allocation[j][i] != 0:
                    new_set = True
                    iter += 1
                    next[0][0] = i
                    next[1].append(j+1)
                    break
            continue
        else:
            is_end = True
            for j in range(num_clients):  
                if allocation[j][i] != 0:
                    is_end = False
                    if j+1 not in next[1]:
                        next[1].append(j+1)
                    break
            if is_end:
                next[0][1] = i
                B.update({iter:next})
                next = ([0,0], [])
                new_set = False

    print('The initial blocks')
    for bita in list(B.keys()):
        print(f'block {bita}')
        print(f'start-finish: {B[bita][0][0],B[bita][0][1]}')
        print(f'clients in set: {B[bita][1]}')

    for bita in list(B.keys()):
        if len(B[bita][1]) == 1: # no need to reschedule
            continue
        
        bloc = B[bita]
        # step - 3 find l-client
        l = -1
        min = duration+100
        for client in bloc[1]:
            temp = bloc[0][1]+ll[client-1]
            if temp < min:
                min = temp
                l = client
        print(f'l client is {l}')

        # step 4 - reschedule



    '''


    for bita in list(B.keys()):
        if len(B[bita]) == 1:
            continue
        subset = [bita]
        fixed = 0 # number of sub-blocks that cannot change any more
        while fixed < len(subset):
            for i in range(len(subset)):
                if len(subset[i]) == 1:
                    fixed += 1

                # step - 3 find l-client
                l = 0

                # step 4 - reschedule

                #if cannot reschedule:
                fixed += 1

                # step 5 - update subset
                # remove subset[i] from subset
                # ftiakse nea subset thewrontas ton l idle
                subset.append()
    '''
    
    # show result
    makespan = 0
    finish_clients = [0 for i in range(num_clients)]


if __name__ == '__main__':
    main()