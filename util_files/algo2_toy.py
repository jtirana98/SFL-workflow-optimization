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
        if alloc[i] > 0:
            client = alloc[i]
        
        print(f'{client}', end='\t')
    print('')

def main():
    num_clients = 5
    arriv = np.array([0, 2, 3, 1, 9]) # arrival time at helper 
    proc = np.array([1, 2, 3, 2, 1]) # process time at helper
    ll = np.array([5, 3, 8, 1, 1]) # transmission an proc at the client

    '''
    arriv = np.array([1, 2, 3, 1, 4]) 
    proc = np.array([1, 2, 4, 2, 1])
    ll = np.array([5, 3, 1, 1, 2])
    '''
    duration = sum(arriv) + sum(proc)

    allocation = np.zeros((duration)) # if allocation[i] = 0 there is no task allocated, if allocation = k, k>0 then client k has been allocated at machine
    
    # step - 1 order clients by releasing date
    order = np.argsort(arriv)
    print(order)
    for ii in range(num_clients):
        client = order[ii]
        start = arriv[client]
        for i in range(start, duration):
            if allocation[i] == 0:
                break
            start = start + 1

        for j in range(proc[client]):
            allocation[start+j] = client+1


    print('Initial allocation')
    print_res(allocation, duration, num_clients)

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
    for i in range(duration):
        if new_set == False:
            if allocation[i] != 0:
                new_set = True
                iter += 1
                next[0][0] = i
                next[1].append(allocation[i])
        else:
            is_end = True
            if allocation[i] != 0:
                is_end = False
                if allocation[i] not in next[1]:
                    next[1].append(allocation[i])

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
        
        print(f'Starting exploring new MAIN block')
        
        sub_blocks = [B[bita]]
        block_i = 0
        while len(sub_blocks) > 0: 
            bloc = sub_blocks.pop(block_i)
            print(f'Starting exploring new sub-block block: start-{bloc[0][0]} end-{bloc[0][1]}  clients {bloc[1]}')
            if len(bloc[1]) == 1:
                print('skip block, has only one client')
                if len(sub_blocks) != 0:
                        block_i = (block_i + 1)%len(sub_blocks)
                continue
            
            

            # step - 3 find l-client
            l = -1
            min = duration+100
            for client in bloc[1]:
                temp = bloc[0][1]+ll[int(client)-1]
                if temp < min:
                    min = temp
                    l = client
            print(f'l-client is {l}')

            # step 4 - reschedule
            # l tasks should allocated in slots where no other client has been released

            # find start of l
            start_l = -1
            for i in range(bloc[0][0], bloc[0][1]):
                if allocation[i] == l:
                    start_l = i
                    break
            
            allocated_slots = 0
            
            slot_i = start_l
            at_least_one = False
            while slot_i <= bloc[0][1]:
                no_change = True
                for jj in range(num_clients):
                    give_priority = -1
                    client_c = order[jj]
                    if client_c + 1 == l:
                        continue
                    #print(f'!!! {client_c+1} {slot_i} {arriv[client_c]}')
                    if arriv[client_c] <= slot_i and (client_c+1 in bloc[1]):
                        #print(f'checking {client_c+1}')
                        is_allocated = False
                        ii = bloc[0][0]
                        while ii <= slot_i:
                            if allocation[ii] == client_c+1:
                                is_allocated = True
                            ii += 1
                        
                        if is_allocated:
                            #print('ignore')
                            continue
                        
                        #print('this is the one')
                        give_priority = client_c+1

                    #print(give_priority)
                    if give_priority != -1:
                        #print('here2')
                        for j in range(proc[give_priority-1]): # shift client
                            allocation[slot_i+j] = give_priority
                        slot_i = slot_i +  proc[give_priority-1]
                        no_change = False
                        at_least_one = True
                        break
                    
                if no_change:
                    if allocated_slots == 0:
                        start_l = slot_i
                    allocated_slots += 1
                    allocation[slot_i] = l
                    slot_i += 1
                    if allocated_slots == proc[int(l)-1]:
                        break
                        

            #if cannot reschedule:
            if not at_least_one:
                print('block cannot be updated any more')
                if len(sub_blocks) != 0:
                    block_i = (block_i + 1)%len(sub_blocks)
                continue
            
            print('New scheduling')
            print_res(allocation, duration, num_clients)
            
            # step 5 - update subset
            slot_i = bloc[0][0]
            
            new_block_f = True
            print(bloc[0][1])
            while slot_i < bloc[0][1]:
                #print(f'In {slot_i}')
                if new_block_f:
                    #print('new block')
                    new_block = [[slot_i,-1],[]]
                    new_block_f = False
                
                if allocation[slot_i] == l:
                    #print('skip')
                    if len(new_block[1]) > 0:
                        new_block[0][1] = slot_i
                        #print('added block')
                        sub_blocks.append(new_block)
                    
                    new_block_f = True
                    slot_i += 1
                    continue
                    #while allocation[slot_i] == l:
                    #slot_i += 1

                elif allocation[slot_i] not in new_block[1]:
                    #print('added client')
                    new_block[1].append(allocation[slot_i])
                
                new_block[0][1] = slot_i
                if slot_i == bloc[0][1]-1:
                    #print('added block')
                    sub_blocks.append(new_block)
                slot_i += 1
            

            print(f'Remaining blocks: {len(sub_blocks)}')
            if len(sub_blocks) != 0:
                block_i = (block_i + 1)%len(sub_blocks)

            

   
    # show result
    makespan = 0
    finish_clients = [0 for i in range(num_clients)]
    print('FINAL scheduling')
    print_res(allocation, duration, num_clients)

    for i in range(duration):
        if allocation[i] != 0:
            finish_clients[int(allocation[i])-1] = i
    
    for i in range(num_clients):
        finish_clients[i] += ll[i] + 1

    print(f'THE MAKESPAN IS {max(finish_clients)}')
if __name__ == '__main__':
    main()