import random

ref = "fully_heterogeneous"
K = 100
H = 5
# fully_symmetric
# fully_heterogeneous
# symmetric_machines
# symmetric_data_owners

fwd_compute_node = [9, 16]
back_compute_node = [10, 20]

release_fwd = [4, 7]
release_back = [4, 8]

fwd_last = [1, 3]
bac_last = [2, 4]

trans_act = [3, 6]
trans_grad = [3, 5]

data_owners = []
for i in range(K):
    data_owners.append((random.randint(0,1), random.randint(0,1))) # 0 slow, 1 fast

compute_nodes = []
for i in range(H):
    compute_nodes.append(random.randint(0,1)) # 0 slow, 1 fast


# Compute Nodes

print("process fwd")

for i in range(H):
    print(fwd_compute_node[compute_nodes[i]])


print("process back")
for i in range(H):
    print(back_compute_node[compute_nodes[i]])



#Data owners


print("release date fwd")
for i in range(K):
    for j in range(H):
        print(release_fwd[data_owners[i][1]], end="\t")
    print("")

print("release date back")
for i in range(K):
    for j in range(H):
        print(release_back[data_owners[i][1]], end="\t")
    print("")


print("fwd last layers")
for i in range(K):
    print(fwd_last[data_owners[i][0]])

print("back last layers")
for i in range(K):
    print(bac_last[data_owners[i][0]])

print("trans back")
for i in range(K):
    for j in range(H):    
        print(trans_act[data_owners[i][1]], end="\t")
    print("")

print("trans gradientd back")
for i in range(K):
    for j in range(H):    
        print(trans_grad[data_owners[i][1]], end="\t")
    print("")