# Workflow Optimization for Parallel Split Learning

This code is for the paper: _Tirana, J., Tsigkari D., Iosifidis G., Chatzopoulos, D. Workflow Optimization for Parallel Split Learning, in proc. of IEEE INFOCOM 2024._

### Set-up and requirements steps

1. Python > 3.9
2. Install cvxpy, run: `pip install cvxpy`
3. Install GUROBI for cvxpy, run:  `pip install gurobipy`
4. Get access to GUROBI license. There is a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).
5. run `pip install -r requirements.txt`
    
### Main code
The following scripts implement the main algorithms of the paper (i.e, Algorithm 1 and 2), and are used to run the the testing scripts:

- utils.py: reads the input file with the profiled data and creates the input tensors.
- ILP_solver.py: solves the problem using only the solver.
- heuristic_FCFS.py: solves the problem using the balanced_greedy algorthm described in Section IV. In this algorithm the client assigment is implemented using a greedy approach, whereas the scheduling takes place using the FCFS policy.
- random_benchmark.py: this the benchmark approach that is used to compare the proposed approach. The client assigment problem is implemented using a random function, whereas the scheduling takes place using the FCFS policy.
- forwardprop_admm.py: Implements Algorithm 1 and solves Pf problem
- backprop_only.py: Solves the Pb problem using the algorithm 2.
- ADMM_solution.py: Implements the ADMM-base solution described in figure 3. To do so calls the forwardprop_admm.py and backprop_only.py


### Testing files
The following scripts can be used to reproduced the experiments of the paper.

**observation_1.py:** 
        This script can be used to reproduce tha Table II from the paper. 
        This script measures the optimality gap between the ADMM-based solution and the optimal solution. Also, it compares the computing time of the two approaches.
        

| Parameter of observation_1                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `log` | Filename for the logging. The scripts writes intermediate resutls into a file. |
| `clients`, `K`| The number of clients. |
| `helpers`, `H`| The number of helpers. |
| `model`, `m` | The model architecture. Options: `resnet101`, `vgg19`. |
| `dataset`, `d` | Dataset to use. Options: `mnist`, `cifar10`. |
| `scenario`, `s` | . |

**observation_2.py:**
**observation_3.py:**
**observation_4.py:**

**Citation:**
If you find this repository useful, please cite our paper:

```
@inproceedings{tirana2024workflow,
      title={Workflow Optimization for Parallel Split Learning},
      author={Tirana, J., Tsigkari D., Iosifidis G., Chatzopoulos, D.},
      booktitle={proc. of IEEE INFOCOM},
      year={2024}
}
```