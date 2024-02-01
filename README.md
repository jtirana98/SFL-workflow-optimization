# Workflow Optimization for Parallel Split Learning

This code is for the paper: _Tirana, J., Tsigkari D., Iosifidis G., Chatzopoulos, D. Workflow Optimization for Parallel Split Learning, in proc. of IEEE INFOCOM 2024._

### Set-up and requirements steps

1. Python > 3.9
2. Install cvxpy, run: `pip install cvxpy`
3. Install GUROBI for cvxpy, run:  `pip install gurobipy`
4. Get access to GUROBI license. There is a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).
5. run `pip install -r requirements.txt`
    
### Main code
The following scripts, which can be found inside the folder util_files, implement the main algorithms of the paper (i.e, Algorithm 1 and 2), and are used to run the testing scripts:

- utils.py: reads the input file with the profiled data and creates the input tensors.
- ILP_solver.py: solves the problem using only the solver.
- heuristic_FCFS.py: solves the problem using the balanced_greedy algorthm described in Section IV. In this algorithm, the client assignment is implemented using a greedy approach, whereas the scheduling takes place using the FCFS policy.
- random_benchmark.py: this is the benchmark approach that is used to compare the proposed approach. The client assignment problem is implemented using a random function, whereas the scheduling takes place using the FCFS policy.
- ADMM_solution.py: Implements the ADMM-base solution described in Figure 3. Implements Algorithm 1 and solves the $P_f$ problem. Also, solves the $P_b$ problem using the Algorithm 2 (see comment in util_files/README).

The scripts above can be used as function calls for other scripts, just import the respective file. However, a scenario and input parameters should be defined.
In case you want to build your own scenarios you can follow the flow of the scripts described below (i.e., the testing scripts).  

### Testing files
The following scripts, which can be found inside the folder test_files, can be used to reproduce the experiments of the paper.
In a nutshell, they build scenarios and call the respective algorithms.
In more detail:

**observation_1.py:** 
        This script can be used to reproduce experiments shown in Table II in the paper. 
        This script measures the optimality gap between the ADMM-based solution and the optimal solution. Also, it compares the computing time of the two approaches.
        

| Parameter of observation_1                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `log` | Filename for the logging. The script writes intermediate results into a file. |
| `clients`, `K`| The number of clients. |
| `helpers`, `H`| The number of helpers. |
| `model`, `m` | The model architecture. Options: `resnet101`, `vgg19`. |
| `dataset`, `d` | Dataset to use. Options: `mnist`, `cifar10`. |
| `scenario`, `s` | Scenario 1 for low heterogeneity and 2 for high. |

**observation_2.py:**
      This script can be used to reproduce the experiments shown in Figure 6 in the paper. 
      This script compares the resulting makespan using the ADMM approach for different slot lengths.
      By default the K = 50 and H = [5, 10], and slot duration = [50, 150, 200]

| Parameter of observation_2                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `log` | Filename for the logging. The script writes intermediate results into a file. |
| `model`, `m` | The model architecture. Options: `resnet101`, `vgg19`. |
| `dataset`, `d` | Dataset to use. Options: `mnist`, `cifar10`. |


**observation_3.py:**
      This script can be used to reproduce the experiments shown in Figure 7 in the paper. 
      This script compares the output makespan and computing time amongst the ADMM-approach the balanced-greedy and the benchmark approach.
        

| Parameter of observation_3                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `log` | Filename for the logging. The script writes intermediate results into a file. |
| `clients`, `K`| The number of clients. |
| `helpers`, `H`| The number of helpers. |
| `model`, `m` | The model architecture. Options: `resnet101`, `vgg19`. |
| `dataset`, `d` | Dataset to use. Options: `mnist`, `cifar10`. |
| `scenario`, `s` | Scenario 1 for low heterogeneity and 2 for high. |


**observation_4.py:**
This script can be used to reproduce the experiments shown in Figure 8 in the paper. 
This script compares makespan when adding more helpers to the system.
By default the K = 100 and H = [1, 2, 5, 10, 20, 25].

| Parameter of observation_4                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `log` | Filename for the logging. The script writes intermediate results into a file. |
| `model`, `m` | The model architecture. Options: `resnet101`, `vgg19`. |
| `dataset`, `d` | Dataset to use. Options: `mnist`, `cifar10`. |


**Citation:**
If you find this repository useful, please cite our paper:

```
@inproceedings{tirana2024workflow,
      title={Workflow Optimization for Parallel Split Learning},
      author={Tirana, Joana and Tsigkari, Dimitra and Iosifidis, George and Chatzopoulos, Dimitris},
      booktitle={proc. of IEEE INFOCOM},
      year={2024}
}
```
