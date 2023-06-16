import gurobi_solver
import gurobi_approach1a
import gurobi_approach1b
import gurobi_approach3
import gurobi_approach3_reverse
import gurobi_approach1_freeze
import gurobi_approach2_a
import argparse
import gurobi_admm_cont_int

import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase', type=str, default='fully_symmetric', help='fully_symmetric or fully_heterogeneous')
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--data_owners', type=int, default=50, help='the number of data owners')
    parser.add_argument('--compute_nodes', type=int, default=2, help='the number of compute nodes')
    parser.add_argument('--approach', type=str, default='approach3', help='select one of the approaches: approach1a/approach1aFreeze/approach2/approach3')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    f = open(args.log, "w")
    f.write(f"Experiment for {args.approach} for {args.data_owners} on {args.testcase} \n")
    f.close()

    gurobi_solver.K = args.data_owners
    gurobi_solver.H = args.compute_nodes
    w_start = 472
    w_start = gurobi_solver.run(args.log, args.testcase)
    #
    violations = []
    if args.approach == 'approach1a':
        gurobi_approach1a.K = args.data_owners
        gurobi_approach1a.H = args.compute_nodes
        ws, violations_1, violations_2, violations_3, max_c, accepted = gurobi_approach1a.run(args.log, args.testcase)
        violations = violations_3
    elif args.approach == 'approach1aFreeze':
        gurobi_approach1_freeze.K = args.data_owners
        gurobi_approach1_freeze.H = args.compute_nodes
        ws, violations_1, violations_2, violations_3, max_c, accepted = gurobi_approach1_freeze.run(args.log, args.testcase)
        violations = violations_3
    elif args.approach == 'approach2':
        gurobi_approach2_a.K = args.data_owners
        gurobi_approach2_a.H = args.compute_nodes
        ws, violations_1, violations_2, max_c, accepted = gurobi_approach2_a.run(args.log, args.testcase)
    elif args.approach == 'approach3':
        gurobi_approach3.K = args.data_owners
        gurobi_approach3.H = args.compute_nodes
        ws, violations_1, violations_2, max_c, accepted = gurobi_approach3.run(args.log, args.testcase)
    elif args.approach == 'continium':
        gurobi_admm_cont_int.K = args.data_owners
        gurobi_admm_cont_int.H = args.compute_nodes
        violations_1, violations_2, max_c = gurobi_admm_cont_int.run(args.log, args.testcase)
        ws = []
        accepted = []

    utils.plot_approach(w_start, ws, violations_1, violations_2, max_c, accepted, violations)





