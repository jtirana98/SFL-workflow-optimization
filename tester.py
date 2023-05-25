import gurobi_solver
import gurobi_approach1a
import gurobi_approach1b
import gurobi_approach3
import gurobi_approach3_reverse
import gurobi_approach2_a
import argparse

import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase', type=str, default='fully_symmetric', help='fully_symmetric or fully_heterogeneous')
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--data_owners', type=int, default=50, help='50 or 100')
    parser.add_argument('--approach', type=str, default='approach3', help='select one of the approaches')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    f = open(args.log, "w")
    f.write(f"Experiment for {args.approach} for {args.data_owners} on {args.testcase} \n")
    f.close()

    gurobi_solver.K = args.data_owners
    #w_start = gurobi_solver.run(args.log, args.testcase)
    w_start = 648
    
    if args.approach == 'approach1a':
        gurobi_approach1a.K = args.data_owners
        ws, violations_1, violations_2, max_c, accepted = gurobi_approach1a.run(args.log, args.testcase)
    elif args.approach == 'approach2':
        gurobi_approach2_a.K = args.data_owners
        ws, violations_1, violations_2, max_c, accepted = gurobi_approach2_a.run(args.log, args.testcase)
    elif args.approach == 'approach3':
        gurobi_approach3.K = args.data_owners
        ws, violations_1, violations_2, max_c, accepted = gurobi_approach3.run(args.log, args.testcase)




    utils.plot_approach(w_start, ws, violations_1, violations_2, max_c, accepted)





