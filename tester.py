import gurobi_solver
import gurobi_approach1a
import gurobi_approach1b
import argparse

import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase', type=str, default='fully_symmetric', help='fully_symmetric or fully_heterogeneous')
    parser.add_argument('--log', type=str, default='test1.txt', help='filename for the logging')
    parser.add_argument('--data_owners', type=int, default=50, help='50 or 100')
    parser.add_argument('--approach', type=str, default='approach1a', help='select one of the approaches')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    f = open(args.log, "w")
    f.write(f"Experiment for {args.approach} for {args.data_owners} on {args.testcase} \n")
    f.close()

    gurobi_solver.K = args.data_owners
    w_start = gurobi_solver.run(args.log, args.testcase)

    if args.approach == 'approach1a':
        gurobi_approach1a.K = args.data_owners
        ws_, violations_ = gurobi_approach1a.run(args.log, args.testcase)
    
    utils.plot_approach(w_start, ws_, violations_)





