import optunity
import random
import pdb
from associative_neural_net_model import run_optimizer_on_subject
import pickle
import sys
import argparse
from scipy.optimize import minimize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', dest='experiment', type=str, default='no_test')
    parser.add_argument('--sub', dest='sub', type=int, default=0)
    args = parser.parse_args()
    sub = args.sub

    details_list = []


    experiment = args.experiment

    for i in range(1):
        run = run_optimizer_on_subject(sub, experiment)
        if experiment == 'no_test':
            res = minimize(run, [0.5, 0.25, 0.9], method='nelder-mead')
        elif experiment == 'all_weights_learning':
            res = minimize(run, [0.5, 0.25, 0.5, 0.1], method='nelder-mead')
        elif experiment == 'stochastic_learning':
            res = minimize(run, [0.5, 0.25, 0.5, 0.1], method='nelder-mead')

        details_list.append(res)

        f = open('pickles/' + experiment + '_' + str(sub) + '.pkl', 'wb')
        pickle.dump(details_list, f)
        f.close()
