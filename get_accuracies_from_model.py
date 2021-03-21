import sys
import numpy as np
import pickle
from associative_neural_net_model import run_optimizer_on_subject
import pdb

experiment = sys.argv[1]
sub = sys.argv[2]

f = open('pickles/' + experiment+'_'+ str(sub) + '.pkl', 'rb')
data = pickle.load(f)

mu = data[0].x[0]
sigma = data[0].x[1]
rho = data[0].x[2]

mut = None
sigmat = None
params = [mu, sigma, rho]
if experiment != 'no_test':
    mut = data[0].x[3]
    params.append(mut)

run = run_optimizer_on_subject(sub, experiment)
gsquared, time_data, accuracy_stats = run(params, mode='stats')
pickle.dump([gsquared, time_data, accuracy_stats, params], open('pickles_fts/model_' + experiment+'_' + str(sub) +'.pkl', 'wb'))  
