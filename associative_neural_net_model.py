#-------------------------------------------
# Author: Sneha Reddy Aenugu
# 
# Description: Hopfield network simulation
# of paired-associative recall task
#-------------------------------------------

import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import norm
import pickle
import random
from random import shuffle
import itertools
import copy
import math
import pdb 
import argparse
from multiprocessing import Pool
import time
from analyze_data import get_data_stats
import argparse


class HopNet:
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((2*N, 2*N))

    def store_weights(self, A, B, mu1, mu2, pf, pb, nodes, mut=1, mode='train'):

        N = self.N
        self.W1 = self.W[nodes[:N][:,None], nodes[:N]]
        self.W2 = self.W[nodes[:N][:,None], nodes[N:]]
        self.W3 = self.W[nodes[N:][:,None], nodes[:N]]
        self.W4 = self.W[nodes[N:][:,None], nodes[N:]]


        A_associative = (np.matrix(A).T)*np.matrix(A)
        store = np.random.rand(N,N) < mu1
        if mode=='test':
            store = mut*store
        self.W1 += np.multiply(A_associative, store)

        B_associative = (np.matrix(B).T)*np.matrix(B)
        store = np.random.rand(N,N) < mu2
        if mode=='test':
            store = mut*store
        self.W3 += np.multiply(B_associative, store)

        BA_associative = (np.matrix(B).T)*np.matrix(A)
        store = np.random.rand(N,N) < pf
        if mode=='test':
            store = mut*store
        self.W2 += np.multiply(BA_associative, store)

        AB_associative = (np.matrix(A).T)*np.matrix(B)
        store = np.random.rand(N,N) < pb
        if mode=='test':
            store = mut*store
        self.W4 += np.multiply(AB_associative, store)

        self.W[nodes[:N][:,None], nodes[:N]] = self.W1
        self.W[nodes[:N][:,None], nodes[N:]] = self.W2
        self.W[nodes[N:][:,None], nodes[:N]] = self.W3
        self.W[nodes[N:][:,None], nodes[N:]] = self.W4


    def recall(self, cue, target, nodes, reverse=True):
        N = self.N
        self.W1 = self.W[nodes[:N][:,None], nodes[:N]]
        self.W2 = self.W[nodes[:N][:,None], nodes[N:]]
        self.W3 = self.W[nodes[N:][:,None], nodes[:N]]
        self.W4 = self.W[nodes[N:][:,None], nodes[N:]]
        state = 2*np.random.randint(2, size=self.N)-1
        if reverse:
            hetero_inpt = self.W4*(np.matrix(cue).T)
        else:
            hetero_inpt = self.W2*(np.matrix(cue).T)
        for k in range(800):
            i = np.random.randint(self.N)
            if reverse:
                auto_inpt = np.dot(self.W1[i], state)
            else:
                auto_inpt = np.dot(self.W3[i], state)
            state[i] = np.sign(auto_inpt + hetero_inpt[i])
            cos_dist = np.dot(state, target)/(norm(state)*norm(target))
            if cos_dist >= 0.99:
                return 1,k
        return 0,-1


def get_pdf(mu, sigma, rho):
    means = [mu, mu]
    cov_matrix = np.array([[sigma*sigma, rho*sigma*sigma],[rho*sigma*sigma, sigma*sigma]])
    pdf = multivariate_normal(means, cov_matrix)
    return pdf


def run_simuation(pdfs, mus, sigmas, experiment):
    hopnet = HopNet(70)
    nodes = np.array(list(range(140)))
    nodes_store = np.zeros((12,140)).astype(int)

    A = 2*np.random.randint(2,size=(12,70))-1
    B = 2*np.random.randint(2,size=(12,70))-1

    
    ids = list(range(12))
    shuffle(ids)
    
    ### Store weights
    for j in ids:
        np.random.shuffle(nodes)
        nodes_store[j] = nodes
        while(1):
            pr_f, pr_b = pdfs[int(j/4)].rvs(1)
            if pr_f > 0 and pr_f < 1 and pr_b >0 and pr_b < 1:
                break
        hopnet.store_weights(A[j], B[j], mus[int(j/4)], mus[int(j/4)], pr_f, pr_b, nodes)

    
    results1 = np.zeros(12)
    reactions1 = np.zeros(12)
    shuffle(ids)
    correct_updated = []
    correct_notupdated = []
    
    ### First test
    for j in ids:
        if j%4 == 0:
            results1[j], reactions1[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        elif j%4 == 1:
            results1[j], reactions1[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        elif j%4 == 2:
            results1[j], reactions1[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)
        elif j%4 == 3:
            results1[j], reactions1[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)


        if results1[j] and experiment != 'no_test':
            if experiment == 'asymm_test':
                while(1):
                    pr_f, pr_b = pdfs[-1].rvs(1)
                    if pr_f > 0 and pr_f < 1 and pr_b >0 and pr_b < 1:
                        break
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], pr_f, pr_b, nodes_store[j])
            elif experiment == 'stochastic_learning':
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], mus[-1], mus[-1], nodes_store[j])
            elif experiment == 'all_weights_learning':
                mu_test = 1
                hopnet.store_weights(A[j], B[j], mu_test, mu_test, mu_test, mu_test, nodes_store[j], mut=mus[-1], mode='test')
               
            

    results2 = np.zeros(12)
    reactions2 = np.zeros(12)
    shuffle(ids)

    correct_updated_correct = []
    correct_notupdated_correct = []

    ### Second test
    for j in ids:
        if j%4 == 0:
            results2[j], reactions2[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        if j%4 == 1:
            results2[j], reactions2[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)
        if j%4 == 2:
            results2[j], reactions2[j] = hopnet.recall(A[j],B[j], nodes_store[j], reverse=False)
        if j%4 == 3:
            results2[j], reactions2[j] = hopnet.recall(B[j],A[j], nodes_store[j], reverse=True)


        if results2[j] and experiment != 'no_test':
            if j in correct_updated:
                correct_updated_correct.append(j)
            elif j in correct_notupdated:
                correct_notupdated_correct.append(j)
            if experiment=='asymm_test':
                while(1):
                    pr_f, pr_b = pdfs[-1].rvs(1)
                    if pr_f > 0 and pr_f < 1 and pr_b >0 and pr_b < 1:
                        break
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], pr_f, pr_b, nodes_store[j])
            elif experiment == 'stochastic_learning':
                hopnet.store_weights(A[j], B[j], mus[-1], mus[-1], mus[-1], mus[-1], nodes_store[j])
            elif experiment == 'all_weights_learning':
                mu_test = 1
                hopnet.store_weights(A[j], B[j], mu_test, mu_test, mu_test, mu_test, nodes_store[j], mut=mus[-1], mode='test')

    correct_first = 0
    correct_first_i = 0
    correct_first_r = 0
    incorrect_first_i = 0
    incorrect_first_r = 0
    incorrect_second_correct_first_i = 0
    incorrect_second_correct_first_r = 0
    correct_second_incorrect_first_i = 0
    correct_second_incorrect_first_r = 0

    identical_recall_diff = []
    reverse_recall_diff = []

    for i in range(12):
        result1 = results1[i]
        result2 = results2[i]

        if result1 == 1:
            correct_first += 1
            if i% 4 == 0 or i%4 == 3:
                correct_first_i += 1
            else:
                correct_first_r += 1
        if result1 == 0:
            if i% 4 == 0 or i%4 == 3:
                incorrect_first_i += 1
            else:
                incorrect_first_r += 1
        if result1 == 1 and result2 == 0:
            if i% 4 == 0 or i%4 == 3:
                incorrect_second_correct_first_i += 1
            else:
                incorrect_second_correct_first_r += 1
        if result1 == 0 and result2 == 1:
            if i% 4 == 0 or i%4 == 3:
                correct_second_incorrect_first_i += 1
            else:
                correct_second_incorrect_first_r += 1

        if result1==1 and result2 == 1:
            if i%4 == 0 or i%4 == 3:
                identical_recall_diff.append(reactions1[i]-reactions2[i])
            else:
                reverse_recall_diff.append(reactions1[i]-reactions2[i])


    stats = [correct_first, incorrect_second_correct_first_i, correct_second_incorrect_first_i, incorrect_second_correct_first_r, correct_second_incorrect_first_r, correct_first_i, correct_first_r, incorrect_first_i, incorrect_first_r]
    stats = np.array(stats).astype(float)

    return stats, [identical_recall_diff, reverse_recall_diff], [correct_updated, correct_updated_correct, correct_notupdated, correct_notupdated_correct]


def get_g_squared_value(O, E):

    if len(np.where(E==0)[0]) > 0:
        return 1000

    O[np.where(O==0)] = 0.001

    g_squared = 0
    if O[0] == 0:
        g_squared +=  2*72*(1-O[0])*math.log((1-O[0])/(1-E[0]))
    else:
        g_squared += 2*72*(O[0])*math.log(O[0]/E[0]) + 2*72*(1-O[0])*math.log((1-O[0])/(1-E[0]))

    if O[1] == 0:
        g_squared += 2*36*O[-4]*O[1]*math.log((O[-4]*O[1])/(E[-4]*E[1])) 
    else:
        g_squared += 2*36*O[-4]*O[1]*math.log((O[-4]*O[1])/(E[-4]*E[1])) + 2*36*O[-4]*(1-O[1])*math.log((O[-4]*(1-O[1]))/(E[-4]*(1-E[1])))

    if O[2] == 0:
        g_squared += 2*36*O[-2]*O[2]*math.log((O[-2]*O[2])/(E[-2]*E[2]))
    else:
        g_squared += 2*36*O[-2]*O[2]*math.log((O[-2]*O[2])/(E[-2]*E[2])) + 2*36*O[-2]*(1-O[2])*math.log((O[-2]*(1-O[2]))/(E[-2]*(1-E[2])))

    if O[3] == 0:
        g_squared += 2*36*O[-3]*O[3]*math.log((O[-3]*O[3])/(E[-3]*E[3]))
    else:
        g_squared += 2*36*O[-3]*O[3]*math.log((O[-3]*O[3])/(E[-3]*E[3])) + 2*36*O[-3]*(1-O[3])*math.log((O[-3]*(1-O[3]))/(E[-3]*(1-E[3])))

    if O[4] == 0:
        g_squared += 2*36*O[-1]*O[4]*math.log((O[-1]*O[4])/(E[-1]*E[4]))
    else:
        g_squared += 2*36*O[-1]*O[4]*math.log((O[-1]*O[4])/(E[-1]*E[4])) + 2*36*O[-1]*(1-O[4])*math.log((O[-1]*(1-O[4]))/(E[-1]*(1-E[4])))


    return g_squared

def run_optimizer_on_subject(sub, exp=None):
    #def run(mu, sigma, rho, mu_t=None, sigma_t=None):
    def run(args, mode='optimize'):
        mu = args[0]
        sigma=args[1]
        rho = args[2]
        if mu <= 0 or mu >=1 or sigma <=0 or sigma >= 1 or rho <=0 or rho >= 1:
            return 10000
        if len(args) > 3:
            mu_t = args[3]
            if mu_t <= 0 or mu_t >=1:
                return 10000
        else:
            mu_t = None
        if len(args) > 4:
            sigma_t = args[4]
        else:
            sigma_t = None

        if not mu_t and not sigma_t:
            experiment = 'no_test'
        elif mu_t and not sigma_t:
            experiment = 'stochastic_learning'
        elif mu_t and sigma_t:
            experiment = 'asymm_test'
        if exp:
            experiment = exp
        print(experiment)

        # Using the same mu and sigma for all repetitions of 1,3 and 5 
        if experiment=='no_test': 
            mus = [mu, mu, mu]
            sigmas = [sigma, sigma, sigma]
        elif experiment == 'stochastic_learning' or experiment == 'all_weights_learning':
            mus = [mu, mu, mu, mu_t]
            sigmas = [sigma, sigma, sigma]
        elif experiment == 'asymm_test':
            mus = [mu, mu, mu, mu_t]
            sigmas = [sigma, sigma, sigma, sigma_t]

        nsim = 30

        data_stats, data_reactions = get_data_stats()

        stats_array = np.zeros(9)
        reactions_array_identical = []
        reactions_array_reverse = []
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0

        for k in range(nsim):
            if k%10 == 0:
                print(k)
            if experiment == 'stochastic_learning' or experiment == 'all_weights_learning':
                pdfs = [get_pdf(mus[i], sigmas[i], rho) for i in range(len(mus)-1)]
            else:
                pdfs = [get_pdf(mus[i], sigmas[i], rho) for i in range(len(mus))]

            stats, reaction_stats, updated = run_simuation(pdfs, mus, sigmas, experiment)
            count1 += len(updated[0])
            count2 += len(updated[1])
            count3 += len(updated[2])
            count4 += len(updated[3])
            stats_array += stats
            reactions_array_identical.extend(reaction_stats[0])
            reactions_array_reverse.extend(reaction_stats[1])

        stats_array[1] = stats_array[1]/stats_array[-4]
        stats_array[2] = stats_array[2]/stats_array[-2]
        stats_array[3] = stats_array[3]/stats_array[-3]
        stats_array[4] = stats_array[4]/stats_array[-1]
        stats_array[0] = stats_array[0]/(12*nsim)
        stats_array[5] = stats_array[5]/(6*nsim)
        stats_array[6] = stats_array[6]/(6*nsim)
        stats_array[7] = stats_array[7]/(6*nsim)
        stats_array[8] = stats_array[8]/(6*nsim)


        gsquared = get_g_squared_value(data_stats[int(sub)], stats_array)
        
        if mode == 'optimize':
            return gsquared
        else:
            return gsquared, [np.mean(reactions_array_identical), np.mean(reactions_array_reverse)], stats_array 
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu', dest='mu', type=float,  default=0.624)
    parser.add_argument('--sigma', dest='sigma', type=float,  default=0.277)
    parser.add_argument('--rho', dest='rho', default=0.523)
    parser.add_argument('--mut', dest='mut', type=float, default=0.3)
    parser.add_argument('--sigmat', dest='sigmat', type=float,  default=None)
    parser.add_argument('--experiment', dest='experiment', type=str,  default='all_weights_learning')
    parser.add_argument('--num', dest='num', type=str, default=1)

    args = parser.parse_args()
    mu = float(args.mu)
    sigma = float(args.sigma)
    rho = float(args.rho)
    mut = args.mut
    sigmat = args.sigmat
    num = args.num
    experiment = args.experiment
    print(mu, sigma, rho, mut, num)

    run = run_optimizer_on_subject(num, exp=experiment)
    gsq, time, stats = run([mu, sigma, rho, mut, sigmat])

    f = open(experiment + str(num) + '_' + str(mu) + '_' + str(sigma) + '_' + str(rho) + '_' + str(mut) + '_5.0.pkl', 'wb')
    pickle.dump([gsq, time, stats], f)

        
    



