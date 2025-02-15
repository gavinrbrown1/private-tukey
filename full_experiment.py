# Code accompanying "Tukey Depth Mechanisms for Practical Private Mean Estimation"
# Gavin Brown and Lydia Zakynthinou

# This code is for research purposes only:
#   do no use it to protect sensitive information

# main script to run different examples.

import numpy as np
from time import time
import math
import sys

from utils import produce_directions
from mechanisms import restricted_exponential_mechanism, box_exponential_mechanism


# set parameters
eps, delta = 2, 1e-6
n = 500
d = 2

t = n // 4  # for restricted exponential mechanism
R = 10      # for box exponential mechanism
k = 30      # number of random directions, for approximate depth

np.random.seed(123)

print('Four basic experiments')
X = np.random.normal(size=(n,d)) # mean zero
print(f'n={n} samples, d={d} dimensions')
print(f'({eps},{delta})-DP for REM; ({eps},0)-DP for BoxEM; no formal privacy for approx volume\n')

mu_hat = np.mean(X, axis=0)
print(f'Empirical mean error: {np.linalg.norm(mu_hat):0.3f}\n')

print('Running REM, exact depth')
mu, PTR_flag = restricted_exponential_mechanism(X, n//4, eps, delta,
                                                    depth_method='exact',
                                                    num_directions=None,
                                                    volume_method='exact',
                                                    sampling_method='exact')
if PTR_flag == 'pass':
    print('PTR passed')
    print(f'Error: {np.linalg.norm(mu):0.3f}\n')
else:
    print('PTR failed\n')

print('Running BoxEM, exact depth')
mu = box_exponential_mechanism(X, eps, delta, R,
                                    depth_method='exact', 
                                    num_directions=None,
                                    volume_method='exact',
                                    sampling_method='exact')
print(f'Error: {np.linalg.norm(mu):0.3f}\n')

print('Running BoxEM, approx depth')
mu = box_exponential_mechanism(X, eps, delta, R,
                                    depth_method='approx', 
                                    num_directions=k,
                                    volume_method='exact',
                                    sampling_method='exact',
                                    halfspace_orientation='random')
print(f'Error: {np.linalg.norm(mu):0.3f}\n')

print('Running BoxEM, approx depth, approx volume')
mu = box_exponential_mechanism(X, eps, delta, R,
                                    depth_method='approx',
                                    num_directions=k,
                                    volume_method='MCMC',
                                    sampling_method='exact',
                                    halfspace_orientation='random')
print(f'Error: {np.linalg.norm(mu):0.3f}\n')

print('One bonus experiment in d=3, which uses VINCI volume software')
print('d=3, BoxEM, approx depth')
X = np.random.normal(size=(n,3)) # mean zero
mu = box_exponential_mechanism(X, eps, delta, R,
                                    depth_method='approx', 
                                    num_directions=k,
                                    volume_method='exact',
                                    sampling_method='exact',
                                    halfspace_orientation='random')
print(f'Empirical Error: {np.linalg.norm(np.mean(X,axis=0)):0.3f}')
print(f'Error: {np.linalg.norm(mu):0.3f}\n')



