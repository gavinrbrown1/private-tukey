# Code accompanying "Tukey Depth Mechanisms for Practical Private Mean Estimation"
# Gavin Brown and Lydia Zakynthinou

# This code is for research purposes only:
#   do not use it to protect sensitive information

# Run a single mechanism.
# This code does not require installing anything beyond Python packages.
# (Some other functions require R and/or VINCI.)

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

print('One basic experiment, using only Python')
X = np.random.normal(size=(n,d)) # mean zero
print(f'n={n} samples, d={d} dimensions')
print(f'({eps},{delta})-DP for REM\n')

mu_hat = np.mean(X, axis=0)
print(f'Empirical mean error: {np.linalg.norm(mu_hat):0.3f}\n')

print('Running REM, approx depth')
mu, PTR_flag = restricted_exponential_mechanism(X, t, eps, delta, 
                                    depth_method='approx', 
                                    num_directions=k,
                                    volume_method='exact',
                                    sampling_method='exact',
                                    halfspace_orientation='random')
if PTR_flag == 'pass':
    print('PTR passed')
    print(f'Error: {np.linalg.norm(mu):0.3f}\n')
else:
    print('PTR failed\n')

