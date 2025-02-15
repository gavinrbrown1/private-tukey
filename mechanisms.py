# Code accompanying "Tukey Depth Mechanisms for Practical Private Mean Estimation"
# Gavin Brown and Lydia Zakynthinou

# This code is for research purposes only:
#   do no use it to protect sensitive information

# main code for private algorithms

import numpy as np
import math

from utils import produce_directions, approx_distance, PTR_test  
from volume_computation import all_volumes
from sampling import sample_from_level_set, level_selection

def restricted_exponential_mechanism(X, t, eps, delta, depth_method='exact', num_directions=None,
                                     volume_method='exact', sampling_method='exact', halfspace_orientation=None):
    """
    Restricted exponential mechanism on Tukey depth (BGSUZ)
    Input
        X: dataset shape (n,d)
        t: cutoff threshold for sampling
        eps, delta: privacy parameters
        depth_method: use 'exact' Tukey depth or 'approx' (ie, random)
        num_directions: if 'approx' depth, how many random directions to consider
        volume_method: {'exact', 'MCMC'}
        halfspace_orientation: None if using exact depth,
                                'random' for random directions, 
                                'axis' for axis-aligned
    Output
        mu: mean estimate or None
        PTR_flag: {'pass','fail'}
    """
    n, d = X.shape

    # rescaling of privacy parameters so that overall we satisfy (eps,delta)-DP
    eps = eps/2
    delta = delta/(math.exp(2*eps))

    if depth_method == 'approx':
        H = produce_directions(num_directions, d, halfspace_orientation)
    elif depth_method == 'exact':
        H = None

    if volume_method != 'exact':        # set PAC guarantees, these need to be adjusted
        eta, beta = 0.1, delta/2
    else:
        eta, beta = None, None

    # compute volumes
    volume_list = all_volumes(X, depth_method, H, volume_method, R=None, eta=eta, beta=beta)

    # approximate distance-to-unsafety and privacy check
    dist = approx_distance(volume_list, t, eps, delta)
    PTR_flag = PTR_test(dist, eps, delta)
    if PTR_flag == 'fail':
        print('PTR failed')
        return None, 'fail'

    # select level to sample from
    target_depth = level_selection(volume_list, eps, t)
    # the following code breaks privacy, but can be useful for debugging 
    #print('Sampling mean from depth', target_depth)

    # select point
    mu = sample_from_level_set(X, target_depth, 1, 
                               H, depth_method, sampling_method)

    return mu, 'pass' 

def box_exponential_mechanism(X, eps, delta, R,
                              depth_method='exact', num_directions=None,
                              volume_method='exact', sampling_method='exact',
                              halfspace_orientation=None,
                              verbose=False):
    """
    Exponential mechanism on Tukey depth over [-R,+R]^d (Kaplan et al, LKO)
    Input
        X: dataset shape (n,d)
        eps, delta: privacy parameter
        R: box size parameter
        depth_method: use 'exact' Tukey depth or 'approx' (ie, random)
        num_directions: if 'approx' depth, how many random directions to consider
        volume_method: {'exact', 'rejection', 'MCMC'}
        halfspace_orientation: None if using exact depth,
                                'random' for random directions, 
                                'axis' for axis-aligned
    Output
        mean estimate
    """
    n, d = X.shape

    if depth_method == 'approx':
        H = produce_directions(num_directions, d, halfspace_orientation)
    elif depth_method == 'exact':
        H = None

    # compute all volumes of Tukey level sets
    if volume_method == 'exact':
        volume_list = all_volumes(X, depth_method, H, volume_method, R=R, verbose=verbose)
    else:
        volume_list = all_volumes(X, depth_method, H, volume_method, R=R, eta=0.1, beta=delta)
            
    # select level to sample from. 
    target_depth = level_selection(volume_list, eps, t=0)
    if verbose:
        # this breaks privacy but is useful for debugging
        print('Sampling mean from depth', target_depth)

    # select point
    if target_depth == 0:
        mu = np.random.uniform(low=-R, high=R, size=d) 
    else:
        mu = sample_from_level_set(X, target_depth, 1, 
                                H, depth_method, sampling_method)

    return mu

def standard_gaussian_mechanism(X, eps, delta, R):
    """
    Naive approach of projecting to d-dimensional ball B(0,R) and adding Gaussian noise (KV, KLSU, with ball not box)
        Input
            X: dataset shape (n,d)
            eps, delta: privacy parameters
            R: radius of ball
        Output
            mean estimate
    """
    n, d = X.shape

    # Shrink each x to fit in B(0,R)
    X_magnitudes = np.linalg.norm(X, axis=1)
    outside_ball = (X_magnitudes > R)
    X_normalized = (X.T / X_magnitudes).T
    X[outside_ball] = (X_normalized[outside_ball] * R)  # rescale each coordinate of x by R/norm(x) only if norm(x)>R

    # Add spherical Gaussian noise
    sd = 2*R*((2*np.log(1.25/delta))**0.5)/(eps*n)
    noise = np.random.normal(0, sd, size=d)
    mu = np.sum(X, axis=0)/n + noise

    return mu

