# Code accompanying "Tukey Depth Mechanisms for Practical Private Mean Estimation"
# Gavin Brown and Lydia Zakynthinou

# This code is for research purposes only:
#   do not use it to protect sensitive information

# functions for sampling: from level sets and for PTR

import numpy as np
import warnings
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

from utils import polytope_bounding_box, make_polytope_representation
from utils import check_polytope_inclusion, is_in_convex_hull

def sample_from_level_set(X, depth, num_samples,
                          H, depth_method, sampling_method):
    """
    Produces (possible approximate) sample from target Tukey level set.
    Input
        X: dataset, (n,d)
        depth: Tukey depth to sample from
        num_samples: number of positive samples to produce
        H: directions, (num_directions,d) or None
        depth_method: {'exact', 'approx'}
        sampling_method: {'exact', 'MCMC'}, where 'exact' implies rejection sampling
    Output
        sample: the sample (d,)
    """
    if sampling_method == 'MCMC':
        return sample_from_level_set_MCMC(X, depth, num_samples, H, depth_method)
    elif sampling_method == 'exact':
        return sample_from_level_set_exact(X, depth, num_samples, H, depth_method)

def sample_from_level_set_with_MCMC(X, depth, num_samples, H, depth_method):
    """Call the Volesti R package"""
    A, b = make_polytope_representation(X, H, depth, depth_method='approx')
    rpy2.robjects.numpy2ri.activate()  # automatic conversion to R objects
    volesti = importr('volesti')
    b_num = rpy2.robjects.FloatVector(b)
    P = volesti.Hpolytope(A=A, b=b_num)
    num_samples = 1
    samples = volesti.sample_points(P, n=num_samples)
    return samples[:, 0]

def sample_from_level_set_exact(X, depth, num_samples, H, depth_method):
    """Use rejection sampling"""
    if depth_method == 'exact':
        polytope = make_polytope_representation(X, None, depth, depth_method='exact')
        representation = 'vertex'

    elif depth_method == 'approx':
        polytope = make_polytope_representation(X, H, depth, depth_method='approx')
        representation = 'halfspace'

    r_lo, r_hi = polytope_bounding_box(polytope, representation)
    # this is not good, what if we don't get any acceptances?
    # start with few proposals, then go to more
    proposal_schedule = [10, 100, 1000] + 10 * [10000]
    for num_proposals in proposal_schedule:
        accepted = polytope_rejection_sampling(polytope, representation, 'box', (r_lo, r_hi), num_proposals)
        try:
            return accepted[0, :]
        except IndexError:
            pass
    print('did not sample anything')
    return None

def level_selection(volume_list, eps, t):
    """
    Which level set to sample from. 
    Uses racing sampling as in Amin et al. See Medina and Gillenwater for reference.
    Note that these volumes are of the convex set (ie all points geq depth).
    Input
        volume_list: all n volumes in numpy array
            uses "true" indexing, so zeroth entry is volume of zero level set, which may be infinity
        eps: privacy param
        t: threshold cutoff for sampling
    Output:
        target_level: integer (greater than or equal to t)
    """
    # first, throw away volumes at the top that are zero
    volume_list = volume_list[volume_list != 0]

    # then, anything that is below t. We'll add t back at the end.
    volume_list = volume_list[t:]

    # now do sampling
    m = len(volume_list)
    unifs = np.random.rand(m)
    
    # check for runtime warnings 
    warnings.simplefilter("error", RuntimeWarning)

    outputs = np.log(np.log(1/unifs)) - np.log(volume_list) - (eps/2)*(np.arange(m)+t) - np.log(1 - np.exp(-eps/2))

    target_index = np.argmin(outputs)
    return target_index + t

def polytope_rejection_sampling(polytope, representation, region_type, proposal_region, num_proposals,
                                return_counts=False):
    """
    Do the rejection sampling from a ball or box
    Input
        polytope: either (A,b) or vertices V
        representation: {'halfspace', 'vertex'}
        region_type: is the proposal region a 'ball' or 'box'
        proposal_region: tuple (r_lo, r_hi)
        num_proposals: how many to make
        return_counts: whether to return only the number accepted
    Output:
        array of all accepted points
    """
    # if number of proposals too big, batch them
    threshold = int(1e7)
    if (num_proposals > threshold) and (return_counts is False):
        print('polytope rejection sampling, too many samples to return points')
        return None

    if num_proposals < threshold:
        proposals = make_proposals(num_proposals, region_type, proposal_region)
        accepted_mask = check_polytope_inclusion(polytope, proposals, representation)
        if return_counts:
            return np.sum(accepted_mask)
        return proposals[accepted_mask==1,:]
    remaining_proposals = num_proposals
    accepted_count = 0
    while remaining_proposals > threshold:
        remaining_proposals -= threshold
        proposals = make_proposals(threshold, region_type, proposal_region)
        accepted_mask = check_polytope_inclusion(polytope, proposals, representation)
        accepted_count += np.sum(accepted_mask)
    proposals = make_proposals(remaining_proposals, region_type, proposal_region)
    accepted_mask = check_polytope_inclusion(polytope, proposals, representation)
    accepted_count += np.sum(accepted_mask)
    return accepted_count

def make_proposals(num_proposals, region_type, proposal_region):
    """Helper function, for profiling purposes."""
    if region_type == 'box':
        r_lo, r_hi = proposal_region
        d = len(r_lo)
        proposals = np.random.rand(num_proposals, d)
        proposals = (r_hi - r_lo) * proposals + r_lo  # proposals[i,j] is uniform in [r_lo[j], r_hi[j]]
    elif region_type == 'ball':
        center, radius = proposal_region
        d = len(center)
        # use the 'dropped coordinates' method
        # https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
        higher_dim_pts = np.random.normal(size=(num_proposals,d+2))
        higher_dim_pts = higher_dim_pts / np.reshape(np.linalg.norm(higher_dim_pts,axis=1), (num_proposals,1))
        proposals = radius*higher_dim_pts[:,0:d] + center
    return proposals

