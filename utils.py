# Code accompanying "Tukey Depth Mechanisms for Practical Private Mean Estimation"
# Gavin Brown and Lydia Zakynthinou

# This code is for research purposes only:
#   do not use it to protect sensitive information

# collected utility functions

import numpy as np
from scipy.optimize import linprog
import math
import warnings

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

def produce_directions(num_directions, d, orientation='random'):
    """
    Generate random directions (normal vectors for halfspaces) to consider.
    Input
        num_directions: number of directions to generate
        d: dimension
        orientation: 'random' for random directions, 'axis' for axis-aligned (as in AJRV)
    Output
        V: shape (num_directions,d), or (d,d)
    """
    if orientation == 'random':
        V = np.random.normal(0, 1, (num_directions,d))
    elif orientation == 'axis':
        V = np.eye(d)
    else:
        print('orientation not supported')
        V = None
    return V

def approx_distance(volume_list, t, eps, delta):
    """
    Approx distance/score function from Amin et al.
    Input
        volume_list: all n volumes in numpy array
        t: threshold cutoff for sampling
        eps, delta: privacy params
    Output
        dist: approximate distance
    """
    # find the max depth
    max_depth = 0
    for depth, volume in enumerate(volume_list):
        if volume > 0:
            max_depth = depth

    # compute the distance function 
    for k in range(t-1,-1,-1):
        for g in range(1,max_depth+1):
            lo_depth = t - k - 1
            hi_depth = t + k + g + 1
            if (1 <= lo_depth) and (hi_depth <= max_depth):
                volume_ratio = volume_list[lo_depth] / volume_list[hi_depth]
                if volume_ratio * math.exp(-eps * g / 2) <= delta:
                    return k
    return -1 

def PTR_test(dist, eps, delta):
    """
    private check for distance being sufficiently large
    Input
        dist: integer
        eps, delta: privacy parameters
    Output
        flag: 'pass' or 'fail'
    """
    noise = np.random.laplace(loc=0, scale=1/eps)
    threshold = math.log(1/(2*delta)) / eps
    if dist + noise >= threshold:
        return 'pass'
    return 'fail'

def make_polytope_representation(X, H, depth, depth_method):
    """
    Return polytope Ax <= b corresponding to Tukey region.
    each direction yields an interval between the upper and lower quantiles.
    Inclusion test is np.all(A@z <= b)
    Input
        X: (n,d) dataset
        H: (k,d) halfspaces/directions to consider
        depth: integer, Tukey region
        depth_method: approx or exact
    Output:
        (A, b) or V, vertices
    """
    n, d = X.shape
    
    if depth_method == 'approx':
        k = H.shape[0]
        q_lo, q_hi = (depth-1)/n, 1-(depth-1)/n  # upper and lower quantiles
            # for instance, when depth=1, want min and max of this projection
        b_lo, b_hi = np.zeros(k), np.zeros(k)

        # vectorize this if bottleneck
        for i in range(k):
            projections = X @ H[i,:]
            b_lo[i], b_hi[i] = np.quantile(projections, [q_lo, q_hi])

        # stack to make single (A,b) pair
        A = np.vstack((H, -H))
        b = np.hstack((b_hi, -b_lo))

        return A, b
    elif depth_method == 'exact':
        rpy2.robjects.numpy2ri.activate()  # automatic conversion to R objects
        TukeyRegion = importr('TukeyRegion')
        Tr = TukeyRegion.TukeyRegion(X, depth, retVertices=True)
                                    #retFacets=False, retVolume=False,
                                    #retBarycenter=False, retHalfspaces=False)
        V = Tr.rx2('vertices')
        return V

def polytope_bounding_box(polytope, representation):
    """
    Find an axis-aligned rectangle to use for rejection sampling.
    The smallest (up to some tolerance) rectangle that contains the polytope.
    Input:
        polytope: either tuple (A,b) as halfspaces or V, list of vertices
        representation: {'halfspace', 'vertex'}
    Output
        r_hi: (d,), upper bounds for each axis
        r_lo: (d,), lower bounds
    """
    if representation == 'vertex':
        r_lo = np.min(polytope, axis=0)
        r_hi = np.max(polytope, axis=0)
    elif representation == 'halfspace':
        # do linear programming thing
        A, b = polytope[0], polytope[1]
        d = A.shape[1]
        r_hi, r_lo = np.zeros(d), np.zeros(d)

        # solve two linear programs for each dimension, one up and one down.
        for j in range(d):
            # find the smallest value in the box, we minimize c@x
            c = np.zeros(d)
            c[j] = 1
            res = linprog(c, A, b, bounds=(None,None), method='highs-ds')
            x = res.x
            r_lo[j] = x[j]

            # now, largest value
            c = np.zeros(d)
            c[j] = -1
            res = linprog(c, A, b, bounds=(None,None), method='highs-ds')
            x = res.x
            r_hi[j] = x[j]

    return r_lo, r_hi

def check_polytope_inclusion(polytope, proposals, representation):
    """Helper function, for profiling purposes."""
    if representation == 'halfspace':
        A, b = polytope[0], polytope[1]
        accepted = np.all(proposals@A.T <= b, axis=1)
    elif representation == 'vertex':
        V = polytope
        num_proposals = proposals.shape[0]
        accepted = np.zeros(num_proposals)
        for i in range(num_proposals):
            z = proposals[i,:]
            if is_in_convex_hull(z,V):
                accepted[i] = 1
    return accepted
        
def is_in_convex_hull(x, V):
    """
    Checks whether x is in the convex hull of vertices V.
    Input
        x: dimension d
        V: dimension (num_points,d)
    Output
        True or False
    """
    num_points = V.shape[0]
    c = np.zeros(num_points)
    A_eq = np.vstack([np.ones(num_points), V.T])
    b_eq = np.hstack([1, x])
    bounds = np.array([(0, 1) for _ in range(num_points)])
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={"disp": False})
    return result.success

def chebyshev_center(A, b):
    """
    Finds the largest ball contained in P = {x : Ax <= b}.
    This code assumes the polytope is not empty.
    Input
        A, b: polytope
    Output
        ball_center: array in d dimensions
        ball_radius: positive float
    Alternatively, could use Volesti.inner_ball(Hpolytope P)
    """
    m, d = A.shape
    norms = np.linalg.norm(A, axis=1, keepdims=True)

    A_ub = np.hstack((A,norms))
    c = np.zeros(d+1)
    c[-1] = -1      # because we want to maximize r
    bounds = np.full((d+1, 2), None)
    bounds[-1,0] = 0 # r must be nonnegative
    
    res = linprog(c=c, A_ub=A_ub, b_ub=b, bounds=bounds, method='highs-ds')
    return res.x[:d], res.x[-1]

def bounding_sphere(points):
    """
    Uses Ritter's algorithm
    https://gist.github.com/chriselion/2168f0d9aa217d858620bac58403bbeb
    """
    m = points.shape[0]     # number of points

    # pick random starting point
    start_i = np.random.randint(m)
    x = points[start_i,:]    #any arbitrary point in the point cloud works
    #y = max(points,key= lambda p: dist(p,x) )    #choose point y furthest away from x
    y = points[np.argmax(np.linalg.norm(x - points,axis=1)), :]
    #z = max(points,key= lambda p: dist(p,y) )    #choose point z furthest away from y
    z = points[np.argmax(np.linalg.norm(y - points,axis=1)), :]
    #center, radius = (((y[0]+z[0])/2,(y[1]+z[1])/2,(y[2]+z[2])/2), dist(y,z)/2)    #initial bounding sphere
    center = (y + z)/2
    radius = np.linalg.norm(y - z) / 2

    # Note this doesn't use the radius^2 optimization that Ritter uses
    for i in range(m):
        d = np.linalg.norm(center - points[i,:])
        if d < radius:
            continue
        radius = .5 * (radius + d)
        old_to_new = d - radius
        center = (center*radius + points[i,:]*old_to_new)/d
    return center, radius

def bounding_sphere_approx_best(X, num_iters=5):
    """calls Ritter's multiple times to find smallest radius"""
    d = X.shape[1]
    best_R = np.float('inf')
    best_center = np.zeros(d)

    for i in range(num_iters):
        center, R = bounding_sphere(X)
        if R < best_R:
            best_R = R
            best_center = center
    return best_center, best_R








