# Code accompanying "Tukey Depth Mechanisms for Practical Private Mean Estimation"
# Gavin Brown and Lydia Zakynthinou

# This code is for research purposes only:
#   do not use it to protect sensitive information

# Functions to compute volumes of polytopes and Tukey depth level sets

import math
import numpy as np
import subprocess
import re
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import polytope as pc
import sys

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

from utils import polytope_bounding_box, make_polytope_representation
from sampling import polytope_rejection_sampling

def all_volumes(X, depth_method, H, volume_method, R=None, eta=None, beta=None, verbose=False):
    """
    Wrapper for finding volume list.
    Let Vi* be the true volume of the ith level set. We return estimate Vi.
    For the approximate methods, wp 1-beta we have, for all i, (1-eta)Vi* <= Vi <= (1+eta)Vi*.
    Input
        X: dataset (n,d)
        depth_method: 'exact' or 'approx'
        H: either None (if 'exact') or (num_directions,d) directions to consider
        volume_method: {'exact','rejection','MCMC'}
        R: side of bounding box [-R,+R]^2, or None
        eta: volume approximation guarantee
        beta: high probability guarantee
    Output
        volume_list: array of floor(n/2) volumes. zeroth entry is volume of zero depth set, which is np.inf if R==None
    """
    n, d = X.shape
    volume_list = np.zeros(n // 2 + 1)

    # first, the zeroth depth
    if R is None:
        volume_list[0] = np.inf
    else:
        volume_list[0] = np.power(2 * R, d)

    # if exact depth and exact volumes,
    #   use TukeyKRegions, which finds all volumes at once
    if depth_method == 'exact' and volume_method == 'exact':
        volume_list[1:] = all_exact_volumes(X)[1:]

    else:
        # otherwise, remaining depths one by one
        for depth in range(1, n // 2 + 1):
            if depth_method == 'approx' and volume_method == 'exact':
                # compute polytope
                k, d = H.shape
                A, b = make_polytope_representation(X, H, depth, depth_method='approx')

                # get its volume
                volume_list[depth] = exact_polytope_volume(A,b)
            elif depth_method == 'approx' and volume_method == 'MCMC':
                volume_list[depth] = MCMC_volume(X, H, depth, eta)
            else:
                print('the volume method asked for is not implemented')

            # if zero depth, bail out
            # Volesti returns volume = -1 after a certain depth
            if volume_list[depth] <= 0:
                volume_list[depth] = 0      # for value = -1.0
                return volume_list

            # print for tracking progress, if verbose
            print_step = n // 10
            if (depth % print_step == 0) and verbose:
                print('volume of depth', depth, 'is', volume_list[depth])

    return volume_list

def MCMC_volume(X, H, depth, eta):
    """
    Use Volesti code to compute volume of a polytope.
    This is not private and included for demonstration purposes only:
        even though the algorithms themselves are practical, all 
        current analyses of MCMC-based volume computation algorithms
        have large constants that are incompatible with practical
        implementations that, like in our setting, require provable
        "probably approximately correct" guarantees.
    Input
        X: (n,d) dataset
        H: halfspaces to consider if depth=='approx'
        depth: level
    Output:
        volume: float
    """
    n, d = X.shape
    A, b = make_polytope_representation(X, H, depth, depth_method='approx')
    rpy2.robjects.numpy2ri.activate()  # automatic conversion to R objects
    volesti = importr('volesti')
    b_num = rpy2.robjects.FloatVector(b)
    P = volesti.Hpolytope(A=A, b=b_num)
    settings = ['CB', eta, 'CDHR', 2]
    volume = volesti.volume(P, settings=settings, rounding=False)
    # default settings: algorithm = 'CB' for CoolingBodies, error=0.1, random_walk = 'CDHR'
    # for Coordinate Directions Hit-and-Run, walk_length=1, win_length=400+3*d**2, rounding = FALSE
    return volume

def all_exact_volumes(X):
    """Wrapper for TukeyRegion R package."""
    n, d = X.shape
    hi_depth = math.floor((n - d + 1) / 2)  # highest possible depth
    volume_list = np.zeros(n // 2 + 1)

    rpy2.robjects.numpy2ri.activate()  # automatic conversion to R objects
    TukeyRegion = importr('TukeyRegion')
    Tr = TukeyRegion.TukeyKRegions(X, maxDepth=hi_depth,
                                   retFacets=False, retVolume=True,
                                   retBarycenter=False, retHalfspaces=False)

    for depth in range(1, hi_depth + 1):
        region = Tr.rx2(depth)
        if region.rx2('innerPointFound')[0]:
            volume_list[depth] = region.rx2('volume')[0]
    return volume_list

def exact_polytope_volume(A, b):
    """
    Use VINCI code to compute exact volume with Lasserre's algorithm.
    Function writes a text file (.ine) with polytope description, then
        calls vinci script with python's subprocess package.
    We read the results from the command line.

    In d=2, use Qhull.
    """
    m, d = A.shape
    if d == 2:
        # first, check that it's feasible
        d = A.shape[1]
        res = linprog(c=np.zeros(d), A_ub=A, b_ub=b, bounds=(None, None), method='highs-ds')
        if not res.success:
            return 0

        # if not empty, proceed
        # convert to polytope package's representation
        P = pc.Polytope(A, b)

        # convert to V-representation, getting vertices (slow in high dimensions)
        vertices = pc.extreme(P)

        # find volume with qhull (also slow in high dimensions)
        hull = ConvexHull(vertices, qhull_options='FS')
        volume = hull.volume
    else:
        sys.path.append('./vinci-1.0.5')

        # using VINCI code, and Lasserre's algorithm
        b = b.reshape(-1, 1)  # Reshape b to be a column vector
        data = np.hstack([b, -A])

        # Prepare the header and footer
        header = f"begin\n{m} {d + 1} real\n"  # assuming the number type is float
        footer = "end\n"

        # Define the file path
        file_path = "polytope.ine"

        # Write to the file
        with open(file_path, "w") as file:
            file.write("Your comments here\n")  # Write any initial comments
            file.write(header)
            for row in data:
                line = ' '.join(map(str, row.tolist())) + "\n"
                file.write(line)
            file.write(footer)
            file.write("Your options here\n")  # Write any options here

        volume = vinci_subprocess()
    return volume

def vinci_subprocess():
    """wrapper to call the vinci subprocess"""
    command = ['./vinci-1.0.5/vinci', "polytope", "-m", "rlass"]  

    # Running the command
    result = subprocess.run(command, text=True, capture_output=True)

    match = re.search(r"Volume:\s+(\d+\.\d+e[+-]\d+|\d+\.\d+)", result.stdout)

    # Convert the matched number to float if found
    if match:
        return float(match.group(1))
    else:
        print('vinci didn\'t find volume')
        return None

