r"""
Numbers (:mod:`skbio.core.numbers`)
===================================

.. currentmodule:: skbio.core.numbers

Numbers holds a sequence of numbers, and defines several statistical
operations (mean, stdev, etc.) FrequencyDistribution holds a mapping from
items (not necessarily numbers) to counts, and defines operations such as
Shannon entropy and frequency normalization.


Classes
-------

.. autosummary::
   :toctree: generated/

   Numbers

"""
# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

from random import shuffle
import math

import numpy

import cogent.maths.stats.test as stats


def hommola_cospeciation(host_dist, par_dist, matrix, permutations):
    """Performs the cospeciation test from Hommola et al recursively over a tree.

    Takes numpy matrices of jxj host distances, ixi 'parasite' (OTU) distances, 
    and a binary ixj association matrix. 

    test data from Hommola et al MB&E 2009: 
    hdist = numpy.array([[0,3,8,8,9],[3,0,7,7,8],[8,7,0,6,7],[8,7,6,0,3],[9,8,7,3,0]])
    pdist = numpy.array([[0,5,8,8,8],[5,0,7,7,7],[8,7,0,4,4],[8,7,4,0,2],[8,7,4,2,0]])
    int = numpy.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,1]])

    This is basically a direct translation from the R code, and not optimized
    in any way for Python.

    NOTE: the method return signature is now changed.
    For backwards compatibility purposes - 
    when this method is called, 'result' has changed to 'result[0]'
    """


    m = matrix.sum()

    hosts = [0] * m
    pars = [0] * m

    # Generate lists of host and symbiont edges, such that the index
    # of the lists represents an edge connecting the host to the parasite.
    s = 0
    while s < m:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 1:
                    hosts[s] = j
                    pars[s] = i
                    s += 1

    # get a vector of pairwise distances for each interaction edge
    x = get_dist(hosts, host_dist, range(matrix.shape[1]))
    y = get_dist(pars, par_dist, range(matrix.shape[0]))

    # calculate the observed correlation coefficient for this host/symbionts
    r = stats.correlation(x, y)[0]

    # now do permutaitons. Initialize index lists of the appropriate size.
    mp = range(par_dist.shape[1])
    mh = range(host_dist.shape[1])
    below = 0

    perm_stats = []  # initialize list of shuffled correlation vals

    for i in range(permutations):
        # Generate a shuffled list of indexes for each permutation. This effectively
        # randomizes which host is associated with which symbiont, but maintains
        # the distribution of genetic distances.
        shuffle(mp)
        shuffle(mh)

        # Get pairwise distances in shuffled order
        y_p = get_dist(pars, par_dist, mp)
        x_p = get_dist(hosts, host_dist, mh)

        # calculate shuffled correlation.
        # If greater than observed value, iterate counter below.
        r_p = stats.correlation(x_p, y_p)[0]
        perm_stats.append(r_p)
        if r_p >= r:
            below += 1

    # print "Below: " + str(below)
    # print "Pemutations: " + str(permutations)

    p_val = float(below + 1) / float(permutations + 1)

    return p_val, r, perm_stats


def get_dist(labels, dists, index):
    """Function for picking a subset of pairwise distances from a distance matrix
    according to a set of (randomizable) indices. Derived from Hommola et al R code"""

    m = len(labels)
    vec = []

    for i in range(m - 1):
        k = index[labels[i]]
        for j in range(i + 1, m):
            t = index[labels[j]]
            vec.append(dists[k, t])

    return vec

