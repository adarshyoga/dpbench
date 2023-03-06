# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base_pair_wise
import dpnp as np
import numba as nb
from numba_dpex import dpjit


# Naieve pairwise distance impl - take an array representing M points in N dimensions, and return the M x M matrix of Euclidean distances
@dpjit
def pw_distance(X1, X2, D):
    # Size of imputs
    M = X1.shape[0]
    N = X2.shape[0]
    O = X1.shape[1]

    # Outermost parallel loop over the matrix X1
    for i in nb.prange(M):
        # Loop over the matrix X2
        for j in range(N):
            d = 0.0
            # Compute exclidean distance
            for k in range(O):
                tmp = X1[i, k] - X2[j, k]
                d += tmp * tmp
            # Write computed distance to distance matrix
            D[i, j] = np.sqrt(d)


base_pair_wise.run("Numba par_for", pw_distance)
