# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba
import numba_mlir as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True, gpu_fp64_truncate="auto")
def rambo(nevts, nout, C1, F1, Q1, output):
    for i in numba.prange(nevts):
        for j in numba.prange(nout):
            C = 2.0 * C1[i, j] - 1.0
            S = np.sqrt(1 - np.square(C))
            F = 2.0 * np.pi * F1[i, j]
            Q = -np.log(Q1[i, j])

            output[i, j, 0] = Q
            output[i, j, 1] = Q * S * np.sin(F)
            output[i, j, 2] = Q * S * np.cos(F)
            output[i, j, 3] = Q * C
