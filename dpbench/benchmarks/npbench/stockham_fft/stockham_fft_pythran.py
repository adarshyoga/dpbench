# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


# pythran export mgrid(int, int)
def mgrid(xn, yn):
    Xi = np.empty((xn, yn), dtype=np.uint32)
    Yi = np.empty((xn, yn), dtype=np.uint32)
    for i in range(xn):
        Xi[i, :] = i
    for j in range(yn):
        Yi[:, j] = j
    return Xi, Yi


# pythran export stockham_fft(int, int, int, float64[:], float64[:])
def stockham_fft(N, R, K, x, y):
    # Generate DFT matrix for radix R.
    # Define transient variable for matrix.
    # i_coord, j_coord = np.mgrid[0:R, 0:R]
    i_coord, j_coord = mgrid(R, R)
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat = np.exp(-2.0j * np.pi * i_coord * j_coord / R)
    # Move input x to output y
    # to avoid overwriting the input.
    y[:] = x[:]

    # ii_coord, jj_coord = np.mgrid[0:R, 0:R**K]
    ii_coord, jj_coord = mgrid(R, R**K)

    # Main Stockham loop
    for i in range(K):
        # Stride permutation
        # yv = np.reshape(y, (R**i, R, R**(K-i-1)))
        yv = y.reshape(R**i, R, R ** (K - i - 1))
        tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        # Twiddle Factor multiplication
        D = np.empty((R, R**i, R ** (K - i - 1)), dtype=np.complex128)
        tmp = np.exp(
            -2.0j
            * np.pi
            * ii_coord[:, : R**i]
            * jj_coord[:, : R**i]
            / R ** (i + 1)
        )
        # D[:] = np.repeat(np.reshape(tmp, (R, R**i, 1)), R ** (K-i-1), axis=2)
        D[:] = np.repeat(tmp.reshape(R, R**i, 1), R ** (K - i - 1), axis=2)
        # tmp_twid = np.reshape(tmp_perm, (N, )) * np.reshape(D, (N, ))
        tmp_twid = tmp_perm.reshape(N) * D.reshape(N)
        # Product with Butterfly
        # y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R**(K-1))), (N, ))
        tmp2 = dft_mat @ tmp_twid.reshape(R, R ** (K - 1))
        y[:] = tmp2.reshape(N)
