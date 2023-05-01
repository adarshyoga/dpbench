# SPDX-FileCopyrightText: 2017 Nicolas P. Rougier
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# more information at https://github.com/rougier/numpy-book

import dpnp as np


def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
    Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xi.reshape(xn * yn)
    Yi.reshape(xn * yn)
    C.reshape(xn * yn)

    Z = np.zeros(C.shape, dtype=np.complex128)
    for i in range(itermax):
        if not len(Z):
            break

        # Compute for relevant points only
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)

        # Failed convergence
        I = abs(Z) > horizon  # noqa: E741 math variable
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]

        # Keep going with those who have not diverged yet
        np.logical_not(I, I)  # np.negative(I, I) not working any longer
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    return Z_.T, N_.T
