# SPDX-FileCopyrightText: 2023 Stefan Behnel, Robert Bradshaw,
#   Dag Sverre Seljebotn, Greg Ewing, William Stein, Gabriel Gellner, et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

import numpy as np


# pythran export compute(int64[:,:], int64[:,:], int64, int64, int64)
def compute(array_1, array_2, a, b, c):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c
