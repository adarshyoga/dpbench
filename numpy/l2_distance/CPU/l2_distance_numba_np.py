# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import numba as nb
import base_l2_distance

@nb.njit(parallel=True, fastmath=True)
def l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1)
    d[:] = np.sqrt(sum)

base_l2_distance.run("l2 distance kernel", l2_norm)
