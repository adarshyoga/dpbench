# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpnp as np
import numba as nb
from numba_dpex import dpjit
import base_l2_distance

@dpjit
def l2_norm(a, d):
    for i in nb.prange(a.shape[0]):
        for k in range(a.shape[1]):
            d[i] += np.square(a[i, k])
        d[i] = np.sqrt(d[i])

base_l2_distance.run("l2 distance kernel", l2_norm)
