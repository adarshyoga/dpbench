# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numba_dpex as nbdx
import numpy as np
import base_l2_distance

@nbdx.kernel
def l2_norm_kernel(a, d):
    i = nbdx.get_global_id(0)
    O = a.shape[1]
    d[i] = 0.0
    for k in range(O):
        d[i] += a[i, k] * a[i, k]
    d[i] = np.sqrt(d[i])


def l2_norm(a, d):
    l2_norm_kernel[a.shape[0],](a, d)

base_l2_distance.run("l2 distance kernel", l2_norm)
