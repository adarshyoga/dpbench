# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpnp as np
from numba_dpex import dpjit
import base_l2_distance

@dpjit
def l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1)
    d[:] = np.sqrt(sum)

base_l2_distance.run("l2 distance kernel", l2_norm)
