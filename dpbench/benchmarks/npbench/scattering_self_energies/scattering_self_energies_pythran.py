# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


# pythran export scattering_self_energies(
#     int32[:,:], complex128[:,:,:,:,:], complex128[:,:,:,:,:],
#     complex128[:,:,:,:,:,:], complex128[:,:,:,:,:])
def scattering_self_energies(neigh_idx, dH, G, D, Sigma):
    for k in range(G.shape[0]):
        for E in range(G.shape[1]):
            for q in range(D.shape[0]):
                for w in range(D.shape[1]):
                    for i in range(D.shape[-2]):
                        for j in range(D.shape[-1]):
                            for a in range(neigh_idx.shape[0]):
                                for b in range(neigh_idx.shape[1]):
                                    if E - w >= 0:
                                        dHG = (
                                            G[k, E - w, neigh_idx[a, b]]
                                            @ dH[a, b, i]
                                        )
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] += dHG @ dHD
