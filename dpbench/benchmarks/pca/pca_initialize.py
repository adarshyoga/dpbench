# Copyright 2022 Intel Corp.
# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


def initialize(npoints, dims):
    from sklearn.datasets import make_regression

    return make_regression(n_samples=npoints, n_features=dims, random_state=0)
