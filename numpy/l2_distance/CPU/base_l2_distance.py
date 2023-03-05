# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sys
import numpy as np
from dpbench_python.l2_distance.l2_distance_python import l2_distance_python
from dpbench_datagen.l2_distance import gen_data

try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0), t1 - t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

RISK_FREE = 0.1
VOLATILITY = 0.2
    
###############################################

def gen_data_np(nopt, dims):
    return gen_data(nopt, dims)
     
##############################################

# create input data, call l2_distance computation function (alg)
def run(name, alg, sizes=10, step=2, nopt=2**20):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", required=False, default=sizes, help="Number of steps"
    )
    parser.add_argument(
        "--step", required=False, default=step, help="Factor for each step"
    )
    parser.add_argument(
        "--size", required=False, default=nopt, help="Initial data size"
    )
    parser.add_argument(
        "--repeat",
        required=False,
        default=1,
        help="Iterations inside measured region",
    )
    parser.add_argument(
        "--text", required=False, default="", help="Print with each result"
    )
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )
    parser.add_argument("-d", type=int, default=3, help="Dimensions")

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)
    dims = int(args.d)

    if args.test:
        X, Y = gen_data_np(nopt, dims)
        l2_distance_python(
            X, Y
        )

        X_np, Y_np = gen_data_np(nopt, dims)
        # pass numpy generated data to kernel
        alg(X_np, Y_np)

        if np.allclose(Y_np, Y):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f1 = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        # generate input data
        X, Y = gen_data_np(nopt, dims)

        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nopt), end=" ", flush=True)
        sys.stdout.flush()

        # call algorithm
        alg(X, Y)  # warmup

        t0 = now()
        for _ in iterations:
            alg(X, Y)

        mops, time = get_mops(t0, now(), nopt)

        # record performance data - mops, time
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nopt, mops * 2 * repeat, time
            ),
            flush=True,
        )
        f1.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    f1.close()
    f2.close()
