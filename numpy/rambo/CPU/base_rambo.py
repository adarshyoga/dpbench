# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sys
import numpy as np
from dpbench_python.rambo.rambo_python import rambo_python
from dpbench_datagen.rambo import gen_rand_data

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
    
###############################################

def gen_data_np(nevts, nout):
    return gen_rand_data(nevts, nout)
     
##############################################

# create input data, call rambo computation function (alg)
def run(name, alg, sizes=5, step=2, nevts=2**20):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", required=False, default=sizes, help="Number of steps"
    )
    parser.add_argument(
        "--step", required=False, default=step, help="Factor for each step"
    )
    parser.add_argument(
        "--size", required=False, default=nevts, help="Initial data size"
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

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nevts = int(args.size)
    repeat = int(args.repeat)
    nout = 4

    if args.test:
        C1, F1, Q1, output = gen_data_np(nevts, nout)
        rambo_python(
            nevts, nout, C1, F1, Q1, output
        )

        C1_np, F1_np, Q1_np, output_np = gen_data_np(nevts, nout)
        # pass numpy generated data to kernel
        alg(nevts, nout, C1_np, F1_np, Q1_np, output_np)

        if np.allclose(output_np, output):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f1 = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        # generate input data
        C1_np, F1_np, Q1_np, output_np = gen_data_np(nevts, nout)

        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nevts), end=" ", flush=True)
        sys.stdout.flush()

        # call algorithm
        alg(nevts, nout, C1_np, F1_np, Q1_np, output_np)

        t0 = now()
        for _ in iterations:
            alg(nevts, nout, C1_np, F1_np, Q1_np, output_np)

        mops, time = get_mops(t0, now(), nevts)

        # record performance data - mops, time
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nevts, mops * 2 * repeat, time
            ),
            flush=True,
        )
        f1.write(str(nevts) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nevts) + "," + str(time) + "\n")
        nevts *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    f1.close()
    f2.close()
