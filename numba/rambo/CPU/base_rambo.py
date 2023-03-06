# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sys
import dpnp as np
import numpy
from dpbench_python.rambo.rambo_python import rambo_python
from dpbench_datagen.rambo import gen_rand_data
import dpctl

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

def to_dpnp(ref_array):
    if ref_array.flags["C_CONTIGUOUS"]:
        order = "C"
    elif ref_array.flags["F_CONTIGUOUS"]:
        order = "F"
    else:
        order = "K"
    return np.asarray(
        ref_array,
        dtype=ref_array.dtype,
        order=order,
        like=None,
        device="cpu",
        usm_type=None,
        sycl_queue=None,
    )

def to_numpy(ref_array):
    return np.asnumpy(ref_array)


def gen_data_dpnp(nevts, nout):
    C1, F1, Q1, output = gen_rand_data(nevts, nout)

    #convert to dpnp
    return (to_dpnp(C1), to_dpnp(F1), to_dpnp(Q1), to_dpnp(output))
     
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

    dpctl.SyclDevice("cpu")

    if args.test:
        C1, F1, Q1, output = gen_data_np(nevts, nout)
        rambo_python(
            nevts, nout, C1, F1, Q1, output
        )

        C1_dpnp, F1_dpnp, Q1_dpnp, output_dpnp = gen_data_dpnp(nevts, nout)
        # pass numpy generated data to kernel
        alg(nevts, nout, C1_dpnp, F1_dpnp, Q1_dpnp, output_dpnp)

        if numpy.allclose(to_numpy(output_dpnp), output):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f1 = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        # generate input data
        C1_dpnp, F1_dpnp, Q1_dpnp, output_dpnp = gen_data_dpnp(nevts, nout)

        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nevts), end=" ", flush=True)
        sys.stdout.flush()

        # call algorithm
        alg(nevts, nout, C1_dpnp, F1_dpnp, Q1_dpnp, output_dpnp)

        t0 = now()
        for _ in iterations:
            alg(nevts, nout, C1_dpnp, F1_dpnp, Q1_dpnp, output_dpnp)

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
