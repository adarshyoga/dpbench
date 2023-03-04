# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sys
import numpy as np
from dpbench_python.blackscholes.bs_python import black_scholes_python
from dpbench_datagen.blackscholes import gen_rand_data

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

def gen_data_np(nopt):
    price, strike, t = gen_rand_data(nopt)
    call = np.zeros(nopt, dtype=np.float64)
    put = np.ones(nopt, dtype=np.float64)
    return (
        price,
        strike,
        t,
        call,
        put,
    )

##############################################

# create input data, call blackscholes computation function (alg)
def run(name, alg, sizes=14, step=2, nopt=2**19):
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

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)

    if args.test:
        price, strike, t, p_call, p_put = gen_data_np(nopt)
        black_scholes_python(
            nopt, price, strike, t, RISK_FREE, VOLATILITY, p_call, p_put
        )

        price_1, strike_1, t_1, n_call, n_put = gen_data_np(nopt)
        # pass numpy generated data to kernel
        alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, n_call, n_put)

        if np.allclose(n_call, p_call) and np.allclose(n_put, p_put):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f1 = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        # generate input data
        price, strike, t, call, put = gen_data_np(nopt)

        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nopt), end=" ", flush=True)
        sys.stdout.flush()

        # call algorithm
        alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put)  # warmup

        t0 = now()
        for _ in iterations:
            alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put)

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
