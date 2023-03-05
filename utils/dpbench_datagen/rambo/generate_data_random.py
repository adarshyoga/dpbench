import numpy as np

# constants used for input data generation

# write input data to a file in binary format
def __dump_binary__(X, Y):
    with open("X.bin", "w") as fd:
        X.tofile(fd)

    with open("Y.bin", "w") as fd:
        Y.tofile(fd)


# write input data to a file in text format
def __dump_text__(X, Y):
    with open("X.txt", "w") as fd:
        X.tofile(fd, "\n", "%s")

    with open("Y.txt", "w") as fd:
        Y.tofile(fd, "\n", "%s")


def gen_rand_data(nevts, nout):
    C1 = np.empty((nevts, nout))
    F1 = np.empty((nevts, nout))
    Q1 = np.empty((nevts, nout))

    np.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = np.random.rand()
            F1[i, j] = np.random.rand()
            Q1[i, j] = np.random.rand() * np.random.rand()

    return (
        C1,
        F1,
        Q1,
        np.empty((nevts, nout, 4)),
    )

# call numpy.random.uniform to generate input data and write the input as binary to a file
def gen_data_to_file(nevts, nout, dtype=np.float64):
    X, Y = gen_rand_data(nevts, nout)
    __dump_binary__(X, Y)
    # __dump_text__(X, Y) #for verification purpose only
