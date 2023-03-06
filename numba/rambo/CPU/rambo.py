import base_rambo
import dpnp as np
from numba_dpex import dpjit


@dpjit
def rambo(nevts, nout, C1, F1, Q1, output):
    C = 2.0 * C1 - 1.0
    S = np.sqrt(1 - np.square(C))
    F = 2.0 * np.pi * F1
    Q = -np.log(Q1)

    output[:, :, 0] = Q
    output[:, :, 1] = Q * S * np.sin(F)
    output[:, :, 2] = Q * S * np.cos(F)
    output[:, :, 3] = Q * C

base_rambo.run("Rambo Numba", rambo)