import numpy as np


def one_body_density_oo(rho, t, l, o, v):
    N = o.stop

    rho[o, o] += np.eye(N)
    rho[o, o] += np.einsum("abkj,abik->ij", l, t) / 2

    return rho


def one_body_density_vv(rho, t, l, o, v):
    rho[v, v] += np.einsum("acij,bcij", l, t) / 2

    return rho


def one_body_density(rho, t, l, o, v):
    rho = one_body_density_oo(rho, t, l, o, v)
    rho = one_body_density_vv(rho, t, l, o, v)

    return rho
