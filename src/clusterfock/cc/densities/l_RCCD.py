import numpy as np

def one_body_density_oo(rho, t, l, o, v):
    N = o.stop

    rho[o, o] += 2*np.eye(N, dtype=rho.dtype)
    rho[o, o] -= 2*np.einsum("abkj,baik->ij", l, t)

    return rho


def one_body_density_vv(rho, t, l, o, v):
    rho[v, v] += 2*np.einsum("acij,bcij", l, t)

    return rho


def one_body_density_restricted(rho, t, l, o, v):
    rho = one_body_density_oo(rho, t, l, o, v)
    rho = one_body_density_vv(rho, t, l, o, v)

    return rho
