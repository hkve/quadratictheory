import numpy as np


def one_body_density_oo(rho, t1, t2, l1, l2, o, v):
    N = o.stop

    rho[o, o] += 2*np.eye(N, dtype=rho.dtype)
    rho[o, o] -= np.einsum("aj,ai->ij", l1, t1)
    rho[o, o] -= 2*np.einsum("abkj,baik->ij", l2, t2)

    return rho


def one_body_density_vv(rho, t1, t2, l1, l2, o, v):
    rho[v, v] += np.einsum("ai,bi->ab", l1, t1)
    rho[v, v] += 2*np.einsum("acij,bcij->ab", l2, t2)

    return rho


def one_body_density_ov(rho, t1, t2, l1, l2, o, v):
    rho[o, v] += 2*t1.T
    rho[o, v] -= 2*np.einsum("aj,bcjk,bcik->ia", t1, l2, t2, optimize=True)
    rho[o, v] -= 2*np.einsum("bi,bcjk,acjk->ia", t1, l2, t2, optimize=True)
    rho[o, v] += 2*np.einsum("bj,abij->ia", l1, t2)
    rho[o, v] -= np.einsum("bj,abji->ia", l1, t2)
    rho[o, v] -= np.einsum("bj,aj,bi->ia", l1, t1, t1)

    return rho


def one_body_density_vo(rho, t1, t2, l1, l2, o, v):
    rho[v, o] += l1

    return rho


def one_body_density_restricted(rho, t1, t2, l1, l2, o, v):
    rho = one_body_density_oo(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_vv(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_ov(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_vo(rho, t1, t2, l1, l2, o, v)

    return rho


def two_body_density_restricted(rho, t1, t2, l1, l2, o, v):
    pass