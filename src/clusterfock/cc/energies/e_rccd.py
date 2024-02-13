import numpy as np


def td_energy_addition_restricted(t2, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    e = 0

    e -= 2 * np.einsum("abij,acik,kbjc->", l2, t2, u[o, v, o, v], optimize=True)

    e -= 2 * np.einsum("abij,acki,kbcj->", l2, t2, u[o, v, v, o], optimize=True)

    e -= 2 * np.einsum("abji,acki,kbjc->", l2, t2, u[o, v, o, v], optimize=True)

    e += 4 * np.einsum("abij,acik,kbcj->", l2, t2, u[o, v, v, o], optimize=True)

    e -= 2 * np.einsum("ki,abij,bajk->", f[o, o], l2, t2, optimize=True)

    e += np.einsum("abij,cdij,abcd->", l2, t2, u[v, v, v, v], optimize=True)

    e += 2 * np.einsum("ac,abij,bcji->", f[v, v], l2, t2, optimize=True)

    e += np.einsum("abij,abkl,klij->", l2, t2, u[o, o, o, o], optimize=True)

    e += np.einsum("abij,abij->", l2, u[v, v, o, o], optimize=True)

    e += np.einsum("abij,abkl,cdij,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e += np.einsum("abij,acki,bdlj,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e += np.einsum("abij,adlj,bcki,kldc->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e -= 4 * np.einsum("abij,abik,cdjl,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e -= 4 * np.einsum("abij,acij,bdkl,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e -= 4 * np.einsum("abij,acik,bdlj,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e -= 2 * np.einsum("abij,acik,bdjl,kldc->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e += 2 * np.einsum("abij,abik,cdjl,kldc->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e += 2 * np.einsum("abij,acij,bdkl,kldc->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e += 2 * np.einsum("abij,acik,bdlj,kldc->", l2, t2, t2, u[o, o, v, v], optimize=True)

    e += 4 * np.einsum("abij,acik,bdjl,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True)

    return e
