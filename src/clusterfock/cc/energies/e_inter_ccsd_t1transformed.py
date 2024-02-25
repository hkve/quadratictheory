import numpy as np

def td_energy_addition(t2, l1, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)
    tau0 = zeros((N, N, M, M))

    tau0 += 2 * np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau0 -= np.einsum(
        "caki,jkbc->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau1 = zeros((N, N, M, M))

    tau1 -= 4 * np.einsum(
        "caki,jkbc->ijab", t2, tau0, optimize=True
    )

    tau0 = None

    tau1 += 2 * np.einsum(
        "baji->ijab", u[v, v, o, o], optimize=True
    )

    tau1 += np.einsum(
        "dcji,badc->ijab", t2, u[v, v, v, v], optimize=True
    )

    e = 0

    e += np.einsum(
        "abij,ijab->", l2, tau1, optimize=True
    ) / 8

    tau1 = None

    tau2 = zeros((N, N, N, N))

    tau2 += np.einsum(
        "baij,bakl->ijkl", l2, t2, optimize=True
    )

    tau3 = zeros((N, N, N, N))

    tau3 += 2 * np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau3 += np.einsum(
        "balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True
    )

    e += np.einsum(
        "ijkl,klij->", tau2, tau3, optimize=True
    ) / 16

    tau2 = None

    tau3 = None

    tau4 = zeros((M, M))

    tau4 += np.einsum(
        "caji,cbji->ab", l2, t2, optimize=True
    )

    tau5 = zeros((M, M))

    tau5 += 2 * np.einsum(
        "ab->ab", f[v, v], optimize=True
    )

    tau5 -= np.einsum(
        "caji,jicb->ab", t2, u[o, o, v, v], optimize=True
    )

    e += np.einsum(
        "ab,ab->", tau4, tau5, optimize=True
    ) / 4

    tau4 = None

    tau5 = None

    tau6 = zeros((N, M))

    tau6 += 2 * np.einsum(
        "ai->ia", f[v, o], optimize=True
    )

    tau6 += 2 * np.einsum(
        "jb,baji->ia", f[o, v], t2, optimize=True
    )

    tau6 += np.einsum(
        "bakj,kjib->ia", t2, u[o, o, o, v], optimize=True
    )

    tau6 += np.einsum(
        "cbji,jacb->ia", t2, u[o, v, v, v], optimize=True
    )

    e += np.einsum(
        "ai,ia->", l1, tau6, optimize=True
    ) / 2

    tau6 = None

    tau7 = zeros((N, N))

    tau7 += np.einsum(
        "baki,bakj->ij", l2, t2, optimize=True
    )

    tau8 = zeros((N, N))

    tau8 += 2 * np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    tau8 += np.einsum(
        "bakj,kiba->ij", t2, u[o, o, v, v], optimize=True
    )

    e -= np.einsum(
        "ij,ji->", tau7, tau8, optimize=True
    ) / 4

    tau8 = None

    tau7 = None

    return e