import numpy as np

def td_energy_addition_opti_restricted(t2, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N))

    tau0 += np.einsum(
        "abik,bakj->ij", l2, t2, optimize=True
    )

    e = 0

    e -= 2 * np.einsum(
        "ji,ij->", f[o, o], tau0, optimize=True
    )

    tau0 = None

    tau1 = zeros((M, M))

    tau1 += np.einsum(
        "acij,cbji->ab", l2, t2, optimize=True
    )

    e += 2 * np.einsum(
        "ba,ba->", f[v, v], tau1, optimize=True
    )

    tau1 = None

    tau2 = zeros((N, N, M, M))

    tau2 -= np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau2 += np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau3 = zeros((N, N, M, M))

    tau3 -= np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau3 += 2 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum(
        "kiac,kjcb->ijab", tau2, tau3, optimize=True
    )

    tau2 = None

    tau11 = zeros((M, M))

    tau11 += np.einsum(
        "acji,ijcb->ab", t2, tau3, optimize=True
    )

    tau3 = None

    tau4 += 2 * np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau4 -= np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau6 = zeros((N, N, M, M))

    tau6 += 2 * np.einsum(
        "bcjk,ikac->ijab", t2, tau4, optimize=True
    )

    tau4 = None

    tau5 = zeros((N, N, M, M))

    tau5 += 2 * np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau5 -= np.einsum(
        "acki,jkbc->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau6 -= np.einsum(
        "bckj,ikac->ijab", t2, tau5, optimize=True
    )

    tau5 = None

    tau6 += np.einsum(
        "baji->ijab", u[v, v, o, o], optimize=True
    )

    tau6 += np.einsum(
        "dcji,abcd->ijab", t2, u[v, v, v, v], optimize=True
    )

    e += np.einsum(
        "baji,ijab->", l2, tau6, optimize=True
    )

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += 2 * np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau7 -= np.einsum(
        "acki,jkcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum(
        "acki,jkbc->ijab", t2, tau7, optimize=True
    )

    tau7 = None

    e -= np.einsum(
        "abji,ijab->", l2, tau8, optimize=True
    )

    tau8 = None

    tau9 = zeros((N, N, N, N))

    tau9 += np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau9 += np.einsum(
        "balk,ijab->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau10 = zeros((N, N, N, N))

    tau10 += np.einsum(
        "abij,abkl->ijkl", l2, t2, optimize=True
    )

    e += np.einsum(
        "lkji,ijkl->", tau10, tau9, optimize=True
    )

    tau9 = None

    tau10 = None

    tau12 = zeros((M, M))

    tau12 += np.einsum(
        "caij,cbij->ab", l2, t2, optimize=True
    )

    e -= 2 * np.einsum(
        "ab,ab->", tau11, tau12, optimize=True
    )

    tau11 = None

    tau12 = None

    tau13 = zeros((N, N, M, M))

    tau13 += 2 * np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau13 -= np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau14 = zeros((N, N))

    tau14 += np.einsum(
        "abik,kjab->ij", t2, tau13, optimize=True
    )

    tau13 = None

    tau15 = zeros((N, N))

    tau15 += np.einsum(
        "abki,abkj->ij", l2, t2, optimize=True
    )

    e -= 2 * np.einsum(
        "ij,ij->", tau14, tau15, optimize=True
    )

    tau14 = None

    tau15 = None

    return e