import numpy as np


def lambda_amplitudes_intermediates_ccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("ijlk,lkab->abij", tau0, u[o, o, v, v], optimize=True) / 4

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum("caki,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("ijab->ijab", tau1, optimize=True)

    tau1 = None

    tau2 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("caki,kjcb->ijab", l2, tau2, optimize=True)

    tau2 = None

    r2 += np.einsum("ijab->abij", tau3, optimize=True)

    r2 -= np.einsum("ijba->abij", tau3, optimize=True)

    r2 -= np.einsum("jiab->abij", tau3, optimize=True)

    r2 += np.einsum("jiba->abij", tau3, optimize=True)

    tau3 = None

    tau4 = zeros((N, N))

    tau4 -= np.einsum("baki,jkba->ij", t2, u[o, o, v, v], optimize=True)

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("kj,abik->ijab", tau4, l2, optimize=True)

    tau4 = None

    tau8 = zeros((N, N, M, M))

    tau8 -= np.einsum("ijba->ijab", tau5, optimize=True)

    tau5 = None

    tau6 = zeros((N, N))

    tau6 -= np.einsum("baik,bakj->ij", l2, t2, optimize=True)

    tau7 = zeros((N, N, M, M))

    tau7 -= np.einsum("ik,jkab->ijab", tau6, u[o, o, v, v], optimize=True)

    tau6 = None

    tau8 -= np.einsum("ijba->ijab", tau7, optimize=True)

    tau7 = None

    r2 += np.einsum("ijba->abij", tau8, optimize=True) / 2

    r2 -= np.einsum("jiba->abij", tau8, optimize=True) / 2

    tau8 = None

    tau9 = zeros((M, M))

    tau9 -= np.einsum("acji,cbji->ab", l2, t2, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 -= np.einsum("ac,ijbc->ijab", tau9, u[o, o, v, v], optimize=True)

    tau9 = None

    tau14 = zeros((N, N, M, M))

    tau14 -= np.einsum("ijab->ijab", tau10, optimize=True)

    tau10 = None

    tau11 = zeros((M, M))

    tau11 -= np.einsum("caji,jibc->ab", t2, u[o, o, v, v], optimize=True)

    tau12 = zeros((M, M))

    tau12 -= np.einsum("ab->ab", tau11, optimize=True)

    tau11 = None

    tau12 += 2 * np.einsum("ab->ab", f[v, v], optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("cb,caij->ijab", tau12, l2, optimize=True)

    tau12 = None

    tau14 += np.einsum("jiab->ijab", tau13, optimize=True)

    tau13 = None

    r2 += np.einsum("ijab->abij", tau14, optimize=True) / 2

    r2 -= np.einsum("ijba->abij", tau14, optimize=True) / 2

    tau14 = None

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("jk,abik->ijab", f[o, o], l2, optimize=True)

    r2 += np.einsum("ijba->abij", tau15, optimize=True)

    r2 -= np.einsum("jiba->abij", tau15, optimize=True)

    tau15 = None

    tau16 = zeros((N, N, N, N))

    tau16 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau16 += np.einsum("balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bakl,jikl->abij", l2, tau16, optimize=True) / 4

    tau16 = None

    r2 += np.einsum("dcji,dcba->abij", l2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("jiba->abij", u[o, o, v, v], optimize=True)

    return r2
