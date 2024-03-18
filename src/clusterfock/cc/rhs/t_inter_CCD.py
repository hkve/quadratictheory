import numpy as np


def amplitudes_intermediates_ccd(t2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum("ki,abjk->ijab", f[o, o], t2, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 -= 2 * np.einsum("ijab->ijab", tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum("ac,cbji->ijab", f[v, v], t2, optimize=True)

    tau10 += 2 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau1 = None

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("caik,kbjc->ijab", t2, u[o, v, o, v], optimize=True)

    tau10 -= 2 * np.einsum("ijab->ijab", tau2, optimize=True)

    tau2 = None

    tau3 = zeros((N, N))

    tau3 -= np.einsum("baik,kjba->ij", t2, u[o, o, v, v], optimize=True)

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("jk,abik->ijab", tau3, t2, optimize=True)

    tau3 = None

    tau10 += np.einsum("ijab->ijab", tau4, optimize=True)

    tau4 = None

    tau5 = zeros((M, M))

    tau5 += np.einsum("caji,jicb->ab", t2, u[o, o, v, v], optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("bc,caji->ijab", tau5, t2, optimize=True)

    tau5 = None

    tau10 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("caik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("ijab->ijab", tau7, optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("caik,jkbc->ijab", t2, tau7, optimize=True)

    tau7 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("ijab->ijab", tau13, optimize=True)

    tau13 = None

    tau8 += 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("caki,jkbc->ijab", t2, tau8, optimize=True)

    tau8 = None

    tau10 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau9 = None

    r2 = zeros((M, M, N, N))

    r2 -= np.einsum("ijab->abij", tau10, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau10, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau10, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau10, optimize=True) / 4

    tau10 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("acik,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("acik,jkbc->ijab", t2, tau11, optimize=True)

    tau11 = None

    tau14 += np.einsum("ijab->ijab", tau12, optimize=True)

    tau12 = None

    r2 -= np.einsum("jiab->abij", tau14, optimize=True) / 4

    r2 += np.einsum("jiba->abij", tau14, optimize=True) / 4

    tau14 = None

    tau15 = zeros((N, N, N, N))

    tau15 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau15 -= np.einsum("ablk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bakl,klji->abij", t2, tau15, optimize=True) / 4

    tau15 = None

    r2 -= np.einsum("cdji,badc->abij", t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    r2 = 0.25 * (
        r2 - r2.transpose(1, 0, 2, 3) - r2.transpose(0, 1, 3, 2) + r2.transpose(1, 0, 3, 2)
    )

    return r2
