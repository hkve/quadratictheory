import numpy as np


def amplitudes_intermediates_ccd(t2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("ki,abjk->ijab", f[o, o], t2, optimize=True)

    r2 = np.zeros((M, M, N, N))

    r2 -= np.einsum("ijba->abij", tau0, optimize=True)

    r2 += np.einsum("jiba->abij", tau0, optimize=True)

    tau0 = None

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    tau6 = np.zeros((N, N, M, M))

    tau6 -= 2 * np.einsum("jiab->ijab", tau1, optimize=True)

    tau1 = None

    tau2 = np.zeros((M, M))

    tau2 -= np.einsum("acji,jicb->ab", t2, u[o, o, v, v], optimize=True)

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("bc,acij->ijab", tau2, t2, optimize=True)

    tau2 = None

    tau6 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau3 = None

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum("acik,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("acik,jkbc->ijab", t2, tau4, optimize=True)

    tau4 = None

    tau6 += 2 * np.einsum("ijba->ijab", tau5, optimize=True)

    tau5 = None

    r2 -= np.einsum("ijab->abij", tau6, optimize=True) / 2

    r2 += np.einsum("ijba->abij", tau6, optimize=True) / 2

    tau6 = None

    tau7 = np.zeros((N, N))

    tau7 -= np.einsum("baik,kjba->ij", t2, u[o, o, v, v], optimize=True)

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("jk,abik->ijab", tau7, t2, optimize=True)

    tau7 = None

    r2 -= np.einsum("ijab->abij", tau8, optimize=True) / 2

    r2 += np.einsum("jiab->abij", tau8, optimize=True) / 2

    tau8 = None

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum("acik,kbjc->ijab", t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ijab->abij", tau9, optimize=True)

    r2 += np.einsum("ijba->abij", tau9, optimize=True)

    r2 += np.einsum("jiab->abij", tau9, optimize=True)

    r2 -= np.einsum("jiba->abij", tau9, optimize=True)

    tau9 = None

    tau10 = np.zeros((N, N, N, N))

    tau10 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau10 += np.einsum("balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bakl,klji->abij", t2, tau10, optimize=True) / 4

    tau10 = None

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    r2 += np.einsum("dcji,badc->abij", t2, u[v, v, v, v], optimize=True) / 2

    return r2
