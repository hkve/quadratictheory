import numpy as np


def amplitudes_intermediates_rccd(t2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("ijba->abij", tau0, optimize=True)

    r2 += np.einsum("jiab->abij", tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau1 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("acki,kjcb->ijab", t2, tau1, optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("acik,jkbc->ijab", t2, tau2, optimize=True)

    tau2 = None

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau3 = None

    tau8 = zeros((N, N, M, M))

    tau8 += 2 * np.einsum("acik,kjcb->ijab", t2, tau1, optimize=True)

    tau10 = zeros((N, N))

    tau10 += np.einsum("abik,kjba->ij", t2, tau1, optimize=True)

    tau1 = None

    tau4 = zeros((N, N, M, M))

    tau4 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau4 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau5 = zeros((M, M))

    tau5 += np.einsum("acij,ijcb->ab", t2, tau4, optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("bc,acij->ijab", tau5, t2, optimize=True)

    tau5 = None

    tau7 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau6 = None

    r2 -= np.einsum("ijab->abij", tau7, optimize=True)

    r2 -= np.einsum("jiba->abij", tau7, optimize=True)

    tau7 = None

    tau11 = zeros((N, N))

    tau11 += np.einsum("abki,kjba->ij", t2, tau4, optimize=True)

    tau4 = None

    tau8 += 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau8 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    r2 += np.einsum("acik,jkbc->abij", t2, tau8, optimize=True)

    tau8 = None

    tau9 = zeros((N, N, N, N))

    tau9 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau9 += np.einsum("abkl,ijab->ijkl", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("abkl,klij->abij", t2, tau9, optimize=True)

    tau9 = None

    tau10 += np.einsum("ji->ij", f[o, o], optimize=True)

    r2 -= np.einsum("ik,abkj->abij", tau10, t2, optimize=True)

    tau10 = None

    tau11 += np.einsum("ji->ij", f[o, o], optimize=True)

    r2 -= np.einsum("jk,abik->abij", tau11, t2, optimize=True)

    tau11 = None

    tau12 = zeros((N, N, M, M))

    tau12 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau12 += np.einsum("acki,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bckj,ikac->abij", t2, tau12, optimize=True)

    tau12 = None

    tau13 = zeros((N, N, M, M))

    tau13 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau13 += np.einsum("acki,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ackj,ikbc->abij", t2, tau13, optimize=True)

    tau13 = None

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    r2 += np.einsum("cdij,badc->abij", t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("bcjk,kaic->abij", t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("acki,kbcj->abij", t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("bcki,kajc->abij", t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("bcjk,kaci->abij", t2, u[o, v, v, o], optimize=True)

    return r2
