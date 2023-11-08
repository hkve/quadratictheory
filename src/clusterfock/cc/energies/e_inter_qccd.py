import numpy as np


def energy_intermediates_qccd(t2, l2, u, f, o, v):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau14 = np.zeros((N, N, N, N))

    tau14 += np.einsum("mikn,njlm->ijkl", tau0, tau0, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau2 = np.zeros((N, N, N, N))

    tau2 += np.einsum("ijab,kalb->ijkl", tau1, u[o, v, o, v], optimize=True)

    e = 0

    e += np.einsum("jilk,kjli->", tau0, tau2, optimize=True) / 2

    tau2 = None

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("caki,kjcb->ijab", t2, tau1, optimize=True)

    tau6 = np.zeros((N, N, N, N))

    tau6 += np.einsum("baij,klba->ijkl", l2, tau5, optimize=True)

    tau5 = None

    e += np.einsum("lkij,jilk->", tau6, u[o, o, o, o], optimize=True) / 4

    tau6 = None

    tau8 = np.zeros((M, M, M, M))

    tau8 += 4 * np.einsum("ijab,jcid->abcd", tau1, u[o, v, o, v], optimize=True)

    tau11 = np.zeros((N, N, M, M))

    tau11 += 4 * np.einsum("ikca,kcjb->ijab", tau1, u[o, v, o, v], optimize=True)

    tau11 += 2 * np.einsum("ijcd,cadb->ijab", tau1, u[v, v, v, v], optimize=True)

    tau14 += 4 * np.einsum("ikab,jlba->ijkl", tau1, tau1, optimize=True)

    tau1 = None

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau4 = np.zeros((N, N, N, N))

    tau4 -= np.einsum("bajk,ilab->ijkl", t2, tau3, optimize=True)

    e -= np.einsum("lkji,ilkj->", tau0, tau4, optimize=True) / 4

    tau0 = None

    tau4 = None

    tau10 = np.zeros((N, N, M, M))

    tau10 += 4 * np.einsum("ijab->ijab", tau3, optimize=True)

    tau17 = np.zeros((N, N, M, M))

    tau17 += 4 * np.einsum("ijab->ijab", tau3, optimize=True)

    tau3 = None

    tau7 = np.zeros((M, M, M, M))

    tau7 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau8 -= np.einsum("eafc,befd->abcd", tau7, u[v, v, v, v], optimize=True)

    tau7 = None

    tau12 = np.zeros((N, N, M, M))

    tau12 -= np.einsum("cdji,bcda->ijab", l2, tau8, optimize=True)

    tau8 = None

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau10 -= np.einsum("jiba->ijab", tau9, optimize=True)

    tau9 = None

    tau11 -= np.einsum("cakj,ikcb->ijab", t2, tau10, optimize=True)

    tau12 += 2 * np.einsum("caki,jkcb->ijab", l2, tau11, optimize=True)

    tau11 = None

    tau16 = np.zeros((M, M))

    tau16 -= np.einsum("caij,ijcb->ab", t2, tau10, optimize=True)

    tau10 = None

    tau12 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    e += np.einsum("abij,ijab->", t2, tau12, optimize=True) / 8

    tau12 = None

    tau13 = np.zeros((N, N))

    tau13 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau14 -= np.einsum("ik,jl->ijkl", tau13, tau13, optimize=True)

    e += np.einsum("lkji,ijkl->", tau14, u[o, o, o, o], optimize=True) / 8

    tau14 = None

    tau15 = np.zeros((M, M))

    tau15 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau16 -= np.einsum("cd,cadb->ab", tau15, u[v, v, v, v], optimize=True)

    tau18 = np.zeros((N, N))

    tau18 += 2 * np.einsum("ab,jaib->ij", tau15, u[o, v, o, v], optimize=True)

    e += np.einsum("ab,ab->", tau15, tau16, optimize=True) / 8

    tau16 = None

    tau15 = None

    tau17 -= np.einsum("balk,jilk->ijab", l2, u[o, o, o, o], optimize=True)

    tau18 -= np.einsum("abki,kjab->ij", t2, tau17, optimize=True)

    tau17 = None

    e += np.einsum("ij,ij->", tau13, tau18, optimize=True) / 8

    tau18 = None

    tau13 = None

    return e
