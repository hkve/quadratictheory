import numpy as np


def lambda_amplitudes_intermediates_rccd(t2, l2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += 2 * np.einsum("ablk,lkij->abij", l2, tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, N, N))

    tau1 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    r2 += 2 * np.einsum("ijlk,lkab->abij", tau1, u[o, o, v, v], optimize=True)

    tau1 = None

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("acki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("ijab->ijab", tau2, optimize=True)

    tau2 = None

    tau3 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("caki,kjcb->ijab", l2, tau3, optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 -= np.einsum("ijab->ijab", tau4, optimize=True)

    tau4 = None

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("acki,kjcb->ijab", l2, tau3, optimize=True)

    tau3 = None

    r2 += 2 * np.einsum("ijba->abij", tau23, optimize=True)

    r2 += 2 * np.einsum("jiab->abij", tau23, optimize=True)

    tau23 = None

    tau5 = zeros((N, N, M, M))

    tau5 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau5 -= np.einsum("baji->ijab", t2, optimize=True)

    tau6 = zeros((M, M))

    tau6 += np.einsum("ijbc,ijca->ab", tau5, u[o, o, v, v], optimize=True)

    tau5 = None

    tau7 = zeros((M, M))

    tau7 -= np.einsum("ba->ab", tau6, optimize=True)

    tau6 = None

    tau7 += np.einsum("ab->ab", f[v, v], optimize=True)

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("cb,caij->ijab", tau7, l2, optimize=True)

    tau7 = None

    tau13 -= np.einsum("jiab->ijab", tau8, optimize=True)

    tau8 = None

    tau9 = zeros((N, N, M, M))

    tau9 -= np.einsum("abji->ijab", t2, optimize=True)

    tau9 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau10 = zeros((N, N))

    tau10 += np.einsum("kjab,kiab->ij", tau9, u[o, o, v, v], optimize=True)

    tau11 = zeros((N, N))

    tau11 += np.einsum("ji->ij", tau10, optimize=True)

    tau10 = None

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("kjcb,kica->ijab", tau9, u[o, o, v, v], optimize=True)

    tau9 = None

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("jiba->ijab", tau15, optimize=True)

    tau15 = None

    tau11 += np.einsum("ji->ij", f[o, o], optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("kj,abki->ijab", tau11, l2, optimize=True)

    tau11 = None

    tau13 += np.einsum("ijba->ijab", tau12, optimize=True)

    tau12 = None

    r2 -= 2 * np.einsum("ijab->abij", tau13, optimize=True)

    r2 -= 2 * np.einsum("jiba->abij", tau13, optimize=True)

    tau13 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("caki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau16 -= np.einsum("ijab->ijab", tau14, optimize=True)

    tau14 = None

    tau16 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("caki,kjcb->ijab", l2, tau16, optimize=True)

    tau16 = None

    r2 += 4 * np.einsum("ijab->abij", tau17, optimize=True)

    r2 -= 2 * np.einsum("ijba->abij", tau17, optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", tau17, optimize=True)

    r2 += 4 * np.einsum("jiba->abij", tau17, optimize=True)

    tau17 = None

    tau18 = zeros((M, M))

    tau18 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("ac,jicb->ijab", tau18, u[o, o, v, v], optimize=True)

    tau18 = None

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum("ijab->ijab", tau19, optimize=True)

    tau19 = None

    tau20 = zeros((N, N))

    tau20 += np.einsum("abik,bakj->ij", l2, t2, optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 += np.einsum("ik,jkab->ijab", tau20, u[o, o, v, v], optimize=True)

    tau20 = None

    tau22 += np.einsum("ijab->ijab", tau21, optimize=True)

    tau21 = None

    r2 += 2 * np.einsum("ijab->abij", tau22, optimize=True)

    r2 -= 4 * np.einsum("ijba->abij", tau22, optimize=True)

    r2 -= 4 * np.einsum("jiab->abij", tau22, optimize=True)

    r2 += 2 * np.einsum("jiba->abij", tau22, optimize=True)

    tau22 = None

    r2 += 2 * np.einsum("dcji,dcba->abij", l2, u[v, v, v, v], optimize=True)

    r2 += 2 * np.einsum("balk,jilk->abij", l2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", u[o, o, v, v], optimize=True)

    r2 += 4 * np.einsum("jiba->abij", u[o, o, v, v], optimize=True)

    return 0.5 * r2
