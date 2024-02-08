import numpy as np

def amplitudes_intermediates_rccd(t2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum(
        "cdij,badc->ijab", t2, u[v, v, v, v], optimize=True
    )

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum(
        "jiba->ijab", tau0, optimize=True
    )

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum(
        "caik,kjcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum(
        "caik,jkbc->ijab", t2, tau1, optimize=True
    )

    tau1 = None

    tau9 += np.einsum(
        "ijab->ijab", tau2, optimize=True
    )

    tau2 = None

    tau3 = zeros((N, N, M, M))

    tau3 -= np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau3 += 2 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum(
        "caki,kjcb->ijab", t2, tau3, optimize=True
    )

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum(
        "caki,jkbc->ijab", t2, tau4, optimize=True
    )

    tau4 = None

    tau9 += 2 * np.einsum(
        "jiba->ijab", tau5, optimize=True
    )

    tau5 = None

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum(
        "acki,kjcb->ijab", t2, tau3, optimize=True
    )

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum(
        "ijab->ijab", tau15, optimize=True
    )

    tau15 = None

    tau21 = zeros((N, N))

    tau21 += np.einsum(
        "abki,kjab->ij", t2, tau3, optimize=True
    )

    tau3 = None

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum(
        "jk,abki->ijab", tau21, t2, optimize=True
    )

    tau21 = None

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum(
        "ijba->ijab", tau22, optimize=True
    )

    tau22 = None

    tau6 = zeros((N, N, N, N))

    tau6 += np.einsum(
        "abij,lkba->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau7 = zeros((N, N, N, N))

    tau7 += np.einsum(
        "lkji->ijkl", tau6, optimize=True
    )

    tau6 = None

    tau7 += np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum(
        "abkl,klij->ijab", t2, tau7, optimize=True
    )

    tau7 = None

    tau9 += np.einsum(
        "ijab->ijab", tau8, optimize=True
    )

    tau8 = None

    tau9 += np.einsum(
        "baji->ijab", u[v, v, o, o], optimize=True
    )

    r2 = zeros((M, M, N, N))

    r2 -= 2 * np.einsum(
        "ijba->abij", tau9, optimize=True
    )

    r2 += 4 * np.einsum(
        "ijab->abij", tau9, optimize=True
    )

    tau9 = None

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum(
        "ki,abjk->ijab", f[o, o], t2, optimize=True
    )

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum(
        "ijab->ijab", tau10, optimize=True
    )

    tau10 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum(
        "ac,cbji->ijab", f[v, v], t2, optimize=True
    )

    tau13 -= np.einsum(
        "ijab->ijab", tau11, optimize=True
    )

    tau11 = None

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum(
        "caik,kbjc->ijab", t2, u[o, v, o, v], optimize=True
    )

    tau13 += np.einsum(
        "ijab->ijab", tau12, optimize=True
    )

    tau12 = None

    r2 += 2 * np.einsum(
        "ijab->abij", tau13, optimize=True
    )

    r2 -= 4 * np.einsum(
        "ijba->abij", tau13, optimize=True
    )

    r2 -= 4 * np.einsum(
        "jiab->abij", tau13, optimize=True
    )

    r2 += 2 * np.einsum(
        "jiba->abij", tau13, optimize=True
    )

    tau13 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum(
        "caik,kbcj->ijab", t2, u[o, v, v, o], optimize=True
    )

    tau23 += np.einsum(
        "ijab->ijab", tau14, optimize=True
    )

    tau14 = None

    tau16 += np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau16 -= 2 * np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum(
        "caki,jkbc->ijab", t2, tau16, optimize=True
    )

    tau16 = None

    tau23 += np.einsum(
        "ijab->ijab", tau17, optimize=True
    )

    tau17 = None

    tau18 = zeros((N, N, M, M))

    tau18 += 2 * np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau18 -= np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau19 = zeros((M, M))

    tau19 += np.einsum(
        "acij,ijcb->ab", t2, tau18, optimize=True
    )

    tau18 = None

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum(
        "bc,caij->ijab", tau19, t2, optimize=True
    )

    tau19 = None

    tau23 += np.einsum(
        "jiab->ijab", tau20, optimize=True
    )

    tau20 = None

    r2 -= 4 * np.einsum(
        "ijab->abij", tau23, optimize=True
    )

    r2 += 2 * np.einsum(
        "ijba->abij", tau23, optimize=True
    )

    r2 += 2 * np.einsum(
        "jiab->abij", tau23, optimize=True
    )

    r2 -= 4 * np.einsum(
        "jiba->abij", tau23, optimize=True
    )

    tau23 = None

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum(
        "caik,kjbc->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum(
        "caik,jkbc->ijab", t2, tau24, optimize=True
    )

    tau24 = None

    r2 += 4 * np.einsum(
        "jiab->abij", tau25, optimize=True
    )

    r2 -= 2 * np.einsum(
        "jiba->abij", tau25, optimize=True
    )

    tau25 = None


    return r2
