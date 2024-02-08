import numpy as np

def lambda_amplitudes_intermediates_rccd(t2, l2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum(
        "abij,balk->ijkl", l2, t2, optimize=True
    )

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum(
        "jikl,lkab->ijab", tau0, u[o, o, v, v], optimize=True
    )

    tau0 = None

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum(
        "ijab->ijab", tau1, optimize=True
    )

    tau1 = None

    tau2 = zeros((N, N, N, N))

    tau2 += np.einsum(
        "baij,lkab->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau3 = zeros((N, N, N, N))

    tau3 += np.einsum(
        "lkji->ijkl", tau2, optimize=True
    )

    tau2 = None

    tau3 += np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum(
        "abkl,ijlk->ijab", l2, tau3, optimize=True
    )

    tau3 = None

    tau5 += np.einsum(
        "jiab->ijab", tau4, optimize=True
    )

    tau4 = None

    tau5 += np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    r2 = zeros((M, M, N, N))

    r2 -= 2 * np.einsum(
        "ijba->abij", tau5, optimize=True
    )

    r2 += 4 * np.einsum(
        "ijab->abij", tau5, optimize=True
    )

    tau5 = None

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum(
        "caik,bcjk->ijab", l2, t2, optimize=True
    )

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum(
        "ijab->ijab", tau6, optimize=True
    )

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 -= np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau7 += 2 * np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum(
        "caki,kjcb->ijab", l2, tau7, optimize=True
    )

    tau9 -= np.einsum(
        "ijab->ijab", tau8, optimize=True
    )

    tau8 = None

    tau17 = zeros((M, M))

    tau17 += np.einsum(
        "ijbc,ijac->ab", tau7, u[o, o, v, v], optimize=True
    )

    tau18 = zeros((M, M))

    tau18 -= np.einsum(
        "ba->ab", tau17, optimize=True
    )

    tau17 = None

    tau20 = zeros((N, N))

    tau20 += np.einsum(
        "kjab,kiab->ij", tau7, u[o, o, v, v], optimize=True
    )

    tau21 = zeros((N, N))

    tau21 += np.einsum(
        "ji->ij", tau20, optimize=True
    )

    tau20 = None

    tau31 = zeros((N, N))

    tau31 += np.einsum(
        "abki,kjab->ij", l2, tau7, optimize=True
    )

    tau7 = None

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum(
        "jk,kiab->ijab", tau31, u[o, o, v, v], optimize=True
    )

    tau31 = None

    tau33 = zeros((N, N, M, M))

    tau33 -= np.einsum(
        "jiba->ijab", tau32, optimize=True
    )

    tau32 = None

    tau10 = zeros((N, N, M, M))

    tau10 -= np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau10 += 2 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum(
        "kjcb,ikac->ijab", tau10, tau9, optimize=True
    )

    tau9 = None

    tau10 = None

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum(
        "ijab->ijab", tau11, optimize=True
    )

    tau11 = None

    tau12 = zeros((N, N, M, M))

    tau12 += 2 * np.einsum(
        "iabj->ijab", u[o, v, v, o], optimize=True
    )

    tau12 -= np.einsum(
        "iajb->ijab", u[o, v, o, v], optimize=True
    )

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum(
        "caki,jkcb->ijab", l2, tau12, optimize=True
    )

    tau12 = None

    tau23 -= np.einsum(
        "ijab->ijab", tau13, optimize=True
    )

    tau13 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum(
        "acki,jkbc->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum(
        "ijab->ijab", tau14, optimize=True
    )

    tau14 = None

    tau15 -= np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum(
        "acki,kjcb->ijab", l2, tau15, optimize=True
    )

    tau15 = None

    tau23 -= np.einsum(
        "ijab->ijab", tau16, optimize=True
    )

    tau16 = None

    tau18 += np.einsum(
        "ab->ab", f[v, v], optimize=True
    )

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum(
        "cb,caij->ijab", tau18, l2, optimize=True
    )

    tau18 = None

    tau23 -= np.einsum(
        "jiab->ijab", tau19, optimize=True
    )

    tau19 = None

    tau21 += np.einsum(
        "ji->ij", f[o, o], optimize=True
    )

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum(
        "kj,abki->ijab", tau21, l2, optimize=True
    )

    tau21 = None

    tau23 += np.einsum(
        "ijba->ijab", tau22, optimize=True
    )

    tau22 = None

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
        "cdij,dcba->ijab", l2, u[v, v, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "jiab->abij", tau24, optimize=True
    )

    r2 += 7 * np.einsum(
        "jiba->abij", tau24, optimize=True
    ) / 2

    tau24 = None

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum(
        "caik,jckb->ijab", l2, u[o, v, o, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "ijab->abij", tau25, optimize=True
    )

    r2 -= 4 * np.einsum(
        "ijba->abij", tau25, optimize=True
    )

    r2 -= 3 * np.einsum(
        "jiab->abij", tau25, optimize=True
    )

    r2 += 2 * np.einsum(
        "jiba->abij", tau25, optimize=True
    )

    tau25 = None

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum(
        "acki,jkcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum(
        "caik,kjcb->ijab", l2, tau26, optimize=True
    )

    tau26 = None

    tau33 += np.einsum(
        "ijab->ijab", tau27, optimize=True
    )

    tau27 = None

    tau28 = zeros((N, N, M, M))

    tau28 += 2 * np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau28 -= np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau29 = zeros((M, M))

    tau29 += np.einsum(
        "acij,ijcb->ab", l2, tau28, optimize=True
    )

    tau28 = None

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum(
        "bc,ijca->ijab", tau29, u[o, o, v, v], optimize=True
    )

    tau29 = None

    tau33 -= np.einsum(
        "jiba->ijab", tau30, optimize=True
    )

    tau30 = None

    r2 -= 2 * np.einsum(
        "ijab->abij", tau33, optimize=True
    )

    r2 += 4 * np.einsum(
        "ijba->abij", tau33, optimize=True
    )

    r2 += 4 * np.einsum(
        "jiab->abij", tau33, optimize=True
    )

    r2 -= 2 * np.einsum(
        "jiba->abij", tau33, optimize=True
    )

    tau33 = None

    return r2