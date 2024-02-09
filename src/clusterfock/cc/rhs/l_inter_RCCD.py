import numpy as np

def lambda_amplitudes_intermediates_rccd(t2, l2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    l2 = l2.transpose(2,3,0,1)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum(
        "ijab,balk->ijkl", l2, t2, optimize=True
    )

    rhs = zeros((N, N, M, M))

    rhs += 2 * np.einsum(
        "jikl,lkab->ijab", tau0, u[o, o, v, v], optimize=True
    )

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum(
        "ik,jkab->ijab", f[o, o], l2, optimize=True
    )

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum(
        "ijab->ijab", tau1, optimize=True
    )

    tau1 = None

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum(
        "ca,ijbc->ijab", f[v, v], l2, optimize=True
    )

    tau6 -= np.einsum(
        "ijab->ijab", tau2, optimize=True
    )

    tau2 = None

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum(
        "acki,jkcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum(
        "ijab->ijab", tau3, optimize=True
    )

    tau3 = None

    tau4 -= np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum(
        "kiac,kjcb->ijab", l2, tau4, optimize=True
    )

    tau6 -= np.einsum(
        "ijab->ijab", tau5, optimize=True
    )

    tau5 = None

    rhs -= 2 * np.einsum(
        "ijba->ijab", tau6, optimize=True
    )

    rhs -= 2 * np.einsum(
        "jiab->ijab", tau6, optimize=True
    )

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum(
        "kica,kjcb->ijab", l2, tau4, optimize=True
    )

    tau4 = None

    tau14 = zeros((N, N, M, M))

    tau14 -= np.einsum(
        "ijab->ijab", tau7, optimize=True
    )

    tau7 = None

    tau8 = zeros((N, N, M, M))

    tau8 += 2 * np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau8 -= np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau9 = zeros((M, M))

    tau9 += np.einsum(
        "ijbc,ijca->ab", tau8, u[o, o, v, v], optimize=True
    )

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum(
        "bc,ijca->ijab", tau9, l2, optimize=True
    )

    tau9 = None

    tau14 += np.einsum(
        "jiab->ijab", tau10, optimize=True
    )

    tau10 = None

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum(
        "kjbc,kica->ijab", tau8, u[o, o, v, v], optimize=True
    )

    tau8 = None

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum(
        "jiba->ijab", tau16, optimize=True
    )

    tau16 = None

    tau11 = zeros((N, N, M, M))

    tau11 -= np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau11 += 2 * np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau12 = zeros((N, N))

    tau12 += np.einsum(
        "kjab,kiab->ij", tau11, u[o, o, v, v], optimize=True
    )

    tau11 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum(
        "jk,kiab->ijab", tau12, l2, optimize=True
    )

    tau12 = None

    tau14 += np.einsum(
        "ijba->ijab", tau13, optimize=True
    )

    tau13 = None

    rhs -= 2 * np.einsum(
        "ijab->ijab", tau14, optimize=True
    )

    rhs -= 2 * np.einsum(
        "jiba->ijab", tau14, optimize=True
    )

    tau14 = None

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum(
        "caki,jkcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau17 -= np.einsum(
        "ijab->ijab", tau15, optimize=True
    )

    tau15 = None

    tau17 += np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau18 = zeros((N, N, M, M))

    tau18 += np.einsum(
        "kica,kjcb->ijab", l2, tau17, optimize=True
    )

    tau17 = None

    rhs += 4 * np.einsum(
        "ijab->ijab", tau18, optimize=True
    )

    rhs -= 2 * np.einsum(
        "ijba->ijab", tau18, optimize=True
    )

    rhs -= 2 * np.einsum(
        "jiab->ijab", tau18, optimize=True
    )

    rhs += 4 * np.einsum(
        "jiba->ijab", tau18, optimize=True
    )

    tau18 = None

    tau19 = zeros((N, N))

    tau19 += np.einsum(
        "ikab,bakj->ij", l2, t2, optimize=True
    )

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum(
        "ik,jkab->ijab", tau19, u[o, o, v, v], optimize=True
    )

    tau19 = None

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum(
        "ijab->ijab", tau20, optimize=True
    )

    tau20 = None

    tau21 = zeros((M, M))

    tau21 += np.einsum(
        "jica,cbji->ab", l2, t2, optimize=True
    )

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum(
        "ac,ijbc->ijab", tau21, u[o, o, v, v], optimize=True
    )

    tau21 = None

    tau23 += np.einsum(
        "ijab->ijab", tau22, optimize=True
    )

    tau22 = None

    rhs += 2 * np.einsum(
        "ijab->ijab", tau23, optimize=True
    )

    rhs -= 4 * np.einsum(
        "ijba->ijab", tau23, optimize=True
    )

    rhs -= 4 * np.einsum(
        "jiab->ijab", tau23, optimize=True
    )

    rhs += 2 * np.einsum(
        "jiba->ijab", tau23, optimize=True
    )

    tau23 = None

    tau24 = zeros((N, N, N, N))

    tau24 += np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau24 += np.einsum(
        "balk,ijab->ijkl", t2, u[o, o, v, v], optimize=True
    )

    rhs += 2 * np.einsum(
        "klba,jikl->ijab", l2, tau24, optimize=True
    )

    tau24 = None

    rhs += 2 * np.einsum(
        "jicd,dcab->ijab", l2, u[v, v, v, v], optimize=True
    )

    rhs -= 2 * np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    rhs += 4 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    return rhs.transpose(2,3,0,1)