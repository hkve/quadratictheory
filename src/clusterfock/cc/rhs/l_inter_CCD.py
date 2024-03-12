import numpy as np


def lambda_amplitudes_intermediates_ccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((N, N, N, N))

    tau0 += np.einsum(
        "abij,abkl->ijkl", l2, t2, optimize=True
    )

    r2 = np.zeros((M, M, N, N))

    r2 += np.einsum(
        "ijlk,lkab->abij", tau0, u[o, o, v, v], optimize=True
    ) / 4

    tau0 = None

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum(
        "ik,abjk->ijab", f[o, o], l2, optimize=True
    )

    tau12 = np.zeros((N, N, M, M))

    tau12 -= 2 * np.einsum(
        "ijab->ijab", tau1, optimize=True
    )

    tau1 = None

    tau2 = np.zeros((N, N, M, M))

    tau2 += np.einsum(
        "ca,cbji->ijab", f[v, v], l2, optimize=True
    )

    tau12 += 2 * np.einsum(
        "ijab->ijab", tau2, optimize=True
    )

    tau2 = None

    tau3 = np.zeros((N, N))

    tau3 += np.einsum(
        "abki,jkba->ij", t2, u[o, o, v, v], optimize=True
    )

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum(
        "kj,abik->ijab", tau3, l2, optimize=True
    )

    tau3 = None

    tau12 += np.einsum(
        "ijab->ijab", tau4, optimize=True
    )

    tau4 = None

    tau5 = np.zeros((M, M))

    tau5 -= np.einsum(
        "caji,jibc->ab", t2, u[o, o, v, v], optimize=True
    )

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum(
        "cb,caji->ijab", tau5, l2, optimize=True
    )

    tau5 = None

    tau12 += np.einsum(
        "ijab->ijab", tau6, optimize=True
    )

    tau6 = None

    tau7 = np.zeros((N, N, M, M))

    tau7 -= np.einsum(
        "abji->ijab", l2, optimize=True
    )

    tau7 += np.einsum(
        "baji->ijab", l2, optimize=True
    )

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau8 -= np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum(
        "kjbc,kica->ijab", tau8, u[o, o, v, v], optimize=True
    )

    tau10 = np.zeros((N, N, M, M))

    tau10 -= np.einsum(
        "jiba->ijab", tau9, optimize=True
    )

    tau9 = None

    tau13 = np.zeros((M, M))

    tau13 += np.einsum(
        "caij,ijcb->ab", l2, tau8, optimize=True
    )

    tau14 = np.zeros((N, N, M, M))

    tau14 -= np.einsum(
        "bc,ijca->ijab", tau13, u[o, o, v, v], optimize=True
    )

    tau13 = None

    r2 -= np.einsum(
        "ijba->abij", tau14, optimize=True
    ) / 4

    r2 += np.einsum(
        "ijab->abij", tau14, optimize=True
    ) / 4

    tau14 = None

    tau15 = np.zeros((N, N))

    tau15 += np.einsum(
        "abki,kjab->ij", l2, tau8, optimize=True
    )

    tau8 = None

    tau16 = np.zeros((N, N, M, M))

    tau16 -= np.einsum(
        "jk,kiab->ijab", tau15, u[o, o, v, v], optimize=True
    )

    tau15 = None

    r2 -= np.einsum(
        "jiab->abij", tau16, optimize=True
    ) / 4

    r2 += np.einsum(
        "ijab->abij", tau16, optimize=True
    ) / 4

    tau16 = None

    tau10 += 2 * np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum(
        "kjcb,kiac->ijab", tau10, tau7, optimize=True
    )

    tau10 = None

    tau7 = None

    tau12 -= np.einsum(
        "ijab->ijab", tau11, optimize=True
    )

    tau11 = None

    r2 -= np.einsum(
        "ijab->abij", tau12, optimize=True
    ) / 4

    r2 += np.einsum(
        "ijba->abij", tau12, optimize=True
    ) / 4

    r2 += np.einsum(
        "jiab->abij", tau12, optimize=True
    ) / 4

    r2 -= np.einsum(
        "jiba->abij", tau12, optimize=True
    ) / 4

    tau12 = None

    tau17 = np.zeros((N, N, N, N))

    tau17 += 2 * np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau17 -= np.einsum(
        "ablk,jiba->ijkl", t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bakl,jikl->abij", l2, tau17, optimize=True
    ) / 4

    tau17 = None

    r2 -= np.einsum(
        "cdji,dcba->abij", l2, u[v, v, v, v], optimize=True
    ) / 2

    r2 += np.einsum(
        "jiba->abij", u[o, o, v, v], optimize=True
    )

    r2 = 0.25*(r2 - r2.transpose(1,0,2,3) - r2.transpose(0,1,3,2) + r2.transpose(1,0,3,2))

    return r2
