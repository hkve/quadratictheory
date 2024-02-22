import numpy as np


def t1_transform_intermediates_ccsd(t2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    r1 = np.zeros((M,N), dtype=dtype)

    r1 += np.einsum(
        "ai->ai", f[v, o], optimize=True
    )

    r1 -= np.einsum(
        "abkj,kjib->ai", t2, u[o, o, o, v], optimize=True
    ) / 2

    r1 += np.einsum(
        "jb,abij->ai", f[o, v], t2, optimize=True
    )

    r1 -= np.einsum(
        "cbij,jacb->ai", t2, u[o, v, v, v], optimize=True
    ) / 2

    return r1

def t1_transform_lambda_intermediates_ccsd(t2, l1, l2, u, f, v, o):
    M, N = l1.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M))

    tau0 += np.einsum(
        "bj,abij->ia", l1, t2, optimize=True
    )

    r1 = zeros((M, N))

    r1 += np.einsum(
        "jb,ijab->ai", tau0, u[o, o, v, v], optimize=True
    )

    tau0 = None

    tau1 = zeros((N, N))

    tau1 -= np.einsum(
        "baik,bakj->ij", l2, t2, optimize=True
    )

    tau23 = zeros((N, N, M, M))

    tau23 -= np.einsum(
        "ik,jkab->ijab", tau1, u[o, o, v, v], optimize=True
    )

    tau24 = zeros((N, N, M, M))

    tau24 -= np.einsum(
        "ijba->ijab", tau23, optimize=True
    )

    tau23 = None

    r1 -= np.einsum(
        "ja,ij->ai", f[o, v], tau1, optimize=True
    ) / 2

    r1 += np.einsum(
        "jk,ikja->ai", tau1, u[o, o, o, v], optimize=True
    ) / 2

    tau1 = None

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum(
        "bi,abjk->ijka", l1, t2, optimize=True
    )

    r1 += np.einsum(
        "ikjb,kjab->ai", tau2, u[o, o, v, v], optimize=True
    ) / 2

    tau2 = None

    tau3 = zeros((N, N, N, N))

    tau3 += np.einsum(
        "baij,bakl->ijkl", l2, t2, optimize=True
    )

    r1 -= np.einsum(
        "ijlk,lkja->ai", tau3, u[o, o, o, v], optimize=True
    ) / 4

    r2 = zeros((M, M, N, N))

    r2 += np.einsum(
        "ijlk,lkab->abij", tau3, u[o, o, v, v], optimize=True
    ) / 4

    tau3 = None

    tau4 = zeros((N, N, M, M))

    tau4 -= np.einsum(
        "acki,bcjk->ijab", l2, t2, optimize=True
    )

    r1 -= np.einsum(
        "ijbc,jbac->ai", tau4, u[o, v, v, v], optimize=True
    )

    tau4 = None

    tau5 = zeros((M, M))

    tau5 -= np.einsum(
        "acji,cbji->ab", l2, t2, optimize=True
    )

    tau17 = zeros((N, N, M, M))

    tau17 -= np.einsum(
        "ac,ijbc->ijab", tau5, u[o, o, v, v], optimize=True
    )

    tau21 = zeros((N, N, M, M))

    tau21 -= np.einsum(
        "ijab->ijab", tau17, optimize=True
    )

    tau17 = None

    r1 += np.einsum(
        "bc,ibac->ai", tau5, u[o, v, v, v], optimize=True
    ) / 2

    tau5 = None

    tau6 = zeros((N, N, N, M))

    tau6 += 2 * np.einsum(
        "iakj->ijka", u[o, v, o, o], optimize=True
    )

    tau6 -= 2 * np.einsum(
        "ib,abkj->ijka", f[o, v], t2, optimize=True
    )

    tau6 += 4 * np.einsum(
        "balj,ilkb->ijka", t2, u[o, o, o, v], optimize=True
    )

    tau6 += np.einsum(
        "cbkj,iacb->ijka", t2, u[o, v, v, v], optimize=True
    )

    r1 -= np.einsum(
        "bajk,ijkb->ai", l2, tau6, optimize=True
    ) / 4

    tau6 = None

    tau7 = zeros((N, N))

    tau7 -= np.einsum(
        "baki,jkba->ij", t2, u[o, o, v, v], optimize=True
    )

    tau8 = zeros((N, N))

    tau8 += np.einsum(
        "ji->ij", tau7, optimize=True
    )

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum(
        "kj,abik->ijab", tau7, l2, optimize=True
    )

    tau7 = None

    tau24 -= np.einsum(
        "ijba->ijab", tau22, optimize=True
    )

    tau22 = None

    r2 += np.einsum(
        "ijba->abij", tau24, optimize=True
    ) / 2

    r2 -= np.einsum(
        "jiba->abij", tau24, optimize=True
    ) / 2

    tau24 = None

    tau8 += 2 * np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    r1 -= np.einsum(
        "aj,ij->ai", l1, tau8, optimize=True
    ) / 2

    tau8 = None

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum(
        "caki,jkbc->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum(
        "ijab->ijab", tau9, optimize=True
    )

    tau9 = None

    tau10 -= np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum(
        "caki,kjcb->ijab", l2, tau10, optimize=True
    )

    tau10 = None

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum(
        "ijab->ijab", tau11, optimize=True
    )

    tau11 = None

    tau12 += np.einsum(
        "ai,jb->ijab", l1, f[o, v], optimize=True
    )

    r2 += np.einsum(
        "ijab->abij", tau12, optimize=True
    )

    r2 -= np.einsum(
        "ijba->abij", tau12, optimize=True
    )

    r2 -= np.einsum(
        "jiab->abij", tau12, optimize=True
    )

    r2 += np.einsum(
        "jiba->abij", tau12, optimize=True
    )

    tau12 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum(
        "jk,abik->ijab", f[o, o], l2, optimize=True
    )

    tau15 = zeros((N, N, M, M))

    tau15 -= np.einsum(
        "ijba->ijab", tau13, optimize=True
    )

    tau13 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum(
        "ci,jcab->ijab", l1, u[o, v, v, v], optimize=True
    )

    tau15 -= np.einsum(
        "ijba->ijab", tau14, optimize=True
    )

    tau14 = None

    r2 -= np.einsum(
        "ijab->abij", tau15, optimize=True
    )

    r2 += np.einsum(
        "jiab->abij", tau15, optimize=True
    )

    tau15 = None

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum(
        "ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True
    )

    tau21 += 2 * np.einsum(
        "jiab->ijab", tau16, optimize=True
    )

    tau16 = None

    tau18 = zeros((M, M))

    tau18 -= np.einsum(
        "caji,jibc->ab", t2, u[o, o, v, v], optimize=True
    )

    tau19 = zeros((M, M))

    tau19 -= np.einsum(
        "ab->ab", tau18, optimize=True
    )

    tau18 = None

    tau19 += 2 * np.einsum(
        "ab->ab", f[v, v], optimize=True
    )

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum(
        "cb,caij->ijab", tau19, l2, optimize=True
    )

    tau19 = None

    tau21 += np.einsum(
        "jiab->ijab", tau20, optimize=True
    )

    tau20 = None

    r2 += np.einsum(
        "ijab->abij", tau21, optimize=True
    ) / 2

    r2 -= np.einsum(
        "ijba->abij", tau21, optimize=True
    ) / 2

    tau21 = None

    tau25 = zeros((N, N, N, N))

    tau25 += 2 * np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau25 += np.einsum(
        "balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bakl,jikl->abij", l2, tau25, optimize=True
    ) / 4

    tau25 = None

    r1 += np.einsum(
        "ia->ai", f[o, v], optimize=True
    )

    r1 += np.einsum(
        "bi,ba->ai", l1, f[v, v], optimize=True
    )

    r1 -= np.einsum(
        "bj,ibja->ai", l1, u[o, v, o, v], optimize=True
    )

    r1 -= np.einsum(
        "cbij,cbja->ai", l2, u[v, v, o, v], optimize=True
    ) / 2

    r2 += np.einsum(
        "jiba->abij", u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "dcji,dcba->abij", l2, u[v, v, v, v], optimize=True
    ) / 2

    return r1, r2