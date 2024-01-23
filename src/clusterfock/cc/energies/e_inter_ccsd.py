import numpy as np

def td_energy_addition(t1, t2, l1, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((M, M))

    tau0 += np.einsum(
        "caji,cbji->ab", l2, t2, optimize=True
    )

    tau21 = zeros((M, M))

    tau21 += np.einsum(
        "ab->ab", tau0, optimize=True
    )

    e = 0

    e += np.einsum(
        "ba,ba->", f[v, v], tau0, optimize=True
    ) / 2

    tau1 = zeros((M, M))

    tau1 += np.einsum(
        "caji,jicb->ab", t2, u[o, o, v, v], optimize=True
    )

    e -= np.einsum(
        "ab,ab->", tau0, tau1, optimize=True
    ) / 4

    tau0 = None

    tau1 = None

    tau2 = zeros((N, N))

    tau2 += np.einsum(
        "ia,aj->ij", f[o, v], t1, optimize=True
    )

    tau3 = zeros((N, N))

    tau3 += np.einsum(
        "baki,bakj->ij", l2, t2, optimize=True
    )

    tau24 = zeros((N, N))

    tau24 += np.einsum(
        "ij->ij", tau3, optimize=True
    )

    e -= np.einsum(
        "ij,ji->", tau2, tau3, optimize=True
    ) / 2

    tau3 = None

    tau4 = zeros((N, N))

    tau4 += np.einsum(
        "ai,aj->ij", l1, t1, optimize=True
    )

    tau24 += 2 * np.einsum(
        "ij->ij", tau4, optimize=True
    )

    e -= np.einsum(
        "ji,ij->", tau2, tau4, optimize=True
    )

    tau2 = None

    tau4 = None

    tau5 = zeros((N, N, M, M))

    tau5 += 2 * np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau5 -= np.einsum(
        "caki,jkbc->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau6 = zeros((N, N, M, M))

    tau6 -= 4 * np.einsum(
        "cbkj,ikac->ijab", t2, tau5, optimize=True
    )

    tau5 = None

    tau6 += 2 * np.einsum(
        "baji->ijab", u[v, v, o, o], optimize=True
    )

    tau6 += np.einsum(
        "dcji,badc->ijab", t2, u[v, v, v, v], optimize=True
    )

    e += np.einsum(
        "abij,ijab->", l2, tau6, optimize=True
    ) / 8

    tau6 = None

    tau7 = zeros((N, N, N, N))

    tau7 += 2 * np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau7 += np.einsum(
        "balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau8 = zeros((N, N, N, N))

    tau8 += np.einsum(
        "baij,bakl->ijkl", l2, t2, optimize=True
    )

    tau22 = zeros((N, M))

    tau22 += 2 * np.einsum(
        "jilk,lkja->ia", tau8, u[o, o, o, v], optimize=True
    )

    e += np.einsum(
        "ijkl,klij->", tau7, tau8, optimize=True
    ) / 16

    tau8 = None

    tau7 = None

    tau9 = zeros((N, M))

    tau9 += np.einsum(
        "bj,abij->ia", l1, t2, optimize=True
    )

    tau22 += 8 * np.einsum(
        "jb,jiba->ia", tau9, u[o, o, v, v], optimize=True
    )

    tau9 = None

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum(
        "caki,bcjk->ijab", l2, t2, optimize=True
    )

    tau22 += 8 * np.einsum(
        "ijbc,jbca->ia", tau10, u[o, v, v, v], optimize=True
    )

    tau10 = None

    tau11 = zeros((N, N, N, M))

    tau11 += np.einsum(
        "kjia->ijka", u[o, o, o, v], optimize=True
    )

    tau11 -= np.einsum(
        "bi,kjab->ijka", t1, u[o, o, v, v], optimize=True
    )

    tau16 = zeros((N, N, N, M))

    tau16 += 8 * np.einsum(
        "balj,klib->ijka", t2, tau11, optimize=True
    )

    tau11 = None

    tau12 = zeros((N, M, M, M))

    tau12 -= 2 * np.einsum(
        "iacb->iabc", u[o, v, v, v], optimize=True
    )

    tau12 += np.einsum(
        "aj,ijcb->iabc", t1, u[o, o, v, v], optimize=True
    )

    tau16 += np.einsum(
        "bckj,iabc->ijka", t2, tau12, optimize=True
    )

    tau12 = None

    tau13 = zeros((N, M))

    tau13 += np.einsum(
        "ia->ia", f[o, v], optimize=True
    )

    tau13 += np.einsum(
        "bj,jiba->ia", t1, u[o, o, v, v], optimize=True
    )

    tau16 += 4 * np.einsum(
        "ib,bakj->ijka", tau13, t2, optimize=True
    )

    tau13 = None

    tau14 = zeros((N, N, M, M))

    tau14 += 2 * np.einsum(
        "iajb->ijab", u[o, v, o, v], optimize=True
    )

    tau14 -= np.einsum(
        "cj,iabc->ijab", t1, u[o, v, v, v], optimize=True
    )

    tau16 -= 4 * np.einsum(
        "bk,ijab->ijka", t1, tau14, optimize=True
    )

    tau14 = None

    tau15 = zeros((N, N, N, N))

    tau15 -= np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau15 -= 2 * np.einsum(
        "ak,jila->ijkl", t1, u[o, o, o, v], optimize=True
    )

    tau16 -= 2 * np.einsum(
        "al,likj->ijka", t1, tau15, optimize=True
    )

    tau15 = None

    tau16 += 4 * np.einsum(
        "iakj->ijka", u[o, v, o, o], optimize=True
    )

    tau22 -= np.einsum(
        "bajk,ijkb->ia", l2, tau16, optimize=True
    )

    tau16 = None

    tau17 = zeros((N, M, M, M))

    tau17 -= 2 * np.einsum(
        "baic->iabc", u[v, v, o, v], optimize=True
    )

    tau17 -= np.einsum(
        "di,badc->iabc", t1, u[v, v, v, v], optimize=True
    )

    tau22 += 2 * np.einsum(
        "bcji,jbca->ia", l2, tau17, optimize=True
    )

    tau17 = None

    tau18 = zeros((N, N, M, M))

    tau18 -= np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau18 += 2 * np.einsum(
        "aj,bi->ijab", t1, t1, optimize=True
    )

    tau19 = zeros((N, N, N, N))

    tau19 += np.einsum(
        "abji,lkab->ijkl", l2, tau18, optimize=True
    )

    tau18 = None

    tau20 = zeros((N, N, N, M))

    tau20 -= np.einsum(
        "al,likj->ijka", t1, tau19, optimize=True
    )

    tau19 = None

    tau20 -= 4 * np.einsum(
        "bi,abkj->ijka", l1, t2, optimize=True
    )

    tau22 -= np.einsum(
        "ijkb,jkba->ia", tau20, u[o, o, v, v], optimize=True
    )

    tau20 = None

    tau21 += 2 * np.einsum(
        "ai,bi->ab", l1, t1, optimize=True
    )

    tau22 -= 4 * np.einsum(
        "bc,ibca->ia", tau21, u[o, v, v, v], optimize=True
    )

    tau21 = None

    tau22 += 8 * np.einsum(
        "bi,ba->ia", l1, f[v, v], optimize=True
    )

    tau22 -= 8 * np.einsum(
        "bj,ibja->ia", l1, u[o, v, o, v], optimize=True
    )

    e += np.einsum(
        "ai,ia->", t1, tau22, optimize=True
    ) / 8

    tau22 = None

    tau23 = zeros((N, M))

    tau23 += 2 * np.einsum(
        "ai->ia", f[v, o], optimize=True
    )

    tau23 += 2 * np.einsum(
        "jb,baji->ia", f[o, v], t2, optimize=True
    )

    tau23 += np.einsum(
        "bakj,kjib->ia", t2, u[o, o, o, v], optimize=True
    )

    tau23 += np.einsum(
        "cbji,jacb->ia", t2, u[o, v, v, v], optimize=True
    )

    e += np.einsum(
        "ai,ia->", l1, tau23, optimize=True
    ) / 2

    tau23 = None

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau25 += 2 * np.einsum(
        "ai,bj->ijab", t1, t1, optimize=True
    )

    tau26 = zeros((N, N))

    tau26 += np.einsum(
        "kjab,kiab->ij", tau25, u[o, o, v, v], optimize=True
    )

    tau25 = None

    tau26 += 2 * np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    tau26 -= 2 * np.einsum(
        "ak,kija->ij", t1, u[o, o, o, v], optimize=True
    )

    e -= np.einsum(
        "ij,ji->", tau24, tau26, optimize=True
    ) / 4

    tau24 = None

    tau26 = None

    return e