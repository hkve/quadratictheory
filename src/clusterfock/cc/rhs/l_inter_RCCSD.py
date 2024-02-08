import numpy as np

def lambda_amplitudes_intermediates_rccsd(t1, t2, l1, l2, u, f, v, o):
    rhs1 = lambda_amplitudes_intermediates_rccsd_l1(t1, t2, l1, l2, u, f, v, o)
    rhs2 = lambda_amplitudes_intermediates_rccsd_l2(t1, t2, l1, l2, u, f, v, o)

    return rhs1, rhs2

def lambda_amplitudes_intermediates_rccsd_l1(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, M))

    tau0 += np.einsum(
        "ib,abjk->ijka", f[o, v], t2, optimize=True
    )

    tau15 = zeros((N, N, N, M))

    tau15 -= 2 * np.einsum(
        "jika->ijka", tau0, optimize=True
    )

    tau15 += np.einsum(
        "jkia->ijka", tau0, optimize=True
    )

    tau0 = None

    tau1 = zeros((N, M, M, M))

    tau1 += np.einsum(
        "aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True
    )

    tau2 = zeros((N, M, M, M))

    tau2 -= np.einsum(
        "iabc->iabc", tau1, optimize=True
    )

    tau1 = None

    tau2 += np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau3 = zeros((N, N, N, M))

    tau3 += np.einsum(
        "bcij,kabc->ijka", t2, tau2, optimize=True
    )

    tau2 = None

    tau15 += np.einsum(
        "ikja->ijka", tau3, optimize=True
    )

    tau15 -= 2 * np.einsum(
        "kija->ijka", tau3, optimize=True
    )

    tau3 = None

    tau4 = zeros((N, N, M, M))

    tau4 += 2 * np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau4 -= np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau5 = zeros((N, M))

    tau5 += np.einsum(
        "bj,jiab->ia", t1, tau4, optimize=True
    )

    tau6 = zeros((N, N, N, M))

    tau6 += np.einsum(
        "kb,abij->ijka", tau5, t2, optimize=True
    )

    tau5 = None

    tau15 -= 2 * np.einsum(
        "ikja->ijka", tau6, optimize=True
    )

    tau15 += np.einsum(
        "kija->ijka", tau6, optimize=True
    )

    tau6 = None

    tau7 = zeros((N, N, N, M))

    tau7 += np.einsum(
        "bi,kjba->ijka", t1, u[o, o, v, v], optimize=True
    )

    tau8 = zeros((N, N, N, M))

    tau8 -= np.einsum(
        "ijka->ijka", tau7, optimize=True
    )

    tau8 += 2 * np.einsum(
        "ikja->ijka", tau7, optimize=True
    )

    tau9 = zeros((N, N, N, M))

    tau9 += 2 * np.einsum(
        "ijka->ijka", tau7, optimize=True
    )

    tau9 -= np.einsum(
        "ikja->ijka", tau7, optimize=True
    )

    tau12 = zeros((N, N, N, N))

    tau12 += np.einsum(
        "aj,ilka->ijkl", t1, tau7, optimize=True
    )

    tau7 = None

    tau14 = zeros((N, N, N, N))

    tau14 += np.einsum(
        "ijkl->ijkl", tau12, optimize=True
    )

    tau23 = zeros((N, N, N, N))

    tau23 += 2 * np.einsum(
        "ijkl->ijkl", tau12, optimize=True
    )

    tau12 = None

    tau8 += 2 * np.einsum(
        "jkia->ijka", u[o, o, o, v], optimize=True
    )

    tau8 -= np.einsum(
        "kjia->ijka", u[o, o, o, v], optimize=True
    )

    tau15 += np.einsum(
        "abkl,ijlb->ijka", t2, tau8, optimize=True
    )

    tau8 = None

    tau9 -= np.einsum(
        "jkia->ijka", u[o, o, o, v], optimize=True
    )

    tau9 += 2 * np.einsum(
        "kjia->ijka", u[o, o, o, v], optimize=True
    )

    tau15 += np.einsum(
        "ablk,ijlb->ijka", t2, tau9, optimize=True
    )

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum(
        "ci,jabc->ijab", t1, u[o, v, v, v], optimize=True
    )

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum(
        "jiab->ijab", tau10, optimize=True
    )

    tau22 = zeros((N, N, M, M))

    tau22 += 2 * np.einsum(
        "jiab->ijab", tau10, optimize=True
    )

    tau10 = None

    tau11 += np.einsum(
        "iabj->ijab", u[o, v, v, o], optimize=True
    )

    tau11 -= 2 * np.einsum(
        "iajb->ijab", u[o, v, o, v], optimize=True
    )

    tau15 += np.einsum(
        "bi,jkab->ijka", t1, tau11, optimize=True
    )

    tau11 = None

    tau13 = zeros((N, N, N, N))

    tau13 += np.einsum(
        "ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True
    )

    tau14 -= 2 * np.einsum(
        "iklj->ijkl", tau13, optimize=True
    )

    tau14 += np.einsum(
        "ilkj->ijkl", tau13, optimize=True
    )

    tau23 -= np.einsum(
        "iklj->ijkl", tau13, optimize=True
    )

    tau23 += 2 * np.einsum(
        "ilkj->ijkl", tau13, optimize=True
    )

    tau13 = None

    tau24 = zeros((N, N, N, M))

    tau24 += np.einsum(
        "al,ikjl->ijka", t1, tau23, optimize=True
    )

    tau23 = None

    tau14 -= 2 * np.einsum(
        "lkij->ijkl", u[o, o, o, o], optimize=True
    )

    tau14 += np.einsum(
        "lkji->ijkl", u[o, o, o, o], optimize=True
    )

    tau15 -= np.einsum(
        "al,ikjl->ijka", t1, tau14, optimize=True
    )

    tau14 = None

    tau15 += np.einsum(
        "jaik->ijka", u[o, v, o, o], optimize=True
    )

    tau15 -= 2 * np.einsum(
        "jaki->ijka", u[o, v, o, o], optimize=True
    )

    r1 = zeros((M, N))

    r1 += 2 * np.einsum(
        "abjk,kijb->ai", l2, tau15, optimize=True
    )

    tau15 = None

    tau16 = zeros((N, M, M, M))

    tau16 += 2 * np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau16 -= np.einsum(
        "iacb->iabc", u[o, v, v, v], optimize=True
    )

    tau17 = zeros((N, N, M, M))

    tau17 += 2 * np.einsum(
        "abij->ijab", l2, optimize=True
    )

    tau17 -= np.einsum(
        "baij->ijab", l2, optimize=True
    )

    tau18 = zeros((N, N, M, M))

    tau18 += np.einsum(
        "cbkj,ikac->ijab", t2, tau17, optimize=True
    )

    tau40 = zeros((N, N, N, M))

    tau40 += np.einsum(
        "bi,jkab->ijka", t1, tau17, optimize=True
    )

    tau17 = None

    tau41 = zeros((N, N, M, M))

    tau41 += np.einsum(
        "bk,jika->ijab", t1, tau40, optimize=True
    )

    r1 -= 2 * np.einsum(
        "ijbc,jbca->ai", tau41, u[o, v, v, v], optimize=True
    )

    tau41 = None

    r1 -= 2 * np.einsum(
        "jikb,jbka->ai", tau40, u[o, v, o, v], optimize=True
    )

    tau40 = None

    tau18 += np.einsum(
        "ai,bj->ijab", l1, t1, optimize=True
    )

    tau18 -= np.einsum(
        "acik,cbjk->ijab", l2, t2, optimize=True
    )

    r1 += 2 * np.einsum(
        "jbca,ijbc->ai", tau16, tau18, optimize=True
    )

    tau18 = None

    tau19 = zeros((N, M, M, M))

    tau19 += np.einsum(
        "cj,abij->iabc", t1, l2, optimize=True
    )

    tau20 = zeros((M, M, M, M))

    tau20 -= np.einsum(
        "bacd->abcd", u[v, v, v, v], optimize=True
    )

    tau20 += 2 * np.einsum(
        "badc->abcd", u[v, v, v, v], optimize=True
    )

    r1 += 2 * np.einsum(
        "ibcd,bcad->ai", tau19, tau20, optimize=True
    )

    tau19 = None

    tau20 = None

    tau21 = zeros((N, N, M, M))

    tau21 += 2 * np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau21 -= np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau24 -= np.einsum(
        "klab,iljb->ijka", tau21, tau9, optimize=True
    )

    tau9 = None

    tau48 = zeros((M, M))

    tau48 += np.einsum(
        "caij,ijcb->ab", l2, tau21, optimize=True
    )

    tau22 += 2 * np.einsum(
        "iabj->ijab", u[o, v, v, o], optimize=True
    )

    tau22 -= np.einsum(
        "iajb->ijab", u[o, v, o, v], optimize=True
    )

    tau24 -= np.einsum(
        "bi,jkab->ijka", t1, tau22, optimize=True
    )

    tau22 = None

    r1 += 2 * np.einsum(
        "abjk,jikb->ai", l2, tau24, optimize=True
    )

    tau24 = None

    tau25 = zeros((N, N, N, M))

    tau25 += np.einsum(
        "bi,bajk->ijka", l1, t2, optimize=True
    )

    tau34 = zeros((N, N, N, M))

    tau34 += 2 * np.einsum(
        "ijka->ijka", tau25, optimize=True
    )

    tau39 = zeros((N, N, N, M))

    tau39 += np.einsum(
        "ijka->ijka", tau25, optimize=True
    )

    tau25 = None

    tau26 = zeros((N, N, N, N))

    tau26 += np.einsum(
        "baij,bakl->ijkl", l2, t2, optimize=True
    )

    tau34 += np.einsum(
        "al,ilkj->ijka", t1, tau26, optimize=True
    )

    tau38 = zeros((N, N, N, N))

    tau38 += np.einsum(
        "ijkl->ijkl", tau26, optimize=True
    )

    tau27 = zeros((N, N, N, M))

    tau27 += np.einsum(
        "bk,baij->ijka", t1, l2, optimize=True
    )

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum(
        "ai,bj->ijab", t1, t1, optimize=True
    )

    tau28 -= 2 * np.einsum(
        "baij->ijab", t2, optimize=True
    )

    tau28 += np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau34 += np.einsum(
        "ilkb,ljab->ijka", tau27, tau28, optimize=True
    )

    tau28 = None

    tau29 = zeros((N, N, N, M))

    tau29 += np.einsum(
        "bk,abij->ijka", t1, l2, optimize=True
    )

    tau37 = zeros((N, N, N, N))

    tau37 += np.einsum(
        "ak,ijla->ijkl", t1, tau29, optimize=True
    )

    tau38 += np.einsum(
        "ijkl->ijkl", tau37, optimize=True
    )

    tau39 += 2 * np.einsum(
        "al,ilkj->ijka", t1, tau38, optimize=True
    )

    tau38 = None

    tau30 = zeros((N, N, M, M))

    tau30 -= np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau30 += 2 * np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau32 = zeros((N, N))

    tau32 += np.einsum(
        "abik,kjab->ij", l2, tau30, optimize=True
    )

    tau33 = zeros((N, N))

    tau33 += np.einsum(
        "ij->ij", tau32, optimize=True
    )

    tau32 = None

    tau34 -= np.einsum(
        "ilkb,ljba->ijka", tau29, tau30, optimize=True
    )

    tau29 = None

    tau51 = zeros((N, M))

    tau51 += np.einsum(
        "jkib,jkba->ia", tau27, tau30, optimize=True
    )

    tau27 = None

    tau51 -= np.einsum(
        "bj,ijba->ia", l1, tau30, optimize=True
    )

    tau30 = None

    tau31 = zeros((N, N))

    tau31 += np.einsum(
        "ai,aj->ij", l1, t1, optimize=True
    )

    tau33 += np.einsum(
        "ij->ij", tau31, optimize=True
    )

    tau34 -= np.einsum(
        "aj,ik->ijka", t1, tau33, optimize=True
    )

    r1 -= 2 * np.einsum(
        "ijkb,jkab->ai", tau34, u[o, o, v, v], optimize=True
    )

    tau34 = None

    tau39 -= 2 * np.einsum(
        "aj,ik->ijka", t1, tau33, optimize=True
    )

    r1 -= 2 * np.einsum(
        "ja,ij->ai", f[o, v], tau33, optimize=True
    )

    tau33 = None

    tau50 = zeros((N, N))

    tau50 += np.einsum(
        "ij->ij", tau31, optimize=True
    )

    tau53 = zeros((N, N))

    tau53 += np.einsum(
        "ij->ij", tau31, optimize=True
    )

    tau31 = None

    tau35 = zeros((N, N, M, M))

    tau35 -= np.einsum(
        "abij->ijab", l2, optimize=True
    )

    tau35 += 2 * np.einsum(
        "baij->ijab", l2, optimize=True
    )

    tau36 = zeros((N, N, N, M))

    tau36 += np.einsum(
        "bi,jkab->ijka", t1, tau35, optimize=True
    )

    tau35 = None

    tau39 -= np.einsum(
        "ljba,kilb->ijka", tau21, tau36, optimize=True
    )

    tau21 = None

    r1 += 2 * np.einsum(
        "ijkb,jkba->ai", tau39, u[o, o, v, v], optimize=True
    )

    tau39 = None

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum(
        "bk,jika->ijab", t1, tau36, optimize=True
    )

    r1 -= 2 * np.einsum(
        "ijbc,jbac->ai", tau42, u[o, v, v, v], optimize=True
    )

    tau42 = None

    r1 -= 2 * np.einsum(
        "jikb,jbak->ai", tau36, u[o, v, v, o], optimize=True
    )

    tau36 = None

    tau43 = zeros((N, N, M, M))

    tau43 += np.einsum(
        "caik,cbjk->ijab", l2, t2, optimize=True
    )

    tau44 = zeros((N, M, M, M))

    tau44 -= np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau44 += 2 * np.einsum(
        "iacb->iabc", u[o, v, v, v], optimize=True
    )

    r1 -= 2 * np.einsum(
        "ijbc,jbca->ai", tau43, tau44, optimize=True
    )

    tau43 = None

    tau44 = None

    tau45 = zeros((N, M, M, M))

    tau45 += 2 * np.einsum(
        "abic->iabc", u[v, v, o, v], optimize=True
    )

    tau45 -= np.einsum(
        "baic->iabc", u[v, v, o, v], optimize=True
    )

    r1 += 2 * np.einsum(
        "bcij,jcba->ai", l2, tau45, optimize=True
    )

    tau45 = None

    tau46 = zeros((N, N, N, M))

    tau46 += 2 * np.einsum(
        "ijka->ijka", u[o, o, o, v], optimize=True
    )

    tau46 -= np.einsum(
        "jika->ijka", u[o, o, o, v], optimize=True
    )

    tau57 = zeros((N, N))

    tau57 += np.einsum(
        "ak,ikja->ij", t1, tau46, optimize=True
    )

    r1 += 2 * np.einsum(
        "ijkl,lkja->ai", tau37, tau46, optimize=True
    )

    tau37 = None

    tau46 = None

    tau47 = zeros((N, N, N, M))

    tau47 -= np.einsum(
        "ijka->ijka", u[o, o, o, v], optimize=True
    )

    tau47 += 2 * np.einsum(
        "jika->ijka", u[o, o, o, v], optimize=True
    )

    r1 += 2 * np.einsum(
        "ijkl,klja->ai", tau26, tau47, optimize=True
    )

    tau26 = None

    tau48 += np.einsum(
        "ai,bi->ab", l1, t1, optimize=True
    )

    r1 += 2 * np.einsum(
        "bc,ibac->ai", tau48, tau16, optimize=True
    )

    tau16 = None

    tau48 = None

    tau49 = zeros((N, N, M, M))

    tau49 -= np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau49 += 2 * np.einsum(
        "baij->ijab", t2, optimize=True
    )

    tau50 += np.einsum(
        "abik,kjab->ij", l2, tau49, optimize=True
    )

    tau49 = None

    tau51 += np.einsum(
        "aj,ji->ia", t1, tau50, optimize=True
    )

    tau50 = None

    tau51 -= np.einsum(
        "ai->ia", t1, optimize=True
    )

    r1 -= 2 * np.einsum(
        "jb,jiab->ai", tau51, tau4, optimize=True
    )

    tau51 = None

    tau4 = None

    tau52 = zeros((N, N, M, M))

    tau52 += 2 * np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau52 -= np.einsum(
        "baij->ijab", t2, optimize=True
    )

    tau53 += np.einsum(
        "abki,kjab->ij", l2, tau52, optimize=True
    )

    tau52 = None

    r1 -= 2 * np.einsum(
        "kj,ijka->ai", tau53, tau47, optimize=True
    )

    tau53 = None

    tau47 = None

    tau54 = zeros((N, N, M, M))

    tau54 += 2 * np.einsum(
        "iabj->ijab", u[o, v, v, o], optimize=True
    )

    tau54 -= np.einsum(
        "iajb->ijab", u[o, v, o, v], optimize=True
    )

    r1 += 2 * np.einsum(
        "bj,ijba->ai", l1, tau54, optimize=True
    )

    tau54 = None

    tau55 = zeros((N, N, M, M))

    tau55 += np.einsum(
        "ai,bj->ijab", t1, t1, optimize=True
    )

    tau55 += np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau55 -= 2 * np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau57 -= np.einsum(
        "kjba,kiab->ij", tau55, u[o, o, v, v], optimize=True
    )

    tau55 = None

    tau56 = zeros((N, M))

    tau56 += np.einsum(
        "ia->ia", f[o, v], optimize=True
    )

    tau56 += 2 * np.einsum(
        "bj,ijab->ia", t1, u[o, o, v, v], optimize=True
    )

    tau57 += np.einsum(
        "aj,ia->ij", t1, tau56, optimize=True
    )

    tau56 = None

    tau57 += np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    r1 -= 2 * np.einsum(
        "aj,ij->ai", l1, tau57, optimize=True
    )

    tau57 = None

    r1 += 2 * np.einsum(
        "bi,ba->ai", l1, f[v, v], optimize=True
    )

    r1 += 2 * np.einsum(
        "ia->ai", f[o, v], optimize=True
    )

    return r1

def lambda_amplitudes_intermediates_rccsd_l2(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum(
        "ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True
    )

    tau24 = zeros((N, N, M, M))

    tau24 -= np.einsum(
        "ijab->ijab", tau0, optimize=True
    )

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 += 2 * np.einsum(
        "abij->ijab", l2, optimize=True
    )

    tau1 -= np.einsum(
        "abji->ijab", l2, optimize=True
    )

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum(
        "ak,kijb->ijab", t1, u[o, o, o, v], optimize=True
    )

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum(
        "jiab->ijab", tau2, optimize=True
    )

    tau2 = None

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum(
        "ci,jabc->ijab", t1, u[o, v, v, v], optimize=True
    )

    tau8 -= np.einsum(
        "ijab->ijab", tau3, optimize=True
    )

    tau3 = None

    tau4 = zeros((N, N, M, M))

    tau4 -= np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau4 += 2 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum(
        "acik,kjcb->ijab", t2, tau4, optimize=True
    )

    tau8 -= np.einsum(
        "ijab->ijab", tau5, optimize=True
    )

    tau5 = None

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau6 += np.einsum(
        "ai,bj->ijab", t1, t1, optimize=True
    )

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum(
        "kjbc,kica->ijab", tau6, u[o, o, v, v], optimize=True
    )

    tau8 += np.einsum(
        "jiba->ijab", tau7, optimize=True
    )

    tau7 = None

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum(
        "kjbc,kiac->ijab", tau6, u[o, o, v, v], optimize=True
    )

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum(
        "jiba->ijab", tau12, optimize=True
    )

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum(
        "jiba->ijab", tau12, optimize=True
    )

    tau12 = None

    tau16 = zeros((M, M))

    tau16 += np.einsum(
        "ijca,jibc->ab", tau4, tau6, optimize=True
    )

    tau4 = None

    tau19 = zeros((M, M))

    tau19 += np.einsum(
        "ba->ab", tau16, optimize=True
    )

    tau16 = None

    tau50 = zeros((N, N, N, N))

    tau50 += np.einsum(
        "abkl,ijab->ijkl", l2, tau6, optimize=True
    )

    tau6 = None

    tau51 = zeros((N, N, M, M))

    tau51 += np.einsum(
        "lkij,klab->ijab", tau50, u[o, o, v, v], optimize=True
    )

    tau50 = None

    tau52 = zeros((N, N, M, M))

    tau52 += np.einsum(
        "ijba->ijab", tau51, optimize=True
    )

    tau51 = None

    tau8 -= np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum(
        "ikac,kjcb->ijab", tau1, tau8, optimize=True
    )

    tau8 = None

    tau24 -= np.einsum(
        "ijab->ijab", tau9, optimize=True
    )

    tau9 = None

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum(
        "ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True
    )

    tau13 += np.einsum(
        "jiab->ijab", tau10, optimize=True
    )

    tau28 += np.einsum(
        "jiab->ijab", tau10, optimize=True
    )

    tau10 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum(
        "ci,jacb->ijab", t1, u[o, v, v, v], optimize=True
    )

    tau13 -= np.einsum(
        "ijab->ijab", tau11, optimize=True
    )

    tau28 -= np.einsum(
        "ijab->ijab", tau11, optimize=True
    )

    tau11 = None

    tau29 = zeros((N, N, M, M))

    tau29 += np.einsum(
        "bckj,kica->ijab", l2, tau28, optimize=True
    )

    tau28 = None

    tau35 = zeros((N, N, M, M))

    tau35 += np.einsum(
        "jiba->ijab", tau29, optimize=True
    )

    tau29 = None

    tau13 -= np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum(
        "bcjk,kica->ijab", l2, tau13, optimize=True
    )

    tau13 = None

    tau24 += np.einsum(
        "jiba->ijab", tau14, optimize=True
    )

    tau14 = None

    tau15 = zeros((M, M))

    tau15 += np.einsum(
        "ia,bi->ab", f[o, v], t1, optimize=True
    )

    tau19 += np.einsum(
        "ba->ab", tau15, optimize=True
    )

    tau15 = None

    tau17 = zeros((N, M, M, M))

    tau17 -= np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau17 += 2 * np.einsum(
        "iacb->iabc", u[o, v, v, v], optimize=True
    )

    tau18 = zeros((M, M))

    tau18 += np.einsum(
        "ci,iabc->ab", t1, tau17, optimize=True
    )

    tau17 = None

    tau19 -= np.einsum(
        "ab->ab", tau18, optimize=True
    )

    tau18 = None

    tau19 -= np.einsum(
        "ab->ab", f[v, v], optimize=True
    )

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum(
        "ca,bcij->ijab", tau19, l2, optimize=True
    )

    tau19 = None

    tau24 -= np.einsum(
        "ijba->ijab", tau20, optimize=True
    )

    tau20 = None

    tau21 = zeros((N, N, M, M))

    tau21 += 2 * np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau21 -= np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau22 = zeros((N, M))

    tau22 += np.einsum(
        "bj,jiab->ia", t1, tau21, optimize=True
    )

    tau21 = None

    tau23 = zeros((N, M))

    tau23 += np.einsum(
        "ia->ia", tau22, optimize=True
    )

    tau39 = zeros((N, N))

    tau39 += np.einsum(
        "aj,ia->ij", t1, tau22, optimize=True
    )

    tau22 = None

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum(
        "ik,abkj->ijab", tau39, l2, optimize=True
    )

    tau41 = zeros((N, N, M, M))

    tau41 -= np.einsum(
        "jiab->ijab", tau40, optimize=True
    )

    tau40 = None

    tau46 = zeros((N, N, M, M))

    tau46 += np.einsum(
        "ik,abjk->ijab", tau39, l2, optimize=True
    )

    tau39 = None

    tau47 = zeros((N, N, M, M))

    tau47 -= np.einsum(
        "jiab->ijab", tau46, optimize=True
    )

    tau46 = None

    tau23 += np.einsum(
        "ia->ia", f[o, v], optimize=True
    )

    tau24 += np.einsum(
        "ai,jb->ijab", l1, tau23, optimize=True
    )

    tau23 = None

    r2 = zeros((M, M, N, N))

    r2 += 4 * np.einsum(
        "ijab->abij", tau24, optimize=True
    )

    r2 -= 2 * np.einsum(
        "ijba->abij", tau24, optimize=True
    )

    r2 -= 2 * np.einsum(
        "jiab->abij", tau24, optimize=True
    )

    r2 += 4 * np.einsum(
        "jiba->abij", tau24, optimize=True
    )

    tau24 = None

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum(
        "ci,jcab->ijab", l1, u[o, v, v, v], optimize=True
    )

    tau35 += np.einsum(
        "ijab->ijab", tau25, optimize=True
    )

    tau25 = None

    tau26 = zeros((N, N))

    tau26 += np.einsum(
        "ai,aj->ij", l1, t1, optimize=True
    )

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum(
        "ik,jkab->ijab", tau26, u[o, o, v, v], optimize=True
    )

    tau26 = None

    tau35 -= np.einsum(
        "ijab->ijab", tau27, optimize=True
    )

    tau27 = None

    tau30 = zeros((M, M))

    tau30 += np.einsum(
        "ai,bi->ab", l1, t1, optimize=True
    )

    tau33 = zeros((M, M))

    tau33 += np.einsum(
        "ab->ab", tau30, optimize=True
    )

    tau30 = None

    tau31 = zeros((N, N, M, M))

    tau31 -= np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau31 += 2 * np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau32 = zeros((M, M))

    tau32 += np.einsum(
        "bcij,ijca->ab", l2, tau31, optimize=True
    )

    tau33 += np.einsum(
        "ba->ab", tau32, optimize=True
    )

    tau32 = None

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum(
        "bc,ijca->ijab", tau33, u[o, o, v, v], optimize=True
    )

    tau33 = None

    tau35 -= np.einsum(
        "jiba->ijab", tau34, optimize=True
    )

    tau34 = None

    r2 -= 2 * np.einsum(
        "ijab->abij", tau35, optimize=True
    )

    r2 += 4 * np.einsum(
        "ijba->abij", tau35, optimize=True
    )

    r2 += 4 * np.einsum(
        "jiab->abij", tau35, optimize=True
    )

    r2 -= 2 * np.einsum(
        "jiba->abij", tau35, optimize=True
    )

    tau35 = None

    tau55 = zeros((N, N))

    tau55 += np.einsum(
        "abjk,kiab->ij", l2, tau31, optimize=True
    )

    tau56 = zeros((N, N, M, M))

    tau56 += np.einsum(
        "kj,kiab->ijab", tau55, u[o, o, v, v], optimize=True
    )

    tau55 = None

    tau57 = zeros((N, N, M, M))

    tau57 += np.einsum(
        "jiba->ijab", tau56, optimize=True
    )

    tau56 = None

    tau68 = zeros((N, N))

    tau68 += np.einsum(
        "kjba,kiab->ij", tau31, u[o, o, v, v], optimize=True
    )

    tau31 = None

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum(
        "cdij,dcba->ijab", l2, u[v, v, v, v], optimize=True
    )

    r2 += 7 * np.einsum(
        "ijab->abij", tau36, optimize=True
    ) / 2

    r2 -= 2 * np.einsum(
        "ijba->abij", tau36, optimize=True
    )

    tau36 = None

    tau37 = zeros((N, N, N, N))

    tau37 += np.einsum(
        "ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True
    )

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum(
        "abkl,kijl->ijab", l2, tau37, optimize=True
    )

    tau41 += np.einsum(
        "ijab->ijab", tau38, optimize=True
    )

    tau38 = None

    r2 -= 2 * np.einsum(
        "ijab->abij", tau41, optimize=True
    )

    r2 += 4 * np.einsum(
        "jiab->abij", tau41, optimize=True
    )

    tau41 = None

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum(
        "ablk,kijl->ijab", l2, tau37, optimize=True
    )

    tau37 = None

    tau47 += np.einsum(
        "ijab->ijab", tau42, optimize=True
    )

    tau42 = None

    tau43 = zeros((N, N, N, M))

    tau43 += np.einsum(
        "bi,jkab->ijka", t1, u[o, o, v, v], optimize=True
    )

    tau44 = zeros((N, N, N, N))

    tau44 += np.einsum(
        "ai,jkla->ijkl", t1, tau43, optimize=True
    )

    tau43 = None

    tau45 = zeros((N, N, M, M))

    tau45 += np.einsum(
        "abkl,klij->ijab", l2, tau44, optimize=True
    )

    tau44 = None

    tau47 += np.einsum(
        "ijab->ijab", tau45, optimize=True
    )

    tau45 = None

    r2 += 4 * np.einsum(
        "ijab->abij", tau47, optimize=True
    )

    r2 -= 2 * np.einsum(
        "jiab->abij", tau47, optimize=True
    )

    tau47 = None

    tau48 = zeros((N, N, N, M))

    tau48 += np.einsum(
        "bk,baij->ijka", t1, l2, optimize=True
    )

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum(
        "ijkc,kcab->ijab", tau48, u[o, v, v, v], optimize=True
    )

    tau48 = None

    tau52 -= np.einsum(
        "ijab->ijab", tau49, optimize=True
    )

    tau49 = None

    r2 += 4 * np.einsum(
        "ijab->abij", tau52, optimize=True
    )

    r2 -= 2 * np.einsum(
        "ijba->abij", tau52, optimize=True
    )

    tau52 = None

    tau53 = zeros((N, N, N, M))

    tau53 += np.einsum(
        "bk,abij->ijka", t1, l2, optimize=True
    )

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum(
        "ijkc,kcab->ijab", tau53, u[o, v, v, v], optimize=True
    )

    tau53 = None

    tau57 += np.einsum(
        "ijab->ijab", tau54, optimize=True
    )

    tau54 = None

    r2 += 2 * np.einsum(
        "ijab->abij", tau57, optimize=True
    )

    r2 -= 4 * np.einsum(
        "ijba->abij", tau57, optimize=True
    )

    tau57 = None

    tau58 = zeros((N, N, M, M))

    tau58 += np.einsum(
        "acki,jckb->ijab", l2, u[o, v, o, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "ijab->abij", tau58, optimize=True
    )

    r2 -= 4 * np.einsum(
        "ijba->abij", tau58, optimize=True
    )

    r2 -= 3 * np.einsum(
        "jiab->abij", tau58, optimize=True
    )

    r2 += 2 * np.einsum(
        "jiba->abij", tau58, optimize=True
    )

    tau58 = None

    tau59 = zeros((N, N, M, M))

    tau59 += 2 * np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau59 -= np.einsum(
        "abji->ijab", t2, optimize=True
    )

    tau60 = zeros((N, N))

    tau60 += np.einsum(
        "abkj,kiab->ij", l2, tau59, optimize=True
    )

    tau61 = zeros((N, N, M, M))

    tau61 += np.einsum(
        "kj,kiab->ijab", tau60, u[o, o, v, v], optimize=True
    )

    tau60 = None

    r2 -= 4 * np.einsum(
        "ijba->abij", tau61, optimize=True
    )

    r2 += 2 * np.einsum(
        "ijab->abij", tau61, optimize=True
    )

    tau61 = None

    tau67 = zeros((N, N))

    tau67 += np.einsum(
        "kjab,kiab->ij", tau59, u[o, o, v, v], optimize=True
    )

    tau59 = None

    tau62 = zeros((N, N, N, N))

    tau62 += np.einsum(
        "abij,klab->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau63 = zeros((N, N, N, N))

    tau63 += 2 * np.einsum(
        "klji->ijkl", tau62, optimize=True
    )

    tau63 -= np.einsum(
        "lkji->ijkl", tau62, optimize=True
    )

    tau62 = None

    tau63 += 2 * np.einsum(
        "jikl->ijkl", u[o, o, o, o], optimize=True
    )

    tau63 -= np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    r2 += 2 * np.einsum(
        "abkl,jikl->abij", l2, tau63, optimize=True
    )

    tau63 = None

    tau64 = zeros((N, N))

    tau64 += np.einsum(
        "ia,aj->ij", f[o, v], t1, optimize=True
    )

    tau67 += np.einsum(
        "ij->ij", tau64, optimize=True
    )

    tau68 += np.einsum(
        "ij->ij", tau64, optimize=True
    )

    tau64 = None

    tau65 = zeros((N, N, N, M))

    tau65 += 2 * np.einsum(
        "ijka->ijka", u[o, o, o, v], optimize=True
    )

    tau65 -= np.einsum(
        "jika->ijka", u[o, o, o, v], optimize=True
    )

    tau66 = zeros((N, N))

    tau66 += np.einsum(
        "ak,ikja->ij", t1, tau65, optimize=True
    )

    tau65 = None

    tau67 += np.einsum(
        "ij->ij", tau66, optimize=True
    )

    tau68 += np.einsum(
        "ij->ij", tau66, optimize=True
    )

    tau66 = None

    tau67 += np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    r2 -= 2 * np.einsum(
        "jk,ikab->abij", tau67, tau1, optimize=True
    )

    tau67 = None

    tau68 += np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ik,kjab->abij", tau68, tau1, optimize=True
    )

    tau1 = None

    tau68 = None

    r2 -= 2 * np.einsum(
        "jiab->abij", u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "jiba->abij", u[o, o, v, v], optimize=True
    )

    return r2