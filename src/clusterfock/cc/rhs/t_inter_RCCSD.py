import numpy as np

def amplitudes_intermediates_rccsd(t1, t2, u, f, v, o):
    rhs1 = amplitudes_intermediates_rccsd_t1(t1, t2, u, f, v, o)
    rhs2 = amplitudes_intermediates_rccsd_t2(t1, t2, u, f, v, o)

    return rhs1, rhs2

def amplitudes_intermediates_rccsd_t1(t1, t2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M, M, M))

    tau0 -= np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau0 += 2 * np.einsum(
        "iacb->iabc", u[o, v, v, v], optimize=True
    )

    r = zeros((M, N))

    r += 2 * np.einsum(
        "bcij,jabc->ai", t2, tau0, optimize=True
    )

    tau0 = None

    tau1 = zeros((N, N, N, M))

    tau1 += np.einsum(
        "bi,jkab->ijka", t1, u[o, o, v, v], optimize=True
    )

    tau2 = zeros((N, N, N, M))

    tau2 -= np.einsum(
        "ijka->ijka", tau1, optimize=True
    )

    tau2 += 2 * np.einsum(
        "ikja->ijka", tau1, optimize=True
    )

    tau1 = None

    tau2 += 2 * np.einsum(
        "jkia->ijka", u[o, o, o, v], optimize=True
    )

    tau2 -= np.einsum(
        "kjia->ijka", u[o, o, o, v], optimize=True
    )

    r -= 2 * np.einsum(
        "abjk,ijkb->ai", t2, tau2, optimize=True
    )

    tau2 = None

    tau3 = zeros((N, N, M, M))

    tau3 -= np.einsum(
        "jiab->ijab", u[o, o, v, v], optimize=True
    )

    tau3 += 2 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau4 = zeros((N, M))

    tau4 += np.einsum(
        "bj,jiba->ia", t1, tau3, optimize=True
    )

    tau5 = zeros((N, M))

    tau5 += np.einsum(
        "ia->ia", tau4, optimize=True
    )

    tau4 = None

    tau11 = zeros((N, N))

    tau11 += np.einsum(
        "abjk,kiba->ij", t2, tau3, optimize=True
    )

    tau3 = None

    tau5 += np.einsum(
        "ia->ia", f[o, v], optimize=True
    )

    tau11 += np.einsum(
        "aj,ia->ij", t1, tau5, optimize=True
    )

    tau6 = zeros((N, N, M, M))

    tau6 += 2 * np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau6 -= np.einsum(
        "abji->ijab", t2, optimize=True
    )

    r += 2 * np.einsum(
        "jb,ijab->ai", tau5, tau6, optimize=True
    )

    tau5 = None

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += 2 * np.einsum(
        "iabj->ijab", u[o, v, v, o], optimize=True
    )

    tau7 -= np.einsum(
        "iajb->ijab", u[o, v, o, v], optimize=True
    )

    r += 2 * np.einsum(
        "bj,jiab->ai", t1, tau7, optimize=True
    )

    tau7 = None

    tau8 = zeros((N, M, M, M))

    tau8 += 2 * np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau8 -= np.einsum(
        "iacb->iabc", u[o, v, v, v], optimize=True
    )

    tau9 = zeros((M, M))

    tau9 += np.einsum(
        "ci,iacb->ab", t1, tau8, optimize=True
    )

    tau8 = None

    tau9 += np.einsum(
        "ab->ab", f[v, v], optimize=True
    )

    r += 2 * np.einsum(
        "bi,ab->ai", t1, tau9, optimize=True
    )

    tau9 = None

    tau10 = zeros((N, N, N, M))

    tau10 -= np.einsum(
        "ijka->ijka", u[o, o, o, v], optimize=True
    )

    tau10 += 2 * np.einsum(
        "jika->ijka", u[o, o, o, v], optimize=True
    )

    tau11 += np.einsum(
        "ak,kija->ij", t1, tau10, optimize=True
    )

    tau10 = None

    tau11 += np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    r -= 2 * np.einsum(
        "aj,ji->ai", t1, tau11, optimize=True
    )

    tau11 = None

    r += 2 * np.einsum(
        "ai->ai", f[v, o], optimize=True
    )

    return r


def amplitudes_intermediates_rccsd_t2(t1, t2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M, M, M))

    tau0 += np.einsum(
        "di,abcd->iabc", t1, u[v, v, v, v], optimize=True
    )

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum(
        "ci,jabc->ijab", t1, tau0, optimize=True
    )

    tau0 = None

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum(
        "ijab->ijab", tau1, optimize=True
    )

    tau1 = None

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum(
        "acki,kjcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum(
        "bckj,ikac->ijab", t2, tau2, optimize=True
    )

    tau2 = None

    tau12 += np.einsum(
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
        "bcjk,kica->ijab", t2, tau4, optimize=True
    )

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum(
        "bcjk,kica->ijab", t2, tau5, optimize=True
    )

    tau5 = None

    tau12 += 2 * np.einsum(
        "jiba->ijab", tau6, optimize=True
    )

    tau6 = None

    tau43 = zeros((M, M))

    tau43 += np.einsum(
        "bcij,ijac->ab", t2, tau4, optimize=True
    )

    tau48 = zeros((M, M))

    tau48 += np.einsum(
        "ba->ab", tau43, optimize=True
    )

    tau43 = None

    tau45 = zeros((N, M))

    tau45 += np.einsum(
        "bj,jiba->ia", t1, tau4, optimize=True
    )

    tau46 = zeros((N, M))

    tau46 += np.einsum(
        "ia->ia", tau45, optimize=True
    )

    tau45 = None

    tau58 = zeros((N, N))

    tau58 += np.einsum(
        "bajk,kiab->ij", t2, tau4, optimize=True
    )

    tau61 = zeros((N, N))

    tau61 += np.einsum(
        "ij->ij", tau58, optimize=True
    )

    tau63 = zeros((N, N))

    tau63 += np.einsum(
        "ji->ij", tau58, optimize=True
    )

    tau58 = None

    tau7 = zeros((N, N, N, M))

    tau7 += np.einsum(
        "bi,jkab->ijka", t1, u[o, o, v, v], optimize=True
    )

    tau8 = zeros((N, N, N, N))

    tau8 += np.einsum(
        "aj,ilka->ijkl", t1, tau7, optimize=True
    )

    tau9 = zeros((N, N, N, N))

    tau9 += np.einsum(
        "lkji->ijkl", tau8, optimize=True
    )

    tau8 = None

    tau16 = zeros((N, N, N, M))

    tau16 += np.einsum(
        "ijka->ijka", tau7, optimize=True
    )

    tau55 = zeros((N, N, N, M))

    tau55 += np.einsum(
        "kjia->ijka", tau7, optimize=True
    )

    tau7 = None

    tau9 += np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau10 = zeros((N, N, N, M))

    tau10 += np.einsum(
        "al,lijk->ijka", t1, tau9, optimize=True
    )

    tau9 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum(
        "bk,kjia->ijab", t1, tau10, optimize=True
    )

    tau10 = None

    tau12 += np.einsum(
        "ijba->ijab", tau11, optimize=True
    )

    tau11 = None

    tau12 += np.einsum(
        "baji->ijab", u[v, v, o, o], optimize=True
    )

    r = zeros((M, M, N, N))

    r -= 2 * np.einsum(
        "ijba->abij", tau12, optimize=True
    )

    r += 4 * np.einsum(
        "ijab->abij", tau12, optimize=True
    )

    tau12 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum(
        "ac,bcij->ijab", f[v, v], t2, optimize=True
    )

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum(
        "ijab->ijab", tau13, optimize=True
    )

    tau13 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum(
        "ci,abjc->ijab", t1, u[v, v, o, v], optimize=True
    )

    tau25 += np.einsum(
        "ijab->ijab", tau14, optimize=True
    )

    tau14 = None

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum(
        "ci,jacb->ijab", t1, u[o, v, v, v], optimize=True
    )

    tau18 = zeros((N, N, M, M))

    tau18 -= np.einsum(
        "ijab->ijab", tau15, optimize=True
    )

    tau15 = None

    tau16 += np.einsum(
        "kjia->ijka", u[o, o, o, v], optimize=True
    )

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum(
        "bk,ikja->ijab", t1, tau16, optimize=True
    )

    tau18 += np.einsum(
        "ijba->ijab", tau17, optimize=True
    )

    tau17 = None

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum(
        "bk,ijka->ijab", t1, tau16, optimize=True
    )

    tau16 = None

    tau41 = zeros((N, N, M, M))

    tau41 += np.einsum(
        "ijba->ijab", tau40, optimize=True
    )

    tau40 = None

    tau18 -= np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum(
        "bckj,ikac->ijab", t2, tau18, optimize=True
    )

    tau18 = None

    tau25 += np.einsum(
        "jiba->ijab", tau19, optimize=True
    )

    tau19 = None

    tau20 = zeros((N, N, N, M))

    tau20 += np.einsum(
        "bi,jakb->ijka", t1, u[o, v, o, v], optimize=True
    )

    tau23 = zeros((N, N, N, M))

    tau23 += np.einsum(
        "ijka->ijka", tau20, optimize=True
    )

    tau20 = None

    tau21 = zeros((N, N, N, N))

    tau21 += np.einsum(
        "ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True
    )

    tau22 = zeros((N, N, N, M))

    tau22 += np.einsum(
        "al,ijlk->ijka", t1, tau21, optimize=True
    )

    tau23 -= np.einsum(
        "ijka->ijka", tau22, optimize=True
    )

    tau22 = None

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum(
        "bk,ikja->ijab", t1, tau23, optimize=True
    )

    tau23 = None

    tau25 -= np.einsum(
        "ijba->ijab", tau24, optimize=True
    )

    tau24 = None

    r -= 2 * np.einsum(
        "ijab->abij", tau25, optimize=True
    )

    r += 4 * np.einsum(
        "ijba->abij", tau25, optimize=True
    )

    r += 4 * np.einsum(
        "jiab->abij", tau25, optimize=True
    )

    r -= 2 * np.einsum(
        "jiba->abij", tau25, optimize=True
    )

    tau25 = None

    tau67 = zeros((N, N, M, M))

    tau67 += np.einsum(
        "ablk,ilkj->ijab", t2, tau21, optimize=True
    )

    tau21 = None

    tau70 = zeros((N, N, M, M))

    tau70 -= np.einsum(
        "ijab->ijab", tau67, optimize=True
    )

    tau67 = None

    tau26 = zeros((N, M, M, M))

    tau26 += np.einsum(
        "aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True
    )

    tau27 = zeros((N, M, M, M))

    tau27 -= np.einsum(
        "iabc->iabc", tau26, optimize=True
    )

    tau26 = None

    tau27 += np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau28 = zeros((M, M, M, M))

    tau28 += np.einsum(
        "di,iabc->abcd", t1, tau27, optimize=True
    )

    tau27 = None

    tau29 = zeros((M, M, M, M))

    tau29 -= np.einsum(
        "adcb->abcd", tau28, optimize=True
    )

    tau28 = None

    tau29 += np.einsum(
        "badc->abcd", u[v, v, v, v], optimize=True
    )

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum(
        "dcij,abcd->ijab", t2, tau29, optimize=True
    )

    tau29 = None

    r += 4 * np.einsum(
        "ijba->abij", tau30, optimize=True
    )

    r -= 2 * np.einsum(
        "ijab->abij", tau30, optimize=True
    )

    tau30 = None

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum(
        "abij->ijab", t2, optimize=True
    )

    tau31 += np.einsum(
        "ai,bj->ijab", t1, t1, optimize=True
    )

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum(
        "kjbc,kica->ijab", tau31, tau4, optimize=True
    )

    tau4 = None

    tau31 = None

    tau37 = zeros((N, N, M, M))

    tau37 += np.einsum(
        "jiba->ijab", tau32, optimize=True
    )

    tau32 = None

    tau33 = zeros((N, M, M, M))

    tau33 += 2 * np.einsum(
        "iabc->iabc", u[o, v, v, v], optimize=True
    )

    tau33 -= np.einsum(
        "iacb->iabc", u[o, v, v, v], optimize=True
    )

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum(
        "cj,iabc->ijab", t1, tau33, optimize=True
    )

    tau37 -= np.einsum(
        "jiab->ijab", tau34, optimize=True
    )

    tau34 = None

    tau44 = zeros((M, M))

    tau44 += np.einsum(
        "ci,iacb->ab", t1, tau33, optimize=True
    )

    tau33 = None

    tau48 -= np.einsum(
        "ab->ab", tau44, optimize=True
    )

    tau44 = None

    tau35 = zeros((N, N, N, M))

    tau35 -= np.einsum(
        "ijka->ijka", u[o, o, o, v], optimize=True
    )

    tau35 += 2 * np.einsum(
        "jika->ijka", u[o, o, o, v], optimize=True
    )

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum(
        "bk,ikja->ijab", t1, tau35, optimize=True
    )

    tau37 += np.einsum(
        "jiba->ijab", tau36, optimize=True
    )

    tau36 = None

    tau59 = zeros((N, N))

    tau59 += np.einsum(
        "ak,kija->ij", t1, tau35, optimize=True
    )

    tau35 = None

    tau61 += np.einsum(
        "ij->ij", tau59, optimize=True
    )

    tau68 = zeros((N, N))

    tau68 += np.einsum(
        "ij->ij", tau59, optimize=True
    )

    tau59 = None

    tau37 += np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau37 -= 2 * np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum(
        "bcjk,ikac->ijab", t2, tau37, optimize=True
    )

    tau37 = None

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum(
        "jiba->ijab", tau38, optimize=True
    )

    tau38 = None

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum(
        "ci,jabc->ijab", t1, u[o, v, v, v], optimize=True
    )

    tau41 -= np.einsum(
        "ijab->ijab", tau39, optimize=True
    )

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum(
        "jiab->ijab", tau39, optimize=True
    )

    tau39 = None

    tau41 -= np.einsum(
        "jabi->ijab", u[o, v, v, o], optimize=True
    )

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum(
        "bckj,ikac->ijab", t2, tau41, optimize=True
    )

    tau41 = None

    tau54 -= np.einsum(
        "jiba->ijab", tau42, optimize=True
    )

    tau42 = None

    tau46 += np.einsum(
        "ia->ia", f[o, v], optimize=True
    )

    tau47 = zeros((M, M))

    tau47 += np.einsum(
        "bi,ia->ab", t1, tau46, optimize=True
    )

    tau48 += np.einsum(
        "ba->ab", tau47, optimize=True
    )

    tau47 = None

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum(
        "ac,bcij->ijab", tau48, t2, optimize=True
    )

    tau48 = None

    tau54 += np.einsum(
        "ijba->ijab", tau49, optimize=True
    )

    tau49 = None

    tau60 = zeros((N, N))

    tau60 += np.einsum(
        "aj,ia->ij", t1, tau46, optimize=True
    )

    tau46 = None

    tau61 += np.einsum(
        "ij->ij", tau60, optimize=True
    )

    tau62 = zeros((N, N, M, M))

    tau62 += np.einsum(
        "ki,abjk->ijab", tau61, t2, optimize=True
    )

    tau61 = None

    tau65 = zeros((N, N, M, M))

    tau65 += np.einsum(
        "jiab->ijab", tau62, optimize=True
    )

    tau62 = None

    tau68 += np.einsum(
        "ij->ij", tau60, optimize=True
    )

    tau60 = None

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum(
        "ki,abkj->ijab", tau68, t2, optimize=True
    )

    tau68 = None

    tau70 += np.einsum(
        "jiab->ijab", tau69, optimize=True
    )

    tau69 = None

    tau50 += np.einsum(
        "iabj->ijab", u[o, v, v, o], optimize=True
    )

    tau51 = zeros((N, N, N, M))

    tau51 += np.einsum(
        "bk,ijab->ijka", t1, tau50, optimize=True
    )

    tau50 = None

    tau52 = zeros((N, N, N, M))

    tau52 += np.einsum(
        "jkia->ijka", tau51, optimize=True
    )

    tau51 = None

    tau52 += np.einsum(
        "jaik->ijka", u[o, v, o, o], optimize=True
    )

    tau53 = zeros((N, N, M, M))

    tau53 += np.einsum(
        "bk,ikja->ijab", t1, tau52, optimize=True
    )

    tau52 = None

    tau54 += np.einsum(
        "ijba->ijab", tau53, optimize=True
    )

    tau53 = None

    r -= 4 * np.einsum(
        "ijab->abij", tau54, optimize=True
    )

    r += 2 * np.einsum(
        "ijba->abij", tau54, optimize=True
    )

    r += 2 * np.einsum(
        "jiab->abij", tau54, optimize=True
    )

    r -= 4 * np.einsum(
        "jiba->abij", tau54, optimize=True
    )

    tau54 = None

    tau55 += np.einsum(
        "ijka->ijka", u[o, o, o, v], optimize=True
    )

    tau56 = zeros((N, N, N, N))

    tau56 += np.einsum(
        "al,ijka->ijkl", t1, tau55, optimize=True
    )

    tau55 = None

    tau57 = zeros((N, N, M, M))

    tau57 += np.einsum(
        "ablk,klji->ijab", t2, tau56, optimize=True
    )

    tau56 = None

    tau65 -= np.einsum(
        "ijab->ijab", tau57, optimize=True
    )

    tau57 = None

    tau63 += np.einsum(
        "ji->ij", f[o, o], optimize=True
    )

    tau64 = zeros((N, N, M, M))

    tau64 += np.einsum(
        "ik,abkj->ijab", tau63, t2, optimize=True
    )

    tau63 = None

    tau65 += np.einsum(
        "ijab->ijab", tau64, optimize=True
    )

    tau64 = None

    r -= 4 * np.einsum(
        "ijab->abij", tau65, optimize=True
    )

    r += 2 * np.einsum(
        "jiab->abij", tau65, optimize=True
    )

    tau65 = None

    tau66 = zeros((N, N, M, M))

    tau66 += np.einsum(
        "ki,abjk->ijab", f[o, o], t2, optimize=True
    )

    tau70 += np.einsum(
        "ijab->ijab", tau66, optimize=True
    )

    tau66 = None

    r += 2 * np.einsum(
        "ijab->abij", tau70, optimize=True
    )

    r -= 4 * np.einsum(
        "jiab->abij", tau70, optimize=True
    )

    tau70 = None

    tau71 = zeros((N, N, N, M))

    tau71 += np.einsum(
        "cbij,kabc->ijka", t2, u[o, v, v, v], optimize=True
    )

    tau72 = zeros((N, N, M, M))

    tau72 += np.einsum(
        "ak,ijkb->ijab", t1, tau71, optimize=True
    )

    tau71 = None

    r += 2 * np.einsum(
        "ijab->abij", tau72, optimize=True
    )

    r -= 4 * np.einsum(
        "ijba->abij", tau72, optimize=True
    )

    tau72 = None

    tau73 = zeros((N, N, M, M))

    tau73 += np.einsum(
        "acki,kjbc->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau74 = zeros((N, N, M, M))

    tau74 += np.einsum(
        "bckj,ikac->ijab", t2, tau73, optimize=True
    )

    tau73 = None

    r += 4 * np.einsum(
        "jiab->abij", tau74, optimize=True
    )

    r -= 2 * np.einsum(
        "jiba->abij", tau74, optimize=True
    )

    tau74 = None

    tau75 = zeros((N, N, N, N))

    tau75 += np.einsum(
        "baij,lkab->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau76 = zeros((N, N, N, N))

    tau76 += 2 * np.einsum(
        "lkij->ijkl", tau75, optimize=True
    )

    tau76 -= np.einsum(
        "lkji->ijkl", tau75, optimize=True
    )

    tau75 = None

    tau76 += 2 * np.einsum(
        "jikl->ijkl", u[o, o, o, o], optimize=True
    )

    tau76 -= np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    r += 2 * np.einsum(
        "abkl,klji->abij", t2, tau76, optimize=True
    )

    tau76 = None

    return r
