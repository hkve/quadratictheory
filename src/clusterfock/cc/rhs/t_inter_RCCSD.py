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

    tau0 -= np.einsum("iabc->iabc", u[o, v, v, v], optimize=True)

    tau0 += 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau8 = zeros((M, M))

    tau8 += np.einsum("ci,iabc->ab", t1, tau0, optimize=True)

    r1 = zeros((M, N))

    r1 += np.einsum("bcji,jacb->ai", t2, tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, N, M))

    tau1 += np.einsum("bi,kjba->ijka", t1, u[o, o, v, v], optimize=True)

    tau2 = zeros((N, N, N, M))

    tau2 -= np.einsum("ijka->ijka", tau1, optimize=True)

    tau2 += 2 * np.einsum("ikja->ijka", tau1, optimize=True)

    tau1 = None

    tau2 += 2 * np.einsum("jkia->ijka", u[o, o, o, v], optimize=True)

    tau2 -= np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bajk,ikjb->ai", t2, tau2, optimize=True)

    tau2 = None

    tau3 = zeros((N, N, M, M))

    tau3 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau3 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau4 = zeros((N, M))

    tau4 += np.einsum("bj,jiba->ia", t1, tau3, optimize=True)

    tau5 = zeros((N, M))

    tau5 += np.einsum("ia->ia", tau4, optimize=True)

    tau4 = None

    tau10 = zeros((N, N))

    tau10 += np.einsum("abkj,kiab->ij", t2, tau3, optimize=True)

    tau3 = None

    tau5 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau10 += np.einsum("aj,ia->ij", t1, tau5, optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau6 -= np.einsum("baji->ijab", t2, optimize=True)

    r1 += np.einsum("jb,jiab->ai", tau5, tau6, optimize=True)

    tau5 = None

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau7 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    r1 += np.einsum("bj,jiab->ai", t1, tau7, optimize=True)

    tau7 = None

    tau8 += np.einsum("ab->ab", f[v, v], optimize=True)

    r1 += np.einsum("bi,ab->ai", t1, tau8, optimize=True)

    tau8 = None

    tau9 = zeros((N, N, N, M))

    tau9 -= np.einsum("ijka->ijka", u[o, o, o, v], optimize=True)

    tau9 += 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau10 += np.einsum("ak,kija->ij", t1, tau9, optimize=True)

    tau9 = None

    tau10 += np.einsum("ij->ij", f[o, o], optimize=True)

    r1 -= np.einsum("aj,ji->ai", t1, tau10, optimize=True)

    tau10 = None

    r1 += np.einsum("ai->ai", f[v, o], optimize=True)

    return r1


def amplitudes_intermediates_rccsd_t2(t1, t2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M, M, M))

    tau0 += np.einsum("di,abcd->iabc", t1, u[v, v, v, v], optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("cj,ibac->abij", t1, tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, N, N))

    tau1 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ablk,ijlk->abij", t2, tau1, optimize=True)

    tau1 = None

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("ijab->ijab", tau2, optimize=True)

    tau2 = None

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("ci,abjc->ijab", t1, u[v, v, o, v], optimize=True)

    tau26 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau3 = None

    tau4 = zeros((N, N, N, N))

    tau4 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("ablk,ilkj->ijab", t2, tau4, optimize=True)

    tau4 = None

    tau26 += np.einsum("ijab->ijab", tau5, optimize=True)

    tau5 = None

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("cajk,ikbc->ijab", t2, tau6, optimize=True)

    tau26 -= np.einsum("ijab->ijab", tau7, optimize=True)

    tau7 = None

    tau41 = zeros((N, N, M, M))

    tau41 += np.einsum("jiab->ijab", tau6, optimize=True)

    tau6 = None

    tau8 = zeros((N, M, M, M))

    tau8 -= np.einsum("iabc->iabc", u[o, v, v, v], optimize=True)

    tau8 += 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("cj,iacb->ijab", t1, tau8, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("cbkj,kiac->ijab", t2, tau9, optimize=True)

    tau9 = None

    tau26 += np.einsum("ijba->ijab", tau10, optimize=True)

    tau10 = None

    tau34 = zeros((M, M))

    tau34 += np.einsum("ci,iabc->ab", t1, tau8, optimize=True)

    tau8 = None

    tau35 = zeros((M, M))

    tau35 -= np.einsum("ab->ab", tau34, optimize=True)

    tau34 = None

    tau11 = zeros((N, N, N, M))

    tau11 += np.einsum("bi,jakb->ijka", t1, u[o, v, o, v], optimize=True)

    tau21 = zeros((N, N, N, M))

    tau21 -= np.einsum("jika->ijka", tau11, optimize=True)

    tau11 = None

    tau12 = zeros((N, N, N, M))

    tau12 += np.einsum("abil,ljkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau21 += np.einsum("jika->ijka", tau12, optimize=True)

    tau12 = None

    tau13 = zeros((N, N, N, M))

    tau13 += np.einsum("bi,kjba->ijka", t1, u[o, o, v, v], optimize=True)

    tau14 = zeros((N, N, N, M))

    tau14 += np.einsum("bajl,iklb->ijka", t2, tau13, optimize=True)

    tau21 += np.einsum("jkia->ijka", tau14, optimize=True)

    tau14 = None

    tau29 = zeros((N, N, M, M))

    tau29 += np.einsum("ak,ijkb->ijab", t1, tau13, optimize=True)

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("ijab->ijab", tau29, optimize=True)

    tau29 = None

    tau39 = zeros((N, N, N, M))

    tau39 += 2 * np.einsum("ijka->ijka", tau13, optimize=True)

    tau39 -= np.einsum("ikja->ijka", tau13, optimize=True)

    tau40 = zeros((N, N, N, M))

    tau40 += np.einsum("balk,iljb->ijka", t2, tau39, optimize=True)

    tau39 = None

    tau43 = zeros((N, N, N, M))

    tau43 += np.einsum("ijka->ijka", tau40, optimize=True)

    tau40 = None

    tau53 = zeros((N, N, N, N))

    tau53 += np.einsum("aj,ilka->ijkl", t1, tau13, optimize=True)

    tau13 = None

    tau54 = zeros((N, N, N, N))

    tau54 += np.einsum("lkji->ijkl", tau53, optimize=True)

    tau53 = None

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau15 -= 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau15 += np.einsum("baji->ijab", t2, optimize=True)

    tau16 = zeros((N, N, N, M))

    tau16 += np.einsum("liab,jlkb->ijka", tau15, u[o, o, o, v], optimize=True)

    tau15 = None

    tau21 += np.einsum("jika->ijka", tau16, optimize=True)

    tau16 = None

    tau17 = zeros((N, N, M, M))

    tau17 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau17 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau18 = zeros((N, M))

    tau18 += np.einsum("bj,jiba->ia", t1, tau17, optimize=True)

    tau19 = zeros((N, M))

    tau19 += np.einsum("ia->ia", tau18, optimize=True)

    tau18 = None

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("cbkj,kica->ijab", t2, tau17, optimize=True)

    tau31 -= np.einsum("jiba->ijab", tau30, optimize=True)

    tau30 = None

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("bckj,ikac->ijab", t2, tau31, optimize=True)

    tau31 = None

    tau50 = zeros((N, N, M, M))

    tau50 -= np.einsum("ijab->ijab", tau32, optimize=True)

    tau32 = None

    tau33 = zeros((M, M))

    tau33 += np.einsum("bcij,ijac->ab", t2, tau17, optimize=True)

    tau35 += np.einsum("ba->ab", tau33, optimize=True)

    tau33 = None

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum("ac,cbij->ijab", tau35, t2, optimize=True)

    tau35 = None

    tau50 += np.einsum("jiba->ijab", tau36, optimize=True)

    tau36 = None

    tau45 = zeros((N, N))

    tau45 += np.einsum("abkj,kiab->ij", t2, tau17, optimize=True)

    tau17 = None

    tau48 = zeros((N, N))

    tau48 += np.einsum("ij->ij", tau45, optimize=True)

    tau45 = None

    tau19 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau20 = zeros((N, N, N, M))

    tau20 += np.einsum("ib,bajk->ijka", tau19, t2, optimize=True)

    tau21 -= np.einsum("ikja->ijka", tau20, optimize=True)

    tau20 = None

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum("bk,kija->ijab", t1, tau21, optimize=True)

    tau21 = None

    tau26 += np.einsum("ijba->ijab", tau22, optimize=True)

    tau22 = None

    tau23 = zeros((N, N))

    tau23 += np.einsum("aj,ia->ij", t1, tau19, optimize=True)

    tau19 = None

    tau24 = zeros((N, N))

    tau24 += np.einsum("ji->ij", tau23, optimize=True)

    tau23 = None

    tau24 += np.einsum("ji->ij", f[o, o], optimize=True)

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum("ik,abkj->ijab", tau24, t2, optimize=True)

    tau24 = None

    tau26 -= np.einsum("ijba->ijab", tau25, optimize=True)

    tau25 = None

    r2 += np.einsum("ijba->abij", tau26, optimize=True)

    r2 += np.einsum("jiab->abij", tau26, optimize=True)

    tau26 = None

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("ci,jacb->ijab", t1, u[o, v, v, v], optimize=True)

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("cajk,ikbc->ijab", t2, tau27, optimize=True)

    tau27 = None

    tau50 += np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau37 = zeros((N, N, N, M))

    tau37 += np.einsum("bail,ljkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau43 -= np.einsum("ijka->ijka", tau37, optimize=True)

    tau37 = None

    tau38 = zeros((N, N, N, M))

    tau38 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau43 += np.einsum("ikja->ijka", tau38, optimize=True)

    tau38 = None

    tau41 += np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau42 = zeros((N, N, N, M))

    tau42 += np.einsum("bk,ijab->ijka", t1, tau41, optimize=True)

    tau41 = None

    tau43 += np.einsum("jkia->ijka", tau42, optimize=True)

    tau42 = None

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("bk,ikja->ijab", t1, tau43, optimize=True)

    tau43 = None

    tau50 += np.einsum("ijba->ijab", tau44, optimize=True)

    tau44 = None

    tau46 = zeros((N, N, N, M))

    tau46 -= np.einsum("ijka->ijka", u[o, o, o, v], optimize=True)

    tau46 += 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau47 = zeros((N, N))

    tau47 += np.einsum("ak,kija->ij", t1, tau46, optimize=True)

    tau46 = None

    tau48 += np.einsum("ij->ij", tau47, optimize=True)

    tau47 = None

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum("ki,abkj->ijab", tau48, t2, optimize=True)

    tau48 = None

    tau50 += np.einsum("jiba->ijab", tau49, optimize=True)

    tau49 = None

    r2 -= np.einsum("ijab->abij", tau50, optimize=True)

    r2 -= np.einsum("jiba->abij", tau50, optimize=True)

    tau50 = None

    tau51 = zeros((N, N, M, M))

    tau51 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau51 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau52 = zeros((N, N, M, M))

    tau52 += 2 * np.einsum("caki,kjbc->ijab", t2, tau51, optimize=True)

    tau51 = None

    tau52 += 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau52 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    r2 += np.einsum("cbkj,ikac->abij", t2, tau52, optimize=True)

    tau52 = None

    tau54 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau57 = zeros((N, N, N, M))

    tau57 += np.einsum("al,likj->ijka", t1, tau54, optimize=True)

    r2 += np.einsum("bakl,klji->abij", t2, tau54, optimize=True)

    tau54 = None

    tau55 = zeros((N, M, M, M))

    tau55 += np.einsum("aj,jicb->iabc", t1, u[o, o, v, v], optimize=True)

    tau56 = zeros((M, M, M, M))

    tau56 += np.einsum("ai,ibcd->abcd", t1, tau55, optimize=True)

    tau55 = None

    tau56 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    r2 += np.einsum("cdji,bacd->abij", t2, tau56, optimize=True)

    tau56 = None

    tau57 -= np.einsum("iajk->ijka", u[o, v, o, o], optimize=True)

    r2 += np.einsum("bk,kjia->abij", t1, tau57, optimize=True)

    tau57 = None

    tau58 = zeros((N, N, M, M))

    tau58 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau58 += np.einsum("caik,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bckj,ikac->abij", t2, tau58, optimize=True)

    tau58 = None

    tau59 = zeros((N, N, M, M))

    tau59 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau59 += np.einsum("caik,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bcki,jkac->abij", t2, tau59, optimize=True)

    tau59 = None

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    r2 -= np.einsum("ak,kbij->abij", t1, u[o, v, o, o], optimize=True)

    r2 -= np.einsum("acik,kbjc->abij", t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("caik,kbcj->abij", t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("cajk,kbic->abij", t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("acik,kbcj->abij", t2, u[o, v, v, o], optimize=True)

    return r2
