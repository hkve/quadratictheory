import numpy as np


def lambda_amplitudes_intermediates_rccsd(t1, t2, l1, l2, u, f, v, o):
    rhs1 = lambda_amplitudes_intermediates_rccsd_l1(t1, t2, l1, l2, u, f, v, o)
    rhs2 = lambda_amplitudes_intermediates_rccsd_l2(t1, t2, l1, l2, u, f, v, o)

    return rhs1, 0.5 * rhs2


def lambda_amplitudes_intermediates_rccsd_l1(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau25 = zeros((N, N, N, M))

    tau25 += 2 * np.einsum("al,iljk->ijka", t1, tau0, optimize=True)

    r1 = zeros((M, N))

    r1 += 2 * np.einsum("ijkl,lkja->ai", tau0, u[o, o, o, v], optimize=True)

    tau0 = None

    tau1 = zeros((N, N, N, M))

    tau1 += np.einsum("bk,baji->ijka", t1, l2, optimize=True)

    tau2 = zeros((N, N, N, N))

    tau2 += np.einsum("ak,ijla->ijkl", t1, tau1, optimize=True)

    r1 += 2 * np.einsum("iljk,kjla->ai", tau2, u[o, o, o, v], optimize=True)

    tau2 = None

    tau25 += 2 * np.einsum("abjl,likb->ijka", t2, tau1, optimize=True)

    tau25 += 2 * np.einsum("bajl,ilkb->ijka", t2, tau1, optimize=True)

    tau31 = zeros((N, M))

    tau31 += 2 * np.einsum("abkj,jkib->ia", t2, tau1, optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    r1 -= 2 * np.einsum("ijbc,jbac->ai", tau3, u[o, v, v, v], optimize=True)

    tau3 = None

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    r1 -= 2 * np.einsum("ijbc,jbca->ai", tau4, u[o, v, v, v], optimize=True)

    tau4 = None

    tau5 = zeros((N, N, N, M))

    tau5 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau6 = zeros((N, N, N, M))

    tau6 += np.einsum("ijka->ijka", tau5, optimize=True)

    tau12 = zeros((N, N, N, M))

    tau12 += 2 * np.einsum("ijka->ijka", tau5, optimize=True)

    tau12 -= np.einsum("ikja->ijka", tau5, optimize=True)

    tau15 = zeros((N, N, N, M))

    tau15 += np.einsum("kjia->ijka", tau5, optimize=True)

    tau5 = None

    tau6 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau11 = zeros((N, N, N, M))

    tau11 += np.einsum("ablk,jilb->ijka", t2, tau6, optimize=True)

    tau17 = zeros((N, N, N, M))

    tau17 += np.einsum("ablk,iljb->ijka", t2, tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, M, M, M))

    tau7 += np.einsum("iabc->iabc", u[o, v, v, v], optimize=True)

    tau7 -= np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau11 -= np.einsum("bckj,iabc->ijka", t2, tau7, optimize=True)

    tau7 = None

    tau8 = zeros((N, N, M, M))

    tau8 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau8 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau9 = zeros((N, M))

    tau9 += np.einsum("bj,jiba->ia", t1, tau8, optimize=True)

    tau9 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau11 -= np.einsum("ib,bakj->ijka", tau9, t2, optimize=True)

    tau9 = None

    tau10 = zeros((N, N, N, N))

    tau10 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau10 += np.einsum("ak,jila->ijkl", t1, u[o, o, o, v], optimize=True)

    tau11 += np.einsum("al,lijk->ijka", t1, tau10, optimize=True)

    tau10 = None

    tau11 -= np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau11 -= np.einsum("bj,iakb->ijka", t1, u[o, v, o, v], optimize=True)

    r1 += 2 * np.einsum("abkj,ijkb->ai", l2, tau11, optimize=True)

    tau11 = None

    tau12 -= np.einsum("jkia->ijka", u[o, o, o, v], optimize=True)

    tau12 += 2 * np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau17 -= np.einsum("balk,iljb->ijka", t2, tau12, optimize=True)

    tau12 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("jiab->ijab", tau13, optimize=True)

    tau13 = None

    tau14 += np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau17 -= np.einsum("bi,jkab->ijka", t1, tau14, optimize=True)

    r1 -= 2 * np.einsum("jikb,kjba->ai", tau1, tau14, optimize=True)

    tau14 = None

    tau15 += np.einsum("ijka->ijka", u[o, o, o, v], optimize=True)

    tau16 = zeros((N, N, N, N))

    tau16 += np.einsum("ai,jkla->ijkl", t1, tau15, optimize=True)

    tau15 = None

    tau17 += np.einsum("al,iljk->ijka", t1, tau16, optimize=True)

    tau16 = None

    r1 += 2 * np.einsum("bakj,jikb->ai", l2, tau17, optimize=True)

    tau17 = None

    tau18 = zeros((N, M, M, M))

    tau18 += 2 * np.einsum("iabc->iabc", u[o, v, v, v], optimize=True)

    tau18 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau19 += 2 * np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    r1 += np.einsum("jbca,ijbc->ai", tau18, tau19, optimize=True)

    tau19 = None

    tau20 = zeros((N, M, M, M))

    tau20 += np.einsum("abic->iabc", u[v, v, o, v], optimize=True)

    tau20 += np.einsum("di,bacd->iabc", t1, u[v, v, v, v], optimize=True)

    r1 += 2 * np.einsum("bcji,jbca->ai", l2, tau20, optimize=True)

    tau20 = None

    tau21 = zeros((N, N, N, M))

    tau21 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau25 += np.einsum("ijka->ijka", tau21, optimize=True)

    tau25 -= 2 * np.einsum("ikja->ijka", tau21, optimize=True)

    tau21 = None

    tau22 = zeros((N, N))

    tau22 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau24 = zeros((N, N))

    tau24 += np.einsum("ij->ij", tau22, optimize=True)

    tau22 = None

    tau23 = zeros((N, N))

    tau23 += np.einsum("baik,abkj->ij", l2, t2, optimize=True)

    tau24 += 2 * np.einsum("ij->ij", tau23, optimize=True)

    tau23 = None

    tau25 += np.einsum("aj,ik->ijka", t1, tau24, optimize=True)

    r1 += np.einsum("ijkb,jkab->ai", tau25, u[o, o, v, v], optimize=True)

    tau25 = None

    tau27 = zeros((N, N, N, M))

    tau27 += np.einsum("aj,ik->ijka", t1, tau24, optimize=True)

    tau31 += np.einsum("aj,ji->ia", t1, tau24, optimize=True)

    r1 -= np.einsum("ja,ij->ai", f[o, v], tau24, optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau26 -= 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau26 += np.einsum("baji->ijab", t2, optimize=True)

    tau27 -= np.einsum("likb,ljab->ijka", tau1, tau26, optimize=True)

    r1 -= 2 * np.einsum("ijkb,jkba->ai", tau27, u[o, o, v, v], optimize=True)

    tau27 = None

    tau36 = zeros((N, N))

    tau36 -= np.einsum("kjab,kiba->ij", tau26, u[o, o, v, v], optimize=True)

    tau26 = None

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau28 += np.einsum("cj,iacb->ijab", t1, u[o, v, v, v], optimize=True)

    r1 -= 2 * np.einsum("ijkb,kjba->ai", tau1, tau28, optimize=True)

    tau1 = None

    tau28 = None

    tau29 = zeros((M, M))

    tau29 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau29 += 2 * np.einsum("acji,bcji->ab", l2, t2, optimize=True)

    r1 += np.einsum("bc,ibac->ai", tau29, tau18, optimize=True)

    tau18 = None

    tau29 = None

    tau30 = zeros((N, N, M, M))

    tau30 -= np.einsum("abji->ijab", t2, optimize=True)

    tau30 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau31 -= np.einsum("bj,jiba->ia", l1, tau30, optimize=True)

    tau30 = None

    tau31 -= 2 * np.einsum("ai->ia", t1, optimize=True)

    r1 -= np.einsum("jb,jiba->ai", tau31, tau8, optimize=True)

    tau8 = None

    tau31 = None

    tau32 = zeros((N, N, N, M))

    tau32 += 2 * np.einsum("ijka->ijka", u[o, o, o, v], optimize=True)

    tau32 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    r1 -= np.einsum("jk,kija->ai", tau24, tau32, optimize=True)

    tau24 = None

    tau32 = None

    tau33 = zeros((N, N, M, M))

    tau33 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau33 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    r1 += np.einsum("bj,ijba->ai", l1, tau33, optimize=True)

    tau33 = None

    tau34 = zeros((N, N, N, M))

    tau34 -= np.einsum("ijka->ijka", u[o, o, o, v], optimize=True)

    tau34 += 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau36 += np.einsum("ak,kija->ij", t1, tau34, optimize=True)

    tau34 = None

    tau35 = zeros((N, M))

    tau35 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau35 += 2 * np.einsum("bj,ijab->ia", t1, u[o, o, v, v], optimize=True)

    tau36 += np.einsum("aj,ia->ij", t1, tau35, optimize=True)

    tau35 = None

    tau36 += np.einsum("ij->ij", f[o, o], optimize=True)

    r1 -= np.einsum("aj,ij->ai", l1, tau36, optimize=True)

    tau36 = None

    r1 += np.einsum("bi,ba->ai", l1, f[v, v], optimize=True)

    r1 += 2 * np.einsum("ia->ai", f[o, v], optimize=True)

    return r1


def lambda_amplitudes_intermediates_rccsd_l2(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum("abij,lkba->ijkl", t2, u[o, o, v, v], optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += 2 * np.einsum("balk,lkji->abij", l2, tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, N, N))

    tau1 += np.einsum("baij,ablk->ijkl", l2, t2, optimize=True)

    r2 += 2 * np.einsum("ijkl,lkba->abij", tau1, u[o, o, v, v], optimize=True)

    tau1 = None

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau3 = zeros((N, N, N, N))

    tau3 += np.einsum("al,jika->ijkl", t1, tau2, optimize=True)

    r2 += 2 * np.einsum("ijlk,lkab->abij", tau3, u[o, o, v, v], optimize=True)

    tau3 = None

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("jikc,kcab->ijab", tau2, u[o, v, v, v], optimize=True)

    tau2 = None

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum("ijab->ijab", tau26, optimize=True)

    tau26 = None

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True)

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("ijab->ijab", tau4, optimize=True)

    tau4 = None

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("ak,kijb->ijab", t1, u[o, o, o, v], optimize=True)

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("jiab->ijab", tau5, optimize=True)

    tau5 = None

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau11 -= np.einsum("ijab->ijab", tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau7 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("cbkj,kica->ijab", t2, tau7, optimize=True)

    tau11 -= np.einsum("jiba->ijab", tau8, optimize=True)

    tau8 = None

    tau13 = zeros((N, M))

    tau13 += np.einsum("bj,jiba->ia", t1, tau7, optimize=True)

    tau7 = None

    tau14 = zeros((N, M))

    tau14 += np.einsum("ia->ia", tau13, optimize=True)

    tau13 = None

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("baji->ijab", t2, optimize=True)

    tau9 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("kjbc,kica->ijab", tau9, u[o, o, v, v], optimize=True)

    tau11 += np.einsum("jiba->ijab", tau10, optimize=True)

    tau10 = None

    tau29 = zeros((N, N, M, M))

    tau29 += np.einsum("kjbc,kiac->ijab", tau9, u[o, o, v, v], optimize=True)

    tau9 = None

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("jiba->ijab", tau29, optimize=True)

    tau29 = None

    tau11 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("cbkj,kica->ijab", l2, tau11, optimize=True)

    tau11 = None

    tau15 += 2 * np.einsum("jiba->ijab", tau12, optimize=True)

    tau12 = None

    tau14 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau15 -= np.einsum("ai,jb->ijab", l1, tau14, optimize=True)

    tau14 = None

    r2 -= 2 * np.einsum("ijab->abij", tau15, optimize=True)

    r2 += np.einsum("ijba->abij", tau15, optimize=True)

    r2 += np.einsum("jiab->abij", tau15, optimize=True)

    r2 -= 2 * np.einsum("jiba->abij", tau15, optimize=True)

    tau15 = None

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau25 = zeros((N, N, M, M))

    tau25 -= np.einsum("ijab->ijab", tau16, optimize=True)

    tau16 = None

    tau17 = zeros((M, M))

    tau17 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau19 = zeros((M, M))

    tau19 += np.einsum("ab->ab", tau17, optimize=True)

    tau17 = None

    tau18 = zeros((M, M))

    tau18 += np.einsum("acji,bcji->ab", l2, t2, optimize=True)

    tau19 += 2 * np.einsum("ab->ab", tau18, optimize=True)

    tau18 = None

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum("bc,ijca->ijab", tau19, u[o, o, v, v], optimize=True)

    tau19 = None

    tau25 += np.einsum("jiba->ijab", tau20, optimize=True)

    tau20 = None

    tau21 = zeros((N, N))

    tau21 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau23 = zeros((N, N))

    tau23 += np.einsum("ij->ij", tau21, optimize=True)

    tau21 = None

    tau22 = zeros((N, N))

    tau22 += np.einsum("abik,bakj->ij", l2, t2, optimize=True)

    tau23 += 2 * np.einsum("ij->ij", tau22, optimize=True)

    tau22 = None

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("jk,kiab->ijab", tau23, u[o, o, v, v], optimize=True)

    tau23 = None

    tau25 += np.einsum("jiba->ijab", tau24, optimize=True)

    tau24 = None

    r2 += np.einsum("ijab->abij", tau25, optimize=True)

    r2 -= 2 * np.einsum("ijba->abij", tau25, optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", tau25, optimize=True)

    r2 += np.einsum("jiba->abij", tau25, optimize=True)

    tau25 = None

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau30 += np.einsum("jiab->ijab", tau27, optimize=True)

    tau27 = None

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("ci,jacb->ijab", t1, u[o, v, v, v], optimize=True)

    tau30 -= np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau30 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("cbkj,kica->ijab", l2, tau30, optimize=True)

    tau50 -= np.einsum("jiba->ijab", tau31, optimize=True)

    tau31 = None

    tau53 = zeros((N, N, M, M))

    tau53 += np.einsum("bckj,kica->ijab", l2, tau30, optimize=True)

    tau30 = None

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("jiba->ijab", tau53, optimize=True)

    tau53 = None

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau32 -= 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau32 += np.einsum("baji->ijab", t2, optimize=True)

    tau33 = zeros((M, M))

    tau33 += np.einsum("ijcb,ijac->ab", tau32, u[o, o, v, v], optimize=True)

    tau32 = None

    tau39 = zeros((M, M))

    tau39 -= np.einsum("ba->ab", tau33, optimize=True)

    tau33 = None

    tau34 = zeros((N, M, M, M))

    tau34 += 2 * np.einsum("iabc->iabc", u[o, v, v, v], optimize=True)

    tau34 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau35 = zeros((M, M))

    tau35 += np.einsum("ci,iacb->ab", t1, tau34, optimize=True)

    tau34 = None

    tau39 -= np.einsum("ab->ab", tau35, optimize=True)

    tau35 = None

    tau36 = zeros((N, M))

    tau36 += np.einsum("bj,ijab->ia", t1, u[o, o, v, v], optimize=True)

    tau37 = zeros((N, M))

    tau37 += 2 * np.einsum("ia->ia", tau36, optimize=True)

    tau36 = None

    tau37 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau38 = zeros((M, M))

    tau38 += np.einsum("bi,ia->ab", t1, tau37, optimize=True)

    tau37 = None

    tau39 += np.einsum("ba->ab", tau38, optimize=True)

    tau38 = None

    tau39 -= np.einsum("ab->ab", f[v, v], optimize=True)

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum("ca,cbij->ijab", tau39, l2, optimize=True)

    tau39 = None

    tau50 += np.einsum("jiba->ijab", tau40, optimize=True)

    tau40 = None

    tau41 = zeros((N, N, M, M))

    tau41 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau41 -= np.einsum("abji->ijab", t2, optimize=True)

    tau41 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau42 = zeros((N, N))

    tau42 += np.einsum("kjab,kiab->ij", tau41, u[o, o, v, v], optimize=True)

    tau41 = None

    tau48 = zeros((N, N))

    tau48 += np.einsum("ij->ij", tau42, optimize=True)

    tau42 = None

    tau43 = zeros((N, N, N, M))

    tau43 -= np.einsum("ijka->ijka", u[o, o, o, v], optimize=True)

    tau43 += 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau44 = zeros((N, N))

    tau44 += np.einsum("ak,kija->ij", t1, tau43, optimize=True)

    tau43 = None

    tau48 += np.einsum("ij->ij", tau44, optimize=True)

    tau44 = None

    tau45 = zeros((N, M))

    tau45 += np.einsum("bj,ijba->ia", t1, u[o, o, v, v], optimize=True)

    tau46 = zeros((N, M))

    tau46 -= np.einsum("ia->ia", tau45, optimize=True)

    tau45 = None

    tau46 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau47 = zeros((N, N))

    tau47 += np.einsum("aj,ia->ij", t1, tau46, optimize=True)

    tau46 = None

    tau48 += np.einsum("ij->ij", tau47, optimize=True)

    tau47 = None

    tau48 += np.einsum("ij->ij", f[o, o], optimize=True)

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum("ik,abkj->ijab", tau48, l2, optimize=True)

    tau48 = None

    tau50 += np.einsum("jiba->ijab", tau49, optimize=True)

    tau49 = None

    r2 -= 2 * np.einsum("ijab->abij", tau50, optimize=True)

    r2 -= 2 * np.einsum("jiba->abij", tau50, optimize=True)

    tau50 = None

    tau51 = zeros((N, N, N, N))

    tau51 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau52 = zeros((N, N, M, M))

    tau52 += np.einsum("balk,kijl->ijab", l2, tau51, optimize=True)

    tau51 = None

    tau54 += np.einsum("ijab->ijab", tau52, optimize=True)

    tau52 = None

    r2 += 2 * np.einsum("ijba->abij", tau54, optimize=True)

    r2 += 2 * np.einsum("jiab->abij", tau54, optimize=True)

    tau54 = None

    tau55 = zeros((N, N, N, M))

    tau55 += np.einsum("bi,kjba->ijka", t1, u[o, o, v, v], optimize=True)

    tau56 = zeros((N, N, N, N))

    tau56 += np.einsum("al,kjia->ijkl", t1, tau55, optimize=True)

    tau55 = None

    tau56 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("ablk,jikl->abij", l2, tau56, optimize=True)

    tau56 = None

    r2 += 2 * np.einsum("dcji,dcba->abij", l2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", u[o, o, v, v], optimize=True)

    r2 += 4 * np.einsum("jiba->abij", u[o, o, v, v], optimize=True)

    return r2
