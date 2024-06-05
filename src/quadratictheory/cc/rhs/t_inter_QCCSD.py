import numpy as np
from quadratictheory.cc.rhs.t_inter_CCSD import amplitudes_intermediates_ccsd


def amplitudes_intermediates_qccsd(t1, t2, l1, l2, u, f, v, o):
    r1, r2 = amplitudes_intermediates_ccsd(t1, t2, u, f, v, o)

    r1 += amplitudes_intermediates_qccsd_t1_addition(t1, t2, l1, l2, u, f, v, o)
    r2 += amplitudes_intermediates_qccsd_t2_addition(t1, t2, l1, l2, u, f, v, o)

    r2 = 0.25 * (
        r2 - r2.transpose(1, 0, 2, 3) - r2.transpose(0, 1, 3, 2) + r2.transpose(1, 0, 3, 2)
    )

    return r1, r2


def amplitudes_intermediates_qccsd_t1_addition(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M))

    tau0 += np.einsum("bj,abij->ia", l1, t2, optimize=True)

    tau41 = zeros((N, N, N, M))

    tau41 += 4 * np.einsum("kb,jiab->ijka", tau0, u[o, o, v, v], optimize=True)

    tau46 = zeros((N, N, M, M))

    tau46 += 8 * np.einsum("ai,jb->ijab", t1, tau0, optimize=True)

    tau46 += 8 * np.einsum("bj,ia->ijab", t1, tau0, optimize=True)

    tau58 = zeros((N, M))

    tau58 -= 2 * np.einsum("ia->ia", tau0, optimize=True)

    r1 = zeros((M, N))

    r1 += np.einsum("ab,ib->ai", f[v, v], tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, N, N))

    tau1 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau24 = zeros((N, N, N, M))

    tau24 += np.einsum("al,iljk->ijka", t1, tau1, optimize=True)

    tau28 = zeros((N, N, N, M))

    tau28 -= np.einsum("ikja->ijka", tau24, optimize=True)

    tau42 = zeros((N, N, N, M))

    tau42 += np.einsum("ikja->ijka", tau24, optimize=True)

    tau47 = zeros((N, N, N, M))

    tau47 += np.einsum("ikja->ijka", tau24, optimize=True)

    tau53 = zeros((N, N, N, M))

    tau53 -= np.einsum("ikja->ijka", tau24, optimize=True)

    tau24 = None

    tau41 += 4 * np.einsum("ilkm,jmla->ijka", tau1, u[o, o, o, v], optimize=True)

    tau46 += np.einsum("ablk,lkij->ijab", t2, tau1, optimize=True)

    tau61 = zeros((N, M))

    tau61 += np.einsum("ijlk,lkja->ia", tau1, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("lkij,jalk->ai", tau1, u[o, v, o, o], optimize=True) / 4

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau3 = zeros((N, N, N, M))

    tau3 -= np.einsum("ablk,lijb->ijka", t2, tau2, optimize=True)

    tau28 += 2 * np.einsum("ikja->ijka", tau3, optimize=True)

    tau42 += 4 * np.einsum("ijka->ijka", tau3, optimize=True)

    r1 -= np.einsum("kjib,jakb->ai", tau3, u[o, v, o, v], optimize=True)

    tau3 = None

    tau43 = zeros((N, M))

    tau43 += np.einsum("bakj,kjib->ia", t2, tau2, optimize=True)

    tau45 = zeros((N, M))

    tau45 += np.einsum("ia->ia", tau43, optimize=True)

    tau58 += np.einsum("ia->ia", tau43, optimize=True)

    tau43 = None

    tau49 = zeros((N, N, N, N))

    tau49 += np.einsum("ak,ijla->ijkl", t1, tau2, optimize=True)

    tau61 -= 2 * np.einsum("iljk,kjla->ia", tau49, u[o, o, o, v], optimize=True)

    tau49 = None

    tau4 = zeros((N, M, M, M))

    tau4 -= np.einsum("di,abdc->iabc", t1, u[v, v, v, v], optimize=True)

    tau11 = zeros((N, M, M, M))

    tau11 += np.einsum("ibac->iabc", tau4, optimize=True)

    tau52 = zeros((N, M, M, M))

    tau52 += np.einsum("ibac->iabc", tau4, optimize=True)

    tau4 = None

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("jiab->ijab", tau5, optimize=True)

    tau64 = zeros((N, N, M, M))

    tau64 += np.einsum("jiab->ijab", tau5, optimize=True)

    tau5 = None

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau8 += 2 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau54 = zeros((N, N, M, M))

    tau54 -= np.einsum("jiab->ijab", tau6, optimize=True)

    tau64 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau7 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau8 += np.einsum("kica,kjcb->ijab", tau7, u[o, o, v, v], optimize=True)

    tau53 += 2 * np.einsum("likb,ljba->ijka", tau2, tau7, optimize=True)

    tau7 = None

    tau8 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau11 -= np.einsum("bj,ijac->iabc", t1, tau8, optimize=True)

    tau8 = None

    tau9 = zeros((N, M))

    tau9 += np.einsum("bj,jiba->ia", t1, u[o, o, v, v], optimize=True)

    tau10 = zeros((N, M))

    tau10 += np.einsum("ia->ia", tau9, optimize=True)

    tau9 = None

    tau10 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau11 -= np.einsum("jc,baji->iabc", tau10, t2, optimize=True)

    tau21 = zeros((N, N, N, M))

    tau21 += np.einsum("kb,baij->ijka", tau10, t2, optimize=True)

    tau22 = zeros((N, N, N, M))

    tau22 += 2 * np.einsum("jika->ijka", tau21, optimize=True)

    tau51 = zeros((N, N, N, M))

    tau51 += 2 * np.einsum("kjia->ijka", tau21, optimize=True)

    tau21 = None

    tau66 = zeros((N, N, M, M))

    tau66 += 8 * np.einsum("bi,ja->ijab", l1, tau10, optimize=True)

    tau10 = None

    tau11 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau11 += 2 * np.einsum("bdji,jadc->iabc", t2, u[o, v, v, v], optimize=True)

    tau41 -= 2 * np.einsum("bcji,kbca->ijka", l2, tau11, optimize=True)

    tau11 = None

    tau12 = zeros((N, N, N, M))

    tau12 += np.einsum("bi,jakb->ijka", t1, u[o, v, o, v], optimize=True)

    tau15 = zeros((N, N, N, M))

    tau15 += np.einsum("ijka->ijka", tau12, optimize=True)

    tau51 -= 4 * np.einsum("kija->ijka", tau12, optimize=True)

    tau12 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("baji->ijab", t2, optimize=True)

    tau13 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau14 = zeros((N, N, N, M))

    tau14 += np.einsum("lkba,lijb->ijka", tau13, u[o, o, o, v], optimize=True)

    tau15 -= np.einsum("jkia->ijka", tau14, optimize=True)

    tau14 = None

    tau22 += 2 * np.einsum("ikja->ijka", tau15, optimize=True)

    tau22 -= 2 * np.einsum("jkia->ijka", tau15, optimize=True)

    tau15 = None

    tau64 += np.einsum("kica,kjcb->ijab", tau13, u[o, o, v, v], optimize=True)

    tau13 = None

    tau16 = zeros((N, N, N, M))

    tau16 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau17 = zeros((N, N, N, M))

    tau17 -= np.einsum("bajl,ilkb->ijka", t2, tau16, optimize=True)

    tau22 -= 2 * np.einsum("ijka->ijka", tau17, optimize=True)

    tau22 += 2 * np.einsum("jika->ijka", tau17, optimize=True)

    tau17 = None

    tau29 = zeros((N, N, N, M))

    tau29 -= np.einsum("ikja->ijka", tau16, optimize=True)

    tau32 = zeros((N, N, N, M))

    tau32 += np.einsum("kjia->ijka", tau16, optimize=True)

    tau16 = None

    tau18 = zeros((N, N, M, M))

    tau18 -= np.einsum("baji->ijab", t2, optimize=True)

    tau18 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau19 = zeros((N, M, M, M))

    tau19 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau20 = zeros((N, M, M, M))

    tau20 += np.einsum("iacb->iabc", tau19, optimize=True)

    tau19 = None

    tau20 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau22 -= np.einsum("jibc,kabc->ijka", tau18, tau20, optimize=True)

    tau18 = None

    tau22 += 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau22 += 2 * np.einsum("al,lkji->ijka", t1, u[o, o, o, o], optimize=True)

    tau41 += 4 * np.einsum("balj,lkib->ijka", l2, tau22, optimize=True)

    tau22 = None

    tau23 = zeros((N, N, N, M))

    tau23 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau28 -= 2 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau42 += 4 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau46 += 2 * np.einsum("ak,kjib->ijab", t1, tau42, optimize=True)

    tau42 = None

    tau47 += 2 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau53 -= 2 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau23 = None

    tau25 = zeros((N, N))

    tau25 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau27 = zeros((N, N))

    tau27 += 2 * np.einsum("ij->ij", tau25, optimize=True)

    tau25 = None

    tau26 = zeros((N, N))

    tau26 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau27 += np.einsum("ij->ij", tau26, optimize=True)

    tau26 = None

    tau28 += np.einsum("aj,ik->ijka", t1, tau27, optimize=True)

    tau41 -= 4 * np.einsum("iklb,ljba->ijka", tau28, u[o, o, v, v], optimize=True)

    tau28 = None

    tau41 -= 4 * np.einsum("il,ljka->ijka", tau27, u[o, o, o, v], optimize=True)

    tau44 = zeros((N, M))

    tau44 += np.einsum("aj,ji->ia", t1, tau27, optimize=True)

    tau45 += np.einsum("ia->ia", tau44, optimize=True)

    tau46 -= 4 * np.einsum("ai,jb->ijab", t1, tau45, optimize=True)

    tau45 = None

    tau58 += np.einsum("ia->ia", tau44, optimize=True)

    tau44 = None

    tau59 = zeros((N, M))

    tau59 += np.einsum("jb,jiba->ia", tau58, u[o, o, v, v], optimize=True)

    tau61 += 2 * np.einsum("ia->ia", tau59, optimize=True)

    tau67 = zeros((N, M))

    tau67 -= np.einsum("ia->ia", tau59, optimize=True)

    tau59 = None

    tau68 = zeros((N, N))

    tau68 += 4 * np.einsum("ka,kija->ij", tau58, u[o, o, o, v], optimize=True)

    r1 += np.einsum("jb,jaib->ai", tau58, u[o, v, o, v], optimize=True) / 2

    tau58 = None

    tau46 -= 2 * np.einsum("kj,baki->ijab", tau27, t2, optimize=True)

    tau47 += np.einsum("ak,ij->ijka", t1, tau27, optimize=True)

    r1 -= np.einsum("kjib,jakb->ai", tau47, u[o, v, o, v], optimize=True) / 2

    tau47 = None

    tau53 += 2 * np.einsum("aj,ik->ijka", t1, tau27, optimize=True)

    tau61 += np.einsum("ijkb,jkba->ia", tau53, u[o, o, v, v], optimize=True)

    tau53 = None

    tau60 = zeros((N, M))

    tau60 += np.einsum("kj,jika->ia", tau27, u[o, o, o, v], optimize=True)

    tau61 += 2 * np.einsum("ia->ia", tau60, optimize=True)

    tau67 -= np.einsum("ia->ia", tau60, optimize=True)

    tau60 = None

    tau61 += 2 * np.einsum("ja,ij->ia", f[o, v], tau27, optimize=True)

    tau66 -= 2 * np.einsum("ik,kjba->ijab", tau27, u[o, o, v, v], optimize=True)

    tau68 -= 4 * np.einsum("lk,kilj->ij", tau27, u[o, o, o, o], optimize=True)

    r1 -= np.einsum("kj,jaki->ai", tau27, u[o, v, o, o], optimize=True) / 2

    tau27 = None

    tau29 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau51 += 4 * np.einsum("balj,klib->ijka", t2, tau29, optimize=True)

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("ijab->ijab", tau30, optimize=True)

    tau46 += 8 * np.einsum("bcjk,kica->ijab", t2, tau30, optimize=True)

    tau62 = zeros((N, M, M, M))

    tau62 += 2 * np.einsum("bj,jiac->iabc", t1, tau30, optimize=True)

    tau30 = None

    tau31 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau41 += 4 * np.einsum("ljib,lkab->ijka", tau29, tau31, optimize=True)

    tau41 -= 8 * np.einsum("ikbc,jbca->ijka", tau31, u[o, v, v, v], optimize=True)

    tau61 -= 4 * np.einsum("ijbc,jbca->ia", tau31, u[o, v, v, v], optimize=True)

    r1 += np.einsum("jibc,bajc->ai", tau31, u[v, v, o, v], optimize=True)

    tau31 = None

    tau32 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau41 -= np.einsum("jilm,lmka->ijka", tau1, tau32, optimize=True)

    tau32 = None

    tau1 = None

    tau33 = zeros((N, N, N, N))

    tau33 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau34 = zeros((N, N, N, N))

    tau34 += 2 * np.einsum("kjil->ijkl", tau33, optimize=True)

    tau50 = zeros((N, N, N, N))

    tau50 -= 2 * np.einsum("kjil->ijkl", tau33, optimize=True)

    tau65 = zeros((N, N, N, N))

    tau65 -= 4 * np.einsum("ljik->ijkl", tau33, optimize=True)

    tau33 = None

    tau34 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau34 += np.einsum("balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau41 -= 2 * np.einsum("al,jilk->ijka", l1, tau34, optimize=True)

    tau34 = None

    tau35 = zeros((M, M))

    tau35 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau41 -= 2 * np.einsum("ab,kjib->ijka", tau35, tau29, optimize=True)

    tau29 = None

    tau46 += 4 * np.einsum("ca,bcij->ijab", tau35, t2, optimize=True)

    r1 += np.einsum("ijcb,jabc->ai", tau46, u[o, v, v, v], optimize=True) / 8

    tau46 = None

    tau56 = zeros((M, M))

    tau56 += np.einsum("ab->ab", tau35, optimize=True)

    tau62 -= np.einsum("bi,ac->iabc", t1, tau35, optimize=True)

    tau35 = None

    r1 -= np.einsum("ibdc,bacd->ai", tau62, u[v, v, v, v], optimize=True) / 2

    tau62 = None

    tau36 = zeros((N, N))

    tau36 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau40 = zeros((N, N))

    tau40 += 2 * np.einsum("ij->ij", tau36, optimize=True)

    tau36 = None

    tau37 = zeros((N, N))

    tau37 -= np.einsum("ak,kija->ij", t1, u[o, o, o, v], optimize=True)

    tau40 += 2 * np.einsum("ij->ij", tau37, optimize=True)

    tau37 = None

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum("baji->ijab", t2, optimize=True)

    tau38 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau39 = zeros((N, N))

    tau39 += np.einsum("kjab,kiab->ij", tau38, u[o, o, v, v], optimize=True)

    tau40 += np.einsum("ij->ij", tau39, optimize=True)

    tau39 = None

    tau51 += np.einsum("iabc,kjbc->ijka", tau20, tau38, optimize=True)

    tau20 = None

    tau65 += np.einsum("lkab,jiab->ijkl", tau38, u[o, o, v, v], optimize=True)

    tau38 = None

    tau40 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    tau41 += 4 * np.einsum("aj,ik->ijka", l1, tau40, optimize=True)

    tau61 += 2 * np.einsum("aj,ij->ia", l1, tau40, optimize=True)

    tau40 = None

    tau41 -= 8 * np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bajk,kjib->ai", t2, tau41, optimize=True) / 8

    tau41 = None

    tau48 = zeros((N, M))

    tau48 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau61 += 4 * np.einsum("ia->ia", tau48, optimize=True)

    tau67 -= 2 * np.einsum("ia->ia", tau48, optimize=True)

    tau48 = None

    tau50 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau51 -= 2 * np.einsum("al,likj->ijka", t1, tau50, optimize=True)

    tau50 = None

    tau51 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau61 += np.einsum("bajk,ijkb->ia", l2, tau51, optimize=True)

    tau51 = None

    tau52 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau61 -= 2 * np.einsum("bcji,jbca->ia", l2, tau52, optimize=True)

    tau52 = None

    tau54 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau61 -= 4 * np.einsum("jikb,kjba->ia", tau2, tau54, optimize=True)

    tau2 = None

    tau54 = None

    tau55 = zeros((M, M))

    tau55 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau56 += 2 * np.einsum("ab->ab", tau55, optimize=True)

    tau57 = zeros((N, M))

    tau57 += np.einsum("bc,ibca->ia", tau56, u[o, v, v, v], optimize=True)

    tau61 += 2 * np.einsum("ia->ia", tau57, optimize=True)

    tau67 -= np.einsum("ia->ia", tau57, optimize=True)

    tau57 = None

    tau68 += 4 * np.einsum("aj,ia->ij", t1, tau67, optimize=True)

    tau67 = None

    tau66 -= 4 * np.einsum("ac,jicb->ijab", tau56, u[o, o, v, v], optimize=True)

    tau68 += 4 * np.einsum("ab,iajb->ij", tau56, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bc,baic->ai", tau56, u[v, v, o, v], optimize=True) / 2

    tau56 = None

    tau63 = zeros((N, M, M, M))

    tau63 += 2 * np.einsum("ci,ab->iabc", t1, tau55, optimize=True)

    tau55 = None

    tau61 -= 4 * np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    r1 -= np.einsum("jb,baji->ai", tau61, t2, optimize=True) / 4

    tau61 = None

    tau63 += np.einsum("aj,cbij->iabc", l1, t2, optimize=True)

    r1 += np.einsum("ibcd,bacd->ai", tau63, u[v, v, v, v], optimize=True) / 2

    tau63 = None

    tau64 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau66 += 8 * np.einsum("cbki,kjca->ijab", l2, tau64, optimize=True)

    tau64 = None

    tau65 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau66 -= np.einsum("bakl,jikl->ijab", l2, tau65, optimize=True)

    tau65 = None

    tau66 -= 8 * np.einsum("ak,jikb->ijab", l1, u[o, o, o, v], optimize=True)

    tau66 -= 4 * np.einsum("ci,jcba->ijab", l1, u[o, v, v, v], optimize=True)

    tau68 += np.einsum("abkj,kiba->ij", t2, tau66, optimize=True)

    tau66 = None

    tau68 += 8 * np.einsum("ak,iajk->ij", l1, u[o, v, o, o], optimize=True)

    r1 -= np.einsum("aj,ji->ai", t1, tau68, optimize=True) / 8

    tau68 = None

    r1 += np.einsum("bj,abij->ai", l1, u[v, v, o, o], optimize=True)

    return r1


def amplitudes_intermediates_qccsd_t2_addition(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 -= np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau1 = zeros((N, N, M, M))

    tau1 -= np.einsum("acki,kjcb->ijab", t2, tau0, optimize=True)

    tau31 = zeros((N, N, M, M))

    tau31 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau89 = zeros((N, N, M, M))

    tau89 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau91 = zeros((N, N, M, M))

    tau91 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau98 = zeros((N, N, M, M))

    tau98 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau151 = zeros((N, N, N, M))

    tau151 += np.einsum("ijcb,kacb->ijka", tau1, u[o, v, v, v], optimize=True)

    tau154 = zeros((N, N, N, M))

    tau154 -= np.einsum("kjia->ijka", tau151, optimize=True)

    tau151 = None

    tau199 = zeros((N, N, M, M))

    tau199 -= 2 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau202 = zeros((N, N, N, N))

    tau202 += 2 * np.einsum("lkba,jiba->ijkl", tau1, u[o, o, v, v], optimize=True)

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum("ijab->ijab", tau0, optimize=True)

    tau79 = zeros((N, M, M, M))

    tau79 += np.einsum("bj,jiac->iabc", t1, tau0, optimize=True)

    tau80 = zeros((N, M, M, M))

    tau80 += 2 * np.einsum("iacb->iabc", tau79, optimize=True)

    tau79 = None

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau3 = zeros((N, N, N, N))

    tau3 -= np.einsum("aj,ikla->ijkl", t1, tau2, optimize=True)

    tau152 = zeros((N, N, N, N))

    tau152 -= np.einsum("lkij->ijkl", tau3, optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("klab,ijkl->abij", tau1, tau3, optimize=True)

    tau3 = None

    tau112 = zeros((N, N, N, M))

    tau112 += np.einsum("kjia->ijka", tau2, optimize=True)

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True)

    tau22 = zeros((N, N, M, M))

    tau22 -= 2 * np.einsum("jiba->ijab", tau4, optimize=True)

    tau134 = zeros((N, N, M, M))

    tau134 -= 4 * np.einsum("jiab->ijab", tau4, optimize=True)

    tau182 = zeros((N, N, M, M))

    tau182 -= 8 * np.einsum("jiab->ijab", tau4, optimize=True)

    tau200 = zeros((N, N, M, M))

    tau200 += 2 * np.einsum("jiab->ijab", tau4, optimize=True)

    tau4 = None

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau22 += 2 * np.einsum("ijba->ijab", tau5, optimize=True)

    tau134 -= 8 * np.einsum("ijba->ijab", tau5, optimize=True)

    tau182 -= 4 * np.einsum("ijba->ijab", tau5, optimize=True)

    tau200 += 2 * np.einsum("ijba->ijab", tau5, optimize=True)

    tau5 = None

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("jiab->ijab", tau6, optimize=True)

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("jiab->ijab", tau6, optimize=True)

    tau55 = zeros((N, N, M, M))

    tau55 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau204 = zeros((N, M, M, M))

    tau204 += np.einsum("aj,jibc->iabc", t1, tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau10 += np.einsum("ijab->ijab", tau7, optimize=True)

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum("ijab->ijab", tau7, optimize=True)

    tau57 = zeros((N, N, N, M))

    tau57 += np.einsum("bi,jkba->ijka", l1, tau7, optimize=True)

    tau60 = zeros((N, N, N, M))

    tau60 += np.einsum("ijka->ijka", tau57, optimize=True)

    tau57 = None

    tau72 = zeros((N, N, M, M))

    tau72 += np.einsum("ijab->ijab", tau7, optimize=True)

    tau108 = zeros((N, N, M, M))

    tau108 += np.einsum("ijab->ijab", tau7, optimize=True)

    tau156 = zeros((N, N, N, M))

    tau156 -= np.einsum("bj,ikab->ijka", t1, tau7, optimize=True)

    tau157 = zeros((N, N, N, M))

    tau157 += np.einsum("ijka->ijka", tau156, optimize=True)

    tau201 = zeros((N, N, N, M))

    tau201 -= np.einsum("kjia->ijka", tau156, optimize=True)

    tau156 = None

    tau165 = zeros((N, N, N, N))

    tau165 -= np.einsum("ikab,jlab->ijkl", tau0, tau7, optimize=True)

    tau179 = zeros((N, N, N, N))

    tau179 += 4 * np.einsum("ijlk->ijkl", tau165, optimize=True)

    tau165 = None

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("baji->ijab", t2, optimize=True)

    tau8 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("kica,kjcb->ijab", tau8, u[o, o, v, v], optimize=True)

    tau8 = None

    tau10 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau44 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau45 = zeros((N, N, M, M))

    tau45 += np.einsum("kjbc,kiac->ijab", tau0, tau44, optimize=True)

    tau44 = None

    tau62 = zeros((N, N, M, M))

    tau62 += 4 * np.einsum("ijba->ijab", tau45, optimize=True)

    tau45 = None

    tau55 += np.einsum("jiab->ijab", tau9, optimize=True)

    tau9 = None

    tau10 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("cbkj,kica->ijab", l2, tau10, optimize=True)

    tau10 = None

    tau22 += 2 * np.einsum("jiba->ijab", tau11, optimize=True)

    tau134 += 8 * np.einsum("ijba->ijab", tau11, optimize=True)

    tau182 += 8 * np.einsum("jiab->ijab", tau11, optimize=True)

    tau200 += 4 * np.einsum("jiba->ijab", tau11, optimize=True)

    tau11 = None

    tau12 = zeros((M, M))

    tau12 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau14 = zeros((M, M))

    tau14 += 2 * np.einsum("ab->ab", tau12, optimize=True)

    tau12 = None

    tau13 = zeros((M, M))

    tau13 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau14 += np.einsum("ab->ab", tau13, optimize=True)

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("ac,ijcb->ijab", tau14, u[o, o, v, v], optimize=True)

    tau22 -= np.einsum("jiba->ijab", tau15, optimize=True)

    tau134 -= 2 * np.einsum("jiab->ijab", tau15, optimize=True)

    tau182 -= 4 * np.einsum("jiab->ijab", tau15, optimize=True)

    tau200 += np.einsum("jiab->ijab", tau15, optimize=True)

    tau15 = None

    tau80 += np.einsum("bi,ac->iabc", t1, tau14, optimize=True)

    tau87 = zeros((N, N, M, M))

    tau87 += np.einsum("ca,cbij->ijab", tau14, t2, optimize=True)

    tau89 += 2 * np.einsum("jiab->ijab", tau87, optimize=True)

    tau199 += np.einsum("jiba->ijab", tau87, optimize=True)

    tau87 = None

    tau136 = zeros((M, M))

    tau136 += np.einsum("cd,cadb->ab", tau14, u[v, v, v, v], optimize=True)

    tau142 = zeros((M, M))

    tau142 -= 4 * np.einsum("ab->ab", tau136, optimize=True)

    tau136 = None

    tau145 = zeros((N, M))

    tau145 += np.einsum("bc,ibca->ia", tau14, u[o, v, v, v], optimize=True)

    tau148 = zeros((N, M))

    tau148 -= np.einsum("ia->ia", tau145, optimize=True)

    tau145 = None

    tau184 = zeros((N, N))

    tau184 += np.einsum("ab,iajb->ij", tau14, u[o, v, o, v], optimize=True)

    tau14 = None

    tau187 = zeros((N, N))

    tau187 += 4 * np.einsum("ij->ij", tau184, optimize=True)

    tau184 = None

    tau53 = zeros((N, M, M, M))

    tau53 -= np.einsum("bi,ac->iabc", t1, tau13, optimize=True)

    tau68 = zeros((N, N, M, M))

    tau68 += np.einsum("ac,ijbc->ijab", tau13, tau7, optimize=True)

    tau75 = zeros((N, N, M, M))

    tau75 += np.einsum("ijab->ijab", tau68, optimize=True)

    tau68 = None

    tau111 = zeros((N, M, M, M))

    tau111 -= np.einsum("ad,ibcd->iabc", tau13, u[o, v, v, v], optimize=True)

    tau13 = None

    tau117 = zeros((N, M, M, M))

    tau117 -= np.einsum("ibac->iabc", tau111, optimize=True)

    tau111 = None

    tau16 = zeros((N, N))

    tau16 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau18 = zeros((N, N))

    tau18 += 2 * np.einsum("ij->ij", tau16, optimize=True)

    tau16 = None

    tau17 = zeros((N, N))

    tau17 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau18 += np.einsum("ij->ij", tau17, optimize=True)

    tau17 = None

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("ik,kjab->ijab", tau18, u[o, o, v, v], optimize=True)

    tau22 += np.einsum("ijba->ijab", tau19, optimize=True)

    tau134 -= 4 * np.einsum("ijba->ijab", tau19, optimize=True)

    tau182 -= 2 * np.einsum("ijba->ijab", tau19, optimize=True)

    tau200 += np.einsum("ijba->ijab", tau19, optimize=True)

    tau19 = None

    tau49 = zeros((N, N, N, M))

    tau49 += np.einsum("aj,ik->ijka", t1, tau18, optimize=True)

    tau70 = zeros((N, N, N, M))

    tau70 += np.einsum("aj,ik->ijka", t1, tau18, optimize=True)

    tau88 = zeros((N, N, M, M))

    tau88 += np.einsum("ki,abkj->ijab", tau18, t2, optimize=True)

    tau89 -= 2 * np.einsum("jiba->ijab", tau88, optimize=True)

    tau88 = None

    tau138 = zeros((N, M))

    tau138 += np.einsum("aj,ji->ia", t1, tau18, optimize=True)

    tau139 = zeros((N, M))

    tau139 += np.einsum("ia->ia", tau138, optimize=True)

    tau138 = None

    tau141 = zeros((M, M))

    tau141 += np.einsum("ij,jaib->ab", tau18, u[o, v, o, v], optimize=True)

    tau142 += 4 * np.einsum("ab->ab", tau141, optimize=True)

    tau141 = None

    tau147 = zeros((N, M))

    tau147 += np.einsum("jk,kija->ia", tau18, u[o, o, o, v], optimize=True)

    tau148 -= np.einsum("ia->ia", tau147, optimize=True)

    tau147 = None

    tau168 = zeros((N, N, N, M))

    tau168 += np.einsum("ak,ij->ijka", t1, tau18, optimize=True)

    tau186 = zeros((N, N))

    tau186 += np.einsum("kl,likj->ij", tau18, u[o, o, o, o], optimize=True)

    tau187 -= 4 * np.einsum("ij->ij", tau186, optimize=True)

    tau186 = None

    tau20 = zeros((N, M))

    tau20 += np.einsum("bj,jiba->ia", t1, u[o, o, v, v], optimize=True)

    tau21 = zeros((N, M))

    tau21 += np.einsum("ia->ia", tau20, optimize=True)

    tau20 = None

    tau21 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau22 += 2 * np.einsum("ai,jb->ijab", l1, tau21, optimize=True)

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("cbkj,kiac->ijab", t2, tau22, optimize=True)

    tau22 = None

    tau62 -= 2 * np.einsum("ijab->ijab", tau23, optimize=True)

    tau23 = None

    tau134 += 8 * np.einsum("aj,ib->ijab", l1, tau21, optimize=True)

    tau182 += 8 * np.einsum("bi,ja->ijab", l1, tau21, optimize=True)

    tau200 += 4 * np.einsum("ai,jb->ijab", l1, tau21, optimize=True)

    tau21 = None

    tau202 += np.einsum("balk,ijab->ijkl", t2, tau200, optimize=True)

    tau200 = None

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("acik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau25 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau33 = zeros((N, N, M, M))

    tau33 += np.einsum("jkcb,ikca->ijab", tau0, tau25, optimize=True)

    tau62 += 4 * np.einsum("jiab->ijab", tau33, optimize=True)

    tau33 = None

    tau108 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau166 = zeros((N, N, M, M))

    tau166 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau24 = None

    tau26 = zeros((M, M, M, M))

    tau26 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("ijcd,cadb->ijab", tau25, tau26, optimize=True)

    tau25 = None

    tau62 -= 2 * np.einsum("jiab->ijab", tau27, optimize=True)

    tau27 = None

    tau78 = zeros((N, N, M, M))

    tau78 -= np.einsum("cabd,icjd->ijab", tau26, u[o, v, o, v], optimize=True)

    tau85 = zeros((N, N, M, M))

    tau85 -= np.einsum("ijba->ijab", tau78, optimize=True)

    tau78 = None

    tau28 = zeros((N, N, N, N))

    tau28 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau29 = zeros((N, N, M, M))

    tau29 -= np.einsum("ablk,lkji->ijab", t2, tau28, optimize=True)

    tau31 -= np.einsum("ijba->ijab", tau29, optimize=True)

    tau89 -= np.einsum("ijba->ijab", tau29, optimize=True)

    tau91 -= np.einsum("ijba->ijab", tau29, optimize=True)

    tau98 -= np.einsum("ijba->ijab", tau29, optimize=True)

    tau29 = None

    tau99 = zeros((N, N, N, M))

    tau99 += np.einsum("jlkb,ilba->ijka", tau2, tau98, optimize=True)

    tau98 = None

    tau100 = zeros((N, N, N, M))

    tau100 += np.einsum("kija->ijka", tau99, optimize=True)

    tau99 = None

    tau56 = zeros((N, N, M, M))

    tau56 += np.einsum("likj,klab->ijab", tau28, tau55, optimize=True)

    tau55 = None

    tau62 -= 2 * np.einsum("ijba->ijab", tau56, optimize=True)

    tau56 = None

    tau67 = zeros((N, N, M, M))

    tau67 += np.einsum("ikjl,lakb->ijab", tau28, u[o, v, o, v], optimize=True)

    tau75 -= np.einsum("jiba->ijab", tau67, optimize=True)

    tau67 = None

    tau69 = zeros((N, N, N, M))

    tau69 += np.einsum("al,iljk->ijka", t1, tau28, optimize=True)

    tau70 -= np.einsum("ikja->ijka", tau69, optimize=True)

    tau168 += np.einsum("ikja->ijka", tau69, optimize=True)

    tau170 = zeros((N, N, N, M))

    tau170 -= np.einsum("ikja->ijka", tau69, optimize=True)

    tau69 = None

    tau121 = zeros((N, N, M, M))

    tau121 -= np.einsum("jilk,lkab->ijab", tau28, u[o, o, v, v], optimize=True)

    tau130 = zeros((N, N, M, M))

    tau130 += np.einsum("ijba->ijab", tau121, optimize=True)

    tau134 += np.einsum("ijba->ijab", tau121, optimize=True)

    tau121 = None

    tau164 = zeros((N, N, N, N))

    tau164 -= np.einsum("imjn,nklm->ijkl", tau28, u[o, o, o, o], optimize=True)

    tau179 += 2 * np.einsum("ijkl->ijkl", tau164, optimize=True)

    tau164 = None

    tau30 = zeros((N, M))

    tau30 -= np.einsum("bj,baij->ia", l1, t2, optimize=True)

    tau31 += 4 * np.einsum("ai,jb->ijab", t1, tau30, optimize=True)

    tau31 += 4 * np.einsum("bj,ia->ijab", t1, tau30, optimize=True)

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("ikca,kjcb->ijab", tau31, u[o, o, v, v], optimize=True)

    tau31 = None

    tau62 -= np.einsum("jiba->ijab", tau32, optimize=True)

    tau32 = None

    tau66 = zeros((N, N, M, M))

    tau66 -= np.einsum("ic,jabc->ijab", tau30, u[o, v, v, v], optimize=True)

    tau75 += 2 * np.einsum("ijba->ijab", tau66, optimize=True)

    tau66 = None

    tau77 = zeros((N, N, M, M))

    tau77 -= np.einsum("ka,ikjb->ijab", tau30, u[o, o, o, v], optimize=True)

    tau85 += 2 * np.einsum("ijab->ijab", tau77, optimize=True)

    tau77 = None

    tau89 += 4 * np.einsum("ai,jb->ijab", t1, tau30, optimize=True)

    tau89 += 4 * np.einsum("bj,ia->ijab", t1, tau30, optimize=True)

    tau90 = zeros((N, N, M, M))

    tau90 += np.einsum("ikca,kbjc->ijab", tau89, u[o, v, o, v], optimize=True)

    tau89 = None

    tau102 = zeros((N, N, M, M))

    tau102 -= np.einsum("ijab->ijab", tau90, optimize=True)

    tau90 = None

    tau91 += 4 * np.einsum("ai,jb->ijab", t1, tau30, optimize=True)

    tau92 = zeros((N, N, M, M))

    tau92 += np.einsum("ikac,kjbc->ijab", tau7, tau91, optimize=True)

    tau7 = None

    tau102 -= np.einsum("ijba->ijab", tau92, optimize=True)

    tau92 = None

    tau97 = zeros((N, N, N, M))

    tau97 += np.einsum("ilba,ljkb->ijka", tau91, u[o, o, o, v], optimize=True)

    tau91 = None

    tau100 += np.einsum("ijka->ijka", tau97, optimize=True)

    tau97 = None

    tau101 = zeros((N, N, M, M))

    tau101 += np.einsum("bk,ikja->ijab", t1, tau100, optimize=True)

    tau100 = None

    tau102 -= np.einsum("ijba->ijab", tau101, optimize=True)

    tau101 = None

    tau105 = zeros((M, M, M, M))

    tau105 += np.einsum("ia,ibdc->abcd", tau30, u[o, v, v, v], optimize=True)

    tau119 = zeros((M, M, M, M))

    tau119 -= 2 * np.einsum("bcad->abcd", tau105, optimize=True)

    tau105 = None

    tau110 = zeros((N, M, M, M))

    tau110 += np.einsum("ja,ijcb->iabc", tau30, u[o, o, v, v], optimize=True)

    tau117 -= np.einsum("iacb->iabc", tau110, optimize=True)

    tau110 = None

    tau139 -= 2 * np.einsum("ia->ia", tau30, optimize=True)

    tau160 = zeros((N, N, M, M))

    tau160 -= np.einsum("ic,abjc->ijab", tau30, u[v, v, o, v], optimize=True)

    tau191 = zeros((N, N, M, M))

    tau191 += 8 * np.einsum("ijba->ijab", tau160, optimize=True)

    tau160 = None

    tau161 = zeros((N, M, M, M))

    tau161 += np.einsum("id,abdc->iabc", tau30, u[v, v, v, v], optimize=True)

    tau162 = zeros((N, N, M, M))

    tau162 += np.einsum("ci,jabc->ijab", t1, tau161, optimize=True)

    tau161 = None

    tau191 -= 8 * np.einsum("ijba->ijab", tau162, optimize=True)

    tau162 = None

    tau163 = zeros((N, N, N, N))

    tau163 += np.einsum("ia,kjla->ijkl", tau30, u[o, o, o, v], optimize=True)

    tau179 -= 2 * np.einsum("jikl->ijkl", tau163, optimize=True)

    tau194 = zeros((N, N, N, N))

    tau194 -= np.einsum("ikjl->ijkl", tau163, optimize=True)

    tau163 = None

    tau176 = zeros((N, N, N, M))

    tau176 += np.einsum("ib,kjab->ijka", tau30, u[o, o, v, v], optimize=True)

    tau177 = zeros((N, N, N, M))

    tau177 += np.einsum("ikja->ijka", tau176, optimize=True)

    tau193 = zeros((N, N, N, N))

    tau193 -= np.einsum("ai,jlka->ijkl", t1, tau176, optimize=True)

    tau176 = None

    tau194 += np.einsum("ilkj->ijkl", tau193, optimize=True)

    tau193 = None

    tau195 = zeros((N, N, N, M))

    tau195 += np.einsum("al,iljk->ijka", t1, tau194, optimize=True)

    tau194 = None

    tau196 = zeros((N, N, M, M))

    tau196 -= np.einsum("bk,ikja->ijab", t1, tau195, optimize=True)

    tau195 = None

    tau197 = zeros((N, N, M, M))

    tau197 += np.einsum("ijab->ijab", tau196, optimize=True)

    tau196 = None

    tau34 = zeros((M, M, M, M))

    tau34 += np.einsum("ai,ibcd->abcd", t1, u[o, v, v, v], optimize=True)

    tau36 = zeros((M, M, M, M))

    tau36 -= 2 * np.einsum("abdc->abcd", tau34, optimize=True)

    tau106 = zeros((M, M, M, M))

    tau106 += 2 * np.einsum("abdc->abcd", tau34, optimize=True)

    tau34 = None

    tau35 = zeros((M, M, M, M))

    tau35 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau36 -= np.einsum("badc->abcd", tau35, optimize=True)

    tau37 = zeros((N, N, M, M))

    tau37 += np.einsum("ijcd,acdb->ijab", tau0, tau36, optimize=True)

    tau36 = None

    tau62 -= 2 * np.einsum("ijba->ijab", tau37, optimize=True)

    tau37 = None

    tau106 += np.einsum("badc->abcd", tau35, optimize=True)

    tau35 = None

    tau38 = zeros((N, N, N, N))

    tau38 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau41 = zeros((N, N, N, N))

    tau41 -= 2 * np.einsum("ikjl->ijkl", tau38, optimize=True)

    tau128 = zeros((N, N, N, N))

    tau128 -= 4 * np.einsum("ljik->ijkl", tau38, optimize=True)

    tau173 = zeros((N, N, N, N))

    tau173 -= 2 * np.einsum("ikjl->ijkl", tau38, optimize=True)

    tau192 = zeros((N, N, M, M))

    tau192 += np.einsum("klba,ilkj->ijab", tau1, tau38, optimize=True)

    tau38 = None

    tau197 -= np.einsum("ijba->ijab", tau192, optimize=True)

    tau192 = None

    r2 -= np.einsum("ijab->abij", tau197, optimize=True)

    r2 += np.einsum("jiab->abij", tau197, optimize=True)

    tau197 = None

    tau39 = zeros((N, N, M, M))

    tau39 -= np.einsum("baji->ijab", t2, optimize=True)

    tau39 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau40 = zeros((N, N, N, N))

    tau40 += np.einsum("ijab,klab->ijkl", tau39, u[o, o, v, v], optimize=True)

    tau39 = None

    tau41 += np.einsum("likj->ijkl", tau40, optimize=True)

    tau40 = None

    tau42 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau43 = zeros((N, N, M, M))

    tau43 += np.einsum("ikjl,lkab->ijab", tau41, tau42, optimize=True)

    tau41 = None

    tau62 -= 2 * np.einsum("jiab->ijab", tau43, optimize=True)

    tau43 = None

    tau74 = zeros((N, N, M, M))

    tau74 += np.einsum("ijcd,cadb->ijab", tau42, u[v, v, v, v], optimize=True)

    tau75 += 2 * np.einsum("jiba->ijab", tau74, optimize=True)

    tau74 = None

    tau83 = zeros((N, N, M, M))

    tau83 += np.einsum("ikca,kcjb->ijab", tau42, u[o, v, o, v], optimize=True)

    tau85 += 2 * np.einsum("ijab->ijab", tau83, optimize=True)

    tau83 = None

    tau84 = zeros((N, N, M, M))

    tau84 += np.einsum("klab,likj->ijab", tau42, u[o, o, o, o], optimize=True)

    tau85 += 2 * np.einsum("ijba->ijab", tau84, optimize=True)

    tau84 = None

    tau46 = zeros((N, N, N, M))

    tau46 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau49 -= np.einsum("ikja->ijka", tau46, optimize=True)

    tau58 = zeros((N, N, N, M))

    tau58 += np.einsum("ikja->ijka", tau46, optimize=True)

    tau70 -= 2 * np.einsum("ikja->ijka", tau46, optimize=True)

    tau168 += 2 * np.einsum("ikja->ijka", tau46, optimize=True)

    tau170 -= 2 * np.einsum("ikja->ijka", tau46, optimize=True)

    tau47 = zeros((N, N, N, M))

    tau47 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau48 = zeros((N, N, N, M))

    tau48 -= np.einsum("bakl,lijb->ijka", t2, tau47, optimize=True)

    tau49 -= 2 * np.einsum("ijka->ijka", tau48, optimize=True)

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum("jlkb,ikla->ijab", tau2, tau49, optimize=True)

    tau62 += 2 * np.einsum("ijba->ijab", tau50, optimize=True)

    tau50 = None

    tau82 = zeros((N, N, M, M))

    tau82 += np.einsum("ikla,lkjb->ijab", tau49, u[o, o, o, v], optimize=True)

    tau49 = None

    tau85 += np.einsum("ijab->ijab", tau82, optimize=True)

    tau82 = None

    tau58 += np.einsum("ijka->ijka", tau48, optimize=True)

    tau59 = zeros((N, N, N, M))

    tau59 += np.einsum("iljb,lkba->ijka", tau58, u[o, o, v, v], optimize=True)

    tau58 = None

    tau60 -= np.einsum("ijka->ijka", tau59, optimize=True)

    tau59 = None

    tau61 = zeros((N, N, M, M))

    tau61 += np.einsum("bk,ijka->ijab", t1, tau60, optimize=True)

    tau60 = None

    tau62 += 4 * np.einsum("ijab->ijab", tau61, optimize=True)

    tau61 = None

    tau70 += 2 * np.einsum("ikja->ijka", tau48, optimize=True)

    tau71 = zeros((N, N, M, M))

    tau71 += np.einsum("ijkc,kacb->ijab", tau70, u[o, v, v, v], optimize=True)

    tau70 = None

    tau75 -= np.einsum("jiba->ijab", tau71, optimize=True)

    tau71 = None

    tau168 += 2 * np.einsum("ijka->ijka", tau48, optimize=True)

    tau169 = zeros((N, N, N, N))

    tau169 += np.einsum("imja,mkla->ijkl", tau168, u[o, o, o, v], optimize=True)

    tau168 = None

    tau179 -= 2 * np.einsum("ijkl->ijkl", tau169, optimize=True)

    tau169 = None

    tau170 -= 2 * np.einsum("ijka->ijka", tau48, optimize=True)

    tau48 = None

    tau171 = zeros((N, N, N, N))

    tau171 += np.einsum("imja,kmla->ijkl", tau170, tau2, optimize=True)

    tau170 = None

    tau179 += 2 * np.einsum("iljk->ijkl", tau171, optimize=True)

    tau171 = None

    tau52 = zeros((N, M, M, M))

    tau52 += np.einsum("bckj,kjia->iabc", t2, tau47, optimize=True)

    tau53 += np.einsum("iacb->iabc", tau52, optimize=True)

    tau80 -= np.einsum("iacb->iabc", tau52, optimize=True)

    tau52 = None

    tau137 = zeros((N, M))

    tau137 += np.einsum("bakj,kjib->ia", t2, tau47, optimize=True)

    tau47 = None

    tau139 += np.einsum("ia->ia", tau137, optimize=True)

    tau137 = None

    tau140 = zeros((M, M))

    tau140 += np.einsum("ic,iacb->ab", tau139, u[o, v, v, v], optimize=True)

    tau142 += 4 * np.einsum("ab->ab", tau140, optimize=True)

    tau140 = None

    tau146 = zeros((N, M))

    tau146 += np.einsum("jb,jiba->ia", tau139, u[o, o, v, v], optimize=True)

    tau148 -= np.einsum("ia->ia", tau146, optimize=True)

    tau146 = None

    tau185 = zeros((N, N))

    tau185 += np.einsum("ka,kija->ij", tau139, u[o, o, o, v], optimize=True)

    tau139 = None

    tau187 += 4 * np.einsum("ij->ij", tau185, optimize=True)

    tau185 = None

    tau51 = zeros((N, M, M, M))

    tau51 += np.einsum("aj,bcij->iabc", l1, t2, optimize=True)

    tau53 += 2 * np.einsum("iacb->iabc", tau51, optimize=True)

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("ikjc,kabc->ijab", tau2, tau53, optimize=True)

    tau2 = None

    tau53 = None

    tau62 += 2 * np.einsum("jiab->ijab", tau54, optimize=True)

    tau54 = None

    tau63 = zeros((N, N, M, M))

    tau63 += np.einsum("cbkj,kica->ijab", t2, tau62, optimize=True)

    tau62 = None

    tau102 -= np.einsum("ijab->ijab", tau63, optimize=True)

    tau63 = None

    tau80 -= 2 * np.einsum("iacb->iabc", tau51, optimize=True)

    tau51 = None

    tau81 = zeros((N, N, M, M))

    tau81 += np.einsum("kabc,kijc->ijab", tau80, u[o, o, o, v], optimize=True)

    tau80 = None

    tau85 -= np.einsum("ijba->ijab", tau81, optimize=True)

    tau81 = None

    tau86 = zeros((N, N, M, M))

    tau86 += np.einsum("cbkj,kiac->ijab", t2, tau85, optimize=True)

    tau85 = None

    tau102 -= 2 * np.einsum("jiab->ijab", tau86, optimize=True)

    tau86 = None

    tau64 = zeros((N, N, N, M))

    tau64 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau65 = zeros((N, N, M, M))

    tau65 += np.einsum("ak,ikjb->ijab", l1, tau64, optimize=True)

    tau64 = None

    tau75 -= np.einsum("ijab->ijab", tau65, optimize=True)

    tau65 = None

    tau72 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau73 = zeros((N, N, M, M))

    tau73 += np.einsum("kiac,kjbc->ijab", tau42, tau72, optimize=True)

    tau72 = None

    tau75 -= 2 * np.einsum("ijab->ijab", tau73, optimize=True)

    tau73 = None

    tau76 = zeros((N, N, M, M))

    tau76 += np.einsum("cbkj,ikca->ijab", t2, tau75, optimize=True)

    tau75 = None

    tau102 -= 2 * np.einsum("ijba->ijab", tau76, optimize=True)

    tau76 = None

    tau93 = zeros((N, N, M, M))

    tau93 += np.einsum("ak,ibjk->ijab", l1, u[o, v, o, o], optimize=True)

    tau95 = zeros((N, N, M, M))

    tau95 += np.einsum("ijab->ijab", tau93, optimize=True)

    tau93 = None

    tau94 = zeros((N, N, M, M))

    tau94 += np.einsum("ci,acjb->ijab", l1, u[v, v, o, v], optimize=True)

    tau95 += np.einsum("ijba->ijab", tau94, optimize=True)

    tau94 = None

    tau96 = zeros((N, N, M, M))

    tau96 += np.einsum("cbkj,kica->ijab", t2, tau95, optimize=True)

    tau95 = None

    tau102 -= 4 * np.einsum("jiba->ijab", tau96, optimize=True)

    tau96 = None

    r2 -= np.einsum("ijab->abij", tau102, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau102, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau102, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau102, optimize=True) / 4

    tau102 = None

    tau103 = zeros((N, M, M, M))

    tau103 -= np.einsum("adij,jbdc->iabc", t2, u[o, v, v, v], optimize=True)

    tau104 = zeros((M, M, M, M))

    tau104 += np.einsum("ai,ibcd->abcd", l1, tau103, optimize=True)

    tau103 = None

    tau119 -= 4 * np.einsum("abcd->abcd", tau104, optimize=True)

    tau104 = None

    tau106 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau107 = zeros((M, M, M, M))

    tau107 += np.einsum("eafb,ecfd->abcd", tau106, tau26, optimize=True)

    tau106 = None

    tau119 += np.einsum("cdab->abcd", tau107, optimize=True)

    tau107 = None

    tau108 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau109 = zeros((M, M, M, M))

    tau109 += np.einsum("ijcd,ijab->abcd", tau108, tau42, optimize=True)

    tau42 = None

    tau108 = None

    tau119 += 4 * np.einsum("abcd->abcd", tau109, optimize=True)

    tau109 = None

    tau112 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau113 = zeros((N, M, M, M))

    tau113 += np.einsum("kjbc,jika->iabc", tau0, tau112, optimize=True)

    tau112 = None

    tau117 -= 2 * np.einsum("icba->iabc", tau113, optimize=True)

    tau113 = None

    tau114 = zeros((N, M, M, M))

    tau114 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau115 = zeros((N, M, M, M))

    tau115 += np.einsum("iacb->iabc", tau114, optimize=True)

    tau122 = zeros((N, M, M, M))

    tau122 += np.einsum("iacb->iabc", tau114, optimize=True)

    tau198 = zeros((M, M, M, M))

    tau198 += np.einsum("ai,ibcd->abcd", t1, tau114, optimize=True)

    tau114 = None

    tau115 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau116 = zeros((N, M, M, M))

    tau116 += np.einsum("idea,dbec->iabc", tau115, tau26, optimize=True)

    tau26 = None

    tau115 = None

    tau117 -= np.einsum("icba->iabc", tau116, optimize=True)

    tau116 = None

    tau118 = zeros((M, M, M, M))

    tau118 += np.einsum("di,iabc->abcd", t1, tau117, optimize=True)

    tau117 = None

    tau119 += 2 * np.einsum("cadb->abcd", tau118, optimize=True)

    tau118 = None

    tau120 = zeros((N, N, M, M))

    tau120 += np.einsum("dcij,cabd->ijab", t2, tau119, optimize=True)

    tau119 = None

    tau159 = zeros((N, N, M, M))

    tau159 += 2 * np.einsum("jiab->ijab", tau120, optimize=True)

    tau120 = None

    tau122 -= 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau123 = zeros((M, M, M, M))

    tau123 += np.einsum("di,iabc->abcd", t1, tau122, optimize=True)

    tau122 = None

    tau124 = zeros((M, M, M, M))

    tau124 -= np.einsum("adcb->abcd", tau123, optimize=True)

    tau123 = None

    tau124 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau125 = zeros((N, N, M, M))

    tau125 += np.einsum("cdij,cdab->ijab", l2, tau124, optimize=True)

    tau124 = None

    tau130 -= 2 * np.einsum("jiba->ijab", tau125, optimize=True)

    tau134 -= 2 * np.einsum("jiba->ijab", tau125, optimize=True)

    tau125 = None

    tau135 = zeros((M, M))

    tau135 += np.einsum("cbji,ijca->ab", t2, tau134, optimize=True)

    tau134 = None

    tau142 += np.einsum("ba->ab", tau135, optimize=True)

    tau135 = None

    tau126 = zeros((N, N, M, M))

    tau126 += np.einsum("baji->ijab", t2, optimize=True)

    tau126 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau127 = zeros((N, N, N, N))

    tau127 += np.einsum("ijab,klab->ijkl", tau126, u[o, o, v, v], optimize=True)

    tau126 = None

    tau128 += np.einsum("lkji->ijkl", tau127, optimize=True)

    tau127 = None

    tau128 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau129 = zeros((N, N, M, M))

    tau129 += np.einsum("abkl,ijkl->ijab", l2, tau128, optimize=True)

    tau128 = None

    tau130 -= np.einsum("jiba->ijab", tau129, optimize=True)

    tau131 = zeros((N, N, M, M))

    tau131 += np.einsum("cbkj,ikca->ijab", t2, tau130, optimize=True)

    tau130 = None

    tau132 = zeros((N, N, M, M))

    tau132 += np.einsum("cbkj,kica->ijab", t2, tau131, optimize=True)

    tau131 = None

    tau159 -= 2 * np.einsum("ijab->ijab", tau132, optimize=True)

    tau132 = None

    tau182 -= np.einsum("jiba->ijab", tau129, optimize=True)

    tau129 = None

    tau183 = zeros((N, N))

    tau183 += np.einsum("bakj,kiab->ij", t2, tau182, optimize=True)

    tau182 = None

    tau187 += np.einsum("ij->ij", tau183, optimize=True)

    tau183 = None

    tau133 = zeros((M, M))

    tau133 += np.einsum("ci,acib->ab", l1, u[v, v, o, v], optimize=True)

    tau142 += 8 * np.einsum("ab->ab", tau133, optimize=True)

    tau133 = None

    tau143 = zeros((N, N, M, M))

    tau143 += np.einsum("ac,cbij->ijab", tau142, t2, optimize=True)

    tau142 = None

    tau159 += np.einsum("jiba->ijab", tau143, optimize=True)

    tau143 = None

    tau144 = zeros((N, M))

    tau144 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau148 -= 2 * np.einsum("ia->ia", tau144, optimize=True)

    tau144 = None

    tau149 = zeros((M, M))

    tau149 += np.einsum("bi,ia->ab", t1, tau148, optimize=True)

    tau150 = zeros((N, N, M, M))

    tau150 += np.einsum("ca,cbij->ijab", tau149, t2, optimize=True)

    tau149 = None

    tau159 -= 4 * np.einsum("jiab->ijab", tau150, optimize=True)

    tau150 = None

    tau189 = zeros((N, N))

    tau189 += np.einsum("aj,ia->ij", t1, tau148, optimize=True)

    tau148 = None

    tau190 = zeros((N, N, M, M))

    tau190 += np.einsum("ki,abkj->ijab", tau189, t2, optimize=True)

    tau189 = None

    tau191 -= 4 * np.einsum("ijba->ijab", tau190, optimize=True)

    tau190 = None

    tau152 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau153 = zeros((N, N, N, M))

    tau153 += np.einsum("la,lijk->ijka", tau30, tau152, optimize=True)

    tau154 -= np.einsum("ikja->ijka", tau153, optimize=True)

    tau153 = None

    tau155 = zeros((N, N, M, M))

    tau155 += np.einsum("bk,kija->ijab", t1, tau154, optimize=True)

    tau154 = None

    tau159 += 8 * np.einsum("ijba->ijab", tau155, optimize=True)

    tau155 = None

    tau202 += 2 * np.einsum("im,mjlk->ijkl", tau18, tau152, optimize=True)

    tau152 = None

    tau18 = None

    tau157 -= np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau158 = zeros((N, N, M, M))

    tau158 += np.einsum("kb,ijka->ijab", tau30, tau157, optimize=True)

    tau157 = None

    tau30 = None

    tau159 += 8 * np.einsum("ijba->ijab", tau158, optimize=True)

    tau158 = None

    r2 += np.einsum("jiab->abij", tau159, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau159, optimize=True) / 8

    tau159 = None

    tau166 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau167 = zeros((N, N, N, N))

    tau167 += np.einsum("klab,ijab->ijkl", tau0, tau166, optimize=True)

    tau0 = None

    tau166 = None

    tau179 += 4 * np.einsum("lkij->ijkl", tau167, optimize=True)

    tau167 = None

    tau172 = zeros((N, N, N, N))

    tau172 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau173 -= np.einsum("likj->ijkl", tau172, optimize=True)

    tau174 = zeros((N, N, N, N))

    tau174 += np.einsum("imjn,nkml->ijkl", tau173, tau28, optimize=True)

    tau28 = None

    tau173 = None

    tau179 -= np.einsum("jkil->ijkl", tau174, optimize=True)

    tau174 = None

    tau203 = zeros((N, N, N, N))

    tau203 += np.einsum("lkji->ijkl", tau172, optimize=True)

    tau172 = None

    tau175 = zeros((N, N, N, M))

    tau175 += np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    tau177 += 2 * np.einsum("jkia->ijka", tau175, optimize=True)

    tau175 = None

    tau178 = zeros((N, N, N, N))

    tau178 += np.einsum("al,ijka->ijkl", t1, tau177, optimize=True)

    tau177 = None

    tau179 -= 2 * np.einsum("likj->ijkl", tau178, optimize=True)

    tau178 = None

    tau180 = zeros((N, N, M, M))

    tau180 += np.einsum("ablk,kilj->ijab", t2, tau179, optimize=True)

    tau179 = None

    tau191 += 2 * np.einsum("ijba->ijab", tau180, optimize=True)

    tau180 = None

    tau181 = zeros((N, N))

    tau181 += np.einsum("ak,iajk->ij", l1, u[o, v, o, o], optimize=True)

    tau187 += 8 * np.einsum("ij->ij", tau181, optimize=True)

    tau181 = None

    tau188 = zeros((N, N, M, M))

    tau188 += np.einsum("ki,abkj->ijab", tau187, t2, optimize=True)

    tau187 = None

    tau191 += np.einsum("jiba->ijab", tau188, optimize=True)

    tau188 = None

    r2 += np.einsum("ijba->abij", tau191, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau191, optimize=True) / 8

    tau191 = None

    tau198 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    r2 += np.einsum("bacd,ijcd->abij", tau198, tau199, optimize=True) / 2

    tau198 = None

    tau199 = None

    tau201 -= np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau202 += 4 * np.einsum("ai,jlka->ijkl", l1, tau201, optimize=True)

    tau201 = None

    r2 -= np.einsum("balk,klji->abij", t2, tau202, optimize=True) / 4

    tau202 = None

    tau203 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 -= np.einsum("klab,klji->abij", tau1, tau203, optimize=True) / 2

    tau1 = None

    tau203 = None

    tau204 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    r2 += np.einsum("kbac,kjic->abij", tau204, tau46, optimize=True)

    tau204 = None

    tau46 = None

    return r2
