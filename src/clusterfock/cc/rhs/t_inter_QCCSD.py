import numpy as np
from clusterfock.cc.rhs.t_inter_CCSD import amplitudes_intermediates_ccsd


def amplitudes_intermediates_qccsd(t1, t2, l1, l2, u, f, v, o):
    r1, r2 = amplitudes_intermediates_ccsd(t1, t2, u, f, v, o)

    amplitudes_intermediates_qccsd_t1_addition(r1, t1, t2, l1, l2, u, f, v, o)
    amplitudes_intermediates_qccsd_t2_addition(r2, t1, t2, l1, l2, u, f, v, o)

    return r1, r2


def amplitudes_intermediates_qccsd_t1_addition(r1, t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau24 = np.zeros((N, N, N, M))

    tau24 += np.einsum("al,iljk->ijka", t1, tau0, optimize=True)

    tau28 = np.zeros((N, N, N, M))

    tau28 -= np.einsum("ikja->ijka", tau24, optimize=True)

    tau42 = np.zeros((N, N, N, M))

    tau42 += np.einsum("ikja->ijka", tau24, optimize=True)

    tau47 = np.zeros((N, N, N, M))

    tau47 += np.einsum("ikja->ijka", tau24, optimize=True)

    tau53 = np.zeros((N, N, N, M))

    tau53 -= np.einsum("ikja->ijka", tau24, optimize=True)

    tau24 = None

    tau41 = np.zeros((N, N, N, M))

    tau41 += 4 * np.einsum("ilkm,jmla->ijka", tau0, u[o, o, o, v], optimize=True)

    tau46 = np.zeros((N, N, M, M))

    tau46 += np.einsum("balk,lkji->ijab", t2, tau0, optimize=True)

    tau61 = np.zeros((N, M))

    tau61 += np.einsum("ijlk,lkja->ia", tau0, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("lkij,jalk->ai", tau0, u[o, v, o, o], optimize=True) / 4

    tau1 = np.zeros((N, M))

    tau1 += np.einsum("bj,abij->ia", l1, t2, optimize=True)

    tau41 += 4 * np.einsum("kb,jiab->ijka", tau1, u[o, o, v, v], optimize=True)

    tau46 += 8 * np.einsum("ai,jb->ijab", t1, tau1, optimize=True)

    tau46 += 8 * np.einsum("bj,ia->ijab", t1, tau1, optimize=True)

    tau58 = np.zeros((N, M))

    tau58 -= 2 * np.einsum("ia->ia", tau1, optimize=True)

    r1 += np.einsum("ab,ib->ai", f[v, v], tau1, optimize=True)

    tau1 = None

    tau2 = np.zeros((N, N, N, M))

    tau2 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau3 = np.zeros((N, N, N, M))

    tau3 -= np.einsum("ablk,lijb->ijka", t2, tau2, optimize=True)

    tau28 += 2 * np.einsum("ikja->ijka", tau3, optimize=True)

    tau42 += 4 * np.einsum("ijka->ijka", tau3, optimize=True)

    r1 -= np.einsum("kjib,jakb->ai", tau3, u[o, v, o, v], optimize=True)

    tau3 = None

    tau43 = np.zeros((N, M))

    tau43 += np.einsum("bakj,kjib->ia", t2, tau2, optimize=True)

    tau45 = np.zeros((N, M))

    tau45 += np.einsum("ia->ia", tau43, optimize=True)

    tau58 += np.einsum("ia->ia", tau43, optimize=True)

    tau43 = None

    tau49 = np.zeros((N, N, N, N))

    tau49 += np.einsum("ak,ijla->ijkl", t1, tau2, optimize=True)

    tau61 -= 2 * np.einsum("iljk,kjla->ia", tau49, u[o, o, o, v], optimize=True)

    tau49 = None

    tau4 = np.zeros((N, M, M, M))

    tau4 += np.einsum("di,abcd->iabc", t1, u[v, v, v, v], optimize=True)

    tau11 = np.zeros((N, M, M, M))

    tau11 += np.einsum("ibac->iabc", tau4, optimize=True)

    tau52 = np.zeros((N, M, M, M))

    tau52 += np.einsum("ibac->iabc", tau4, optimize=True)

    tau4 = None

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("jiab->ijab", tau5, optimize=True)

    tau64 = np.zeros((N, N, M, M))

    tau64 += np.einsum("jiab->ijab", tau5, optimize=True)

    tau5 = None

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau8 += 2 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau54 = np.zeros((N, N, M, M))

    tau54 -= np.einsum("jiab->ijab", tau6, optimize=True)

    tau64 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau6 = None

    tau7 = np.zeros((N, N, M, M))

    tau7 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau7 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau8 += np.einsum("kica,kjcb->ijab", tau7, u[o, o, v, v], optimize=True)

    tau53 += 2 * np.einsum("likb,ljba->ijka", tau2, tau7, optimize=True)

    tau7 = None

    tau8 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau11 -= np.einsum("bj,ijac->iabc", t1, tau8, optimize=True)

    tau8 = None

    tau9 = np.zeros((N, M))

    tau9 += np.einsum("bj,jiba->ia", t1, u[o, o, v, v], optimize=True)

    tau10 = np.zeros((N, M))

    tau10 += np.einsum("ia->ia", tau9, optimize=True)

    tau9 = None

    tau10 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau11 -= np.einsum("jc,baji->iabc", tau10, t2, optimize=True)

    tau21 = np.zeros((N, N, N, M))

    tau21 += np.einsum("kb,baij->ijka", tau10, t2, optimize=True)

    tau22 = np.zeros((N, N, N, M))

    tau22 += 2 * np.einsum("jika->ijka", tau21, optimize=True)

    tau51 = np.zeros((N, N, N, M))

    tau51 += 2 * np.einsum("kjia->ijka", tau21, optimize=True)

    tau21 = None

    tau66 = np.zeros((N, N, M, M))

    tau66 += 8 * np.einsum("bi,ja->ijab", l1, tau10, optimize=True)

    tau10 = None

    tau11 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau11 += 2 * np.einsum("bdji,jadc->iabc", t2, u[o, v, v, v], optimize=True)

    tau41 -= 2 * np.einsum("bcji,kbca->ijka", l2, tau11, optimize=True)

    tau11 = None

    tau12 = np.zeros((N, N, N, M))

    tau12 += np.einsum("bi,jakb->ijka", t1, u[o, v, o, v], optimize=True)

    tau15 = np.zeros((N, N, N, M))

    tau15 += np.einsum("ijka->ijka", tau12, optimize=True)

    tau51 -= 4 * np.einsum("kija->ijka", tau12, optimize=True)

    tau12 = None

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum("baji->ijab", t2, optimize=True)

    tau13 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau14 = np.zeros((N, N, N, M))

    tau14 += np.einsum("lkba,lijb->ijka", tau13, u[o, o, o, v], optimize=True)

    tau15 -= np.einsum("jkia->ijka", tau14, optimize=True)

    tau14 = None

    tau22 += 2 * np.einsum("ikja->ijka", tau15, optimize=True)

    tau22 -= 2 * np.einsum("jkia->ijka", tau15, optimize=True)

    tau15 = None

    tau64 += np.einsum("kica,kjcb->ijab", tau13, u[o, o, v, v], optimize=True)

    tau13 = None

    tau16 = np.zeros((N, N, N, M))

    tau16 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau17 = np.zeros((N, N, N, M))

    tau17 -= np.einsum("bajl,ilkb->ijka", t2, tau16, optimize=True)

    tau22 -= 2 * np.einsum("ijka->ijka", tau17, optimize=True)

    tau22 += 2 * np.einsum("jika->ijka", tau17, optimize=True)

    tau17 = None

    tau29 = np.zeros((N, N, N, M))

    tau29 -= np.einsum("ikja->ijka", tau16, optimize=True)

    tau32 = np.zeros((N, N, N, M))

    tau32 += np.einsum("kjia->ijka", tau16, optimize=True)

    tau16 = None

    tau18 = np.zeros((N, N, M, M))

    tau18 -= np.einsum("baji->ijab", t2, optimize=True)

    tau18 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau65 = np.zeros((N, N, N, N))

    tau65 -= np.einsum("lkab,jiab->ijkl", tau18, u[o, o, v, v], optimize=True)

    tau19 = np.zeros((N, M, M, M))

    tau19 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau20 = np.zeros((N, M, M, M))

    tau20 += np.einsum("iacb->iabc", tau19, optimize=True)

    tau19 = None

    tau20 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau22 -= np.einsum("jibc,kabc->ijka", tau18, tau20, optimize=True)

    tau18 = None

    tau22 += 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau22 -= 2 * np.einsum("al,klji->ijka", t1, u[o, o, o, o], optimize=True)

    tau41 += 4 * np.einsum("balj,lkib->ijka", l2, tau22, optimize=True)

    tau22 = None

    tau23 = np.zeros((N, N, N, M))

    tau23 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau28 -= 2 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau42 += 4 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau46 += 2 * np.einsum("ak,kjib->ijab", t1, tau42, optimize=True)

    tau42 = None

    tau47 += 2 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau53 -= 2 * np.einsum("ikja->ijka", tau23, optimize=True)

    tau23 = None

    tau25 = np.zeros((N, N))

    tau25 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau27 = np.zeros((N, N))

    tau27 += 2 * np.einsum("ij->ij", tau25, optimize=True)

    tau25 = None

    tau26 = np.zeros((N, N))

    tau26 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau27 += np.einsum("ij->ij", tau26, optimize=True)

    tau26 = None

    tau28 += np.einsum("aj,ik->ijka", t1, tau27, optimize=True)

    tau41 -= 4 * np.einsum("iklb,ljba->ijka", tau28, u[o, o, v, v], optimize=True)

    tau28 = None

    tau41 -= 4 * np.einsum("il,ljka->ijka", tau27, u[o, o, o, v], optimize=True)

    tau44 = np.zeros((N, M))

    tau44 += np.einsum("aj,ji->ia", t1, tau27, optimize=True)

    tau45 += np.einsum("ia->ia", tau44, optimize=True)

    tau46 -= 4 * np.einsum("ai,jb->ijab", t1, tau45, optimize=True)

    tau45 = None

    tau58 += np.einsum("ia->ia", tau44, optimize=True)

    tau44 = None

    tau59 = np.zeros((N, M))

    tau59 += np.einsum("jb,jiba->ia", tau58, u[o, o, v, v], optimize=True)

    tau61 += 2 * np.einsum("ia->ia", tau59, optimize=True)

    tau67 = np.zeros((N, M))

    tau67 -= np.einsum("ia->ia", tau59, optimize=True)

    tau59 = None

    tau68 = np.zeros((N, N))

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

    tau60 = np.zeros((N, M))

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

    tau30 = np.zeros((N, N, M, M))

    tau30 -= np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum("ijab->ijab", tau30, optimize=True)

    tau46 -= 8 * np.einsum("caik,kjcb->ijab", t2, tau30, optimize=True)

    tau63 = np.zeros((N, M, M, M))

    tau63 += 2 * np.einsum("bj,jiac->iabc", t1, tau30, optimize=True)

    tau30 = None

    tau31 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau41 += 4 * np.einsum("ljib,lkab->ijka", tau29, tau31, optimize=True)

    tau41 -= 8 * np.einsum("ikbc,jbca->ijka", tau31, u[o, v, v, v], optimize=True)

    tau61 -= 4 * np.einsum("ijbc,jbca->ia", tau31, u[o, v, v, v], optimize=True)

    r1 += np.einsum("jibc,bajc->ai", tau31, u[v, v, o, v], optimize=True)

    tau31 = None

    tau32 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau41 -= np.einsum("jilm,lmka->ijka", tau0, tau32, optimize=True)

    tau0 = None

    tau32 = None

    tau33 = np.zeros((N, N, N, N))

    tau33 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau34 = np.zeros((N, N, N, N))

    tau34 += 2 * np.einsum("kjil->ijkl", tau33, optimize=True)

    tau50 = np.zeros((N, N, N, N))

    tau50 -= 2 * np.einsum("kjil->ijkl", tau33, optimize=True)

    tau65 -= 4 * np.einsum("ljik->ijkl", tau33, optimize=True)

    tau33 = None

    tau34 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau34 += np.einsum("balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau41 -= 2 * np.einsum("al,jilk->ijka", l1, tau34, optimize=True)

    tau34 = None

    tau35 = np.zeros((M, M))

    tau35 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau41 -= 2 * np.einsum("ab,kjib->ijka", tau35, tau29, optimize=True)

    tau29 = None

    tau46 += 4 * np.einsum("ca,bcij->ijab", tau35, t2, optimize=True)

    r1 += np.einsum("ijcb,jabc->ai", tau46, u[o, v, v, v], optimize=True) / 8

    tau46 = None

    tau56 = np.zeros((M, M))

    tau56 += np.einsum("ab->ab", tau35, optimize=True)

    tau63 -= np.einsum("bi,ac->iabc", t1, tau35, optimize=True)

    tau35 = None

    r1 -= np.einsum("ibdc,bacd->ai", tau63, u[v, v, v, v], optimize=True) / 2

    tau63 = None

    tau36 = np.zeros((N, N))

    tau36 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau40 = np.zeros((N, N))

    tau40 += 2 * np.einsum("ij->ij", tau36, optimize=True)

    tau36 = None

    tau37 = np.zeros((N, N))

    tau37 -= np.einsum("ak,kija->ij", t1, u[o, o, o, v], optimize=True)

    tau40 += 2 * np.einsum("ij->ij", tau37, optimize=True)

    tau37 = None

    tau38 = np.zeros((N, N, M, M))

    tau38 += np.einsum("baji->ijab", t2, optimize=True)

    tau38 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau39 = np.zeros((N, N))

    tau39 += np.einsum("kjab,kiab->ij", tau38, u[o, o, v, v], optimize=True)

    tau40 += np.einsum("ij->ij", tau39, optimize=True)

    tau39 = None

    tau51 += np.einsum("iabc,kjbc->ijka", tau20, tau38, optimize=True)

    tau20 = None

    tau38 = None

    tau40 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    tau41 += 4 * np.einsum("aj,ik->ijka", l1, tau40, optimize=True)

    tau61 += 2 * np.einsum("aj,ij->ia", l1, tau40, optimize=True)

    tau40 = None

    tau41 -= 8 * np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bajk,kjib->ai", t2, tau41, optimize=True) / 8

    tau41 = None

    tau48 = np.zeros((N, M))

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

    tau55 = np.zeros((M, M))

    tau55 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau56 += 2 * np.einsum("ab->ab", tau55, optimize=True)

    tau57 = np.zeros((N, M))

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

    tau62 = np.zeros((N, M, M, M))

    tau62 += 2 * np.einsum("ci,ab->iabc", t1, tau55, optimize=True)

    tau55 = None

    tau61 -= 4 * np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    r1 -= np.einsum("jb,baji->ai", tau61, t2, optimize=True) / 4

    tau61 = None

    tau62 += np.einsum("aj,cbij->iabc", l1, t2, optimize=True)

    r1 += np.einsum("ibcd,bacd->ai", tau62, u[v, v, v, v], optimize=True) / 2

    tau62 = None

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

    # return r1


def amplitudes_intermediates_qccsd_t2_addition(r2, t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True)

    tau18 = np.zeros((N, N, M, M))

    tau18 -= 2 * np.einsum("jiba->ijab", tau0, optimize=True)

    tau128 = np.zeros((N, N, M, M))

    tau128 -= 8 * np.einsum("jiab->ijab", tau0, optimize=True)

    tau173 = np.zeros((N, N, M, M))

    tau173 -= 4 * np.einsum("jiab->ijab", tau0, optimize=True)

    tau200 = np.zeros((N, N, M, M))

    tau200 += 2 * np.einsum("jiab->ijab", tau0, optimize=True)

    tau0 = None

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau18 += 2 * np.einsum("ijba->ijab", tau1, optimize=True)

    tau128 -= 4 * np.einsum("ijba->ijab", tau1, optimize=True)

    tau173 -= 8 * np.einsum("ijba->ijab", tau1, optimize=True)

    tau200 += 2 * np.einsum("ijba->ijab", tau1, optimize=True)

    tau1 = None

    tau2 = np.zeros((N, N, M, M))

    tau2 -= np.einsum("ak,kijb->ijab", t1, u[o, o, o, v], optimize=True)

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("jiab->ijab", tau2, optimize=True)

    tau42 = np.zeros((N, N, M, M))

    tau42 += np.einsum("jiab->ijab", tau2, optimize=True)

    tau54 = np.zeros((N, N, M, M))

    tau54 += np.einsum("ijab->ijab", tau2, optimize=True)

    tau204 = np.zeros((N, M, M, M))

    tau204 += np.einsum("aj,jibc->iabc", t1, tau2, optimize=True)

    tau2 = None

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau6 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau21 = np.zeros((N, N, M, M))

    tau21 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau56 = np.zeros((N, N, N, M))

    tau56 += np.einsum("bi,jkba->ijka", l1, tau3, optimize=True)

    tau59 = np.zeros((N, N, N, M))

    tau59 += np.einsum("ijka->ijka", tau56, optimize=True)

    tau56 = None

    tau71 = np.zeros((N, N, M, M))

    tau71 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau151 = np.zeros((N, N, M, M))

    tau151 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau188 = np.zeros((N, N, N, M))

    tau188 -= np.einsum("bj,ikab->ijka", t1, tau3, optimize=True)

    tau189 = np.zeros((N, N, N, M))

    tau189 += np.einsum("ijka->ijka", tau188, optimize=True)

    tau201 = np.zeros((N, N, N, M))

    tau201 -= np.einsum("kjia->ijka", tau188, optimize=True)

    tau188 = None

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum("baji->ijab", t2, optimize=True)

    tau4 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("kica,kjcb->ijab", tau4, u[o, o, v, v], optimize=True)

    tau4 = None

    tau6 += np.einsum("ijab->ijab", tau5, optimize=True)

    tau42 += np.einsum("ijab->ijab", tau5, optimize=True)

    tau54 += np.einsum("jiab->ijab", tau5, optimize=True)

    tau5 = None

    tau6 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau7 = np.zeros((N, N, M, M))

    tau7 += np.einsum("cbkj,kica->ijab", l2, tau6, optimize=True)

    tau6 = None

    tau18 += 2 * np.einsum("jiba->ijab", tau7, optimize=True)

    tau128 += 8 * np.einsum("jiab->ijab", tau7, optimize=True)

    tau173 += 8 * np.einsum("ijba->ijab", tau7, optimize=True)

    tau200 += 4 * np.einsum("jiba->ijab", tau7, optimize=True)

    tau7 = None

    tau8 = np.zeros((M, M))

    tau8 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau10 = np.zeros((M, M))

    tau10 += 2 * np.einsum("ab->ab", tau8, optimize=True)

    tau8 = None

    tau9 = np.zeros((M, M))

    tau9 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau10 += np.einsum("ab->ab", tau9, optimize=True)

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("ac,ijcb->ijab", tau10, u[o, o, v, v], optimize=True)

    tau18 -= np.einsum("jiba->ijab", tau11, optimize=True)

    tau128 -= 4 * np.einsum("jiab->ijab", tau11, optimize=True)

    tau173 -= 2 * np.einsum("jiab->ijab", tau11, optimize=True)

    tau200 += np.einsum("jiab->ijab", tau11, optimize=True)

    tau11 = None

    tau79 = np.zeros((N, M, M, M))

    tau79 += np.einsum("bi,ac->iabc", t1, tau10, optimize=True)

    tau86 = np.zeros((N, N, M, M))

    tau86 += np.einsum("ca,cbij->ijab", tau10, t2, optimize=True)

    tau88 = np.zeros((N, N, M, M))

    tau88 += 2 * np.einsum("jiab->ijab", tau86, optimize=True)

    tau199 = np.zeros((N, N, M, M))

    tau199 += np.einsum("jiba->ijab", tau86, optimize=True)

    tau86 = None

    tau130 = np.zeros((N, N))

    tau130 += np.einsum("ab,iajb->ij", tau10, u[o, v, o, v], optimize=True)

    tau136 = np.zeros((N, N))

    tau136 += 4 * np.einsum("ij->ij", tau130, optimize=True)

    tau130 = None

    tau139 = np.zeros((N, M))

    tau139 += np.einsum("bc,ibca->ia", tau10, u[o, v, v, v], optimize=True)

    tau142 = np.zeros((N, M))

    tau142 -= np.einsum("ia->ia", tau139, optimize=True)

    tau139 = None

    tau175 = np.zeros((M, M))

    tau175 += np.einsum("cd,cadb->ab", tau10, u[v, v, v, v], optimize=True)

    tau10 = None

    tau178 = np.zeros((M, M))

    tau178 -= 4 * np.einsum("ab->ab", tau175, optimize=True)

    tau175 = None

    tau52 = np.zeros((N, M, M, M))

    tau52 -= np.einsum("bi,ac->iabc", t1, tau9, optimize=True)

    tau67 = np.zeros((N, N, M, M))

    tau67 += np.einsum("ac,ijbc->ijab", tau9, tau3, optimize=True)

    tau74 = np.zeros((N, N, M, M))

    tau74 += np.einsum("ijab->ijab", tau67, optimize=True)

    tau67 = None

    tau154 = np.zeros((N, M, M, M))

    tau154 -= np.einsum("ad,ibcd->iabc", tau9, u[o, v, v, v], optimize=True)

    tau9 = None

    tau160 = np.zeros((N, M, M, M))

    tau160 -= np.einsum("ibac->iabc", tau154, optimize=True)

    tau154 = None

    tau12 = np.zeros((N, N))

    tau12 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau14 = np.zeros((N, N))

    tau14 += 2 * np.einsum("ij->ij", tau12, optimize=True)

    tau12 = None

    tau13 = np.zeros((N, N))

    tau13 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau14 += np.einsum("ij->ij", tau13, optimize=True)

    tau13 = None

    tau15 = np.zeros((N, N, M, M))

    tau15 += np.einsum("ik,kjab->ijab", tau14, u[o, o, v, v], optimize=True)

    tau18 += np.einsum("ijba->ijab", tau15, optimize=True)

    tau128 -= 2 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau173 -= 4 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau200 += np.einsum("ijba->ijab", tau15, optimize=True)

    tau15 = None

    tau47 = np.zeros((N, N, N, M))

    tau47 += np.einsum("aj,ik->ijka", t1, tau14, optimize=True)

    tau69 = np.zeros((N, N, N, M))

    tau69 += np.einsum("aj,ik->ijka", t1, tau14, optimize=True)

    tau87 = np.zeros((N, N, M, M))

    tau87 += np.einsum("ki,abkj->ijab", tau14, t2, optimize=True)

    tau88 -= 2 * np.einsum("jiba->ijab", tau87, optimize=True)

    tau87 = None

    tau110 = np.zeros((N, N, N, M))

    tau110 += np.einsum("ak,ij->ijka", t1, tau14, optimize=True)

    tau132 = np.zeros((N, M))

    tau132 += np.einsum("aj,ji->ia", t1, tau14, optimize=True)

    tau133 = np.zeros((N, M))

    tau133 += np.einsum("ia->ia", tau132, optimize=True)

    tau132 = None

    tau135 = np.zeros((N, N))

    tau135 += np.einsum("kl,likj->ij", tau14, u[o, o, o, o], optimize=True)

    tau136 -= 4 * np.einsum("ij->ij", tau135, optimize=True)

    tau135 = None

    tau141 = np.zeros((N, M))

    tau141 += np.einsum("jk,kija->ia", tau14, u[o, o, o, v], optimize=True)

    tau142 -= np.einsum("ia->ia", tau141, optimize=True)

    tau141 = None

    tau177 = np.zeros((M, M))

    tau177 += np.einsum("ij,jaib->ab", tau14, u[o, v, o, v], optimize=True)

    tau178 += 4 * np.einsum("ab->ab", tau177, optimize=True)

    tau177 = None

    tau16 = np.zeros((N, M))

    tau16 += np.einsum("bj,jiba->ia", t1, u[o, o, v, v], optimize=True)

    tau17 = np.zeros((N, M))

    tau17 += np.einsum("ia->ia", tau16, optimize=True)

    tau16 = None

    tau17 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau18 += 2 * np.einsum("ai,jb->ijab", l1, tau17, optimize=True)

    tau19 = np.zeros((N, N, M, M))

    tau19 += np.einsum("cbkj,kiac->ijab", t2, tau18, optimize=True)

    tau18 = None

    tau61 = np.zeros((N, N, M, M))

    tau61 -= 2 * np.einsum("ijab->ijab", tau19, optimize=True)

    tau19 = None

    tau128 += 8 * np.einsum("bi,ja->ijab", l1, tau17, optimize=True)

    tau173 += 8 * np.einsum("aj,ib->ijab", l1, tau17, optimize=True)

    tau200 += 4 * np.einsum("ai,jb->ijab", l1, tau17, optimize=True)

    tau17 = None

    tau202 = np.zeros((N, N, N, N))

    tau202 += np.einsum("balk,ijab->ijkl", t2, tau200, optimize=True)

    tau200 = None

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum("acik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau21 += np.einsum("ijab->ijab", tau20, optimize=True)

    tau108 = np.zeros((N, N, M, M))

    tau108 += np.einsum("ijab->ijab", tau20, optimize=True)

    tau151 += np.einsum("ijab->ijab", tau20, optimize=True)

    tau20 = None

    tau22 = np.zeros((M, M, M, M))

    tau22 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("ijcd,cadb->ijab", tau21, tau22, optimize=True)

    tau61 -= 2 * np.einsum("jiab->ijab", tau23, optimize=True)

    tau23 = None

    tau77 = np.zeros((N, N, M, M))

    tau77 += np.einsum("acbd,icjd->ijab", tau22, u[o, v, o, v], optimize=True)

    tau84 = np.zeros((N, N, M, M))

    tau84 -= np.einsum("ijba->ijab", tau77, optimize=True)

    tau77 = None

    tau24 = np.zeros((N, N, N, N))

    tau24 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau25 = np.zeros((N, N, M, M))

    tau25 -= np.einsum("ablk,lkji->ijab", t2, tau24, optimize=True)

    tau29 = np.zeros((N, N, M, M))

    tau29 -= np.einsum("ijba->ijab", tau25, optimize=True)

    tau88 -= np.einsum("ijba->ijab", tau25, optimize=True)

    tau90 = np.zeros((N, N, M, M))

    tau90 -= np.einsum("ijba->ijab", tau25, optimize=True)

    tau97 = np.zeros((N, N, M, M))

    tau97 -= np.einsum("ijba->ijab", tau25, optimize=True)

    tau25 = None

    tau55 = np.zeros((N, N, M, M))

    tau55 += np.einsum("likj,klab->ijab", tau24, tau54, optimize=True)

    tau54 = None

    tau61 -= 2 * np.einsum("ijba->ijab", tau55, optimize=True)

    tau55 = None

    tau66 = np.zeros((N, N, M, M))

    tau66 -= np.einsum("kijl,lakb->ijab", tau24, u[o, v, o, v], optimize=True)

    tau74 -= np.einsum("jiba->ijab", tau66, optimize=True)

    tau66 = None

    tau68 = np.zeros((N, N, N, M))

    tau68 += np.einsum("al,iljk->ijka", t1, tau24, optimize=True)

    tau69 -= np.einsum("ikja->ijka", tau68, optimize=True)

    tau110 += np.einsum("ikja->ijka", tau68, optimize=True)

    tau112 = np.zeros((N, N, N, M))

    tau112 -= np.einsum("ikja->ijka", tau68, optimize=True)

    tau68 = None

    tau106 = np.zeros((N, N, N, N))

    tau106 += np.einsum("mijn,nklm->ijkl", tau24, u[o, o, o, o], optimize=True)

    tau121 = np.zeros((N, N, N, N))

    tau121 += 2 * np.einsum("ijkl->ijkl", tau106, optimize=True)

    tau106 = None

    tau164 = np.zeros((N, N, M, M))

    tau164 -= np.einsum("jilk,lkab->ijab", tau24, u[o, o, v, v], optimize=True)

    tau169 = np.zeros((N, N, M, M))

    tau169 += np.einsum("ijba->ijab", tau164, optimize=True)

    tau173 += np.einsum("ijba->ijab", tau164, optimize=True)

    tau164 = None

    tau26 = np.zeros((N, N, M, M))

    tau26 -= np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau27 = np.zeros((N, N, M, M))

    tau27 -= np.einsum("acki,kjcb->ijab", t2, tau26, optimize=True)

    tau29 += 4 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau88 += 4 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau90 += 4 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau97 += 4 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau182 = np.zeros((N, N, N, M))

    tau182 += np.einsum("ijcb,kacb->ijka", tau27, u[o, v, v, v], optimize=True)

    tau186 = np.zeros((N, N, N, M))

    tau186 -= np.einsum("kjia->ijka", tau182, optimize=True)

    tau182 = None

    tau199 -= 2 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau202 += 2 * np.einsum("lkba,jiba->ijkl", tau27, u[o, o, v, v], optimize=True)

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum("ikca,jkcb->ijab", tau21, tau26, optimize=True)

    tau21 = None

    tau61 += 4 * np.einsum("jiab->ijab", tau31, optimize=True)

    tau31 = None

    tau40 = np.zeros((N, N, M, M))

    tau40 += np.einsum("ijab->ijab", tau26, optimize=True)

    tau43 = np.zeros((N, N, M, M))

    tau43 += np.einsum("kjbc,kiac->ijab", tau26, tau42, optimize=True)

    tau42 = None

    tau61 += 4 * np.einsum("ijba->ijab", tau43, optimize=True)

    tau43 = None

    tau78 = np.zeros((N, M, M, M))

    tau78 += np.einsum("bj,jiac->iabc", t1, tau26, optimize=True)

    tau79 += 2 * np.einsum("iacb->iabc", tau78, optimize=True)

    tau78 = None

    tau107 = np.zeros((N, N, N, N))

    tau107 -= np.einsum("ikab,jlab->ijkl", tau26, tau3, optimize=True)

    tau121 += 4 * np.einsum("ijlk->ijkl", tau107, optimize=True)

    tau107 = None

    tau28 = np.zeros((N, M))

    tau28 -= np.einsum("bj,abji->ia", l1, t2, optimize=True)

    tau29 += 4 * np.einsum("ai,jb->ijab", t1, tau28, optimize=True)

    tau29 += 4 * np.einsum("bj,ia->ijab", t1, tau28, optimize=True)

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("ikca,kjcb->ijab", tau29, u[o, o, v, v], optimize=True)

    tau29 = None

    tau61 -= np.einsum("jiba->ijab", tau30, optimize=True)

    tau30 = None

    tau65 = np.zeros((N, N, M, M))

    tau65 -= np.einsum("ic,jabc->ijab", tau28, u[o, v, v, v], optimize=True)

    tau74 += 2 * np.einsum("ijba->ijab", tau65, optimize=True)

    tau65 = None

    tau76 = np.zeros((N, N, M, M))

    tau76 -= np.einsum("ka,ikjb->ijab", tau28, u[o, o, o, v], optimize=True)

    tau84 += 2 * np.einsum("ijab->ijab", tau76, optimize=True)

    tau76 = None

    tau88 += 4 * np.einsum("ai,jb->ijab", t1, tau28, optimize=True)

    tau88 += 4 * np.einsum("bj,ia->ijab", t1, tau28, optimize=True)

    tau89 = np.zeros((N, N, M, M))

    tau89 += np.einsum("ikca,kbjc->ijab", tau88, u[o, v, o, v], optimize=True)

    tau88 = None

    tau101 = np.zeros((N, N, M, M))

    tau101 -= np.einsum("ijab->ijab", tau89, optimize=True)

    tau89 = None

    tau90 += 4 * np.einsum("ai,jb->ijab", t1, tau28, optimize=True)

    tau91 = np.zeros((N, N, M, M))

    tau91 += np.einsum("jkbc,kiac->ijab", tau3, tau90, optimize=True)

    tau3 = None

    tau101 -= np.einsum("jiab->ijab", tau91, optimize=True)

    tau91 = None

    tau96 = np.zeros((N, N, N, M))

    tau96 += np.einsum("ilba,ljkb->ijka", tau90, u[o, o, o, v], optimize=True)

    tau90 = None

    tau99 = np.zeros((N, N, N, M))

    tau99 += np.einsum("ijka->ijka", tau96, optimize=True)

    tau96 = None

    tau102 = np.zeros((N, N, M, M))

    tau102 -= np.einsum("ic,abjc->ijab", tau28, u[v, v, o, v], optimize=True)

    tau145 = np.zeros((N, N, M, M))

    tau145 += 8 * np.einsum("ijba->ijab", tau102, optimize=True)

    tau102 = None

    tau103 = np.zeros((N, M, M, M))

    tau103 += np.einsum("id,abdc->iabc", tau28, u[v, v, v, v], optimize=True)

    tau104 = np.zeros((N, N, M, M))

    tau104 += np.einsum("ci,jabc->ijab", t1, tau103, optimize=True)

    tau103 = None

    tau145 -= 8 * np.einsum("ijba->ijab", tau104, optimize=True)

    tau104 = None

    tau105 = np.zeros((N, N, N, N))

    tau105 += np.einsum("ia,kjla->ijkl", tau28, u[o, o, o, v], optimize=True)

    tau121 -= 2 * np.einsum("jikl->ijkl", tau105, optimize=True)

    tau194 = np.zeros((N, N, N, N))

    tau194 -= np.einsum("ikjl->ijkl", tau105, optimize=True)

    tau105 = None

    tau118 = np.zeros((N, N, N, M))

    tau118 += np.einsum("ib,kjab->ijka", tau28, u[o, o, v, v], optimize=True)

    tau119 = np.zeros((N, N, N, M))

    tau119 += np.einsum("ikja->ijka", tau118, optimize=True)

    tau193 = np.zeros((N, N, N, N))

    tau193 -= np.einsum("ai,jlka->ijkl", t1, tau118, optimize=True)

    tau118 = None

    tau194 += np.einsum("ilkj->ijkl", tau193, optimize=True)

    tau193 = None

    tau195 = np.zeros((N, N, N, M))

    tau195 += np.einsum("al,iljk->ijka", t1, tau194, optimize=True)

    tau194 = None

    tau196 = np.zeros((N, N, M, M))

    tau196 -= np.einsum("bk,ikja->ijab", t1, tau195, optimize=True)

    tau195 = None

    tau197 = np.zeros((N, N, M, M))

    tau197 -= np.einsum("ijba->ijab", tau196, optimize=True)

    tau196 = None

    tau133 -= 2 * np.einsum("ia->ia", tau28, optimize=True)

    tau148 = np.zeros((M, M, M, M))

    tau148 += np.einsum("ia,ibdc->abcd", tau28, u[o, v, v, v], optimize=True)

    tau162 = np.zeros((M, M, M, M))

    tau162 -= 2 * np.einsum("bcad->abcd", tau148, optimize=True)

    tau148 = None

    tau153 = np.zeros((N, M, M, M))

    tau153 += np.einsum("ja,ijcb->iabc", tau28, u[o, o, v, v], optimize=True)

    tau160 -= np.einsum("iacb->iabc", tau153, optimize=True)

    tau153 = None

    tau32 = np.zeros((M, M, M, M))

    tau32 += np.einsum("ai,ibcd->abcd", t1, u[o, v, v, v], optimize=True)

    tau34 = np.zeros((M, M, M, M))

    tau34 -= 2 * np.einsum("abdc->abcd", tau32, optimize=True)

    tau149 = np.zeros((M, M, M, M))

    tau149 += 2 * np.einsum("abdc->abcd", tau32, optimize=True)

    tau32 = None

    tau33 = np.zeros((M, M, M, M))

    tau33 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau34 -= np.einsum("badc->abcd", tau33, optimize=True)

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("ijcd,acdb->ijab", tau26, tau34, optimize=True)

    tau34 = None

    tau61 -= 2 * np.einsum("ijba->ijab", tau35, optimize=True)

    tau35 = None

    tau149 += np.einsum("badc->abcd", tau33, optimize=True)

    tau33 = None

    tau36 = np.zeros((N, N, N, N))

    tau36 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau39 = np.zeros((N, N, N, N))

    tau39 -= 2 * np.einsum("ikjl->ijkl", tau36, optimize=True)

    tau115 = np.zeros((N, N, N, N))

    tau115 -= 2 * np.einsum("ikjl->ijkl", tau36, optimize=True)

    tau126 = np.zeros((N, N, N, N))

    tau126 -= 4 * np.einsum("ljik->ijkl", tau36, optimize=True)

    tau192 = np.zeros((N, N, M, M))

    tau192 += np.einsum("klba,ilkj->ijab", tau27, tau36, optimize=True)

    tau36 = None

    tau197 -= np.einsum("ijba->ijab", tau192, optimize=True)

    tau192 = None

    r2 -= np.einsum("ijab->abij", tau197, optimize=True)

    r2 += np.einsum("jiab->abij", tau197, optimize=True)

    tau197 = None

    tau37 = np.zeros((N, N, M, M))

    tau37 -= np.einsum("baji->ijab", t2, optimize=True)

    tau37 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau38 = np.zeros((N, N, N, N))

    tau38 += np.einsum("ijab,klab->ijkl", tau37, u[o, o, v, v], optimize=True)

    tau37 = None

    tau39 += np.einsum("likj->ijkl", tau38, optimize=True)

    tau38 = None

    tau40 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau41 = np.zeros((N, N, M, M))

    tau41 += np.einsum("ikjl,lkab->ijab", tau39, tau40, optimize=True)

    tau39 = None

    tau61 -= 2 * np.einsum("jiab->ijab", tau41, optimize=True)

    tau41 = None

    tau73 = np.zeros((N, N, M, M))

    tau73 += np.einsum("ijcd,cadb->ijab", tau40, u[v, v, v, v], optimize=True)

    tau74 += 2 * np.einsum("jiba->ijab", tau73, optimize=True)

    tau73 = None

    tau82 = np.zeros((N, N, M, M))

    tau82 += np.einsum("ikca,kcjb->ijab", tau40, u[o, v, o, v], optimize=True)

    tau84 += 2 * np.einsum("ijab->ijab", tau82, optimize=True)

    tau82 = None

    tau83 = np.zeros((N, N, M, M))

    tau83 += np.einsum("klab,likj->ijab", tau40, u[o, o, o, o], optimize=True)

    tau84 += 2 * np.einsum("ijba->ijab", tau83, optimize=True)

    tau83 = None

    tau44 = np.zeros((N, N, N, M))

    tau44 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau47 -= np.einsum("ikja->ijka", tau44, optimize=True)

    tau57 = np.zeros((N, N, N, M))

    tau57 += np.einsum("ikja->ijka", tau44, optimize=True)

    tau69 -= 2 * np.einsum("ikja->ijka", tau44, optimize=True)

    tau110 += 2 * np.einsum("ikja->ijka", tau44, optimize=True)

    tau112 -= 2 * np.einsum("ikja->ijka", tau44, optimize=True)

    tau45 = np.zeros((N, N, N, M))

    tau45 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau46 = np.zeros((N, N, N, M))

    tau46 -= np.einsum("bakl,lijb->ijka", t2, tau45, optimize=True)

    tau47 -= 2 * np.einsum("ijka->ijka", tau46, optimize=True)

    tau81 = np.zeros((N, N, M, M))

    tau81 += np.einsum("ikla,lkjb->ijab", tau47, u[o, o, o, v], optimize=True)

    tau84 += np.einsum("ijab->ijab", tau81, optimize=True)

    tau81 = None

    tau57 += np.einsum("ijka->ijka", tau46, optimize=True)

    tau58 = np.zeros((N, N, N, M))

    tau58 += np.einsum("iljb,lkba->ijka", tau57, u[o, o, v, v], optimize=True)

    tau57 = None

    tau59 -= np.einsum("ijka->ijka", tau58, optimize=True)

    tau58 = None

    tau60 = np.zeros((N, N, M, M))

    tau60 += np.einsum("bk,ijka->ijab", t1, tau59, optimize=True)

    tau59 = None

    tau61 += 4 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau60 = None

    tau69 += 2 * np.einsum("ikja->ijka", tau46, optimize=True)

    tau70 = np.zeros((N, N, M, M))

    tau70 += np.einsum("ijkc,kacb->ijab", tau69, u[o, v, v, v], optimize=True)

    tau69 = None

    tau74 -= np.einsum("jiba->ijab", tau70, optimize=True)

    tau70 = None

    tau110 += 2 * np.einsum("ijka->ijka", tau46, optimize=True)

    tau111 = np.zeros((N, N, N, N))

    tau111 += np.einsum("imja,mkla->ijkl", tau110, u[o, o, o, v], optimize=True)

    tau110 = None

    tau121 -= 2 * np.einsum("ijkl->ijkl", tau111, optimize=True)

    tau111 = None

    tau112 -= 2 * np.einsum("ijka->ijka", tau46, optimize=True)

    tau46 = None

    tau51 = np.zeros((N, M, M, M))

    tau51 += np.einsum("bckj,kjia->iabc", t2, tau45, optimize=True)

    tau52 += np.einsum("iacb->iabc", tau51, optimize=True)

    tau79 -= np.einsum("iacb->iabc", tau51, optimize=True)

    tau51 = None

    tau131 = np.zeros((N, M))

    tau131 += np.einsum("bakj,kjib->ia", t2, tau45, optimize=True)

    tau45 = None

    tau133 += np.einsum("ia->ia", tau131, optimize=True)

    tau131 = None

    tau134 = np.zeros((N, N))

    tau134 += np.einsum("ka,kija->ij", tau133, u[o, o, o, v], optimize=True)

    tau136 += 4 * np.einsum("ij->ij", tau134, optimize=True)

    tau134 = None

    tau140 = np.zeros((N, M))

    tau140 += np.einsum("jb,jiba->ia", tau133, u[o, o, v, v], optimize=True)

    tau142 -= np.einsum("ia->ia", tau140, optimize=True)

    tau140 = None

    tau176 = np.zeros((M, M))

    tau176 += np.einsum("ic,iacb->ab", tau133, u[o, v, v, v], optimize=True)

    tau133 = None

    tau178 += 4 * np.einsum("ab->ab", tau176, optimize=True)

    tau176 = None

    tau48 = np.zeros((N, N, N, M))

    tau48 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau49 = np.zeros((N, N, M, M))

    tau49 += np.einsum("ikla,jlkb->ijab", tau47, tau48, optimize=True)

    tau47 = None

    tau61 += 2 * np.einsum("ijba->ijab", tau49, optimize=True)

    tau49 = None

    tau98 = np.zeros((N, N, N, M))

    tau98 += np.einsum("jlkb,ilba->ijka", tau48, tau97, optimize=True)

    tau97 = None

    tau99 += np.einsum("kija->ijka", tau98, optimize=True)

    tau98 = None

    tau100 = np.zeros((N, N, M, M))

    tau100 += np.einsum("bk,ikja->ijab", t1, tau99, optimize=True)

    tau99 = None

    tau101 -= np.einsum("ijba->ijab", tau100, optimize=True)

    tau100 = None

    tau113 = np.zeros((N, N, N, N))

    tau113 += np.einsum("imja,kmla->ijkl", tau112, tau48, optimize=True)

    tau112 = None

    tau121 += 2 * np.einsum("iljk->ijkl", tau113, optimize=True)

    tau113 = None

    tau155 = np.zeros((N, N, N, M))

    tau155 += np.einsum("kjia->ijka", tau48, optimize=True)

    tau183 = np.zeros((N, N, N, N))

    tau183 -= np.einsum("aj,ikla->ijkl", t1, tau48, optimize=True)

    tau184 = np.zeros((N, N, N, N))

    tau184 -= np.einsum("lkij->ijkl", tau183, optimize=True)

    tau183 = None

    tau50 = np.zeros((N, M, M, M))

    tau50 += np.einsum("aj,bcij->iabc", l1, t2, optimize=True)

    tau52 += 2 * np.einsum("iacb->iabc", tau50, optimize=True)

    tau53 = np.zeros((N, N, M, M))

    tau53 += np.einsum("ikjc,kabc->ijab", tau48, tau52, optimize=True)

    tau48 = None

    tau52 = None

    tau61 += 2 * np.einsum("jiab->ijab", tau53, optimize=True)

    tau53 = None

    tau62 = np.zeros((N, N, M, M))

    tau62 += np.einsum("cbkj,kica->ijab", t2, tau61, optimize=True)

    tau61 = None

    tau101 -= np.einsum("ijab->ijab", tau62, optimize=True)

    tau62 = None

    tau79 -= 2 * np.einsum("iacb->iabc", tau50, optimize=True)

    tau50 = None

    tau80 = np.zeros((N, N, M, M))

    tau80 += np.einsum("kabc,kijc->ijab", tau79, u[o, o, o, v], optimize=True)

    tau79 = None

    tau84 -= np.einsum("ijba->ijab", tau80, optimize=True)

    tau80 = None

    tau85 = np.zeros((N, N, M, M))

    tau85 += np.einsum("cbkj,kiac->ijab", t2, tau84, optimize=True)

    tau84 = None

    tau101 -= 2 * np.einsum("jiab->ijab", tau85, optimize=True)

    tau85 = None

    tau63 = np.zeros((N, N, N, M))

    tau63 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau64 = np.zeros((N, N, M, M))

    tau64 += np.einsum("ak,ikjb->ijab", l1, tau63, optimize=True)

    tau63 = None

    tau74 -= np.einsum("ijab->ijab", tau64, optimize=True)

    tau64 = None

    tau71 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau72 = np.zeros((N, N, M, M))

    tau72 += np.einsum("kiac,kjbc->ijab", tau40, tau71, optimize=True)

    tau71 = None

    tau74 -= 2 * np.einsum("ijab->ijab", tau72, optimize=True)

    tau72 = None

    tau75 = np.zeros((N, N, M, M))

    tau75 += np.einsum("cbkj,ikca->ijab", t2, tau74, optimize=True)

    tau74 = None

    tau101 -= 2 * np.einsum("ijba->ijab", tau75, optimize=True)

    tau75 = None

    tau92 = np.zeros((N, N, M, M))

    tau92 += np.einsum("ak,ibjk->ijab", l1, u[o, v, o, o], optimize=True)

    tau94 = np.zeros((N, N, M, M))

    tau94 += np.einsum("ijab->ijab", tau92, optimize=True)

    tau92 = None

    tau93 = np.zeros((N, N, M, M))

    tau93 += np.einsum("ci,acjb->ijab", l1, u[v, v, o, v], optimize=True)

    tau94 += np.einsum("ijba->ijab", tau93, optimize=True)

    tau93 = None

    tau95 = np.zeros((N, N, M, M))

    tau95 += np.einsum("cbkj,kica->ijab", t2, tau94, optimize=True)

    tau94 = None

    tau101 -= 4 * np.einsum("jiba->ijab", tau95, optimize=True)

    tau95 = None

    r2 -= np.einsum("ijab->abij", tau101, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau101, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau101, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau101, optimize=True) / 4

    tau101 = None

    tau108 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau109 = np.zeros((N, N, N, N))

    tau109 += np.einsum("ijab,klab->ijkl", tau108, tau26, optimize=True)

    tau108 = None

    tau121 += 4 * np.einsum("lkij->ijkl", tau109, optimize=True)

    tau109 = None

    tau114 = np.zeros((N, N, N, N))

    tau114 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau115 -= np.einsum("likj->ijkl", tau114, optimize=True)

    tau114 = None

    tau116 = np.zeros((N, N, N, N))

    tau116 += np.einsum("imjn,nkml->ijkl", tau115, tau24, optimize=True)

    tau24 = None

    tau115 = None

    tau121 -= np.einsum("jkil->ijkl", tau116, optimize=True)

    tau116 = None

    tau117 = np.zeros((N, N, N, M))

    tau117 += np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    tau119 += 2 * np.einsum("jkia->ijka", tau117, optimize=True)

    tau117 = None

    tau120 = np.zeros((N, N, N, N))

    tau120 += np.einsum("al,ijka->ijkl", t1, tau119, optimize=True)

    tau119 = None

    tau121 -= 2 * np.einsum("likj->ijkl", tau120, optimize=True)

    tau120 = None

    tau122 = np.zeros((N, N, M, M))

    tau122 += np.einsum("ablk,kilj->ijab", t2, tau121, optimize=True)

    tau121 = None

    tau145 += 2 * np.einsum("ijba->ijab", tau122, optimize=True)

    tau122 = None

    tau123 = np.zeros((N, N))

    tau123 += np.einsum("ak,iajk->ij", l1, u[o, v, o, o], optimize=True)

    tau136 += 8 * np.einsum("ij->ij", tau123, optimize=True)

    tau123 = None

    tau124 = np.zeros((N, N, M, M))

    tau124 += np.einsum("baji->ijab", t2, optimize=True)

    tau124 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau125 = np.zeros((N, N, N, N))

    tau125 += np.einsum("ijab,klab->ijkl", tau124, u[o, o, v, v], optimize=True)

    tau124 = None

    tau126 += np.einsum("lkji->ijkl", tau125, optimize=True)

    tau203 = np.zeros((N, N, N, N))

    tau203 += np.einsum("lkji->ijkl", tau125, optimize=True)

    tau125 = None

    tau126 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau127 = np.zeros((N, N, M, M))

    tau127 += np.einsum("abkl,ijkl->ijab", l2, tau126, optimize=True)

    tau126 = None

    tau128 -= np.einsum("jiba->ijab", tau127, optimize=True)

    tau129 = np.zeros((N, N))

    tau129 += np.einsum("bakj,kiab->ij", t2, tau128, optimize=True)

    tau128 = None

    tau136 += np.einsum("ij->ij", tau129, optimize=True)

    tau129 = None

    tau137 = np.zeros((N, N, M, M))

    tau137 += np.einsum("ki,abkj->ijab", tau136, t2, optimize=True)

    tau136 = None

    tau145 += np.einsum("jiba->ijab", tau137, optimize=True)

    tau137 = None

    tau169 -= np.einsum("jiba->ijab", tau127, optimize=True)

    tau127 = None

    tau138 = np.zeros((N, M))

    tau138 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau142 -= 2 * np.einsum("ia->ia", tau138, optimize=True)

    tau138 = None

    tau143 = np.zeros((N, N))

    tau143 += np.einsum("aj,ia->ij", t1, tau142, optimize=True)

    tau144 = np.zeros((N, N, M, M))

    tau144 += np.einsum("ki,abkj->ijab", tau143, t2, optimize=True)

    tau143 = None

    tau145 -= 4 * np.einsum("ijba->ijab", tau144, optimize=True)

    tau144 = None

    r2 += np.einsum("ijba->abij", tau145, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau145, optimize=True) / 8

    tau145 = None

    tau180 = np.zeros((M, M))

    tau180 += np.einsum("bi,ia->ab", t1, tau142, optimize=True)

    tau142 = None

    tau181 = np.zeros((N, N, M, M))

    tau181 += np.einsum("ca,cbij->ijab", tau180, t2, optimize=True)

    tau180 = None

    tau191 = np.zeros((N, N, M, M))

    tau191 -= 4 * np.einsum("jiab->ijab", tau181, optimize=True)

    tau181 = None

    tau146 = np.zeros((N, M, M, M))

    tau146 -= np.einsum("adij,jbdc->iabc", t2, u[o, v, v, v], optimize=True)

    tau147 = np.zeros((M, M, M, M))

    tau147 += np.einsum("ai,ibcd->abcd", l1, tau146, optimize=True)

    tau146 = None

    tau162 -= 4 * np.einsum("abcd->abcd", tau147, optimize=True)

    tau147 = None

    tau149 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau150 = np.zeros((M, M, M, M))

    tau150 += np.einsum("eafb,ecfd->abcd", tau149, tau22, optimize=True)

    tau149 = None

    tau162 += np.einsum("cdab->abcd", tau150, optimize=True)

    tau150 = None

    tau151 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau152 = np.zeros((M, M, M, M))

    tau152 += np.einsum("ijcd,ijab->abcd", tau151, tau40, optimize=True)

    tau40 = None

    tau151 = None

    tau162 += 4 * np.einsum("abcd->abcd", tau152, optimize=True)

    tau152 = None

    tau155 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau156 = np.zeros((N, M, M, M))

    tau156 += np.einsum("jika,kjbc->iabc", tau155, tau26, optimize=True)

    tau26 = None

    tau155 = None

    tau160 -= 2 * np.einsum("icba->iabc", tau156, optimize=True)

    tau156 = None

    tau157 = np.zeros((N, M, M, M))

    tau157 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau158 = np.zeros((N, M, M, M))

    tau158 += np.einsum("iacb->iabc", tau157, optimize=True)

    tau165 = np.zeros((N, M, M, M))

    tau165 += np.einsum("iacb->iabc", tau157, optimize=True)

    tau198 = np.zeros((M, M, M, M))

    tau198 += np.einsum("ai,ibcd->abcd", t1, tau157, optimize=True)

    tau157 = None

    tau158 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau159 = np.zeros((N, M, M, M))

    tau159 += np.einsum("idea,dbec->iabc", tau158, tau22, optimize=True)

    tau158 = None

    tau22 = None

    tau160 -= np.einsum("icba->iabc", tau159, optimize=True)

    tau159 = None

    tau161 = np.zeros((M, M, M, M))

    tau161 += np.einsum("di,iabc->abcd", t1, tau160, optimize=True)

    tau160 = None

    tau162 += 2 * np.einsum("cadb->abcd", tau161, optimize=True)

    tau161 = None

    tau163 = np.zeros((N, N, M, M))

    tau163 += np.einsum("dcij,cabd->ijab", t2, tau162, optimize=True)

    tau162 = None

    tau191 += 2 * np.einsum("jiab->ijab", tau163, optimize=True)

    tau163 = None

    tau165 -= 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau166 = np.zeros((M, M, M, M))

    tau166 += np.einsum("di,iabc->abcd", t1, tau165, optimize=True)

    tau165 = None

    tau167 = np.zeros((M, M, M, M))

    tau167 -= np.einsum("adcb->abcd", tau166, optimize=True)

    tau166 = None

    tau167 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau168 = np.zeros((N, N, M, M))

    tau168 += np.einsum("cdij,cdab->ijab", l2, tau167, optimize=True)

    tau167 = None

    tau169 -= 2 * np.einsum("jiba->ijab", tau168, optimize=True)

    tau170 = np.zeros((N, N, M, M))

    tau170 += np.einsum("cbkj,ikca->ijab", t2, tau169, optimize=True)

    tau169 = None

    tau171 = np.zeros((N, N, M, M))

    tau171 += np.einsum("cbkj,kica->ijab", t2, tau170, optimize=True)

    tau170 = None

    tau191 -= 2 * np.einsum("ijab->ijab", tau171, optimize=True)

    tau171 = None

    tau173 -= 2 * np.einsum("jiba->ijab", tau168, optimize=True)

    tau168 = None

    tau174 = np.zeros((M, M))

    tau174 += np.einsum("cbji,ijca->ab", t2, tau173, optimize=True)

    tau173 = None

    tau178 += np.einsum("ba->ab", tau174, optimize=True)

    tau174 = None

    tau172 = np.zeros((M, M))

    tau172 += np.einsum("ci,acib->ab", l1, u[v, v, o, v], optimize=True)

    tau178 += 8 * np.einsum("ab->ab", tau172, optimize=True)

    tau172 = None

    tau179 = np.zeros((N, N, M, M))

    tau179 += np.einsum("ac,cbij->ijab", tau178, t2, optimize=True)

    tau178 = None

    tau191 += np.einsum("jiba->ijab", tau179, optimize=True)

    tau179 = None

    tau184 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau185 = np.zeros((N, N, N, M))

    tau185 += np.einsum("la,lijk->ijka", tau28, tau184, optimize=True)

    tau186 -= np.einsum("ikja->ijka", tau185, optimize=True)

    tau185 = None

    tau187 = np.zeros((N, N, M, M))

    tau187 += np.einsum("bk,kija->ijab", t1, tau186, optimize=True)

    tau186 = None

    tau191 += 8 * np.einsum("ijba->ijab", tau187, optimize=True)

    tau187 = None

    tau202 += 2 * np.einsum("im,mjlk->ijkl", tau14, tau184, optimize=True)

    tau184 = None

    tau14 = None

    tau189 -= np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau190 = np.zeros((N, N, M, M))

    tau190 += np.einsum("kb,ijka->ijab", tau28, tau189, optimize=True)

    tau28 = None

    tau189 = None

    tau191 += 8 * np.einsum("ijba->ijab", tau190, optimize=True)

    tau190 = None

    r2 += np.einsum("jiab->abij", tau191, optimize=True) / 8

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

    r2 += np.einsum("klji,klba->abij", tau203, tau27, optimize=True) / 2

    tau27 = None

    tau203 = None

    tau204 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    r2 += np.einsum("kbac,kjic->abij", tau204, tau44, optimize=True)

    tau204 = None

    tau44 = None

    return r2
