import numpy as np
from clusterfock.cc.rhs.l_inter_CCSD import lambda_amplitudes_intermediates_ccsd


def lambda_amplitudes_intermediates_qccsd(t1, t2, l1, l2, u, f, v, o):
    r1, r2 = lambda_amplitudes_intermediates_ccsd(t1, t2, l1, l2, u, f, v, o)

    lambda_amplitudes_intermediates_qccsd_l1_addition(r1, t1, t2, l1, l2, u, f, v, o)

    lambda_amplitudes_intermediates_qccsd_l2_addition_L1L1(r2, t1, t2, l1, l2, u, f, v, o)
    lambda_amplitudes_intermediates_qccsd_l2_addition_L1L2(r2, t1, t2, l1, l2, u, f, v, o)
    lambda_amplitudes_intermediates_qccsd_l2_addition_L1L2(r2, t1, t2, l1, l2, u, f, v, o)

    return r1, r2


def lambda_amplitudes_intermediates_qccsd_l1_addition(r1, t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((M, M))

    tau0 -= np.einsum("ci,caib->ab", l1, u[v, v, o, v], optimize=True)

    r1 -= np.einsum("bi,ba->ai", l1, tau0, optimize=True)

    tau0 = None

    tau1 = np.zeros((N, M))

    tau1 += np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau2 = np.zeros((N, N))

    tau2 += np.einsum("ai,ja->ij", l1, tau1, optimize=True)

    tau66 = np.zeros((N, N, N, M))

    tau66 += 8 * np.einsum("aj,ik->ijka", t1, tau2, optimize=True)

    tau78 = np.zeros((N, M))

    tau78 += 4 * np.einsum("aj,ji->ia", t1, tau2, optimize=True)

    r1 += np.einsum("jk,ikja->ai", tau2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("ja,ij->ai", f[o, v], tau2, optimize=True)

    tau2 = None

    tau3 = np.zeros((M, M))

    tau3 += np.einsum("ai,ib->ab", l1, tau1, optimize=True)

    r1 += np.einsum("bc,ibac->ai", tau3, u[o, v, v, v], optimize=True)

    tau3 = None

    tau15 = np.zeros((N, N, M, M))

    tau15 += 4 * np.einsum("ai,jb->ijab", t1, tau1, optimize=True)

    tau21 = np.zeros((N, M))

    tau21 -= 2 * np.einsum("ia->ia", tau1, optimize=True)

    tau25 = np.zeros((N, N, N, N))

    tau25 += np.einsum("ia,kjla->ijkl", tau1, u[o, o, o, v], optimize=True)

    tau49 = np.zeros((N, N, N, M))

    tau49 += 4 * np.einsum("al,iljk->ijka", t1, tau25, optimize=True)

    tau25 = None

    tau26 = np.zeros((N, N, M, M))

    tau26 -= np.einsum("ic,jabc->ijab", tau1, u[o, v, v, v], optimize=True)

    tau49 += 4 * np.einsum("bk,ijab->ijka", t1, tau26, optimize=True)

    tau26 = None

    tau33 = np.zeros((N, M, M, M))

    tau33 += np.einsum("jb,ijca->iabc", tau1, u[o, o, v, v], optimize=True)

    tau40 = np.zeros((N, N, N, M))

    tau40 -= 2 * np.einsum("ib,kjab->ijka", tau1, u[o, o, v, v], optimize=True)

    tau41 = np.zeros((N, N, M, M))

    tau41 += 2 * np.einsum("ai,jb->ijab", t1, tau1, optimize=True)

    tau41 += 4 * np.einsum("bj,ia->ijab", t1, tau1, optimize=True)

    tau42 = np.zeros((N, N, M, M))

    tau42 += 4 * np.einsum("ai,jb->ijab", t1, tau1, optimize=True)

    tau49 -= 2 * np.einsum("la,jlik->ijka", tau1, u[o, o, o, o], optimize=True)

    tau49 += 4 * np.einsum("kb,jaib->ijka", tau1, u[o, v, o, v], optimize=True)

    tau51 = np.zeros((N, N, M, M))

    tau51 += 4 * np.einsum("bi,ja->ijab", t1, tau1, optimize=True)

    tau51 += 2 * np.einsum("aj,ib->ijab", t1, tau1, optimize=True)

    tau53 = np.zeros((N, N, N, M))

    tau53 -= np.einsum("kb,baji->ijka", tau1, l2, optimize=True)

    tau55 = np.zeros((N, N, N, M))

    tau55 -= 2 * np.einsum("ijka->ijka", tau53, optimize=True)

    tau68 = np.zeros((N, N, N, M))

    tau68 += np.einsum("ijka->ijka", tau53, optimize=True)

    tau71 = np.zeros((N, N, N, M))

    tau71 -= 2 * np.einsum("ijka->ijka", tau53, optimize=True)

    tau77 = np.zeros((N, N, N, M))

    tau77 -= 2 * np.einsum("ijka->ijka", tau53, optimize=True)

    tau53 = None

    tau67 = np.zeros((N, M, M, M))

    tau67 -= 2 * np.einsum("jc,baji->iabc", tau1, l2, optimize=True)

    tau74 = np.zeros((N, N, M, M))

    tau74 += 2 * np.einsum("ai,jb->ijab", t1, tau1, optimize=True)

    tau4 = np.zeros((N, N, N, M))

    tau4 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau17 = np.zeros((N, N, N, M))

    tau17 += np.einsum("balk,lijb->ijka", t2, tau4, optimize=True)

    tau18 = np.zeros((N, N, N, M))

    tau18 += 2 * np.einsum("ikja->ijka", tau17, optimize=True)

    tau34 = np.zeros((N, N, N, M))

    tau34 += 2 * np.einsum("ikja->ijka", tau17, optimize=True)

    tau56 = np.zeros((N, N, N, M))

    tau56 += 2 * np.einsum("ijka->ijka", tau17, optimize=True)

    tau60 = np.zeros((N, N, N, M))

    tau60 += 2 * np.einsum("ijka->ijka", tau17, optimize=True)

    tau19 = np.zeros((N, M))

    tau19 += np.einsum("bakj,kjib->ia", t2, tau4, optimize=True)

    tau21 += np.einsum("ia->ia", tau19, optimize=True)

    tau65 = np.zeros((N, M))

    tau65 += np.einsum("ia->ia", tau19, optimize=True)

    tau19 = None

    tau27 = np.zeros((N, M, M, M))

    tau27 += np.einsum("bckj,kjia->iabc", t2, tau4, optimize=True)

    tau29 = np.zeros((N, M, M, M))

    tau29 -= np.einsum("iacb->iabc", tau27, optimize=True)

    tau57 = np.zeros((N, M, M, M))

    tau57 += np.einsum("iacb->iabc", tau27, optimize=True)

    tau27 = None

    tau5 = np.zeros((N, N, N, M))

    tau5 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau18 -= 2 * np.einsum("ikja->ijka", tau5, optimize=True)

    tau24 = np.zeros((N, N, M, M))

    tau24 += 2 * np.einsum("lkja,ilkb->ijab", tau4, tau5, optimize=True)

    tau34 -= 2 * np.einsum("ikja->ijka", tau5, optimize=True)

    tau54 = np.zeros((N, N, N, M))

    tau54 -= np.einsum("ablj,ilkb->ijka", l2, tau5, optimize=True)

    tau55 -= 2 * np.einsum("ijka->ijka", tau54, optimize=True)

    tau55 += 2 * np.einsum("jika->ijka", tau54, optimize=True)

    tau68 += np.einsum("ijka->ijka", tau54, optimize=True)

    tau68 -= np.einsum("jika->ijka", tau54, optimize=True)

    tau69 = np.zeros((N, N, M, M))

    tau69 += np.einsum("bk,ikja->ijab", t1, tau68, optimize=True)

    tau68 = None

    r1 += np.einsum("ijbc,jbca->ai", tau69, u[o, v, v, v], optimize=True)

    tau69 = None

    tau70 = np.zeros((N, N, N, M))

    tau70 += 2 * np.einsum("jika->ijka", tau54, optimize=True)

    tau71 += 2 * np.einsum("jika->ijka", tau54, optimize=True)

    tau77 += 4 * np.einsum("jika->ijka", tau54, optimize=True)

    tau54 = None

    tau56 += np.einsum("ikja->ijka", tau5, optimize=True)

    tau61 = np.zeros((N, N, N, N))

    tau61 -= np.einsum("aj,ikla->ijkl", l1, tau5, optimize=True)

    tau64 = np.zeros((N, N, N, N))

    tau64 -= 2 * np.einsum("ijlk->ijkl", tau61, optimize=True)

    tau73 = np.zeros((N, N, N, N))

    tau73 += 2 * np.einsum("ijlk->ijkl", tau61, optimize=True)

    tau61 = None

    tau62 = np.zeros((N, N, N, N))

    tau62 += np.einsum("mjka,imla->ijkl", tau4, tau5, optimize=True)

    tau64 -= 4 * np.einsum("ijkl->ijkl", tau62, optimize=True)

    tau73 += 4 * np.einsum("jilk->ijkl", tau62, optimize=True)

    tau75 = np.zeros((N, N, N, N))

    tau75 += 2 * np.einsum("ijkl->ijkl", tau62, optimize=True)

    tau62 = None

    tau67 += np.einsum("bakj,ikjc->iabc", l2, tau5, optimize=True)

    r1 += np.einsum("ibcd,bcda->ai", tau67, u[v, v, v, v], optimize=True) / 4

    tau67 = None

    tau6 = np.zeros((N, N, M, M))

    tau6 -= np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("cbkj,kica->ijab", t2, tau6, optimize=True)

    tau15 += 4 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau41 += 4 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau42 += 4 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau51 += 4 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau63 = np.zeros((N, N, M, M))

    tau63 += 2 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau72 = np.zeros((N, N, M, M))

    tau72 += 2 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau24 -= 4 * np.einsum("ikcb,kjac->ijab", tau6, tau6, optimize=True)

    tau28 = np.zeros((N, M, M, M))

    tau28 += np.einsum("bj,jiac->iabc", t1, tau6, optimize=True)

    tau29 += 2 * np.einsum("iacb->iabc", tau28, optimize=True)

    tau57 += 2 * np.einsum("iabc->iabc", tau28, optimize=True)

    tau28 = None

    tau33 += 2 * np.einsum("jkab,ikjc->iabc", tau6, u[o, o, o, v], optimize=True)

    tau37 = np.zeros((N, N, M, M))

    tau37 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau66 += 4 * np.einsum("lkjb,ilba->ijka", tau56, tau6, optimize=True)

    tau56 = None

    tau66 -= 4 * np.einsum("kbac,ijcb->ijka", tau57, tau6, optimize=True)

    tau57 = None

    tau73 += 4 * np.einsum("ikab,jlba->ijkl", tau6, tau6, optimize=True)

    tau76 = np.zeros((N, M, M, M))

    tau76 -= 2 * np.einsum("aj,ijbc->iabc", l1, tau6, optimize=True)

    tau7 = np.zeros((M, M, M, M))

    tau7 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau24 -= 2 * np.einsum("ijdc,cabd->ijab", tau6, tau7, optimize=True)

    tau33 += np.einsum("daeb,idce->iabc", tau7, u[o, v, v, v], optimize=True)

    tau7 = None

    tau8 = np.zeros((N, N, N, N))

    tau8 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 -= np.einsum("ablk,lkji->ijab", t2, tau8, optimize=True)

    tau15 -= np.einsum("ijba->ijab", tau9, optimize=True)

    tau41 -= np.einsum("ijba->ijab", tau9, optimize=True)

    tau42 -= np.einsum("ijba->ijab", tau9, optimize=True)

    tau49 += np.einsum("klba,ljib->ijka", tau42, u[o, o, o, v], optimize=True)

    tau42 = None

    tau51 += np.einsum("ijba->ijab", tau9, optimize=True)

    tau9 = None

    tau16 = np.zeros((N, N, N, M))

    tau16 -= np.einsum("al,ilkj->ijka", t1, tau8, optimize=True)

    tau18 -= np.einsum("ikja->ijka", tau16, optimize=True)

    tau24 -= 2 * np.einsum("ak,ikjb->ijab", l1, tau18, optimize=True)

    tau18 = None

    tau34 -= np.einsum("ikja->ijka", tau16, optimize=True)

    tau60 += np.einsum("ikja->ijka", tau16, optimize=True)

    tau16 = None

    tau24 -= 2 * np.einsum("klab,lijk->ijab", tau6, tau8, optimize=True)

    tau6 = None

    tau40 += np.einsum("lkmi,jmla->ijka", tau8, u[o, o, o, v], optimize=True)

    tau52 = np.zeros((N, N, N, M))

    tau52 -= np.einsum("al,jikl->ijka", l1, tau8, optimize=True)

    tau55 += np.einsum("ijka->ijka", tau52, optimize=True)

    tau70 += np.einsum("ijka->ijka", tau52, optimize=True)

    tau77 += np.einsum("ijka->ijka", tau52, optimize=True)

    tau52 = None

    tau66 -= 2 * np.einsum("la,ilkj->ijka", tau1, tau8, optimize=True)

    tau73 -= np.einsum("jnlm,mikn->ijkl", tau8, tau8, optimize=True)

    tau11 = np.zeros((M, M))

    tau11 -= np.einsum("acji,cbji->ab", l2, t2, optimize=True)

    tau15 += 2 * np.einsum("cb,acji->ijab", tau11, t2, optimize=True)

    tau23 = np.zeros((M, M))

    tau23 += np.einsum("ab->ab", tau11, optimize=True)

    tau33 += np.einsum("ad,ibcd->iabc", tau11, u[o, v, v, v], optimize=True)

    tau66 += 2 * np.einsum("ba,ikjb->ijka", tau11, tau5, optimize=True)

    tau12 = np.zeros((N, N))

    tau12 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau14 = np.zeros((N, N))

    tau14 += 2 * np.einsum("ij->ij", tau12, optimize=True)

    tau60 += 2 * np.einsum("ak,ij->ijka", t1, tau12, optimize=True)

    tau66 -= 2 * np.einsum("lkma,milj->ijka", tau60, tau8, optimize=True)

    tau8 = None

    tau60 = None

    tau66 += 8 * np.einsum("lk,ijla->ijka", tau12, tau17, optimize=True)

    tau17 = None

    tau73 += 4 * np.einsum("il,jk->ijkl", tau12, tau12, optimize=True)

    tau74 -= np.einsum("ki,bajk->ijab", tau12, t2, optimize=True)

    tau75 -= np.einsum("baji,klab->ijkl", l2, tau74, optimize=True)

    tau74 = None

    tau13 = np.zeros((N, N))

    tau13 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau14 += np.einsum("ij->ij", tau13, optimize=True)

    tau15 += 2 * np.einsum("ki,bakj->ijab", tau14, t2, optimize=True)

    tau24 += np.einsum("caki,jkcb->ijab", l2, tau15, optimize=True)

    tau15 = None

    tau20 = np.zeros((N, M))

    tau20 += np.einsum("aj,ji->ia", t1, tau14, optimize=True)

    tau21 += np.einsum("ia->ia", tau20, optimize=True)

    tau24 += 2 * np.einsum("ai,jb->ijab", l1, tau21, optimize=True)

    tau46 = np.zeros((N, M))

    tau46 += np.einsum("jb,jiba->ia", tau21, u[o, o, v, v], optimize=True)

    tau48 = np.zeros((N, M))

    tau48 -= np.einsum("ia->ia", tau46, optimize=True)

    tau46 = None

    tau66 -= 4 * np.einsum("ik,ja->ijka", tau12, tau21, optimize=True)

    tau84 = np.zeros((N, N))

    tau84 += 4 * np.einsum("ka,kija->ij", tau21, u[o, o, o, v], optimize=True)

    tau21 = None

    tau65 += np.einsum("ia->ia", tau20, optimize=True)

    tau20 = None

    tau34 += np.einsum("aj,ik->ijka", t1, tau14, optimize=True)

    tau40 -= np.einsum("kilb,ljba->ijka", tau34, u[o, o, v, v], optimize=True)

    tau34 = None

    tau40 -= np.einsum("kl,ljia->ijka", tau14, u[o, o, o, v], optimize=True)

    tau47 = np.zeros((N, M))

    tau47 += np.einsum("jk,kija->ia", tau14, u[o, o, o, v], optimize=True)

    tau48 -= np.einsum("ia->ia", tau47, optimize=True)

    tau47 = None

    tau55 += np.einsum("ai,jk->ijka", l1, tau14, optimize=True)

    tau66 -= 4 * np.einsum("balj,ilkb->ijka", t2, tau55, optimize=True)

    tau55 = None

    tau70 += np.einsum("ai,jk->ijka", l1, tau14, optimize=True)

    r1 -= np.einsum("ijkb,kbja->ai", tau70, u[o, v, o, v], optimize=True) / 2

    tau70 = None

    tau71 += np.einsum("ai,jk->ijka", l1, tau14, optimize=True)

    r1 += np.einsum("jikb,kbja->ai", tau71, u[o, v, o, v], optimize=True) / 2

    tau71 = None

    tau77 += 2 * np.einsum("ai,jk->ijka", l1, tau14, optimize=True)

    tau78 += np.einsum("bajk,jkib->ia", t2, tau77, optimize=True)

    tau77 = None

    r1 -= np.einsum("jb,jiba->ai", tau78, u[o, o, v, v], optimize=True) / 4

    tau78 = None

    tau83 = np.zeros((N, N, M, M))

    tau83 -= 2 * np.einsum("ik,kjba->ijab", tau14, u[o, o, v, v], optimize=True)

    tau84 -= 4 * np.einsum("lk,kilj->ij", tau14, u[o, o, o, o], optimize=True)

    tau50 = np.zeros((N, N, M, M))

    tau50 += np.einsum("kj,abik->ijab", tau13, t2, optimize=True)

    tau51 -= 2 * np.einsum("ijba->ijab", tau50, optimize=True)

    tau63 -= np.einsum("ijba->ijab", tau50, optimize=True)

    tau64 -= np.einsum("abji,lkab->ijkl", l2, tau63, optimize=True)

    tau63 = None

    tau66 -= 2 * np.einsum("al,ilkj->ijka", t1, tau64, optimize=True)

    tau64 = None

    tau72 += np.einsum("ijba->ijab", tau50, optimize=True)

    tau50 = None

    tau73 += np.einsum("abji,klab->ijkl", l2, tau72, optimize=True)

    tau72 = None

    tau66 += 4 * np.einsum("ja,ik->ijka", tau1, tau13, optimize=True)

    tau1 = None

    tau66 += 2 * np.einsum("ij,ka->ijka", tau13, tau65, optimize=True)

    tau65 = None

    tau73 -= np.einsum("ik,jl->ijkl", tau13, tau14, optimize=True)

    r1 += np.einsum("ijkl,klja->ai", tau73, u[o, o, o, v], optimize=True) / 4

    tau73 = None

    tau75 -= np.einsum("ik,jl->ijkl", tau12, tau13, optimize=True)

    tau12 = None

    tau13 = None

    r1 -= np.einsum("ijkl,lkja->ai", tau75, u[o, o, o, v], optimize=True) / 2

    tau75 = None

    tau22 = np.zeros((M, M))

    tau22 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau23 += 2 * np.einsum("ab->ab", tau22, optimize=True)

    tau22 = None

    tau24 -= np.einsum("ij,ab->ijab", tau14, tau23, optimize=True)

    tau14 = None

    r1 -= np.einsum("ijbc,jbca->ai", tau24, u[o, v, v, v], optimize=True) / 4

    tau24 = None

    tau29 += np.einsum("bi,ac->iabc", t1, tau23, optimize=True)

    tau45 = np.zeros((N, M))

    tau45 += np.einsum("bc,ibca->ia", tau23, u[o, v, v, v], optimize=True)

    tau48 -= np.einsum("ia->ia", tau45, optimize=True)

    tau45 = None

    tau51 += 2 * np.einsum("cb,caji->ijab", tau23, t2, optimize=True)

    tau66 -= 2 * np.einsum("likb,ljba->ijka", tau4, tau51, optimize=True)

    tau51 = None

    tau4 = None

    tau76 += np.einsum("ai,bc->iabc", l1, tau23, optimize=True)

    r1 += np.einsum("ibcd,cbda->ai", tau76, u[v, v, v, v], optimize=True) / 2

    tau76 = None

    tau83 -= 4 * np.einsum("ac,jicb->ijab", tau23, u[o, o, v, v], optimize=True)

    tau84 += 4 * np.einsum("ab,iajb->ij", tau23, u[o, v, o, v], optimize=True)

    tau23 = None

    tau29 -= 2 * np.einsum("aj,cbij->iabc", l1, t2, optimize=True)

    tau33 -= np.einsum("jabd,jidc->iabc", tau29, u[o, o, v, v], optimize=True)

    tau29 = None

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau32 = np.zeros((N, N, M, M))

    tau32 += np.einsum("jiab->ijab", tau30, optimize=True)

    tau80 = np.zeros((N, N, M, M))

    tau80 += np.einsum("jiab->ijab", tau30, optimize=True)

    tau30 = None

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau32 += np.einsum("ijab->ijab", tau31, optimize=True)

    tau80 += np.einsum("ijab->ijab", tau31, optimize=True)

    tau31 = None

    tau32 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau33 += 2 * np.einsum("aj,jibc->iabc", l1, tau32, optimize=True)

    tau32 = None

    tau49 += np.einsum("cbki,jbac->ijka", t2, tau33, optimize=True)

    tau33 = None

    tau35 = np.zeros((N, N, N, M))

    tau35 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau36 = np.zeros((N, N, N, M))

    tau36 -= np.einsum("ikja->ijka", tau35, optimize=True)

    tau49 += np.einsum("kljb,ilba->ijka", tau35, tau41, optimize=True)

    tau41 = None

    tau35 = None

    tau36 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau40 += np.einsum("ab,ikjb->ijka", tau11, tau36, optimize=True)

    tau11 = None

    tau37 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau40 -= 2 * np.einsum("lkjb,liab->ijka", tau36, tau37, optimize=True)

    tau36 = None

    tau40 -= 2 * np.einsum("kibc,jbca->ijka", tau37, u[o, v, v, v], optimize=True)

    tau37 = None

    tau38 = np.zeros((N, N, N, N))

    tau38 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau39 = np.zeros((N, N, N, N))

    tau39 += 2 * np.einsum("kjil->ijkl", tau38, optimize=True)

    tau81 = np.zeros((N, N, N, N))

    tau81 -= 4 * np.einsum("ljik->ijkl", tau38, optimize=True)

    tau38 = None

    tau39 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau39 += np.einsum("balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau40 += np.einsum("al,kjli->ijka", l1, tau39, optimize=True)

    tau39 = None

    tau40 -= 2 * np.einsum("bk,jbia->ijka", l1, u[o, v, o, v], optimize=True)

    tau49 += 2 * np.einsum("bali,kjlb->ijka", t2, tau40, optimize=True)

    tau40 = None

    tau43 = np.zeros((N, M, M, M))

    tau43 += np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau43 -= np.einsum("aj,ijcb->iabc", t1, u[o, o, v, v], optimize=True)

    tau49 += 2 * np.einsum("ikcb,jabc->ijka", tau10, tau43, optimize=True)

    tau10 = None

    tau43 = None

    tau44 = np.zeros((N, M))

    tau44 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau48 -= 2 * np.einsum("ia->ia", tau44, optimize=True)

    tau44 = None

    tau49 -= np.einsum("jb,baki->ijka", tau48, t2, optimize=True)

    r1 += np.einsum("bajk,jikb->ai", l2, tau49, optimize=True) / 4

    tau49 = None

    tau84 += 4 * np.einsum("aj,ia->ij", t1, tau48, optimize=True)

    tau48 = None

    tau58 = np.zeros((N, N, M, M))

    tau58 -= np.einsum("baji->ijab", t2, optimize=True)

    tau58 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau59 = np.zeros((N, N, N, N))

    tau59 += np.einsum("abji,lkab->ijkl", l2, tau58, optimize=True)

    tau66 += np.einsum("ilma,lmkj->ijka", tau5, tau59, optimize=True)

    tau59 = None

    tau5 = None

    r1 -= np.einsum("ijkb,jkba->ai", tau66, u[o, o, v, v], optimize=True) / 8

    tau66 = None

    tau81 -= np.einsum("lkab,jiab->ijkl", tau58, u[o, o, v, v], optimize=True)

    tau58 = None

    tau79 = np.zeros((N, N, M, M))

    tau79 += np.einsum("baji->ijab", t2, optimize=True)

    tau79 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau80 += np.einsum("kica,kjcb->ijab", tau79, u[o, o, v, v], optimize=True)

    tau79 = None

    tau80 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau83 += 8 * np.einsum("cbki,kjca->ijab", l2, tau80, optimize=True)

    tau80 = None

    tau81 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau83 -= np.einsum("bakl,jikl->ijab", l2, tau81, optimize=True)

    tau81 = None

    tau82 = np.zeros((N, M))

    tau82 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau82 -= np.einsum("bj,ijba->ia", t1, u[o, o, v, v], optimize=True)

    tau83 += 8 * np.einsum("bi,ja->ijab", l1, tau82, optimize=True)

    tau82 = None

    tau83 -= 8 * np.einsum("ak,jikb->ijab", l1, u[o, o, o, v], optimize=True)

    tau83 -= 4 * np.einsum("ci,jcba->ijab", l1, u[o, v, v, v], optimize=True)

    tau84 += np.einsum("bakj,kiab->ij", t2, tau83, optimize=True)

    tau83 = None

    tau84 -= 8 * np.einsum("ak,iakj->ij", l1, u[o, v, o, o], optimize=True)

    r1 -= np.einsum("aj,ij->ai", l1, tau84, optimize=True) / 8

    tau84 = None


def lambda_amplitudes_intermediates_qccsd_l2_addition_L1L1(r2, t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((N, M, M, M))

    tau0 += np.einsum("di,adbc->iabc", l1, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cj,icab->abij", l1, tau0, optimize=True)

    tau0 = None

    tau1 = np.zeros((N, N))

    tau1 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau2 = np.zeros((N, N, N, M))

    tau2 += np.einsum("il,jlka->ijka", tau1, u[o, o, o, v], optimize=True)

    tau9 = np.zeros((N, N, N, M))

    tau9 += np.einsum("ijka->ijka", tau2, optimize=True)

    tau2 = None

    tau4 = np.zeros((N, N, N, M))

    tau4 -= np.einsum("ak,ij->ijka", t1, tau1, optimize=True)

    tau17 = np.zeros((N, M))

    tau17 += np.einsum("aj,ji->ia", t1, tau1, optimize=True)

    tau18 = np.zeros((N, M))

    tau18 -= np.einsum("ia->ia", tau17, optimize=True)

    tau17 = None

    tau20 = np.zeros((N, N, N, M))

    tau20 += 2 * np.einsum("aj,ik->ijka", t1, tau1, optimize=True)

    tau40 = np.zeros((N, N, N, N))

    tau40 += 2 * np.einsum("ik,jl->ijkl", tau1, tau1, optimize=True)

    tau42 = np.zeros((N, M))

    tau42 += np.einsum("ja,ij->ia", f[o, v], tau1, optimize=True)

    tau48 = np.zeros((N, M))

    tau48 += np.einsum("ia->ia", tau42, optimize=True)

    tau42 = None

    tau43 = np.zeros((N, M))

    tau43 += np.einsum("jk,ikja->ia", tau1, u[o, o, o, v], optimize=True)

    tau48 -= np.einsum("ia->ia", tau43, optimize=True)

    tau43 = None

    tau3 = np.zeros((N, N, N, M))

    tau3 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau4 -= np.einsum("ikja->ijka", tau3, optimize=True)

    tau5 = np.zeros((N, N, N, M))

    tau5 += np.einsum("iljb,lkba->ijka", tau4, u[o, o, v, v], optimize=True)

    tau4 = None

    tau9 += np.einsum("ikja->ijka", tau5, optimize=True)

    tau5 = None

    tau20 -= np.einsum("ikja->ijka", tau3, optimize=True)

    tau21 = np.zeros((N, M))

    tau21 += np.einsum("ijkb,jkba->ia", tau20, u[o, o, v, v], optimize=True)

    tau20 = None

    tau25 = np.zeros((N, M))

    tau25 += np.einsum("ia->ia", tau21, optimize=True)

    tau21 = None

    tau40 -= np.einsum("aj,ikla->ijkl", l1, tau3, optimize=True)

    tau3 = None

    r2 -= np.einsum("ijkl,klba->abij", tau40, u[o, o, v, v], optimize=True) / 2

    tau40 = None

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau7 = np.zeros((N, N, M, M))

    tau7 -= np.einsum("jiab->ijab", tau6, optimize=True)

    tau6 = None

    tau7 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau8 = np.zeros((N, N, N, M))

    tau8 += np.einsum("bk,ijba->ijka", l1, tau7, optimize=True)

    tau7 = None

    tau9 -= np.einsum("jkia->ijka", tau8, optimize=True)

    tau8 = None

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("bk,ijka->ijab", l1, tau9, optimize=True)

    tau9 = None

    tau26 = np.zeros((N, N, M, M))

    tau26 += 2 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau10 = None

    tau11 = np.zeros((N, M))

    tau11 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau25 += 2 * np.einsum("ia->ia", tau11, optimize=True)

    tau11 = None

    tau12 = np.zeros((M, M))

    tau12 += np.einsum("ci,iabc->ab", t1, u[o, v, v, v], optimize=True)

    tau13 = np.zeros((N, M))

    tau13 += np.einsum("bi,ba->ia", l1, tau12, optimize=True)

    tau12 = None

    tau25 += 2 * np.einsum("ia->ia", tau13, optimize=True)

    tau13 = None

    tau14 = np.zeros((M, M))

    tau14 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau15 = np.zeros((N, M))

    tau15 += np.einsum("bc,ibac->ia", tau14, u[o, v, v, v], optimize=True)

    tau14 = None

    tau25 -= 2 * np.einsum("ia->ia", tau15, optimize=True)

    tau15 = None

    tau16 = np.zeros((N, M))

    tau16 += np.einsum("bj,abij->ia", l1, t2, optimize=True)

    tau18 += np.einsum("ia->ia", tau16, optimize=True)

    tau19 = np.zeros((N, M))

    tau19 += np.einsum("jb,jiba->ia", tau18, u[o, o, v, v], optimize=True)

    tau18 = None

    tau25 -= 2 * np.einsum("ia->ia", tau19, optimize=True)

    tau19 = None

    tau29 = np.zeros((N, N, N, M))

    tau29 -= np.einsum("ib,jkab->ijka", tau16, u[o, o, v, v], optimize=True)

    tau30 = np.zeros((N, N, N, M))

    tau30 -= np.einsum("ikja->ijka", tau29, optimize=True)

    tau29 = None

    tau34 = np.zeros((N, N))

    tau34 += np.einsum("ai,ja->ij", l1, tau16, optimize=True)

    tau16 = None

    tau35 = np.zeros((N, N, M, M))

    tau35 -= np.einsum("ik,jkab->ijab", tau34, u[o, o, v, v], optimize=True)

    tau34 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 -= np.einsum("ijba->ijab", tau35, optimize=True)

    tau35 = None

    tau22 = np.zeros((N, N, M, M))

    tau22 += np.einsum("baji->ijab", t2, optimize=True)

    tau22 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau23 = np.zeros((N, N))

    tau23 += np.einsum("kiab,kjab->ij", tau22, u[o, o, v, v], optimize=True)

    tau22 = None

    tau24 = np.zeros((N, M))

    tau24 += np.einsum("aj,ji->ia", l1, tau23, optimize=True)

    tau23 = None

    tau25 += np.einsum("ia->ia", tau24, optimize=True)

    tau24 = None

    tau26 += np.einsum("ai,jb->ijab", l1, tau25, optimize=True)

    tau25 = None

    r2 -= np.einsum("ijab->abij", tau26, optimize=True) / 2

    r2 += np.einsum("ijba->abij", tau26, optimize=True) / 2

    r2 += np.einsum("jiab->abij", tau26, optimize=True) / 2

    r2 -= np.einsum("jiba->abij", tau26, optimize=True) / 2

    tau26 = None

    tau27 = np.zeros((N, N, N, N))

    tau27 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau28 = np.zeros((N, N, N, M))

    tau28 += np.einsum("al,ijkl->ijka", l1, tau27, optimize=True)

    tau27 = None

    tau30 -= np.einsum("ikja->ijka", tau28, optimize=True)

    tau28 = None

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum("bk,kija->ijab", l1, tau30, optimize=True)

    tau30 = None

    r2 += np.einsum("jiba->abij", tau31, optimize=True)

    r2 -= np.einsum("jiab->abij", tau31, optimize=True)

    tau31 = None

    tau32 = np.zeros((N, N, M, M))

    tau32 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("ik,jkab->ijab", tau1, tau32, optimize=True)

    tau32 = None

    tau1 = None

    tau36 -= np.einsum("ijba->ijab", tau33, optimize=True)

    tau33 = None

    r2 += np.einsum("ijba->abij", tau36, optimize=True)

    r2 -= np.einsum("jiba->abij", tau36, optimize=True)

    tau36 = None

    tau37 = np.zeros((N, N, M, M))

    tau37 -= np.einsum("baji->ijab", t2, optimize=True)

    tau37 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau38 = np.zeros((N, N, N, N))

    tau38 -= np.einsum("lkab,jiab->ijkl", tau37, u[o, o, v, v], optimize=True)

    tau37 = None

    tau38 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau39 = np.zeros((N, N, N, M))

    tau39 += np.einsum("al,kjli->ijka", l1, tau38, optimize=True)

    tau38 = None

    r2 -= np.einsum("ak,kjib->abij", l1, tau39, optimize=True) / 2

    tau39 = None

    tau41 = np.zeros((N, M))

    tau41 += np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    tau48 -= np.einsum("ia->ia", tau41, optimize=True)

    tau41 = None

    tau44 = np.zeros((N, N))

    tau44 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau46 = np.zeros((N, N))

    tau46 += np.einsum("ij->ij", tau44, optimize=True)

    tau44 = None

    tau45 = np.zeros((N, N))

    tau45 += np.einsum("ak,ikja->ij", t1, u[o, o, o, v], optimize=True)

    tau46 += np.einsum("ij->ij", tau45, optimize=True)

    tau45 = None

    tau46 += np.einsum("ij->ij", f[o, o], optimize=True)

    tau47 = np.zeros((N, M))

    tau47 += np.einsum("aj,ij->ia", l1, tau46, optimize=True)

    tau46 = None

    tau48 += np.einsum("ia->ia", tau47, optimize=True)

    tau47 = None

    r2 -= np.einsum("ai,jb->abij", l1, tau48, optimize=True)

    r2 -= np.einsum("bj,ia->abij", l1, tau48, optimize=True)

    r2 += np.einsum("aj,ib->abij", l1, tau48, optimize=True)

    r2 += np.einsum("bi,ja->abij", l1, tau48, optimize=True)

    tau48 = None


def lambda_amplitudes_intermediates_qccsd_l2_addition_L1L2(r2, t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("ijba->ijab", tau0, optimize=True)

    tau86 = np.zeros((N, N, M, M))

    tau86 -= 2 * np.einsum("ijba->ijab", tau0, optimize=True)

    tau114 = np.zeros((N, N, M, M))

    tau114 += np.einsum("ijba->ijab", tau0, optimize=True)

    tau131 = np.zeros((N, N, M, M))

    tau131 -= np.einsum("ijba->ijab", tau0, optimize=True)

    tau0 = None

    tau1 = np.zeros((N, N))

    tau1 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau2 = np.zeros((N, N, M, M))

    tau2 -= np.einsum("ik,jkba->ijab", tau1, u[o, o, v, v], optimize=True)

    tau5 -= np.einsum("ijba->ijab", tau2, optimize=True)

    tau86 += 2 * np.einsum("ijba->ijab", tau2, optimize=True)

    tau114 -= np.einsum("ijba->ijab", tau2, optimize=True)

    tau131 += np.einsum("ijba->ijab", tau2, optimize=True)

    tau2 = None

    tau10 = np.zeros((N, N, N, M))

    tau10 += 2 * np.einsum("aj,ik->ijka", t1, tau1, optimize=True)

    tau82 = np.zeros((M, M))

    tau82 += np.einsum("ij,jaib->ab", tau1, u[o, v, o, v], optimize=True)

    tau97 = np.zeros((M, M))

    tau97 += 2 * np.einsum("ab->ab", tau82, optimize=True)

    tau82 = None

    tau88 = np.zeros((N, M))

    tau88 += np.einsum("aj,ji->ia", t1, tau1, optimize=True)

    tau89 = np.zeros((N, M))

    tau89 -= np.einsum("ia->ia", tau88, optimize=True)

    tau88 = None

    tau92 = np.zeros((N, M))

    tau92 += np.einsum("jk,ikja->ia", tau1, u[o, o, o, v], optimize=True)

    tau95 = np.zeros((N, M))

    tau95 += np.einsum("ia->ia", tau92, optimize=True)

    tau92 = None

    tau113 = np.zeros((N, N, N, N))

    tau113 -= np.einsum("im,jmlk->ijkl", tau1, u[o, o, o, o], optimize=True)

    tau124 = np.zeros((N, N, N, N))

    tau124 += 2 * np.einsum("ijlk->ijkl", tau113, optimize=True)

    tau113 = None

    tau120 = np.zeros((N, N, N, M))

    tau120 -= np.einsum("ak,ij->ijka", t1, tau1, optimize=True)

    tau122 = np.zeros((N, N, N, M))

    tau122 -= np.einsum("ak,ij->ijka", t1, tau1, optimize=True)

    tau129 = np.zeros((N, N))

    tau129 += np.einsum("kl,iljk->ij", tau1, u[o, o, o, o], optimize=True)

    tau135 = np.zeros((N, N))

    tau135 -= 2 * np.einsum("ij->ij", tau129, optimize=True)

    tau129 = None

    tau3 = np.zeros((N, M))

    tau3 -= np.einsum("bj,ijba->ia", t1, u[o, o, v, v], optimize=True)

    tau4 = np.zeros((N, M))

    tau4 += np.einsum("ia->ia", tau3, optimize=True)

    tau3 = None

    tau4 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau5 += np.einsum("ai,jb->ijab", l1, tau4, optimize=True)

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("caki,jkcb->ijab", t2, tau5, optimize=True)

    tau5 = None

    tau23 = np.zeros((N, N, M, M))

    tau23 += 2 * np.einsum("jiba->ijab", tau6, optimize=True)

    tau6 = None

    tau42 = np.zeros((N, N, N, M))

    tau42 += np.einsum("kb,baij->ijka", tau4, t2, optimize=True)

    tau43 = np.zeros((N, N, N, M))

    tau43 += 2 * np.einsum("kjia->ijka", tau42, optimize=True)

    tau66 = np.zeros((N, N, N, M))

    tau66 += 2 * np.einsum("kjia->ijka", tau42, optimize=True)

    tau42 = None

    tau86 += 2 * np.einsum("aj,ib->ijab", l1, tau4, optimize=True)

    tau104 = np.zeros((N, M, M, M))

    tau104 += np.einsum("jc,abji->iabc", tau4, t2, optimize=True)

    tau105 = np.zeros((N, M, M, M))

    tau105 -= np.einsum("ibac->iabc", tau104, optimize=True)

    tau104 = None

    tau114 += 2 * np.einsum("ai,jb->ijab", l1, tau4, optimize=True)

    tau115 = np.zeros((N, N, N, N))

    tau115 += np.einsum("abij,klba->ijkl", t2, tau114, optimize=True)

    tau114 = None

    tau124 -= np.einsum("lkij->ijkl", tau115, optimize=True)

    tau115 = None

    tau131 += 2 * np.einsum("bi,ja->ijab", l1, tau4, optimize=True)

    tau4 = None

    tau7 = np.zeros((N, N, N, M))

    tau7 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau8 = np.zeros((N, N, N, M))

    tau8 += np.einsum("kjia->ijka", tau7, optimize=True)

    tau39 = np.zeros((N, N, N, M))

    tau39 -= np.einsum("balj,iklb->ijka", t2, tau7, optimize=True)

    tau43 -= 2 * np.einsum("jkia->ijka", tau39, optimize=True)

    tau43 += 2 * np.einsum("kjia->ijka", tau39, optimize=True)

    tau39 = None

    tau52 = np.zeros((N, N, N, M))

    tau52 -= np.einsum("ikja->ijka", tau7, optimize=True)

    tau8 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau9 = np.zeros((N, N, N, M))

    tau9 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau10 -= np.einsum("ikja->ijka", tau9, optimize=True)

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("jklb,klia->ijab", tau10, tau8, optimize=True)

    tau10 = None

    tau23 -= np.einsum("jiab->ijab", tau11, optimize=True)

    tau11 = None

    tau26 = np.zeros((N, M, M, M))

    tau26 += np.einsum("abkj,ikjc->iabc", l2, tau9, optimize=True)

    tau31 = np.zeros((N, M, M, M))

    tau31 -= np.einsum("ibac->iabc", tau26, optimize=True)

    tau26 = None

    tau50 = np.zeros((N, N, N, M))

    tau50 += np.einsum("abjl,ilkb->ijka", l2, tau9, optimize=True)

    tau51 = np.zeros((N, N, N, M))

    tau51 -= 2 * np.einsum("ijka->ijka", tau50, optimize=True)

    tau51 += 2 * np.einsum("jika->ijka", tau50, optimize=True)

    tau126 = np.zeros((N, N, N, M))

    tau126 += 2 * np.einsum("ijka->ijka", tau50, optimize=True)

    tau50 = None

    tau120 -= np.einsum("ikja->ijka", tau9, optimize=True)

    tau121 = np.zeros((N, N, N, N))

    tau121 += np.einsum("kmla,mija->ijkl", tau120, u[o, o, o, v], optimize=True)

    tau120 = None

    tau124 -= 4 * np.einsum("jkil->ijkl", tau121, optimize=True)

    tau121 = None

    tau122 -= 2 * np.einsum("ikja->ijka", tau9, optimize=True)

    tau123 = np.zeros((N, N, N, N))

    tau123 += np.einsum("kmla,imja->ijkl", tau122, tau7, optimize=True)

    tau122 = None

    tau7 = None

    tau124 -= 2 * np.einsum("ljik->ijkl", tau123, optimize=True)

    tau123 = None

    tau138 = np.zeros((N, N, N, N))

    tau138 += np.einsum("lkma,mjia->ijkl", tau8, tau9, optimize=True)

    tau12 = np.zeros((N, M, M, M))

    tau12 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau13 = np.zeros((N, M, M, M))

    tau13 += np.einsum("iacb->iabc", tau12, optimize=True)

    tau32 = np.zeros((N, M, M, M))

    tau32 -= np.einsum("iacb->iabc", tau12, optimize=True)

    tau12 = None

    tau13 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau14 = np.zeros((N, N, M, M))

    tau14 += np.einsum("kacb,ikjc->ijab", tau13, tau9, optimize=True)

    tau9 = None

    tau23 += 2 * np.einsum("ijba->ijab", tau14, optimize=True)

    tau14 = None

    tau15 = np.zeros((N, M, M, M))

    tau15 += np.einsum("di,abcd->iabc", t1, u[v, v, v, v], optimize=True)

    tau16 = np.zeros((N, M, M, M))

    tau16 += np.einsum("ibac->iabc", tau15, optimize=True)

    tau105 += np.einsum("ibac->iabc", tau15, optimize=True)

    tau15 = None

    tau16 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau17 = np.zeros((N, N, M, M))

    tau17 += np.einsum("ci,jcab->ijab", l1, tau16, optimize=True)

    tau23 += 2 * np.einsum("ijba->ijab", tau17, optimize=True)

    tau17 = None

    tau68 = np.zeros((N, M))

    tau68 += np.einsum("bcji,jbca->ia", l2, tau16, optimize=True)

    tau16 = None

    tau79 = np.zeros((N, M))

    tau79 += 2 * np.einsum("ia->ia", tau68, optimize=True)

    tau68 = None

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau19 = np.zeros((N, N, M, M))

    tau19 -= np.einsum("jiab->ijab", tau18, optimize=True)

    tau102 = np.zeros((N, N, M, M))

    tau102 += 2 * np.einsum("ijab->ijab", tau18, optimize=True)

    tau116 = np.zeros((N, N, M, M))

    tau116 -= np.einsum("jiab->ijab", tau18, optimize=True)

    tau18 = None

    tau19 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau20 = np.zeros((N, N, N, M))

    tau20 += np.einsum("bi,jkba->ijka", l1, tau19, optimize=True)

    tau21 = np.zeros((N, N, M, M))

    tau21 += np.einsum("ak,ikjb->ijab", t1, tau20, optimize=True)

    tau20 = None

    tau23 += 2 * np.einsum("ijba->ijab", tau21, optimize=True)

    tau21 = None

    tau22 = np.zeros((N, N, M, M))

    tau22 += np.einsum("ik,kjab->ijab", tau1, tau19, optimize=True)

    tau1 = None

    tau23 -= 2 * np.einsum("ijba->ijab", tau22, optimize=True)

    tau22 = None

    tau24 = np.zeros((N, N, M, M))

    tau24 += np.einsum("caki,jkbc->ijab", l2, tau23, optimize=True)

    tau23 = None

    tau80 = np.zeros((N, N, M, M))

    tau80 -= 2 * np.einsum("jiab->ijab", tau24, optimize=True)

    tau24 = None

    tau25 = np.zeros((M, M))

    tau25 -= np.einsum("acji,cbji->ab", l2, t2, optimize=True)

    tau31 += np.einsum("ai,bc->iabc", l1, tau25, optimize=True)

    tau61 = np.zeros((N, M))

    tau61 += np.einsum("bc,ibac->ia", tau25, u[o, v, v, v], optimize=True)

    tau79 += 2 * np.einsum("ia->ia", tau61, optimize=True)

    tau61 = None

    tau27 = np.zeros((N, M))

    tau27 += np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau28 = np.zeros((N, M, M, M))

    tau28 -= np.einsum("jc,abij->iabc", tau27, l2, optimize=True)

    tau31 += 2 * np.einsum("ibac->iabc", tau28, optimize=True)

    tau28 = None

    tau49 = np.zeros((N, N, N, M))

    tau49 -= np.einsum("kb,abij->ijka", tau27, l2, optimize=True)

    tau51 -= 2 * np.einsum("ijka->ijka", tau49, optimize=True)

    tau139 = np.zeros((N, N, N, M))

    tau139 += 2 * np.einsum("ijka->ijka", tau49, optimize=True)

    tau49 = None

    tau89 += np.einsum("ia->ia", tau27, optimize=True)

    tau90 = np.zeros((M, M))

    tau90 += np.einsum("ic,iacb->ab", tau89, u[o, v, v, v], optimize=True)

    tau97 -= 2 * np.einsum("ab->ab", tau90, optimize=True)

    tau90 = None

    tau94 = np.zeros((N, M))

    tau94 += np.einsum("jb,jiba->ia", tau89, u[o, o, v, v], optimize=True)

    tau95 += np.einsum("ia->ia", tau94, optimize=True)

    tau94 = None

    tau133 = np.zeros((N, N))

    tau133 += np.einsum("ka,kija->ij", tau89, u[o, o, o, v], optimize=True)

    tau89 = None

    tau135 -= 2 * np.einsum("ij->ij", tau133, optimize=True)

    tau133 = None

    tau138 += 2 * np.einsum("ia,lkja->ijkl", tau27, tau8, optimize=True)

    tau27 = None

    r2 -= np.einsum("bakl,klij->abij", l2, tau138, optimize=True) / 2

    tau138 = None

    tau29 = np.zeros((N, N, M, M))

    tau29 += np.einsum("acik,cbkj->ijab", l2, t2, optimize=True)

    tau30 = np.zeros((N, M, M, M))

    tau30 += np.einsum("aj,ijbc->iabc", l1, tau29, optimize=True)

    tau31 -= 2 * np.einsum("iabc->iabc", tau30, optimize=True)

    tau31 += 2 * np.einsum("ibac->iabc", tau30, optimize=True)

    tau30 = None

    tau60 = np.zeros((N, M))

    tau60 -= np.einsum("ijbc,jbac->ia", tau29, u[o, v, v, v], optimize=True)

    tau79 += 4 * np.einsum("ia->ia", tau60, optimize=True)

    tau60 = None

    tau32 += np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("icad,jcdb->ijab", tau31, tau32, optimize=True)

    tau32 = None

    tau31 = None

    tau80 += 2 * np.einsum("ijab->ijab", tau33, optimize=True)

    tau33 = None

    tau34 = np.zeros((N, N, N, M))

    tau34 += np.einsum("al,iljk->ijka", t1, u[o, o, o, o], optimize=True)

    tau43 -= 2 * np.einsum("ikja->ijka", tau34, optimize=True)

    tau34 = None

    tau35 = np.zeros((N, N, N, M))

    tau35 += np.einsum("bi,jakb->ijka", t1, u[o, v, o, v], optimize=True)

    tau38 = np.zeros((N, N, N, M))

    tau38 += np.einsum("ijka->ijka", tau35, optimize=True)

    tau66 -= 4 * np.einsum("kija->ijka", tau35, optimize=True)

    tau35 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum("baji->ijab", t2, optimize=True)

    tau36 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau37 = np.zeros((N, N, N, M))

    tau37 += np.einsum("lkba,lijb->ijka", tau36, u[o, o, o, v], optimize=True)

    tau36 = None

    tau38 -= np.einsum("jkia->ijka", tau37, optimize=True)

    tau37 = None

    tau43 += 2 * np.einsum("jika->ijka", tau38, optimize=True)

    tau43 -= 2 * np.einsum("kija->ijka", tau38, optimize=True)

    tau38 = None

    tau40 = np.zeros((N, N, M, M))

    tau40 -= np.einsum("baji->ijab", t2, optimize=True)

    tau40 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau41 = np.zeros((N, N, N, M))

    tau41 += np.einsum("iabc,jkbc->ijka", tau13, tau40, optimize=True)

    tau40 = None

    tau43 -= np.einsum("ikja->ijka", tau41, optimize=True)

    tau66 -= np.einsum("ikja->ijka", tau41, optimize=True)

    tau41 = None

    tau43 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau44 = np.zeros((N, N, M, M))

    tau44 += np.einsum("ak,ikjb->ijab", l1, tau43, optimize=True)

    tau43 = None

    tau45 = np.zeros((N, N, M, M))

    tau45 -= np.einsum("caki,jkbc->ijab", l2, tau44, optimize=True)

    tau44 = None

    tau80 -= 2 * np.einsum("ijba->ijab", tau45, optimize=True)

    tau45 = None

    tau46 = np.zeros((N, N))

    tau46 -= np.einsum("baik,bakj->ij", l2, t2, optimize=True)

    tau51 += np.einsum("ai,jk->ijka", l1, tau46, optimize=True)

    tau54 = np.zeros((N, M))

    tau54 -= np.einsum("ja,ij->ia", f[o, v], tau46, optimize=True)

    tau79 += 2 * np.einsum("ia->ia", tau54, optimize=True)

    tau54 = None

    tau56 = np.zeros((N, M))

    tau56 += np.einsum("jk,ikja->ia", tau46, u[o, o, o, v], optimize=True)

    tau79 += 2 * np.einsum("ia->ia", tau56, optimize=True)

    tau56 = None

    tau72 = np.zeros((N, N, N, M))

    tau72 += 2 * np.einsum("aj,ik->ijka", t1, tau46, optimize=True)

    tau75 = np.zeros((N, M))

    tau75 += np.einsum("aj,ji->ia", t1, tau46, optimize=True)

    tau77 = np.zeros((N, M))

    tau77 += np.einsum("ia->ia", tau75, optimize=True)

    tau75 = None

    tau126 -= np.einsum("ai,jk->ijka", l1, tau46, optimize=True)

    tau46 = None

    tau127 = np.zeros((N, N, M, M))

    tau127 += np.einsum("ijkc,kcab->ijab", tau126, tau13, optimize=True)

    tau126 = None

    tau137 = np.zeros((N, N, M, M))

    tau137 -= 2 * np.einsum("ijba->ijab", tau127, optimize=True)

    tau127 = None

    tau47 = np.zeros((N, N, N, N))

    tau47 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau48 = np.zeros((N, N, N, M))

    tau48 += np.einsum("al,ijkl->ijka", l1, tau47, optimize=True)

    tau51 += np.einsum("ijka->ijka", tau48, optimize=True)

    tau139 -= np.einsum("ijka->ijka", tau48, optimize=True)

    tau48 = None

    r2 += np.einsum("kcba,ijkc->abij", tau13, tau139, optimize=True) / 2

    tau139 = None

    tau13 = None

    tau55 = np.zeros((N, M))

    tau55 += np.einsum("ijlk,lkja->ia", tau47, u[o, o, o, v], optimize=True)

    tau79 -= np.einsum("ia->ia", tau55, optimize=True)

    tau55 = None

    tau69 = np.zeros((N, N, N, M))

    tau69 -= np.einsum("al,ilkj->ijka", t1, tau47, optimize=True)

    tau72 -= np.einsum("ikja->ijka", tau69, optimize=True)

    tau69 = None

    tau108 = np.zeros((N, N, N, M))

    tau108 += np.einsum("ijlm,lmka->ijka", tau47, tau8, optimize=True)

    tau8 = None

    tau47 = None

    tau110 = np.zeros((N, N, N, M))

    tau110 -= np.einsum("jika->ijka", tau108, optimize=True)

    tau108 = None

    tau52 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau53 = np.zeros((N, N, M, M))

    tau53 += np.einsum("kila,kljb->ijab", tau51, tau52, optimize=True)

    tau51 = None

    tau80 += 2 * np.einsum("ijab->ijab", tau53, optimize=True)

    tau53 = None

    tau62 = np.zeros((N, N, N, M))

    tau62 += np.einsum("bali,jlkb->ijka", t2, tau52, optimize=True)

    tau66 += 4 * np.einsum("jkia->ijka", tau62, optimize=True)

    tau62 = None

    tau85 = np.zeros((N, N, M, M))

    tau85 += np.einsum("ak,kijb->ijab", l1, tau52, optimize=True)

    tau86 += np.einsum("jiab->ijab", tau85, optimize=True)

    tau87 = np.zeros((M, M))

    tau87 += np.einsum("caij,jicb->ab", t2, tau86, optimize=True)

    tau86 = None

    tau97 += np.einsum("ab->ab", tau87, optimize=True)

    tau87 = None

    tau131 += 2 * np.einsum("jiab->ijab", tau85, optimize=True)

    tau85 = None

    tau132 = np.zeros((N, N))

    tau132 += np.einsum("abki,kjba->ij", t2, tau131, optimize=True)

    tau131 = None

    tau135 += np.einsum("ji->ij", tau132, optimize=True)

    tau132 = None

    tau107 = np.zeros((N, N, N, M))

    tau107 += np.einsum("liab,ljkb->ijka", tau29, tau52, optimize=True)

    tau29 = None

    tau110 += 4 * np.einsum("kjia->ijka", tau107, optimize=True)

    tau107 = None

    tau109 = np.zeros((N, N, N, M))

    tau109 += np.einsum("ab,ijkb->ijka", tau25, tau52, optimize=True)

    tau25 = None

    tau52 = None

    tau110 -= 2 * np.einsum("kjia->ijka", tau109, optimize=True)

    tau109 = None

    tau57 = np.zeros((N, N, N, M))

    tau57 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau58 = np.zeros((N, N, N, N))

    tau58 += np.einsum("ak,ijla->ijkl", t1, tau57, optimize=True)

    tau59 = np.zeros((N, M))

    tau59 -= np.einsum("iljk,kjla->ia", tau58, u[o, o, o, v], optimize=True)

    tau58 = None

    tau79 -= 2 * np.einsum("ia->ia", tau59, optimize=True)

    tau59 = None

    tau74 = np.zeros((N, M))

    tau74 += np.einsum("kjba,jikb->ia", tau19, tau57, optimize=True)

    tau19 = None

    tau79 += 4 * np.einsum("ia->ia", tau74, optimize=True)

    tau74 = None

    tau76 = np.zeros((N, M))

    tau76 += np.einsum("bakj,kjib->ia", t2, tau57, optimize=True)

    tau77 += np.einsum("ia->ia", tau76, optimize=True)

    tau76 = None

    tau78 = np.zeros((N, M))

    tau78 += np.einsum("jb,jiba->ia", tau77, u[o, o, v, v], optimize=True)

    tau77 = None

    tau79 -= 2 * np.einsum("ia->ia", tau78, optimize=True)

    tau78 = None

    tau63 = np.zeros((N, N, N, N))

    tau63 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau64 = np.zeros((N, N, N, N))

    tau64 -= 2 * np.einsum("kjil->ijkl", tau63, optimize=True)

    tau63 = None

    tau64 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau65 = np.zeros((N, N, N, M))

    tau65 += np.einsum("al,lijk->ijka", t1, tau64, optimize=True)

    tau64 = None

    tau66 -= 2 * np.einsum("ikja->ijka", tau65, optimize=True)

    tau65 = None

    tau66 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau67 = np.zeros((N, M))

    tau67 += np.einsum("bajk,ijkb->ia", l2, tau66, optimize=True)

    tau66 = None

    tau79 -= np.einsum("ia->ia", tau67, optimize=True)

    tau67 = None

    tau70 = np.zeros((N, N, M, M))

    tau70 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau70 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau71 = np.zeros((N, N, N, M))

    tau71 += np.einsum("lijb,lkba->ijka", tau57, tau70, optimize=True)

    tau57 = None

    tau72 += 2 * np.einsum("ikja->ijka", tau71, optimize=True)

    tau71 = None

    tau73 = np.zeros((N, M))

    tau73 += np.einsum("ijkb,jkba->ia", tau72, u[o, o, v, v], optimize=True)

    tau72 = None

    tau79 -= np.einsum("ia->ia", tau73, optimize=True)

    tau73 = None

    tau80 -= np.einsum("ai,jb->ijab", l1, tau79, optimize=True)

    tau79 = None

    r2 -= np.einsum("ijab->abij", tau80, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau80, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau80, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau80, optimize=True) / 4

    tau80 = None

    tau101 = np.zeros((N, N, M, M))

    tau101 += np.einsum("kjcb,kica->ijab", tau70, u[o, o, v, v], optimize=True)

    tau70 = None

    tau102 += np.einsum("jiba->ijab", tau101, optimize=True)

    tau101 = None

    tau81 = np.zeros((M, M))

    tau81 -= np.einsum("ci,caib->ab", l1, u[v, v, o, v], optimize=True)

    tau97 += 2 * np.einsum("ab->ab", tau81, optimize=True)

    tau81 = None

    tau83 = np.zeros((M, M))

    tau83 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau84 = np.zeros((M, M))

    tau84 += np.einsum("cd,acbd->ab", tau83, u[v, v, v, v], optimize=True)

    tau97 -= 2 * np.einsum("ab->ab", tau84, optimize=True)

    tau84 = None

    tau93 = np.zeros((N, M))

    tau93 += np.einsum("bc,ibac->ia", tau83, u[o, v, v, v], optimize=True)

    tau95 += np.einsum("ia->ia", tau93, optimize=True)

    tau93 = None

    tau130 = np.zeros((N, N))

    tau130 += np.einsum("ab,iajb->ij", tau83, u[o, v, o, v], optimize=True)

    tau83 = None

    tau135 += 2 * np.einsum("ij->ij", tau130, optimize=True)

    tau130 = None

    tau91 = np.zeros((N, M))

    tau91 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau95 -= np.einsum("ia->ia", tau91, optimize=True)

    tau91 = None

    tau96 = np.zeros((M, M))

    tau96 += np.einsum("ai,ib->ab", t1, tau95, optimize=True)

    tau97 += 2 * np.einsum("ab->ab", tau96, optimize=True)

    tau96 = None

    tau98 = np.zeros((N, N, M, M))

    tau98 += np.einsum("cb,caij->ijab", tau97, l2, optimize=True)

    tau97 = None

    tau112 = np.zeros((N, N, M, M))

    tau112 += 2 * np.einsum("jiab->ijab", tau98, optimize=True)

    tau98 = None

    tau134 = np.zeros((N, N))

    tau134 += np.einsum("ai,ja->ij", t1, tau95, optimize=True)

    tau95 = None

    tau135 += 2 * np.einsum("ji->ij", tau134, optimize=True)

    tau134 = None

    tau99 = np.zeros((N, M, M, M))

    tau99 += np.einsum("daji,jbcd->iabc", t2, u[o, v, v, v], optimize=True)

    tau105 += 2 * np.einsum("ibac->iabc", tau99, optimize=True)

    tau99 = None

    tau100 = np.zeros((N, N, M, M))

    tau100 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau102 += np.einsum("jiab->ijab", tau100, optimize=True)

    tau100 = None

    tau102 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau103 = np.zeros((N, M, M, M))

    tau103 += np.einsum("aj,ijbc->iabc", t1, tau102, optimize=True)

    tau102 = None

    tau105 -= np.einsum("ibac->iabc", tau103, optimize=True)

    tau103 = None

    tau105 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau106 = np.zeros((N, N, N, M))

    tau106 += np.einsum("bcij,kbca->ijka", l2, tau105, optimize=True)

    tau105 = None

    tau110 -= 2 * np.einsum("jika->ijka", tau106, optimize=True)

    tau106 = None

    tau111 = np.zeros((N, N, M, M))

    tau111 += np.einsum("ak,ijkb->ijab", l1, tau110, optimize=True)

    tau110 = None

    tau112 -= np.einsum("jiab->ijab", tau111, optimize=True)

    tau111 = None

    r2 += np.einsum("jiab->abij", tau112, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau112, optimize=True) / 4

    tau112 = None

    tau116 += 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau117 = np.zeros((N, N, N, M))

    tau117 += np.einsum("bi,jkab->ijka", t1, tau116, optimize=True)

    tau116 = None

    tau118 = np.zeros((N, N, N, M))

    tau118 -= np.einsum("jika->ijka", tau117, optimize=True)

    tau117 = None

    tau118 -= np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau119 = np.zeros((N, N, N, N))

    tau119 += np.einsum("ai,jkla->ijkl", l1, tau118, optimize=True)

    tau118 = None

    tau124 -= 2 * np.einsum("ijlk->ijkl", tau119, optimize=True)

    tau119 = None

    tau125 = np.zeros((N, N, M, M))

    tau125 += np.einsum("abkl,ijkl->ijab", l2, tau124, optimize=True)

    tau124 = None

    tau137 -= np.einsum("ijba->ijab", tau125, optimize=True)

    tau125 = None

    tau128 = np.zeros((N, N))

    tau128 -= np.einsum("ak,iakj->ij", l1, u[o, v, o, o], optimize=True)

    tau135 += 2 * np.einsum("ij->ij", tau128, optimize=True)

    tau128 = None

    tau136 = np.zeros((N, N, M, M))

    tau136 += np.einsum("jk,abki->ijab", tau135, l2, optimize=True)

    tau135 = None

    tau137 += 2 * np.einsum("ijba->ijab", tau136, optimize=True)

    tau136 = None

    r2 += np.einsum("ijba->abij", tau137, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau137, optimize=True) / 4

    tau137 = None


def lambda_amplitudes_intermediates_qccsd_l2_addition_L2L2(r2, t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((M, M, M, M))

    tau0 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau106 = np.zeros((M, M, M, M))

    tau106 -= np.einsum("aefb,cedf->abcd", tau0, u[v, v, v, v], optimize=True)

    tau117 = np.zeros((M, M, M, M))

    tau117 += 2 * np.einsum("acbd->abcd", tau106, optimize=True)

    tau106 = None

    tau130 = np.zeros((M, M, M, M))

    tau130 += np.einsum("afde,becf->abcd", tau0, tau0, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("ijab->ijab", tau1, optimize=True)

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("ijab->ijab", tau1, optimize=True)

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum("ijab->ijab", tau1, optimize=True)

    tau54 = np.zeros((N, N, M, M))

    tau54 += np.einsum("jiab->ijab", tau1, optimize=True)

    tau59 = np.zeros((N, N, M, M))

    tau59 -= np.einsum("jiab->ijab", tau1, optimize=True)

    tau113 = np.zeros((N, N, M, M))

    tau113 += np.einsum("ijab->ijab", tau1, optimize=True)

    tau1 = None

    tau2 = np.zeros((N, N, M, M))

    tau2 -= np.einsum("caki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau3 += np.einsum("ijab->ijab", tau2, optimize=True)

    tau3 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum("cadb,ijcd->ijab", tau0, tau3, optimize=True)

    tau33 = np.zeros((N, N, M, M))

    tau33 -= 2 * np.einsum("ijab->ijab", tau4, optimize=True)

    tau4 = None

    tau5 = np.zeros((M, M))

    tau5 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau6 = np.zeros((N, N, M, M))

    tau6 -= np.einsum("ac,ijbc->ijab", tau5, u[o, o, v, v], optimize=True)

    tau17 = np.zeros((N, N, M, M))

    tau17 -= 2 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau92 = np.zeros((N, N, M, M))

    tau92 -= 4 * np.einsum("ijba->ijab", tau6, optimize=True)

    tau119 = np.zeros((N, N, M, M))

    tau119 -= np.einsum("ijab->ijab", tau6, optimize=True)

    tau123 = np.zeros((N, N, M, M))

    tau123 += 2 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau6 = None

    tau26 = np.zeros((N, M, M, M))

    tau26 += np.einsum("bi,ac->iabc", t1, tau5, optimize=True)

    tau69 = np.zeros((N, N, M, M))

    tau69 -= np.einsum("cb,acji->ijab", tau5, t2, optimize=True)

    tau70 = np.zeros((N, N, M, M))

    tau70 += 2 * np.einsum("ijab->ijab", tau69, optimize=True)

    tau133 = np.zeros((N, N, M, M))

    tau133 += np.einsum("ijab->ijab", tau69, optimize=True)

    tau69 = None

    tau91 = np.zeros((N, N))

    tau91 += np.einsum("ab,iajb->ij", tau5, u[o, v, o, v], optimize=True)

    tau103 = np.zeros((N, N))

    tau103 += 4 * np.einsum("ji->ij", tau91, optimize=True)

    tau91 = None

    tau99 = np.zeros((N, M))

    tau99 -= np.einsum("bc,ibca->ia", tau5, u[o, v, v, v], optimize=True)

    tau101 = np.zeros((N, M))

    tau101 += np.einsum("ia->ia", tau99, optimize=True)

    tau99 = None

    tau122 = np.zeros((M, M))

    tau122 -= np.einsum("cd,cabd->ab", tau5, u[v, v, v, v], optimize=True)

    tau127 = np.zeros((M, M))

    tau127 -= 4 * np.einsum("ab->ab", tau122, optimize=True)

    tau122 = None

    tau130 += np.einsum("ac,bd->abcd", tau5, tau5, optimize=True)

    tau7 = np.zeros((N, N, M, M))

    tau7 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau10 += np.einsum("jiab->ijab", tau7, optimize=True)

    tau54 += np.einsum("ijab->ijab", tau7, optimize=True)

    tau7 = None

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("baji->ijab", t2, optimize=True)

    tau8 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum("kjcb,kica->ijab", tau8, u[o, o, v, v], optimize=True)

    tau8 = None

    tau10 += np.einsum("jiba->ijab", tau9, optimize=True)

    tau54 += np.einsum("ijba->ijab", tau9, optimize=True)

    tau9 = None

    tau10 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("caki,kjcb->ijab", l2, tau10, optimize=True)

    tau17 += 4 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau47 = np.zeros((N, N, M, M))

    tau47 -= 4 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau76 = np.zeros((N, N, M, M))

    tau76 += 4 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau92 -= 8 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau119 += 4 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau123 -= 8 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau11 = None

    tau12 = np.zeros((N, N, N, N))

    tau12 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau15 = np.zeros((N, N, N, N))

    tau15 -= 4 * np.einsum("ljik->ijkl", tau12, optimize=True)

    tau29 = np.zeros((N, N, N, N))

    tau29 -= 2 * np.einsum("ljik->ijkl", tau12, optimize=True)

    tau82 = np.zeros((N, N, N, N))

    tau82 -= np.einsum("ljik->ijkl", tau12, optimize=True)

    tau12 = None

    tau13 = np.zeros((N, N, M, M))

    tau13 -= np.einsum("baji->ijab", t2, optimize=True)

    tau13 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau14 = np.zeros((N, N, N, N))

    tau14 += np.einsum("klab,ijab->ijkl", tau13, u[o, o, v, v], optimize=True)

    tau13 = None

    tau15 -= np.einsum("jilk->ijkl", tau14, optimize=True)

    tau14 = None

    tau15 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau16 = np.zeros((N, N, M, M))

    tau16 += np.einsum("abkl,ijkl->ijab", l2, tau15, optimize=True)

    tau15 = None

    tau17 += np.einsum("jiba->ijab", tau16, optimize=True)

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum("caki,kjbc->ijab", t2, tau17, optimize=True)

    tau17 = None

    tau33 -= np.einsum("ijba->ijab", tau18, optimize=True)

    tau18 = None

    tau92 -= np.einsum("jiba->ijab", tau16, optimize=True)

    tau16 = None

    tau19 = np.zeros((N, N, M, M))

    tau19 -= np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum("kjbc,kiac->ijab", tau10, tau19, optimize=True)

    tau10 = None

    tau33 += 4 * np.einsum("ijab->ijab", tau20, optimize=True)

    tau20 = None

    tau25 = np.zeros((N, M, M, M))

    tau25 += np.einsum("bj,jiac->iabc", t1, tau19, optimize=True)

    tau26 += 2 * np.einsum("iacb->iabc", tau25, optimize=True)

    tau25 = None

    tau53 = np.zeros((N, N, M, M))

    tau53 += np.einsum("ikca,jkcb->ijab", tau19, tau3, optimize=True)

    tau61 = np.zeros((N, N, M, M))

    tau61 += 4 * np.einsum("ijab->ijab", tau53, optimize=True)

    tau53 = None

    tau63 = np.zeros((N, N, M, M))

    tau63 += np.einsum("acbd,ijdc->ijab", tau0, tau19, optimize=True)

    tau72 = np.zeros((N, N, M, M))

    tau72 += 2 * np.einsum("ijab->ijab", tau63, optimize=True)

    tau63 = None

    tau65 = np.zeros((N, N, M, M))

    tau65 += np.einsum("ikcb,kjac->ijab", tau19, tau19, optimize=True)

    tau72 -= 4 * np.einsum("ijab->ijab", tau65, optimize=True)

    tau65 = None

    tau67 = np.zeros((N, N, M, M))

    tau67 += np.einsum("caki,kjcb->ijab", t2, tau19, optimize=True)

    tau70 += 4 * np.einsum("ijba->ijab", tau67, optimize=True)

    tau131 = np.zeros((N, N, M, M))

    tau131 += 2 * np.einsum("ijab->ijab", tau67, optimize=True)

    tau133 += 2 * np.einsum("ijba->ijab", tau67, optimize=True)

    tau67 = None

    tau134 = np.zeros((N, N, N, N))

    tau134 += np.einsum("ijab,lkab->ijkl", tau133, u[o, o, v, v], optimize=True)

    tau133 = None

    r2 -= np.einsum("bakl,klij->abij", l2, tau134, optimize=True) / 4

    tau134 = None

    tau78 = np.zeros((N, N, N, N))

    tau78 += np.einsum("ijab,klab->ijkl", tau19, tau3, optimize=True)

    tau3 = None

    tau88 = np.zeros((N, N, N, N))

    tau88 += 8 * np.einsum("ijlk->ijkl", tau78, optimize=True)

    tau78 = None

    tau107 = np.zeros((M, M, M, M))

    tau107 += np.einsum("ijab,ijcd->abcd", tau19, tau2, optimize=True)

    tau2 = None

    tau117 += 4 * np.einsum("acbd->abcd", tau107, optimize=True)

    tau107 = None

    tau130 += 4 * np.einsum("ijad,jibc->abcd", tau19, tau19, optimize=True)

    r2 -= np.einsum("abcd,jicd->abij", tau130, u[o, o, v, v], optimize=True) / 4

    tau130 = None

    tau132 = np.zeros((N, N, N, N))

    tau132 += 4 * np.einsum("ikba,jlab->ijkl", tau19, tau19, optimize=True)

    tau21 = np.zeros((N, N, N, M))

    tau21 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau22 = np.zeros((N, N, N, M))

    tau22 += np.einsum("kjia->ijka", tau21, optimize=True)

    tau57 = np.zeros((N, N, N, M))

    tau57 -= np.einsum("ikja->ijka", tau21, optimize=True)

    tau84 = np.zeros((N, N, N, M))

    tau84 += np.einsum("kjia->ijka", tau21, optimize=True)

    tau21 = None

    tau22 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau23 = np.zeros((N, N, N, M))

    tau23 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau24 = np.zeros((N, M, M, M))

    tau24 += np.einsum("bckj,kjia->iabc", t2, tau23, optimize=True)

    tau26 -= np.einsum("iacb->iabc", tau24, optimize=True)

    tau24 = None

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("kijc,kabc->ijab", tau22, tau26, optimize=True)

    tau26 = None

    tau33 += 2 * np.einsum("jiab->ijab", tau27, optimize=True)

    tau27 = None

    tau35 = np.zeros((N, N, N, M))

    tau35 -= np.einsum("balk,iljb->ijka", t2, tau23, optimize=True)

    tau36 = np.zeros((N, N, N, M))

    tau36 -= np.einsum("iljb,klab->ijka", tau35, u[o, o, v, v], optimize=True)

    tau37 = np.zeros((N, N, M, M))

    tau37 += np.einsum("ak,ijkb->ijab", t1, tau36, optimize=True)

    tau36 = None

    tau61 += 4 * np.einsum("ijab->ijab", tau37, optimize=True)

    tau37 = None

    tau56 = np.zeros((N, N, N, M))

    tau56 += 2 * np.einsum("ijka->ijka", tau35, optimize=True)

    tau80 = np.zeros((N, N, N, M))

    tau80 += 2 * np.einsum("ijka->ijka", tau35, optimize=True)

    tau35 = None

    tau95 = np.zeros((N, M))

    tau95 += np.einsum("bakj,kjib->ia", t2, tau23, optimize=True)

    tau23 = None

    tau96 = np.zeros((N, M))

    tau96 += np.einsum("ia->ia", tau95, optimize=True)

    tau95 = None

    tau28 = np.zeros((N, N, N, N))

    tau28 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau29 += np.einsum("lkji->ijkl", tau28, optimize=True)

    tau29 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("klab,likj->ijab", tau19, tau29, optimize=True)

    tau29 = None

    tau33 -= 2 * np.einsum("jiab->ijab", tau30, optimize=True)

    tau30 = None

    tau31 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau32 = np.zeros((N, N, M, M))

    tau32 += np.einsum("ac,ijbc->ijab", tau5, tau31, optimize=True)

    tau31 = None

    tau33 -= 2 * np.einsum("ijab->ijab", tau32, optimize=True)

    tau32 = None

    tau34 = np.zeros((N, N, M, M))

    tau34 += np.einsum("caki,kjbc->ijab", l2, tau33, optimize=True)

    tau33 = None

    tau74 = np.zeros((N, N, M, M))

    tau74 -= np.einsum("ijba->ijab", tau34, optimize=True)

    tau34 = None

    tau38 = np.zeros((N, N, N, N))

    tau38 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau39 = np.zeros((N, N, M, M))

    tau39 -= np.einsum("jilk,lkab->ijab", tau38, u[o, o, v, v], optimize=True)

    tau47 += np.einsum("ijba->ijab", tau39, optimize=True)

    tau123 += np.einsum("ijba->ijab", tau39, optimize=True)

    tau39 = None

    tau64 = np.zeros((N, N, M, M))

    tau64 += np.einsum("klab,iljk->ijab", tau19, tau38, optimize=True)

    tau72 += 2 * np.einsum("ijab->ijab", tau64, optimize=True)

    tau64 = None

    tau66 = np.zeros((N, N, M, M))

    tau66 -= np.einsum("ablk,lkji->ijab", t2, tau38, optimize=True)

    tau70 += np.einsum("ijba->ijab", tau66, optimize=True)

    tau66 = None

    tau75 = np.zeros((N, N, N, N))

    tau75 += np.einsum("mjln,imnk->ijkl", tau28, tau38, optimize=True)

    tau28 = None

    tau88 -= 2 * np.einsum("ijlk->ijkl", tau75, optimize=True)

    tau75 = None

    tau79 = np.zeros((N, N, N, M))

    tau79 -= np.einsum("al,ilkj->ijka", t1, tau38, optimize=True)

    tau80 += np.einsum("ikja->ijka", tau79, optimize=True)

    tau79 = None

    tau81 = np.zeros((N, N, N, N))

    tau81 += np.einsum("mija,kmla->ijkl", tau22, tau80, optimize=True)

    tau80 = None

    tau22 = None

    tau88 -= 4 * np.einsum("klij->ijkl", tau81, optimize=True)

    tau81 = None

    tau120 = np.zeros((N, N, M, M))

    tau120 += np.einsum("klab,ijlk->ijab", tau119, tau38, optimize=True)

    tau119 = None

    tau129 = np.zeros((N, N, M, M))

    tau129 -= np.einsum("jiab->ijab", tau120, optimize=True)

    tau120 = None

    tau132 += np.einsum("inkm,jmln->ijkl", tau38, tau38, optimize=True)

    tau40 = np.zeros((N, N))

    tau40 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau41 = np.zeros((N, N, M, M))

    tau41 -= np.einsum("ik,jkab->ijab", tau40, u[o, o, v, v], optimize=True)

    tau47 -= 2 * np.einsum("ijba->ijab", tau41, optimize=True)

    tau76 += np.einsum("ijba->ijab", tau41, optimize=True)

    tau77 = np.zeros((N, N, N, N))

    tau77 += np.einsum("abij,klba->ijkl", t2, tau76, optimize=True)

    tau76 = None

    tau88 -= np.einsum("ljik->ijkl", tau77, optimize=True)

    tau77 = None

    tau92 -= 2 * np.einsum("ijba->ijab", tau41, optimize=True)

    tau93 = np.zeros((N, N))

    tau93 += np.einsum("abki,kjab->ij", t2, tau92, optimize=True)

    tau92 = None

    tau103 -= np.einsum("ij->ij", tau93, optimize=True)

    tau93 = None

    tau123 += 4 * np.einsum("jiba->ijab", tau41, optimize=True)

    tau41 = None

    tau56 -= np.einsum("aj,ik->ijka", t1, tau40, optimize=True)

    tau68 = np.zeros((N, N, M, M))

    tau68 += np.einsum("kj,abik->ijab", tau40, t2, optimize=True)

    tau70 -= 2 * np.einsum("ijba->ijab", tau68, optimize=True)

    tau71 = np.zeros((N, N, M, M))

    tau71 += np.einsum("caki,kjcb->ijab", l2, tau70, optimize=True)

    tau70 = None

    tau72 += np.einsum("ijab->ijab", tau71, optimize=True)

    tau71 = None

    tau131 += np.einsum("ijba->ijab", tau68, optimize=True)

    tau68 = None

    tau132 += np.einsum("abji,klab->ijkl", l2, tau131, optimize=True)

    tau131 = None

    tau72 -= np.einsum("ij,ab->ijab", tau40, tau5, optimize=True)

    tau73 = np.zeros((N, N, M, M))

    tau73 += np.einsum("jkbc,kica->ijab", tau72, u[o, o, v, v], optimize=True)

    tau72 = None

    tau74 -= np.einsum("jiba->ijab", tau73, optimize=True)

    tau73 = None

    tau90 = np.zeros((N, N))

    tau90 -= np.einsum("kl,ilkj->ij", tau40, u[o, o, o, o], optimize=True)

    tau103 -= 4 * np.einsum("ji->ij", tau90, optimize=True)

    tau90 = None

    tau94 = np.zeros((N, M))

    tau94 += np.einsum("aj,ji->ia", t1, tau40, optimize=True)

    tau96 += np.einsum("ia->ia", tau94, optimize=True)

    tau94 = None

    tau97 = np.zeros((N, N))

    tau97 += np.einsum("ka,kija->ij", tau96, u[o, o, o, v], optimize=True)

    tau103 += 4 * np.einsum("ji->ij", tau97, optimize=True)

    tau97 = None

    tau100 = np.zeros((N, M))

    tau100 += np.einsum("jb,jiba->ia", tau96, u[o, o, v, v], optimize=True)

    tau101 -= np.einsum("ia->ia", tau100, optimize=True)

    tau100 = None

    tau125 = np.zeros((M, M))

    tau125 += np.einsum("ic,iacb->ab", tau96, u[o, v, v, v], optimize=True)

    tau96 = None

    tau127 += 4 * np.einsum("ab->ab", tau125, optimize=True)

    tau125 = None

    tau98 = np.zeros((N, M))

    tau98 += np.einsum("jk,ikja->ia", tau40, u[o, o, o, v], optimize=True)

    tau101 += np.einsum("ia->ia", tau98, optimize=True)

    tau98 = None

    tau102 = np.zeros((N, N))

    tau102 += np.einsum("aj,ia->ij", t1, tau101, optimize=True)

    tau103 += 4 * np.einsum("ji->ij", tau102, optimize=True)

    tau102 = None

    tau104 = np.zeros((N, N, M, M))

    tau104 += np.einsum("kj,abki->ijab", tau103, l2, optimize=True)

    tau103 = None

    tau105 = np.zeros((N, N, M, M))

    tau105 += np.einsum("ijba->ijab", tau104, optimize=True)

    tau104 = None

    tau126 = np.zeros((M, M))

    tau126 += np.einsum("bi,ia->ab", t1, tau101, optimize=True)

    tau101 = None

    tau127 += 4 * np.einsum("ba->ab", tau126, optimize=True)

    tau126 = None

    tau121 = np.zeros((M, M))

    tau121 += np.einsum("ij,jaib->ab", tau40, u[o, v, o, v], optimize=True)

    tau127 += 4 * np.einsum("ab->ab", tau121, optimize=True)

    tau121 = None

    tau132 -= np.einsum("ik,jl->ijkl", tau40, tau40, optimize=True)

    r2 += np.einsum("ijkl,klba->abij", tau132, u[o, o, v, v], optimize=True) / 4

    tau132 = None

    tau42 = np.zeros((N, M, M, M))

    tau42 -= np.einsum("aj,jibc->iabc", t1, u[o, o, v, v], optimize=True)

    tau43 = np.zeros((N, M, M, M))

    tau43 += np.einsum("iacb->iabc", tau42, optimize=True)

    tau42 = None

    tau43 -= 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau44 = np.zeros((M, M, M, M))

    tau44 += np.einsum("ai,ibcd->abcd", t1, tau43, optimize=True)

    tau43 = None

    tau45 = np.zeros((M, M, M, M))

    tau45 -= np.einsum("badc->abcd", tau44, optimize=True)

    tau115 = np.zeros((M, M, M, M))

    tau115 += np.einsum("abdc->abcd", tau44, optimize=True)

    tau44 = None

    tau45 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau46 = np.zeros((N, N, M, M))

    tau46 += np.einsum("cdij,cdab->ijab", l2, tau45, optimize=True)

    tau45 = None

    tau47 -= 2 * np.einsum("jiba->ijab", tau46, optimize=True)

    tau48 = np.zeros((N, N, M, M))

    tau48 += np.einsum("caki,jkcb->ijab", t2, tau47, optimize=True)

    tau47 = None

    tau61 += np.einsum("jiab->ijab", tau48, optimize=True)

    tau48 = None

    tau123 -= 2 * np.einsum("jiba->ijab", tau46, optimize=True)

    tau46 = None

    tau124 = np.zeros((M, M))

    tau124 += np.einsum("cbij,ijca->ab", t2, tau123, optimize=True)

    tau123 = None

    tau127 -= np.einsum("ba->ab", tau124, optimize=True)

    tau124 = None

    tau128 = np.zeros((N, N, M, M))

    tau128 += np.einsum("ca,cbij->ijab", tau127, l2, optimize=True)

    tau127 = None

    tau129 += np.einsum("jiba->ijab", tau128, optimize=True)

    tau128 = None

    tau49 = np.zeros((M, M, M, M))

    tau49 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau51 = np.zeros((M, M, M, M))

    tau51 += np.einsum("badc->abcd", tau49, optimize=True)

    tau49 = None

    tau50 = np.zeros((M, M, M, M))

    tau50 += np.einsum("ai,ibcd->abcd", t1, u[o, v, v, v], optimize=True)

    tau51 += 2 * np.einsum("abdc->abcd", tau50, optimize=True)

    tau51 -= 2 * np.einsum("badc->abcd", tau50, optimize=True)

    tau110 = np.zeros((M, M, M, M))

    tau110 += 2 * np.einsum("abdc->abcd", tau50, optimize=True)

    tau110 -= 2 * np.einsum("badc->abcd", tau50, optimize=True)

    tau50 = None

    tau51 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau52 = np.zeros((N, N, M, M))

    tau52 += np.einsum("ijcd,cadb->ijab", tau19, tau51, optimize=True)

    tau51 = None

    tau61 -= 2 * np.einsum("ijab->ijab", tau52, optimize=True)

    tau52 = None

    tau54 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau55 = np.zeros((N, N, M, M))

    tau55 += np.einsum("kilj,lkab->ijab", tau38, tau54, optimize=True)

    tau54 = None

    tau61 -= 2 * np.einsum("ijab->ijab", tau55, optimize=True)

    tau55 = None

    tau57 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau58 = np.zeros((N, N, M, M))

    tau58 += np.einsum("ikla,jlkb->ijab", tau56, tau57, optimize=True)

    tau56 = None

    tau61 -= 2 * np.einsum("ijab->ijab", tau58, optimize=True)

    tau58 = None

    tau112 = np.zeros((N, N, M, M))

    tau112 += np.einsum("ak,ikjb->ijab", t1, tau57, optimize=True)

    tau57 = None

    tau113 += np.einsum("ijab->ijab", tau112, optimize=True)

    tau112 = None

    tau59 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau60 = np.zeros((N, N, M, M))

    tau60 += np.einsum("ik,kjab->ijab", tau40, tau59, optimize=True)

    tau59 = None

    tau61 += 2 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau60 = None

    tau62 = np.zeros((N, N, M, M))

    tau62 += np.einsum("caki,jkcb->ijab", l2, tau61, optimize=True)

    tau61 = None

    tau74 -= np.einsum("jiab->ijab", tau62, optimize=True)

    tau62 = None

    r2 += np.einsum("ijab->abij", tau74, optimize=True) / 4

    r2 -= np.einsum("ijba->abij", tau74, optimize=True) / 4

    r2 -= np.einsum("jiab->abij", tau74, optimize=True) / 4

    r2 += np.einsum("jiba->abij", tau74, optimize=True) / 4

    tau74 = None

    tau82 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau83 = np.zeros((N, N, N, N))

    tau83 += np.einsum("minj,nkml->ijkl", tau38, tau82, optimize=True)

    tau82 = None

    tau38 = None

    tau88 += 4 * np.einsum("ijkl->ijkl", tau83, optimize=True)

    tau83 = None

    tau84 -= 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau85 = np.zeros((N, N, N, N))

    tau85 += np.einsum("ai,jkla->ijkl", t1, tau84, optimize=True)

    tau84 = None

    tau86 = np.zeros((N, N, N, N))

    tau86 -= np.einsum("kjil->ijkl", tau85, optimize=True)

    tau85 = None

    tau86 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau87 = np.zeros((N, N, N, N))

    tau87 += np.einsum("im,mjkl->ijkl", tau40, tau86, optimize=True)

    tau40 = None

    tau86 = None

    tau88 += 2 * np.einsum("iklj->ijkl", tau87, optimize=True)

    tau87 = None

    tau89 = np.zeros((N, N, M, M))

    tau89 += np.einsum("abkl,ikjl->ijab", l2, tau88, optimize=True)

    tau88 = None

    tau105 -= np.einsum("ijba->ijab", tau89, optimize=True)

    tau89 = None

    r2 += np.einsum("ijba->abij", tau105, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau105, optimize=True) / 8

    tau105 = None

    tau108 = np.zeros((N, N, M, M))

    tau108 += np.einsum("baji->ijab", t2, optimize=True)

    tau108 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau109 = np.zeros((M, M, M, M))

    tau109 += np.einsum("ijab,ijcd->abcd", tau108, u[o, o, v, v], optimize=True)

    tau108 = None

    tau110 += np.einsum("badc->abcd", tau109, optimize=True)

    tau109 = None

    tau111 = np.zeros((M, M, M, M))

    tau111 += np.einsum("ecfd,eafb->abcd", tau0, tau110, optimize=True)

    tau0 = None

    tau110 = None

    tau117 -= np.einsum("cdab->abcd", tau111, optimize=True)

    tau111 = None

    tau113 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau114 = np.zeros((M, M, M, M))

    tau114 += np.einsum("ijab,ijcd->abcd", tau113, tau19, optimize=True)

    tau113 = None

    tau19 = None

    tau117 -= 4 * np.einsum("cdab->abcd", tau114, optimize=True)

    tau114 = None

    tau115 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau116 = np.zeros((M, M, M, M))

    tau116 += np.einsum("de,abec->abcd", tau5, tau115, optimize=True)

    tau115 = None

    tau5 = None

    tau117 += np.einsum("cbda->abcd", tau116, optimize=True)

    tau116 = None

    tau118 = np.zeros((N, N, M, M))

    tau118 += np.einsum("cdij,acdb->ijab", l2, tau117, optimize=True)

    tau117 = None

    tau129 += 2 * np.einsum("jiab->ijab", tau118, optimize=True)

    tau118 = None

    r2 += np.einsum("jiab->abij", tau129, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau129, optimize=True) / 8

    tau129 = None
