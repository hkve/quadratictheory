import numpy as np
from clusterfock.cc.rhs.l_inter_CCSD import lambda_amplitudes_intermediates_ccsd


def lambda_amplitudes_intermediates_qccsd(t1, t2, l1, l2, u, f, v, o):
    r1, r2 = lambda_amplitudes_intermediates_ccsd(t1, t2, l1, l2, u, f, v, o)

    r1 += lambda_amplitudes_intermediates_qccsd_l1_addition(t1, t2, l1, l2, u, f, v, o)
    r2 += lambda_amplitudes_intermediates_qccsd_l2_addition(t1, t2, l1, l2, u, f, v, o)

    r2 = 0.25 * (
        r2 - r2.transpose(1, 0, 2, 3) - r2.transpose(0, 1, 3, 2) + r2.transpose(1, 0, 3, 2)
    )

    return r1, r2


def lambda_amplitudes_intermediates_qccsd_l1_addition(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M))

    tau0 += np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau1 = zeros((N, N))

    tau1 += np.einsum("ai,ja->ij", l1, tau0, optimize=True)

    tau66 = zeros((N, N, N, M))

    tau66 += 8 * np.einsum("aj,ik->ijka", t1, tau1, optimize=True)

    tau78 = zeros((N, M))

    tau78 += 4 * np.einsum("aj,ji->ia", t1, tau1, optimize=True)

    r1 = zeros((M, N))

    r1 -= np.einsum("ja,ij->ai", f[o, v], tau1, optimize=True)

    r1 += np.einsum("jk,ikja->ai", tau1, u[o, o, o, v], optimize=True)

    tau1 = None

    tau3 = zeros((M, M))

    tau3 += np.einsum("ai,ib->ab", l1, tau0, optimize=True)

    r1 += np.einsum("bc,ibac->ai", tau3, u[o, v, v, v], optimize=True)

    tau3 = None

    tau15 = zeros((N, N, M, M))

    tau15 += 4 * np.einsum("ai,jb->ijab", t1, tau0, optimize=True)

    tau21 = zeros((N, M))

    tau21 -= 2 * np.einsum("ia->ia", tau0, optimize=True)

    tau25 = zeros((N, N, N, N))

    tau25 += np.einsum("ia,kjla->ijkl", tau0, u[o, o, o, v], optimize=True)

    tau49 = zeros((N, N, N, M))

    tau49 += 4 * np.einsum("al,iljk->ijka", t1, tau25, optimize=True)

    tau25 = None

    tau26 = zeros((N, N, M, M))

    tau26 -= np.einsum("ic,jabc->ijab", tau0, u[o, v, v, v], optimize=True)

    tau49 += 4 * np.einsum("bk,ijab->ijka", t1, tau26, optimize=True)

    tau26 = None

    tau33 = zeros((N, M, M, M))

    tau33 += np.einsum("jb,ijca->iabc", tau0, u[o, o, v, v], optimize=True)

    tau40 = zeros((N, N, N, M))

    tau40 -= 2 * np.einsum("ib,kjab->ijka", tau0, u[o, o, v, v], optimize=True)

    tau41 = zeros((N, N, M, M))

    tau41 += 2 * np.einsum("ai,jb->ijab", t1, tau0, optimize=True)

    tau41 += 4 * np.einsum("bj,ia->ijab", t1, tau0, optimize=True)

    tau42 = zeros((N, N, M, M))

    tau42 += 4 * np.einsum("ai,jb->ijab", t1, tau0, optimize=True)

    tau49 -= 2 * np.einsum("la,jlik->ijka", tau0, u[o, o, o, o], optimize=True)

    tau49 += 4 * np.einsum("kb,jaib->ijka", tau0, u[o, v, o, v], optimize=True)

    tau51 = zeros((N, N, M, M))

    tau51 += 4 * np.einsum("bi,ja->ijab", t1, tau0, optimize=True)

    tau51 += 2 * np.einsum("aj,ib->ijab", t1, tau0, optimize=True)

    tau53 = zeros((N, N, N, M))

    tau53 -= np.einsum("kb,baji->ijka", tau0, l2, optimize=True)

    tau55 = zeros((N, N, N, M))

    tau55 -= 2 * np.einsum("ijka->ijka", tau53, optimize=True)

    tau68 = zeros((N, N, N, M))

    tau68 += np.einsum("ijka->ijka", tau53, optimize=True)

    tau71 = zeros((N, N, N, M))

    tau71 -= 2 * np.einsum("ijka->ijka", tau53, optimize=True)

    tau77 = zeros((N, N, N, M))

    tau77 -= 2 * np.einsum("ijka->ijka", tau53, optimize=True)

    tau53 = None

    tau67 = zeros((N, M, M, M))

    tau67 -= 2 * np.einsum("jc,baji->iabc", tau0, l2, optimize=True)

    tau74 = zeros((N, N, M, M))

    tau74 += 2 * np.einsum("ai,jb->ijab", t1, tau0, optimize=True)

    tau2 = zeros((M, M))

    tau2 -= np.einsum("ci,caib->ab", l1, u[v, v, o, v], optimize=True)

    r1 -= np.einsum("bi,ba->ai", l1, tau2, optimize=True)

    tau2 = None

    tau4 = zeros((N, N, N, M))

    tau4 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau18 = zeros((N, N, N, M))

    tau18 -= 2 * np.einsum("ikja->ijka", tau4, optimize=True)

    tau34 = zeros((N, N, N, M))

    tau34 -= 2 * np.einsum("ikja->ijka", tau4, optimize=True)

    tau54 = zeros((N, N, N, M))

    tau54 -= np.einsum("ablj,ilkb->ijka", l2, tau4, optimize=True)

    tau55 -= 2 * np.einsum("ijka->ijka", tau54, optimize=True)

    tau55 += 2 * np.einsum("jika->ijka", tau54, optimize=True)

    tau68 += np.einsum("ijka->ijka", tau54, optimize=True)

    tau68 -= np.einsum("jika->ijka", tau54, optimize=True)

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum("bk,ikja->ijab", t1, tau68, optimize=True)

    tau68 = None

    r1 += np.einsum("ijbc,jbca->ai", tau69, u[o, v, v, v], optimize=True)

    tau69 = None

    tau70 = zeros((N, N, N, M))

    tau70 += 2 * np.einsum("jika->ijka", tau54, optimize=True)

    tau71 += 2 * np.einsum("jika->ijka", tau54, optimize=True)

    tau77 += 4 * np.einsum("jika->ijka", tau54, optimize=True)

    tau54 = None

    tau56 = zeros((N, N, N, M))

    tau56 += np.einsum("ikja->ijka", tau4, optimize=True)

    tau61 = zeros((N, N, N, N))

    tau61 -= np.einsum("aj,ikla->ijkl", l1, tau4, optimize=True)

    tau64 = zeros((N, N, N, N))

    tau64 -= 2 * np.einsum("ijlk->ijkl", tau61, optimize=True)

    tau73 = zeros((N, N, N, N))

    tau73 += 2 * np.einsum("ijlk->ijkl", tau61, optimize=True)

    tau61 = None

    tau67 += np.einsum("bakj,ikjc->iabc", l2, tau4, optimize=True)

    r1 += np.einsum("ibcd,bcda->ai", tau67, u[v, v, v, v], optimize=True) / 4

    tau67 = None

    tau5 = zeros((N, N, N, M))

    tau5 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau17 = zeros((N, N, N, M))

    tau17 += np.einsum("balk,lijb->ijka", t2, tau5, optimize=True)

    tau18 += 2 * np.einsum("ikja->ijka", tau17, optimize=True)

    tau34 += 2 * np.einsum("ikja->ijka", tau17, optimize=True)

    tau56 += 2 * np.einsum("ijka->ijka", tau17, optimize=True)

    tau60 = zeros((N, N, N, M))

    tau60 += 2 * np.einsum("ijka->ijka", tau17, optimize=True)

    tau19 = zeros((N, M))

    tau19 += np.einsum("bakj,kjib->ia", t2, tau5, optimize=True)

    tau21 += np.einsum("ia->ia", tau19, optimize=True)

    tau65 = zeros((N, M))

    tau65 += np.einsum("ia->ia", tau19, optimize=True)

    tau19 = None

    tau24 = zeros((N, N, M, M))

    tau24 += 2 * np.einsum("ilkb,lkja->ijab", tau4, tau5, optimize=True)

    tau27 = zeros((N, M, M, M))

    tau27 += np.einsum("bckj,kjia->iabc", t2, tau5, optimize=True)

    tau29 = zeros((N, M, M, M))

    tau29 -= np.einsum("iacb->iabc", tau27, optimize=True)

    tau57 = zeros((N, M, M, M))

    tau57 += np.einsum("iacb->iabc", tau27, optimize=True)

    tau27 = None

    tau62 = zeros((N, N, N, N))

    tau62 += np.einsum("imla,mjka->ijkl", tau4, tau5, optimize=True)

    tau64 -= 4 * np.einsum("ijkl->ijkl", tau62, optimize=True)

    tau73 += 4 * np.einsum("jilk->ijkl", tau62, optimize=True)

    tau75 = zeros((N, N, N, N))

    tau75 += 2 * np.einsum("ijkl->ijkl", tau62, optimize=True)

    tau62 = None

    tau6 = zeros((N, N, M, M))

    tau6 -= np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("caki,kjcb->ijab", t2, tau6, optimize=True)

    tau15 += 4 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau41 += 4 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau42 += 4 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau51 += 4 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau63 = zeros((N, N, M, M))

    tau63 += 2 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau72 = zeros((N, N, M, M))

    tau72 += 2 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau24 -= 4 * np.einsum("ikcb,kjac->ijab", tau6, tau6, optimize=True)

    tau28 = zeros((N, M, M, M))

    tau28 += np.einsum("bj,jiac->iabc", t1, tau6, optimize=True)

    tau29 += 2 * np.einsum("iacb->iabc", tau28, optimize=True)

    tau57 += 2 * np.einsum("iabc->iabc", tau28, optimize=True)

    tau28 = None

    tau33 += 2 * np.einsum("jkab,ikjc->iabc", tau6, u[o, o, o, v], optimize=True)

    tau37 = zeros((N, N, M, M))

    tau37 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau66 += 4 * np.einsum("lkjb,ilba->ijka", tau56, tau6, optimize=True)

    tau56 = None

    tau66 -= 4 * np.einsum("kbac,ijcb->ijka", tau57, tau6, optimize=True)

    tau57 = None

    tau73 += 4 * np.einsum("ikab,jlba->ijkl", tau6, tau6, optimize=True)

    tau76 = zeros((N, M, M, M))

    tau76 -= 2 * np.einsum("aj,ijbc->iabc", l1, tau6, optimize=True)

    tau7 = zeros((M, M, M, M))

    tau7 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau24 += 2 * np.einsum("ijdc,acbd->ijab", tau6, tau7, optimize=True)

    tau33 += np.einsum("daeb,idce->iabc", tau7, u[o, v, v, v], optimize=True)

    tau7 = None

    tau8 = zeros((N, N, N, N))

    tau8 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 -= np.einsum("ablk,lkji->ijab", t2, tau8, optimize=True)

    tau15 -= np.einsum("ijba->ijab", tau9, optimize=True)

    tau41 -= np.einsum("ijba->ijab", tau9, optimize=True)

    tau42 -= np.einsum("ijba->ijab", tau9, optimize=True)

    tau49 += np.einsum("klba,ljib->ijka", tau42, u[o, o, o, v], optimize=True)

    tau42 = None

    tau51 += np.einsum("ijba->ijab", tau9, optimize=True)

    tau9 = None

    tau16 = zeros((N, N, N, M))

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

    tau52 = zeros((N, N, N, M))

    tau52 -= np.einsum("al,jikl->ijka", l1, tau8, optimize=True)

    tau55 += np.einsum("ijka->ijka", tau52, optimize=True)

    tau70 += np.einsum("ijka->ijka", tau52, optimize=True)

    tau77 += np.einsum("ijka->ijka", tau52, optimize=True)

    tau52 = None

    tau66 -= 2 * np.einsum("la,ilkj->ijka", tau0, tau8, optimize=True)

    tau73 -= np.einsum("jnlm,mikn->ijkl", tau8, tau8, optimize=True)

    tau11 = zeros((M, M))

    tau11 -= np.einsum("acji,cbji->ab", l2, t2, optimize=True)

    tau15 += 2 * np.einsum("cb,acji->ijab", tau11, t2, optimize=True)

    tau23 = zeros((M, M))

    tau23 += np.einsum("ab->ab", tau11, optimize=True)

    tau33 += np.einsum("ad,ibcd->iabc", tau11, u[o, v, v, v], optimize=True)

    tau66 += 2 * np.einsum("ba,ikjb->ijka", tau11, tau4, optimize=True)

    tau12 = zeros((N, N))

    tau12 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau14 = zeros((N, N))

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

    tau13 = zeros((N, N))

    tau13 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau14 += np.einsum("ij->ij", tau13, optimize=True)

    tau15 += 2 * np.einsum("ki,bakj->ijab", tau14, t2, optimize=True)

    tau24 += np.einsum("caki,jkcb->ijab", l2, tau15, optimize=True)

    tau15 = None

    tau20 = zeros((N, M))

    tau20 += np.einsum("aj,ji->ia", t1, tau14, optimize=True)

    tau21 += np.einsum("ia->ia", tau20, optimize=True)

    tau24 += 2 * np.einsum("ai,jb->ijab", l1, tau21, optimize=True)

    tau46 = zeros((N, M))

    tau46 += np.einsum("jb,jiba->ia", tau21, u[o, o, v, v], optimize=True)

    tau48 = zeros((N, M))

    tau48 -= np.einsum("ia->ia", tau46, optimize=True)

    tau46 = None

    tau66 -= 4 * np.einsum("ik,ja->ijka", tau12, tau21, optimize=True)

    tau84 = zeros((N, N))

    tau84 += 4 * np.einsum("ka,kija->ij", tau21, u[o, o, o, v], optimize=True)

    tau21 = None

    tau65 += np.einsum("ia->ia", tau20, optimize=True)

    tau20 = None

    tau34 += np.einsum("aj,ik->ijka", t1, tau14, optimize=True)

    tau40 -= np.einsum("kilb,ljba->ijka", tau34, u[o, o, v, v], optimize=True)

    tau34 = None

    tau40 -= np.einsum("kl,ljia->ijka", tau14, u[o, o, o, v], optimize=True)

    tau47 = zeros((N, M))

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

    tau83 = zeros((N, N, M, M))

    tau83 -= 2 * np.einsum("ik,kjba->ijab", tau14, u[o, o, v, v], optimize=True)

    tau84 -= 4 * np.einsum("kl,likj->ij", tau14, u[o, o, o, o], optimize=True)

    tau50 = zeros((N, N, M, M))

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

    tau66 += 4 * np.einsum("ja,ik->ijka", tau0, tau13, optimize=True)

    tau0 = None

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

    tau22 = zeros((M, M))

    tau22 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau23 += 2 * np.einsum("ab->ab", tau22, optimize=True)

    tau22 = None

    tau24 -= np.einsum("ij,ab->ijab", tau14, tau23, optimize=True)

    tau14 = None

    r1 -= np.einsum("ijbc,jbca->ai", tau24, u[o, v, v, v], optimize=True) / 4

    tau24 = None

    tau29 += np.einsum("bi,ac->iabc", t1, tau23, optimize=True)

    tau45 = zeros((N, M))

    tau45 += np.einsum("bc,ibca->ia", tau23, u[o, v, v, v], optimize=True)

    tau48 -= np.einsum("ia->ia", tau45, optimize=True)

    tau45 = None

    tau51 += 2 * np.einsum("cb,caji->ijab", tau23, t2, optimize=True)

    tau66 -= 2 * np.einsum("likb,ljba->ijka", tau5, tau51, optimize=True)

    tau51 = None

    tau5 = None

    tau76 += np.einsum("ai,bc->iabc", l1, tau23, optimize=True)

    r1 += np.einsum("icbd,bcda->ai", tau76, u[v, v, v, v], optimize=True) / 2

    tau76 = None

    tau83 -= 4 * np.einsum("ac,jicb->ijab", tau23, u[o, o, v, v], optimize=True)

    tau84 += 4 * np.einsum("ab,iajb->ij", tau23, u[o, v, o, v], optimize=True)

    tau23 = None

    tau29 -= 2 * np.einsum("aj,cbij->iabc", l1, t2, optimize=True)

    tau33 -= np.einsum("jabd,jidc->iabc", tau29, u[o, o, v, v], optimize=True)

    tau29 = None

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("jiab->ijab", tau30, optimize=True)

    tau80 = zeros((N, N, M, M))

    tau80 += np.einsum("jiab->ijab", tau30, optimize=True)

    tau30 = None

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau32 += np.einsum("ijab->ijab", tau31, optimize=True)

    tau80 += np.einsum("ijab->ijab", tau31, optimize=True)

    tau31 = None

    tau32 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau33 += 2 * np.einsum("aj,jibc->iabc", l1, tau32, optimize=True)

    tau32 = None

    tau49 += np.einsum("cbki,jbac->ijka", t2, tau33, optimize=True)

    tau33 = None

    tau35 = zeros((N, N, N, M))

    tau35 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau36 = zeros((N, N, N, M))

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

    tau38 = zeros((N, N, N, N))

    tau38 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau39 = zeros((N, N, N, N))

    tau39 += 2 * np.einsum("kjil->ijkl", tau38, optimize=True)

    tau81 = zeros((N, N, N, N))

    tau81 -= 4 * np.einsum("ljik->ijkl", tau38, optimize=True)

    tau38 = None

    tau39 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau39 += np.einsum("balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau40 += np.einsum("al,kjli->ijka", l1, tau39, optimize=True)

    tau39 = None

    tau40 -= 2 * np.einsum("bk,jbia->ijka", l1, u[o, v, o, v], optimize=True)

    tau49 += 2 * np.einsum("bali,kjlb->ijka", t2, tau40, optimize=True)

    tau40 = None

    tau43 = zeros((N, M, M, M))

    tau43 += np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau43 -= np.einsum("aj,ijcb->iabc", t1, u[o, o, v, v], optimize=True)

    tau49 += 2 * np.einsum("ikcb,jabc->ijka", tau10, tau43, optimize=True)

    tau10 = None

    tau43 = None

    tau44 = zeros((N, M))

    tau44 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau48 -= 2 * np.einsum("ia->ia", tau44, optimize=True)

    tau44 = None

    tau49 -= np.einsum("jb,baki->ijka", tau48, t2, optimize=True)

    r1 += np.einsum("bajk,jikb->ai", l2, tau49, optimize=True) / 4

    tau49 = None

    tau84 += 4 * np.einsum("aj,ia->ij", t1, tau48, optimize=True)

    tau48 = None

    tau58 = zeros((N, N, M, M))

    tau58 -= np.einsum("baji->ijab", t2, optimize=True)

    tau58 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau59 = zeros((N, N, N, N))

    tau59 += np.einsum("abji,lkab->ijkl", l2, tau58, optimize=True)

    tau66 += np.einsum("ilma,lmkj->ijka", tau4, tau59, optimize=True)

    tau59 = None

    tau4 = None

    r1 -= np.einsum("ijkb,jkba->ai", tau66, u[o, o, v, v], optimize=True) / 8

    tau66 = None

    tau81 -= np.einsum("lkab,jiab->ijkl", tau58, u[o, o, v, v], optimize=True)

    tau58 = None

    tau79 = zeros((N, N, M, M))

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

    tau82 = zeros((N, M))

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

    return r1


def lambda_amplitudes_intermediates_qccsd_l2_addition(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, M))

    tau0 += np.einsum("al,ijkl->ijka", l1, u[o, o, o, o], optimize=True)

    r2 = zeros((M, M, N, N))

    r2 -= np.einsum("ak,jikb->abij", l1, tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, M, M, M))

    tau1 += np.einsum("di,adbc->iabc", l1, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cj,icab->abij", l1, tau1, optimize=True)

    tau1 = None

    tau2 = zeros((M, M))

    tau2 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau3 = zeros((M, M, M, M))

    tau3 += np.einsum("ae,cbde->abcd", tau2, u[v, v, v, v], optimize=True)

    tau35 = zeros((M, M, M, M))

    tau35 -= np.einsum("acbd->abcd", tau3, optimize=True)

    tau3 = None

    tau44 = zeros((M, M))

    tau44 += np.einsum("ab->ab", tau2, optimize=True)

    tau215 = zeros((M, M, M, M))

    tau215 -= np.einsum("ad,bc->abcd", tau2, tau2, optimize=True)

    tau4 = zeros((M, M, M, M))

    tau4 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau5 = zeros((M, M, M, M))

    tau5 -= np.einsum("aefb,cedf->abcd", tau4, u[v, v, v, v], optimize=True)

    tau35 -= 2 * np.einsum("acbd->abcd", tau5, optimize=True)

    tau5 = None

    tau215 -= np.einsum("afce,bedf->abcd", tau4, tau4, optimize=True)

    tau6 = zeros((M, M, M, M))

    tau6 += np.einsum("ai,ibcd->abcd", t1, u[o, v, v, v], optimize=True)

    tau9 = zeros((M, M, M, M))

    tau9 += 2 * np.einsum("abdc->abcd", tau6, optimize=True)

    tau9 -= 2 * np.einsum("badc->abcd", tau6, optimize=True)

    tau93 = zeros((M, M, M, M))

    tau93 += 2 * np.einsum("abdc->abcd", tau6, optimize=True)

    tau93 -= 2 * np.einsum("badc->abcd", tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("baji->ijab", t2, optimize=True)

    tau7 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau8 = zeros((M, M, M, M))

    tau8 += np.einsum("ijab,ijcd->abcd", tau7, u[o, o, v, v], optimize=True)

    tau9 += np.einsum("badc->abcd", tau8, optimize=True)

    tau8 = None

    tau10 = zeros((M, M, M, M))

    tau10 += np.einsum("ecfd,eafb->abcd", tau4, tau9, optimize=True)

    tau9 = None

    tau35 += np.einsum("cdab->abcd", tau10, optimize=True)

    tau10 = None

    tau180 = zeros((N, N))

    tau180 += np.einsum("kiab,kjab->ij", tau7, u[o, o, v, v], optimize=True)

    tau7 = None

    tau181 = zeros((N, M))

    tau181 += np.einsum("aj,ji->ia", l1, tau180, optimize=True)

    tau180 = None

    tau182 = zeros((N, M))

    tau182 += 2 * np.einsum("ia->ia", tau181, optimize=True)

    tau181 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("ijab->ijab", tau11, optimize=True)

    tau119 = zeros((N, M, M, M))

    tau119 += np.einsum("bj,jiac->iabc", t1, tau11, optimize=True)

    tau120 = zeros((N, M, M, M))

    tau120 += 2 * np.einsum("iacb->iabc", tau119, optimize=True)

    tau119 = None

    tau136 = zeros((N, N, M, M))

    tau136 += np.einsum("ikcb,kjac->ijab", tau11, tau11, optimize=True)

    tau148 = zeros((N, N, M, M))

    tau148 += 4 * np.einsum("ijab->ijab", tau136, optimize=True)

    tau136 = None

    tau137 = zeros((N, N, M, M))

    tau137 += np.einsum("ijdc,acbd->ijab", tau11, tau4, optimize=True)

    tau148 -= 2 * np.einsum("ijab->ijab", tau137, optimize=True)

    tau137 = None

    tau140 = zeros((N, N, M, M))

    tau140 += np.einsum("bcjk,kica->ijab", t2, tau11, optimize=True)

    tau143 = zeros((N, N, M, M))

    tau143 += 4 * np.einsum("ijba->ijab", tau140, optimize=True)

    tau216 = zeros((N, N, M, M))

    tau216 += 2 * np.einsum("ijab->ijab", tau140, optimize=True)

    tau218 = zeros((N, N, M, M))

    tau218 -= 2 * np.einsum("ijba->ijab", tau140, optimize=True)

    tau140 = None

    tau152 = zeros((N, M, M, M))

    tau152 += np.einsum("aj,ijbc->iabc", l1, tau11, optimize=True)

    tau153 = zeros((N, M, M, M))

    tau153 -= 2 * np.einsum("iabc->iabc", tau152, optimize=True)

    tau153 += 2 * np.einsum("ibac->iabc", tau152, optimize=True)

    tau152 = None

    tau196 = zeros((N, N, M, M))

    tau196 += 2 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau215 += 4 * np.einsum("ijad,jibc->abcd", tau11, tau11, optimize=True)

    tau217 = zeros((N, N, N, N))

    tau217 -= 4 * np.einsum("ikba,jlab->ijkl", tau11, tau11, optimize=True)

    tau12 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau179 = zeros((N, M))

    tau179 += np.einsum("ijbc,jbca->ia", tau12, u[o, v, v, v], optimize=True)

    tau182 -= 4 * np.einsum("ia->ia", tau179, optimize=True)

    tau179 = None

    tau195 = zeros((N, N, N, N))

    tau195 += np.einsum("ijab,kalb->ijkl", tau12, u[o, v, o, v], optimize=True)

    tau201 = zeros((N, N, N, N))

    tau201 += 8 * np.einsum("iljk->ijkl", tau195, optimize=True)

    tau195 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("jiab->ijab", tau13, optimize=True)

    tau41 = zeros((N, N, M, M))

    tau41 += np.einsum("ijab->ijab", tau13, optimize=True)

    tau95 = zeros((N, N, M, M))

    tau95 += np.einsum("ijab->ijab", tau13, optimize=True)

    tau100 = zeros((N, N, M, M))

    tau100 += np.einsum("jiab->ijab", tau13, optimize=True)

    tau125 = zeros((N, N, M, M))

    tau125 += np.einsum("ijab->ijab", tau13, optimize=True)

    tau177 = zeros((N, N, M, M))

    tau177 -= np.einsum("jiab->ijab", tau13, optimize=True)

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("caki,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau15 += np.einsum("jiab->ijab", tau14, optimize=True)

    tau95 += np.einsum("ijab->ijab", tau14, optimize=True)

    tau187 = zeros((N, N, N, N))

    tau187 += np.einsum("ijab,klab->ijkl", tau11, tau14, optimize=True)

    tau14 = None

    tau201 -= 8 * np.einsum("ilkj->ijkl", tau187, optimize=True)

    tau187 = None

    tau15 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau16 = zeros((M, M, M, M))

    tau16 += np.einsum("ijab,jicd->abcd", tau12, tau15, optimize=True)

    tau15 = None

    tau35 -= 4 * np.einsum("acbd->abcd", tau16, optimize=True)

    tau16 = None

    tau17 = zeros((N, M, M, M))

    tau17 += np.einsum("di,abcd->iabc", t1, u[v, v, v, v], optimize=True)

    tau26 = zeros((N, M, M, M))

    tau26 -= np.einsum("ibac->iabc", tau17, optimize=True)

    tau104 = zeros((N, M, M, M))

    tau104 += np.einsum("ibac->iabc", tau17, optimize=True)

    tau17 = None

    tau18 = zeros((N, M, M, M))

    tau18 += np.einsum("daji,jbcd->iabc", t2, u[o, v, v, v], optimize=True)

    tau26 += 2 * np.einsum("iabc->iabc", tau18, optimize=True)

    tau18 = None

    tau19 = zeros((N, M))

    tau19 -= np.einsum("bj,ijba->ia", t1, u[o, o, v, v], optimize=True)

    tau20 = zeros((N, M))

    tau20 += np.einsum("ia->ia", tau19, optimize=True)

    tau19 = None

    tau20 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau21 = zeros((N, M, M, M))

    tau21 += np.einsum("ja,bcji->iabc", tau20, t2, optimize=True)

    tau26 += np.einsum("icba->iabc", tau21, optimize=True)

    tau21 = None

    tau59 = zeros((N, N, M, M))

    tau59 += 8 * np.einsum("aj,ib->ijab", l1, tau20, optimize=True)

    tau90 = zeros((N, N, M, M))

    tau90 += 4 * np.einsum("ai,jb->ijab", l1, tau20, optimize=True)

    tau114 = zeros((N, N, M, M))

    tau114 += 4 * np.einsum("ai,jb->ijab", l1, tau20, optimize=True)

    tau167 = zeros((N, N, N, M))

    tau167 += np.einsum("ib,bajk->ijka", tau20, t2, optimize=True)

    tau170 = zeros((N, N, N, M))

    tau170 += 2 * np.einsum("ikja->ijka", tau167, optimize=True)

    tau167 = None

    tau188 = zeros((N, N, M, M))

    tau188 += 4 * np.einsum("ai,jb->ijab", l1, tau20, optimize=True)

    tau206 = zeros((N, N, M, M))

    tau206 += 8 * np.einsum("bi,ja->ijab", l1, tau20, optimize=True)

    tau20 = None

    tau22 = zeros((N, N, N, M))

    tau22 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau23 = zeros((N, N, N, M))

    tau23 -= np.einsum("ikja->ijka", tau22, optimize=True)

    tau28 = zeros((N, N, N, M))

    tau28 += np.einsum("kjia->ijka", tau22, optimize=True)

    tau23 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("bk,ikja->ijab", t1, tau23, optimize=True)

    tau25 = zeros((N, M, M, M))

    tau25 -= np.einsum("cj,ijba->iabc", t1, tau24, optimize=True)

    tau24 = None

    tau26 += np.einsum("ibca->iabc", tau25, optimize=True)

    tau25 = None

    tau80 = zeros((N, N, N, M))

    tau80 += np.einsum("lkab,lijb->ijka", tau11, tau23, optimize=True)

    tau82 = zeros((N, N, N, M))

    tau82 += 2 * np.einsum("kjia->ijka", tau80, optimize=True)

    tau80 = None

    tau81 = zeros((N, N, N, M))

    tau81 += np.einsum("ab,ijkb->ijka", tau2, tau23, optimize=True)

    tau82 -= np.einsum("ikja->ijka", tau81, optimize=True)

    tau81 = None

    tau164 = zeros((N, N, N, M))

    tau164 += np.einsum("balk,iljb->ijka", t2, tau23, optimize=True)

    tau23 = None

    tau170 += 4 * np.einsum("kija->ijka", tau164, optimize=True)

    tau164 = None

    tau26 += np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau27 = zeros((M, M, M, M))

    tau27 += np.einsum("di,iabc->abcd", l1, tau26, optimize=True)

    tau26 = None

    tau35 += 2 * np.einsum("cbda->abcd", tau27, optimize=True)

    tau27 = None

    tau28 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau29 = zeros((N, M, M, M))

    tau29 += np.einsum("kjbc,jika->iabc", tau11, tau28, optimize=True)

    tau33 = zeros((N, M, M, M))

    tau33 -= 4 * np.einsum("icab->iabc", tau29, optimize=True)

    tau29 = None

    tau30 = zeros((N, M, M, M))

    tau30 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau31 = zeros((N, M, M, M))

    tau31 += np.einsum("iacb->iabc", tau30, optimize=True)

    tau165 = zeros((N, M, M, M))

    tau165 += np.einsum("iacb->iabc", tau30, optimize=True)

    tau30 = None

    tau31 -= 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau32 = zeros((N, M, M, M))

    tau32 += np.einsum("cd,iadb->iabc", tau2, tau31, optimize=True)

    tau2 = None

    tau33 -= np.einsum("ibca->iabc", tau32, optimize=True)

    tau32 = None

    tau34 = zeros((M, M, M, M))

    tau34 += np.einsum("di,iabc->abcd", t1, tau33, optimize=True)

    tau33 = None

    tau35 -= np.einsum("abdc->abcd", tau34, optimize=True)

    tau34 = None

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum("cdij,acdb->ijab", l2, tau35, optimize=True)

    tau35 = None

    tau84 = zeros((N, N, M, M))

    tau84 -= 2 * np.einsum("jiab->ijab", tau36, optimize=True)

    tau36 = None

    tau52 = zeros((M, M, M, M))

    tau52 += np.einsum("di,iabc->abcd", t1, tau31, optimize=True)

    tau31 = None

    tau53 = zeros((M, M, M, M))

    tau53 -= np.einsum("adcb->abcd", tau52, optimize=True)

    tau52 = None

    tau37 = zeros((N, N, M, M))

    tau37 += np.einsum("ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True)

    tau46 = zeros((N, N, M, M))

    tau46 += 2 * np.einsum("jiab->ijab", tau37, optimize=True)

    tau59 -= 4 * np.einsum("jiab->ijab", tau37, optimize=True)

    tau114 += 4 * np.einsum("jiab->ijab", tau37, optimize=True)

    tau206 -= 8 * np.einsum("jiab->ijab", tau37, optimize=True)

    tau37 = None

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau41 += np.einsum("jiab->ijab", tau38, optimize=True)

    tau100 += np.einsum("ijab->ijab", tau38, optimize=True)

    tau38 = None

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("baji->ijab", t2, optimize=True)

    tau39 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum("kica,kjcb->ijab", tau39, u[o, o, v, v], optimize=True)

    tau39 = None

    tau41 += np.einsum("ijab->ijab", tau40, optimize=True)

    tau100 += np.einsum("jiab->ijab", tau40, optimize=True)

    tau40 = None

    tau41 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum("cbkj,kica->ijab", l2, tau41, optimize=True)

    tau46 += 4 * np.einsum("jiba->ijab", tau42, optimize=True)

    tau59 += 8 * np.einsum("ijba->ijab", tau42, optimize=True)

    tau90 += 4 * np.einsum("jiba->ijab", tau42, optimize=True)

    tau114 += 4 * np.einsum("jiba->ijab", tau42, optimize=True)

    tau188 += 4 * np.einsum("jiba->ijab", tau42, optimize=True)

    tau206 += 8 * np.einsum("jiab->ijab", tau42, optimize=True)

    tau42 = None

    tau116 = zeros((N, N, M, M))

    tau116 += np.einsum("kjbc,kiac->ijab", tau11, tau41, optimize=True)

    tau41 = None

    tau131 = zeros((N, N, M, M))

    tau131 += 4 * np.einsum("ijba->ijab", tau116, optimize=True)

    tau116 = None

    tau43 = zeros((M, M))

    tau43 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau44 += 2 * np.einsum("ab->ab", tau43, optimize=True)

    tau45 = zeros((N, N, M, M))

    tau45 += np.einsum("ac,ijcb->ijab", tau44, u[o, o, v, v], optimize=True)

    tau46 += np.einsum("jiab->ijab", tau45, optimize=True)

    tau59 -= 2 * np.einsum("jiab->ijab", tau45, optimize=True)

    tau114 += 2 * np.einsum("jiab->ijab", tau45, optimize=True)

    tau206 -= 4 * np.einsum("jiab->ijab", tau45, optimize=True)

    tau45 = None

    tau61 = zeros((M, M))

    tau61 += np.einsum("cd,cadb->ab", tau44, u[v, v, v, v], optimize=True)

    tau75 = zeros((M, M))

    tau75 -= 4 * np.einsum("ab->ab", tau61, optimize=True)

    tau61 = None

    tau70 = zeros((N, M))

    tau70 += np.einsum("bc,ibca->ia", tau44, u[o, v, v, v], optimize=True)

    tau73 = zeros((N, M))

    tau73 -= np.einsum("ia->ia", tau70, optimize=True)

    tau182 += 2 * np.einsum("ia->ia", tau70, optimize=True)

    tau70 = None

    tau120 += np.einsum("bi,ac->iabc", t1, tau44, optimize=True)

    tau141 = zeros((N, N, M, M))

    tau141 += np.einsum("ca,cbij->ijab", tau44, t2, optimize=True)

    tau143 -= 2 * np.einsum("jiab->ijab", tau141, optimize=True)

    tau218 += np.einsum("jiab->ijab", tau141, optimize=True)

    tau141 = None

    tau219 = zeros((N, N, N, N))

    tau219 -= np.einsum("ijab,lkba->ijkl", tau218, u[o, o, v, v], optimize=True)

    tau218 = None

    tau153 += np.einsum("ai,bc->iabc", l1, tau44, optimize=True)

    tau208 = zeros((N, N))

    tau208 += np.einsum("ab,iajb->ij", tau44, u[o, v, o, v], optimize=True)

    tau212 = zeros((N, N))

    tau212 += 4 * np.einsum("ij->ij", tau208, optimize=True)

    tau208 = None

    tau215 += 4 * np.einsum("ac,bd->abcd", tau43, tau43, optimize=True)

    tau43 = None

    tau47 = zeros((N, N, N, N))

    tau47 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau48 = zeros((N, N, M, M))

    tau48 += np.einsum("klab,ijlk->ijab", tau46, tau47, optimize=True)

    tau46 = None

    tau84 -= np.einsum("jiab->ijab", tau48, optimize=True)

    tau48 = None

    tau51 = zeros((N, N, M, M))

    tau51 -= np.einsum("jilk,lkab->ijab", tau47, u[o, o, v, v], optimize=True)

    tau59 += np.einsum("ijba->ijab", tau51, optimize=True)

    tau90 -= np.einsum("ijba->ijab", tau51, optimize=True)

    tau51 = None

    tau138 = zeros((N, N, M, M))

    tau138 += np.einsum("klab,iljk->ijab", tau11, tau47, optimize=True)

    tau148 -= 2 * np.einsum("ijab->ijab", tau138, optimize=True)

    tau138 = None

    tau139 = zeros((N, N, M, M))

    tau139 -= np.einsum("ablk,lkji->ijab", t2, tau47, optimize=True)

    tau143 += np.einsum("ijba->ijab", tau139, optimize=True)

    tau139 = None

    tau145 = zeros((N, N, N, M))

    tau145 -= np.einsum("al,ilkj->ijka", t1, tau47, optimize=True)

    tau146 = zeros((N, N, N, M))

    tau146 -= np.einsum("ikja->ijka", tau145, optimize=True)

    tau175 = zeros((N, N, N, M))

    tau175 -= np.einsum("ikja->ijka", tau145, optimize=True)

    tau193 = zeros((N, N, N, M))

    tau193 -= np.einsum("ikja->ijka", tau145, optimize=True)

    tau145 = None

    tau155 = zeros((N, N, N, M))

    tau155 += np.einsum("al,ijkl->ijka", l1, tau47, optimize=True)

    tau157 = zeros((N, N, N, M))

    tau157 += np.einsum("ijka->ijka", tau155, optimize=True)

    tau220 = zeros((N, N, N, M))

    tau220 -= np.einsum("ijka->ijka", tau155, optimize=True)

    tau155 = None

    tau160 = zeros((N, M))

    tau160 += np.einsum("ijlk,lkja->ia", tau47, u[o, o, o, v], optimize=True)

    tau182 += np.einsum("ia->ia", tau160, optimize=True)

    tau160 = None

    tau217 -= np.einsum("inkm,jmln->ijkl", tau47, tau47, optimize=True)

    tau49 = zeros((M, M))

    tau49 -= np.einsum("ci,caib->ab", l1, u[v, v, o, v], optimize=True)

    tau75 += 8 * np.einsum("ab->ab", tau49, optimize=True)

    tau49 = None

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau59 -= 8 * np.einsum("ijba->ijab", tau50, optimize=True)

    tau90 += 4 * np.einsum("ijba->ijab", tau50, optimize=True)

    tau188 += 2 * np.einsum("ijba->ijab", tau50, optimize=True)

    tau206 -= 4 * np.einsum("ijba->ijab", tau50, optimize=True)

    tau50 = None

    tau53 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("cdij,cdab->ijab", l2, tau53, optimize=True)

    tau53 = None

    tau59 -= 2 * np.einsum("jiba->ijab", tau54, optimize=True)

    tau90 += 2 * np.einsum("jiba->ijab", tau54, optimize=True)

    tau54 = None

    tau55 = zeros((N, N))

    tau55 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau57 = zeros((N, N))

    tau57 += 2 * np.einsum("ij->ij", tau55, optimize=True)

    tau98 = zeros((N, N, N, M))

    tau98 += 2 * np.einsum("ak,ij->ijka", t1, tau55, optimize=True)

    tau217 += 4 * np.einsum("ik,jl->ijkl", tau55, tau55, optimize=True)

    tau220 -= 2 * np.einsum("ai,jk->ijka", l1, tau55, optimize=True)

    tau220 += 2 * np.einsum("aj,ik->ijka", l1, tau55, optimize=True)

    tau222 = zeros((N, M))

    tau222 += np.einsum("ja,ij->ia", f[o, v], tau55, optimize=True)

    tau228 = zeros((N, M))

    tau228 += np.einsum("ia->ia", tau222, optimize=True)

    tau222 = None

    tau223 = zeros((N, M))

    tau223 += np.einsum("jk,ikja->ia", tau55, u[o, o, o, v], optimize=True)

    tau55 = None

    tau228 -= np.einsum("ia->ia", tau223, optimize=True)

    tau223 = None

    tau56 = zeros((N, N))

    tau56 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau57 += np.einsum("ij->ij", tau56, optimize=True)

    tau58 = zeros((N, N, M, M))

    tau58 += np.einsum("ik,kjab->ijab", tau57, u[o, o, v, v], optimize=True)

    tau59 -= 4 * np.einsum("ijba->ijab", tau58, optimize=True)

    tau60 = zeros((M, M))

    tau60 += np.einsum("cbji,ijca->ab", t2, tau59, optimize=True)

    tau59 = None

    tau75 += np.einsum("ba->ab", tau60, optimize=True)

    tau60 = None

    tau90 += 2 * np.einsum("ijba->ijab", tau58, optimize=True)

    tau91 = zeros((N, N, M, M))

    tau91 += np.einsum("cbkj,ikca->ijab", t2, tau90, optimize=True)

    tau90 = None

    tau107 = zeros((N, N, M, M))

    tau107 -= np.einsum("ijab->ijab", tau91, optimize=True)

    tau91 = None

    tau188 += np.einsum("ijba->ijab", tau58, optimize=True)

    tau189 = zeros((N, N, N, N))

    tau189 += np.einsum("bakl,ijab->ijkl", t2, tau188, optimize=True)

    tau188 = None

    tau201 -= np.einsum("ijlk->ijkl", tau189, optimize=True)

    tau189 = None

    tau206 -= 2 * np.einsum("ijba->ijab", tau58, optimize=True)

    tau58 = None

    tau65 = zeros((N, M))

    tau65 += np.einsum("aj,ji->ia", t1, tau57, optimize=True)

    tau66 = zeros((N, M))

    tau66 += np.einsum("ia->ia", tau65, optimize=True)

    tau65 = None

    tau68 = zeros((M, M))

    tau68 += np.einsum("ij,jaib->ab", tau57, u[o, v, o, v], optimize=True)

    tau75 += 4 * np.einsum("ab->ab", tau68, optimize=True)

    tau68 = None

    tau72 = zeros((N, M))

    tau72 += np.einsum("jk,kija->ia", tau57, u[o, o, o, v], optimize=True)

    tau73 -= np.einsum("ia->ia", tau72, optimize=True)

    tau72 = None

    tau102 = zeros((N, N, N, M))

    tau102 += np.einsum("aj,ik->ijka", t1, tau57, optimize=True)

    tau106 = zeros((N, N, M, M))

    tau106 += np.einsum("ik,kajb->ijab", tau57, u[o, v, o, v], optimize=True)

    tau107 += 2 * np.einsum("ijba->ijab", tau106, optimize=True)

    tau106 = None

    tau142 = zeros((N, N, M, M))

    tau142 += np.einsum("ki,abkj->ijab", tau57, t2, optimize=True)

    tau143 -= 2 * np.einsum("ijba->ijab", tau142, optimize=True)

    tau216 -= np.einsum("jiba->ijab", tau142, optimize=True)

    tau142 = None

    tau148 += np.einsum("ab,ij->ijab", tau44, tau57, optimize=True)

    tau157 += np.einsum("ai,jk->ijka", l1, tau57, optimize=True)

    tau175 += 2 * np.einsum("aj,ik->ijka", t1, tau57, optimize=True)

    tau193 += np.einsum("aj,ik->ijka", t1, tau57, optimize=True)

    tau198 = zeros((N, N, N, M))

    tau198 += np.einsum("aj,ik->ijka", t1, tau57, optimize=True)

    tau200 = zeros((N, N, N, N))

    tau200 += np.einsum("im,mjkl->ijkl", tau57, u[o, o, o, o], optimize=True)

    tau201 -= 2 * np.einsum("ijlk->ijkl", tau200, optimize=True)

    tau200 = None

    tau210 = zeros((N, N))

    tau210 += np.einsum("kl,likj->ij", tau57, u[o, o, o, o], optimize=True)

    tau57 = None

    tau212 -= 4 * np.einsum("ij->ij", tau210, optimize=True)

    tau210 = None

    tau98 -= np.einsum("aj,ik->ijka", t1, tau56, optimize=True)

    tau159 = zeros((N, M))

    tau159 -= np.einsum("ja,ij->ia", f[o, v], tau56, optimize=True)

    tau182 -= 2 * np.einsum("ia->ia", tau159, optimize=True)

    tau159 = None

    tau161 = zeros((N, M))

    tau161 += np.einsum("jk,ikja->ia", tau56, u[o, o, o, v], optimize=True)

    tau182 -= 2 * np.einsum("ia->ia", tau161, optimize=True)

    tau161 = None

    tau203 = zeros((N, N, N, M))

    tau203 -= np.einsum("ai,jk->ijka", l1, tau56, optimize=True)

    tau217 += np.einsum("ik,jl->ijkl", tau56, tau56, optimize=True)

    tau56 = None

    tau62 = zeros((N, M))

    tau62 += np.einsum("bj,abij->ia", l1, t2, optimize=True)

    tau66 -= 2 * np.einsum("ia->ia", tau62, optimize=True)

    tau79 = zeros((N, N, N, M))

    tau79 -= np.einsum("ib,jkab->ijka", tau62, u[o, o, v, v], optimize=True)

    tau82 -= 2 * np.einsum("ikja->ijka", tau79, optimize=True)

    tau79 = None

    tau143 += 4 * np.einsum("bi,ja->ijab", t1, tau62, optimize=True)

    tau143 += 4 * np.einsum("aj,ib->ijab", t1, tau62, optimize=True)

    tau144 = zeros((N, N, M, M))

    tau144 += np.einsum("cbkj,ikac->ijab", l2, tau143, optimize=True)

    tau143 = None

    tau148 -= np.einsum("jiba->ijab", tau144, optimize=True)

    tau144 = None

    tau151 = zeros((N, M, M, M))

    tau151 -= np.einsum("jc,abij->iabc", tau62, l2, optimize=True)

    tau153 += 2 * np.einsum("ibac->iabc", tau151, optimize=True)

    tau215 += 4 * np.einsum("di,iabc->abcd", t1, tau151, optimize=True)

    tau151 = None

    tau156 = zeros((N, N, N, M))

    tau156 -= np.einsum("kb,abij->ijka", tau62, l2, optimize=True)

    tau157 -= 2 * np.einsum("ijka->ijka", tau156, optimize=True)

    tau220 += 2 * np.einsum("ijka->ijka", tau156, optimize=True)

    tau156 = None

    r2 += np.einsum("ijkc,kcba->abij", tau220, u[o, v, v, v], optimize=True) / 2

    tau220 = None

    tau184 = zeros((N, N))

    tau184 += np.einsum("ai,ja->ij", l1, tau62, optimize=True)

    tau185 = zeros((N, N, M, M))

    tau185 -= np.einsum("ik,jkab->ijab", tau184, u[o, o, v, v], optimize=True)

    tau184 = None

    tau214 = zeros((N, N, M, M))

    tau214 -= 8 * np.einsum("ijba->ijab", tau185, optimize=True)

    tau185 = None

    tau216 += 4 * np.einsum("bj,ia->ijab", t1, tau62, optimize=True)

    tau217 -= np.einsum("abji,klab->ijkl", l2, tau216, optimize=True)

    tau216 = None

    tau219 += 4 * np.einsum("ja,lkia->ijkl", tau62, u[o, o, o, v], optimize=True)

    tau62 = None

    tau63 = zeros((N, N, N, M))

    tau63 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau64 = zeros((N, M))

    tau64 += np.einsum("bakj,kjib->ia", t2, tau63, optimize=True)

    tau66 += np.einsum("ia->ia", tau64, optimize=True)

    tau64 = None

    tau67 = zeros((M, M))

    tau67 += np.einsum("ic,iacb->ab", tau66, u[o, v, v, v], optimize=True)

    tau75 += 4 * np.einsum("ab->ab", tau67, optimize=True)

    tau67 = None

    tau71 = zeros((N, M))

    tau71 += np.einsum("jb,jiba->ia", tau66, u[o, o, v, v], optimize=True)

    tau73 -= np.einsum("ia->ia", tau71, optimize=True)

    tau182 += 2 * np.einsum("ia->ia", tau71, optimize=True)

    tau71 = None

    tau209 = zeros((N, N))

    tau209 += np.einsum("ka,kija->ij", tau66, u[o, o, o, v], optimize=True)

    tau66 = None

    tau212 += 4 * np.einsum("ij->ij", tau209, optimize=True)

    tau209 = None

    tau87 = zeros((N, N, N, M))

    tau87 += np.einsum("balk,lijb->ijka", t2, tau63, optimize=True)

    tau88 = zeros((N, N, N, M))

    tau88 -= np.einsum("iljb,klab->ijka", tau87, u[o, o, v, v], optimize=True)

    tau89 = zeros((N, N, M, M))

    tau89 += np.einsum("ak,ijkb->ijab", t1, tau88, optimize=True)

    tau88 = None

    tau107 += 4 * np.einsum("ijba->ijab", tau89, optimize=True)

    tau89 = None

    tau98 += 2 * np.einsum("ijka->ijka", tau87, optimize=True)

    tau146 += 2 * np.einsum("ikja->ijka", tau87, optimize=True)

    tau193 += 2 * np.einsum("ikja->ijka", tau87, optimize=True)

    tau194 = zeros((N, N, N, N))

    tau194 += np.einsum("ijma,mkla->ijkl", tau193, u[o, o, o, v], optimize=True)

    tau193 = None

    tau201 += 4 * np.einsum("iljk->ijkl", tau194, optimize=True)

    tau194 = None

    tau198 += 4 * np.einsum("ikja->ijka", tau87, optimize=True)

    tau87 = None

    tau199 = zeros((N, N, N, N))

    tau199 += np.einsum("ijma,kmla->ijkl", tau198, tau22, optimize=True)

    tau198 = None

    tau22 = None

    tau201 += 2 * np.einsum("iklj->ijkl", tau199, optimize=True)

    tau199 = None

    tau118 = zeros((N, M, M, M))

    tau118 += np.einsum("bckj,kjia->iabc", t2, tau63, optimize=True)

    tau120 -= np.einsum("iacb->iabc", tau118, optimize=True)

    tau118 = None

    tau162 = zeros((N, N, N, N))

    tau162 += np.einsum("ak,ijla->ijkl", t1, tau63, optimize=True)

    tau163 = zeros((N, M))

    tau163 -= np.einsum("iljk,kjla->ia", tau162, u[o, o, o, v], optimize=True)

    tau162 = None

    tau182 += 2 * np.einsum("ia->ia", tau163, optimize=True)

    tau163 = None

    tau69 = zeros((N, M))

    tau69 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau73 -= 2 * np.einsum("ia->ia", tau69, optimize=True)

    tau74 = zeros((M, M))

    tau74 += np.einsum("bi,ia->ab", t1, tau73, optimize=True)

    tau75 += 4 * np.einsum("ba->ab", tau74, optimize=True)

    tau74 = None

    tau76 = zeros((N, N, M, M))

    tau76 += np.einsum("ca,cbij->ijab", tau75, l2, optimize=True)

    tau75 = None

    tau84 += np.einsum("jiba->ijab", tau76, optimize=True)

    tau76 = None

    tau211 = zeros((N, N))

    tau211 += np.einsum("aj,ia->ij", t1, tau73, optimize=True)

    tau73 = None

    tau212 += 4 * np.einsum("ij->ij", tau211, optimize=True)

    tau211 = None

    tau182 += 4 * np.einsum("ia->ia", tau69, optimize=True)

    tau69 = None

    tau77 = zeros((N, N, N, N))

    tau77 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau78 = zeros((N, N, N, M))

    tau78 += np.einsum("al,ijkl->ijka", l1, tau77, optimize=True)

    tau82 -= 2 * np.einsum("ikja->ijka", tau78, optimize=True)

    tau78 = None

    tau83 = zeros((N, N, M, M))

    tau83 += np.einsum("bk,kija->ijab", l1, tau82, optimize=True)

    tau82 = None

    tau84 += 4 * np.einsum("ijba->ijab", tau83, optimize=True)

    tau83 = None

    r2 += np.einsum("jiab->abij", tau84, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau84, optimize=True) / 8

    tau84 = None

    tau112 = zeros((N, N, N, N))

    tau112 -= 4 * np.einsum("ljik->ijkl", tau77, optimize=True)

    tau123 = zeros((N, N, N, N))

    tau123 -= 2 * np.einsum("ljik->ijkl", tau77, optimize=True)

    tau168 = zeros((N, N, N, N))

    tau168 -= 2 * np.einsum("kjil->ijkl", tau77, optimize=True)

    tau191 = zeros((N, N, N, N))

    tau191 -= 2 * np.einsum("ljik->ijkl", tau77, optimize=True)

    tau77 = None

    tau85 = zeros((N, N, N, M))

    tau85 += np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    tau86 = zeros((N, N, M, M))

    tau86 += np.einsum("ak,ijkb->ijab", l1, tau85, optimize=True)

    tau85 = None

    tau183 = zeros((N, N, M, M))

    tau183 -= 4 * np.einsum("ijab->ijab", tau86, optimize=True)

    tau86 = None

    tau92 = zeros((M, M, M, M))

    tau92 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau93 += np.einsum("badc->abcd", tau92, optimize=True)

    tau92 = None

    tau93 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau94 = zeros((N, N, M, M))

    tau94 += np.einsum("ijcd,cadb->ijab", tau11, tau93, optimize=True)

    tau11 = None

    tau93 = None

    tau107 -= 2 * np.einsum("ijba->ijab", tau94, optimize=True)

    tau94 = None

    tau95 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau96 = zeros((N, N, M, M))

    tau96 += np.einsum("ikca,jkcb->ijab", tau12, tau95, optimize=True)

    tau107 += 4 * np.einsum("ijba->ijab", tau96, optimize=True)

    tau96 = None

    tau109 = zeros((N, N, M, M))

    tau109 += np.einsum("cadb,ijcd->ijab", tau4, tau95, optimize=True)

    tau4 = None

    tau95 = None

    tau131 -= 2 * np.einsum("jiab->ijab", tau109, optimize=True)

    tau109 = None

    tau97 = zeros((N, N, N, M))

    tau97 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau98 += np.einsum("ikja->ijka", tau97, optimize=True)

    tau99 = zeros((N, N, M, M))

    tau99 += np.einsum("klia,jlkb->ijab", tau28, tau98, optimize=True)

    tau98 = None

    tau107 += 2 * np.einsum("jiab->ijab", tau99, optimize=True)

    tau99 = None

    tau102 -= 2 * np.einsum("ikja->ijka", tau97, optimize=True)

    tau103 = zeros((N, N, M, M))

    tau103 += np.einsum("ijkc,kacb->ijab", tau102, u[o, v, v, v], optimize=True)

    tau102 = None

    tau107 += 2 * np.einsum("ijba->ijab", tau103, optimize=True)

    tau103 = None

    tau133 = zeros((N, N, N, M))

    tau133 += np.einsum("abjl,ilkb->ijka", l2, tau97, optimize=True)

    tau134 = zeros((N, N, M, M))

    tau134 += np.einsum("bk,ikja->ijab", t1, tau133, optimize=True)

    tau148 += 4 * np.einsum("ijab->ijab", tau134, optimize=True)

    tau134 = None

    tau157 -= 2 * np.einsum("ijka->ijka", tau133, optimize=True)

    tau157 += 2 * np.einsum("jika->ijka", tau133, optimize=True)

    tau158 = zeros((N, N, M, M))

    tau158 += np.einsum("kila,ljkb->ijab", tau157, u[o, o, o, v], optimize=True)

    tau157 = None

    tau183 -= 2 * np.einsum("ijab->ijab", tau158, optimize=True)

    tau158 = None

    tau203 += 2 * np.einsum("ijka->ijka", tau133, optimize=True)

    tau133 = None

    tau135 = zeros((N, N, M, M))

    tau135 += np.einsum("lkja,ilkb->ijab", tau63, tau97, optimize=True)

    tau148 -= 2 * np.einsum("ijab->ijab", tau135, optimize=True)

    tau135 = None

    tau146 -= 2 * np.einsum("ikja->ijka", tau97, optimize=True)

    tau147 = zeros((N, N, M, M))

    tau147 += np.einsum("bk,ikja->ijab", l1, tau146, optimize=True)

    tau146 = None

    tau148 += 2 * np.einsum("ijba->ijab", tau147, optimize=True)

    tau147 = None

    tau149 = zeros((N, N, M, M))

    tau149 += np.einsum("ikac,kjcb->ijab", tau148, u[o, o, v, v], optimize=True)

    tau148 = None

    tau183 -= np.einsum("ijab->ijab", tau149, optimize=True)

    tau149 = None

    tau150 = zeros((N, M, M, M))

    tau150 += np.einsum("abkj,ikjc->iabc", l2, tau97, optimize=True)

    tau153 -= np.einsum("ibac->iabc", tau150, optimize=True)

    tau150 = None

    tau154 = zeros((N, N, M, M))

    tau154 += np.einsum("jcbd,icda->ijab", tau153, u[o, v, v, v], optimize=True)

    tau153 = None

    tau183 -= 2 * np.einsum("jiba->ijab", tau154, optimize=True)

    tau154 = None

    tau175 -= 2 * np.einsum("ikja->ijka", tau97, optimize=True)

    tau190 = zeros((N, N, N, N))

    tau190 += np.einsum("mija,kmla->ijkl", tau28, tau97, optimize=True)

    tau201 += 8 * np.einsum("jlik->ijkl", tau190, optimize=True)

    tau190 = None

    tau217 -= 2 * np.einsum("aj,ikla->ijkl", l1, tau97, optimize=True)

    r2 -= np.einsum("ijkl,klba->abij", tau217, u[o, o, v, v], optimize=True) / 4

    tau217 = None

    tau219 -= 2 * np.einsum("mjia,lkma->ijkl", tau97, u[o, o, o, v], optimize=True)

    tau97 = None

    r2 += np.einsum("bakl,klij->abij", l2, tau219, optimize=True) / 4

    tau219 = None

    tau100 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau101 = zeros((N, N, M, M))

    tau101 += np.einsum("klab,likj->ijab", tau100, tau47, optimize=True)

    tau100 = None

    tau107 -= 2 * np.einsum("ijba->ijab", tau101, optimize=True)

    tau101 = None

    tau104 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau105 = zeros((N, N, M, M))

    tau105 += np.einsum("cj,icab->ijab", l1, tau104, optimize=True)

    tau107 -= 4 * np.einsum("jiba->ijab", tau105, optimize=True)

    tau105 = None

    tau108 = zeros((N, N, M, M))

    tau108 += np.einsum("cbkj,ikac->ijab", l2, tau107, optimize=True)

    tau107 = None

    tau183 += np.einsum("ijba->ijab", tau108, optimize=True)

    tau108 = None

    tau172 = zeros((N, M))

    tau172 += np.einsum("bcji,jbca->ia", l2, tau104, optimize=True)

    tau104 = None

    tau182 -= 2 * np.einsum("ia->ia", tau172, optimize=True)

    tau172 = None

    tau110 = zeros((N, N, M, M))

    tau110 -= np.einsum("baji->ijab", t2, optimize=True)

    tau110 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau111 = zeros((N, N, N, N))

    tau111 += np.einsum("ijab,klab->ijkl", tau110, u[o, o, v, v], optimize=True)

    tau112 -= np.einsum("lkji->ijkl", tau111, optimize=True)

    tau191 -= np.einsum("lkji->ijkl", tau111, optimize=True)

    tau111 = None

    tau112 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau113 = zeros((N, N, M, M))

    tau113 += np.einsum("abkl,ijkl->ijab", l2, tau112, optimize=True)

    tau112 = None

    tau114 += np.einsum("jiba->ijab", tau113, optimize=True)

    tau115 = zeros((N, N, M, M))

    tau115 += np.einsum("cbkj,kiac->ijab", t2, tau114, optimize=True)

    tau114 = None

    tau131 -= np.einsum("ijab->ijab", tau115, optimize=True)

    tau115 = None

    tau206 -= np.einsum("jiba->ijab", tau113, optimize=True)

    tau113 = None

    tau207 = zeros((N, N))

    tau207 += np.einsum("bakj,kiab->ij", t2, tau206, optimize=True)

    tau206 = None

    tau212 += np.einsum("ij->ij", tau207, optimize=True)

    tau207 = None

    tau117 = zeros((N, M, M, M))

    tau117 += np.einsum("aj,bcij->iabc", l1, t2, optimize=True)

    tau120 -= 2 * np.einsum("iacb->iabc", tau117, optimize=True)

    tau121 = zeros((N, N, M, M))

    tau121 += np.einsum("kabc,kijc->ijab", tau120, tau28, optimize=True)

    tau120 = None

    tau28 = None

    tau131 += 2 * np.einsum("ijab->ijab", tau121, optimize=True)

    tau121 = None

    tau215 += 2 * np.einsum("ai,ibcd->abcd", l1, tau117, optimize=True)

    tau117 = None

    r2 += np.einsum("bacd,jicd->abij", tau215, u[o, o, v, v], optimize=True) / 4

    tau215 = None

    tau122 = zeros((N, N, N, N))

    tau122 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau123 += np.einsum("lkji->ijkl", tau122, optimize=True)

    tau122 = None

    tau123 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau124 = zeros((N, N, M, M))

    tau124 += np.einsum("klab,likj->ijab", tau12, tau123, optimize=True)

    tau123 = None

    tau12 = None

    tau131 -= 2 * np.einsum("ijab->ijab", tau124, optimize=True)

    tau124 = None

    tau125 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau126 = zeros((N, N, M, M))

    tau126 += np.einsum("ac,ijbc->ijab", tau44, tau125, optimize=True)

    tau44 = None

    tau125 = None

    tau131 -= 2 * np.einsum("jiab->ijab", tau126, optimize=True)

    tau126 = None

    tau127 = zeros((N, N, N, M))

    tau127 += np.einsum("bi,jakb->ijka", t1, u[o, v, o, v], optimize=True)

    tau129 = zeros((N, N, N, M))

    tau129 += 2 * np.einsum("ijka->ijka", tau127, optimize=True)

    tau170 -= 4 * np.einsum("kija->ijka", tau127, optimize=True)

    tau127 = None

    tau128 = zeros((N, N, N, M))

    tau128 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau129 += np.einsum("kija->ijka", tau128, optimize=True)

    tau128 = None

    tau129 += 2 * np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    tau130 = zeros((N, N, M, M))

    tau130 += np.einsum("bk,ijka->ijab", l1, tau129, optimize=True)

    tau129 = None

    tau131 -= 2 * np.einsum("jiba->ijab", tau130, optimize=True)

    tau130 = None

    tau132 = zeros((N, N, M, M))

    tau132 += np.einsum("cbkj,ikac->ijab", l2, tau131, optimize=True)

    tau131 = None

    tau183 += np.einsum("jiab->ijab", tau132, optimize=True)

    tau132 = None

    tau165 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau166 = zeros((N, N, N, M))

    tau166 += np.einsum("ijbc,kabc->ijka", tau110, tau165, optimize=True)

    tau110 = None

    tau170 -= np.einsum("kjia->ijka", tau166, optimize=True)

    tau166 = None

    tau204 = zeros((N, N, M, M))

    tau204 += np.einsum("kcab,ijkc->ijab", tau165, tau203, optimize=True)

    tau203 = None

    tau165 = None

    tau214 -= 4 * np.einsum("ijba->ijab", tau204, optimize=True)

    tau204 = None

    tau168 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau169 = zeros((N, N, N, M))

    tau169 += np.einsum("al,lijk->ijka", t1, tau168, optimize=True)

    tau168 = None

    tau170 -= 2 * np.einsum("ikja->ijka", tau169, optimize=True)

    tau169 = None

    tau170 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau171 = zeros((N, M))

    tau171 += np.einsum("bajk,ijkb->ia", l2, tau170, optimize=True)

    tau170 = None

    tau182 += np.einsum("ia->ia", tau171, optimize=True)

    tau171 = None

    tau173 = zeros((N, N, M, M))

    tau173 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau173 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau174 = zeros((N, N, N, M))

    tau174 += np.einsum("liba,ljkb->ijka", tau173, tau63, optimize=True)

    tau173 = None

    tau175 += 2 * np.einsum("jika->ijka", tau174, optimize=True)

    tau174 = None

    tau176 = zeros((N, M))

    tau176 += np.einsum("ijkb,jkba->ia", tau175, u[o, o, v, v], optimize=True)

    tau175 = None

    tau182 += np.einsum("ia->ia", tau176, optimize=True)

    tau176 = None

    tau177 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau178 = zeros((N, M))

    tau178 += np.einsum("jkba,kijb->ia", tau177, tau63, optimize=True)

    tau177 = None

    tau63 = None

    tau182 -= 4 * np.einsum("ia->ia", tau178, optimize=True)

    tau178 = None

    tau183 += np.einsum("ai,jb->ijab", l1, tau182, optimize=True)

    tau182 = None

    r2 -= np.einsum("ijab->abij", tau183, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau183, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau183, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau183, optimize=True) / 4

    tau183 = None

    tau186 = zeros((N, N, N, N))

    tau186 += np.einsum("ai,jakl->ijkl", l1, u[o, v, o, o], optimize=True)

    tau201 -= 4 * np.einsum("ijlk->ijkl", tau186, optimize=True)

    tau186 = None

    tau191 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau192 = zeros((N, N, N, N))

    tau192 += np.einsum("minj,nkml->ijkl", tau191, tau47, optimize=True)

    tau191 = None

    tau47 = None

    tau201 += 2 * np.einsum("jlik->ijkl", tau192, optimize=True)

    tau192 = None

    tau196 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau197 = zeros((N, N, N, N))

    tau197 += np.einsum("ijab,klab->ijkl", tau13, tau196, optimize=True)

    tau196 = None

    tau13 = None

    tau201 += 4 * np.einsum("ljik->ijkl", tau197, optimize=True)

    tau197 = None

    tau202 = zeros((N, N, M, M))

    tau202 += np.einsum("abkl,ijkl->ijab", l2, tau201, optimize=True)

    tau201 = None

    tau214 -= np.einsum("ijba->ijab", tau202, optimize=True)

    tau202 = None

    tau205 = zeros((N, N))

    tau205 -= np.einsum("ak,iakj->ij", l1, u[o, v, o, o], optimize=True)

    tau212 += 8 * np.einsum("ij->ij", tau205, optimize=True)

    tau205 = None

    tau213 = zeros((N, N, M, M))

    tau213 += np.einsum("ik,abkj->ijab", tau212, l2, optimize=True)

    tau212 = None

    tau214 += np.einsum("jiba->ijab", tau213, optimize=True)

    tau213 = None

    r2 += np.einsum("ijba->abij", tau214, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau214, optimize=True) / 8

    tau214 = None

    tau221 = zeros((N, M))

    tau221 += np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    tau228 -= np.einsum("ia->ia", tau221, optimize=True)

    tau221 = None

    tau224 = zeros((N, N))

    tau224 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau226 = zeros((N, N))

    tau226 += np.einsum("ij->ij", tau224, optimize=True)

    tau224 = None

    tau225 = zeros((N, N))

    tau225 += np.einsum("ak,ikja->ij", t1, u[o, o, o, v], optimize=True)

    tau226 += np.einsum("ij->ij", tau225, optimize=True)

    tau225 = None

    tau226 += np.einsum("ij->ij", f[o, o], optimize=True)

    tau227 = zeros((N, M))

    tau227 += np.einsum("aj,ij->ia", l1, tau226, optimize=True)

    tau226 = None

    tau228 += np.einsum("ia->ia", tau227, optimize=True)

    tau227 = None

    r2 += np.einsum("aj,ib->abij", l1, tau228, optimize=True)

    r2 += np.einsum("bi,ja->abij", l1, tau228, optimize=True)

    r2 -= np.einsum("bj,ia->abij", l1, tau228, optimize=True)

    r2 -= np.einsum("ai,jb->abij", l1, tau228, optimize=True)

    tau228 = None

    return r2
