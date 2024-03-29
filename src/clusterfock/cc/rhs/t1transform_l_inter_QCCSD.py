import numpy as np

from clusterfock.cc.rhs.t1transform_CCSD import t1_transform_lambda_intermediates_ccsd


def t1_transform_l_intermediates_qccsd(t2, l1, l2, u, f, v, o):
    l1, l2 = t1_transform_lambda_intermediates_ccsd(t2, l1, l2, u, f, v, o)

    l1 += t1_transform_l1_intermediates_qccsd_addition(t2, l1, l2, u, f, v, o)
    l2 += t1_transform_l2_intermediates_qccsd_addition(t2, l1, l2, u, f, v, o)

    l2 = 0.25 * (
        l2 - l2.transpose(1, 0, 2, 3) - l2.transpose(0, 1, 3, 2) + l2.transpose(1, 0, 3, 2)
    )

    return l1, l2


def t1_transform_l1_intermediates_qccsd_addition(t2, l1, l2, u, f, v, o):
    M, N = l1.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M))

    tau0 += np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau1 = zeros((M, M))

    tau1 += np.einsum("ai,ib->ab", l1, tau0, optimize=True)

    r1 = zeros((M, N))

    r1 += np.einsum("bc,ibac->ai", tau1, u[o, v, v, v], optimize=True)

    tau1 = None

    tau3 = zeros((N, N))

    tau3 += np.einsum("ai,ja->ij", l1, tau0, optimize=True)

    r1 += np.einsum("jk,ikja->ai", tau3, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("ja,ij->ai", f[o, v], tau3, optimize=True)

    tau3 = None

    tau14 = zeros((N, N, M, M))

    tau14 -= 4 * np.einsum("ai,jb->ijab", l1, tau0, optimize=True)

    tau15 = zeros((N, M, M, M))

    tau15 += 2 * np.einsum("jc,baij->iabc", tau0, l2, optimize=True)

    tau18 = zeros((N, M, M, M))

    tau18 += np.einsum("jb,ijca->iabc", tau0, u[o, o, v, v], optimize=True)

    tau21 = zeros((N, N, N, M))

    tau21 += 2 * np.einsum("ib,kjab->ijka", tau0, u[o, o, v, v], optimize=True)

    tau23 = zeros((N, M))

    tau23 -= 2 * np.einsum("jb,ijba->ia", tau0, u[o, o, v, v], optimize=True)

    tau24 = zeros((N, N, N, M))

    tau24 -= 2 * np.einsum("la,jlik->ijka", tau0, u[o, o, o, o], optimize=True)

    tau24 += 4 * np.einsum("kb,jaib->ijka", tau0, u[o, v, o, v], optimize=True)

    tau26 = zeros((N, N, N, M))

    tau26 -= np.einsum("kb,abij->ijka", tau0, l2, optimize=True)

    tau28 = zeros((N, N, N, M))

    tau28 -= 2 * np.einsum("ijka->ijka", tau26, optimize=True)

    tau31 = zeros((N, N, N, M))

    tau31 -= 2 * np.einsum("ijka->ijka", tau26, optimize=True)

    tau35 = zeros((N, N, N, M))

    tau35 -= 2 * np.einsum("ijka->ijka", tau26, optimize=True)

    tau26 = None

    tau38 = zeros((N, N))

    tau38 += 8 * np.einsum("ka,ikja->ij", tau0, u[o, o, o, v], optimize=True)

    tau2 = zeros((M, M))

    tau2 -= np.einsum("ci,caib->ab", l1, u[v, v, o, v], optimize=True)

    r1 -= np.einsum("bi,ba->ai", l1, tau2, optimize=True)

    tau2 = None

    tau4 = zeros((N, N, N, M))

    tau4 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau14 += 4 * np.einsum("ak,ijkb->ijab", l1, tau4, optimize=True)

    tau15 += np.einsum("bakj,ikjc->iabc", l2, tau4, optimize=True)

    r1 += np.einsum("ibcd,bcda->ai", tau15, u[v, v, v, v], optimize=True) / 4

    tau15 = None

    tau21 += 2 * np.einsum("klib,jlba->ijka", tau4, u[o, o, v, v], optimize=True)

    tau27 = zeros((N, N, N, M))

    tau27 += np.einsum("balj,ilkb->ijka", l2, tau4, optimize=True)

    tau28 -= 2 * np.einsum("ijka->ijka", tau27, optimize=True)

    tau28 += 2 * np.einsum("jika->ijka", tau27, optimize=True)

    tau30 = zeros((N, N, N, M))

    tau30 += 2 * np.einsum("jika->ijka", tau27, optimize=True)

    tau31 += 2 * np.einsum("jika->ijka", tau27, optimize=True)

    tau35 += 4 * np.einsum("jika->ijka", tau27, optimize=True)

    tau27 = None

    tau33 = zeros((N, N, N, N))

    tau33 += 2 * np.einsum("aj,ilka->ijkl", l1, tau4, optimize=True)

    tau5 = zeros((M, M, M, M))

    tau5 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau18 += np.einsum("daeb,idce->iabc", tau5, u[o, v, v, v], optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("caki,kjcb->ijab", t2, tau6, optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 += 4 * np.einsum("ijba->ijab", tau11, optimize=True)

    tau22 = zeros((N, N, M, M))

    tau22 += 4 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau24 -= 2 * np.einsum("ikbc,jacb->ijka", tau11, u[o, v, v, v], optimize=True)

    tau32 = zeros((N, N, M, M))

    tau32 += 2 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau11 = None

    tau14 += 2 * np.einsum("acbd,ijdc->ijab", tau5, tau6, optimize=True)

    tau5 = None

    tau14 -= 4 * np.einsum("ikcb,kjac->ijab", tau6, tau6, optimize=True)

    tau18 += 2 * np.einsum("jkab,ikjc->iabc", tau6, u[o, o, o, v], optimize=True)

    tau21 -= 2 * np.einsum("liab,kjlb->ijka", tau6, u[o, o, o, v], optimize=True)

    tau21 -= 2 * np.einsum("kibc,jbac->ijka", tau6, u[o, v, v, v], optimize=True)

    tau29 = zeros((N, N, N, M))

    tau29 += 4 * np.einsum("lkjb,ilba->ijka", tau4, tau6, optimize=True)

    tau33 -= 4 * np.einsum("ikab,jlba->ijkl", tau6, tau6, optimize=True)

    tau34 = zeros((N, M, M, M))

    tau34 += 2 * np.einsum("aj,ijbc->iabc", l1, tau6, optimize=True)

    tau7 = zeros((N, N, N, N))

    tau7 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 -= np.einsum("ablk,lkji->ijab", t2, tau7, optimize=True)

    tau13 += np.einsum("ijba->ijab", tau10, optimize=True)

    tau22 -= np.einsum("ijba->ijab", tau10, optimize=True)

    tau10 = None

    tau24 += np.einsum("klba,ljib->ijka", tau22, u[o, o, o, v], optimize=True)

    tau22 = None

    tau14 += 2 * np.einsum("klab,iljk->ijab", tau6, tau7, optimize=True)

    tau6 = None

    tau21 -= np.einsum("lkmi,jmla->ijka", tau7, u[o, o, o, v], optimize=True)

    tau25 = zeros((N, N, N, M))

    tau25 -= np.einsum("al,jikl->ijka", l1, tau7, optimize=True)

    tau28 += np.einsum("ijka->ijka", tau25, optimize=True)

    tau30 += np.einsum("ijka->ijka", tau25, optimize=True)

    tau35 += np.einsum("ijka->ijka", tau25, optimize=True)

    tau25 = None

    tau29 += np.einsum("imla,mlkj->ijka", tau4, tau7, optimize=True)

    tau29 += 2 * np.einsum("la,ilkj->ijka", tau0, tau7, optimize=True)

    tau33 -= np.einsum("imkn,jnlm->ijkl", tau7, tau7, optimize=True)

    tau7 = None

    tau8 = zeros((N, N))

    tau8 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("kj,abik->ijab", tau8, t2, optimize=True)

    tau13 -= 2 * np.einsum("ijba->ijab", tau12, optimize=True)

    tau32 += np.einsum("ijba->ijab", tau12, optimize=True)

    tau12 = None

    tau33 -= np.einsum("abji,klab->ijkl", l2, tau32, optimize=True)

    tau32 = None

    tau21 -= np.einsum("kl,jlia->ijka", tau8, u[o, o, o, v], optimize=True)

    tau23 += np.einsum("jk,ikja->ia", tau8, u[o, o, o, v], optimize=True)

    tau28 += np.einsum("ai,jk->ijka", l1, tau8, optimize=True)

    tau29 -= 4 * np.einsum("balk,iljb->ijka", t2, tau28, optimize=True)

    tau28 = None

    tau29 += 4 * np.einsum("ka,ij->ijka", tau0, tau8, optimize=True)

    tau0 = None

    tau30 += np.einsum("ai,jk->ijka", l1, tau8, optimize=True)

    r1 -= np.einsum("ikjb,jbka->ai", tau30, u[o, v, o, v], optimize=True) / 2

    tau30 = None

    tau31 += np.einsum("ai,jk->ijka", l1, tau8, optimize=True)

    r1 += np.einsum("kijb,jbka->ai", tau31, u[o, v, o, v], optimize=True) / 2

    tau31 = None

    tau33 += np.einsum("ik,jl->ijkl", tau8, tau8, optimize=True)

    r1 -= np.einsum("ijkl,klja->ai", tau33, u[o, o, o, v], optimize=True) / 4

    tau33 = None

    tau35 += 2 * np.einsum("ai,jk->ijka", l1, tau8, optimize=True)

    tau36 = zeros((N, M))

    tau36 += np.einsum("bajk,jkib->ia", t2, tau35, optimize=True)

    tau35 = None

    r1 -= np.einsum("jb,jiba->ai", tau36, u[o, o, v, v], optimize=True) / 4

    tau36 = None

    tau37 = zeros((N, N, M, M))

    tau37 -= 2 * np.einsum("ik,jkba->ijab", tau8, u[o, o, v, v], optimize=True)

    tau38 += 4 * np.einsum("kl,ilkj->ij", tau8, u[o, o, o, o], optimize=True)

    tau9 = zeros((M, M))

    tau9 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau13 -= 2 * np.einsum("cb,acji->ijab", tau9, t2, optimize=True)

    tau14 += np.einsum("caki,kjcb->ijab", l2, tau13, optimize=True)

    tau13 = None

    tau14 -= np.einsum("ij,ab->ijab", tau8, tau9, optimize=True)

    tau8 = None

    r1 -= np.einsum("ijbc,jbca->ai", tau14, u[o, v, v, v], optimize=True) / 4

    tau14 = None

    tau18 += np.einsum("ad,ibcd->iabc", tau9, u[o, v, v, v], optimize=True)

    tau21 += np.einsum("ab,kjib->ijka", tau9, u[o, o, o, v], optimize=True)

    tau23 += np.einsum("bc,ibac->ia", tau9, u[o, v, v, v], optimize=True)

    tau29 -= 2 * np.einsum("ba,ikjb->ijka", tau9, tau4, optimize=True)

    tau4 = None

    r1 -= np.einsum("ikjb,jkba->ai", tau29, u[o, o, v, v], optimize=True) / 8

    tau29 = None

    tau34 -= np.einsum("ai,bc->iabc", l1, tau9, optimize=True)

    r1 -= np.einsum("icbd,bcda->ai", tau34, u[v, v, v, v], optimize=True) / 2

    tau34 = None

    tau37 += 4 * np.einsum("ac,ijbc->ijab", tau9, u[o, o, v, v], optimize=True)

    tau38 += 4 * np.einsum("ab,iajb->ij", tau9, u[o, v, o, v], optimize=True)

    tau9 = None

    tau16 = zeros((N, N, M, M))

    tau16 -= np.einsum("caki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("ijab->ijab", tau16, optimize=True)

    tau16 = None

    tau17 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau18 += 2 * np.einsum("aj,jibc->iabc", l1, tau17, optimize=True)

    tau24 += np.einsum("cbki,jbac->ijka", t2, tau18, optimize=True)

    tau18 = None

    tau37 -= 8 * np.einsum("cbki,kjca->ijab", l2, tau17, optimize=True)

    tau17 = None

    tau19 = zeros((N, N, N, N))

    tau19 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau20 = zeros((N, N, N, N))

    tau20 += np.einsum("lkji->ijkl", tau19, optimize=True)

    tau19 = None

    tau20 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau21 -= np.einsum("al,kjli->ijka", l1, tau20, optimize=True)

    tau37 += np.einsum("bakl,jikl->ijab", l2, tau20, optimize=True)

    tau20 = None

    tau21 += 2 * np.einsum("bk,jbia->ijka", l1, u[o, v, o, v], optimize=True)

    tau24 -= 2 * np.einsum("bali,kjlb->ijka", t2, tau21, optimize=True)

    tau21 = None

    tau23 -= 2 * np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau24 -= np.einsum("jb,baki->ijka", tau23, t2, optimize=True)

    tau23 = None

    r1 += np.einsum("bajk,jikb->ai", l2, tau24, optimize=True) / 4

    tau24 = None

    tau37 -= 8 * np.einsum("bi,ja->ijab", l1, f[o, v], optimize=True)

    tau37 += 8 * np.einsum("ak,jikb->ijab", l1, u[o, o, o, v], optimize=True)

    tau37 += 4 * np.einsum("ci,jcba->ijab", l1, u[o, v, v, v], optimize=True)

    tau38 -= np.einsum("bakj,kiab->ij", t2, tau37, optimize=True)

    tau37 = None

    tau38 -= 8 * np.einsum("ak,iakj->ij", l1, u[o, v, o, o], optimize=True)

    r1 -= np.einsum("aj,ij->ai", l1, tau38, optimize=True) / 8

    tau38 = None

    return r1


def t1_transform_l2_intermediates_qccsd_addition(t2, l1, l2, u, f, v, o):
    M, N = l1.shape
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

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("ak,ijkb->ijab", l1, tau2, optimize=True)

    tau2 = None

    tau87 = zeros((N, N, M, M))

    tau87 += 4 * np.einsum("ijab->ijab", tau3, optimize=True)

    tau3 = None

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("ci,acjb->ijab", l1, u[v, v, o, v], optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += 4 * np.einsum("ijba->ijab", tau4, optimize=True)

    tau4 = None

    tau5 = zeros((N, N, N, M))

    tau5 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ilka,lkjb->ijab", tau5, u[o, o, o, v], optimize=True)

    tau26 += 2 * np.einsum("ijba->ijab", tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("ikjc,kabc->ijab", tau5, u[o, v, v, v], optimize=True)

    tau26 += 4 * np.einsum("ijba->ijab", tau7, optimize=True)

    tau7 = None

    tau47 = zeros((N, N, M, M))

    tau47 += np.einsum("ak,ijkb->ijab", l1, tau5, optimize=True)

    tau57 = zeros((N, N, M, M))

    tau57 += 4 * np.einsum("ijab->ijab", tau47, optimize=True)

    tau47 = None

    tau59 = zeros((N, M, M, M))

    tau59 -= np.einsum("bakj,ikjc->iabc", l2, tau5, optimize=True)

    tau63 = zeros((N, M, M, M))

    tau63 -= np.einsum("ibac->iabc", tau59, optimize=True)

    tau59 = None

    tau67 = zeros((N, N, N, M))

    tau67 -= np.einsum("bajl,ilkb->ijka", l2, tau5, optimize=True)

    tau68 = zeros((N, N, N, M))

    tau68 -= 2 * np.einsum("ijka->ijka", tau67, optimize=True)

    tau68 += 2 * np.einsum("jika->ijka", tau67, optimize=True)

    tau134 = zeros((N, N, N, M))

    tau134 += 2 * np.einsum("ijka->ijka", tau67, optimize=True)

    tau67 = None

    tau75 = zeros((N, M))

    tau75 -= np.einsum("ikjb,kjab->ia", tau5, u[o, o, v, v], optimize=True)

    tau86 = zeros((N, M))

    tau86 -= 2 * np.einsum("ia->ia", tau75, optimize=True)

    tau75 = None

    tau117 = zeros((N, N, N, N))

    tau117 += np.einsum("imja,kmla->ijkl", tau5, u[o, o, o, v], optimize=True)

    tau124 = zeros((N, N, N, N))

    tau124 -= 8 * np.einsum("ikjl->ijkl", tau117, optimize=True)

    tau117 = None

    tau140 = zeros((N, N, N, N))

    tau140 -= 2 * np.einsum("aj,ikla->ijkl", l1, tau5, optimize=True)

    tau142 = zeros((N, N, N, N))

    tau142 -= 2 * np.einsum("mjia,lkma->ijkl", tau5, u[o, o, o, v], optimize=True)

    tau5 = None

    tau8 = zeros((N, N))

    tau8 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 -= np.einsum("ik,kajb->ijab", tau8, u[o, v, o, v], optimize=True)

    tau26 -= 2 * np.einsum("ijba->ijab", tau9, optimize=True)

    tau9 = None

    tau14 = zeros((N, N, M, M))

    tau14 -= np.einsum("ik,jkab->ijab", tau8, u[o, o, v, v], optimize=True)

    tau18 = zeros((N, N, M, M))

    tau18 -= 2 * np.einsum("ijba->ijab", tau14, optimize=True)

    tau104 = zeros((N, N, M, M))

    tau104 += 4 * np.einsum("ijba->ijab", tau14, optimize=True)

    tau121 = zeros((N, N, M, M))

    tau121 += np.einsum("ijba->ijab", tau14, optimize=True)

    tau130 = zeros((N, N, M, M))

    tau130 += 2 * np.einsum("ijba->ijab", tau14, optimize=True)

    tau14 = None

    tau53 = zeros((N, N, M, M))

    tau53 += np.einsum("kj,abik->ijab", tau8, t2, optimize=True)

    tau55 = zeros((N, N, M, M))

    tau55 -= 2 * np.einsum("ijba->ijab", tau53, optimize=True)

    tau139 = zeros((N, N, M, M))

    tau139 += np.einsum("ijba->ijab", tau53, optimize=True)

    tau53 = None

    tau68 += np.einsum("ai,jk->ijka", l1, tau8, optimize=True)

    tau74 = zeros((N, M))

    tau74 -= np.einsum("ja,ij->ia", f[o, v], tau8, optimize=True)

    tau86 += 2 * np.einsum("ia->ia", tau74, optimize=True)

    tau74 = None

    tau77 = zeros((N, M))

    tau77 += np.einsum("jk,ikja->ia", tau8, u[o, o, o, v], optimize=True)

    tau86 += 2 * np.einsum("ia->ia", tau77, optimize=True)

    tau77 = None

    tau102 = zeros((M, M))

    tau102 += np.einsum("ij,jaib->ab", tau8, u[o, v, o, v], optimize=True)

    tau106 = zeros((M, M))

    tau106 += 4 * np.einsum("ab->ab", tau102, optimize=True)

    tau102 = None

    tau118 = zeros((N, N, N, N))

    tau118 += np.einsum("im,jmlk->ijkl", tau8, u[o, o, o, o], optimize=True)

    tau124 -= 2 * np.einsum("ijlk->ijkl", tau118, optimize=True)

    tau118 = None

    tau128 = zeros((N, N))

    tau128 -= np.einsum("kl,ilkj->ij", tau8, u[o, o, o, o], optimize=True)

    tau132 = zeros((N, N))

    tau132 -= 4 * np.einsum("ij->ij", tau128, optimize=True)

    tau128 = None

    tau134 -= np.einsum("ai,jk->ijka", l1, tau8, optimize=True)

    tau135 = zeros((N, N, M, M))

    tau135 += np.einsum("ijkc,kcab->ijab", tau134, u[o, v, v, v], optimize=True)

    tau134 = None

    tau136 = zeros((N, N, M, M))

    tau136 -= 4 * np.einsum("ijba->ijab", tau135, optimize=True)

    tau135 = None

    tau140 += np.einsum("ik,jl->ijkl", tau8, tau8, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau18 -= 4 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau104 += 8 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau121 += 2 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau130 += 4 * np.einsum("ijba->ijab", tau10, optimize=True)

    tau10 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau18 -= 2 * np.einsum("jiba->ijab", tau11, optimize=True)

    tau104 += 2 * np.einsum("jiba->ijab", tau11, optimize=True)

    tau11 = None

    tau12 = zeros((N, N, N, N))

    tau12 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 -= np.einsum("jilk,lkab->ijab", tau12, u[o, o, v, v], optimize=True)

    tau18 += np.einsum("ijba->ijab", tau13, optimize=True)

    tau104 -= np.einsum("ijba->ijab", tau13, optimize=True)

    tau13 = None

    tau51 = zeros((N, N, M, M))

    tau51 -= np.einsum("ablk,lkji->ijab", t2, tau12, optimize=True)

    tau55 += np.einsum("ijba->ijab", tau51, optimize=True)

    tau51 = None

    tau65 = zeros((N, N, N, M))

    tau65 += np.einsum("al,ijkl->ijka", l1, tau12, optimize=True)

    tau68 += np.einsum("ijka->ijka", tau65, optimize=True)

    tau143 = zeros((N, N, N, M))

    tau143 -= np.einsum("ijka->ijka", tau65, optimize=True)

    tau65 = None

    tau76 = zeros((N, M))

    tau76 += np.einsum("ijlk,lkja->ia", tau12, u[o, o, o, v], optimize=True)

    tau86 -= np.einsum("ia->ia", tau76, optimize=True)

    tau76 = None

    tau119 = zeros((N, N, N, N))

    tau119 += np.einsum("imnj,knml->ijkl", tau12, u[o, o, o, o], optimize=True)

    tau124 -= 4 * np.einsum("iljk->ijkl", tau119, optimize=True)

    tau119 = None

    tau140 -= np.einsum("inkm,jmln->ijkl", tau12, tau12, optimize=True)

    tau15 = zeros((N, N, M, M))

    tau15 -= np.einsum("caki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("ijab->ijab", tau15, optimize=True)

    tau15 = None

    tau16 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("caki,kjcb->ijab", l2, tau16, optimize=True)

    tau18 -= 4 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau37 = zeros((N, N, M, M))

    tau37 += 4 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau98 = zeros((N, N, M, M))

    tau98 += 4 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau104 -= 8 * np.einsum("jiab->ijab", tau17, optimize=True)

    tau121 += 4 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau130 -= 8 * np.einsum("ijba->ijab", tau17, optimize=True)

    tau17 = None

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum("kilj,klab->ijab", tau12, tau16, optimize=True)

    tau26 -= 2 * np.einsum("ijba->ijab", tau25, optimize=True)

    tau25 = None

    tau18 -= 4 * np.einsum("ai,jb->ijab", l1, f[o, v], optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("caki,jkcb->ijab", t2, tau18, optimize=True)

    tau18 = None

    tau26 += np.einsum("jiba->ijab", tau19, optimize=True)

    tau19 = None

    tau20 = zeros((N, N, M, M))

    tau20 -= np.einsum("caik,cbkj->ijab", l2, t2, optimize=True)

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("jkcb,ikca->ijab", tau16, tau20, optimize=True)

    tau26 += 4 * np.einsum("ijba->ijab", tau24, optimize=True)

    tau24 = None

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("kjbc,kiac->ijab", tau16, tau20, optimize=True)

    tau45 = zeros((N, N, M, M))

    tau45 += 4 * np.einsum("jiab->ijab", tau39, optimize=True)

    tau39 = None

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum("iljk,klab->ijab", tau12, tau20, optimize=True)

    tau57 += 2 * np.einsum("ijab->ijab", tau49, optimize=True)

    tau49 = None

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum("ikcb,kjac->ijab", tau20, tau20, optimize=True)

    tau57 -= 4 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau50 = None

    tau52 = zeros((N, N, M, M))

    tau52 += np.einsum("bcjk,kica->ijab", t2, tau20, optimize=True)

    tau55 += 4 * np.einsum("ijba->ijab", tau52, optimize=True)

    tau139 += 2 * np.einsum("ijab->ijab", tau52, optimize=True)

    tau140 -= np.einsum("abji,klab->ijkl", l2, tau139, optimize=True)

    tau139 = None

    tau141 = zeros((N, N, M, M))

    tau141 += 2 * np.einsum("ijab->ijab", tau52, optimize=True)

    tau52 = None

    tau62 = zeros((N, M, M, M))

    tau62 += np.einsum("aj,ijbc->iabc", l1, tau20, optimize=True)

    tau63 -= 2 * np.einsum("iabc->iabc", tau62, optimize=True)

    tau63 += 2 * np.einsum("ibac->iabc", tau62, optimize=True)

    tau62 = None

    tau78 = zeros((N, M))

    tau78 -= np.einsum("ijbc,jbac->ia", tau20, u[o, v, v, v], optimize=True)

    tau86 += 4 * np.einsum("ia->ia", tau78, optimize=True)

    tau78 = None

    tau91 = zeros((M, M, M, M))

    tau91 += np.einsum("ijcd,ijab->abcd", tau16, tau20, optimize=True)

    tau96 = zeros((M, M, M, M))

    tau96 -= 4 * np.einsum("abcd->abcd", tau91, optimize=True)

    tau91 = None

    tau110 = zeros((N, N, N, M))

    tau110 += np.einsum("liab,jklb->ijka", tau20, u[o, o, o, v], optimize=True)

    tau111 = zeros((N, N, N, M))

    tau111 -= 2 * np.einsum("ikja->ijka", tau110, optimize=True)

    tau110 = None

    tau123 = zeros((N, N, N, N))

    tau123 += np.einsum("klab,ijab->ijkl", tau16, tau20, optimize=True)

    tau124 -= 8 * np.einsum("ilkj->ijkl", tau123, optimize=True)

    tau123 = None

    tau138 = zeros((M, M, M, M))

    tau138 += 4 * np.einsum("ijac,jibd->abcd", tau20, tau20, optimize=True)

    tau140 -= 4 * np.einsum("ikba,jlab->ijkl", tau20, tau20, optimize=True)

    r2 -= np.einsum("ijkl,klba->abij", tau140, u[o, o, v, v], optimize=True) / 4

    tau140 = None

    tau21 = zeros((M, M, M, M))

    tau21 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau22 = zeros((M, M, M, M))

    tau22 += np.einsum("badc->abcd", tau21, optimize=True)

    tau22 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("ijcd,cadb->ijab", tau20, tau22, optimize=True)

    tau22 = None

    tau26 -= 2 * np.einsum("ijba->ijab", tau23, optimize=True)

    tau23 = None

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("caki,jkbc->ijab", l2, tau26, optimize=True)

    tau26 = None

    tau87 -= np.einsum("jiab->ijab", tau27, optimize=True)

    tau27 = None

    tau28 = zeros((M, M))

    tau28 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau29 = zeros((N, N, M, M))

    tau29 -= np.einsum("ac,ibjc->ijab", tau28, u[o, v, o, v], optimize=True)

    tau45 -= 2 * np.einsum("ijab->ijab", tau29, optimize=True)

    tau29 = None

    tau33 = zeros((N, N, M, M))

    tau33 -= np.einsum("ac,ijbc->ijab", tau28, u[o, o, v, v], optimize=True)

    tau37 -= 2 * np.einsum("ijab->ijab", tau33, optimize=True)

    tau98 -= np.einsum("ijab->ijab", tau33, optimize=True)

    tau104 -= 2 * np.einsum("ijab->ijab", tau33, optimize=True)

    tau130 -= 4 * np.einsum("ijab->ijab", tau33, optimize=True)

    tau33 = None

    tau54 = zeros((N, N, M, M))

    tau54 -= np.einsum("cb,acji->ijab", tau28, t2, optimize=True)

    tau55 += 2 * np.einsum("ijab->ijab", tau54, optimize=True)

    tau56 = zeros((N, N, M, M))

    tau56 += np.einsum("caki,kjcb->ijab", l2, tau55, optimize=True)

    tau55 = None

    tau57 += np.einsum("ijab->ijab", tau56, optimize=True)

    tau56 = None

    tau141 -= np.einsum("ijab->ijab", tau54, optimize=True)

    tau54 = None

    tau142 += np.einsum("ijab,lkab->ijkl", tau141, u[o, o, v, v], optimize=True)

    tau141 = None

    tau57 -= np.einsum("ab,ij->ijab", tau28, tau8, optimize=True)

    tau8 = None

    tau63 += np.einsum("ai,bc->iabc", l1, tau28, optimize=True)

    tau79 = zeros((N, M))

    tau79 += np.einsum("bc,ibac->ia", tau28, u[o, v, v, v], optimize=True)

    tau86 += 2 * np.einsum("ia->ia", tau79, optimize=True)

    tau79 = None

    tau88 = zeros((M, M, M, M))

    tau88 += np.einsum("ae,cbde->abcd", tau28, u[v, v, v, v], optimize=True)

    tau96 += np.einsum("acbd->abcd", tau88, optimize=True)

    tau88 = None

    tau103 = zeros((M, M))

    tau103 -= np.einsum("cd,cabd->ab", tau28, u[v, v, v, v], optimize=True)

    tau106 -= 4 * np.einsum("ab->ab", tau103, optimize=True)

    tau103 = None

    tau109 = zeros((N, N, N, M))

    tau109 -= np.einsum("ab,ijkb->ijka", tau28, u[o, o, o, v], optimize=True)

    tau111 -= np.einsum("kjia->ijka", tau109, optimize=True)

    tau109 = None

    tau129 = zeros((N, N))

    tau129 += np.einsum("ab,iajb->ij", tau28, u[o, v, o, v], optimize=True)

    tau132 += 4 * np.einsum("ij->ij", tau129, optimize=True)

    tau129 = None

    tau138 += np.einsum("ad,bc->abcd", tau28, tau28, optimize=True)

    tau28 = None

    tau30 = zeros((M, M, M, M))

    tau30 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("ijcd,cadb->ijab", tau16, tau30, optimize=True)

    tau16 = None

    tau45 -= 2 * np.einsum("jiab->ijab", tau31, optimize=True)

    tau31 = None

    tau48 = zeros((N, N, M, M))

    tau48 += np.einsum("ijdc,acbd->ijab", tau20, tau30, optimize=True)

    tau57 += 2 * np.einsum("ijab->ijab", tau48, optimize=True)

    tau48 = None

    tau58 = zeros((N, N, M, M))

    tau58 += np.einsum("jkbc,kica->ijab", tau57, u[o, o, v, v], optimize=True)

    tau57 = None

    tau87 -= np.einsum("jiba->ijab", tau58, optimize=True)

    tau58 = None

    tau89 = zeros((M, M, M, M))

    tau89 -= np.einsum("aefb,cedf->abcd", tau30, u[v, v, v, v], optimize=True)

    tau96 -= 2 * np.einsum("abcd->abcd", tau89, optimize=True)

    tau89 = None

    tau90 = zeros((M, M, M, M))

    tau90 -= np.einsum("cedf,aefb->abcd", tau21, tau30, optimize=True)

    tau21 = None

    tau96 += np.einsum("acbd->abcd", tau90, optimize=True)

    tau90 = None

    tau138 += np.einsum("afce,bedf->abcd", tau30, tau30, optimize=True)

    tau30 = None

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True)

    tau37 += 4 * np.einsum("jiab->ijab", tau32, optimize=True)

    tau98 += 2 * np.einsum("jiab->ijab", tau32, optimize=True)

    tau99 = zeros((N, N, M, M))

    tau99 += np.einsum("ijkl,lkab->ijab", tau12, tau98, optimize=True)

    tau98 = None

    tau113 = zeros((N, N, M, M))

    tau113 -= np.einsum("jiab->ijab", tau99, optimize=True)

    tau99 = None

    tau104 += 4 * np.einsum("jiab->ijab", tau32, optimize=True)

    tau130 += 8 * np.einsum("jiab->ijab", tau32, optimize=True)

    tau32 = None

    tau34 = zeros((N, N, N, N))

    tau34 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau35 = zeros((N, N, N, N))

    tau35 += np.einsum("lkji->ijkl", tau34, optimize=True)

    tau120 = zeros((N, N, N, N))

    tau120 += np.einsum("imnk,mjln->ijkl", tau12, tau34, optimize=True)

    tau34 = None

    tau12 = None

    tau124 += 2 * np.einsum("ilkj->ijkl", tau120, optimize=True)

    tau120 = None

    tau35 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum("abkl,ijkl->ijab", l2, tau35, optimize=True)

    tau37 += np.einsum("jiba->ijab", tau36, optimize=True)

    tau130 += np.einsum("jiba->ijab", tau36, optimize=True)

    tau36 = None

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum("klab,likj->ijab", tau20, tau35, optimize=True)

    tau35 = None

    tau20 = None

    tau45 -= 2 * np.einsum("ijab->ijab", tau40, optimize=True)

    tau40 = None

    tau37 += 4 * np.einsum("ai,jb->ijab", l1, f[o, v], optimize=True)

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum("caki,kjbc->ijab", t2, tau37, optimize=True)

    tau37 = None

    tau45 -= np.einsum("jiba->ijab", tau38, optimize=True)

    tau38 = None

    tau41 = zeros((N, N, N, M))

    tau41 += np.einsum("bali,jlkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau43 = zeros((N, N, N, M))

    tau43 -= 2 * np.einsum("ijka->ijka", tau41, optimize=True)

    tau81 = zeros((N, N, N, M))

    tau81 += 4 * np.einsum("jika->ijka", tau41, optimize=True)

    tau41 = None

    tau42 = zeros((N, N, N, M))

    tau42 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau43 -= np.einsum("kija->ijka", tau42, optimize=True)

    tau81 += np.einsum("kjia->ijka", tau42, optimize=True)

    tau42 = None

    tau43 -= 2 * np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("ak,kijb->ijab", l1, tau43, optimize=True)

    tau43 = None

    tau45 -= 2 * np.einsum("ijab->ijab", tau44, optimize=True)

    tau44 = None

    tau46 = zeros((N, N, M, M))

    tau46 += np.einsum("caki,jkbc->ijab", l2, tau45, optimize=True)

    tau45 = None

    tau87 -= np.einsum("ijba->ijab", tau46, optimize=True)

    tau46 = None

    tau60 = zeros((N, M))

    tau60 += np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau61 = zeros((N, M, M, M))

    tau61 += np.einsum("jc,baij->iabc", tau60, l2, optimize=True)

    tau63 += 2 * np.einsum("ibac->iabc", tau61, optimize=True)

    tau61 = None

    tau64 = zeros((N, N, M, M))

    tau64 += np.einsum("jcbd,icda->ijab", tau63, u[o, v, v, v], optimize=True)

    tau63 = None

    tau87 += 2 * np.einsum("jiba->ijab", tau64, optimize=True)

    tau64 = None

    tau66 = zeros((N, N, N, M))

    tau66 += np.einsum("kb,baij->ijka", tau60, l2, optimize=True)

    tau68 -= 2 * np.einsum("ijka->ijka", tau66, optimize=True)

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum("ljkb,kila->ijab", tau68, u[o, o, o, v], optimize=True)

    tau68 = None

    tau87 += 2 * np.einsum("jiba->ijab", tau69, optimize=True)

    tau69 = None

    tau143 += 2 * np.einsum("ijka->ijka", tau66, optimize=True)

    tau66 = None

    r2 += np.einsum("ijkc,kcba->abij", tau143, u[o, v, v, v], optimize=True) / 2

    tau143 = None

    tau73 = zeros((N, M))

    tau73 += np.einsum("jb,ijab->ia", tau60, u[o, o, v, v], optimize=True)

    tau86 += 4 * np.einsum("ia->ia", tau73, optimize=True)

    tau73 = None

    tau101 = zeros((M, M))

    tau101 += np.einsum("ic,iabc->ab", tau60, u[o, v, v, v], optimize=True)

    tau106 += 8 * np.einsum("ab->ab", tau101, optimize=True)

    tau101 = None

    tau108 = zeros((N, N, N, M))

    tau108 -= np.einsum("ib,jkab->ijka", tau60, u[o, o, v, v], optimize=True)

    tau111 -= 2 * np.einsum("ikja->ijka", tau108, optimize=True)

    tau108 = None

    tau112 = zeros((N, N, M, M))

    tau112 += np.einsum("ak,kijb->ijab", l1, tau111, optimize=True)

    tau111 = None

    tau113 += 4 * np.einsum("ijab->ijab", tau112, optimize=True)

    tau112 = None

    tau114 = zeros((N, N))

    tau114 += np.einsum("ai,ja->ij", l1, tau60, optimize=True)

    tau115 = zeros((N, N, M, M))

    tau115 -= np.einsum("ik,jkab->ijab", tau114, u[o, o, v, v], optimize=True)

    tau114 = None

    tau136 -= 8 * np.einsum("ijba->ijab", tau115, optimize=True)

    tau115 = None

    tau127 = zeros((N, N))

    tau127 += np.einsum("ka,ikja->ij", tau60, u[o, o, o, v], optimize=True)

    tau132 += 8 * np.einsum("ij->ij", tau127, optimize=True)

    tau127 = None

    tau142 += 4 * np.einsum("ja,lkia->ijkl", tau60, u[o, o, o, v], optimize=True)

    tau60 = None

    r2 += np.einsum("bakl,klij->abij", l2, tau142, optimize=True) / 4

    tau142 = None

    tau70 = zeros((N, M))

    tau70 += np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    tau86 += 4 * np.einsum("ia->ia", tau70, optimize=True)

    tau70 = None

    tau71 = zeros((N, M))

    tau71 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau86 -= 4 * np.einsum("ia->ia", tau71, optimize=True)

    tau71 = None

    tau72 = zeros((N, M))

    tau72 += np.einsum("cbij,cbja->ia", l2, u[v, v, o, v], optimize=True)

    tau86 -= 2 * np.einsum("ia->ia", tau72, optimize=True)

    tau72 = None

    tau80 = zeros((N, N, N, M))

    tau80 += np.einsum("ib,abjk->ijka", f[o, v], t2, optimize=True)

    tau81 -= 2 * np.einsum("ikja->ijka", tau80, optimize=True)

    tau80 = None

    tau81 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau82 = zeros((N, M))

    tau82 += np.einsum("bajk,ijkb->ia", l2, tau81, optimize=True)

    tau81 = None

    tau86 -= np.einsum("ia->ia", tau82, optimize=True)

    tau82 = None

    tau83 = zeros((N, N))

    tau83 -= np.einsum("baki,jkba->ij", t2, u[o, o, v, v], optimize=True)

    tau84 = zeros((N, N))

    tau84 += np.einsum("ji->ij", tau83, optimize=True)

    tau83 = None

    tau84 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    tau85 = zeros((N, M))

    tau85 += np.einsum("aj,ij->ia", l1, tau84, optimize=True)

    tau84 = None

    tau86 -= 2 * np.einsum("ia->ia", tau85, optimize=True)

    tau85 = None

    tau87 += np.einsum("ai,jb->ijab", l1, tau86, optimize=True)

    tau86 = None

    r2 += np.einsum("ijab->abij", tau87, optimize=True) / 4

    r2 -= np.einsum("ijba->abij", tau87, optimize=True) / 4

    r2 -= np.einsum("jiab->abij", tau87, optimize=True) / 4

    r2 += np.einsum("jiba->abij", tau87, optimize=True) / 4

    tau87 = None

    tau92 = zeros((N, M, M, M))

    tau92 += np.einsum("ja,bcij->iabc", f[o, v], t2, optimize=True)

    tau94 = zeros((N, M, M, M))

    tau94 -= np.einsum("icba->iabc", tau92, optimize=True)

    tau92 = None

    tau93 = zeros((N, M, M, M))

    tau93 += np.einsum("daji,jbcd->iabc", t2, u[o, v, v, v], optimize=True)

    tau94 += 2 * np.einsum("iabc->iabc", tau93, optimize=True)

    tau93 = None

    tau94 += np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau95 = zeros((M, M, M, M))

    tau95 += np.einsum("ai,ibcd->abcd", l1, tau94, optimize=True)

    tau94 = None

    tau96 += 2 * np.einsum("abcd->abcd", tau95, optimize=True)

    tau95 = None

    tau97 = zeros((N, N, M, M))

    tau97 += np.einsum("cdij,acdb->ijab", l2, tau96, optimize=True)

    tau96 = None

    tau113 += 2 * np.einsum("jiab->ijab", tau97, optimize=True)

    tau97 = None

    tau100 = zeros((M, M))

    tau100 -= np.einsum("ci,caib->ab", l1, u[v, v, o, v], optimize=True)

    tau106 += 8 * np.einsum("ab->ab", tau100, optimize=True)

    tau100 = None

    tau104 -= 8 * np.einsum("aj,ib->ijab", l1, f[o, v], optimize=True)

    tau105 = zeros((M, M))

    tau105 += np.einsum("caij,jicb->ab", t2, tau104, optimize=True)

    tau104 = None

    tau106 -= np.einsum("ab->ab", tau105, optimize=True)

    tau105 = None

    tau107 = zeros((N, N, M, M))

    tau107 += np.einsum("cb,caij->ijab", tau106, l2, optimize=True)

    tau106 = None

    tau113 += np.einsum("jiab->ijab", tau107, optimize=True)

    tau107 = None

    r2 += np.einsum("jiab->abij", tau113, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau113, optimize=True) / 8

    tau113 = None

    tau116 = zeros((N, N, N, N))

    tau116 += np.einsum("ai,jakl->ijkl", l1, u[o, v, o, o], optimize=True)

    tau124 -= 4 * np.einsum("ijlk->ijkl", tau116, optimize=True)

    tau116 = None

    tau121 += 4 * np.einsum("ai,jb->ijab", l1, f[o, v], optimize=True)

    tau122 = zeros((N, N, N, N))

    tau122 += np.einsum("abij,klba->ijkl", t2, tau121, optimize=True)

    tau121 = None

    tau124 -= np.einsum("lkij->ijkl", tau122, optimize=True)

    tau122 = None

    tau125 = zeros((N, N, M, M))

    tau125 += np.einsum("abkl,ijkl->ijab", l2, tau124, optimize=True)

    tau124 = None

    tau136 -= np.einsum("ijba->ijab", tau125, optimize=True)

    tau125 = None

    tau126 = zeros((N, N))

    tau126 -= np.einsum("ak,iakj->ij", l1, u[o, v, o, o], optimize=True)

    tau132 += 8 * np.einsum("ij->ij", tau126, optimize=True)

    tau126 = None

    tau130 -= 8 * np.einsum("bi,ja->ijab", l1, f[o, v], optimize=True)

    tau131 = zeros((N, N))

    tau131 += np.einsum("abki,kjba->ij", t2, tau130, optimize=True)

    tau130 = None

    tau132 -= np.einsum("ji->ij", tau131, optimize=True)

    tau131 = None

    tau133 = zeros((N, N, M, M))

    tau133 += np.einsum("jk,abki->ijab", tau132, l2, optimize=True)

    tau132 = None

    tau136 += np.einsum("ijba->ijab", tau133, optimize=True)

    tau133 = None

    r2 += np.einsum("ijba->abij", tau136, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau136, optimize=True) / 8

    tau136 = None

    tau137 = zeros((N, M, M, M))

    tau137 += np.einsum("aj,bcij->iabc", l1, t2, optimize=True)

    tau138 -= 2 * np.einsum("ai,ibcd->abcd", l1, tau137, optimize=True)

    tau137 = None

    r2 -= np.einsum("bacd,jicd->abij", tau138, u[o, o, v, v], optimize=True) / 4

    tau138 = None

    return r2
