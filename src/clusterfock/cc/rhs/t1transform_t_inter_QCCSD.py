import numpy as np

from clusterfock.cc.rhs.t_inter_CCD import amplitudes_intermediates_ccd
from clusterfock.cc.rhs.t1transform_CCSD import t1_transform_intermediates_ccsd


def t1_transform_t1_intermediates_qccsd(t2, l1, l2, u, f, v, o):
    t1 = t1_transform_intermediates_ccsd(t2, u, f, v, o)
    t1 += t1_transformed_t1_qccsd_addition(t2, l1, l2, u, f, v, o)

    return t1


def t1_transform_t2_intermediates_qccsd(t2, l1, l2, u, f, v, o):
    t2 = amplitudes_intermediates_ccd(t2, u, f, v, o)
    t2 += t1_transformed_t2_qccsd_addition(t2, l1, l2, u, f, v, o)

    return 0.25 * (
        t2 - t2.transpose(1, 0, 2, 3) - t2.transpose(0, 1, 3, 2) + t2.transpose(1, 0, 3, 2)
    )


def t1_transformed_t1_qccsd_addition(t2, l1, l2, u, f, v, o):
    M, N = l1.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 -= np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau15 = zeros((N, N, N, M))

    tau15 += 4 * np.einsum("lkab,ijlb->ijka", tau0, u[o, o, o, v], optimize=True)

    tau15 += 8 * np.einsum("ikbc,jbac->ijka", tau0, u[o, v, v, v], optimize=True)

    tau16 = zeros((N, N, M, M))

    tau16 += 8 * np.einsum("acjk,kicb->ijab", t2, tau0, optimize=True)

    tau19 = zeros((N, M))

    tau19 += 4 * np.einsum("ijbc,jbca->ia", tau0, u[o, v, v, v], optimize=True)

    r1 = zeros((M, N))

    r1 -= np.einsum("jicb,acjb->ai", tau0, u[v, v, o, v], optimize=True)

    tau0 = None

    tau1 = zeros((M, M))

    tau1 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau15 -= 2 * np.einsum("ab,ijkb->ijka", tau1, u[o, o, o, v], optimize=True)

    tau16 += 4 * np.einsum("cb,acij->ijab", tau1, t2, optimize=True)

    tau19 += 2 * np.einsum("bc,ibac->ia", tau1, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bc,abic->ai", tau1, u[v, v, o, v], optimize=True) / 2

    tau1 = None

    tau2 = zeros((N, M))

    tau2 -= np.einsum("bj,baij->ia", l1, t2, optimize=True)

    tau15 -= 4 * np.einsum("kb,ijab->ijka", tau2, u[o, o, v, v], optimize=True)

    tau19 += 4 * np.einsum("jb,jiba->ia", tau2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("ab,ib->ai", f[v, v], tau2, optimize=True)

    r1 -= np.einsum("jb,jaib->ai", tau2, u[o, v, o, v], optimize=True)

    tau2 = None

    tau3 = zeros((N, N))

    tau3 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau15 += 4 * np.einsum("il,jlka->ijka", tau3, u[o, o, o, v], optimize=True)

    tau16 -= 2 * np.einsum("kj,baik->ijab", tau3, t2, optimize=True)

    tau19 -= 2 * np.einsum("ja,ij->ia", f[o, v], tau3, optimize=True)

    tau19 += 2 * np.einsum("jk,ikja->ia", tau3, u[o, o, o, v], optimize=True)

    r1 += np.einsum("jk,kaij->ai", tau3, u[o, v, o, o], optimize=True) / 2

    tau3 = None

    tau4 = zeros((N, N, N, N))

    tau4 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau15 += np.einsum("ijml,mlka->ijka", tau4, u[o, o, o, v], optimize=True)

    tau15 -= 4 * np.einsum("likm,jmla->ijka", tau4, u[o, o, o, v], optimize=True)

    tau16 -= np.einsum("balk,lkji->ijab", t2, tau4, optimize=True)

    r1 += np.einsum("ijbc,jabc->ai", tau16, u[o, v, v, v], optimize=True) / 8

    tau16 = None

    tau19 += np.einsum("jilk,lkja->ia", tau4, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("lkij,jalk->ai", tau4, u[o, v, o, o], optimize=True) / 4

    tau4 = None

    tau5 = zeros((N, M, M, M))

    tau5 += np.einsum("di,adbc->iabc", l1, u[v, v, v, v], optimize=True)

    r1 += np.einsum("cbij,jacb->ai", t2, tau5, optimize=True) / 2

    tau5 = None

    tau6 = zeros((N, N, N, M))

    tau6 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau15 -= 8 * np.einsum("iklb,ljba->ijka", tau6, u[o, o, v, v], optimize=True)

    tau19 -= 2 * np.einsum("ikjb,kjba->ia", tau6, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("jikb,kajb->ai", tau6, u[o, v, o, v], optimize=True)

    tau6 = None

    tau7 = zeros((N, M, M, M))

    tau7 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau7 += np.einsum("jc,baij->iabc", f[o, v], t2, optimize=True)

    tau7 -= 2 * np.einsum("dbji,jadc->iabc", t2, u[o, v, v, v], optimize=True)

    tau15 -= 2 * np.einsum("bcji,kbca->ijka", l2, tau7, optimize=True)

    tau7 = None

    tau8 = zeros((N, N, N, M))

    tau8 += np.einsum("ib,abjk->ijka", f[o, v], t2, optimize=True)

    tau11 = zeros((N, N, N, M))

    tau11 += 2 * np.einsum("ikja->ijka", tau8, optimize=True)

    tau17 = zeros((N, N, N, M))

    tau17 -= 2 * np.einsum("ikja->ijka", tau8, optimize=True)

    tau8 = None

    tau9 = zeros((N, N, N, M))

    tau9 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau11 -= np.einsum("kjia->ijka", tau9, optimize=True)

    tau17 += np.einsum("kjia->ijka", tau9, optimize=True)

    tau9 = None

    tau10 = zeros((N, N, N, M))

    tau10 -= np.einsum("bali,ljkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau11 -= 2 * np.einsum("jika->ijka", tau10, optimize=True)

    tau11 += 2 * np.einsum("kija->ijka", tau10, optimize=True)

    tau17 += 4 * np.einsum("jika->ijka", tau10, optimize=True)

    tau10 = None

    tau11 -= 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau15 -= 4 * np.einsum("balj,ilkb->ijka", l2, tau11, optimize=True)

    tau11 = None

    tau12 = zeros((N, N, N, N))

    tau12 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau12 += np.einsum("balk,jiba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau15 -= 2 * np.einsum("al,jilk->ijka", l1, tau12, optimize=True)

    tau12 = None

    tau13 = zeros((N, N))

    tau13 += np.einsum("baki,kjba->ij", t2, u[o, o, v, v], optimize=True)

    tau14 = zeros((N, N))

    tau14 += np.einsum("ij->ij", tau13, optimize=True)

    tau18 = zeros((N, N))

    tau18 += np.einsum("ji->ij", tau13, optimize=True)

    tau13 = None

    tau14 += 2 * np.einsum("ji->ij", f[o, o], optimize=True)

    tau15 += 4 * np.einsum("aj,ki->ijka", l1, tau14, optimize=True)

    tau14 = None

    tau15 -= 8 * np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bakj,jkib->ai", t2, tau15, optimize=True) / 8

    tau15 = None

    tau17 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau19 -= np.einsum("bajk,ijkb->ia", l2, tau17, optimize=True)

    tau17 = None

    tau18 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    tau19 -= 2 * np.einsum("aj,ij->ia", l1, tau18, optimize=True)

    tau18 = None

    tau19 += 4 * np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    tau19 -= 4 * np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau19 += 2 * np.einsum("cbji,cbja->ia", l2, u[v, v, o, v], optimize=True)

    r1 += np.einsum("jb,baji->ai", tau19, t2, optimize=True) / 4

    tau19 = None

    r1 += np.einsum("bj,abij->ai", l1, u[v, v, o, o], optimize=True)

    return r1


def t1_transformed_t2_qccsd_addition(t2, l1, l2, u, f, v, o):
    M, N = l1.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 -= np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau1 = zeros((N, N, M, M))

    tau1 -= np.einsum("caik,kjcb->ijab", t2, tau0, optimize=True)

    tau25 = zeros((N, N, M, M))

    tau25 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau50 = zeros((N, N, M, M))

    tau50 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau99 = zeros((N, N, M, M))

    tau99 += 2 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau101 = zeros((N, N, N, N))

    tau101 -= 2 * np.einsum("lkba,jiba->ijkl", tau1, u[o, o, v, v], optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("klab,lkji->abij", tau1, u[o, o, o, o], optimize=True)

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum("kiac,jbkc->ijab", tau0, u[o, v, o, v], optimize=True)

    tau37 = zeros((N, N, M, M))

    tau37 += 2 * np.einsum("ijab->ijab", tau34, optimize=True)

    tau34 = None

    tau36 = zeros((N, N, M, M))

    tau36 -= np.einsum("ijcd,acdb->ijab", tau0, u[v, v, v, v], optimize=True)

    tau37 += 2 * np.einsum("jiba->ijab", tau36, optimize=True)

    tau36 = None

    tau43 = zeros((N, N, M, M))

    tau43 += np.einsum("klab,iljk->ijab", tau0, u[o, o, o, o], optimize=True)

    tau46 = zeros((N, N, M, M))

    tau46 += 2 * np.einsum("ijab->ijab", tau43, optimize=True)

    tau43 = None

    tau45 = zeros((N, N, M, M))

    tau45 += np.einsum("ikca,kcjb->ijab", tau0, u[o, v, o, v], optimize=True)

    tau46 += 2 * np.einsum("ijba->ijab", tau45, optimize=True)

    tau45 = None

    tau2 = zeros((N, N, N, N))

    tau2 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau4 = zeros((N, N, M, M))

    tau4 -= np.einsum("klab,ikjl->ijab", tau0, tau2, optimize=True)

    tau27 = zeros((N, N, M, M))

    tau27 += 2 * np.einsum("jiab->ijab", tau4, optimize=True)

    tau4 = None

    tau68 = zeros((N, N, N, N))

    tau68 += np.einsum("lkji->ijkl", tau2, optimize=True)

    r2 -= np.einsum("klba,jilk->abij", tau1, tau2, optimize=True) / 2

    tau1 = None

    tau3 = zeros((N, N, N, M))

    tau3 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau33 = zeros((N, N, M, M))

    tau33 += np.einsum("ijkc,kacb->ijab", tau3, u[o, v, v, v], optimize=True)

    tau37 -= 2 * np.einsum("jiba->ijab", tau33, optimize=True)

    tau33 = None

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("ilka,lkjb->ijab", tau3, u[o, o, o, v], optimize=True)

    tau46 -= np.einsum("ijba->ijab", tau39, optimize=True)

    tau39 = None

    tau83 = zeros((N, N, N, N))

    tau83 += np.einsum("ijma,mkla->ijkl", tau3, u[o, o, o, v], optimize=True)

    tau88 = zeros((N, N, N, N))

    tau88 -= 4 * np.einsum("ijkl->ijkl", tau83, optimize=True)

    tau83 = None

    r2 += np.einsum("kijc,abkc->abij", tau3, u[v, v, o, v], optimize=True)

    tau3 = None

    tau5 = zeros((M, M, M, M))

    tau5 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("acbd,icjd->ijab", tau5, u[o, v, o, v], optimize=True)

    tau46 -= np.einsum("ijab->ijab", tau44, optimize=True)

    tau44 = None

    tau6 = zeros((N, N, M, M))

    tau6 -= np.einsum("caik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("cabd,ijcd->ijab", tau5, tau6, optimize=True)

    tau27 += 2 * np.einsum("jiab->ijab", tau7, optimize=True)

    tau7 = None

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("kiac,kjbc->ijab", tau0, tau6, optimize=True)

    tau27 += 4 * np.einsum("jiab->ijab", tau8, optimize=True)

    tau8 = None

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("ikca,jkcb->ijab", tau0, tau6, optimize=True)

    tau27 += 4 * np.einsum("ijba->ijab", tau9, optimize=True)

    tau9 = None

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau10 = zeros((M, M, M, M))

    tau10 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau11 = zeros((N, N, M, M))

    tau11 -= np.einsum("ijcd,acdb->ijab", tau0, tau10, optimize=True)

    tau27 -= 2 * np.einsum("ijba->ijab", tau11, optimize=True)

    tau11 = None

    tau61 = zeros((M, M, M, M))

    tau61 += np.einsum("badc->abcd", tau10, optimize=True)

    tau10 = None

    tau12 = zeros((N, N, N, N))

    tau12 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 -= np.einsum("kijl,klab->ijab", tau12, tau6, optimize=True)

    tau6 = None

    tau27 -= 2 * np.einsum("ijba->ijab", tau13, optimize=True)

    tau13 = None

    tau24 = zeros((N, N, M, M))

    tau24 -= np.einsum("ablk,lkji->ijab", t2, tau12, optimize=True)

    tau25 -= np.einsum("ijba->ijab", tau24, optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("jkcb,kica->ijab", tau25, u[o, o, v, v], optimize=True)

    tau25 = None

    tau27 -= np.einsum("ijab->ijab", tau26, optimize=True)

    tau26 = None

    tau50 -= np.einsum("ijba->ijab", tau24, optimize=True)

    tau24 = None

    tau35 = zeros((N, N, M, M))

    tau35 += np.einsum("ikjl,lakb->ijab", tau12, u[o, v, o, v], optimize=True)

    tau37 -= np.einsum("jiba->ijab", tau35, optimize=True)

    tau35 = None

    tau67 = zeros((N, N, M, M))

    tau67 -= np.einsum("jilk,lkab->ijab", tau12, u[o, o, v, v], optimize=True)

    tau70 = zeros((N, N, M, M))

    tau70 += np.einsum("ijba->ijab", tau67, optimize=True)

    tau77 = zeros((N, N, M, M))

    tau77 -= np.einsum("ijba->ijab", tau67, optimize=True)

    tau67 = None

    tau85 = zeros((N, N, N, N))

    tau85 -= np.einsum("imjn,nklm->ijkl", tau12, u[o, o, o, o], optimize=True)

    tau88 += 2 * np.einsum("ijkl->ijkl", tau85, optimize=True)

    tau85 = None

    tau86 = zeros((N, N, N, N))

    tau86 += np.einsum("mikn,jmnl->ijkl", tau12, tau2, optimize=True)

    tau2 = None

    tau12 = None

    tau88 -= np.einsum("ijlk->ijkl", tau86, optimize=True)

    tau86 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True)

    tau22 = zeros((N, N, M, M))

    tau22 -= 2 * np.einsum("jiba->ijab", tau14, optimize=True)

    tau77 += 4 * np.einsum("jiab->ijab", tau14, optimize=True)

    tau94 = zeros((N, N, M, M))

    tau94 += 8 * np.einsum("jiab->ijab", tau14, optimize=True)

    tau100 = zeros((N, N, M, M))

    tau100 += 2 * np.einsum("jiab->ijab", tau14, optimize=True)

    tau14 = None

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau22 += 2 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau77 += 8 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau94 += 4 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau100 += 2 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau15 = None

    tau16 = zeros((M, M))

    tau16 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("ac,jibc->ijab", tau16, u[o, o, v, v], optimize=True)

    tau22 += np.einsum("ijba->ijab", tau17, optimize=True)

    tau77 -= 2 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau94 -= 4 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau100 -= np.einsum("ijab->ijab", tau17, optimize=True)

    tau17 = None

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum("cb,acij->ijab", tau16, t2, optimize=True)

    tau50 += 2 * np.einsum("ijba->ijab", tau49, optimize=True)

    tau99 -= np.einsum("ijab->ijab", tau49, optimize=True)

    tau49 = None

    r2 -= np.einsum("ijcd,bacd->abij", tau99, u[v, v, v, v], optimize=True) / 2

    tau99 = None

    tau76 = zeros((M, M))

    tau76 -= np.einsum("cd,acdb->ab", tau16, u[v, v, v, v], optimize=True)

    tau79 = zeros((M, M))

    tau79 -= 4 * np.einsum("ab->ab", tau76, optimize=True)

    tau76 = None

    tau93 = zeros((N, N))

    tau93 += np.einsum("ab,iajb->ij", tau16, u[o, v, o, v], optimize=True)

    tau16 = None

    tau96 = zeros((N, N))

    tau96 += 4 * np.einsum("ij->ij", tau93, optimize=True)

    tau93 = None

    tau18 = zeros((N, N))

    tau18 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 -= np.einsum("ik,jkab->ijab", tau18, u[o, o, v, v], optimize=True)

    tau22 += np.einsum("ijba->ijab", tau19, optimize=True)

    tau77 += 4 * np.einsum("ijba->ijab", tau19, optimize=True)

    tau94 += 2 * np.einsum("ijba->ijab", tau19, optimize=True)

    tau100 += np.einsum("ijba->ijab", tau19, optimize=True)

    tau19 = None

    tau48 = zeros((N, N, M, M))

    tau48 += np.einsum("kj,abik->ijab", tau18, t2, optimize=True)

    tau50 += 2 * np.einsum("ijba->ijab", tau48, optimize=True)

    tau48 = None

    tau51 = zeros((N, N, M, M))

    tau51 += np.einsum("jkcb,kaic->ijab", tau50, u[o, v, o, v], optimize=True)

    tau50 = None

    tau56 = zeros((N, N, M, M))

    tau56 -= np.einsum("jiba->ijab", tau51, optimize=True)

    tau51 = None

    tau75 = zeros((M, M))

    tau75 += np.einsum("ij,jaib->ab", tau18, u[o, v, o, v], optimize=True)

    tau79 += 4 * np.einsum("ab->ab", tau75, optimize=True)

    tau75 = None

    tau92 = zeros((N, N))

    tau92 -= np.einsum("kl,lijk->ij", tau18, u[o, o, o, o], optimize=True)

    tau96 -= 4 * np.einsum("ij->ij", tau92, optimize=True)

    tau92 = None

    tau101 -= 2 * np.einsum("im,mjlk->ijkl", tau18, u[o, o, o, o], optimize=True)

    tau18 = None

    tau20 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 += np.einsum("caki,kjcb->ijab", l2, tau20, optimize=True)

    tau22 += 2 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau77 -= 8 * np.einsum("jiab->ijab", tau21, optimize=True)

    tau94 -= 8 * np.einsum("ijba->ijab", tau21, optimize=True)

    tau100 += 4 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau21 = None

    tau63 = zeros((M, M, M, M))

    tau63 += np.einsum("ijcd,ijab->abcd", tau0, tau20, optimize=True)

    tau64 = zeros((M, M, M, M))

    tau64 += 4 * np.einsum("cdab->abcd", tau63, optimize=True)

    tau63 = None

    tau87 = zeros((N, N, N, N))

    tau87 += np.einsum("klab,ijab->ijkl", tau0, tau20, optimize=True)

    tau0 = None

    tau20 = None

    tau88 += 4 * np.einsum("lkij->ijkl", tau87, optimize=True)

    tau87 = None

    tau22 += 2 * np.einsum("ai,jb->ijab", l1, f[o, v], optimize=True)

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("cbkj,kiac->ijab", t2, tau22, optimize=True)

    tau22 = None

    tau27 -= 2 * np.einsum("ijab->ijab", tau23, optimize=True)

    tau23 = None

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("cbkj,kica->ijab", t2, tau27, optimize=True)

    tau27 = None

    tau56 -= np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau29 = zeros((N, N, N, M))

    tau29 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("ak,ikjb->ijab", l1, tau29, optimize=True)

    tau29 = None

    tau37 -= np.einsum("ijab->ijab", tau30, optimize=True)

    tau30 = None

    tau31 = zeros((N, M))

    tau31 -= np.einsum("bj,baij->ia", l1, t2, optimize=True)

    tau32 = zeros((N, N, M, M))

    tau32 -= np.einsum("ic,jabc->ijab", tau31, u[o, v, v, v], optimize=True)

    tau37 += 2 * np.einsum("ijba->ijab", tau32, optimize=True)

    tau32 = None

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum("cbkj,ikca->ijab", t2, tau37, optimize=True)

    tau37 = None

    tau56 -= 2 * np.einsum("ijba->ijab", tau38, optimize=True)

    tau38 = None

    tau40 = zeros((N, N, M, M))

    tau40 -= np.einsum("ka,ikjb->ijab", tau31, u[o, o, o, v], optimize=True)

    tau46 += 2 * np.einsum("ijba->ijab", tau40, optimize=True)

    tau40 = None

    tau57 = zeros((N, N, M, M))

    tau57 -= np.einsum("ka,kbij->ijab", tau31, u[o, v, o, o], optimize=True)

    tau81 = zeros((N, N, M, M))

    tau81 += 8 * np.einsum("ijab->ijab", tau57, optimize=True)

    tau57 = None

    tau60 = zeros((M, M, M, M))

    tau60 += np.einsum("ia,ibdc->abcd", tau31, u[o, v, v, v], optimize=True)

    tau64 -= 2 * np.einsum("bcad->abcd", tau60, optimize=True)

    tau60 = None

    tau74 = zeros((M, M))

    tau74 -= np.einsum("ic,iacb->ab", tau31, u[o, v, v, v], optimize=True)

    tau79 += 8 * np.einsum("ab->ab", tau74, optimize=True)

    tau74 = None

    tau82 = zeros((N, N, M, M))

    tau82 -= np.einsum("ic,abjc->ijab", tau31, u[v, v, o, v], optimize=True)

    tau98 = zeros((N, N, M, M))

    tau98 -= 8 * np.einsum("ijba->ijab", tau82, optimize=True)

    tau82 = None

    tau84 = zeros((N, N, N, N))

    tau84 += np.einsum("ia,kjla->ijkl", tau31, u[o, o, o, v], optimize=True)

    tau88 -= 2 * np.einsum("jikl->ijkl", tau84, optimize=True)

    tau84 = None

    tau89 = zeros((N, N, M, M))

    tau89 += np.einsum("ablk,kilj->ijab", t2, tau88, optimize=True)

    tau88 = None

    tau98 -= 2 * np.einsum("ijba->ijab", tau89, optimize=True)

    tau89 = None

    tau91 = zeros((N, N))

    tau91 -= np.einsum("ka,kija->ij", tau31, u[o, o, o, v], optimize=True)

    tau31 = None

    tau96 += 8 * np.einsum("ij->ij", tau91, optimize=True)

    tau91 = None

    tau41 = zeros((N, N, N, M))

    tau41 += np.einsum("bail,ljkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum("ak,kijb->ijab", l1, tau41, optimize=True)

    tau41 = None

    tau46 -= 2 * np.einsum("ijab->ijab", tau42, optimize=True)

    tau42 = None

    tau47 = zeros((N, N, M, M))

    tau47 += np.einsum("cbkj,kica->ijab", t2, tau46, optimize=True)

    tau46 = None

    tau56 -= 2 * np.einsum("jiab->ijab", tau47, optimize=True)

    tau47 = None

    tau52 = zeros((N, N, M, M))

    tau52 += np.einsum("ak,ibjk->ijab", l1, u[o, v, o, o], optimize=True)

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("ijab->ijab", tau52, optimize=True)

    tau52 = None

    tau53 = zeros((N, N, M, M))

    tau53 += np.einsum("ci,acjb->ijab", l1, u[v, v, o, v], optimize=True)

    tau54 += np.einsum("ijba->ijab", tau53, optimize=True)

    tau53 = None

    tau55 = zeros((N, N, M, M))

    tau55 += np.einsum("cbkj,kica->ijab", t2, tau54, optimize=True)

    tau54 = None

    tau56 -= 4 * np.einsum("jiba->ijab", tau55, optimize=True)

    tau55 = None

    r2 -= np.einsum("ijab->abij", tau56, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau56, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau56, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau56, optimize=True) / 4

    tau56 = None

    tau58 = zeros((N, M, M, M))

    tau58 -= np.einsum("adij,jbdc->iabc", t2, u[o, v, v, v], optimize=True)

    tau59 = zeros((M, M, M, M))

    tau59 += np.einsum("ai,ibcd->abcd", l1, tau58, optimize=True)

    tau58 = None

    tau64 -= 4 * np.einsum("abcd->abcd", tau59, optimize=True)

    tau59 = None

    tau61 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau62 = zeros((M, M, M, M))

    tau62 += np.einsum("eafb,ecfd->abcd", tau5, tau61, optimize=True)

    tau61 = None

    tau5 = None

    tau64 += np.einsum("abcd->abcd", tau62, optimize=True)

    tau62 = None

    tau65 = zeros((N, N, M, M))

    tau65 += np.einsum("dcij,cabd->ijab", t2, tau64, optimize=True)

    tau64 = None

    tau81 -= 2 * np.einsum("jiab->ijab", tau65, optimize=True)

    tau65 = None

    tau66 = zeros((N, N, M, M))

    tau66 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau70 -= 2 * np.einsum("jiba->ijab", tau66, optimize=True)

    tau77 += 2 * np.einsum("jiba->ijab", tau66, optimize=True)

    tau66 = None

    tau68 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum("abkl,ijkl->ijab", l2, tau68, optimize=True)

    tau68 = None

    tau70 -= np.einsum("jiba->ijab", tau69, optimize=True)

    tau71 = zeros((N, N, M, M))

    tau71 += np.einsum("cbkj,ikca->ijab", t2, tau70, optimize=True)

    tau70 = None

    tau72 = zeros((N, N, M, M))

    tau72 += np.einsum("cbkj,kica->ijab", t2, tau71, optimize=True)

    tau71 = None

    tau81 += 2 * np.einsum("jiba->ijab", tau72, optimize=True)

    tau72 = None

    tau94 += np.einsum("jiba->ijab", tau69, optimize=True)

    tau69 = None

    tau73 = zeros((M, M))

    tau73 += np.einsum("ci,acib->ab", l1, u[v, v, o, v], optimize=True)

    tau79 += 8 * np.einsum("ab->ab", tau73, optimize=True)

    tau73 = None

    tau77 -= 8 * np.einsum("aj,ib->ijab", l1, f[o, v], optimize=True)

    tau78 = zeros((M, M))

    tau78 += np.einsum("cbji,ijca->ab", t2, tau77, optimize=True)

    tau77 = None

    tau79 -= np.einsum("ba->ab", tau78, optimize=True)

    tau78 = None

    tau80 = zeros((N, N, M, M))

    tau80 += np.einsum("ac,cbij->ijab", tau79, t2, optimize=True)

    tau79 = None

    tau81 -= np.einsum("jiba->ijab", tau80, optimize=True)

    tau80 = None

    r2 -= np.einsum("jiab->abij", tau81, optimize=True) / 8

    r2 += np.einsum("jiba->abij", tau81, optimize=True) / 8

    tau81 = None

    tau90 = zeros((N, N))

    tau90 += np.einsum("ak,iajk->ij", l1, u[o, v, o, o], optimize=True)

    tau96 += 8 * np.einsum("ij->ij", tau90, optimize=True)

    tau90 = None

    tau94 -= 8 * np.einsum("bi,ja->ijab", l1, f[o, v], optimize=True)

    tau95 = zeros((N, N))

    tau95 += np.einsum("bakj,kiab->ij", t2, tau94, optimize=True)

    tau94 = None

    tau96 -= np.einsum("ij->ij", tau95, optimize=True)

    tau95 = None

    tau97 = zeros((N, N, M, M))

    tau97 += np.einsum("ki,abkj->ijab", tau96, t2, optimize=True)

    tau96 = None

    tau98 -= np.einsum("jiba->ijab", tau97, optimize=True)

    tau97 = None

    r2 -= np.einsum("ijba->abij", tau98, optimize=True) / 8

    r2 += np.einsum("jiba->abij", tau98, optimize=True) / 8

    tau98 = None

    tau100 += 4 * np.einsum("ai,jb->ijab", l1, f[o, v], optimize=True)

    tau101 -= np.einsum("balk,ijab->ijkl", t2, tau100, optimize=True)

    tau100 = None

    tau101 -= 4 * np.einsum("ai,jalk->ijkl", l1, u[o, v, o, o], optimize=True)

    r2 += np.einsum("balk,klji->abij", t2, tau101, optimize=True) / 4

    tau101 = None

    return r2
