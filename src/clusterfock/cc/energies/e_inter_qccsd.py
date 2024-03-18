import numpy as np
from clusterfock.cc.energies.e_inter_ccsd import td_energy_addition


def energy_intermediates_qccsd(t1, t2, l1, l2, u, f, o, v):
    e = ccsd_gs_energy(t1, t2, u, f, o, v)
    e += td_energy_addition(t1, t2, l1, l2, u, f, o, v)
    e += qccsd_energy_addition(t1, t2, l1, l2, u, f, o, v)
    return e


def ccsd_gs_energy(t1, t2, u, f, o, v):
    e = 0
    e += np.einsum("ia,ai->", f[o, v], t1, optimize=True)
    e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4
    e += np.einsum("ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True) / 2

    return e


def qccsd_energy_addition(t1, t2, l1, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N))

    tau0 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("ki,abjk->ijab", tau0, t2, optimize=True)

    tau40 = zeros((N, N))

    tau40 += np.einsum("kiba,jkba->ij", tau39, u[o, o, v, v], optimize=True)

    tau39 = None

    tau41 = zeros((N, M))

    tau41 += np.einsum("aj,ji->ia", t1, tau0, optimize=True)

    tau42 = zeros((N, M))

    tau42 += np.einsum("jb,ijab->ia", tau41, u[o, o, v, v], optimize=True)

    tau41 = None

    tau43 = zeros((N, N))

    tau43 += np.einsum("ai,ja->ij", t1, tau42, optimize=True)

    tau42 = None

    tau45 = zeros((N, M))

    tau45 += np.einsum("jk,ikja->ia", tau0, u[o, o, o, v], optimize=True)

    tau46 = zeros((N, N))

    tau46 += np.einsum("ai,ja->ij", t1, tau45, optimize=True)

    tau139 = zeros((N, M))

    tau139 += 4 * np.einsum("ia->ia", tau45, optimize=True)

    tau45 = None

    tau78 = zeros((N, N))

    tau78 += 2 * np.einsum("ij->ij", tau0, optimize=True)

    tau135 = zeros((N, N))

    tau135 += 4 * np.einsum("ij->ij", tau0, optimize=True)

    tau141 = zeros((N, N, N, N))

    tau141 += np.einsum("ik,jl->ijkl", tau0, tau0, optimize=True)

    e = 0

    e += np.einsum("ij,ij->", tau0, tau40, optimize=True) / 4

    tau40 = None

    e += np.einsum("ij,ij->", tau0, tau43, optimize=True) / 2

    tau43 = None

    e -= np.einsum("ij,ij->", tau0, tau46, optimize=True)

    tau46 = None

    tau1 = zeros((N, N, N, N))

    tau1 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau47 = zeros((N, N))

    tau47 += np.einsum("mlki,kjml->ij", tau1, u[o, o, o, o], optimize=True)

    e -= np.einsum("ij,ij->", tau0, tau47, optimize=True) / 4

    tau47 = None

    tau75 = zeros((N, N, N, M))

    tau75 -= np.einsum("al,ilkj->ijka", t1, tau1, optimize=True)

    tau79 = zeros((N, N, N, M))

    tau79 -= np.einsum("ikja->ijka", tau75, optimize=True)

    tau106 = zeros((N, N, N, M))

    tau106 -= np.einsum("ikja->ijka", tau75, optimize=True)

    tau75 = None

    tau84 = zeros((N, M))

    tau84 += np.einsum("ijlk,lkja->ia", tau1, u[o, o, o, v], optimize=True)

    tau93 = zeros((N, N, N, N))

    tau93 += 2 * np.einsum("imnj,nkml->ijkl", tau1, u[o, o, o, o], optimize=True)

    tau97 = zeros((N, N, N, M))

    tau97 -= np.einsum("ilmj,kmla->ijka", tau1, u[o, o, o, v], optimize=True)

    tau112 = zeros((N, N, N, M))

    tau112 += 4 * np.einsum("ikja->ijka", tau97, optimize=True)

    tau122 = zeros((N, N, N, M))

    tau122 += np.einsum("kija->ijka", tau97, optimize=True)

    tau97 = None

    tau116 = zeros((N, M))

    tau116 += 2 * np.einsum("lkji,jalk->ia", tau1, u[o, v, o, o], optimize=True)

    tau117 = zeros((N, N, M, M))

    tau117 -= np.einsum("balk,lkij->ijab", t2, tau1, optimize=True)

    tau118 = zeros((N, N, M, M))

    tau118 -= np.einsum("ijba->ijab", tau117, optimize=True)

    tau123 = zeros((N, N, M, M))

    tau123 -= np.einsum("ijba->ijab", tau117, optimize=True)

    tau117 = None

    tau127 = zeros((N, N, N, N))

    tau127 += np.einsum("miln,njkm->ijkl", tau1, tau1, optimize=True)

    tau128 = zeros((N, N, N, N))

    tau128 += np.einsum("ijkl->ijkl", tau127, optimize=True)

    tau127 = None

    tau131 = zeros((N, N, M, M))

    tau131 -= np.einsum("ijlk,lkab->ijab", tau1, u[o, o, v, v], optimize=True)

    tau2 = zeros((N, N, N, N))

    tau2 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau3 = zeros((N, N))

    tau3 -= np.einsum("lkmi,kmjl->ij", tau1, tau2, optimize=True)

    e += np.einsum("ij,ij->", tau0, tau3, optimize=True) / 2

    tau3 = None

    tau71 = zeros((N, N, N, N))

    tau71 -= 2 * np.einsum("kjil->ijkl", tau2, optimize=True)

    tau109 = zeros((N, N, N, N))

    tau109 -= 4 * np.einsum("kjil->ijkl", tau2, optimize=True)

    tau134 = zeros((N, N, N, N))

    tau134 -= 4 * np.einsum("ljik->ijkl", tau2, optimize=True)

    tau2 = None

    tau4 = zeros((N, N, N, M))

    tau4 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau5 = zeros((N, N, N, M))

    tau5 += np.einsum("balk,lijb->ijka", t2, tau4, optimize=True)

    tau6 = zeros((N, N))

    tau6 -= np.einsum("lkia,kjla->ij", tau5, u[o, o, o, v], optimize=True)

    e -= np.einsum("ij,ij->", tau0, tau6, optimize=True)

    tau6 = None

    tau106 -= 2 * np.einsum("ijka->ijka", tau5, optimize=True)

    tau113 = zeros((N, N, M, M))

    tau113 += 8 * np.einsum("bk,kjia->ijab", t1, tau5, optimize=True)

    tau116 -= 8 * np.einsum("kjib,jakb->ia", tau5, u[o, v, o, v], optimize=True)

    tau122 -= np.einsum("klib,jlab->ijka", tau5, u[o, o, v, v], optimize=True)

    tau129 = zeros((N, N, N, M))

    tau129 += 4 * np.einsum("mikl,ljma->ijka", tau1, tau5, optimize=True)

    tau7 = zeros((N, M))

    tau7 += np.einsum("bakj,kjib->ia", t2, tau4, optimize=True)

    tau8 = zeros((N, N))

    tau8 += np.einsum("ka,kija->ij", tau7, u[o, o, o, v], optimize=True)

    e -= np.einsum("ij,ji->", tau0, tau8, optimize=True) / 2

    tau8 = None

    tau25 = zeros((N, M))

    tau25 -= np.einsum("jb,ijab->ia", tau7, u[o, o, v, v], optimize=True)

    tau26 = zeros((N, N))

    tau26 += np.einsum("ai,ja->ij", t1, tau25, optimize=True)

    e -= np.einsum("ij,ij->", tau0, tau26, optimize=True) / 2

    tau26 = None

    tau124 = zeros((N, M))

    tau124 += np.einsum("ia->ia", tau25, optimize=True)

    tau25 = None

    tau88 = zeros((N, M))

    tau88 += np.einsum("ia->ia", tau7, optimize=True)

    tau113 -= 4 * np.einsum("bi,ja->ijab", t1, tau7, optimize=True)

    tau116 += 4 * np.einsum("jb,jaib->ia", tau7, u[o, v, o, v], optimize=True)

    tau137 = zeros((N, M))

    tau137 += np.einsum("ia->ia", tau7, optimize=True)

    tau138 = zeros((N, M))

    tau138 += 2 * np.einsum("ia->ia", tau7, optimize=True)

    tau7 = None

    tau62 = zeros((N, N, N, N))

    tau62 += np.einsum("ak,ijla->ijkl", t1, tau4, optimize=True)

    tau84 -= 2 * np.einsum("iljk,kjla->ia", tau62, u[o, o, o, v], optimize=True)

    tau62 = None

    tau126 = zeros((N, M, M, M))

    tau126 += np.einsum("bckj,kjia->iabc", t2, tau4, optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("acik,kjcb->ijab", l2, tau9, optimize=True)

    tau11 = zeros((N, N))

    tau11 -= np.einsum("baki,kjba->ij", t2, tau10, optimize=True)

    tau10 = None

    e += np.einsum("ij,ij->", tau0, tau11, optimize=True)

    tau11 = None

    tau80 = zeros((N, N, M, M))

    tau80 -= np.einsum("jiab->ijab", tau9, optimize=True)

    tau99 = zeros((N, N, M, M))

    tau99 += 2 * np.einsum("ijab->ijab", tau9, optimize=True)

    tau133 = zeros((N, N, M, M))

    tau133 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau9 = None

    tau12 = zeros((M, M))

    tau12 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau13 = zeros((N, M))

    tau13 += np.einsum("bc,ibac->ia", tau12, u[o, v, v, v], optimize=True)

    tau14 = zeros((N, N))

    tau14 += np.einsum("ai,ja->ij", t1, tau13, optimize=True)

    e -= np.einsum("ij,ij->", tau0, tau14, optimize=True) / 2

    tau14 = None

    tau124 += 2 * np.einsum("ia->ia", tau13, optimize=True)

    tau13 = None

    tau125 = zeros((N, N, N, M))

    tau125 -= np.einsum("jb,baki->ijka", tau124, t2, optimize=True)

    tau124 = None

    tau19 = zeros((N, N, M, M))

    tau19 -= np.einsum("cb,caij->ijab", tau12, t2, optimize=True)

    tau20 = zeros((N, N))

    tau20 -= np.einsum("ikba,kjba->ij", tau19, u[o, o, v, v], optimize=True)

    e += np.einsum("ij,ij->", tau0, tau20, optimize=True) / 2

    tau20 = None

    tau113 += 4 * np.einsum("ijab->ijab", tau19, optimize=True)

    tau118 -= 2 * np.einsum("ijab->ijab", tau19, optimize=True)

    tau19 = None

    tau36 = zeros((N, N))

    tau36 += np.einsum("ab,iajb->ij", tau12, u[o, v, o, v], optimize=True)

    e -= np.einsum("ij,ji->", tau0, tau36, optimize=True) / 2

    tau36 = None

    tau85 = zeros((M, M))

    tau85 += np.einsum("ab->ab", tau12, optimize=True)

    tau114 = zeros((N, M, M, M))

    tau114 -= np.einsum("bi,ac->iabc", t1, tau12, optimize=True)

    tau116 += 4 * np.einsum("bc,abic->ia", tau12, u[v, v, o, v], optimize=True)

    tau131 += np.einsum("ac,jibc->ijab", tau12, u[o, o, v, v], optimize=True)

    tau132 = zeros((M, M))

    tau132 -= 2 * np.einsum("cd,cadb->ab", tau12, u[v, v, v, v], optimize=True)

    tau15 = zeros((N, N))

    tau15 -= np.einsum("ak,iakj->ij", l1, u[o, v, o, o], optimize=True)

    tau140 = zeros((N, N))

    tau140 += 8 * np.einsum("ij->ij", tau15, optimize=True)

    e -= np.einsum("ij,ji->", tau0, tau15, optimize=True)

    tau15 = None

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("caki,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("caki,kjcb->ijab", l2, tau16, optimize=True)

    tau18 = zeros((N, N))

    tau18 += np.einsum("baki,kjba->ij", t2, tau17, optimize=True)

    tau17 = None

    e -= np.einsum("ij,ij->", tau0, tau18, optimize=True)

    tau18 = None

    tau58 = zeros((N, N, M, M))

    tau58 += np.einsum("ijab->ijab", tau16, optimize=True)

    tau21 = zeros((N, N, N, N))

    tau21 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau22 = zeros((N, N))

    tau22 += np.einsum("mlki,mlkj->ij", tau1, tau21, optimize=True)

    e -= np.einsum("ij,ij->", tau0, tau22, optimize=True) / 8

    tau22 = None

    tau93 += np.einsum("minl,mjnk->ijkl", tau1, tau21, optimize=True)

    tau96 = zeros((N, N, N, N))

    tau96 += np.einsum("lkji->ijkl", tau21, optimize=True)

    tau23 = zeros((N, N, N, M))

    tau23 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau24 = zeros((N, N))

    tau24 -= np.einsum("lkja,lkia->ij", tau23, tau5, optimize=True)

    tau5 = None

    e += np.einsum("ij,ij->", tau0, tau24, optimize=True)

    tau24 = None

    tau27 = zeros((N, N, N, N))

    tau27 += np.einsum("ai,jkla->ijkl", t1, tau23, optimize=True)

    tau28 = zeros((N, N))

    tau28 -= np.einsum("lkmi,lkjm->ij", tau1, tau27, optimize=True)

    tau27 = None

    e -= np.einsum("ij,ij->", tau0, tau28, optimize=True) / 4

    tau28 = None

    tau64 = zeros((N, N, N, M))

    tau64 -= np.einsum("ikja->ijka", tau23, optimize=True)

    tau104 = zeros((N, N, N, M))

    tau104 += np.einsum("balj,ilkb->ijka", t2, tau23, optimize=True)

    tau105 = zeros((N, N, N, M))

    tau105 -= 2 * np.einsum("ijka->ijka", tau104, optimize=True)

    tau105 += 2 * np.einsum("jika->ijka", tau104, optimize=True)

    tau104 = None

    tau107 = zeros((N, N, N, M))

    tau107 += np.einsum("kjia->ijka", tau23, optimize=True)

    tau29 = zeros((N, N, M, M))

    tau29 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau30 = zeros((N, N))

    tau30 += np.einsum("baik,kjba->ij", t2, tau29, optimize=True)

    e -= np.einsum("ij,ij->", tau0, tau30, optimize=True) / 2

    tau30 = None

    tau136 = zeros((N, N, M, M))

    tau136 -= 4 * np.einsum("ijba->ijab", tau29, optimize=True)

    tau29 = None

    tau31 = zeros((M, M))

    tau31 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau32 = zeros((N, M))

    tau32 += np.einsum("bc,ibac->ia", tau31, u[o, v, v, v], optimize=True)

    tau33 = zeros((N, N))

    tau33 += np.einsum("ai,ja->ij", t1, tau32, optimize=True)

    tau32 = None

    e -= np.einsum("ij,ij->", tau0, tau33, optimize=True)

    tau33 = None

    tau51 = zeros((N, N))

    tau51 += np.einsum("ab,iajb->ij", tau31, u[o, v, o, v], optimize=True)

    e -= np.einsum("ij,ji->", tau0, tau51, optimize=True)

    tau51 = None

    tau85 += 2 * np.einsum("ab->ab", tau31, optimize=True)

    tau86 = zeros((N, M))

    tau86 += np.einsum("bc,ibca->ia", tau85, u[o, v, v, v], optimize=True)

    tau89 = zeros((N, M))

    tau89 -= np.einsum("ia->ia", tau86, optimize=True)

    tau139 -= 2 * np.einsum("ia->ia", tau86, optimize=True)

    tau86 = None

    tau136 -= 4 * np.einsum("ac,jicb->ijab", tau85, u[o, o, v, v], optimize=True)

    tau140 += 4 * np.einsum("ab,iajb->ij", tau85, u[o, v, o, v], optimize=True)

    tau85 = None

    tau115 = zeros((N, M, M, M))

    tau115 += 2 * np.einsum("ci,ab->iabc", t1, tau31, optimize=True)

    tau31 = None

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau35 = zeros((N, N))

    tau35 -= np.einsum("kiba,jbka->ij", tau34, u[o, v, o, v], optimize=True)

    e -= np.einsum("ij,ij->", tau0, tau35, optimize=True)

    tau35 = None

    tau48 = zeros((N, N, N, N))

    tau48 += np.einsum("ikab,jlba->ijkl", tau34, tau34, optimize=True)

    tau128 += 4 * np.einsum("ijlk->ijkl", tau48, optimize=True)

    tau141 -= np.einsum("ijkl->ijkl", tau48, optimize=True)

    e += np.einsum("lkji,ijkl->", tau141, u[o, o, o, o], optimize=True) / 2

    tau141 = None

    e -= np.einsum("jilk,ijkl->", tau21, tau48, optimize=True) / 4

    tau48 = None

    tau21 = None

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("ikcb,kjac->ijab", tau34, tau34, optimize=True)

    tau90 = zeros((N, N, M, M))

    tau90 -= 16 * np.einsum("jkbc,ikac->ijab", tau54, u[o, o, v, v], optimize=True)

    tau119 = zeros((N, N, M, M))

    tau119 -= 4 * np.einsum("ijab->ijab", tau54, optimize=True)

    tau54 = None

    tau55 = zeros((M, M, M, M))

    tau55 += 4 * np.einsum("ijab,jcid->abcd", tau34, u[o, v, o, v], optimize=True)

    tau55 += 4 * np.einsum("ijbd,ijac->abcd", tau16, tau34, optimize=True)

    tau16 = None

    tau61 = zeros((N, N, M, M))

    tau61 += 4 * np.einsum("kiac,jbkc->ijab", tau34, u[o, v, o, v], optimize=True)

    tau81 = zeros((N, N, M, M))

    tau81 += np.einsum("ijab->ijab", tau34, optimize=True)

    tau91 = zeros((N, N, M, M))

    tau91 += np.einsum("caki,kjcb->ijab", t2, tau34, optimize=True)

    tau92 = zeros((N, N, N, N))

    tau92 -= np.einsum("ijab,klba->ijkl", tau91, u[o, o, v, v], optimize=True)

    tau93 -= 2 * np.einsum("ljik->ijkl", tau92, optimize=True)

    tau125 -= 2 * np.einsum("al,ikjl->ijka", t1, tau92, optimize=True)

    tau92 = None

    tau95 = zeros((N, N, N, N))

    tau95 -= np.einsum("baji,lkab->ijkl", l2, tau91, optimize=True)

    tau128 -= 2 * np.einsum("ijlk->ijkl", tau95, optimize=True)

    tau129 += np.einsum("al,ilkj->ijka", t1, tau128, optimize=True)

    tau130 = zeros((N, M))

    tau130 -= 2 * np.einsum("iljk,jkla->ia", tau128, u[o, o, o, v], optimize=True)

    tau128 = None

    tau113 += 8 * np.einsum("ijba->ijab", tau91, optimize=True)

    tau118 += 4 * np.einsum("ijab->ijab", tau91, optimize=True)

    tau119 += np.einsum("caki,jkcb->ijab", l2, tau118, optimize=True)

    tau118 = None

    tau123 += 4 * np.einsum("ijab->ijab", tau91, optimize=True)

    tau125 -= 2 * np.einsum("ilba,ljkb->ijka", tau123, u[o, o, o, v], optimize=True)

    tau129 -= 2 * np.einsum("klba,lijb->ijka", tau123, tau4, optimize=True)

    tau123 = None

    tau125 -= 4 * np.einsum("kicb,jacb->ijka", tau91, u[o, v, v, v], optimize=True)

    tau91 = None

    tau114 += 2 * np.einsum("bj,jiac->iabc", t1, tau34, optimize=True)

    tau116 -= 4 * np.einsum("ibdc,bacd->ia", tau114, u[v, v, v, v], optimize=True)

    tau114 = None

    tau119 -= 2 * np.einsum("kijl,lkab->ijab", tau1, tau34, optimize=True)

    tau121 = zeros((N, M, M, M))

    tau121 += 4 * np.einsum("jkab,ikjc->iabc", tau34, u[o, o, o, v], optimize=True)

    tau122 += 2 * np.einsum("kibc,jbac->ijka", tau34, u[o, v, v, v], optimize=True)

    tau129 += 4 * np.einsum("jbac,ikcb->ijka", tau126, tau34, optimize=True)

    tau126 = None

    tau130 -= np.einsum("ikjb,jkba->ia", tau129, u[o, o, v, v], optimize=True)

    tau129 = None

    tau37 = zeros((N, N, N, M))

    tau37 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau38 = zeros((N, N))

    tau38 += np.einsum("klja,klia->ij", tau23, tau37, optimize=True)

    tau23 = None

    e += np.einsum("ij,ij->", tau0, tau38, optimize=True)

    tau38 = None

    tau44 = zeros((N, N))

    tau44 -= np.einsum("klia,ljka->ij", tau37, u[o, o, o, v], optimize=True)

    e += np.einsum("ij,ij->", tau0, tau44, optimize=True)

    tau44 = None

    tau79 -= 2 * np.einsum("ikja->ijka", tau37, optimize=True)

    tau106 -= np.einsum("ikja->ijka", tau37, optimize=True)

    tau37 = None

    tau112 += 4 * np.einsum("ilkb,ljba->ijka", tau106, u[o, o, v, v], optimize=True)

    tau106 = None

    tau49 = zeros((N, M))

    tau49 += np.einsum("bj,ibja->ia", l1, u[o, v, o, v], optimize=True)

    tau50 = zeros((N, N))

    tau50 += np.einsum("ai,ja->ij", t1, tau49, optimize=True)

    e += np.einsum("ij,ij->", tau0, tau50, optimize=True)

    tau0 = None

    tau50 = None

    tau89 -= 2 * np.einsum("ia->ia", tau49, optimize=True)

    tau139 -= 4 * np.einsum("ia->ia", tau49, optimize=True)

    tau49 = None

    tau52 = zeros((M, M, M, M))

    tau52 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau53 = zeros((M, M, M, M))

    tau53 -= np.einsum("bedf,face->abcd", tau52, tau52, optimize=True)

    tau90 -= np.einsum("abcd,ijdc->ijab", tau53, u[o, o, v, v], optimize=True)

    tau53 = None

    tau55 -= np.einsum("aefb,cefd->abcd", tau52, u[v, v, v, v], optimize=True)

    tau90 += 2 * np.einsum("cdji,bdca->ijab", l2, tau55, optimize=True)

    tau55 = None

    tau119 += 2 * np.einsum("ijcd,adbc->ijab", tau34, tau52, optimize=True)

    tau130 -= 2 * np.einsum("ijbc,jbca->ia", tau119, u[o, v, v, v], optimize=True)

    tau119 = None

    tau56 = zeros((M, M, M, M))

    tau56 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau56 += np.einsum("baji,jidc->abcd", t2, u[o, o, v, v], optimize=True)

    tau61 += np.einsum("jicd,cbda->ijab", tau34, tau56, optimize=True)

    tau56 = None

    tau57 = zeros((N, N, M, M))

    tau57 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau60 = zeros((N, N, M, M))

    tau60 -= np.einsum("jiba->ijab", tau57, optimize=True)

    tau131 -= 2 * np.einsum("jiba->ijab", tau57, optimize=True)

    tau57 = None

    tau58 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau59 = zeros((N, N, M, M))

    tau59 += np.einsum("caki,kjcb->ijab", l2, tau58, optimize=True)

    tau60 -= 4 * np.einsum("ijab->ijab", tau59, optimize=True)

    tau61 -= np.einsum("cbki,jkca->ijab", t2, tau60, optimize=True)

    tau60 = None

    tau90 += 4 * np.einsum("caki,kjbc->ijab", l2, tau61, optimize=True)

    tau61 = None

    tau94 = zeros((N, N, N, N))

    tau94 -= np.einsum("abkj,ilba->ijkl", t2, tau59, optimize=True)

    e += np.einsum("ijkl,lijk->", tau1, tau94, optimize=True) / 4

    tau94 = None

    tau131 -= 8 * np.einsum("ijab->ijab", tau59, optimize=True)

    tau59 = None

    tau132 -= np.einsum("caij,ijcb->ab", t2, tau131, optimize=True)

    tau131 = None

    e -= np.einsum("ab,ab->", tau12, tau132, optimize=True) / 16

    tau132 = None

    tau93 -= 8 * np.einsum("ijab,lkab->ijkl", tau34, tau58, optimize=True)

    tau58 = None

    e -= np.einsum("ijkl,likj->", tau1, tau93, optimize=True) / 16

    tau93 = None

    tau63 = zeros((N, N, N, M))

    tau63 += np.einsum("bi,jakb->ijka", t1, u[o, v, o, v], optimize=True)

    tau72 = zeros((N, N, N, M))

    tau72 -= 4 * np.einsum("kija->ijka", tau63, optimize=True)

    tau103 = zeros((N, N, N, M))

    tau103 += np.einsum("ijka->ijka", tau63, optimize=True)

    tau116 += 4 * np.einsum("kjli,jlka->ia", tau1, tau63, optimize=True)

    tau63 = None

    tau64 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau72 += 4 * np.einsum("balj,klib->ijka", t2, tau64, optimize=True)

    tau110 = zeros((N, N, N, M))

    tau110 += np.einsum("liab,ljkb->ijka", tau34, tau64, optimize=True)

    tau34 = None

    tau112 += 4 * np.einsum("kjia->ijka", tau110, optimize=True)

    tau122 -= 2 * np.einsum("ikja->ijka", tau110, optimize=True)

    tau110 = None

    tau111 = zeros((N, N, N, M))

    tau111 += np.einsum("ab,ijkb->ijka", tau12, tau64, optimize=True)

    tau64 = None

    tau112 -= 2 * np.einsum("kjia->ijka", tau111, optimize=True)

    tau122 += np.einsum("ikja->ijka", tau111, optimize=True)

    tau111 = None

    tau125 += 4 * np.einsum("bali,kjlb->ijka", t2, tau122, optimize=True)

    tau122 = None

    tau65 = zeros((N, N, M, M))

    tau65 -= np.einsum("baji->ijab", t2, optimize=True)

    tau65 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau108 = zeros((N, N, N, N))

    tau108 += np.einsum("klab,ijab->ijkl", tau65, u[o, o, v, v], optimize=True)

    tau109 += np.einsum("jilk->ijkl", tau108, optimize=True)

    tau134 -= np.einsum("jilk->ijkl", tau108, optimize=True)

    tau108 = None

    tau66 = zeros((N, M, M, M))

    tau66 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau67 = zeros((N, M, M, M))

    tau67 += np.einsum("iacb->iabc", tau66, optimize=True)

    tau120 = zeros((N, M, M, M))

    tau120 += np.einsum("iacb->iabc", tau66, optimize=True)

    tau66 = None

    tau67 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau72 -= np.einsum("kjbc,iabc->ijka", tau65, tau67, optimize=True)

    tau65 = None

    tau68 = zeros((N, M))

    tau68 += np.einsum("bj,jiba->ia", t1, u[o, o, v, v], optimize=True)

    tau69 = zeros((N, M))

    tau69 += np.einsum("ia->ia", tau68, optimize=True)

    tau68 = None

    tau69 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau70 = zeros((N, N, N, M))

    tau70 += np.einsum("kb,baij->ijka", tau69, t2, optimize=True)

    tau72 += 2 * np.einsum("kjia->ijka", tau70, optimize=True)

    tau105 += 2 * np.einsum("jika->ijka", tau70, optimize=True)

    tau70 = None

    tau100 = zeros((N, M, M, M))

    tau100 -= np.einsum("jc,baji->iabc", tau69, t2, optimize=True)

    tau69 = None

    tau71 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau72 -= 2 * np.einsum("al,likj->ijka", t1, tau71, optimize=True)

    tau71 = None

    tau72 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau84 += np.einsum("bajk,ijkb->ia", l2, tau72, optimize=True)

    tau72 = None

    tau73 = zeros((N, M, M, M))

    tau73 -= np.einsum("di,abdc->iabc", t1, u[v, v, v, v], optimize=True)

    tau74 = zeros((N, M, M, M))

    tau74 += np.einsum("ibac->iabc", tau73, optimize=True)

    tau100 += np.einsum("ibac->iabc", tau73, optimize=True)

    tau73 = None

    tau74 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau84 -= 2 * np.einsum("bcji,jbca->ia", l2, tau74, optimize=True)

    tau74 = None

    tau76 = zeros((N, N, M, M))

    tau76 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau76 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau79 += 2 * np.einsum("likb,ljba->ijka", tau4, tau76, optimize=True)

    tau99 += np.einsum("kica,kjcb->ijab", tau76, u[o, o, v, v], optimize=True)

    tau76 = None

    tau77 = zeros((N, N))

    tau77 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau78 += np.einsum("ij->ij", tau77, optimize=True)

    tau79 += 2 * np.einsum("aj,ik->ijka", t1, tau78, optimize=True)

    tau84 += np.einsum("ijkb,jkba->ia", tau79, u[o, o, v, v], optimize=True)

    tau79 = None

    tau84 += 2 * np.einsum("ja,ij->ia", f[o, v], tau78, optimize=True)

    tau87 = zeros((N, M))

    tau87 += np.einsum("aj,ji->ia", t1, tau78, optimize=True)

    tau88 += np.einsum("ia->ia", tau87, optimize=True)

    tau137 += np.einsum("ia->ia", tau87, optimize=True)

    tau87 = None

    tau140 += 4 * np.einsum("ka,kija->ij", tau137, u[o, o, o, v], optimize=True)

    tau137 = None

    tau89 -= np.einsum("kj,jika->ia", tau78, u[o, o, o, v], optimize=True)

    tau78 = None

    tau135 += np.einsum("ij->ij", tau77, optimize=True)

    tau136 -= np.einsum("ik,kjba->ijab", tau135, u[o, o, v, v], optimize=True)

    tau138 += np.einsum("aj,ji->ia", t1, tau135, optimize=True)

    tau139 -= np.einsum("jb,jiba->ia", tau138, u[o, o, v, v], optimize=True)

    tau138 = None

    tau140 += 2 * np.einsum("aj,ia->ij", t1, tau139, optimize=True)

    tau139 = None

    tau140 -= 2 * np.einsum("lk,kilj->ij", tau135, u[o, o, o, o], optimize=True)

    tau135 = None

    tau80 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau84 -= 4 * np.einsum("kijb,jkba->ia", tau4, tau80, optimize=True)

    tau80 = None

    tau4 = None

    tau81 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau84 -= 4 * np.einsum("ijbc,jbca->ia", tau81, u[o, v, v, v], optimize=True)

    tau112 -= 8 * np.einsum("ikbc,jbca->ijka", tau81, u[o, v, v, v], optimize=True)

    tau116 += 8 * np.einsum("jibc,bajc->ia", tau81, u[v, v, o, v], optimize=True)

    tau81 = None

    tau82 = zeros((N, N, M, M))

    tau82 += np.einsum("baji->ijab", t2, optimize=True)

    tau82 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau83 = zeros((N, N))

    tau83 += np.einsum("kjab,kiab->ij", tau82, u[o, o, v, v], optimize=True)

    tau105 += np.einsum("kabc,jibc->ijka", tau67, tau82, optimize=True)

    tau67 = None

    tau113 -= np.einsum("klji,klba->ijab", tau1, tau82, optimize=True)

    tau82 = None

    tau116 += np.einsum("ijbc,jabc->ia", tau113, u[o, v, v, v], optimize=True)

    tau113 = None

    tau83 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    tau83 += 2 * np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau83 -= 2 * np.einsum("ak,kija->ij", t1, u[o, o, o, v], optimize=True)

    tau84 += 2 * np.einsum("aj,ij->ia", l1, tau83, optimize=True)

    tau83 = None

    tau84 -= 4 * np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    tau90 += 4 * np.einsum("ai,jb->ijab", l1, tau84, optimize=True)

    tau84 = None

    tau88 -= np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau89 -= np.einsum("jb,jiba->ia", tau88, u[o, o, v, v], optimize=True)

    tau88 = None

    tau90 -= 8 * np.einsum("bj,ia->ijab", l1, tau89, optimize=True)

    tau89 = None

    e -= np.einsum("abij,ijab->", t2, tau90, optimize=True) / 16

    tau90 = None

    tau96 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    e -= np.einsum("ijkl,lkij->", tau95, tau96, optimize=True) / 8

    tau96 = None

    tau95 = None

    tau98 = zeros((N, N, M, M))

    tau98 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau99 += np.einsum("jiab->ijab", tau98, optimize=True)

    tau133 += np.einsum("jiab->ijab", tau98, optimize=True)

    tau98 = None

    tau99 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau100 -= np.einsum("bj,ijac->iabc", t1, tau99, optimize=True)

    tau99 = None

    tau100 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau100 -= 2 * np.einsum("dbji,jadc->iabc", t2, u[o, v, v, v], optimize=True)

    tau112 -= 2 * np.einsum("bcji,kbca->ijka", l2, tau100, optimize=True)

    tau100 = None

    tau101 = zeros((N, N, M, M))

    tau101 += np.einsum("baji->ijab", t2, optimize=True)

    tau101 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau102 = zeros((N, N, N, M))

    tau102 += np.einsum("lkba,lijb->ijka", tau101, u[o, o, o, v], optimize=True)

    tau103 -= np.einsum("jkia->ijka", tau102, optimize=True)

    tau102 = None

    tau105 += 2 * np.einsum("ikja->ijka", tau103, optimize=True)

    tau105 -= 2 * np.einsum("jkia->ijka", tau103, optimize=True)

    tau103 = None

    tau133 += np.einsum("kica,kjcb->ijab", tau101, u[o, o, v, v], optimize=True)

    tau101 = None

    tau105 += 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau105 += 2 * np.einsum("al,lkji->ijka", t1, u[o, o, o, o], optimize=True)

    tau112 += 4 * np.einsum("balj,lkib->ijka", l2, tau105, optimize=True)

    tau105 = None

    tau107 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau112 -= np.einsum("jilm,lmka->ijka", tau1, tau107, optimize=True)

    tau1 = None

    tau107 = None

    tau109 -= 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau112 += np.einsum("al,jilk->ijka", l1, tau109, optimize=True)

    tau109 = None

    tau112 -= 8 * np.einsum("bi,jbka->ijka", l1, u[o, v, o, v], optimize=True)

    tau116 -= np.einsum("bajk,kjib->ia", t2, tau112, optimize=True)

    tau112 = None

    tau115 += np.einsum("aj,cbij->iabc", l1, t2, optimize=True)

    tau116 += 2 * np.einsum("ibcd,bacd->ia", tau115, u[v, v, v, v], optimize=True)

    tau115 = None

    tau116 += 4 * np.einsum("bj,abij->ia", l1, u[v, v, o, o], optimize=True)

    e += np.einsum("ai,ia->", l1, tau116, optimize=True) / 8

    tau116 = None

    tau120 -= 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau121 -= np.einsum("idec,daeb->iabc", tau120, tau52, optimize=True)

    tau52 = None

    tau121 -= np.einsum("ad,ibdc->iabc", tau12, tau120, optimize=True)

    tau120 = None

    tau12 = None

    tau125 += np.einsum("bcki,jcab->ijka", t2, tau121, optimize=True)

    tau121 = None

    tau130 += np.einsum("bajk,jikb->ia", l2, tau125, optimize=True)

    tau125 = None

    e += np.einsum("ai,ia->", t1, tau130, optimize=True) / 8

    tau130 = None

    tau133 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau136 += 8 * np.einsum("cbki,kjca->ijab", l2, tau133, optimize=True)

    tau133 = None

    tau134 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau136 -= np.einsum("bakl,jikl->ijab", l2, tau134, optimize=True)

    tau134 = None

    tau136 -= 8 * np.einsum("ak,jikb->ijab", l1, u[o, o, o, v], optimize=True)

    tau140 += np.einsum("abkj,kiba->ij", t2, tau136, optimize=True)

    tau136 = None

    e -= np.einsum("ij,ji->", tau140, tau77, optimize=True) / 16

    tau140 = None

    tau77 = None

    return e
