import numpy as np


def energy_intermediates_qccsd(t1, t2, l1, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau4 = zeros((N, N, N, N))

    tau4 += np.einsum("ikab,jlba->ijkl", tau0, tau0, optimize=True)

    tau32 = zeros((N, N, N, N))

    tau32 += 4 * np.einsum("ijlk->ijkl", tau4, optimize=True)

    tau86 = zeros((N, N, N, N))

    tau86 += 4 * np.einsum("ijkl->ijkl", tau4, optimize=True)

    tau99 = zeros((N, N, N, N))

    tau99 += 4 * np.einsum("ijkl->ijkl", tau4, optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("cbkj,kica->ijab", t2, tau0, optimize=True)

    tau7 = zeros((N, N, N, N))

    tau7 -= np.einsum("baij,klab->ijkl", l2, tau6, optimize=True)

    e = 0

    e -= np.einsum("lkij,jilk->", tau7, u[o, o, o, o], optimize=True) / 4

    tau29 = zeros((N, N, M, M))

    tau29 += 2 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau64 = zeros((N, N, M, M))

    tau64 += 4 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau92 = zeros((N, N, M, M))

    tau92 += 4 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau96 = zeros((N, N, M, M))

    tau96 += 2 * np.einsum("ijba->ijab", tau6, optimize=True)

    e += np.einsum("jiba,jiba->", tau6, u[o, o, v, v], optimize=True) / 2

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("ikcb,kjac->ijab", tau0, tau0, optimize=True)

    tau98 = zeros((N, N, M, M))

    tau98 += 2 * np.einsum("ijab->ijab", tau9, optimize=True)

    e -= np.einsum("ijab,jaib->", tau9, u[o, v, o, v], optimize=True)

    tau9 = None

    tau35 = zeros((N, N, N, M))

    tau35 += np.einsum("liab,jklb->ijka", tau0, u[o, o, o, v], optimize=True)

    tau41 = zeros((N, N, N, M))

    tau41 += 4 * np.einsum("kija->ijka", tau35, optimize=True)

    tau83 = zeros((N, N, N, M))

    tau83 += np.einsum("ikja->ijka", tau35, optimize=True)

    tau125 = zeros((N, N, N, M))

    tau125 += 4 * np.einsum("kija->ijka", tau35, optimize=True)

    tau35 = None

    tau36 = zeros((N, N, N, M))

    tau36 += np.einsum("ijbc,kbac->ijka", tau0, u[o, v, v, v], optimize=True)

    tau41 -= 8 * np.einsum("jkia->ijka", tau36, optimize=True)

    tau83 += np.einsum("kija->ijka", tau36, optimize=True)

    tau87 = zeros((N, N, N, M))

    tau87 += 2 * np.einsum("ijka->ijka", tau36, optimize=True)

    tau125 += 4 * np.einsum("ikja->ijka", tau36, optimize=True)

    tau36 = None

    tau45 = zeros((N, N, N, M))

    tau45 += np.einsum("bj,ikba->ijka", t1, tau0, optimize=True)

    tau53 = zeros((N, N, M, M))

    tau53 -= 4 * np.einsum("ikla,lkjb->ijab", tau45, u[o, o, o, v], optimize=True)

    tau74 = zeros((N, M, M, M))

    tau74 -= 2 * np.einsum("ijkb,kjac->iabc", tau45, u[o, o, v, v], optimize=True)

    tau45 = None

    tau49 = zeros((N, N, N, M, M, M))

    tau49 += 4 * np.einsum("kibd,jcda->ijkabc", tau0, u[o, v, v, v], optimize=True)

    tau71 = zeros((N, M, M, M))

    tau71 -= np.einsum("jkab,kijc->iabc", tau0, u[o, o, o, v], optimize=True)

    tau74 += 2 * np.einsum("iabc->iabc", tau71, optimize=True)

    tau82 = zeros((N, M, M, M))

    tau82 += 2 * np.einsum("iabc->iabc", tau71, optimize=True)

    tau101 = zeros((N, M, M, M))

    tau101 -= 2 * np.einsum("icab->iabc", tau71, optimize=True)

    tau71 = None

    tau73 = zeros((N, M, M, M))

    tau73 += np.einsum("ijda,jdbc->iabc", tau0, u[o, v, v, v], optimize=True)

    tau74 -= 2 * np.einsum("ibca->iabc", tau73, optimize=True)

    tau80 = zeros((N, M, M, M))

    tau80 -= np.einsum("iacb->iabc", tau73, optimize=True)

    tau101 -= 2 * np.einsum("iacb->iabc", tau73, optimize=True)

    tau73 = None

    tau91 = zeros((M, M, M, M))

    tau91 += 4 * np.einsum("ijac,jibd->abcd", tau0, tau0, optimize=True)

    tau121 = zeros((N, N, N, N, M, M))

    tau121 -= np.einsum("ijac,lkcb->ijklab", tau0, u[o, o, v, v], optimize=True)

    tau122 = zeros((N, N, N, M))

    tau122 += np.einsum("bl,lijkab->ijka", t1, tau121, optimize=True)

    tau121 = None

    tau125 -= 4 * np.einsum("kija->ijka", tau122, optimize=True)

    tau122 = None

    e -= np.einsum("ijab,jaib->", tau0, u[o, v, o, v], optimize=True)

    tau1 = zeros((M, M, M, M))

    tau1 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau8 = zeros((N, N, M, M))

    tau8 -= np.einsum("ijdc,cabd->ijab", tau0, tau1, optimize=True)

    tau79 = zeros((N, N, M, M))

    tau79 += 2 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau98 -= np.einsum("ijab->ijab", tau8, optimize=True)

    tau100 = zeros((N, N, M, M))

    tau100 += 4 * np.einsum("ikbc,kjca->ijab", tau98, u[o, o, v, v], optimize=True)

    tau98 = None

    e += np.einsum("ijab,jaib->", tau8, u[o, v, o, v], optimize=True) / 2

    tau8 = None

    tau19 = zeros((M, M, M, M))

    tau19 += 2 * np.einsum("aefb,ecdf->abcd", tau1, u[v, v, v, v], optimize=True)

    tau49 -= np.einsum("badc,kjid->ijkabc", tau1, u[o, o, o, v], optimize=True)

    tau72 = zeros((N, M, M, M))

    tau72 -= np.einsum("daeb,idec->iabc", tau1, u[o, v, v, v], optimize=True)

    tau74 += np.einsum("iabc->iabc", tau72, optimize=True)

    tau82 += np.einsum("iabc->iabc", tau72, optimize=True)

    tau84 = zeros((N, N, N, M))

    tau84 += np.einsum("bcki,jcab->ijka", t2, tau82, optimize=True)

    tau82 = None

    tau101 -= np.einsum("icab->iabc", tau72, optimize=True)

    tau72 = None

    tau91 -= np.einsum("ebcf,fade->abcd", tau1, tau1, optimize=True)

    tau100 -= np.einsum("abcd,jicd->ijab", tau91, u[o, o, v, v], optimize=True)

    tau91 = None

    tau103 = zeros((N, N, M, M, M, M))

    tau103 -= np.einsum("abec,jied->ijabcd", tau1, u[o, o, v, v], optimize=True)

    tau104 = zeros((N, M, M, M))

    tau104 -= np.einsum("dj,ijdabc->iabc", t1, tau103, optimize=True)

    tau103 = None

    tau108 = zeros((N, N, M, M))

    tau108 -= np.einsum("ci,jabc->ijab", t1, tau104, optimize=True)

    tau104 = None

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("caki,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("acdb,ijcd->ijab", tau1, tau2, optimize=True)

    e += np.einsum("ijba,ijab->", tau0, tau3, optimize=True) / 2

    tau3 = None

    tau22 = zeros((N, N, M, M))

    tau22 -= 4 * np.einsum("kjbc,kiac->ijab", tau0, tau2, optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("ijab->ijab", tau2, optimize=True)

    tau51 = zeros((N, N, M, M))

    tau51 += 2 * np.einsum("ijab->ijab", tau2, optimize=True)

    tau53 -= 4 * np.einsum("jiab->ijab", tau2, optimize=True)

    tau58 = zeros((N, N, M, M))

    tau58 += np.einsum("cbkj,ikac->ijab", t2, tau2, optimize=True)

    tau59 = zeros((N, N, M, M))

    tau59 += 4 * np.einsum("ijab->ijab", tau58, optimize=True)

    tau59 -= 4 * np.einsum("ijba->ijab", tau58, optimize=True)

    tau58 = None

    tau110 = zeros((N, N, M, M))

    tau110 -= np.einsum("jiab->ijab", tau2, optimize=True)

    tau132 = zeros((N, N, M, M))

    tau132 += np.einsum("jiab->ijab", tau2, optimize=True)

    tau2 = None

    tau5 = zeros((N, N, N, N))

    tau5 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau52 = zeros((N, N, N, N))

    tau52 += np.einsum("lkji->ijkl", tau5, optimize=True)

    e -= np.einsum("ijkl,jilk->", tau4, tau5, optimize=True) / 4

    tau4 = None

    e -= np.einsum("lkji,lkij->", tau5, tau7, optimize=True) / 8

    tau7 = None

    tau10 = zeros((N, M))

    tau10 += np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau11 = zeros((N, N))

    tau11 += np.einsum("ai,ja->ij", l1, tau10, optimize=True)

    e -= np.einsum("ji,ij->", f[o, o], tau11, optimize=True)

    tau65 = zeros((N, M))

    tau65 += np.einsum("ia->ia", tau10, optimize=True)

    tau12 = zeros((N, N))

    tau12 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    e -= np.einsum("ij,ji->", tau11, tau12, optimize=True)

    tau12 = None

    tau13 = zeros((N, N))

    tau13 -= np.einsum("ak,kija->ij", t1, u[o, o, o, v], optimize=True)

    e -= np.einsum("ij,ji->", tau11, tau13, optimize=True)

    tau13 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau68 = zeros((N, N, M, M))

    tau68 += 2 * np.einsum("jiba->ijab", tau14, optimize=True)

    tau100 -= 2 * np.einsum("jiba->ijab", tau14, optimize=True)

    tau102 = zeros((N, N, M, M))

    tau102 += 2 * np.einsum("jiba->ijab", tau14, optimize=True)

    tau113 = zeros((N, N, M, M))

    tau113 += 2 * np.einsum("jiba->ijab", tau14, optimize=True)

    e += np.einsum("jiba,ijab->", tau14, tau6, optimize=True) / 4

    tau14 = None

    tau15 = zeros((N, N))

    tau15 += np.einsum("baki,kjba->ij", t2, u[o, o, v, v], optimize=True)

    e -= np.einsum("ij,ij->", tau11, tau15, optimize=True) / 2

    tau15 = None

    tau16 = zeros((N, M))

    tau16 += np.einsum("bj,jiba->ia", t1, u[o, o, v, v], optimize=True)

    tau17 = zeros((N, N))

    tau17 += np.einsum("ai,ja->ij", t1, tau16, optimize=True)

    e -= np.einsum("ij,ij->", tau11, tau17, optimize=True)

    tau17 = None

    tau11 = None

    tau62 = zeros((N, M))

    tau62 += np.einsum("ia->ia", tau16, optimize=True)

    tau140 = zeros((N, N))

    tau140 += 8 * np.einsum("ja,ia->ij", tau10, tau16, optimize=True)

    tau18 = zeros((M, M, M, M))

    tau18 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau19 += np.einsum("eafc,ebdf->abcd", tau1, tau18, optimize=True)

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("cdji,bcda->ijab", l2, tau19, optimize=True)

    tau19 = None

    tau20 = zeros((M, M, M, M))

    tau20 += np.einsum("badc->abcd", tau18, optimize=True)

    tau18 = None

    tau20 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 += np.einsum("ijcd,cadb->ijab", tau0, tau20, optimize=True)

    tau22 += np.einsum("ijab->ijab", tau21, optimize=True)

    tau23 += 4 * np.einsum("caki,jkcb->ijab", l2, tau22, optimize=True)

    tau22 = None

    tau53 += 2 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau21 = None

    tau69 = zeros((N, M, M, M))

    tau69 -= np.einsum("di,badc->iabc", t1, tau20, optimize=True)

    tau23 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    e -= np.einsum("abij,ijab->", t2, tau23, optimize=True) / 16

    tau23 = None

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("dcij,abdc->ijab", t2, u[v, v, v, v], optimize=True)

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum("jiba->ijab", tau24, optimize=True)

    tau59 += 2 * np.einsum("jiba->ijab", tau24, optimize=True)

    tau24 = None

    tau25 += 2 * np.einsum("baji->ijab", u[v, v, o, o], optimize=True)

    e += np.einsum("abij,ijab->", l2, tau25, optimize=True) / 8

    tau25 = None

    tau26 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("caki,kjcb->ijab", l2, tau26, optimize=True)

    tau33 = zeros((N, N, N, N))

    tau33 -= np.einsum("abkj,ilba->ijkl", t2, tau27, optimize=True)

    tau68 += 8 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau102 -= 4 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau102 -= 4 * np.einsum("jiab->ijab", tau27, optimize=True)

    tau113 += 8 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau117 = zeros((N, N, M, M))

    tau117 += 2 * np.einsum("ijab->ijab", tau27, optimize=True)

    e -= np.einsum("ijba,ijab->", tau27, tau6, optimize=True)

    tau27 = None

    tau6 = None

    tau30 = zeros((N, N, N, N))

    tau30 -= 8 * np.einsum("jlab,kiab->ijkl", tau0, tau26, optimize=True)

    tau53 += 2 * np.einsum("cbda,jicd->ijab", tau1, tau26, optimize=True)

    tau1 = None

    tau53 -= 4 * np.einsum("kjbc,kiac->ijab", tau0, tau26, optimize=True)

    tau53 -= 4 * np.einsum("ikca,jkcb->ijab", tau0, tau26, optimize=True)

    tau127 = zeros((N, N, N, M))

    tau127 += np.einsum("bi,jkab->ijka", t1, tau26, optimize=True)

    tau26 = None

    tau128 = zeros((N, N, N, M))

    tau128 += 2 * np.einsum("jkia->ijka", tau127, optimize=True)

    tau127 = None

    tau28 = zeros((N, N, N, N))

    tau28 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau30 -= np.einsum("jmnk,mlni->ijkl", tau28, tau5, optimize=True)

    tau5 = None

    tau31 = zeros((N, N, N, N))

    tau31 += np.einsum("miln,njkm->ijkl", tau28, tau28, optimize=True)

    tau32 += np.einsum("ijkl->ijkl", tau31, optimize=True)

    e -= np.einsum("lkij,ijkl->", tau32, u[o, o, o, o], optimize=True) / 8

    tau32 = None

    tau99 += np.einsum("ijlk->ijkl", tau31, optimize=True)

    tau31 = None

    tau41 += np.einsum("ijml,mlka->ijka", tau28, u[o, o, o, v], optimize=True)

    tau63 = zeros((N, N, M, M))

    tau63 -= np.einsum("ablk,lkji->ijab", t2, tau28, optimize=True)

    tau64 -= np.einsum("ijba->ijab", tau63, optimize=True)

    tau78 = zeros((N, N, M, M))

    tau78 += np.einsum("caki,jkcb->ijab", l2, tau64, optimize=True)

    tau79 += np.einsum("ijab->ijab", tau78, optimize=True)

    tau112 = zeros((N, M))

    tau112 -= 2 * np.einsum("ijbc,jbca->ia", tau79, u[o, v, v, v], optimize=True)

    tau79 = None

    tau112 -= 2 * np.einsum("kjab,jikb->ia", tau78, u[o, o, o, v], optimize=True)

    tau78 = None

    tau92 -= np.einsum("ijba->ijab", tau63, optimize=True)

    tau63 = None

    tau67 = zeros((N, N, M, M))

    tau67 -= np.einsum("ijlk,lkba->ijab", tau28, u[o, o, v, v], optimize=True)

    tau68 -= np.einsum("ijba->ijab", tau67, optimize=True)

    tau75 = zeros((M, M))

    tau75 += np.einsum("caij,ijcb->ab", t2, tau68, optimize=True)

    tau68 = None

    tau102 -= np.einsum("ijba->ijab", tau67, optimize=True)

    tau113 -= np.einsum("ijba->ijab", tau67, optimize=True)

    tau67 = None

    tau77 = zeros((N, N, N, M))

    tau77 -= np.einsum("limj,mkla->ijka", tau28, u[o, o, o, v], optimize=True)

    tau87 += np.einsum("ijka->ijka", tau77, optimize=True)

    tau112 += 4 * np.einsum("jkab,kjib->ia", tau0, tau77, optimize=True)

    tau77 = None

    tau89 = zeros((N, N, M, M))

    tau89 -= np.einsum("klab,lijk->ijab", tau0, tau28, optimize=True)

    tau100 -= 4 * np.einsum("jkac,kicb->ijab", tau89, u[o, o, v, v], optimize=True)

    tau89 = None

    tau112 -= 2 * np.einsum("jikl,ljka->ia", tau28, tau87, optimize=True)

    tau87 = None

    tau142 = zeros((N, M))

    tau142 += np.einsum("ijlk,lkja->ia", tau28, u[o, o, o, v], optimize=True)

    e += np.einsum("ijkl,lijk->", tau28, tau33, optimize=True) / 4

    tau33 = None

    tau29 += np.einsum("baji->ijab", t2, optimize=True)

    tau30 -= np.einsum("klab,jiab->ijkl", tau29, u[o, o, v, v], optimize=True)

    tau84 += np.einsum("ikbc,jabc->ijka", tau29, u[o, v, v, v], optimize=True)

    tau85 = zeros((N, N, N, N))

    tau85 += np.einsum("abij,klab->ijkl", l2, tau29, optimize=True)

    tau29 = None

    tau86 += np.einsum("jikl->ijkl", tau85, optimize=True)

    tau112 -= 2 * np.einsum("lijk,jkla->ia", tau86, u[o, o, o, v], optimize=True)

    tau86 = None

    tau99 += np.einsum("jikl->ijkl", tau85, optimize=True)

    tau85 = None

    tau100 += np.einsum("jikl,klba->ijab", tau99, u[o, o, v, v], optimize=True)

    tau99 = None

    tau30 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    e += np.einsum("ijkl,klij->", tau28, tau30, optimize=True) / 16

    tau30 = None

    tau34 = zeros((N, N, N, M))

    tau34 -= np.einsum("bali,ljkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau40 = zeros((N, N, N, M))

    tau40 += 2 * np.einsum("ijka->ijka", tau34, optimize=True)

    tau43 = zeros((N, N, N, M))

    tau43 += np.einsum("ijka->ijka", tau34, optimize=True)

    tau76 = zeros((N, M))

    tau76 += 4 * np.einsum("kjli,klja->ia", tau28, tau34, optimize=True)

    tau128 -= 2 * np.einsum("jika->ijka", tau34, optimize=True)

    tau128 += 2 * np.einsum("kija->ijka", tau34, optimize=True)

    tau141 = zeros((N, N, N, M))

    tau141 += 2 * np.einsum("ijka->ijka", tau34, optimize=True)

    tau34 = None

    tau37 = zeros((N, M, M, M))

    tau37 -= np.einsum("daji,jbdc->iabc", t2, u[o, v, v, v], optimize=True)

    tau38 = zeros((N, M, M, M))

    tau38 += 2 * np.einsum("iabc->iabc", tau37, optimize=True)

    tau42 = zeros((N, M, M, M))

    tau42 += np.einsum("iabc->iabc", tau37, optimize=True)

    tau56 = zeros((N, M, M, M))

    tau56 -= 2 * np.einsum("iabc->iabc", tau37, optimize=True)

    tau61 = zeros((N, M))

    tau61 -= 4 * np.einsum("cbji,jcba->ia", l2, tau37, optimize=True)

    tau69 -= 4 * np.einsum("iabc->iabc", tau37, optimize=True)

    tau69 += 4 * np.einsum("ibac->iabc", tau37, optimize=True)

    tau37 = None

    tau38 += np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau41 += 2 * np.einsum("bcji,kbca->ijka", l2, tau38, optimize=True)

    tau38 = None

    tau39 = zeros((N, N, N, M))

    tau39 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau40 += np.einsum("kija->ijka", tau39, optimize=True)

    tau41 += 4 * np.einsum("bali,kjlb->ijka", l2, tau40, optimize=True)

    tau40 = None

    tau44 = zeros((N, N, N, M))

    tau44 -= np.einsum("kjia->ijka", tau39, optimize=True)

    tau61 += np.einsum("bakj,kjib->ia", l2, tau39, optimize=True)

    tau128 -= np.einsum("kjia->ijka", tau39, optimize=True)

    tau39 = None

    tau41 -= 4 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau76 -= np.einsum("bajk,jkib->ia", t2, tau41, optimize=True)

    tau41 = None

    tau42 += np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau76 += 8 * np.einsum("jibc,jbac->ia", tau0, tau42, optimize=True)

    tau42 = None

    tau43 += np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    tau76 += 8 * np.einsum("jkba,jkib->ia", tau0, tau43, optimize=True)

    tau43 = None

    tau44 -= 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau76 -= np.einsum("jkli,ljka->ia", tau28, tau44, optimize=True)

    tau44 = None

    tau46 = zeros((N, N, N, N, N, M))

    tau46 -= np.einsum("abij,lkmb->ijklma", t2, u[o, o, o, v], optimize=True)

    tau49 += 2 * np.einsum("baml,imkjlc->ijkabc", l2, tau46, optimize=True)

    tau46 = None

    tau47 = zeros((N, N, N, N, M, M))

    tau47 += np.einsum("caji,bckl->ijklab", l2, t2, optimize=True)

    tau49 += 4 * np.einsum("mjla,klimbc->ijkabc", u[o, o, o, v], tau47, optimize=True)

    tau49 -= 2 * np.einsum("ldba,kjildc->ijkabc", u[o, v, v, v], tau47, optimize=True)

    tau48 = zeros((N, N, N, M, M, M))

    tau48 -= np.einsum("adij,kbdc->ijkabc", t2, u[o, v, v, v], optimize=True)

    tau49 -= 4 * np.einsum("dblk,iljcda->ijkabc", l2, tau48, optimize=True)

    tau48 = None

    tau53 -= np.einsum("ck,jikcba->ijab", t1, tau49, optimize=True)

    tau49 = None

    tau50 = zeros((N, N, M, M))

    tau50 -= np.einsum("ci,jacb->ijab", t1, u[o, v, v, v], optimize=True)

    tau51 += np.einsum("ijab->ijab", tau50, optimize=True)

    tau132 += np.einsum("jiab->ijab", tau50, optimize=True)

    tau50 = None

    tau51 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau53 += np.einsum("kilj,klab->ijab", tau28, tau51, optimize=True)

    tau51 = None

    tau52 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau53 += 2 * np.einsum("klba,likj->ijab", tau0, tau52, optimize=True)

    tau59 += np.einsum("bakl,klji->ijab", t2, tau52, optimize=True)

    tau102 += np.einsum("bakl,jikl->ijab", l2, tau52, optimize=True)

    tau112 += 2 * np.einsum("jb,jiba->ia", tau10, tau102, optimize=True)

    tau102 = None

    tau119 = zeros((N, N))

    tau119 += np.einsum("klmi,mjkl->ij", tau28, tau52, optimize=True)

    tau138 = zeros((N, N))

    tau138 += np.einsum("ji->ij", tau119, optimize=True)

    tau140 += np.einsum("ji->ij", tau119, optimize=True)

    tau119 = None

    tau53 += 4 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau76 += 2 * np.einsum("bj,jiab->ia", t1, tau53, optimize=True)

    tau53 = None

    tau54 = zeros((M, M))

    tau54 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau60 = zeros((N, M))

    tau60 += np.einsum("bc,ibac->ia", tau54, u[o, v, v, v], optimize=True)

    tau61 += 2 * np.einsum("ia->ia", tau60, optimize=True)

    tau111 = zeros((N, M))

    tau111 += 2 * np.einsum("ia->ia", tau60, optimize=True)

    tau137 = zeros((N, M))

    tau137 += np.einsum("ia->ia", tau60, optimize=True)

    tau139 = zeros((N, M))

    tau139 += 2 * np.einsum("ia->ia", tau60, optimize=True)

    tau60 = None

    tau70 = zeros((M, M))

    tau70 += np.einsum("cd,cadb->ab", tau54, tau20, optimize=True)

    tau20 = None

    tau75 -= 2 * np.einsum("ab->ab", tau70, optimize=True)

    tau114 = zeros((M, M))

    tau114 -= np.einsum("ab->ab", tau70, optimize=True)

    tau70 = None

    tau74 -= np.einsum("ad,ibdc->iabc", tau54, u[o, v, v, v], optimize=True)

    tau75 -= 4 * np.einsum("ci,icab->ab", t1, tau74, optimize=True)

    tau74 = None

    tau90 = zeros((N, N, M, M))

    tau90 -= np.einsum("ac,jicb->ijab", tau54, u[o, o, v, v], optimize=True)

    tau100 += 4 * np.einsum("ikbc,jkca->ijab", tau0, tau90, optimize=True)

    tau0 = None

    tau109 = zeros((N, N, M, M))

    tau109 -= np.einsum("ijab->ijab", tau90, optimize=True)

    tau134 = zeros((N, N, M, M))

    tau134 -= np.einsum("ijab->ijab", tau90, optimize=True)

    tau90 = None

    tau96 += np.einsum("cb,acij->ijab", tau54, t2, optimize=True)

    tau120 = zeros((N, N, N, M))

    tau120 -= np.einsum("ab,ijkb->ijka", tau54, u[o, o, o, v], optimize=True)

    tau125 += 2 * np.einsum("ijka->ijka", tau120, optimize=True)

    tau120 = None

    tau142 += 2 * np.einsum("ib,ab->ia", tau16, tau54, optimize=True)

    tau16 = None

    tau55 = zeros((N, M, M, M))

    tau55 += np.einsum("abkj,kjic->iabc", t2, u[o, o, o, v], optimize=True)

    tau56 -= np.einsum("ibac->iabc", tau55, optimize=True)

    tau69 -= 2 * np.einsum("ibac->iabc", tau55, optimize=True)

    tau55 = None

    tau56 -= 2 * np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau76 += 2 * np.einsum("bc,ibac->ia", tau54, tau56, optimize=True)

    tau56 = None

    tau57 = zeros((N, N, M, M))

    tau57 += np.einsum("caki,kbjc->ijab", t2, u[o, v, o, v], optimize=True)

    tau59 += 8 * np.einsum("ijba->ijab", tau57, optimize=True)

    tau59 -= 8 * np.einsum("jiba->ijab", tau57, optimize=True)

    tau57 = None

    tau59 += 4 * np.einsum("baji->ijab", u[v, v, o, o], optimize=True)

    tau76 -= np.einsum("bj,jiba->ia", l1, tau59, optimize=True)

    tau59 = None

    tau61 += 4 * np.einsum("ia->ia", f[o, v], optimize=True)

    tau76 -= 2 * np.einsum("jb,baji->ia", tau61, t2, optimize=True)

    tau61 = None

    tau62 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau76 += 2 * np.einsum("jb,ijba->ia", tau62, tau64, optimize=True)

    tau64 = None

    tau138 += 8 * np.einsum("ja,ia->ij", tau10, tau62, optimize=True)

    tau62 = None

    tau65 += np.einsum("ai->ia", t1, optimize=True)

    tau143 = zeros((N, M))

    tau143 += np.einsum("ba,ib->ia", tau54, tau65, optimize=True)

    e -= np.einsum("ia,ia->", f[o, v], tau143, optimize=True) / 2

    tau143 = None

    tau144 = zeros((N, N))

    tau144 += np.einsum("ia,ja->ij", f[o, v], tau65, optimize=True)

    tau66 = zeros((M, M))

    tau66 -= 2 * np.einsum("ab->ab", f[v, v], optimize=True)

    tau66 -= 2 * np.einsum("ci,iacb->ab", t1, u[o, v, v, v], optimize=True)

    tau66 += np.einsum("caji,jicb->ab", t2, u[o, o, v, v], optimize=True)

    tau76 += 4 * np.einsum("ib,ab->ia", tau65, tau66, optimize=True)

    tau65 = None

    tau66 = None

    tau69 -= 4 * np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau75 -= 2 * np.einsum("ci,icab->ab", l1, tau69, optimize=True)

    tau69 = None

    tau76 += np.einsum("bi,ab->ia", t1, tau75, optimize=True)

    tau75 = None

    tau76 -= 8 * np.einsum("ai->ia", f[v, o], optimize=True)

    tau76 -= 4 * np.einsum("cbji,jacb->ia", t2, u[o, v, v, v], optimize=True)

    e -= np.einsum("ai,ia->", l1, tau76, optimize=True) / 8

    tau76 = None

    tau80 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau81 = zeros((N, M, M, M))

    tau81 += 2 * np.einsum("daji,jbdc->iabc", t2, tau80, optimize=True)

    tau80 = None

    tau81 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau112 += 4 * np.einsum("bcji,jbca->ia", l2, tau81, optimize=True)

    tau81 = None

    tau83 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau84 += 4 * np.einsum("bali,kjlb->ijka", t2, tau83, optimize=True)

    tau83 = None

    tau84 -= 2 * np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    tau112 += 2 * np.einsum("bajk,jikb->ia", l2, tau84, optimize=True)

    tau84 = None

    tau88 = zeros((N, N, N, M, M, M))

    tau88 -= np.einsum("daij,kdbc->ijkabc", l2, u[o, v, v, v], optimize=True)

    tau100 += 4 * np.einsum("ck,kijbca->ijab", t1, tau88, optimize=True)

    tau88 = None

    tau92 += 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau93 = zeros((N, N, M, M))

    tau93 += np.einsum("ikca,kjcb->ijab", tau92, u[o, o, v, v], optimize=True)

    tau92 = None

    tau93 += 4 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau100 -= 2 * np.einsum("cbki,kjca->ijab", l2, tau93, optimize=True)

    tau93 = None

    tau94 = zeros((N, N, N, N))

    tau94 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau97 = zeros((N, N, N, N))

    tau97 -= 4 * np.einsum("ljik->ijkl", tau94, optimize=True)

    tau95 = zeros((N, N, N, M))

    tau95 -= np.einsum("bi,jkba->ijka", t1, u[o, o, v, v], optimize=True)

    tau97 -= 2 * np.einsum("ak,ljia->ijkl", t1, tau95, optimize=True)

    tau107 = zeros((N, N, N, M, M, M))

    tau107 -= np.einsum("lmka,jlimbc->ijkabc", tau95, tau47, optimize=True)

    tau47 = None

    tau123 = zeros((N, N, N, M))

    tau123 += np.einsum("kjia->ijka", tau95, optimize=True)

    tau96 -= np.einsum("baji->ijab", t2, optimize=True)

    tau97 -= np.einsum("klba,jiab->ijkl", tau96, u[o, o, v, v], optimize=True)

    tau96 = None

    tau97 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau100 -= np.einsum("bakl,jikl->ijab", l2, tau97, optimize=True)

    tau97 = None

    tau100 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau112 -= np.einsum("bj,jiba->ia", t1, tau100, optimize=True)

    tau100 = None

    tau101 -= 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau112 -= 2 * np.einsum("bc,ibca->ia", tau54, tau101, optimize=True)

    tau101 = None

    tau105 = zeros((N, N, N, N, M, M))

    tau105 += np.einsum("acij,lkcb->ijklab", t2, u[o, o, v, v], optimize=True)

    tau106 = zeros((N, N, N, N, N, M))

    tau106 += np.einsum("bi,jklmab->ijklma", t1, tau105, optimize=True)

    tau105 = None

    tau107 -= np.einsum("baml,limkjc->ijkabc", l2, tau106, optimize=True)

    tau106 = None

    tau108 += 2 * np.einsum("ck,ikjcab->ijab", t1, tau107, optimize=True)

    tau107 = None

    tau112 += 2 * np.einsum("bj,jiab->ia", l1, tau108, optimize=True)

    tau108 = None

    tau109 += 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau111 += np.einsum("bj,jiba->ia", t1, tau109, optimize=True)

    tau109 = None

    tau110 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau111 -= 4 * np.einsum("bj,ijba->ia", l1, tau110, optimize=True)

    tau112 -= np.einsum("ib,ab->ia", tau111, tau54, optimize=True)

    tau111 = None

    tau130 = zeros((N, N))

    tau130 += np.einsum("ab,ijab->ij", tau54, tau110, optimize=True)

    tau110 = None

    tau138 += 4 * np.einsum("ij->ij", tau130, optimize=True)

    tau140 += 4 * np.einsum("ij->ij", tau130, optimize=True)

    tau130 = None

    tau112 += 8 * np.einsum("ia->ia", f[o, v], optimize=True)

    e += np.einsum("ai,ia->", t1, tau112, optimize=True) / 8

    tau112 = None

    tau113 += 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau114 += np.einsum("caij,ijcb->ab", t2, tau113, optimize=True)

    tau113 = None

    tau114 -= 8 * np.einsum("ab->ab", f[v, v], optimize=True)

    e -= np.einsum("ab,ab->", tau114, tau54, optimize=True) / 16

    tau114 = None

    tau54 = None

    tau115 = zeros((N, N))

    tau115 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau131 = zeros((N, N))

    tau131 += np.einsum("ij->ij", tau115, optimize=True)

    tau136 = zeros((N, N))

    tau136 += 2 * np.einsum("ij->ij", tau115, optimize=True)

    tau116 = zeros((N, N))

    tau116 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau131 += np.einsum("ij->ij", tau116, optimize=True)

    tau137 += np.einsum("jk,jkia->ia", tau131, tau95, optimize=True)

    tau95 = None

    tau138 -= 2 * np.einsum("lk,kilj->ij", tau131, tau52, optimize=True)

    tau131 = None

    tau136 += np.einsum("ij->ij", tau116, optimize=True)

    tau137 -= np.einsum("kj,jika->ia", tau136, u[o, o, o, v], optimize=True)

    tau136 = None

    tau138 += 4 * np.einsum("kl,klij->ij", tau116, tau94, optimize=True)

    tau94 = None

    tau140 -= np.einsum("kl,likj->ij", tau116, tau52, optimize=True)

    tau52 = None

    e -= np.einsum("ij,ji->", tau116, tau144, optimize=True) / 2

    tau144 = None

    tau117 += np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau118 = zeros((N, N))

    tau118 += np.einsum("abki,kjab->ij", t2, tau117, optimize=True)

    tau117 = None

    tau138 += 4 * np.einsum("ji->ij", tau118, optimize=True)

    tau140 += 4 * np.einsum("ji->ij", tau118, optimize=True)

    tau118 = None

    tau123 -= 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau124 = zeros((N, N, N, M))

    tau124 += np.einsum("mkla,limj->ijka", tau123, tau28, optimize=True)

    tau28 = None

    tau125 -= np.einsum("ikja->ijka", tau124, optimize=True)

    tau124 = None

    tau139 -= np.einsum("jk,kija->ia", tau116, tau123, optimize=True)

    tau123 = None

    tau125 -= 4 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau126 = zeros((N, N))

    tau126 += np.einsum("ak,kija->ij", t1, tau125, optimize=True)

    tau125 = None

    tau138 -= 2 * np.einsum("ij->ij", tau126, optimize=True)

    tau140 -= 2 * np.einsum("ij->ij", tau126, optimize=True)

    tau126 = None

    tau128 -= 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau129 = zeros((N, N))

    tau129 += np.einsum("ak,ikja->ij", l1, tau128, optimize=True)

    tau128 = None

    tau138 -= 4 * np.einsum("ij->ij", tau129, optimize=True)

    tau140 -= 4 * np.einsum("ij->ij", tau129, optimize=True)

    tau129 = None

    tau132 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau133 = zeros((N, M))

    tau133 += np.einsum("bj,ijba->ia", l1, tau132, optimize=True)

    tau132 = None

    tau137 += 2 * np.einsum("ia->ia", tau133, optimize=True)

    tau139 += 4 * np.einsum("ia->ia", tau133, optimize=True)

    tau133 = None

    tau134 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau135 = zeros((N, M))

    tau135 += np.einsum("bj,jiba->ia", t1, tau134, optimize=True)

    tau134 = None

    tau137 += np.einsum("ia->ia", tau135, optimize=True)

    tau139 += 2 * np.einsum("ia->ia", tau135, optimize=True)

    tau135 = None

    tau140 += 2 * np.einsum("aj,ia->ij", t1, tau139, optimize=True)

    tau139 = None

    tau137 += 2 * np.einsum("ia->ia", f[o, v], optimize=True)

    tau138 += 4 * np.einsum("aj,ia->ij", t1, tau137, optimize=True)

    tau137 = None

    tau138 += 8 * np.einsum("ij->ij", f[o, o], optimize=True)

    e -= np.einsum("ij,ji->", tau115, tau138, optimize=True) / 8

    tau138 = None

    tau115 = None

    tau140 += 8 * np.einsum("ij->ij", f[o, o], optimize=True)

    e -= np.einsum("ij,ji->", tau116, tau140, optimize=True) / 16

    tau140 = None

    tau116 = None

    tau141 += np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    tau142 += 2 * np.einsum("bajk,jikb->ia", l2, tau141, optimize=True)

    tau141 = None

    tau142 -= 2 * np.einsum("cbji,cbja->ia", l2, u[v, v, o, v], optimize=True)

    e -= np.einsum("ia,ia->", tau10, tau142, optimize=True) / 4

    tau10 = None

    tau142 = None

    return e
