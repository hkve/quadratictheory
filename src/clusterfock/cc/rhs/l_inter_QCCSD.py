import numpy as np


def lambda_amplitudes_intermediates_qccsd(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 -= np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau8 = zeros((N, N, N, M))

    tau8 += np.einsum("liab,jklb->ijka", tau0, u[o, o, o, v], optimize=True)

    tau15 = zeros((N, N, N, M))

    tau15 += 4 * np.einsum("kija->ijka", tau8, optimize=True)

    tau111 = zeros((N, N, N, M))

    tau111 += 4 * np.einsum("kija->ijka", tau8, optimize=True)

    tau193 = zeros((N, N, N, M))

    tau193 -= np.einsum("ikja->ijka", tau8, optimize=True)

    tau268 = zeros((N, N, N, M))

    tau268 += 4 * np.einsum("ikja->ijka", tau8, optimize=True)

    tau8 = None

    tau9 = zeros((N, N, N, M))

    tau9 += np.einsum("ijbc,kbac->ijka", tau0, u[o, v, v, v], optimize=True)

    tau15 -= 8 * np.einsum("jkia->ijka", tau9, optimize=True)

    tau111 += 4 * np.einsum("ikja->ijka", tau9, optimize=True)

    tau186 = zeros((N, N, N, M))

    tau186 += 2 * np.einsum("ijka->ijka", tau9, optimize=True)

    tau218 = zeros((N, N, N, M))

    tau218 += 2 * np.einsum("ikja->ijka", tau9, optimize=True)

    tau9 = None

    tau18 = zeros((N, N, N, M))

    tau18 += np.einsum("bj,ikba->ijka", t1, tau0, optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("ikla,lkjb->ijab", tau18, u[o, o, o, v], optimize=True)

    tau52 = zeros((N, N, M, M))

    tau52 += 4 * np.einsum("ijab->ijab", tau19, optimize=True)

    tau150 = zeros((N, N, M, M))

    tau150 -= 2 * np.einsum("ijab->ijab", tau19, optimize=True)

    tau19 = None

    tau93 = zeros((N, M, M, M))

    tau93 += np.einsum("ijka,kjbc->iabc", tau18, u[o, o, v, v], optimize=True)

    tau18 = None

    tau94 = zeros((N, M, M, M))

    tau94 += 4 * np.einsum("iacb->iabc", tau93, optimize=True)

    tau289 = zeros((N, M, M, M))

    tau289 += 4 * np.einsum("icba->iabc", tau93, optimize=True)

    tau93 = None

    tau34 = zeros((N, N, N, M, M, M))

    tau34 += np.einsum("ijad,kbdc->ijkabc", tau0, u[o, v, v, v], optimize=True)

    tau47 = zeros((N, N, N, M, M, M))

    tau47 += 4 * np.einsum("kijbca->ijkabc", tau34, optimize=True)

    tau137 = zeros((N, N, M, M))

    tau137 += np.einsum("ck,ikjacb->ijab", l1, tau34, optimize=True)

    tau240 = zeros((N, N, M, M))

    tau240 -= 4 * np.einsum("ijab->ijab", tau137, optimize=True)

    tau137 = None

    tau154 = zeros((N, N, N, M, M, M))

    tau154 -= np.einsum("bdjl,ilkadc->ijkabc", l2, tau34, optimize=True)

    tau34 = None

    tau164 = zeros((N, N, N, M, M, M))

    tau164 += 2 * np.einsum("ikjbac->ijkabc", tau154, optimize=True)

    tau164 += 2 * np.einsum("jikacb->ijkabc", tau154, optimize=True)

    tau154 = None

    tau73 = zeros((N, N, M, M))

    tau73 -= np.einsum("acki,kjcb->ijab", t2, tau0, optimize=True)

    tau74 = zeros((N, N, M, M))

    tau74 += 4 * np.einsum("ijab->ijab", tau73, optimize=True)

    tau167 = zeros((N, N, M, M))

    tau167 += 4 * np.einsum("ijab->ijab", tau73, optimize=True)

    tau303 = zeros((N, N, M, M))

    tau303 -= 2 * np.einsum("ijab->ijab", tau73, optimize=True)

    tau308 = zeros((N, N, M, M))

    tau308 += 2 * np.einsum("ijab->ijab", tau73, optimize=True)

    tau73 = None

    tau83 = zeros((M, M))

    tau83 -= np.einsum("jica,icjb->ab", tau0, u[o, v, o, v], optimize=True)

    tau102 = zeros((M, M))

    tau102 += 8 * np.einsum("ab->ab", tau83, optimize=True)

    tau83 = None

    tau87 = zeros((N, M, M, M))

    tau87 -= np.einsum("jkab,kijc->iabc", tau0, u[o, o, o, v], optimize=True)

    tau94 += 4 * np.einsum("ibac->iabc", tau87, optimize=True)

    tau162 = zeros((N, M, M, M))

    tau162 += 2 * np.einsum("iabc->iabc", tau87, optimize=True)

    tau87 = None

    tau89 = zeros((N, M, M, M))

    tau89 += np.einsum("ijda,jdbc->iabc", tau0, u[o, v, v, v], optimize=True)

    tau94 -= 4 * np.einsum("iacb->iabc", tau89, optimize=True)

    tau152 = zeros((N, N, N, M, M, M))

    tau152 -= np.einsum("adkj,idcb->ijkabc", l2, tau89, optimize=True)

    tau164 += 2 * np.einsum("ikjacb->ijkabc", tau152, optimize=True)

    tau152 = None

    tau289 -= 4 * np.einsum("icba->iabc", tau89, optimize=True)

    tau89 = None

    tau103 = zeros((N, N))

    tau103 -= np.einsum("kiba,jbka->ij", tau0, u[o, v, o, v], optimize=True)

    tau130 = zeros((N, N))

    tau130 += 8 * np.einsum("ji->ij", tau103, optimize=True)

    tau291 = zeros((N, N))

    tau291 += 8 * np.einsum("ji->ij", tau103, optimize=True)

    tau103 = None

    tau108 = zeros((N, N, N, N, M, M))

    tau108 -= np.einsum("ijac,lkcb->ijklab", tau0, u[o, o, v, v], optimize=True)

    tau109 = zeros((N, N, N, M))

    tau109 += np.einsum("bl,lijkab->ijka", t1, tau108, optimize=True)

    tau111 -= 4 * np.einsum("kija->ijka", tau109, optimize=True)

    tau193 += np.einsum("ikja->ijka", tau109, optimize=True)

    tau109 = None

    tau184 = zeros((N, N, N, M))

    tau184 += np.einsum("bl,ijklba->ijka", t1, tau108, optimize=True)

    tau186 += 2 * np.einsum("ijka->ijka", tau184, optimize=True)

    tau184 = None

    tau208 = zeros((N, N, N, N, M, M))

    tau208 -= np.einsum("ilkjba->ijklab", tau108, optimize=True)

    tau262 = zeros((N, N, N, N, M, M))

    tau262 += np.einsum("klijba->ijklab", tau108, optimize=True)

    tau155 = zeros((N, N, N, N, M, M))

    tau155 -= np.einsum("caij,klbc->ijklab", l2, tau0, optimize=True)

    tau249 = zeros((N, N, N, M, M, M))

    tau249 += np.einsum("ldbc,ijklda->ijkabc", u[o, v, v, v], tau155, optimize=True)

    tau252 = zeros((N, N, N, M, M, M))

    tau252 -= 2 * np.einsum("ijkbca->ijkabc", tau249, optimize=True)

    tau249 = None

    tau177 = zeros((N, N, M, M))

    tau177 += np.einsum("ikcb,kjac->ijab", tau0, tau0, optimize=True)

    tau178 = zeros((N, N, M, M))

    tau178 -= 2 * np.einsum("ijab->ijab", tau177, optimize=True)

    tau177 = None

    tau180 = zeros((N, N, N, M))

    tau180 += np.einsum("bi,jkab->ijka", l1, tau0, optimize=True)

    tau181 = zeros((N, N, N, M))

    tau181 += np.einsum("ijka->ijka", tau180, optimize=True)

    tau241 = zeros((M, M, M, M))

    tau241 += np.einsum("ijab,jcid->abcd", tau0, u[o, v, o, v], optimize=True)

    tau245 = zeros((M, M, M, M))

    tau245 += 4 * np.einsum("abcd->abcd", tau241, optimize=True)

    tau241 = None

    tau277 = zeros((N, N, N, N, N, M))

    tau277 += np.einsum("ijab,lkmb->ijklma", tau0, u[o, o, o, v], optimize=True)

    tau278 = zeros((N, N, N, N, N, M))

    tau278 -= 2 * np.einsum("ikmlja->ijklma", tau277, optimize=True)

    tau277 = None

    tau302 = zeros((M, M, M, M))

    tau302 += 4 * np.einsum("ijac,jibd->abcd", tau0, tau0, optimize=True)

    tau310 = zeros((N, N, N, N))

    tau310 += 4 * np.einsum("ilba,jkab->ijkl", tau0, tau0, optimize=True)

    r1 = zeros((M, N))

    r1 -= np.einsum("kjba,jbik->ai", tau0, u[o, v, o, o], optimize=True)

    r1 -= np.einsum("jicb,acjb->ai", tau0, u[v, v, o, v], optimize=True)

    tau1 = zeros((N, N))

    tau1 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau75 = zeros((N, N))

    tau75 += 2 * np.einsum("ij->ij", tau1, optimize=True)

    tau135 = zeros((N, N, M, M))

    tau135 -= np.einsum("ik,kjab->ijab", tau1, u[o, o, v, v], optimize=True)

    tau315 = zeros((N, M))

    tau315 += np.einsum("ja,ij->ia", f[o, v], tau1, optimize=True)

    tau320 = zeros((N, M))

    tau320 += np.einsum("ia->ia", tau315, optimize=True)

    tau315 = None

    tau316 = zeros((N, M))

    tau316 -= np.einsum("jk,kija->ia", tau1, u[o, o, o, v], optimize=True)

    tau320 -= np.einsum("ia->ia", tau316, optimize=True)

    tau316 = None

    r2 = zeros((M, M, N, N))

    r2 -= np.einsum("jk,ikab->abij", tau1, tau135, optimize=True)

    tau135 = None

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("ib,abjk->ijka", f[o, v], t2, optimize=True)

    r1 -= np.einsum("jk,kija->ai", tau1, tau2, optimize=True)

    tau3 = zeros((N, N, N, N))

    tau3 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau15 += np.einsum("ijml,mlka->ijka", tau3, u[o, o, o, v], optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 -= np.einsum("ijlk,lkba->ijab", tau3, u[o, o, v, v], optimize=True)

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("ijba->ijab", tau21, optimize=True)

    tau85 = zeros((N, N, M, M))

    tau85 -= np.einsum("ijba->ijab", tau21, optimize=True)

    tau173 = zeros((N, N, M, M))

    tau173 += np.einsum("ijba->ijab", tau21, optimize=True)

    tau229 = zeros((N, N, M, M))

    tau229 -= np.einsum("ijba->ijab", tau21, optimize=True)

    tau21 = None

    tau63 = zeros((N, M))

    tau63 += np.einsum("ijlk,lkja->ia", tau3, u[o, o, o, v], optimize=True)

    tau69 = zeros((N, M))

    tau69 += np.einsum("ia->ia", tau63, optimize=True)

    tau239 = zeros((N, M))

    tau239 += np.einsum("ia->ia", tau63, optimize=True)

    tau63 = None

    tau72 = zeros((N, N, M, M))

    tau72 -= np.einsum("ablk,lkji->ijab", t2, tau3, optimize=True)

    tau74 -= np.einsum("ijba->ijab", tau72, optimize=True)

    tau167 -= np.einsum("ijba->ijab", tau72, optimize=True)

    tau72 = None

    tau176 = zeros((N, N, M, M))

    tau176 += np.einsum("klab,iljk->ijab", tau0, tau3, optimize=True)

    tau178 += np.einsum("ijab->ijab", tau176, optimize=True)

    tau176 = None

    tau257 = zeros((N, N, N, N))

    tau257 -= np.einsum("jilk->ijkl", tau3, optimize=True)

    tau310 += np.einsum("inlm,jmkn->ijkl", tau3, tau3, optimize=True)

    tau4 = zeros((N, N, N, M))

    tau4 += np.einsum("bail,ljkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau14 = zeros((N, N, N, M))

    tau14 -= 2 * np.einsum("ikja->ijka", tau4, optimize=True)

    tau14 += 2 * np.einsum("jkia->ijka", tau4, optimize=True)

    tau67 = zeros((N, N, N, M))

    tau67 -= 4 * np.einsum("ijka->ijka", tau4, optimize=True)

    tau77 = zeros((N, N, N, M))

    tau77 -= 2 * np.einsum("jika->ijka", tau4, optimize=True)

    tau77 += 2 * np.einsum("kija->ijka", tau4, optimize=True)

    tau113 = zeros((N, N, N, M))

    tau113 -= 2 * np.einsum("jika->ijka", tau4, optimize=True)

    tau113 += 2 * np.einsum("kija->ijka", tau4, optimize=True)

    tau205 = zeros((N, N, N, M))

    tau205 -= 2 * np.einsum("ikja->ijka", tau4, optimize=True)

    tau205 += 2 * np.einsum("jkia->ijka", tau4, optimize=True)

    tau286 = zeros((N, N, N, M))

    tau286 += 4 * np.einsum("jkia->ijka", tau4, optimize=True)

    r1 += np.einsum("kjil,klja->ai", tau3, tau4, optimize=True) / 2

    tau4 = None

    tau5 = zeros((M, M))

    tau5 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 -= np.einsum("cb,acji->ijab", tau5, t2, optimize=True)

    tau303 += np.einsum("ijab->ijab", tau6, optimize=True)

    r1 += np.einsum("jb,jiab->ai", f[o, v], tau6, optimize=True) / 2

    tau6 = None

    tau64 = zeros((N, M))

    tau64 += np.einsum("bc,ibac->ia", tau5, u[o, v, v, v], optimize=True)

    tau69 -= 2 * np.einsum("ia->ia", tau64, optimize=True)

    tau126 = zeros((N, M))

    tau126 += np.einsum("ia->ia", tau64, optimize=True)

    tau239 -= 2 * np.einsum("ia->ia", tau64, optimize=True)

    tau64 = None

    tau90 = zeros((N, M, M, M))

    tau90 += np.einsum("ad,ibdc->iabc", tau5, u[o, v, v, v], optimize=True)

    tau94 -= 2 * np.einsum("ibac->iabc", tau90, optimize=True)

    tau203 = zeros((N, M, M, M))

    tau203 -= np.einsum("iabc->iabc", tau90, optimize=True)

    tau90 = None

    tau107 = zeros((N, N, N, M))

    tau107 += np.einsum("ab,jikb->ijka", tau5, u[o, o, o, v], optimize=True)

    tau111 += 2 * np.einsum("ijka->ijka", tau107, optimize=True)

    tau268 += 2 * np.einsum("kjia->ijka", tau107, optimize=True)

    tau107 = None

    tau123 = zeros((N, N, M, M))

    tau123 -= np.einsum("ac,jicb->ijab", tau5, u[o, o, v, v], optimize=True)

    tau124 = zeros((N, N, M, M))

    tau124 -= np.einsum("ijab->ijab", tau123, optimize=True)

    tau138 = zeros((N, N, M, M))

    tau138 += np.einsum("ikac,kjcb->ijab", tau0, tau123, optimize=True)

    tau240 -= 2 * np.einsum("ijab->ijab", tau138, optimize=True)

    tau138 = None

    tau221 = zeros((N, N, M, M))

    tau221 -= np.einsum("ijab->ijab", tau123, optimize=True)

    tau229 -= 2 * np.einsum("ijab->ijab", tau123, optimize=True)

    tau265 = zeros((N, N, M, M))

    tau265 += 2 * np.einsum("ijba->ijab", tau123, optimize=True)

    r2 -= np.einsum("bc,jiac->abij", tau5, tau123, optimize=True) / 4

    tau123 = None

    tau7 = zeros((N, N))

    tau7 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau75 += np.einsum("ij->ij", tau7, optimize=True)

    tau192 = zeros((N, N, M, M))

    tau192 += np.einsum("kl,ikljab->ijab", tau75, tau108, optimize=True)

    tau108 = None

    tau240 -= 2 * np.einsum("ijab->ijab", tau192, optimize=True)

    tau192 = None

    tau310 -= np.einsum("ml,jimk->ijkl", tau75, tau3, optimize=True)

    tau136 = zeros((N, N, M, M))

    tau136 -= np.einsum("ik,jkab->ijab", tau7, u[o, o, v, v], optimize=True)

    tau296 = zeros((N, N, M, M))

    tau296 -= 2 * np.einsum("jiba->ijab", tau136, optimize=True)

    tau224 = zeros((N, M))

    tau224 -= np.einsum("ja,ij->ia", f[o, v], tau7, optimize=True)

    tau239 -= 2 * np.einsum("ia->ia", tau224, optimize=True)

    tau224 = None

    tau225 = zeros((N, M))

    tau225 -= np.einsum("jk,kija->ia", tau7, u[o, o, o, v], optimize=True)

    tau239 -= 2 * np.einsum("ia->ia", tau225, optimize=True)

    tau225 = None

    tau294 = zeros((N, N))

    tau294 += np.einsum("ij->ij", tau7, optimize=True)

    r1 -= np.einsum("kj,jika->ai", tau7, tau2, optimize=True) / 2

    tau2 = None

    r2 -= np.einsum("jk,ikba->abij", tau7, tau136, optimize=True) / 4

    tau136 = None

    tau10 = zeros((N, M, M, M))

    tau10 -= np.einsum("daji,jbdc->iabc", t2, u[o, v, v, v], optimize=True)

    tau11 = zeros((N, M, M, M))

    tau11 += 2 * np.einsum("iabc->iabc", tau10, optimize=True)

    tau16 = zeros((N, M, M, M))

    tau16 += 2 * np.einsum("daji,jdbc->iabc", l2, tau10, optimize=True)

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("ci,jabc->ijab", t1, tau10, optimize=True)

    tau56 = zeros((N, N, M, M))

    tau56 += np.einsum("ijab->ijab", tau54, optimize=True)

    tau54 = None

    tau70 = zeros((N, M, M, M))

    tau70 -= 2 * np.einsum("iabc->iabc", tau10, optimize=True)

    tau97 = zeros((N, M, M, M))

    tau97 -= 2 * np.einsum("iabc->iabc", tau10, optimize=True)

    tau97 += 2 * np.einsum("ibac->iabc", tau10, optimize=True)

    tau195 = zeros((N, M, M, M))

    tau195 -= 2 * np.einsum("iabc->iabc", tau10, optimize=True)

    tau195 += 2 * np.einsum("ibac->iabc", tau10, optimize=True)

    tau10 = None

    tau11 += np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau12 = zeros((N, N, N, M))

    tau12 += np.einsum("bcij,kbca->ijka", l2, tau11, optimize=True)

    tau15 += 2 * np.einsum("jika->ijka", tau12, optimize=True)

    tau268 -= 2 * np.einsum("kjia->ijka", tau12, optimize=True)

    tau12 = None

    tau66 = zeros((N, M))

    tau66 += np.einsum("bcji,jbca->ia", l2, tau11, optimize=True)

    tau11 = None

    tau69 += 2 * np.einsum("ia->ia", tau66, optimize=True)

    tau239 += 2 * np.einsum("ia->ia", tau66, optimize=True)

    tau66 = None

    tau13 = zeros((N, N, N, M))

    tau13 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau14 -= np.einsum("jika->ijka", tau13, optimize=True)

    tau15 += 4 * np.einsum("bali,lkjb->ijka", l2, tau14, optimize=True)

    tau14 = None

    tau17 = zeros((N, N, N, M))

    tau17 -= np.einsum("kjia->ijka", tau13, optimize=True)

    tau67 -= np.einsum("kija->ijka", tau13, optimize=True)

    tau77 -= np.einsum("kjia->ijka", tau13, optimize=True)

    tau113 -= np.einsum("kjia->ijka", tau13, optimize=True)

    tau205 -= np.einsum("jika->ijka", tau13, optimize=True)

    tau286 -= np.einsum("jika->ijka", tau13, optimize=True)

    tau13 = None

    tau15 -= 4 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    r1 += np.einsum("bajk,jkib->ai", t2, tau15, optimize=True) / 8

    tau15 = None

    tau16 += np.einsum("ibca->iabc", u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bcji,jbac->ai", t2, tau16, optimize=True) / 2

    tau16 = None

    tau17 -= 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    r1 += np.einsum("ljka,jkli->ai", tau17, tau3, optimize=True) / 8

    tau17 = None

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau28 -= 2 * np.einsum("jiba->ijab", tau20, optimize=True)

    tau85 += 2 * np.einsum("jiba->ijab", tau20, optimize=True)

    tau173 -= 2 * np.einsum("jiba->ijab", tau20, optimize=True)

    tau229 += 2 * np.einsum("jiba->ijab", tau20, optimize=True)

    tau259 = zeros((N, N, M, M))

    tau259 += 2 * np.einsum("jiba->ijab", tau20, optimize=True)

    tau265 += 2 * np.einsum("jiba->ijab", tau20, optimize=True)

    r2 += np.einsum("jiba->abij", tau20, optimize=True) / 2

    tau20 = None

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum("acik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("ijab->ijab", tau22, optimize=True)

    tau51 = zeros((N, N, M, M))

    tau51 += 2 * np.einsum("ijab->ijab", tau22, optimize=True)

    tau55 = zeros((N, N, M, M))

    tau55 += np.einsum("bcjk,ikac->ijab", t2, tau22, optimize=True)

    tau56 += np.einsum("ijab->ijab", tau55, optimize=True)

    tau55 = None

    tau62 = zeros((N, N, M, M))

    tau62 += 4 * np.einsum("ijab->ijab", tau56, optimize=True)

    tau62 -= 4 * np.einsum("ijba->ijab", tau56, optimize=True)

    tau56 = None

    tau84 = zeros((N, N, M, M))

    tau84 += np.einsum("caki,kjcb->ijab", l2, tau22, optimize=True)

    tau85 += 8 * np.einsum("ijab->ijab", tau84, optimize=True)

    tau104 = zeros((N, N, M, M))

    tau104 += 2 * np.einsum("ijab->ijab", tau84, optimize=True)

    tau84 = None

    tau100 = zeros((N, N, M, M))

    tau100 += np.einsum("jiab->ijab", tau22, optimize=True)

    tau115 = zeros((N, N, M, M))

    tau115 -= np.einsum("jiab->ijab", tau22, optimize=True)

    tau200 = zeros((N, N, M, M))

    tau200 += np.einsum("ijab->ijab", tau22, optimize=True)

    tau244 = zeros((M, M, M, M))

    tau244 += np.einsum("ijab,ijcd->abcd", tau0, tau22, optimize=True)

    tau22 = None

    tau245 += 4 * np.einsum("acbd->abcd", tau244, optimize=True)

    tau244 = None

    tau23 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("caki,kjcb->ijab", l2, tau23, optimize=True)

    tau28 -= 4 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau28 -= 4 * np.einsum("jiba->ijab", tau24, optimize=True)

    tau171 = zeros((N, N, M, M))

    tau171 -= 4 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau173 -= 4 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau174 = zeros((N, N, M, M))

    tau174 += np.einsum("jkbc,ikca->ijab", tau0, tau173, optimize=True)

    tau173 = None

    tau240 -= np.einsum("ijba->ijab", tau174, optimize=True)

    tau174 = None

    tau213 = zeros((N, N, M, M))

    tau213 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau221 += 2 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau229 -= 4 * np.einsum("ijba->ijab", tau24, optimize=True)

    tau229 -= 4 * np.einsum("jiab->ijab", tau24, optimize=True)

    tau275 = zeros((N, N, N, N))

    tau275 -= np.einsum("abij,klba->ijkl", t2, tau24, optimize=True)

    tau24 = None

    tau283 = zeros((N, N, N, N))

    tau283 += 2 * np.einsum("ljik->ijkl", tau275, optimize=True)

    tau275 = None

    tau48 = zeros((N, N, M, M))

    tau48 += np.einsum("kiac,kjbc->ijab", tau0, tau23, optimize=True)

    tau52 += 4 * np.einsum("jiba->ijab", tau48, optimize=True)

    tau142 = zeros((N, N, M, M))

    tau142 += 2 * np.einsum("jiab->ijab", tau48, optimize=True)

    tau48 = None

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum("ikca,jkcb->ijab", tau0, tau23, optimize=True)

    tau52 += 4 * np.einsum("ijab->ijab", tau49, optimize=True)

    tau150 -= 2 * np.einsum("ijab->ijab", tau49, optimize=True)

    tau49 = None

    tau52 -= 2 * np.einsum("bc,jiac->ijab", tau5, tau23, optimize=True)

    tau76 = zeros((N, N, N, M))

    tau76 += np.einsum("bi,jkab->ijka", t1, tau23, optimize=True)

    tau77 += 2 * np.einsum("jkia->ijka", tau76, optimize=True)

    tau113 += 2 * np.einsum("jkia->ijka", tau76, optimize=True)

    tau286 -= 4 * np.einsum("jika->ijka", tau76, optimize=True)

    tau76 = None

    tau149 = zeros((N, N, M, M))

    tau149 += np.einsum("klab,kilj->ijab", tau23, tau3, optimize=True)

    tau150 += np.einsum("ijab->ijab", tau149, optimize=True)

    tau149 = None

    tau207 = zeros((N, N, N, N, M, M))

    tau207 += np.einsum("caij,klcb->ijklab", l2, tau23, optimize=True)

    tau208 -= np.einsum("jilkab->ijklab", tau207, optimize=True)

    tau209 = zeros((N, N, N, M))

    tau209 += np.einsum("bl,iljkab->ijka", t1, tau208, optimize=True)

    tau208 = None

    tau218 -= 2 * np.einsum("ijka->ijka", tau209, optimize=True)

    tau209 = None

    tau262 -= np.einsum("jilkab->ijklab", tau207, optimize=True)

    tau207 = None

    tau263 = zeros((N, N, N, M))

    tau263 += np.einsum("bl,ijlkba->ijka", t1, tau262, optimize=True)

    tau262 = None

    tau268 -= 4 * np.einsum("kjia->ijka", tau263, optimize=True)

    tau263 = None

    tau276 = zeros((N, N, N, N))

    tau276 += np.einsum("ijab,klab->ijkl", tau0, tau23, optimize=True)

    tau283 += 4 * np.einsum("ijlk->ijkl", tau276, optimize=True)

    tau276 = None

    tau25 = zeros((N, N, N, N))

    tau25 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau26 = zeros((N, N, N, N))

    tau26 += np.einsum("lkji->ijkl", tau25, optimize=True)

    tau118 = zeros((N, N, N, N))

    tau118 += np.einsum("lkji->ijkl", tau25, optimize=True)

    tau227 = zeros((N, N, N, N))

    tau227 += np.einsum("lkji->ijkl", tau25, optimize=True)

    tau274 = zeros((N, N, N, N))

    tau274 += np.einsum("mjln,imnk->ijkl", tau25, tau3, optimize=True)

    tau283 -= np.einsum("ijlk->ijkl", tau274, optimize=True)

    tau274 = None

    tau312 = zeros((N, N, N, N))

    tau312 += np.einsum("lkji->ijkl", tau25, optimize=True)

    tau25 = None

    tau26 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("abkl,ijkl->ijab", l2, tau26, optimize=True)

    tau28 -= np.einsum("jiba->ijab", tau27, optimize=True)

    tau171 -= np.einsum("jiba->ijab", tau27, optimize=True)

    tau27 = None

    tau172 = zeros((N, N, M, M))

    tau172 += np.einsum("jkbc,kiac->ijab", tau0, tau171, optimize=True)

    tau171 = None

    tau240 -= np.einsum("jiab->ijab", tau172, optimize=True)

    tau172 = None

    tau52 -= 2 * np.einsum("klba,likj->ijab", tau0, tau26, optimize=True)

    tau62 += np.einsum("bakl,klji->ijab", t2, tau26, optimize=True)

    tau106 = zeros((N, N))

    tau106 += np.einsum("mjkl,klmi->ij", tau26, tau3, optimize=True)

    tau26 = None

    tau130 += np.einsum("ji->ij", tau106, optimize=True)

    tau291 += np.einsum("ji->ij", tau106, optimize=True)

    tau106 = None

    tau28 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau52 += np.einsum("cakj,ikcb->ijab", t2, tau28, optimize=True)

    tau28 = None

    tau29 = zeros((M, M, M, M))

    tau29 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("ijcd,cadb->ijab", tau23, tau29, optimize=True)

    tau23 = None

    tau52 -= 2 * np.einsum("jiba->ijab", tau30, optimize=True)

    tau142 -= np.einsum("jiab->ijab", tau30, optimize=True)

    tau30 = None

    tau47 -= np.einsum("bacd,jkid->ijkabc", tau29, u[o, o, o, v], optimize=True)

    tau88 = zeros((N, M, M, M))

    tau88 += np.einsum("dabe,idec->iabc", tau29, u[o, v, v, v], optimize=True)

    tau94 += 2 * np.einsum("ibac->iabc", tau88, optimize=True)

    tau162 += np.einsum("iabc->iabc", tau88, optimize=True)

    tau163 = zeros((N, N, N, M, M, M))

    tau163 += np.einsum("dcjk,iadb->ijkabc", l2, tau162, optimize=True)

    tau162 = None

    tau164 -= np.einsum("kjiabc->ijkabc", tau163, optimize=True)

    tau163 = None

    tau203 += np.einsum("iabc->iabc", tau88, optimize=True)

    tau250 = zeros((N, M, M, M))

    tau250 += np.einsum("iabc->iabc", tau88, optimize=True)

    tau88 = None

    tau91 = zeros((N, N, M, M, M, M))

    tau91 += np.einsum("abce,jied->ijabcd", tau29, u[o, o, v, v], optimize=True)

    tau92 = zeros((N, M, M, M))

    tau92 -= np.einsum("dj,ijdabc->iabc", t1, tau91, optimize=True)

    tau94 -= np.einsum("ibac->iabc", tau92, optimize=True)

    tau250 -= np.einsum("iabc->iabc", tau92, optimize=True)

    tau92 = None

    tau198 = zeros((N, N, M, M, M, M))

    tau198 -= np.einsum("ijcabd->ijabcd", tau91, optimize=True)

    tau91 = None

    tau175 = zeros((N, N, M, M))

    tau175 += np.einsum("ijdc,acbd->ijab", tau0, tau29, optimize=True)

    tau178 += np.einsum("ijab->ijab", tau175, optimize=True)

    tau175 = None

    tau179 = zeros((N, N, M, M))

    tau179 += np.einsum("ikac,kjcb->ijab", tau178, u[o, o, v, v], optimize=True)

    tau178 = None

    tau240 -= 2 * np.einsum("ijab->ijab", tau179, optimize=True)

    tau179 = None

    tau242 = zeros((M, M, M, M))

    tau242 -= np.einsum("aefb,cedf->abcd", tau29, u[v, v, v, v], optimize=True)

    tau245 += 2 * np.einsum("acbd->abcd", tau242, optimize=True)

    tau242 = None

    tau302 -= np.einsum("afde,becf->abcd", tau29, tau29, optimize=True)

    r2 += np.einsum("abcd,jicd->abij", tau302, u[o, o, v, v], optimize=True) / 4

    tau302 = None

    tau31 = zeros((M, M, M, M))

    tau31 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau32 = zeros((M, M, M, M))

    tau32 += np.einsum("badc->abcd", tau31, optimize=True)

    tau243 = zeros((M, M, M, M))

    tau243 -= np.einsum("aefb,cedf->abcd", tau29, tau31, optimize=True)

    tau31 = None

    tau245 -= np.einsum("abcd->abcd", tau243, optimize=True)

    tau243 = None

    tau246 = zeros((N, N, M, M))

    tau246 += np.einsum("cdij,acdb->ijab", l2, tau245, optimize=True)

    tau245 = None

    tau272 = zeros((N, N, M, M))

    tau272 -= 2 * np.einsum("jiab->ijab", tau246, optimize=True)

    tau246 = None

    tau32 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau33 = zeros((N, N, M, M))

    tau33 += np.einsum("ijcd,cadb->ijab", tau0, tau32, optimize=True)

    tau52 -= 2 * np.einsum("ijab->ijab", tau33, optimize=True)

    tau150 += np.einsum("ijab->ijab", tau33, optimize=True)

    tau33 = None

    tau96 = zeros((N, M, M, M))

    tau96 += np.einsum("di,abdc->iabc", t1, tau32, optimize=True)

    tau97 -= np.einsum("ibac->iabc", tau96, optimize=True)

    tau96 = None

    tau99 = zeros((M, M))

    tau99 += np.einsum("cd,cadb->ab", tau5, tau32, optimize=True)

    tau102 -= 2 * np.einsum("ab->ab", tau99, optimize=True)

    tau99 = None

    tau197 = zeros((N, N, M, M, M, M))

    tau197 += np.einsum("eaij,ebcd->ijabcd", l2, tau32, optimize=True)

    tau198 += np.einsum("jiabdc->ijabcd", tau197, optimize=True)

    tau197 = None

    tau199 = zeros((N, M, M, M))

    tau199 += np.einsum("dj,ijabdc->iabc", t1, tau198, optimize=True)

    tau198 = None

    tau203 -= np.einsum("iabc->iabc", tau199, optimize=True)

    tau199 = None

    tau311 = zeros((N, M, M, M))

    tau311 += np.einsum("di,dacb->iabc", l1, tau32, optimize=True)

    tau32 = None

    r2 += np.einsum("cj,icba->abij", l1, tau311, optimize=True) / 2

    tau311 = None

    tau35 = zeros((N, N, N, N, M, M))

    tau35 += np.einsum("caji,bckl->ijklab", l2, t2, optimize=True)

    tau36 = zeros((N, N, N, M, M, M))

    tau36 -= np.einsum("ldbc,ijklda->ijkabc", u[o, v, v, v], tau35, optimize=True)

    tau47 += 2 * np.einsum("kjicba->ijkabc", tau36, optimize=True)

    tau147 = zeros((N, N, N, M, M, M))

    tau147 -= 2 * np.einsum("ikjbca->ijkabc", tau36, optimize=True)

    tau36 = None

    tau247 = zeros((N, N, N, N, N, M))

    tau247 -= np.einsum("mbca,ijklbc->ijklma", u[o, v, v, v], tau35, optimize=True)

    tau248 = zeros((N, N, N, M, M, M))

    tau248 += np.einsum("abml,ijmlkc->ijkabc", l2, tau247, optimize=True)

    tau252 += np.einsum("ijkbac->ijkabc", tau248, optimize=True)

    tau248 = None

    tau278 += np.einsum("ilkjma->ijklma", tau247, optimize=True)

    tau247 = None

    tau37 = zeros((N, N, N, M, M, M))

    tau37 -= np.einsum("adij,kbdc->ijkabc", t2, u[o, v, v, v], optimize=True)

    tau38 = zeros((N, N, N, M, M, M))

    tau38 += np.einsum("dali,jlkbdc->ijkabc", l2, tau37, optimize=True)

    tau37 = None

    tau47 -= 4 * np.einsum("kijbca->ijkabc", tau38, optimize=True)

    tau139 = zeros((N, N, N, M, M, M))

    tau139 += 2 * np.einsum("kijbca->ijkabc", tau38, optimize=True)

    tau147 -= 2 * np.einsum("ijkabc->ijkabc", tau38, optimize=True)

    tau38 = None

    tau39 = zeros((N, N, N, M))

    tau39 -= np.einsum("bi,jkba->ijka", t1, u[o, o, v, v], optimize=True)

    tau40 = zeros((N, N, N, M))

    tau40 += np.einsum("kjia->ijka", tau39, optimize=True)

    tau121 = zeros((N, N, N, M))

    tau121 += np.einsum("kjia->ijka", tau39, optimize=True)

    tau156 = zeros((N, N, N, M, M, M))

    tau156 -= np.einsum("lmkc,lijmab->ijkabc", tau39, tau155, optimize=True)

    tau155 = None

    tau164 += 2 * np.einsum("ijkcab->ijkabc", tau156, optimize=True)

    tau164 += 2 * np.einsum("kijabc->ijkabc", tau156, optimize=True)

    tau156 = None

    tau182 = zeros((N, N, N, M))

    tau182 -= np.einsum("ikja->ijka", tau39, optimize=True)

    tau231 = zeros((N, M))

    tau231 += np.einsum("jk,jkia->ia", tau75, tau39, optimize=True)

    tau239 -= 2 * np.einsum("ia->ia", tau231, optimize=True)

    tau231 = None

    tau312 -= 2 * np.einsum("al,kija->ijkl", t1, tau39, optimize=True)

    tau39 = None

    tau40 -= 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau41 = zeros((N, N, N, M, M, M))

    tau41 += np.einsum("mklc,lijmab->ijkabc", tau40, tau35, optimize=True)

    tau47 -= 2 * np.einsum("kijbca->ijkabc", tau41, optimize=True)

    tau139 += np.einsum("kijbca->ijkabc", tau41, optimize=True)

    tau41 = None

    tau110 = zeros((N, N, N, M))

    tau110 += np.einsum("limj,mkla->ijka", tau3, tau40, optimize=True)

    tau111 -= np.einsum("ikja->ijka", tau110, optimize=True)

    tau110 = None

    tau226 = zeros((N, N, N, N))

    tau226 += np.einsum("ai,jkla->ijkl", t1, tau40, optimize=True)

    tau40 = None

    tau227 -= 2 * np.einsum("ljik->ijkl", tau226, optimize=True)

    tau226 = None

    tau42 = zeros((N, N, N, N, N, M))

    tau42 -= np.einsum("abij,lkmb->ijklma", t2, u[o, o, o, v], optimize=True)

    tau45 = zeros((N, N, N, N, N, M))

    tau45 += np.einsum("kjmlia->ijklma", tau42, optimize=True)

    tau157 = zeros((N, N, N, N, N, M))

    tau157 -= np.einsum("bain,jklmnb->ijklma", l2, tau42, optimize=True)

    tau160 = zeros((N, N, N, N, N, M))

    tau160 += np.einsum("ikjmla->ijklma", tau157, optimize=True)

    tau278 += np.einsum("ikjmla->ijklma", tau157, optimize=True)

    tau157 = None

    tau304 = zeros((N, N, N, N, N, M))

    tau304 -= np.einsum("ijlkma->ijklma", tau42, optimize=True)

    tau42 = None

    tau43 = zeros((N, N, N, N, M, M))

    tau43 += np.einsum("acij,lkcb->ijklab", t2, u[o, o, v, v], optimize=True)

    tau44 = zeros((N, N, N, N, N, M))

    tau44 += np.einsum("bi,jklmab->ijklma", t1, tau43, optimize=True)

    tau45 += np.einsum("ikjmla->ijklma", tau44, optimize=True)

    tau46 = zeros((N, N, N, M, M, M))

    tau46 += np.einsum("ablm,miljkc->ijkabc", l2, tau45, optimize=True)

    tau45 = None

    tau47 += 2 * np.einsum("ikjbac->ijkabc", tau46, optimize=True)

    tau52 += np.einsum("ck,jikcba->ijab", t1, tau47, optimize=True)

    tau47 = None

    tau139 -= 2 * np.einsum("ikjbac->ijkabc", tau46, optimize=True)

    tau46 = None

    tau140 = zeros((N, N, M, M))

    tau140 += np.einsum("ck,ijkcab->ijab", t1, tau139, optimize=True)

    tau139 = None

    tau142 -= np.einsum("jiab->ijab", tau140, optimize=True)

    tau140 = None

    tau304 -= np.einsum("mijlka->ijklma", tau44, optimize=True)

    tau44 = None

    tau307 = zeros((N, N, N, N))

    tau307 += 2 * np.einsum("am,lkijma->ijkl", l1, tau304, optimize=True)

    tau304 = None

    tau158 = zeros((N, N, N, N, N, N, M, M))

    tau158 -= np.einsum("caij,klmncb->ijklmnab", l2, tau43, optimize=True)

    tau159 = zeros((N, N, N, N, N, M))

    tau159 += np.einsum("bn,injklmab->ijklma", t1, tau158, optimize=True)

    tau158 = None

    tau160 += np.einsum("ikjmla->ijklma", tau159, optimize=True)

    tau161 = zeros((N, N, N, M, M, M))

    tau161 += np.einsum("bclm,ilmjka->ijkabc", l2, tau160, optimize=True)

    tau160 = None

    tau164 -= np.einsum("ikjcba->ijkabc", tau161, optimize=True)

    tau161 = None

    tau278 += np.einsum("ikjmla->ijklma", tau159, optimize=True)

    tau159 = None

    tau279 = zeros((N, N, N, N))

    tau279 += np.einsum("am,ijkmla->ijkl", t1, tau278, optimize=True)

    tau278 = None

    tau283 += 2 * np.einsum("iljk->ijkl", tau279, optimize=True)

    tau279 = None

    tau50 = zeros((N, N, M, M))

    tau50 -= np.einsum("ci,jacb->ijab", t1, u[o, v, v, v], optimize=True)

    tau51 += np.einsum("ijab->ijab", tau50, optimize=True)

    tau100 += np.einsum("jiab->ijab", tau50, optimize=True)

    tau200 += np.einsum("ijab->ijab", tau50, optimize=True)

    tau285 = zeros((N, N, N, M))

    tau285 += np.einsum("bi,jkab->ijka", t1, tau50, optimize=True)

    tau50 = None

    tau286 += 2 * np.einsum("ijka->ijka", tau285, optimize=True)

    tau285 = None

    tau51 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau52 -= np.einsum("kilj,klab->ijab", tau3, tau51, optimize=True)

    tau51 = None

    tau52 -= 4 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    r1 += np.einsum("bj,jiab->ai", t1, tau52, optimize=True) / 4

    tau52 = None

    tau53 = zeros((N, N, M, M))

    tau53 -= np.einsum("acki,kbjc->ijab", t2, u[o, v, o, v], optimize=True)

    tau62 -= 4 * np.einsum("ijab->ijab", tau53, optimize=True)

    tau62 += 4 * np.einsum("ijba->ijab", tau53, optimize=True)

    tau62 += 4 * np.einsum("jiab->ijab", tau53, optimize=True)

    tau62 -= 4 * np.einsum("jiba->ijab", tau53, optimize=True)

    tau53 = None

    tau57 = zeros((M, M))

    tau57 -= np.einsum("ci,iacb->ab", t1, u[o, v, v, v], optimize=True)

    tau59 = zeros((M, M))

    tau59 += 2 * np.einsum("ab->ab", tau57, optimize=True)

    tau233 = zeros((M, M))

    tau233 += 2 * np.einsum("ab->ab", tau57, optimize=True)

    tau57 = None

    tau58 = zeros((M, M))

    tau58 -= np.einsum("acji,jicb->ab", t2, u[o, o, v, v], optimize=True)

    tau59 += np.einsum("ab->ab", tau58, optimize=True)

    tau233 += np.einsum("ab->ab", tau58, optimize=True)

    tau58 = None

    tau234 = zeros((N, M))

    tau234 += np.einsum("bi,ba->ia", l1, tau233, optimize=True)

    tau233 = None

    tau239 += 2 * np.einsum("ia->ia", tau234, optimize=True)

    tau234 = None

    tau59 -= 2 * np.einsum("ab->ab", f[v, v], optimize=True)

    tau62 += 2 * np.einsum("ac,cbji->ijab", tau59, t2, optimize=True)

    tau60 = zeros((N, M, M, M))

    tau60 += np.einsum("abkj,kjic->iabc", t2, u[o, o, o, v], optimize=True)

    tau61 = zeros((N, M, M, M))

    tau61 -= np.einsum("ibac->iabc", tau60, optimize=True)

    tau70 -= np.einsum("ibac->iabc", tau60, optimize=True)

    tau97 -= np.einsum("ibac->iabc", tau60, optimize=True)

    tau195 -= np.einsum("ibac->iabc", tau60, optimize=True)

    tau60 = None

    tau61 -= 2 * np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau62 += 2 * np.einsum("ci,jbac->ijab", t1, tau61, optimize=True)

    tau61 = None

    tau62 += 4 * np.einsum("baji->ijab", u[v, v, o, o], optimize=True)

    tau62 += 2 * np.einsum("dcji,badc->ijab", t2, u[v, v, v, v], optimize=True)

    r1 += np.einsum("bj,jiba->ai", l1, tau62, optimize=True) / 4

    tau62 = None

    tau65 = zeros((N, M))

    tau65 += np.einsum("bj,jiba->ia", t1, u[o, o, v, v], optimize=True)

    tau69 += 2 * np.einsum("ab,ib->ia", tau5, tau65, optimize=True)

    tau71 = zeros((N, M))

    tau71 += np.einsum("ia->ia", tau65, optimize=True)

    tau77 -= 2 * np.einsum("ib,abjk->ijka", tau65, t2, optimize=True)

    tau235 = zeros((N, N))

    tau235 += np.einsum("ai,ja->ij", t1, tau65, optimize=True)

    tau236 = zeros((N, N))

    tau236 += 2 * np.einsum("ij->ij", tau235, optimize=True)

    tau235 = None

    tau238 = zeros((N, M))

    tau238 += np.einsum("ja,ij->ia", tau65, tau75, optimize=True)

    tau65 = None

    tau239 += 2 * np.einsum("ia->ia", tau238, optimize=True)

    tau238 = None

    tau67 -= 2 * np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    tau68 = zeros((N, M))

    tau68 += np.einsum("bajk,jikb->ia", l2, tau67, optimize=True)

    tau67 = None

    tau69 -= np.einsum("ia->ia", tau68, optimize=True)

    tau239 -= np.einsum("ia->ia", tau68, optimize=True)

    tau68 = None

    tau69 -= 4 * np.einsum("ia->ia", f[o, v], optimize=True)

    r1 -= np.einsum("jb,baji->ai", tau69, t2, optimize=True) / 4

    tau69 = None

    tau70 -= 2 * np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    r1 -= np.einsum("bc,ibac->ai", tau5, tau70, optimize=True) / 4

    tau70 = None

    tau71 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau80 = zeros((N, N))

    tau80 += np.einsum("ai,ja->ij", t1, tau71, optimize=True)

    tau81 = zeros((N, N))

    tau81 += 2 * np.einsum("ji->ij", tau80, optimize=True)

    tau80 = None

    tau216 = zeros((N, N, N, M))

    tau216 += np.einsum("kb,ijab->ijka", tau71, tau0, optimize=True)

    tau218 -= 2 * np.einsum("ikja->ijka", tau216, optimize=True)

    tau216 = None

    tau223 = zeros((N, N, M, M))

    tau223 += np.einsum("ka,ijkb->ijab", tau71, tau180, optimize=True)

    tau180 = None

    tau240 += 4 * np.einsum("ijba->ijab", tau223, optimize=True)

    tau223 = None

    tau232 = zeros((N, M))

    tau232 += np.einsum("ab,ib->ia", tau5, tau71, optimize=True)

    tau239 += 2 * np.einsum("ia->ia", tau232, optimize=True)

    tau232 = None

    tau267 = zeros((N, N, N, M))

    tau267 += np.einsum("la,ijlk->ijka", tau71, tau3, optimize=True)

    tau268 += 2 * np.einsum("kjia->ijka", tau267, optimize=True)

    tau267 = None

    tau288 = zeros((N, M, M, M))

    tau288 += np.einsum("id,abdc->iabc", tau71, tau29, optimize=True)

    tau29 = None

    tau289 -= 2 * np.einsum("ibac->iabc", tau288, optimize=True)

    tau288 = None

    r1 -= np.einsum("jb,jiab->ai", tau71, tau74, optimize=True) / 4

    tau74 = None

    tau77 -= 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    r1 -= np.einsum("jk,kjia->ai", tau75, tau77, optimize=True) / 4

    tau77 = None

    tau78 = zeros((N, N))

    tau78 -= np.einsum("ak,kija->ij", t1, u[o, o, o, v], optimize=True)

    tau81 += 2 * np.einsum("ij->ij", tau78, optimize=True)

    tau318 = zeros((N, N))

    tau318 += np.einsum("ij->ij", tau78, optimize=True)

    tau78 = None

    tau79 = zeros((N, N))

    tau79 -= np.einsum("baik,kjba->ij", t2, u[o, o, v, v], optimize=True)

    tau81 += np.einsum("ji->ij", tau79, optimize=True)

    tau236 += np.einsum("ij->ij", tau79, optimize=True)

    tau79 = None

    tau237 = zeros((N, M))

    tau237 += np.einsum("aj,ji->ia", l1, tau236, optimize=True)

    tau236 = None

    tau239 += 2 * np.einsum("ia->ia", tau237, optimize=True)

    tau237 = None

    tau81 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    tau82 = zeros((N, N, N, M))

    tau82 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    r1 += np.einsum("kj,jkia->ai", tau81, tau82, optimize=True) / 2

    tau82 = None

    tau85 += 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau86 = zeros((M, M))

    tau86 += np.einsum("caij,ijcb->ab", t2, tau85, optimize=True)

    tau85 = None

    tau102 += np.einsum("ab->ab", tau86, optimize=True)

    tau86 = None

    tau94 -= 4 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau95 = zeros((M, M))

    tau95 += np.einsum("ci,iacb->ab", t1, tau94, optimize=True)

    tau94 = None

    tau102 -= 2 * np.einsum("ab->ab", tau95, optimize=True)

    tau95 = None

    tau97 -= 2 * np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau98 = zeros((M, M))

    tau98 += np.einsum("ci,icab->ab", l1, tau97, optimize=True)

    tau97 = None

    tau102 -= 4 * np.einsum("ab->ab", tau98, optimize=True)

    tau98 = None

    tau100 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau101 = zeros((M, M))

    tau101 += np.einsum("ij,jiab->ab", tau75, tau100, optimize=True)

    tau102 -= 4 * np.einsum("ab->ab", tau101, optimize=True)

    tau101 = None

    tau120 = zeros((N, M))

    tau120 += np.einsum("bj,ijba->ia", l1, tau100, optimize=True)

    tau100 = None

    tau126 += 2 * np.einsum("ia->ia", tau120, optimize=True)

    tau239 -= 4 * np.einsum("ia->ia", tau120, optimize=True)

    tau120 = None

    tau102 -= 8 * np.einsum("ab->ab", f[v, v], optimize=True)

    tau255 = zeros((N, N, M, M))

    tau255 += np.einsum("cb,caij->ijab", tau102, l2, optimize=True)

    tau272 -= np.einsum("jiab->ijab", tau255, optimize=True)

    tau255 = None

    r1 -= np.einsum("bi,ab->ai", t1, tau102, optimize=True) / 8

    tau102 = None

    tau104 += np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau105 = zeros((N, N))

    tau105 += np.einsum("abki,kjab->ij", t2, tau104, optimize=True)

    tau104 = None

    tau130 += 4 * np.einsum("ji->ij", tau105, optimize=True)

    tau291 += 4 * np.einsum("ji->ij", tau105, optimize=True)

    tau105 = None

    tau111 -= 4 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau112 = zeros((N, N))

    tau112 += np.einsum("ak,kija->ij", t1, tau111, optimize=True)

    tau111 = None

    tau130 -= 2 * np.einsum("ij->ij", tau112, optimize=True)

    tau291 -= 2 * np.einsum("ij->ij", tau112, optimize=True)

    tau112 = None

    tau113 -= 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau114 = zeros((N, N))

    tau114 += np.einsum("ak,ikja->ij", l1, tau113, optimize=True)

    tau113 = None

    tau130 -= 4 * np.einsum("ij->ij", tau114, optimize=True)

    tau291 -= 4 * np.einsum("ij->ij", tau114, optimize=True)

    tau114 = None

    tau115 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau116 = zeros((N, N))

    tau116 += np.einsum("ab,ijab->ij", tau5, tau115, optimize=True)

    tau115 = None

    tau130 += 4 * np.einsum("ij->ij", tau116, optimize=True)

    tau291 += 4 * np.einsum("ij->ij", tau116, optimize=True)

    tau116 = None

    tau117 = zeros((N, N, N, N))

    tau117 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau118 += 2 * np.einsum("kjil->ijkl", tau117, optimize=True)

    tau261 = zeros((N, N, N, M))

    tau261 += np.einsum("al,ijkl->ijka", l1, tau117, optimize=True)

    tau268 += 4 * np.einsum("ikja->ijka", tau261, optimize=True)

    tau261 = None

    tau280 = zeros((N, N, N, N))

    tau280 += np.einsum("kjil->ijkl", tau117, optimize=True)

    tau117 = None

    tau118 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau119 = zeros((N, N))

    tau119 += np.einsum("kl,likj->ij", tau75, tau118, optimize=True)

    tau130 -= 2 * np.einsum("ij->ij", tau119, optimize=True)

    tau291 -= 2 * np.einsum("ij->ij", tau119, optimize=True)

    tau119 = None

    tau141 = zeros((N, N, M, M))

    tau141 += np.einsum("klab,likj->ijab", tau0, tau118, optimize=True)

    tau142 -= np.einsum("ijab->ijab", tau141, optimize=True)

    tau141 = None

    tau143 = zeros((N, N, M, M))

    tau143 += np.einsum("cbkj,ikac->ijab", l2, tau142, optimize=True)

    tau142 = None

    tau240 -= 2 * np.einsum("jiab->ijab", tau143, optimize=True)

    tau143 = None

    tau121 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau122 = zeros((N, M))

    tau122 += np.einsum("jk,kija->ia", tau75, tau121, optimize=True)

    tau126 -= np.einsum("ia->ia", tau122, optimize=True)

    tau122 = None

    tau146 = zeros((N, N, N, M, M, M))

    tau146 += np.einsum("mklc,ilmjab->ijkabc", tau121, tau35, optimize=True)

    tau35 = None

    tau147 -= 2 * np.einsum("ijkabc->ijkabc", tau146, optimize=True)

    tau146 = None

    tau185 = zeros((N, N, N, M))

    tau185 += np.einsum("mkla,limj->ijka", tau121, tau3, optimize=True)

    tau186 -= np.einsum("ijka->ijka", tau185, optimize=True)

    tau282 = zeros((N, N, N, N))

    tau282 += np.einsum("al,ijka->ijkl", t1, tau186, optimize=True)

    tau283 -= 2 * np.einsum("ijkl->ijkl", tau282, optimize=True)

    tau282 = None

    tau218 -= np.einsum("ikja->ijka", tau185, optimize=True)

    tau185 = None

    tau202 = zeros((N, M, M, M))

    tau202 += np.einsum("jkab,kijc->iabc", tau0, tau121, optimize=True)

    tau0 = None

    tau203 -= 2 * np.einsum("iabc->iabc", tau202, optimize=True)

    tau250 -= 2 * np.einsum("iabc->iabc", tau202, optimize=True)

    tau202 = None

    tau251 = zeros((N, N, N, M, M, M))

    tau251 += np.einsum("dcjk,iadb->ijkabc", l2, tau250, optimize=True)

    tau250 = None

    tau252 += np.einsum("kjibca->ijkabc", tau251, optimize=True)

    tau251 = None

    tau253 = zeros((N, N, M, M))

    tau253 += np.einsum("ck,ijkcab->ijab", t1, tau252, optimize=True)

    tau252 = None

    tau272 -= 4 * np.einsum("ijab->ijab", tau253, optimize=True)

    tau253 = None

    tau217 = zeros((N, N, N, M))

    tau217 += np.einsum("il,ljka->ijka", tau75, tau121, optimize=True)

    tau121 = None

    tau218 -= np.einsum("ijka->ijka", tau217, optimize=True)

    tau217 = None

    tau124 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau125 = zeros((N, M))

    tau125 += np.einsum("bj,jiba->ia", t1, tau124, optimize=True)

    tau124 = None

    tau126 += np.einsum("ia->ia", tau125, optimize=True)

    tau125 = None

    tau126 += 2 * np.einsum("ia->ia", f[o, v], optimize=True)

    tau127 = zeros((N, N))

    tau127 += np.einsum("ai,ja->ij", t1, tau126, optimize=True)

    tau130 += 4 * np.einsum("ji->ij", tau127, optimize=True)

    tau291 += 4 * np.einsum("ji->ij", tau127, optimize=True)

    tau127 = None

    tau128 = zeros((N, M))

    tau128 -= np.einsum("bj,abji->ia", l1, t2, optimize=True)

    tau129 = zeros((N, N))

    tau129 += np.einsum("ia,ja->ij", tau128, tau71, optimize=True)

    tau130 += 8 * np.einsum("ji->ij", tau129, optimize=True)

    tau291 += 8 * np.einsum("ji->ij", tau129, optimize=True)

    tau129 = None

    tau292 = zeros((N, N, M, M))

    tau292 += np.einsum("ik,abkj->ijab", tau291, l2, optimize=True)

    tau291 = None

    tau299 = zeros((N, N, M, M))

    tau299 += np.einsum("jiba->ijab", tau292, optimize=True)

    tau292 = None

    tau133 = zeros((N, N, N, M))

    tau133 += np.einsum("kb,baij->ijka", tau128, l2, optimize=True)

    tau181 += np.einsum("ijka->ijka", tau133, optimize=True)

    tau271 = zeros((N, N, M, M))

    tau271 += np.einsum("ka,ijkb->ijab", tau71, tau133, optimize=True)

    tau71 = None

    tau272 += 8 * np.einsum("ijba->ijab", tau271, optimize=True)

    tau271 = None

    r2 -= np.einsum("ijkc,kcab->abij", tau133, u[o, v, v, v], optimize=True)

    tau133 = None

    tau166 = zeros((N, N, M, M))

    tau166 -= np.einsum("ic,jabc->ijab", tau128, u[o, v, v, v], optimize=True)

    tau169 = zeros((N, N, M, M))

    tau169 += 4 * np.einsum("ijab->ijab", tau166, optimize=True)

    tau166 = None

    tau188 = zeros((N, M))

    tau188 += np.einsum("ia->ia", tau128, optimize=True)

    tau305 = zeros((N, M))

    tau305 += 2 * np.einsum("ia->ia", tau128, optimize=True)

    tau307 -= 4 * np.einsum("ka,ijla->ijkl", tau128, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("ib,ab->ai", tau128, tau59, optimize=True) / 2

    tau59 = None

    r1 -= np.einsum("ja,ji->ai", tau128, tau81, optimize=True) / 2

    tau128 = None

    tau81 = None

    tau130 += 8 * np.einsum("ij->ij", f[o, o], optimize=True)

    r1 -= np.einsum("aj,ji->ai", t1, tau130, optimize=True) / 8

    tau130 = None

    tau131 = zeros((N, N, N, M, M, M))

    tau131 -= np.einsum("adji,kdbc->ijkabc", l2, u[o, v, v, v], optimize=True)

    tau132 = zeros((N, N, M, M))

    tau132 -= np.einsum("ck,jikcab->ijab", t1, tau131, optimize=True)

    tau259 += 4 * np.einsum("ijba->ijab", tau132, optimize=True)

    tau265 += 4 * np.einsum("ijba->ijab", tau132, optimize=True)

    r2 += np.einsum("jiab->abij", tau132, optimize=True)

    tau132 = None

    tau153 = zeros((N, N, N, M, M, M))

    tau153 -= np.einsum("jiml,lkmabc->ijkabc", tau3, tau131, optimize=True)

    tau164 += np.einsum("ijkacb->ijkabc", tau153, optimize=True)

    tau153 = None

    tau164 += 2 * np.einsum("ijkacb->ijkabc", tau131, optimize=True)

    tau165 = zeros((N, N, M, M))

    tau165 += np.einsum("ck,ikjacb->ijab", t1, tau164, optimize=True)

    tau164 = None

    tau240 += 2 * np.einsum("ijab->ijab", tau165, optimize=True)

    tau165 = None

    tau211 = zeros((N, N, M, M))

    tau211 -= np.einsum("ck,kijabc->ijab", t1, tau131, optimize=True)

    tau131 = None

    tau213 += np.einsum("ijab->ijab", tau211, optimize=True)

    tau221 += 2 * np.einsum("ijab->ijab", tau211, optimize=True)

    tau229 -= 2 * np.einsum("ijba->ijab", tau211, optimize=True)

    tau229 -= 4 * np.einsum("jiab->ijab", tau211, optimize=True)

    tau211 = None

    tau134 = zeros((N, N, M, M))

    tau134 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau273 = zeros((N, N, M, M))

    tau273 -= np.einsum("jk,ikab->ijab", tau7, tau134, optimize=True)

    tau299 -= 4 * np.einsum("ijba->ijab", tau273, optimize=True)

    tau273 = None

    tau296 -= 4 * np.einsum("jiba->ijab", tau134, optimize=True)

    tau301 = zeros((N, N, M, M))

    tau301 -= np.einsum("ijba->ijab", tau134, optimize=True)

    r2 += np.einsum("klab,ijlk->abij", tau134, tau3, optimize=True) / 2

    tau134 = None

    tau144 = zeros((N, N, N, M))

    tau144 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau145 = zeros((N, N, N, M, M, M))

    tau145 += np.einsum("imla,jmklbc->ijkabc", tau144, tau43, optimize=True)

    tau43 = None

    tau147 += np.einsum("ijkabc->ijkabc", tau145, optimize=True)

    tau145 = None

    tau148 = zeros((N, N, M, M))

    tau148 += np.einsum("ck,ijkcab->ijab", t1, tau147, optimize=True)

    tau147 = None

    tau150 -= np.einsum("ijab->ijab", tau148, optimize=True)

    tau148 = None

    tau151 = zeros((N, N, M, M))

    tau151 += np.einsum("cbkj,ikca->ijab", l2, tau150, optimize=True)

    tau150 = None

    tau240 += 2 * np.einsum("ijba->ijab", tau151, optimize=True)

    tau151 = None

    tau187 = zeros((N, N, M, M))

    tau187 += np.einsum("kjlb,ikla->ijab", tau144, tau186, optimize=True)

    tau186 = None

    tau240 += 2 * np.einsum("ijba->ijab", tau187, optimize=True)

    tau187 = None

    tau194 = zeros((N, N, M, M))

    tau194 += np.einsum("kjlb,kila->ijab", tau144, tau193, optimize=True)

    tau193 = None

    tau240 += 4 * np.einsum("jiab->ijab", tau194, optimize=True)

    tau194 = None

    tau210 = zeros((N, N, N, M))

    tau210 += np.einsum("limj,mkla->ijka", tau118, tau144, optimize=True)

    tau118 = None

    tau218 -= np.einsum("jkia->ijka", tau210, optimize=True)

    tau210 = None

    tau256 = zeros((N, N, N, N))

    tau256 += np.einsum("ak,ijla->ijkl", t1, tau144, optimize=True)

    tau257 += 2 * np.einsum("ijlk->ijkl", tau256, optimize=True)

    tau256 = None

    tau258 = zeros((N, N, M, M))

    tau258 += np.einsum("ijkl,klab->ijab", tau257, u[o, o, v, v], optimize=True)

    tau259 -= np.einsum("jiba->ijab", tau258, optimize=True)

    tau265 -= np.einsum("jiba->ijab", tau258, optimize=True)

    tau258 = None

    tau266 = zeros((N, N, N, M))

    tau266 += np.einsum("bk,ijba->ijka", t1, tau265, optimize=True)

    tau265 = None

    tau268 -= np.einsum("jkia->ijka", tau266, optimize=True)

    tau266 = None

    tau264 = zeros((N, N, N, M))

    tau264 += np.einsum("ijlm,lmka->ijka", tau257, u[o, o, o, v], optimize=True)

    tau257 = None

    tau268 -= np.einsum("kjia->ijka", tau264, optimize=True)

    tau264 = None

    tau270 = zeros((N, N, M, M))

    tau270 += np.einsum("ka,ijkb->ijab", tau126, tau144, optimize=True)

    tau126 = None

    tau272 += 4 * np.einsum("jiba->ijab", tau270, optimize=True)

    tau270 = None

    tau167 += 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau168 = zeros((N, N, M, M))

    tau168 += np.einsum("ikca,kjcb->ijab", tau167, u[o, o, v, v], optimize=True)

    tau167 = None

    tau169 += np.einsum("ijab->ijab", tau168, optimize=True)

    tau168 = None

    tau169 += 4 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau170 = zeros((N, N, M, M))

    tau170 += np.einsum("cbkj,kica->ijab", l2, tau169, optimize=True)

    tau169 = None

    tau240 -= np.einsum("jiba->ijab", tau170, optimize=True)

    tau170 = None

    tau182 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau183 = zeros((N, N, M, M))

    tau183 += np.einsum("kila,kljb->ijab", tau181, tau182, optimize=True)

    tau181 = None

    tau240 += 4 * np.einsum("ijab->ijab", tau183, optimize=True)

    tau183 = None

    tau212 = zeros((N, N, M, M))

    tau212 += np.einsum("kjlb,klia->ijab", tau144, tau182, optimize=True)

    tau182 = None

    tau213 -= np.einsum("jiba->ijab", tau212, optimize=True)

    tau214 = zeros((N, N, N, M))

    tau214 += np.einsum("bk,ijab->ijka", t1, tau213, optimize=True)

    tau218 -= 2 * np.einsum("ijka->ijka", tau214, optimize=True)

    tau214 = None

    tau220 = zeros((N, N, M, M))

    tau220 += np.einsum("bc,ijac->ijab", tau5, tau213, optimize=True)

    tau240 += 2 * np.einsum("ijba->ijab", tau220, optimize=True)

    tau220 = None

    tau254 = zeros((N, N, M, M))

    tau254 += np.einsum("klab,ijlk->ijab", tau213, tau3, optimize=True)

    tau213 = None

    tau272 += 4 * np.einsum("jiab->ijab", tau254, optimize=True)

    tau254 = None

    tau221 -= 2 * np.einsum("jiba->ijab", tau212, optimize=True)

    tau222 = zeros((N, N, M, M))

    tau222 += np.einsum("jk,ikab->ijab", tau75, tau221, optimize=True)

    tau75 = None

    tau221 = None

    tau240 += np.einsum("jiab->ijab", tau222, optimize=True)

    tau222 = None

    tau229 += 2 * np.einsum("ijba->ijab", tau212, optimize=True)

    tau212 = None

    tau188 += np.einsum("ai->ia", t1, optimize=True)

    tau189 = zeros((N, N, N, M))

    tau189 += np.einsum("ib,jkba->ijka", tau188, u[o, o, v, v], optimize=True)

    tau190 = zeros((N, N, N, M))

    tau190 -= np.einsum("ikja->ijka", tau189, optimize=True)

    tau268 += 4 * np.einsum("ikja->ijka", tau189, optimize=True)

    tau189 = None

    tau269 = zeros((N, N, M, M))

    tau269 += np.einsum("bk,kija->ijab", l1, tau268, optimize=True)

    tau268 = None

    tau272 -= 2 * np.einsum("jiba->ijab", tau269, optimize=True)

    tau269 = None

    tau293 = zeros((N, N))

    tau293 += np.einsum("aj,ia->ij", l1, tau188, optimize=True)

    tau188 = None

    tau294 += 2 * np.einsum("ji->ij", tau293, optimize=True)

    tau293 = None

    tau295 = zeros((N, N, M, M))

    tau295 += np.einsum("ik,kjab->ijab", tau294, u[o, o, v, v], optimize=True)

    tau294 = None

    tau299 -= 4 * np.einsum("ijba->ijab", tau295, optimize=True)

    tau295 = None

    tau190 -= np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau191 = zeros((N, N, M, M))

    tau191 += np.einsum("kjlb,kila->ijab", tau144, tau190, optimize=True)

    tau190 = None

    tau240 -= 4 * np.einsum("jiba->ijab", tau191, optimize=True)

    tau191 = None

    tau195 -= 2 * np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau196 = zeros((N, M, M, M))

    tau196 += np.einsum("dcji,jdab->iabc", l2, tau195, optimize=True)

    tau195 = None

    tau203 += np.einsum("ibca->iabc", tau196, optimize=True)

    tau196 = None

    tau200 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau201 = zeros((N, M, M, M))

    tau201 += np.einsum("jikc,jkab->iabc", tau144, tau200, optimize=True)

    tau144 = None

    tau203 -= 2 * np.einsum("ibca->iabc", tau201, optimize=True)

    tau201 = None

    tau204 = zeros((N, N, M, M))

    tau204 += np.einsum("cj,iacb->ijab", l1, tau203, optimize=True)

    tau203 = None

    tau240 -= 2 * np.einsum("jiab->ijab", tau204, optimize=True)

    tau204 = None

    tau215 = zeros((N, N, N, M))

    tau215 += np.einsum("bk,ijba->ijka", l1, tau200, optimize=True)

    tau200 = None

    tau218 += 2 * np.einsum("kjia->ijka", tau215, optimize=True)

    tau215 = None

    tau205 -= 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau206 = zeros((N, N, N, M))

    tau206 += np.einsum("balk,lijb->ijka", l2, tau205, optimize=True)

    tau205 = None

    tau218 += np.einsum("kjia->ijka", tau206, optimize=True)

    tau206 = None

    tau219 = zeros((N, N, M, M))

    tau219 += np.einsum("bk,ijka->ijab", l1, tau218, optimize=True)

    tau218 = None

    tau240 -= 2 * np.einsum("ijba->ijab", tau219, optimize=True)

    tau219 = None

    tau227 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau228 = zeros((N, N, M, M))

    tau228 += np.einsum("abkl,ijkl->ijab", l2, tau227, optimize=True)

    tau227 = None

    tau229 += np.einsum("jiba->ijab", tau228, optimize=True)

    tau296 += np.einsum("jiba->ijab", tau228, optimize=True)

    tau297 = zeros((N, N, M, M))

    tau297 += np.einsum("jk,kiab->ijab", tau1, tau296, optimize=True)

    tau296 = None

    tau1 = None

    tau299 -= 2 * np.einsum("jiba->ijab", tau297, optimize=True)

    tau297 = None

    tau298 = zeros((N, N, M, M))

    tau298 += np.einsum("jk,ikba->ijab", tau7, tau228, optimize=True)

    tau228 = None

    tau7 = None

    tau299 -= np.einsum("jiba->ijab", tau298, optimize=True)

    tau298 = None

    tau229 += 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau230 = zeros((N, M))

    tau230 += np.einsum("bj,jiba->ia", t1, tau229, optimize=True)

    tau229 = None

    tau239 -= np.einsum("ia->ia", tau230, optimize=True)

    tau230 = None

    tau240 -= np.einsum("ai,jb->ijab", l1, tau239, optimize=True)

    tau239 = None

    r2 += np.einsum("ijab->abij", tau240, optimize=True) / 4

    r2 -= np.einsum("ijba->abij", tau240, optimize=True) / 4

    r2 -= np.einsum("jiab->abij", tau240, optimize=True) / 4

    r2 += np.einsum("jiba->abij", tau240, optimize=True) / 4

    tau240 = None

    tau259 += 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau260 = zeros((N, N, M, M))

    tau260 += np.einsum("bc,ijca->ijab", tau5, tau259, optimize=True)

    tau259 = None

    tau5 = None

    tau272 += np.einsum("jiba->ijab", tau260, optimize=True)

    tau260 = None

    r2 += np.einsum("ijab->abij", tau272, optimize=True) / 8

    r2 -= np.einsum("ijba->abij", tau272, optimize=True) / 8

    tau272 = None

    tau280 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau281 = zeros((N, N, N, N))

    tau281 += np.einsum("minj,nkml->ijkl", tau280, tau3, optimize=True)

    tau280 = None

    tau3 = None

    tau283 += 2 * np.einsum("klij->ijkl", tau281, optimize=True)

    tau281 = None

    tau284 = zeros((N, N, M, M))

    tau284 += np.einsum("abkl,ikjl->ijab", l2, tau283, optimize=True)

    tau283 = None

    tau299 -= 2 * np.einsum("ijba->ijab", tau284, optimize=True)

    tau284 = None

    tau286 -= 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau287 = zeros((N, M, M, M))

    tau287 += np.einsum("bcjk,jkia->iabc", l2, tau286, optimize=True)

    tau286 = None

    tau289 -= np.einsum("icba->iabc", tau287, optimize=True)

    tau287 = None

    tau290 = zeros((N, N, M, M))

    tau290 += np.einsum("cj,iabc->ijab", l1, tau289, optimize=True)

    tau289 = None

    tau299 -= 2 * np.einsum("jiba->ijab", tau290, optimize=True)

    tau290 = None

    r2 += np.einsum("ijba->abij", tau299, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau299, optimize=True) / 8

    tau299 = None

    tau300 = zeros((N, N, M, M))

    tau300 += np.einsum("jk,abik->ijab", f[o, o], l2, optimize=True)

    tau301 -= np.einsum("ijba->ijab", tau300, optimize=True)

    tau300 = None

    r2 -= np.einsum("ijab->abij", tau301, optimize=True)

    r2 += np.einsum("jiab->abij", tau301, optimize=True)

    tau301 = None

    tau303 -= np.einsum("baji->ijab", t2, optimize=True)

    tau307 += np.einsum("klab,jiab->ijkl", tau303, u[o, o, v, v], optimize=True)

    tau303 = None

    tau305 += np.einsum("ai->ia", t1, optimize=True)

    tau306 = zeros((N, N, N, M))

    tau306 += np.einsum("kb,jiba->ijka", tau305, u[o, o, v, v], optimize=True)

    tau309 = zeros((N, N, N, M))

    tau309 += np.einsum("kb,baji->ijka", tau305, l2, optimize=True)

    tau305 = None

    tau310 += 2 * np.einsum("al,jika->ijkl", t1, tau309, optimize=True)

    tau309 = None

    tau306 += 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau307 += 2 * np.einsum("al,jika->ijkl", t1, tau306, optimize=True)

    tau306 = None

    tau307 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 += np.einsum("bakl,jikl->abij", l2, tau307, optimize=True) / 4

    tau307 = None

    tau308 += np.einsum("baji->ijab", t2, optimize=True)

    tau310 -= np.einsum("abji,klab->ijkl", l2, tau308, optimize=True)

    tau308 = None

    r2 += np.einsum("jikl,klba->abij", tau310, u[o, o, v, v], optimize=True) / 4

    tau310 = None

    tau312 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau313 = zeros((N, N, N, M))

    tau313 += np.einsum("al,kjli->ijka", l1, tau312, optimize=True)

    tau312 = None

    tau313 += 2 * np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,kjib->abij", l1, tau313, optimize=True) / 2

    tau313 = None

    tau314 = zeros((N, M))

    tau314 += np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    tau320 -= np.einsum("ia->ia", tau314, optimize=True)

    tau314 = None

    tau317 = zeros((N, N))

    tau317 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau318 += np.einsum("ij->ij", tau317, optimize=True)

    tau317 = None

    tau318 += np.einsum("ij->ij", f[o, o], optimize=True)

    tau319 = zeros((N, M))

    tau319 += np.einsum("aj,ij->ia", l1, tau318, optimize=True)

    tau318 = None

    tau320 += np.einsum("ia->ia", tau319, optimize=True)

    tau319 = None

    tau320 -= np.einsum("ia->ia", f[o, v], optimize=True)

    r2 += np.einsum("aj,ib->abij", l1, tau320, optimize=True)

    r2 += np.einsum("bi,ja->abij", l1, tau320, optimize=True)

    r2 -= np.einsum("bj,ia->abij", l1, tau320, optimize=True)

    r2 -= np.einsum("ai,jb->abij", l1, tau320, optimize=True)

    tau320 = None

    r1 += np.einsum("ai->ai", f[v, o], optimize=True)

    r2 -= np.einsum("bk,jika->abij", l1, u[o, o, o, v], optimize=True)

    r2 += np.einsum("jiba->abij", u[o, o, v, v], optimize=True)

    return r1, r2
