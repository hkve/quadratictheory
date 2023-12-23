import numpy as np


def amplitudes_intermediates_qccsd(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau5 = zeros((N, N, M, M))

    tau5 -= np.einsum("ablk,lkji->ijab", t2, tau0, optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("caik,kjbc->ijab", l2, tau5, optimize=True)

    r1 = zeros((M, N))

    r1 -= np.einsum("ijbc,jbac->ai", tau6, u[o, v, v, v], optimize=True) / 4

    tau6 = None

    tau17 = zeros((N, N, N, M))

    tau17 += np.einsum("ilba,jlkb->ijka", tau5, u[o, o, o, v], optimize=True)

    tau43 = zeros((N, N, M, M))

    tau43 -= np.einsum("ijba->ijab", tau5, optimize=True)

    tau72 = zeros((N, N, M, M))

    tau72 -= np.einsum("ijba->ijab", tau5, optimize=True)

    tau5 = None

    tau20 = zeros((N, N, N, N))

    tau20 -= np.einsum("jnkm,miln->ijkl", tau0, tau0, optimize=True)

    tau22 = zeros((N, N, N, N))

    tau22 += np.einsum("ijkl->ijkl", tau20, optimize=True)

    tau82 = zeros((N, N, N, N))

    tau82 += np.einsum("ijkl->ijkl", tau20, optimize=True)

    tau20 = None

    tau61 = zeros((N, N, M, M))

    tau61 -= np.einsum("jilk,lkab->ijab", tau0, u[o, o, v, v], optimize=True)

    tau63 = zeros((N, N, M, M))

    tau63 += np.einsum("ijba->ijab", tau61, optimize=True)

    tau94 = zeros((N, N, M, M))

    tau94 -= np.einsum("ijba->ijab", tau61, optimize=True)

    tau246 = zeros((N, N, M, M))

    tau246 += np.einsum("ijba->ijab", tau61, optimize=True)

    tau61 = None

    tau172 = zeros((N, N, M, M))

    tau172 -= np.einsum("iklj,lakb->ijab", tau0, u[o, v, o, v], optimize=True)

    tau174 = zeros((N, N, M, M))

    tau174 -= np.einsum("jiba->ijab", tau172, optimize=True)

    tau172 = None

    tau193 = zeros((N, N, N, M))

    tau193 += np.einsum("limj,kmla->ijka", tau0, u[o, o, o, v], optimize=True)

    tau194 = zeros((N, N, N, M))

    tau194 += np.einsum("jika->ijka", tau193, optimize=True)

    tau193 = None

    tau1 = zeros((N, N, M, M))

    tau1 -= np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau2 = zeros((N, N, N, M))

    tau2 -= np.einsum("ijbc,kbca->ijka", tau1, u[o, v, v, v], optimize=True)

    tau16 = zeros((N, N, N, M))

    tau16 += np.einsum("kija->ijka", tau2, optimize=True)

    tau92 = zeros((N, N, N, M))

    tau92 += 2 * np.einsum("jkia->ijka", tau2, optimize=True)

    tau121 = zeros((N, N, N, M))

    tau121 += 4 * np.einsum("ikja->ijka", tau2, optimize=True)

    tau194 += 2 * np.einsum("jika->ijka", tau2, optimize=True)

    tau298 = zeros((N, N, N, M))

    tau298 += 2 * np.einsum("ijka->ijka", tau2, optimize=True)

    r1 += np.einsum("iklj,jkla->ai", tau0, tau2, optimize=True) / 2

    tau2 = None

    tau7 = zeros((N, M, M, M))

    tau7 += np.einsum("ijda,jdbc->iabc", tau1, u[o, v, v, v], optimize=True)

    tau10 = zeros((N, M, M, M))

    tau10 -= 2 * np.einsum("adij,jbcd->iabc", t2, tau7, optimize=True)

    tau56 = zeros((N, M, M, M))

    tau56 -= 2 * np.einsum("ibca->iabc", tau7, optimize=True)

    tau85 = zeros((N, M, M, M))

    tau85 += 2 * np.einsum("iacb->iabc", tau7, optimize=True)

    tau97 = zeros((N, M, M, M))

    tau97 -= 4 * np.einsum("iacb->iabc", tau7, optimize=True)

    tau223 = zeros((N, M, M, M))

    tau223 -= 2 * np.einsum("ibca->iabc", tau7, optimize=True)

    tau7 = None

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("caki,kjcb->ijab", t2, tau1, optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("ijab->ijab", tau8, optimize=True)

    tau11 = zeros((N, N, M, M))

    tau11 += 2 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 -= 2 * np.einsum("acki,kjbc->ijab", l2, tau8, optimize=True)

    tau43 += 4 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau72 += 4 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau73 = zeros((N, N, M, M))

    tau73 += np.einsum("jkcb,kica->ijab", tau72, u[o, o, v, v], optimize=True)

    tau74 = zeros((N, N, M, M))

    tau74 += np.einsum("caki,jkbc->ijab", l2, tau73, optimize=True)

    tau75 = zeros((N, N, M, M))

    tau75 += np.einsum("ijab->ijab", tau74, optimize=True)

    tau74 = None

    tau213 = zeros((N, N, M, M))

    tau213 += np.einsum("jiba->ijab", tau73, optimize=True)

    tau73 = None

    tau201 = zeros((N, N, N, M))

    tau201 += np.einsum("klba,lijb->ijka", tau72, u[o, o, o, v], optimize=True)

    tau216 = zeros((N, N, N, M))

    tau216 += np.einsum("jkia->ijka", tau201, optimize=True)

    tau201 = None

    tau225 = zeros((N, M, M, M))

    tau225 += np.einsum("ijdc,jadb->iabc", tau72, u[o, v, v, v], optimize=True)

    tau233 = zeros((N, M, M, M))

    tau233 -= np.einsum("ibca->iabc", tau225, optimize=True)

    tau225 = None

    tau80 = zeros((N, N, M, M))

    tau80 += 2 * np.einsum("ijba->ijab", tau8, optimize=True)

    tau82 -= 2 * np.einsum("baji,klba->ijkl", l2, tau8, optimize=True)

    tau265 = zeros((N, N, M, M))

    tau265 += 2 * np.einsum("ijba->ijab", tau8, optimize=True)

    tau309 = zeros((N, M, M, M))

    tau309 -= np.einsum("jkab,kjic->iabc", tau8, u[o, o, o, v], optimize=True)

    tau310 = zeros((N, M, M, M))

    tau310 -= np.einsum("ibac->iabc", tau309, optimize=True)

    tau309 = None

    tau319 = zeros((N, N, M, M))

    tau319 -= 2 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau321 = zeros((N, N, M, M))

    tau321 += 2 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau330 = zeros((N, N, M, M))

    tau330 -= 2 * np.einsum("ijba->ijab", tau8, optimize=True)

    tau336 = zeros((N, N, N, N))

    tau336 += np.einsum("jiba,lkba->ijkl", tau8, u[o, o, v, v], optimize=True)

    tau337 = zeros((N, N, N, M))

    tau337 += np.einsum("al,kjil->ijka", t1, tau336, optimize=True)

    tau336 = None

    r2 = zeros((M, M, N, N))

    r2 -= np.einsum("klba,lkji->abij", tau8, u[o, o, o, o], optimize=True)

    tau12 = zeros((N, M, M, M))

    tau12 += np.einsum("jkab,ikjc->iabc", tau1, u[o, o, o, v], optimize=True)

    tau13 = zeros((N, M, M, M))

    tau13 += 2 * np.einsum("iabc->iabc", tau12, optimize=True)

    tau85 += 2 * np.einsum("icab->iabc", tau12, optimize=True)

    tau97 += 4 * np.einsum("ibac->iabc", tau12, optimize=True)

    tau12 = None

    tau15 = zeros((N, N, N, M))

    tau15 += np.einsum("liab,jklb->ijka", tau1, u[o, o, o, v], optimize=True)

    tau16 += np.einsum("ikja->ijka", tau15, optimize=True)

    tau92 -= 2 * np.einsum("kija->ijka", tau15, optimize=True)

    tau121 += 4 * np.einsum("kija->ijka", tau15, optimize=True)

    tau194 -= 2 * np.einsum("ikja->ijka", tau15, optimize=True)

    tau15 = None

    tau195 = zeros((N, N, N, M))

    tau195 += np.einsum("balk,iljb->ijka", t2, tau194, optimize=True)

    tau194 = None

    tau216 += 2 * np.einsum("ijka->ijka", tau195, optimize=True)

    tau195 = None

    tau18 = zeros((N, N, M, M))

    tau18 -= np.einsum("lijk,klab->ijab", tau0, tau1, optimize=True)

    tau19 += np.einsum("ijab->ijab", tau18, optimize=True)

    r1 -= np.einsum("kjab,jikb->ai", tau19, u[o, o, o, v], optimize=True) / 2

    tau19 = None

    tau70 = zeros((N, N, M, M))

    tau70 += np.einsum("ijab->ijab", tau18, optimize=True)

    tau18 = None

    tau21 = zeros((N, N, N, N))

    tau21 += np.einsum("ikab,jlba->ijkl", tau1, tau1, optimize=True)

    tau22 += 4 * np.einsum("ijlk->ijkl", tau21, optimize=True)

    tau82 -= 4 * np.einsum("ijkl->ijkl", tau21, optimize=True)

    tau21 = None

    tau84 = zeros((N, N, M, M))

    tau84 -= np.einsum("ijkl,klba->ijab", tau82, u[o, o, v, v], optimize=True)

    tau82 = None

    tau31 = zeros((N, N, N, M, M, M))

    tau31 -= np.einsum("ijad,kbcd->ijkabc", tau1, u[o, v, v, v], optimize=True)

    tau32 = zeros((N, N, N, M, M, M))

    tau32 -= np.einsum("ijkabc->ijkabc", tau31, optimize=True)

    tau163 = zeros((N, N, N, M, M, M))

    tau163 += np.einsum("ijkabc->ijkabc", tau31, optimize=True)

    tau163 += np.einsum("kjicba->ijkabc", tau31, optimize=True)

    tau31 = None

    tau164 = zeros((N, N, N, M, M, M))

    tau164 += np.einsum("dclk,lijabd->ijkabc", t2, tau163, optimize=True)

    tau163 = None

    tau165 = zeros((N, N, N, M, M, M))

    tau165 -= 2 * np.einsum("ikjacb->ijkabc", tau164, optimize=True)

    tau164 = None

    tau52 = zeros((N, N, N, M))

    tau52 += np.einsum("bj,ikba->ijka", t1, tau1, optimize=True)

    tau53 = zeros((N, M, M, M))

    tau53 += np.einsum("ijka,kjbc->iabc", tau52, u[o, o, v, v], optimize=True)

    tau56 += 2 * np.einsum("ibca->iabc", tau53, optimize=True)

    tau97 += 4 * np.einsum("iacb->iabc", tau53, optimize=True)

    tau223 += 2 * np.einsum("ibca->iabc", tau53, optimize=True)

    tau53 = None

    tau151 = zeros((N, N, M, M))

    tau151 += np.einsum("ikla,lkjb->ijab", tau52, u[o, o, o, v], optimize=True)

    tau52 = None

    tau154 = zeros((N, N, M, M))

    tau154 += 2 * np.einsum("ijba->ijab", tau151, optimize=True)

    tau151 = None

    tau65 = zeros((N, N, N, M))

    tau65 += np.einsum("bi,jkab->ijka", l1, tau1, optimize=True)

    tau66 = zeros((N, N, M, M))

    tau66 -= np.einsum("kila,jlkb->ijab", tau65, u[o, o, o, v], optimize=True)

    tau65 = None

    tau75 += 4 * np.einsum("ijab->ijab", tau66, optimize=True)

    tau66 = None

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum("ikcb,kjac->ijab", tau1, tau1, optimize=True)

    tau70 -= 2 * np.einsum("ijab->ijab", tau69, optimize=True)

    tau69 = None

    tau76 = zeros((M, M, M, M))

    tau76 += 4 * np.einsum("ijac,jibd->abcd", tau1, tau1, optimize=True)

    tau89 = zeros((N, N, N, N, M, M))

    tau89 += np.einsum("ijac,klcb->ijklab", tau1, u[o, o, v, v], optimize=True)

    tau90 = zeros((N, N, N, N, M, M))

    tau90 -= np.einsum("ijlkab->ijklab", tau89, optimize=True)

    tau90 -= np.einsum("ljikba->ijklab", tau89, optimize=True)

    tau92 += 2 * np.einsum("bl,lkijab->ijka", t1, tau90, optimize=True)

    tau90 = None

    tau118 = zeros((N, N, N, M))

    tau118 += np.einsum("bl,lijkab->ijka", t1, tau89, optimize=True)

    tau121 -= 4 * np.einsum("kija->ijka", tau118, optimize=True)

    tau118 = None

    tau197 = zeros((N, N, N, N, M, M))

    tau197 += np.einsum("ijlkab->ijklab", tau89, optimize=True)

    tau197 += np.einsum("ljikba->ijklab", tau89, optimize=True)

    tau198 = zeros((N, N, N, N, M, M))

    tau198 += np.einsum("cbml,mijkac->ijklab", t2, tau197, optimize=True)

    tau197 = None

    tau199 = zeros((N, N, N, N, M, M))

    tau199 += 2 * np.einsum("ilkjab->ijklab", tau198, optimize=True)

    tau198 = None

    tau296 = zeros((N, N, N, M))

    tau296 -= np.einsum("bl,ijlkba->ijka", t1, tau89, optimize=True)

    tau89 = None

    tau298 += 2 * np.einsum("ijka->ijka", tau296, optimize=True)

    tau296 = None

    tau150 = zeros((N, N, M, M))

    tau150 += np.einsum("ikca,kcjb->ijab", tau1, u[o, v, o, v], optimize=True)

    tau154 -= 2 * np.einsum("ijba->ijab", tau150, optimize=True)

    tau150 = None

    tau171 = zeros((N, N, M, M))

    tau171 += np.einsum("kiac,jbkc->ijab", tau1, u[o, v, o, v], optimize=True)

    tau174 += 2 * np.einsum("ijab->ijab", tau171, optimize=True)

    tau171 = None

    tau173 = zeros((N, N, M, M))

    tau173 -= np.einsum("ijcd,cabd->ijab", tau1, u[v, v, v, v], optimize=True)

    tau174 += 2 * np.einsum("jiba->ijab", tau173, optimize=True)

    tau173 = None

    tau175 = zeros((N, N, M, M))

    tau175 += np.einsum("cbkj,ikca->ijab", t2, tau174, optimize=True)

    tau174 = None

    tau235 = zeros((N, N, M, M))

    tau235 -= 2 * np.einsum("ijba->ijab", tau175, optimize=True)

    tau175 = None

    tau285 = zeros((N, N, N, N, N, M))

    tau285 += np.einsum("ijab,lkmb->ijklma", tau1, u[o, o, o, v], optimize=True)

    tau286 = zeros((N, N, N, N))

    tau286 -= np.einsum("am,ijmkla->ijkl", t1, tau285, optimize=True)

    tau285 = None

    tau290 = zeros((N, N, N, N))

    tau290 += 4 * np.einsum("ijkl->ijkl", tau286, optimize=True)

    tau286 = None

    tau3 = zeros((M, M, M, M))

    tau3 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau4 = zeros((N, M, M, M))

    tau4 += np.einsum("daeb,idce->iabc", tau3, u[o, v, v, v], optimize=True)

    tau13 += np.einsum("iabc->iabc", tau4, optimize=True)

    tau14 = zeros((N, N, N, M))

    tau14 += np.einsum("bcij,kcab->ijka", t2, tau13, optimize=True)

    tau13 = None

    tau17 += np.einsum("kija->ijka", tau14, optimize=True)

    tau277 = zeros((N, N, N, M))

    tau277 -= np.einsum("kjia->ijka", tau14, optimize=True)

    tau14 = None

    tau56 += np.einsum("iabc->iabc", tau4, optimize=True)

    tau85 += np.einsum("icab->iabc", tau4, optimize=True)

    tau97 += 2 * np.einsum("ibac->iabc", tau4, optimize=True)

    tau223 += np.einsum("iabc->iabc", tau4, optimize=True)

    r1 += np.einsum("ijcb,jbca->ai", tau1, tau4, optimize=True) / 2

    tau4 = None

    tau23 = zeros((N, N, N, M, M, M))

    tau23 -= np.einsum("abdc,ijkd->ijkabc", tau3, u[o, o, o, v], optimize=True)

    tau38 = zeros((N, N, N, M, M, M))

    tau38 += np.einsum("kjibac->ijkabc", tau23, optimize=True)

    tau156 = zeros((N, N, N, M, M, M))

    tau156 -= np.einsum("cdil,jlkdab->ijkabc", t2, tau23, optimize=True)

    tau23 = None

    tau165 += np.einsum("ikjabc->ijkabc", tau156, optimize=True)

    tau156 = None

    tau50 = zeros((N, N, M, M, M, M))

    tau50 -= np.einsum("abec,jied->ijabcd", tau3, u[o, o, v, v], optimize=True)

    tau51 = zeros((N, M, M, M))

    tau51 -= np.einsum("dj,jiadbc->iabc", t1, tau50, optimize=True)

    tau56 -= np.einsum("iabc->iabc", tau51, optimize=True)

    tau97 -= np.einsum("ibac->iabc", tau51, optimize=True)

    tau51 = None

    tau221 = zeros((N, N, M, M, M, M))

    tau221 -= np.einsum("ceik,jkeabd->ijabcd", t2, tau50, optimize=True)

    tau50 = None

    tau222 = zeros((N, M, M, M))

    tau222 += np.einsum("dj,ijdabc->iabc", t1, tau221, optimize=True)

    tau221 = None

    tau233 += 2 * np.einsum("iabc->iabc", tau222, optimize=True)

    tau222 = None

    tau68 = zeros((N, N, M, M))

    tau68 -= np.einsum("ijdc,cabd->ijab", tau1, tau3, optimize=True)

    tau70 += np.einsum("ijab->ijab", tau68, optimize=True)

    tau68 = None

    tau71 = zeros((N, N, M, M))

    tau71 += np.einsum("jkbc,kica->ijab", tau70, u[o, o, v, v], optimize=True)

    tau70 = None

    tau75 += 2 * np.einsum("jiba->ijab", tau71, optimize=True)

    tau71 = None

    tau76 += np.einsum("becf,fade->abcd", tau3, tau3, optimize=True)

    tau84 -= np.einsum("abdc,jicd->ijab", tau76, u[o, o, v, v], optimize=True)

    tau76 = None

    tau149 = zeros((N, N, M, M))

    tau149 -= np.einsum("acdb,icjd->ijab", tau3, u[o, v, o, v], optimize=True)

    tau154 += np.einsum("ijab->ijab", tau149, optimize=True)

    tau149 = None

    tau249 = zeros((N, N, M, M, M, M))

    tau249 -= np.einsum("deij,eabc->ijabcd", t2, tau3, optimize=True)

    tau250 = zeros((N, N, N, M, M, M))

    tau250 -= np.einsum("kced,ijabde->ijkabc", u[o, v, v, v], tau249, optimize=True)

    tau260 = zeros((N, N, N, M, M, M))

    tau260 -= np.einsum("kjicba->ijkabc", tau250, optimize=True)

    tau250 = None

    tau270 = zeros((N, N, N, N, M, M))

    tau270 += np.einsum("lkdc,ijabcd->ijklab", u[o, o, v, v], tau249, optimize=True)

    tau249 = None

    tau272 = zeros((N, N, N, N, M, M))

    tau272 += np.einsum("ijlkab->ijklab", tau270, optimize=True)

    tau270 = None

    tau9 += np.einsum("baji->ijab", t2, optimize=True)

    tau10 -= 2 * np.einsum("ijda,jbdc->iabc", tau9, u[o, v, v, v], optimize=True)

    tau9 = None

    tau10 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    r1 += np.einsum("bcji,jbca->ai", l2, tau10, optimize=True) / 2

    tau10 = None

    tau11 += np.einsum("baji->ijab", t2, optimize=True)

    tau17 += np.einsum("ikbc,jabc->ijka", tau11, u[o, v, v, v], optimize=True)

    tau22 -= np.einsum("abji,klab->ijkl", l2, tau11, optimize=True)

    tau11 = None

    r1 += np.einsum("lijk,jkla->ai", tau22, u[o, o, o, v], optimize=True) / 4

    tau22 = None

    tau16 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau17 += 4 * np.einsum("bali,kjlb->ijka", t2, tau16, optimize=True)

    tau16 = None

    tau17 -= 2 * np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    r1 += np.einsum("bajk,jikb->ai", l2, tau17, optimize=True) / 4

    tau17 = None

    tau24 = zeros((N, N, N, M, M, M))

    tau24 += np.einsum("daji,kdbc->ijkabc", l2, u[o, v, v, v], optimize=True)

    tau38 += 2 * np.einsum("cdil,kjldba->ijkabc", t2, tau24, optimize=True)

    tau64 = zeros((N, N, M, M))

    tau64 += np.einsum("ck,ikjabc->ijab", t1, tau24, optimize=True)

    tau24 = None

    tau84 -= 2 * np.einsum("ijba->ijab", tau64, optimize=True)

    tau84 -= 4 * np.einsum("jiab->ijab", tau64, optimize=True)

    tau64 = None

    tau25 = zeros((N, N, N, M))

    tau25 -= np.einsum("bk,baij->ijka", t1, l2, optimize=True)

    tau26 = zeros((N, N, N, N, M, M))

    tau26 += np.einsum("acij,lkcb->ijklab", t2, u[o, o, v, v], optimize=True)

    tau36 = zeros((N, N, N, N, N, M))

    tau36 += np.einsum("bi,jklmab->ijklma", t1, tau26, optimize=True)

    tau37 = zeros((N, N, N, N, N, M))

    tau37 += np.einsum("ikjmla->ijklma", tau36, optimize=True)

    tau324 = zeros((N, N, N, N, N, M))

    tau324 -= np.einsum("mijlka->ijklma", tau36, optimize=True)

    tau36 = None

    tau38 += np.einsum("jmla,imklcb->ijkabc", tau25, tau26, optimize=True)

    tau196 = zeros((N, N, N, N, M, M))

    tau196 -= np.einsum("mijn,kmlnab->ijklab", tau0, tau26, optimize=True)

    tau199 += np.einsum("kijlba->ijklab", tau196, optimize=True)

    tau196 = None

    tau200 = zeros((N, N, N, M))

    tau200 += np.einsum("bl,ijlkba->ijka", t1, tau199, optimize=True)

    tau199 = None

    tau216 -= 2 * np.einsum("ikja->ijka", tau200, optimize=True)

    tau200 = None

    tau254 = zeros((N, N, N, N, N, N, M, M))

    tau254 -= np.einsum("ackj,ilmnbc->ijklmnab", t2, tau26, optimize=True)

    tau255 = zeros((N, N, N, N, M, M, M, M))

    tau255 += np.einsum("abnm,imjnkldc->ijklabcd", l2, tau254, optimize=True)

    tau254 = None

    tau258 = zeros((N, N, N, N, M, M, M, M))

    tau258 += np.einsum("ijlkbacd->ijklabcd", tau255, optimize=True)

    tau255 = None

    tau271 = zeros((N, N, N, N, M, M))

    tau271 -= np.einsum("imca,jklmcb->ijklab", tau1, tau26, optimize=True)

    tau272 += 2 * np.einsum("lijkba->ijklab", tau271, optimize=True)

    tau271 = None

    tau273 = zeros((N, N, N, M))

    tau273 += np.einsum("bl,ijklba->ijka", t1, tau272, optimize=True)

    tau272 = None

    tau277 -= np.einsum("kjia->ijka", tau273, optimize=True)

    tau273 = None

    tau322 = zeros((N, N, N, N, N, N, M, M))

    tau322 -= np.einsum("caij,klmncb->ijklmnab", l2, tau26, optimize=True)

    tau323 = zeros((N, N, N, N, N, M))

    tau323 += np.einsum("bn,inkjmlab->ijklma", t1, tau322, optimize=True)

    tau322 = None

    tau27 = zeros((N, N, N, N, M, M))

    tau27 += np.einsum("caji,bckl->ijklab", l2, t2, optimize=True)

    tau28 = zeros((N, N, N, M, M, M))

    tau28 += np.einsum("mklc,lijmab->ijkabc", u[o, o, o, v], tau27, optimize=True)

    tau32 += np.einsum("ijkabc->ijkabc", tau28, optimize=True)

    tau252 = zeros((N, N, N, M, M, M))

    tau252 += np.einsum("ijkabc->ijkabc", tau28, optimize=True)

    tau28 = None

    tau158 = zeros((N, N, N, N, M, M, M, M))

    tau158 += np.einsum("jmlnbd,imknac->ijklabcd", tau26, tau27, optimize=True)

    tau159 = zeros((N, N, N, M, M, M))

    tau159 += np.einsum("dl,lijkabcd->ijkabc", t1, tau158, optimize=True)

    tau158 = None

    tau165 += 2 * np.einsum("ijkabc->ijkabc", tau159, optimize=True)

    tau159 = None

    tau256 = zeros((N, N, N, N, N, N, M, M))

    tau256 -= np.einsum("acml,ijkncb->ijklmnab", t2, tau27, optimize=True)

    tau257 = zeros((N, N, N, N, M, M, M, M))

    tau257 += np.einsum("nmcd,ijkmlnba->ijklabcd", u[o, o, v, v], tau256, optimize=True)

    tau256 = None

    tau258 += np.einsum("lkijcdba->ijklabcd", tau257, optimize=True)

    tau257 = None

    tau259 = zeros((N, N, N, M, M, M))

    tau259 += np.einsum("dl,ijkldabc->ijkabc", t1, tau258, optimize=True)

    tau258 = None

    tau260 += np.einsum("kjicba->ijkabc", tau259, optimize=True)

    tau259 = None

    tau323 += np.einsum("lbca,imkjbc->ijklma", u[o, v, v, v], tau27, optimize=True)

    tau29 = zeros((N, N, N, M, M, M))

    tau29 += np.einsum("adij,kbcd->ijkabc", t2, u[o, v, v, v], optimize=True)

    tau30 = zeros((N, N, N, M, M, M))

    tau30 -= np.einsum("dail,jlkbdc->ijkabc", l2, tau29, optimize=True)

    tau32 += np.einsum("ijkabc->ijkabc", tau30, optimize=True)

    tau38 -= 2 * np.einsum("jikacb->ijkabc", tau32, optimize=True)

    tau38 -= 2 * np.einsum("kijbca->ijkabc", tau32, optimize=True)

    tau32 = None

    tau161 = zeros((N, N, N, M, M, M))

    tau161 += np.einsum("kijacb->ijkabc", tau30, optimize=True)

    tau252 += np.einsum("kjicba->ijkabc", tau30, optimize=True)

    tau30 = None

    tau253 = zeros((N, N, N, M, M, M))

    tau253 += np.einsum("dclk,ijldab->ijkabc", t2, tau252, optimize=True)

    tau252 = None

    tau260 += 2 * np.einsum("ikjbca->ijkabc", tau253, optimize=True)

    tau253 = None

    tau157 = zeros((N, N, N, M, M, M))

    tau157 += np.einsum("lijm,lkmabc->ijkabc", tau0, tau29, optimize=True)

    tau165 += np.einsum("kijbca->ijkabc", tau157, optimize=True)

    tau157 = None

    tau209 = zeros((N, N, M, M))

    tau209 -= np.einsum("ck,kijacb->ijab", l1, tau29, optimize=True)

    tau213 -= 4 * np.einsum("ijab->ijab", tau209, optimize=True)

    tau209 = None

    tau251 = zeros((N, N, N, M, M, M))

    tau251 -= np.einsum("ilda,jkldbc->ijkabc", tau1, tau29, optimize=True)

    tau260 += 2 * np.einsum("ikjbac->ijkabc", tau251, optimize=True)

    tau251 = None

    tau261 = zeros((N, N, M, M))

    tau261 += np.einsum("ck,kijabc->ijab", t1, tau260, optimize=True)

    tau260 = None

    tau282 = zeros((N, N, M, M))

    tau282 += 4 * np.einsum("ijba->ijab", tau261, optimize=True)

    tau261 = None

    tau267 = zeros((N, N, N, N, M, M))

    tau267 += np.einsum("ci,jklabc->ijklab", t1, tau29, optimize=True)

    tau268 = zeros((N, N, N, N, M, M))

    tau268 += np.einsum("lijkab->ijklab", tau267, optimize=True)

    tau267 = None

    tau33 = zeros((N, N, N, M))

    tau33 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau34 = zeros((N, N, N, M, M, M))

    tau34 += np.einsum("lmkc,lijmab->ijkabc", tau33, tau27, optimize=True)

    tau27 = None

    tau38 += 2 * np.einsum("jikacb->ijkabc", tau34, optimize=True)

    tau38 += np.einsum("kijbca->ijkabc", tau34, optimize=True)

    tau34 = None

    tau54 = zeros((N, N, N, M))

    tau54 += np.einsum("kjia->ijka", tau33, optimize=True)

    tau78 = zeros((N, N, N, N))

    tau78 += np.einsum("aj,ilka->ijkl", t1, tau33, optimize=True)

    tau81 = zeros((N, N, N, N))

    tau81 -= 2 * np.einsum("ilkj->ijkl", tau78, optimize=True)

    tau275 = zeros((N, N, N, N))

    tau275 -= 2 * np.einsum("lkij->ijkl", tau78, optimize=True)

    tau334 = zeros((N, N, N, N))

    tau334 -= 2 * np.einsum("lkij->ijkl", tau78, optimize=True)

    tau78 = None

    tau83 = zeros((N, N, N, M))

    tau83 -= np.einsum("ikja->ijka", tau33, optimize=True)

    tau119 = zeros((N, N, N, M))

    tau119 += np.einsum("kjia->ijka", tau33, optimize=True)

    tau33 = None

    tau35 = zeros((N, N, N, N, N, M))

    tau35 -= np.einsum("abij,lkmb->ijklma", t2, u[o, o, o, v], optimize=True)

    tau37 -= np.einsum("ijmlka->ijklma", tau35, optimize=True)

    tau38 += 2 * np.einsum("balm,milkjc->ijkabc", l2, tau37, optimize=True)

    tau37 = None

    tau57 = zeros((N, N, M, M))

    tau57 -= 2 * np.einsum("ck,jikcba->ijab", t1, tau38, optimize=True)

    tau38 = None

    tau160 = zeros((N, N, N, M, M, M))

    tau160 -= np.einsum("abml,iljkmc->ijkabc", l2, tau35, optimize=True)

    tau161 += np.einsum("ikjbac->ijkabc", tau160, optimize=True)

    tau160 = None

    tau162 = zeros((N, N, N, M, M, M))

    tau162 += np.einsum("dclk,iljdab->ijkabc", t2, tau161, optimize=True)

    tau161 = None

    tau165 -= 2 * np.einsum("ikjabc->ijkabc", tau162, optimize=True)

    tau162 = None

    tau166 = zeros((N, N, M, M))

    tau166 += np.einsum("ck,ijkcab->ijab", t1, tau165, optimize=True)

    tau165 = None

    tau235 -= 2 * np.einsum("ijab->ijab", tau166, optimize=True)

    tau166 = None

    tau323 += np.einsum("bani,kjmlnb->ijklma", l2, tau35, optimize=True)

    tau327 = zeros((N, N, N, N))

    tau327 -= 4 * np.einsum("am,jlkima->ijkl", t1, tau323, optimize=True)

    tau323 = None

    tau324 -= np.einsum("ijlkma->ijklma", tau35, optimize=True)

    tau325 = zeros((N, N, N, N))

    tau325 += np.einsum("am,ijklma->ijkl", l1, tau324, optimize=True)

    tau324 = None

    tau327 += 2 * np.einsum("lkij->ijkl", tau325, optimize=True)

    tau334 += 2 * np.einsum("lkij->ijkl", tau325, optimize=True)

    tau325 = None

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("caki,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum("ijab->ijab", tau39, optimize=True)

    tau45 = zeros((N, N, M, M))

    tau45 += np.einsum("ijab->ijab", tau39, optimize=True)

    tau105 = zeros((N, N, M, M))

    tau105 += np.einsum("jiab->ijab", tau39, optimize=True)

    tau128 = zeros((N, N, M, M))

    tau128 -= np.einsum("jiab->ijab", tau39, optimize=True)

    tau141 = zeros((N, N, M, M))

    tau141 += np.einsum("acdb,ijcd->ijab", tau3, tau39, optimize=True)

    tau147 = zeros((N, N, M, M))

    tau147 += np.einsum("ijab->ijab", tau141, optimize=True)

    tau141 = None

    tau142 = zeros((N, N, M, M))

    tau142 += np.einsum("kiac,kjbc->ijab", tau1, tau39, optimize=True)

    tau147 += 2 * np.einsum("ijab->ijab", tau142, optimize=True)

    tau142 = None

    tau143 = zeros((N, N, M, M))

    tau143 += np.einsum("ikca,jkcb->ijab", tau1, tau39, optimize=True)

    tau147 += 2 * np.einsum("jiba->ijab", tau143, optimize=True)

    tau143 = None

    tau145 = zeros((N, N, M, M))

    tau145 -= np.einsum("iklj,klab->ijab", tau0, tau39, optimize=True)

    tau147 -= np.einsum("jiba->ijab", tau145, optimize=True)

    tau145 = None

    tau203 = zeros((N, N, N, N, M, M))

    tau203 -= np.einsum("acji,klbc->ijklab", t2, tau39, optimize=True)

    tau204 = zeros((N, N, N, N, M, M))

    tau204 += np.einsum("ljikab->ijklab", tau203, optimize=True)

    tau203 = None

    tau40 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau57 += 2 * np.einsum("cbda,jicd->ijab", tau3, tau40, optimize=True)

    tau57 -= 4 * np.einsum("kjbc,kiac->ijab", tau1, tau40, optimize=True)

    tau57 -= 4 * np.einsum("ikca,jkcb->ijab", tau1, tau40, optimize=True)

    tau62 = zeros((N, N, M, M))

    tau62 += np.einsum("caki,kjcb->ijab", l2, tau40, optimize=True)

    tau63 += 4 * np.einsum("ijba->ijab", tau62, optimize=True)

    tau63 += 4 * np.einsum("jiab->ijab", tau62, optimize=True)

    tau94 += 8 * np.einsum("ijab->ijab", tau62, optimize=True)

    tau146 = zeros((N, N, M, M))

    tau146 -= np.einsum("caki,kjbc->ijab", t2, tau62, optimize=True)

    tau147 += 2 * np.einsum("ijba->ijab", tau146, optimize=True)

    tau146 = None

    tau320 = zeros((N, N, M, M))

    tau320 -= 4 * np.einsum("ijba->ijab", tau62, optimize=True)

    tau62 = None

    tau115 = zeros((N, N))

    tau115 += np.einsum("kiab,kjab->ij", tau1, tau40, optimize=True)

    tau136 = zeros((N, N))

    tau136 += 8 * np.einsum("ji->ij", tau115, optimize=True)

    tau292 = zeros((N, N))

    tau292 += 8 * np.einsum("ji->ij", tau115, optimize=True)

    tau115 = None

    tau125 = zeros((N, N, N, M))

    tau125 += np.einsum("bi,jkab->ijka", t1, tau40, optimize=True)

    tau126 = zeros((N, N, N, M))

    tau126 += 2 * np.einsum("jkia->ijka", tau125, optimize=True)

    tau188 = zeros((N, N, N, M))

    tau188 -= np.einsum("ikja->ijka", tau125, optimize=True)

    tau125 = None

    tau167 = zeros((N, N, M, M))

    tau167 += np.einsum("ikac,jkcb->ijab", tau40, tau72, optimize=True)

    tau72 = None

    tau235 += np.einsum("jiba->ijab", tau167, optimize=True)

    tau167 = None

    tau184 = zeros((N, N, N, N, M, M))

    tau184 += np.einsum("caij,klbc->ijklab", t2, tau40, optimize=True)

    tau268 -= np.einsum("jilkab->ijklab", tau184, optimize=True)

    tau269 = zeros((N, N, N, M))

    tau269 += np.einsum("bl,ijklba->ijka", l1, tau268, optimize=True)

    tau268 = None

    tau277 -= 2 * np.einsum("kjia->ijka", tau269, optimize=True)

    tau269 = None

    tau242 = zeros((M, M, M, M))

    tau242 += np.einsum("ijab,ijcd->abcd", tau1, tau40, optimize=True)

    tau243 = zeros((M, M, M, M))

    tau243 += 4 * np.einsum("abcd->abcd", tau242, optimize=True)

    tau242 = None

    tau288 = zeros((N, N, N, N))

    tau288 += np.einsum("ijab,klab->ijkl", tau1, tau40, optimize=True)

    tau290 -= 4 * np.einsum("ijlk->ijkl", tau288, optimize=True)

    tau288 = None

    tau297 = zeros((N, N, N, M))

    tau297 += np.einsum("bi,jkba->ijka", l1, tau40, optimize=True)

    tau298 += 2 * np.einsum("ijka->ijka", tau297, optimize=True)

    tau297 = None

    tau41 = zeros((M, M, M, M))

    tau41 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau42 = zeros((M, M, M, M))

    tau42 += np.einsum("badc->abcd", tau41, optimize=True)

    tau144 = zeros((N, N, M, M))

    tau144 -= np.einsum("ijcd,cabd->ijab", tau1, tau41, optimize=True)

    tau147 -= np.einsum("jiba->ijab", tau144, optimize=True)

    tau144 = None

    tau227 = zeros((N, N, M, M, M, M))

    tau227 += np.einsum("ceji,abed->ijabcd", t2, tau41, optimize=True)

    tau41 = None

    tau228 = zeros((N, N, M, M, M, M))

    tau228 -= np.einsum("ijbacd->ijabcd", tau227, optimize=True)

    tau227 = None

    tau42 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau57 += 2 * np.einsum("ijcd,cadb->ijab", tau1, tau42, optimize=True)

    tau101 = zeros((N, M, M, M))

    tau101 += np.einsum("di,abdc->iabc", t1, tau42, optimize=True)

    tau102 = zeros((N, M, M, M))

    tau102 -= np.einsum("ibac->iabc", tau101, optimize=True)

    tau101 = None

    tau241 = zeros((M, M, M, M))

    tau241 += np.einsum("eafb,ecfd->abcd", tau3, tau42, optimize=True)

    tau3 = None

    tau243 += np.einsum("abcd->abcd", tau241, optimize=True)

    tau241 = None

    tau244 = zeros((N, N, M, M))

    tau244 += np.einsum("dcij,cabd->ijab", t2, tau243, optimize=True)

    tau243 = None

    tau282 -= 2 * np.einsum("jiab->ijab", tau244, optimize=True)

    tau244 = None

    tau43 += 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau57 += np.einsum("jkca,kicb->ijab", tau43, u[o, o, v, v], optimize=True)

    tau43 = None

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau45 += np.einsum("ijab->ijab", tau44, optimize=True)

    tau105 += np.einsum("jiab->ijab", tau44, optimize=True)

    tau231 = zeros((N, N, M, M))

    tau231 += np.einsum("ijab->ijab", tau44, optimize=True)

    tau264 = zeros((N, N, N, M))

    tau264 -= np.einsum("bj,ikab->ijka", t1, tau44, optimize=True)

    tau44 = None

    tau277 += 2 * np.einsum("kjia->ijka", tau264, optimize=True)

    tau279 = zeros((N, N, N, M))

    tau279 += 2 * np.einsum("ijka->ijka", tau264, optimize=True)

    tau264 = None

    tau45 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau57 += 2 * np.einsum("kilj,klab->ijab", tau0, tau45, optimize=True)

    tau45 = None

    tau46 = zeros((N, N, N, N))

    tau46 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau47 = zeros((N, N, N, N))

    tau47 += np.einsum("lkji->ijkl", tau46, optimize=True)

    tau130 = zeros((N, N, N, N))

    tau130 += np.einsum("lkji->ijkl", tau46, optimize=True)

    tau140 = zeros((N, N, M, M))

    tau140 += np.einsum("klab,kijl->ijab", tau1, tau46, optimize=True)

    tau147 += np.einsum("ijab->ijab", tau140, optimize=True)

    tau140 = None

    tau148 = zeros((N, N, M, M))

    tau148 += np.einsum("cbkj,ikca->ijab", t2, tau147, optimize=True)

    tau147 = None

    tau235 -= 2 * np.einsum("ijab->ijab", tau148, optimize=True)

    tau148 = None

    tau245 = zeros((N, N, M, M))

    tau245 -= np.einsum("ablk,lkji->ijab", l2, tau46, optimize=True)

    tau246 += np.einsum("ijba->ijab", tau245, optimize=True)

    tau245 = None

    tau275 += np.einsum("lkji->ijkl", tau46, optimize=True)

    tau287 = zeros((N, N, N, N))

    tau287 -= np.einsum("imkn,jmnl->ijkl", tau0, tau46, optimize=True)

    tau290 += np.einsum("ijlk->ijkl", tau287, optimize=True)

    tau287 = None

    r2 += np.einsum("jilk,klab->abij", tau46, tau8, optimize=True) / 2

    tau8 = None

    tau47 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau57 += 2 * np.einsum("klba,likj->ijab", tau1, tau47, optimize=True)

    tau63 -= np.einsum("bakl,jikl->ijab", l2, tau47, optimize=True)

    tau116 = zeros((N, N))

    tau116 += np.einsum("klmi,mjkl->ij", tau0, tau47, optimize=True)

    tau136 += np.einsum("ji->ij", tau116, optimize=True)

    tau292 += np.einsum("ji->ij", tau116, optimize=True)

    tau116 = None

    tau48 = zeros((N, M))

    tau48 += np.einsum("bj,baji->ia", l1, t2, optimize=True)

    tau49 = zeros((N, M))

    tau49 += np.einsum("ia->ia", tau48, optimize=True)

    tau93 = zeros((N, N))

    tau93 += np.einsum("ai,ja->ij", l1, tau48, optimize=True)

    tau208 = zeros((N, N, M, M))

    tau208 -= np.einsum("ic,jabc->ijab", tau48, u[o, v, v, v], optimize=True)

    tau213 += 4 * np.einsum("ijab->ijab", tau208, optimize=True)

    tau208 = None

    tau215 = zeros((N, N, N, M))

    tau215 += np.einsum("ib,jkab->ijka", tau48, tau40, optimize=True)

    tau216 -= 4 * np.einsum("ikja->ijka", tau215, optimize=True)

    tau215 = None

    tau283 = zeros((N, N, M, M))

    tau283 -= np.einsum("ic,abjc->ijab", tau48, u[v, v, o, v], optimize=True)

    tau307 = zeros((N, N, M, M))

    tau307 += 8 * np.einsum("ijba->ijab", tau283, optimize=True)

    tau283 = None

    tau304 = zeros((N, M, M, M))

    tau304 += np.einsum("id,abdc->iabc", tau48, tau42, optimize=True)

    tau305 = zeros((N, M, M, M))

    tau305 -= np.einsum("ibac->iabc", tau304, optimize=True)

    tau304 = None

    tau312 = zeros((N, N, N, M))

    tau312 -= np.einsum("ib,jkab->ijka", tau48, u[o, o, v, v], optimize=True)

    tau313 = zeros((N, N, N, N))

    tau313 += np.einsum("ai,jkla->ijkl", t1, tau312, optimize=True)

    tau312 = None

    tau315 = zeros((N, N, N, N))

    tau315 -= np.einsum("ilkj->ijkl", tau313, optimize=True)

    tau313 = None

    tau49 += np.einsum("ai->ia", t1, optimize=True)

    tau57 += 4 * np.einsum("jc,iacb->ijab", tau49, u[o, v, v, v], optimize=True)

    tau92 += 2 * np.einsum("kb,jiba->ijka", tau49, u[o, o, v, v], optimize=True)

    tau314 = zeros((N, N, N, N))

    tau314 += np.einsum("la,ijka->ijkl", tau49, u[o, o, o, v], optimize=True)

    tau315 -= np.einsum("kjli->ijkl", tau314, optimize=True)

    tau314 = None

    tau316 = zeros((N, N, N, M))

    tau316 += np.einsum("al,iljk->ijka", t1, tau315, optimize=True)

    tau315 = None

    tau317 = zeros((N, N, M, M))

    tau317 -= np.einsum("bk,ikja->ijab", t1, tau316, optimize=True)

    tau316 = None

    tau318 = zeros((N, N, M, M))

    tau318 += np.einsum("ijba->ijab", tau317, optimize=True)

    tau317 = None

    tau54 -= np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau55 = zeros((N, M, M, M))

    tau55 += np.einsum("jkab,kijc->iabc", tau1, tau54, optimize=True)

    tau56 -= 2 * np.einsum("iabc->iabc", tau55, optimize=True)

    tau57 -= 2 * np.einsum("cj,ibac->ijab", t1, tau56, optimize=True)

    tau56 = None

    tau223 -= 2 * np.einsum("iabc->iabc", tau55, optimize=True)

    tau55 = None

    tau224 = zeros((N, M, M, M))

    tau224 += np.einsum("dcji,jdab->iabc", t2, tau223, optimize=True)

    tau223 = None

    tau233 -= 2 * np.einsum("iacb->iabc", tau224, optimize=True)

    tau224 = None

    tau91 = zeros((N, N, N, M))

    tau91 += np.einsum("limj,mkla->ijka", tau0, tau54, optimize=True)

    tau92 -= np.einsum("jkia->ijka", tau91, optimize=True)

    tau298 -= np.einsum("ijka->ijka", tau91, optimize=True)

    tau91 = None

    tau299 = zeros((N, M, M, M))

    tau299 += np.einsum("bckj,jika->iabc", t2, tau298, optimize=True)

    tau298 = None

    tau305 += np.einsum("icba->iabc", tau299, optimize=True)

    tau299 = None

    r1 -= np.einsum("jk,kija->ai", tau93, tau54, optimize=True)

    tau57 += 4 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bj,ijba->ai", l1, tau57, optimize=True) / 4

    tau57 = None

    tau58 = zeros((N, N, M, M))

    tau58 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau63 -= 2 * np.einsum("jiba->ijab", tau58, optimize=True)

    tau94 += 2 * np.einsum("jiba->ijab", tau58, optimize=True)

    tau246 -= 2 * np.einsum("jiba->ijab", tau58, optimize=True)

    tau58 = None

    tau59 = zeros((M, M))

    tau59 -= np.einsum("acji,cbji->ab", l2, t2, optimize=True)

    tau60 = zeros((N, N, M, M))

    tau60 += np.einsum("ac,jibc->ijab", tau59, u[o, o, v, v], optimize=True)

    tau63 += 2 * np.einsum("ijab->ijab", tau60, optimize=True)

    r1 -= np.einsum("jb,jiba->ai", tau49, tau63, optimize=True) / 4

    tau63 = None

    tau67 = zeros((N, N, M, M))

    tau67 -= np.einsum("ikac,jkcb->ijab", tau1, tau60, optimize=True)

    tau75 += 2 * np.einsum("ijab->ijab", tau67, optimize=True)

    tau67 = None

    tau84 += np.einsum("ijba->ijab", tau75, optimize=True)

    tau84 += np.einsum("jiab->ijab", tau75, optimize=True)

    tau75 = None

    tau111 = zeros((N, N, M, M))

    tau111 -= np.einsum("ijab->ijab", tau60, optimize=True)

    tau210 = zeros((N, N, M, M))

    tau210 -= np.einsum("acik,jkcb->ijab", t2, tau60, optimize=True)

    tau60 = None

    tau213 += 2 * np.einsum("ijab->ijab", tau210, optimize=True)

    tau210 = None

    tau79 = zeros((N, N, M, M))

    tau79 += np.einsum("cb,acij->ijab", tau59, t2, optimize=True)

    tau80 += np.einsum("ijab->ijab", tau79, optimize=True)

    tau81 += np.einsum("ilab,kjab->ijkl", tau80, u[o, o, v, v], optimize=True)

    tau80 = None

    tau265 += np.einsum("ijab->ijab", tau79, optimize=True)

    tau319 += np.einsum("ijab->ijab", tau79, optimize=True)

    tau321 -= np.einsum("ijab->ijab", tau79, optimize=True)

    tau327 -= np.einsum("klab,jiab->ijkl", tau321, u[o, o, v, v], optimize=True)

    tau321 = None

    tau333 = zeros((N, N, M, M))

    tau333 += np.einsum("ijab->ijab", tau79, optimize=True)

    tau79 = None

    tau96 = zeros((N, M, M, M))

    tau96 -= np.einsum("ad,ibcd->iabc", tau59, u[o, v, v, v], optimize=True)

    tau97 -= 2 * np.einsum("ibac->iabc", tau96, optimize=True)

    tau220 = zeros((N, M, M, M))

    tau220 -= np.einsum("adij,jdbc->iabc", t2, tau96, optimize=True)

    tau96 = None

    tau233 += 2 * np.einsum("iabc->iabc", tau220, optimize=True)

    tau220 = None

    tau104 = zeros((M, M))

    tau104 += np.einsum("cd,cadb->ab", tau59, tau42, optimize=True)

    tau42 = None

    tau107 = zeros((M, M))

    tau107 -= 2 * np.einsum("ab->ab", tau104, optimize=True)

    tau262 = zeros((M, M))

    tau262 -= 2 * np.einsum("ab->ab", tau104, optimize=True)

    tau104 = None

    tau108 = zeros((N, M))

    tau108 -= np.einsum("bc,ibca->ia", tau59, u[o, v, v, v], optimize=True)

    tau113 = zeros((N, M))

    tau113 += np.einsum("ia->ia", tau108, optimize=True)

    tau108 = None

    tau117 = zeros((N, N, N, M))

    tau117 -= np.einsum("ab,ijkb->ijka", tau59, u[o, o, o, v], optimize=True)

    tau121 += 2 * np.einsum("ijka->ijka", tau117, optimize=True)

    tau192 = zeros((N, N, N, M))

    tau192 -= np.einsum("abil,jlkb->ijka", t2, tau117, optimize=True)

    tau117 = None

    tau216 -= 2 * np.einsum("ijka->ijka", tau192, optimize=True)

    tau192 = None

    tau168 = zeros((N, N, M, M))

    tau168 += np.einsum("ac,ijbc->ijab", tau59, tau40, optimize=True)

    tau40 = None

    tau169 = zeros((N, N, M, M))

    tau169 += np.einsum("jiab->ijab", tau168, optimize=True)

    tau168 = None

    tau77 = zeros((N, N, N, N))

    tau77 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau81 += 4 * np.einsum("ikjl->ijkl", tau77, optimize=True)

    tau84 -= np.einsum("bakl,ljik->ijab", l2, tau81, optimize=True)

    tau81 = None

    tau130 += 2 * np.einsum("kjil->ijkl", tau77, optimize=True)

    tau152 = zeros((N, N, N, N))

    tau152 += np.einsum("kjil->ijkl", tau77, optimize=True)

    tau190 = zeros((N, N, N, M))

    tau190 += np.einsum("la,iljk->ijka", tau48, tau77, optimize=True)

    tau77 = None

    tau216 += 4 * np.einsum("ijka->ijka", tau190, optimize=True)

    tau190 = None

    tau83 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau84 += 2 * np.einsum("kjla,klib->ijab", tau25, tau83, optimize=True)

    tau25 = None

    tau84 += 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,jiba->ai", t1, tau84, optimize=True) / 4

    tau84 = None

    tau85 += 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    r1 += np.einsum("bc,ibca->ai", tau59, tau85, optimize=True) / 4

    tau85 = None

    tau86 = zeros((N, N))

    tau86 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau88 = zeros((N, N))

    tau88 += 2 * np.einsum("ij->ij", tau86, optimize=True)

    tau300 = zeros((N, N, M, M))

    tau300 += np.einsum("ki,abjk->ijab", tau86, t2, optimize=True)

    tau86 = None

    tau302 = zeros((N, N, M, M))

    tau302 -= 2 * np.einsum("ijba->ijab", tau300, optimize=True)

    tau330 += 2 * np.einsum("jiba->ijab", tau300, optimize=True)

    tau300 = None

    tau87 = zeros((N, N))

    tau87 -= np.einsum("baik,bakj->ij", l2, t2, optimize=True)

    tau88 += np.einsum("ij->ij", tau87, optimize=True)

    tau110 = zeros((N, M))

    tau110 += np.einsum("kj,jika->ia", tau88, tau54, optimize=True)

    tau54 = None

    tau113 -= np.einsum("ia->ia", tau110, optimize=True)

    tau110 = None

    tau185 = zeros((N, N, M, M))

    tau185 -= np.einsum("kl,kijlab->ijab", tau88, tau184, optimize=True)

    tau184 = None

    tau235 -= 2 * np.einsum("ijab->ijab", tau185, optimize=True)

    tau185 = None

    tau206 = zeros((N, N, N, M))

    tau206 += np.einsum("lm,ilmjka->ijka", tau88, tau35, optimize=True)

    tau35 = None

    tau216 += 2 * np.einsum("ijka->ijka", tau206, optimize=True)

    tau206 = None

    tau211 = zeros((N, N, M, M))

    tau211 += np.einsum("kl,ikljab->ijab", tau88, tau26, optimize=True)

    tau26 = None

    tau213 -= 2 * np.einsum("ijab->ijab", tau211, optimize=True)

    tau211 = None

    tau230 = zeros((N, M, M, M))

    tau230 += np.einsum("jk,ijkabc->iabc", tau88, tau29, optimize=True)

    tau29 = None

    tau233 += 2 * np.einsum("iabc->iabc", tau230, optimize=True)

    tau230 = None

    tau301 = zeros((N, N, M, M))

    tau301 += np.einsum("kj,abik->ijab", tau87, t2, optimize=True)

    tau87 = None

    tau302 += np.einsum("ijba->ijab", tau301, optimize=True)

    tau303 = zeros((N, M, M, M))

    tau303 += np.einsum("jkab,kjic->iabc", tau302, u[o, o, o, v], optimize=True)

    tau305 -= np.einsum("iabc->iabc", tau303, optimize=True)

    tau303 = None

    tau306 = zeros((N, N, M, M))

    tau306 += np.einsum("cj,iabc->ijab", t1, tau305, optimize=True)

    tau305 = None

    tau307 -= 4 * np.einsum("jiba->ijab", tau306, optimize=True)

    tau306 = None

    r2 += np.einsum("klab,lkji->abij", tau302, tau47, optimize=True) / 4

    tau302 = None

    tau47 = None

    tau330 -= np.einsum("jiba->ijab", tau301, optimize=True)

    tau301 = None

    tau92 += 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    r1 += np.einsum("jk,kija->ai", tau88, tau92, optimize=True) / 4

    tau92 = None

    tau94 += 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau95 = zeros((M, M))

    tau95 += np.einsum("caij,ijcb->ab", t2, tau94, optimize=True)

    tau94 = None

    tau107 += np.einsum("ab->ab", tau95, optimize=True)

    tau262 += np.einsum("ab->ab", tau95, optimize=True)

    tau95 = None

    tau97 -= 4 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau98 = zeros((M, M))

    tau98 += np.einsum("ci,iacb->ab", t1, tau97, optimize=True)

    tau97 = None

    tau107 -= 2 * np.einsum("ab->ab", tau98, optimize=True)

    tau262 -= 2 * np.einsum("ab->ab", tau98, optimize=True)

    tau98 = None

    tau99 = zeros((N, M, M, M))

    tau99 += np.einsum("abkj,kjic->iabc", t2, u[o, o, o, v], optimize=True)

    tau102 -= np.einsum("ibac->iabc", tau99, optimize=True)

    tau177 = zeros((N, N, N, M, M, M))

    tau177 -= np.einsum("cdji,kabd->ijkabc", t2, tau99, optimize=True)

    tau179 = zeros((N, N, N, M, M, M))

    tau179 -= np.einsum("ijkbac->ijkabc", tau177, optimize=True)

    tau177 = None

    tau295 = zeros((N, N, M, M))

    tau295 += np.einsum("jc,iabc->ijab", tau49, tau99, optimize=True)

    tau99 = None

    tau307 -= 4 * np.einsum("jiba->ijab", tau295, optimize=True)

    tau295 = None

    tau100 = zeros((N, M, M, M))

    tau100 += np.einsum("daji,jbcd->iabc", t2, u[o, v, v, v], optimize=True)

    tau102 -= 2 * np.einsum("iabc->iabc", tau100, optimize=True)

    tau102 += 2 * np.einsum("ibac->iabc", tau100, optimize=True)

    tau178 = zeros((N, N, N, M, M, M))

    tau178 -= np.einsum("adji,kbcd->ijkabc", t2, tau100, optimize=True)

    tau179 += 2 * np.einsum("ijkacb->ijkabc", tau178, optimize=True)

    tau179 += 2 * np.einsum("kjiabc->ijkabc", tau178, optimize=True)

    tau237 = zeros((N, N, M, M))

    tau237 += np.einsum("ck,ijkcab->ijab", l1, tau178, optimize=True)

    tau178 = None

    tau282 += 8 * np.einsum("ijab->ijab", tau237, optimize=True)

    tau237 = None

    tau218 = zeros((N, N, M, M))

    tau218 += np.einsum("jc,iabc->ijab", tau49, tau100, optimize=True)

    tau100 = None

    tau235 -= 4 * np.einsum("jiab->ijab", tau218, optimize=True)

    tau218 = None

    tau102 -= 2 * np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau103 = zeros((M, M))

    tau103 += np.einsum("ci,icab->ab", l1, tau102, optimize=True)

    tau102 = None

    tau107 -= 4 * np.einsum("ab->ab", tau103, optimize=True)

    tau262 -= 4 * np.einsum("ab->ab", tau103, optimize=True)

    tau103 = None

    tau105 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau106 = zeros((M, M))

    tau106 += np.einsum("ij,jiab->ab", tau88, tau105, optimize=True)

    tau107 -= 4 * np.einsum("ab->ab", tau106, optimize=True)

    tau262 -= 4 * np.einsum("ab->ab", tau106, optimize=True)

    tau106 = None

    tau263 = zeros((N, N, M, M))

    tau263 += np.einsum("ac,cbij->ijab", tau262, t2, optimize=True)

    tau262 = None

    tau282 -= np.einsum("jiba->ijab", tau263, optimize=True)

    tau263 = None

    tau109 = zeros((N, M))

    tau109 += np.einsum("bj,ijba->ia", l1, tau105, optimize=True)

    tau105 = None

    tau113 += 2 * np.einsum("ia->ia", tau109, optimize=True)

    tau109 = None

    tau107 -= 8 * np.einsum("ab->ab", f[v, v], optimize=True)

    r1 -= np.einsum("bi,ba->ai", l1, tau107, optimize=True) / 8

    tau107 = None

    tau111 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau112 = zeros((N, M))

    tau112 += np.einsum("bj,jiba->ia", t1, tau111, optimize=True)

    tau111 = None

    tau113 += np.einsum("ia->ia", tau112, optimize=True)

    tau112 = None

    tau113 += 2 * np.einsum("ia->ia", f[o, v], optimize=True)

    tau132 = zeros((N, N))

    tau132 += np.einsum("ai,ja->ij", t1, tau113, optimize=True)

    tau136 += 4 * np.einsum("ji->ij", tau132, optimize=True)

    tau294 = zeros((N, N, M, M))

    tau294 += np.einsum("ik,abkj->ijab", tau132, t2, optimize=True)

    tau132 = None

    tau307 -= 4 * np.einsum("ijba->ijab", tau294, optimize=True)

    tau294 = None

    tau274 = zeros((N, N, N, M))

    tau274 += np.einsum("kb,baij->ijka", tau113, t2, optimize=True)

    tau277 += np.einsum("kjia->ijka", tau274, optimize=True)

    tau274 = None

    r1 -= np.einsum("ib,ab->ai", tau113, tau59, optimize=True) / 4

    r1 -= np.einsum("ja,ij->ai", tau113, tau88, optimize=True) / 4

    tau113 = None

    tau114 = zeros((N, N))

    tau114 -= np.einsum("baki,jkba->ij", t2, u[o, o, v, v], optimize=True)

    tau136 += 4 * np.einsum("ji->ij", tau114, optimize=True)

    tau292 += 4 * np.einsum("ji->ij", tau114, optimize=True)

    tau114 = None

    tau119 -= 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau120 = zeros((N, N, N, M))

    tau120 += np.einsum("limj,mkla->ijka", tau0, tau119, optimize=True)

    tau119 = None

    tau121 -= np.einsum("ikja->ijka", tau120, optimize=True)

    tau120 = None

    tau121 -= 4 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau122 = zeros((N, N))

    tau122 += np.einsum("ak,kija->ij", t1, tau121, optimize=True)

    tau121 = None

    tau136 -= 2 * np.einsum("ij->ij", tau122, optimize=True)

    tau292 -= 2 * np.einsum("ij->ij", tau122, optimize=True)

    tau122 = None

    tau123 = zeros((N, N, N, M))

    tau123 += np.einsum("cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True)

    tau126 -= np.einsum("kjia->ijka", tau123, optimize=True)

    tau279 -= np.einsum("jika->ijka", tau123, optimize=True)

    tau326 = zeros((N, N, N, M))

    tau326 -= np.einsum("jika->ijka", tau123, optimize=True)

    tau124 = zeros((N, N, N, M))

    tau124 += np.einsum("bali,jlkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau126 -= 2 * np.einsum("jika->ijka", tau124, optimize=True)

    tau126 += 2 * np.einsum("kija->ijka", tau124, optimize=True)

    tau182 = zeros((N, N, N, M))

    tau182 += np.einsum("ijka->ijka", tau124, optimize=True)

    tau188 += np.einsum("ijka->ijka", tau124, optimize=True)

    tau189 = zeros((N, N, M, M))

    tau189 += np.einsum("kb,ikja->ijab", tau49, tau188, optimize=True)

    tau49 = None

    tau188 = None

    tau235 -= 4 * np.einsum("ijba->ijab", tau189, optimize=True)

    tau189 = None

    tau284 = zeros((N, N, N, N))

    tau284 += np.einsum("ai,jkla->ijkl", l1, tau124, optimize=True)

    tau290 += 4 * np.einsum("ijkl->ijkl", tau284, optimize=True)

    tau284 = None

    tau126 -= 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau127 = zeros((N, N))

    tau127 += np.einsum("ak,ikja->ij", l1, tau126, optimize=True)

    tau126 = None

    tau136 -= 4 * np.einsum("ij->ij", tau127, optimize=True)

    tau292 -= 4 * np.einsum("ij->ij", tau127, optimize=True)

    tau127 = None

    tau128 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau129 = zeros((N, N))

    tau129 += np.einsum("ab,ijab->ij", tau59, tau128, optimize=True)

    tau59 = None

    tau136 += 4 * np.einsum("ij->ij", tau129, optimize=True)

    tau292 += 4 * np.einsum("ij->ij", tau129, optimize=True)

    tau129 = None

    tau181 = zeros((N, N, N, M))

    tau181 += np.einsum("bi,jkab->ijka", t1, tau128, optimize=True)

    tau128 = None

    tau182 -= np.einsum("kjia->ijka", tau181, optimize=True)

    tau181 = None

    tau130 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau131 = zeros((N, N))

    tau131 += np.einsum("kl,likj->ij", tau88, tau130, optimize=True)

    tau88 = None

    tau130 = None

    tau136 -= 2 * np.einsum("ij->ij", tau131, optimize=True)

    tau292 -= 2 * np.einsum("ij->ij", tau131, optimize=True)

    tau131 = None

    tau133 = zeros((N, M))

    tau133 -= np.einsum("bj,ijba->ia", t1, u[o, o, v, v], optimize=True)

    tau134 = zeros((N, M))

    tau134 += np.einsum("ia->ia", tau133, optimize=True)

    tau133 = None

    tau134 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau135 = zeros((N, N))

    tau135 += np.einsum("ja,ia->ij", tau134, tau48, optimize=True)

    tau136 += 8 * np.einsum("ji->ij", tau135, optimize=True)

    tau292 += 8 * np.einsum("ji->ij", tau135, optimize=True)

    tau135 = None

    tau293 = zeros((N, N, M, M))

    tau293 += np.einsum("ki,abkj->ijab", tau292, t2, optimize=True)

    tau292 = None

    tau307 += np.einsum("jiba->ijab", tau293, optimize=True)

    tau293 = None

    tau186 = zeros((N, N, N, M))

    tau186 += np.einsum("kb,baij->ijka", tau134, t2, optimize=True)

    tau281 = zeros((N, N, M, M))

    tau281 += np.einsum("kb,jika->ijab", tau48, tau186, optimize=True)

    tau282 += 8 * np.einsum("jiab->ijab", tau281, optimize=True)

    tau281 = None

    tau326 -= 2 * np.einsum("jika->ijka", tau186, optimize=True)

    r1 -= np.einsum("ja,ij->ai", tau134, tau93, optimize=True)

    tau93 = None

    tau134 = None

    tau136 += 8 * np.einsum("ij->ij", f[o, o], optimize=True)

    r1 -= np.einsum("aj,ij->ai", l1, tau136, optimize=True) / 8

    tau136 = None

    tau137 = zeros((N, N, N, M))

    tau137 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau138 = zeros((N, N, M, M))

    tau138 -= np.einsum("klja,kilb->ijab", tau124, tau137, optimize=True)

    tau124 = None

    tau235 -= 4 * np.einsum("ijab->ijab", tau138, optimize=True)

    tau138 = None

    tau139 = zeros((N, N, M, M))

    tau139 += np.einsum("iklb,kjla->ijab", tau123, tau137, optimize=True)

    tau123 = None

    tau235 += 2 * np.einsum("ijab->ijab", tau139, optimize=True)

    tau139 = None

    tau187 = zeros((N, N, M, M))

    tau187 -= np.einsum("kljb,kila->ijab", tau137, tau186, optimize=True)

    tau186 = None

    tau235 += 4 * np.einsum("ijab->ijab", tau187, optimize=True)

    tau187 = None

    tau191 = zeros((N, N, N, M))

    tau191 -= np.einsum("ljma,ilmk->ijka", tau137, tau46, optimize=True)

    tau46 = None

    tau216 += 2 * np.einsum("ikja->ijka", tau191, optimize=True)

    tau191 = None

    tau212 = zeros((N, N, M, M))

    tau212 += np.einsum("kljb,klia->ijab", tau137, tau83, optimize=True)

    tau83 = None

    tau213 += 4 * np.einsum("jiba->ijab", tau212, optimize=True)

    tau212 = None

    tau214 = zeros((N, N, N, M))

    tau214 += np.einsum("bk,ijab->ijka", t1, tau213, optimize=True)

    tau213 = None

    tau216 += np.einsum("kjia->ijka", tau214, optimize=True)

    tau214 = None

    tau219 = zeros((N, M, M, M))

    tau219 -= np.einsum("jikb,jkac->iabc", tau137, tau39, optimize=True)

    tau39 = None

    tau233 += 4 * np.einsum("iabc->iabc", tau219, optimize=True)

    tau219 = None

    tau152 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau153 = zeros((N, N, M, M))

    tau153 += np.einsum("lkab,kilj->ijab", tau1, tau152, optimize=True)

    tau1 = None

    tau154 -= 2 * np.einsum("ijab->ijab", tau153, optimize=True)

    tau153 = None

    tau155 = zeros((N, N, M, M))

    tau155 += np.einsum("cbkj,kica->ijab", t2, tau154, optimize=True)

    tau154 = None

    tau235 += 2 * np.einsum("jiab->ijab", tau155, optimize=True)

    tau155 = None

    tau207 = zeros((N, N, N, M))

    tau207 += np.einsum("mlka,limj->ijka", tau137, tau152, optimize=True)

    tau216 += 4 * np.einsum("jkia->ijka", tau207, optimize=True)

    tau207 = None

    tau289 = zeros((N, N, N, N))

    tau289 += np.einsum("nkml,minj->ijkl", tau0, tau152, optimize=True)

    tau0 = None

    tau152 = None

    tau290 -= 2 * np.einsum("klij->ijkl", tau289, optimize=True)

    tau289 = None

    tau291 = zeros((N, N, M, M))

    tau291 += np.einsum("ablk,kilj->ijab", t2, tau290, optimize=True)

    tau290 = None

    tau307 -= 2 * np.einsum("ijba->ijab", tau291, optimize=True)

    tau291 = None

    r2 += np.einsum("ijba->abij", tau307, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau307, optimize=True) / 8

    tau307 = None

    tau169 += 2 * np.einsum("ibja->ijab", u[o, v, o, v], optimize=True)

    tau170 = zeros((N, N, M, M))

    tau170 += np.einsum("cbkj,kica->ijab", t2, tau169, optimize=True)

    tau169 = None

    tau235 += 2 * np.einsum("jiba->ijab", tau170, optimize=True)

    tau170 = None

    tau176 = zeros((N, N, N, M, M, M))

    tau176 -= np.einsum("adji,bckd->ijkabc", t2, u[v, v, o, v], optimize=True)

    tau179 -= 2 * np.einsum("ijkacb->ijkabc", tau176, optimize=True)

    tau180 = zeros((N, N, M, M))

    tau180 += np.einsum("ck,ikjacb->ijab", l1, tau179, optimize=True)

    tau179 = None

    tau235 += 2 * np.einsum("ijab->ijab", tau180, optimize=True)

    tau180 = None

    tau328 = zeros((N, N, N, M, M, M))

    tau328 -= np.einsum("ijkacb->ijkabc", tau176, optimize=True)

    tau176 = None

    tau182 += np.einsum("jaki->ijka", u[o, v, o, o], optimize=True)

    tau183 = zeros((N, N, M, M))

    tau183 += np.einsum("lkjb,ikla->ijab", tau137, tau182, optimize=True)

    tau182 = None

    tau235 -= 4 * np.einsum("jiba->ijab", tau183, optimize=True)

    tau183 = None

    tau202 = zeros((N, N, N, N, M, M))

    tau202 -= np.einsum("acji,kblc->ijklab", t2, u[o, v, o, v], optimize=True)

    tau204 += np.einsum("ijklab->ijklab", tau202, optimize=True)

    tau202 = None

    tau205 = zeros((N, N, N, M))

    tau205 += np.einsum("bl,iljkab->ijka", l1, tau204, optimize=True)

    tau204 = None

    tau216 += 4 * np.einsum("ijka->ijka", tau205, optimize=True)

    tau205 = None

    tau217 = zeros((N, N, M, M))

    tau217 += np.einsum("bk,ikja->ijab", t1, tau216, optimize=True)

    tau216 = None

    tau235 -= np.einsum("ijba->ijab", tau217, optimize=True)

    tau217 = None

    tau226 = zeros((N, N, M, M, M, M))

    tau226 += np.einsum("aeji,bced->ijabcd", t2, u[v, v, v, v], optimize=True)

    tau228 -= 2 * np.einsum("ijacbd->ijabcd", tau226, optimize=True)

    tau229 = zeros((N, M, M, M))

    tau229 += np.einsum("dj,ijadbc->iabc", l1, tau228, optimize=True)

    tau228 = None

    tau233 -= 2 * np.einsum("iabc->iabc", tau229, optimize=True)

    tau229 = None

    tau328 -= np.einsum("dk,ijacbd->ijkabc", t1, tau226, optimize=True)

    tau226 = None

    r2 += np.einsum("ck,jikcab->abij", l1, tau328, optimize=True)

    tau328 = None

    tau231 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau232 = zeros((N, M, M, M))

    tau232 += np.einsum("jkic,jkab->iabc", tau137, tau231, optimize=True)

    tau137 = None

    tau231 = None

    tau233 -= 4 * np.einsum("ibca->iabc", tau232, optimize=True)

    tau232 = None

    tau234 = zeros((N, N, M, M))

    tau234 += np.einsum("cj,iabc->ijab", t1, tau233, optimize=True)

    tau233 = None

    tau235 += np.einsum("jiab->ijab", tau234, optimize=True)

    tau234 = None

    r2 -= np.einsum("ijab->abij", tau235, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau235, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau235, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau235, optimize=True) / 4

    tau235 = None

    tau236 = zeros((N, N, M, M))

    tau236 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    tau282 += 8 * np.einsum("jiab->ijab", tau236, optimize=True)

    tau236 = None

    tau238 = zeros((N, N, M, M))

    tau238 += np.einsum("ablk,ijlk->ijab", l2, u[o, o, o, o], optimize=True)

    tau239 = zeros((N, N, M, M))

    tau239 += np.einsum("bcik,kjca->ijab", t2, tau238, optimize=True)

    tau238 = None

    tau240 = zeros((N, N, M, M))

    tau240 += np.einsum("bcjk,ikca->ijab", t2, tau239, optimize=True)

    tau239 = None

    tau282 -= 4 * np.einsum("ijba->ijab", tau240, optimize=True)

    tau240 = None

    tau246 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau247 = zeros((N, N, M, M))

    tau247 += np.einsum("cbkj,ikca->ijab", t2, tau246, optimize=True)

    tau246 = None

    tau248 = zeros((N, N, M, M))

    tau248 += np.einsum("cbkj,kica->ijab", t2, tau247, optimize=True)

    tau247 = None

    tau282 -= 2 * np.einsum("jiab->ijab", tau248, optimize=True)

    tau248 = None

    tau265 -= np.einsum("baji->ijab", t2, optimize=True)

    tau266 = zeros((N, N, N, M))

    tau266 += np.einsum("ijbc,kabc->ijka", tau265, u[o, v, v, v], optimize=True)

    tau265 = None

    tau277 += np.einsum("jkia->ijka", tau266, optimize=True)

    tau266 = None

    tau275 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau276 = zeros((N, N, N, M))

    tau276 += np.einsum("la,lijk->ijka", tau48, tau275, optimize=True)

    tau275 = None

    tau277 += np.einsum("ikja->ijka", tau276, optimize=True)

    tau276 = None

    tau278 = zeros((N, N, M, M))

    tau278 += np.einsum("bk,kija->ijab", t1, tau277, optimize=True)

    tau277 = None

    tau282 -= 4 * np.einsum("jiba->ijab", tau278, optimize=True)

    tau278 = None

    tau279 -= 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau280 = zeros((N, N, M, M))

    tau280 += np.einsum("kb,ijka->ijab", tau48, tau279, optimize=True)

    tau48 = None

    tau279 = None

    tau282 += 4 * np.einsum("jiba->ijab", tau280, optimize=True)

    tau280 = None

    r2 += np.einsum("ijab->abij", tau282, optimize=True) / 8

    r2 -= np.einsum("ijba->abij", tau282, optimize=True) / 8

    tau282 = None

    tau308 = zeros((N, N, M, M))

    tau308 += np.einsum("ki,abjk->ijab", f[o, o], t2, optimize=True)

    tau318 += np.einsum("ijba->ijab", tau308, optimize=True)

    tau308 = None

    tau310 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau311 = zeros((N, N, M, M))

    tau311 += np.einsum("cj,iabc->ijab", t1, tau310, optimize=True)

    tau310 = None

    tau318 -= np.einsum("jiba->ijab", tau311, optimize=True)

    tau311 = None

    r2 -= np.einsum("ijab->abij", tau318, optimize=True)

    r2 += np.einsum("jiab->abij", tau318, optimize=True)

    tau318 = None

    tau319 -= np.einsum("baji->ijab", t2, optimize=True)

    r2 += np.einsum("ijcd,bacd->abij", tau319, u[v, v, v, v], optimize=True) / 2

    tau319 = None

    tau320 += np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau327 += np.einsum("ablk,jiab->ijkl", t2, tau320, optimize=True)

    tau320 = None

    tau326 -= 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    tau327 -= 2 * np.einsum("aj,lkia->ijkl", l1, tau326, optimize=True)

    tau326 = None

    tau327 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 += np.einsum("bakl,klji->abij", t2, tau327, optimize=True) / 4

    tau327 = None

    tau329 = zeros((N, N, M, M))

    tau329 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau331 = zeros((M, M, M, M))

    tau331 += 2 * np.einsum("abji,ijcd->abcd", t2, tau329, optimize=True)

    tau329 = None

    tau330 += np.einsum("baji->ijab", t2, optimize=True)

    tau331 += np.einsum("ijba,ijdc->abcd", tau330, u[o, o, v, v], optimize=True)

    tau330 = None

    tau331 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau332 = zeros((N, M, M, M))

    tau332 += np.einsum("di,badc->iabc", t1, tau331, optimize=True)

    tau331 = None

    r2 += np.einsum("cj,ibac->abij", t1, tau332, optimize=True) / 2

    tau332 = None

    tau333 -= np.einsum("baji->ijab", t2, optimize=True)

    tau334 += np.einsum("klab,jiab->ijkl", tau333, u[o, o, v, v], optimize=True)

    tau333 = None

    tau334 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau335 = zeros((N, N, N, M))

    tau335 -= np.einsum("al,lkji->ijka", t1, tau334, optimize=True)

    tau334 = None

    tau335 -= 2 * np.einsum("kaji->ijka", u[o, v, o, o], optimize=True)

    r2 += np.einsum("ak,jikb->abij", t1, tau335, optimize=True) / 2

    tau335 = None

    tau337 -= np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    r2 -= np.einsum("bk,kjia->abij", t1, tau337, optimize=True)

    tau337 = None

    r1 += np.einsum("ia->ai", f[o, v], optimize=True)

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    return r1, r2
