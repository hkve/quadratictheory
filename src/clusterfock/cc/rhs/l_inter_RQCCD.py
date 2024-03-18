import numpy as np
from clusterfock.cc.rhs.l_inter_RCCD import lambda_amplitudes_intermediates_rccd


def l_intermediates_qccd_restricted(t2, l2, u, f, v, o):
    r2 = lambda_amplitudes_intermediates_rccd(t2, l2, u, f, v, o)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((M, M))

    tau0 += np.einsum("acji,cbij->ab", l2, t2, optimize=True)

    tau2 = np.zeros((N, N, M, M))

    tau2 += np.einsum("ac,ibjc->ijab", tau0, u[o, v, o, v], optimize=True)

    tau26 = np.zeros((N, N, M, M))

    tau26 -= np.einsum("jiab->ijab", tau2, optimize=True)

    tau190 = np.zeros((N, N, M, M))

    tau190 += 3 * np.einsum("jiab->ijab", tau2, optimize=True)

    tau2 = None

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("ac,jicb->ijab", tau0, u[o, o, v, v], optimize=True)

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("jiab->ijab", tau3, optimize=True)

    tau68 = np.zeros((N, N, M, M))

    tau68 += 2 * np.einsum("ijab->ijab", tau3, optimize=True)

    tau68 -= np.einsum("jiab->ijab", tau3, optimize=True)

    tau118 = np.zeros((N, N, M, M))

    tau118 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau182 = np.zeros((N, N, M, M))

    tau182 += 3 * np.einsum("jiab->ijab", tau3, optimize=True)

    tau382 = np.zeros((N, N, M, M))

    tau382 += np.einsum("ijba->ijab", tau3, optimize=True)

    tau382 += np.einsum("jiab->ijab", tau3, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau56 = np.zeros((N, N, M, M))

    tau56 += np.einsum("ijab->ijab", tau1, optimize=True)

    tau108 = np.zeros((N, N, M, M))

    tau108 += np.einsum("jiab->ijab", tau1, optimize=True)

    tau141 = np.zeros((N, N, M, M))

    tau141 += np.einsum("ijba->ijab", tau1, optimize=True)

    tau162 = np.zeros((N, N, M, M))

    tau162 += np.einsum("jiab->ijab", tau1, optimize=True)

    tau215 = np.zeros((N, N, M, M))

    tau215 += np.einsum("ijba->ijab", tau1, optimize=True)

    tau331 = np.zeros((N, N, M, M))

    tau331 += np.einsum("ijab->ijab", tau1, optimize=True)

    tau335 = np.zeros((N, N, M, M))

    tau335 += np.einsum("jiba->ijab", tau1, optimize=True)

    r2 -= 2 * np.einsum("ac,ijcb->abij", tau0, tau1, optimize=True)

    r2 -= 2 * np.einsum("bc,ijac->abij", tau0, tau1, optimize=True)

    tau1 = None

    tau4 = np.zeros((N, N))

    tau4 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("ik,jkab->ijab", tau4, u[o, o, v, v], optimize=True)

    tau11 += np.einsum("ijba->ijab", tau5, optimize=True)

    tau68 += 2 * np.einsum("ijab->ijab", tau5, optimize=True)

    tau68 -= np.einsum("ijba->ijab", tau5, optimize=True)

    tau108 -= 2 * np.einsum("ijab->ijab", tau5, optimize=True)

    tau108 += np.einsum("ijba->ijab", tau5, optimize=True)

    tau118 += np.einsum("ijab->ijab", tau5, optimize=True)

    tau148 = np.zeros((N, N, M, M))

    tau148 += np.einsum("bc,ijac->ijab", tau0, tau118, optimize=True)

    tau149 = np.zeros((N, N, M, M))

    tau149 += 3 * np.einsum("ijba->ijab", tau148, optimize=True)

    tau148 = None

    tau182 += 3 * np.einsum("ijba->ijab", tau5, optimize=True)

    tau315 = np.zeros((N, N, M, M))

    tau315 -= 3 * np.einsum("ijba->ijab", tau5, optimize=True)

    tau327 = np.zeros((N, N, M, M))

    tau327 += np.einsum("ijba->ijab", tau5, optimize=True)

    tau5 = None

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("kj,abki->ijab", tau4, t2, optimize=True)

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum("ijab->ijab", tau35, optimize=True)

    tau72 = np.zeros((N, N, M, M))

    tau72 += np.einsum("ijab->ijab", tau35, optimize=True)

    tau35 = None

    tau52 = np.zeros((N, N, M, M))

    tau52 += np.einsum("kj,abik->ijab", tau4, t2, optimize=True)

    tau235 = np.zeros((N, N, M, M))

    tau235 += np.einsum("ik,kjab->ijab", tau4, tau3, optimize=True)

    tau238 = np.zeros((N, N, M, M))

    tau238 += np.einsum("ijab->ijab", tau235, optimize=True)

    tau235 = None

    tau313 = np.zeros((N, N, M, M))

    tau313 += np.einsum("ik,kajb->ijab", tau4, u[o, v, o, v], optimize=True)

    tau320 = np.zeros((N, N, M, M))

    tau320 += 3 * np.einsum("jiba->ijab", tau313, optimize=True)

    tau329 = np.zeros((N, N, M, M))

    tau329 -= np.einsum("ijba->ijab", tau313, optimize=True)

    tau313 = None

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("acik,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 -= np.einsum("ijab->ijab", tau6, optimize=True)

    tau38 = np.zeros((N, N, M, M))

    tau38 += np.einsum("caki,kjcb->ijab", l2, tau6, optimize=True)

    tau42 = np.zeros((N, N, M, M))

    tau42 -= 2 * np.einsum("ijab->ijab", tau38, optimize=True)

    tau38 = None

    tau243 = np.zeros((N, N, M, M))

    tau243 -= np.einsum("jiab->ijab", tau6, optimize=True)

    tau6 = None

    tau7 = np.zeros((N, N, M, M))

    tau7 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau7 -= np.einsum("abji->ijab", t2, optimize=True)

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("ikac,kjcb->ijab", tau7, u[o, o, v, v], optimize=True)

    tau9 += np.einsum("ijab->ijab", tau8, optimize=True)

    tau243 += np.einsum("jiab->ijab", tau8, optimize=True)

    tau8 = None

    tau328 = np.zeros((N, N, M, M))

    tau328 += np.einsum("acik,jkbc->ijab", l2, tau7, optimize=True)

    tau392 = np.zeros((N, N, M, M))

    tau392 -= np.einsum("kj,ikab->ijab", tau4, tau7, optimize=True)

    tau7 = None

    tau9 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("bcjk,kica->ijab", l2, tau9, optimize=True)

    tau11 -= np.einsum("jiba->ijab", tau10, optimize=True)

    tau12 = np.zeros((N, N, M, M))

    tau12 += np.einsum("bckj,kiac->ijab", t2, tau11, optimize=True)

    tau11 = None

    tau26 += np.einsum("jiab->ijab", tau12, optimize=True)

    tau12 = None

    tau68 += np.einsum("jiba->ijab", tau10, optimize=True)

    tau141 -= np.einsum("jiba->ijab", tau10, optimize=True)

    tau321 = np.zeros((N, N, M, M))

    tau321 -= np.einsum("jiba->ijab", tau10, optimize=True)

    tau382 -= np.einsum("jiba->ijab", tau10, optimize=True)

    tau107 = np.zeros((N, N, M, M))

    tau107 += np.einsum("cbkj,kica->ijab", l2, tau9, optimize=True)

    tau108 -= np.einsum("jiba->ijab", tau107, optimize=True)

    tau327 -= np.einsum("jiba->ijab", tau107, optimize=True)

    tau360 = np.zeros((N, N, N, N))

    tau360 += 3 * np.einsum("ablj,ikba->ijkl", t2, tau107, optimize=True)

    tau107 = None

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum("acki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau14 = np.zeros((N, N, M, M))

    tau14 += np.einsum("ijab->ijab", tau13, optimize=True)

    tau45 = np.zeros((N, N, M, M))

    tau45 += np.einsum("caik,kjcb->ijab", l2, tau13, optimize=True)

    tau49 = np.zeros((N, N, M, M))

    tau49 -= np.einsum("ijab->ijab", tau45, optimize=True)

    tau357 = np.zeros((N, N, M, M))

    tau357 += np.einsum("ijab->ijab", tau45, optimize=True)

    tau45 = None

    tau358 = np.zeros((N, N, M, M))

    tau358 -= np.einsum("jiab->ijab", tau13, optimize=True)

    tau14 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau67 = np.zeros((N, N, M, M))

    tau67 += np.einsum("bckj,kica->ijab", l2, tau14, optimize=True)

    tau68 -= np.einsum("jiba->ijab", tau67, optimize=True)

    tau69 = np.zeros((N, N, M, M))

    tau69 += np.einsum("bcjk,kiac->ijab", t2, tau68, optimize=True)

    tau68 = None

    tau76 = np.zeros((N, N, M, M))

    tau76 -= np.einsum("jiab->ijab", tau69, optimize=True)

    tau69 = None

    tau108 += np.einsum("jiba->ijab", tau67, optimize=True)

    tau154 = np.zeros((N, N, M, M))

    tau154 -= np.einsum("bckj,ikca->ijab", t2, tau67, optimize=True)

    tau156 = np.zeros((N, N, M, M))

    tau156 -= np.einsum("jiab->ijab", tau154, optimize=True)

    tau154 = None

    tau162 += np.einsum("jiba->ijab", tau67, optimize=True)

    tau182 -= np.einsum("jiba->ijab", tau67, optimize=True)

    tau315 += np.einsum("jiba->ijab", tau67, optimize=True)

    tau348 = np.zeros((N, N, N, N))

    tau348 += np.einsum("abkl,jiba->ijkl", t2, tau67, optimize=True)

    tau349 = np.zeros((N, N, N, N))

    tau349 += np.einsum("iljk->ijkl", tau348, optimize=True)

    tau348 = None

    tau393 = np.zeros((N, N, M, M))

    tau393 -= np.einsum("jiba->ijab", tau67, optimize=True)

    tau140 = np.zeros((N, N, M, M))

    tau140 += np.einsum("cbjk,kica->ijab", l2, tau14, optimize=True)

    tau141 += np.einsum("jiba->ijab", tau140, optimize=True)

    tau215 += np.einsum("jiba->ijab", tau140, optimize=True)

    tau305 = np.zeros((N, N, M, M))

    tau305 += np.einsum("jiba->ijab", tau140, optimize=True)

    tau140 = None

    tau198 = np.zeros((N, N, M, M))

    tau198 += np.einsum("bcjk,kica->ijab", l2, tau14, optimize=True)

    tau199 = np.zeros((N, N, M, M))

    tau199 -= np.einsum("bckj,ikca->ijab", t2, tau198, optimize=True)

    tau201 = np.zeros((N, N, M, M))

    tau201 -= np.einsum("jiab->ijab", tau199, optimize=True)

    tau199 = None

    tau331 += np.einsum("jiba->ijab", tau198, optimize=True)

    tau334 = np.zeros((N, N, M, M))

    tau334 += np.einsum("cbkj,kica->ijab", l2, tau14, optimize=True)

    tau335 += np.einsum("jiba->ijab", tau334, optimize=True)

    tau351 = np.zeros((N, N, N, N))

    tau351 += np.einsum("abkl,jiba->ijkl", t2, tau334, optimize=True)

    tau334 = None

    tau352 = np.zeros((N, N, N, N))

    tau352 += np.einsum("iljk->ijkl", tau351, optimize=True)

    tau351 = None

    tau15 = np.zeros((M, M, M, M))

    tau15 += np.einsum("abij,cdij->abcd", l2, t2, optimize=True)

    tau16 = np.zeros((N, N, M, M))

    tau16 += np.einsum("ijcd,acbd->ijab", tau14, tau15, optimize=True)

    tau26 += np.einsum("ijab->ijab", tau16, optimize=True)

    tau190 -= 3 * np.einsum("ijab->ijab", tau16, optimize=True)

    tau16 = None

    tau192 = np.zeros((N, N, M, M))

    tau192 += np.einsum("acbd,ijcd->ijab", tau15, tau9, optimize=True)

    tau201 -= np.einsum("ijab->ijab", tau192, optimize=True)

    tau192 = None

    tau364 = np.zeros((M, M, M, M))

    tau364 += np.einsum("afed,befc->abcd", tau15, tau15, optimize=True)

    tau17 = np.zeros((M, M))

    tau17 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum("cb,caij->ijab", tau17, t2, optimize=True)

    tau22 = np.zeros((N, N, M, M))

    tau22 += np.einsum("ijba->ijab", tau18, optimize=True)

    tau36 += np.einsum("ijab->ijab", tau18, optimize=True)

    tau72 += np.einsum("ijab->ijab", tau18, optimize=True)

    tau72 -= 2 * np.einsum("jiab->ijab", tau18, optimize=True)

    tau187 = np.zeros((N, N, M, M))

    tau187 -= 3 * np.einsum("ijab->ijab", tau18, optimize=True)

    tau354 = np.zeros((N, N, N, N))

    tau354 += np.einsum("ijba,lkab->ijkl", tau18, u[o, o, v, v], optimize=True)

    tau360 -= 3 * np.einsum("jlki->ijkl", tau354, optimize=True)

    tau360 -= 3 * np.einsum("ljik->ijkl", tau354, optimize=True)

    tau354 = None

    tau384 = np.zeros((N, N, M, M))

    tau384 += np.einsum("ijba->ijab", tau18, optimize=True)

    tau384 -= 2 * np.einsum("jiba->ijab", tau18, optimize=True)

    tau385 = np.zeros((N, N, M, M))

    tau385 += 2 * np.einsum("ijba->ijab", tau18, optimize=True)

    tau385 -= np.einsum("jiba->ijab", tau18, optimize=True)

    tau392 += np.einsum("jiba->ijab", tau18, optimize=True)

    tau396 = np.zeros((N, N))

    tau396 -= 2 * np.einsum("kiba,jkba->ij", tau18, u[o, o, v, v], optimize=True)

    tau18 = None

    tau65 = np.zeros((N, N, M, M))

    tau65 += np.einsum("ac,ijbc->ijab", tau17, u[o, o, v, v], optimize=True)

    tau66 = np.zeros((N, N, M, M))

    tau66 += np.einsum("caki,kjcb->ijab", t2, tau65, optimize=True)

    tau65 = None

    tau76 += np.einsum("ijba->ijab", tau66, optimize=True)

    tau66 = None

    tau19 = np.zeros((N, N, M, M))

    tau19 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum("cbkj,kica->ijab", t2, tau19, optimize=True)

    tau22 -= 2 * np.einsum("ijab->ijab", tau20, optimize=True)

    tau72 -= 2 * np.einsum("ijba->ijab", tau20, optimize=True)

    tau187 += 6 * np.einsum("ijba->ijab", tau20, optimize=True)

    tau360 += 6 * np.einsum("jlab,ikab->ijkl", tau20, u[o, o, v, v], optimize=True)

    tau384 -= 2 * np.einsum("ijab->ijab", tau20, optimize=True)

    tau385 -= 4 * np.einsum("ijab->ijab", tau20, optimize=True)

    tau20 = None

    tau21 = np.zeros((N, N, M, M))

    tau21 += np.einsum("caik,kjcb->ijab", t2, tau19, optimize=True)

    tau22 += np.einsum("ijab->ijab", tau21, optimize=True)

    tau22 += np.einsum("jiba->ijab", tau21, optimize=True)

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("ikca,kjbc->ijab", tau22, u[o, o, v, v], optimize=True)

    tau26 += np.einsum("ijba->ijab", tau23, optimize=True)

    tau23 = None

    tau36 += np.einsum("ijba->ijab", tau21, optimize=True)

    tau36 += np.einsum("jiab->ijab", tau21, optimize=True)

    tau72 += np.einsum("ijba->ijab", tau21, optimize=True)

    tau72 += np.einsum("jiab->ijab", tau21, optimize=True)

    tau187 -= 3 * np.einsum("ijba->ijab", tau21, optimize=True)

    tau187 -= 3 * np.einsum("jiab->ijab", tau21, optimize=True)

    tau353 = np.zeros((N, N, N, N))

    tau353 += np.einsum("ijba,lkab->ijkl", tau21, u[o, o, v, v], optimize=True)

    tau360 -= 3 * np.einsum("jlik->ijkl", tau353, optimize=True)

    tau360 -= 3 * np.einsum("ljki->ijkl", tau353, optimize=True)

    tau353 = None

    tau381 = np.zeros((N, N, N, N))

    tau381 -= 3 * np.einsum("abij,klab->ijkl", l2, tau21, optimize=True)

    tau384 += np.einsum("ijab->ijab", tau21, optimize=True)

    tau384 += np.einsum("jiba->ijab", tau21, optimize=True)

    tau385 += 2 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau385 += 2 * np.einsum("jiba->ijab", tau21, optimize=True)

    tau21 = None

    tau129 = np.zeros((N, N, M, M))

    tau129 -= 3 * np.einsum("ijab->ijab", tau19, optimize=True)

    tau133 = np.zeros((N, N, M, M))

    tau133 += np.einsum("ijab->ijab", tau19, optimize=True)

    tau387 = np.zeros((N, N, M, M))

    tau387 += np.einsum("ijab->ijab", tau19, optimize=True)

    tau24 = np.zeros((N, N, M, M))

    tau24 += np.einsum("acik,cbkj->ijab", l2, t2, optimize=True)

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum("kjbc,kiac->ijab", tau24, tau9, optimize=True)

    tau26 -= np.einsum("jiba->ijab", tau25, optimize=True)

    tau25 = None

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("bcjk,kiac->ijab", l2, tau26, optimize=True)

    tau26 = None

    tau64 = np.zeros((N, N, M, M))

    tau64 += 3 * np.einsum("jiba->ijab", tau27, optimize=True)

    tau27 = None

    tau74 = np.zeros((N, N, M, M))

    tau74 += np.einsum("kiac,kjbc->ijab", tau14, tau24, optimize=True)

    tau76 += np.einsum("jiba->ijab", tau74, optimize=True)

    tau74 = None

    tau91 = np.zeros((N, N, M, M))

    tau91 -= 3 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau96 = np.zeros((N, N, M, M))

    tau96 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau100 = np.zeros((N, N, M, M))

    tau100 -= 3 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau121 = np.zeros((M, M, M, M))

    tau121 += np.einsum("ijbc,jiad->abcd", tau24, tau24, optimize=True)

    tau122 = np.zeros((N, N, M, M))

    tau122 += np.einsum("bacd,ijdc->ijab", tau121, u[o, o, v, v], optimize=True)

    tau121 = None

    tau149 += 3 * np.einsum("ijab->ijab", tau122, optimize=True)

    tau122 = None

    tau224 = np.zeros((N, N, M, M))

    tau224 += np.einsum("ikcb,kjac->ijab", tau24, tau24, optimize=True)

    tau225 = np.zeros((N, N, M, M))

    tau225 += np.einsum("ikac,jkcb->ijab", tau224, u[o, o, v, v], optimize=True)

    tau224 = None

    tau226 = np.zeros((N, N, M, M))

    tau226 += np.einsum("ijab->ijab", tau225, optimize=True)

    tau225 = None

    tau236 = np.zeros((N, N, M, M))

    tau236 += np.einsum("ikcb,kjac->ijab", tau19, tau24, optimize=True)

    tau237 = np.zeros((N, N, M, M))

    tau237 += np.einsum("ikac,jkcb->ijab", tau236, u[o, o, v, v], optimize=True)

    tau236 = None

    tau238 += np.einsum("ijab->ijab", tau237, optimize=True)

    tau237 = None

    r2 += 4 * np.einsum("jiab->abij", tau238, optimize=True)

    r2 -= 2 * np.einsum("jiba->abij", tau238, optimize=True)

    tau238 = None

    tau283 = np.zeros((N, N, M, M))

    tau283 += np.einsum("klab,ilkj->ijab", tau24, u[o, o, o, o], optimize=True)

    tau285 = np.zeros((N, N, M, M))

    tau285 += np.einsum("ijab->ijab", tau283, optimize=True)

    tau283 = None

    tau291 = np.zeros((M, M, M, M))

    tau291 += np.einsum("ijab,jcdi->abcd", tau24, u[o, v, v, o], optimize=True)

    tau294 = np.zeros((M, M, M, M))

    tau294 -= np.einsum("abcd->abcd", tau291, optimize=True)

    tau291 = None

    tau292 = np.zeros((N, N, M, M))

    tau292 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau360 += 3 * np.einsum("ilab,jkab->ijkl", tau24, tau9, optimize=True)

    tau28 = np.zeros((N, N, M, M))

    tau28 += np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau29 = np.zeros((M, M, M, M))

    tau29 += np.einsum("ijad,jibc->abcd", tau24, tau28, optimize=True)

    tau32 = np.zeros((M, M, M, M))

    tau32 += 3 * np.einsum("abcd->abcd", tau29, optimize=True)

    tau29 = None

    tau155 = np.zeros((N, N, M, M))

    tau155 += np.einsum("kiac,kjbc->ijab", tau14, tau28, optimize=True)

    tau156 += np.einsum("jiba->ijab", tau155, optimize=True)

    tau155 = None

    tau193 = np.zeros((N, N, M, M))

    tau193 += np.einsum("kjbc,kiac->ijab", tau28, tau9, optimize=True)

    tau201 -= np.einsum("jiba->ijab", tau193, optimize=True)

    tau193 = None

    tau256 = np.zeros((N, N, M, M))

    tau256 += np.einsum("ikcb,kjac->ijab", tau28, tau28, optimize=True)

    tau257 = np.zeros((N, N, M, M))

    tau257 += np.einsum("ijab->ijab", tau256, optimize=True)

    tau256 = None

    tau288 = np.zeros((N, N, M, M))

    tau288 += np.einsum("klab,ilkj->ijab", tau28, u[o, o, o, o], optimize=True)

    tau289 = np.zeros((N, N, M, M))

    tau289 += np.einsum("jiab->ijab", tau288, optimize=True)

    tau288 = None

    tau364 += np.einsum("ijad,jibc->abcd", tau28, tau28, optimize=True)

    r2 += 4 * np.einsum("bacd,jicd->abij", tau364, u[o, o, v, v], optimize=True) / 3

    tau364 = None

    tau377 = np.zeros((N, N, M, M))

    tau377 += np.einsum("ikcb,kjac->ijab", tau19, tau28, optimize=True)

    tau378 = np.zeros((N, N, M, M))

    tau378 += np.einsum("kjac,ikcb->ijab", tau24, tau28, optimize=True)

    tau378 += np.einsum("ikcb,kjac->ijab", tau24, tau28, optimize=True)

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    tau31 = np.zeros((M, M, M, M))

    tau31 += np.einsum("ijad,jibc->abcd", tau28, tau30, optimize=True)

    tau32 -= np.einsum("abcd->abcd", tau31, optimize=True)

    tau31 = None

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("abcd,ijdc->ijab", tau32, u[o, o, v, v], optimize=True)

    tau32 = None

    tau64 += np.einsum("jiab->ijab", tau33, optimize=True)

    tau33 = None

    tau71 = np.zeros((N, N, M, M))

    tau71 += np.einsum("cbjk,kica->ijab", t2, tau30, optimize=True)

    tau72 += np.einsum("jiab->ijab", tau71, optimize=True)

    tau152 = np.zeros((N, N, M, M))

    tau152 += np.einsum("jiab->ijab", tau71, optimize=True)

    tau262 = np.zeros((N, N, N, N))

    tau262 += np.einsum("ijba,lkab->ijkl", tau71, u[o, o, v, v], optimize=True)

    tau263 = np.zeros((N, N, M, M))

    tau263 += np.einsum("ablk,klji->ijab", l2, tau262, optimize=True)

    tau262 = None

    r2 += 2 * np.einsum("ijab->abij", tau263, optimize=True) / 3

    r2 += 4 * np.einsum("jiab->abij", tau263, optimize=True) / 3

    tau263 = None

    tau379 = np.zeros((N, N, M, M))

    tau379 += 2 * np.einsum("jiab->ijab", tau71, optimize=True)

    tau379 += np.einsum("jiba->ijab", tau71, optimize=True)

    tau384 += np.einsum("jiba->ijab", tau71, optimize=True)

    tau71 = None

    tau133 += np.einsum("ijab->ijab", tau30, optimize=True)

    tau134 = np.zeros((N, N, M, M))

    tau134 += np.einsum("ikca,kjbc->ijab", tau133, tau24, optimize=True)

    tau137 = np.zeros((N, N, M, M))

    tau137 -= 3 * np.einsum("ijba->ijab", tau134, optimize=True)

    tau134 = None

    tau151 = np.zeros((N, N, M, M))

    tau151 += np.einsum("kjbc,kiac->ijab", tau30, tau9, optimize=True)

    tau156 -= np.einsum("jiba->ijab", tau151, optimize=True)

    tau151 = None

    tau200 = np.zeros((N, N, M, M))

    tau200 += np.einsum("kiac,kjbc->ijab", tau14, tau30, optimize=True)

    tau201 += np.einsum("jiba->ijab", tau200, optimize=True)

    tau200 = None

    tau203 = np.zeros((M, M, M, M))

    tau203 += np.einsum("ijad,jibc->abcd", tau24, tau30, optimize=True)

    tau205 = np.zeros((M, M, M, M))

    tau205 += 3 * np.einsum("abcd->abcd", tau203, optimize=True)

    tau203 = None

    tau208 = np.zeros((M, M, M, M))

    tau208 += np.einsum("ijac,jibd->abcd", tau30, tau30, optimize=True)

    tau209 = np.zeros((N, N, M, M))

    tau209 += np.einsum("abcd,ijcd->ijab", tau208, u[o, o, v, v], optimize=True)

    tau208 = None

    tau222 = np.zeros((N, N, M, M))

    tau222 += np.einsum("ijab->ijab", tau209, optimize=True)

    tau209 = None

    tau242 = np.zeros((M, M, M, M))

    tau242 += np.einsum("ijcd,ijab->abcd", tau30, tau9, optimize=True)

    tau245 = np.zeros((M, M, M, M))

    tau245 -= np.einsum("cdab->abcd", tau242, optimize=True)

    tau242 = None

    tau248 = np.zeros((N, N, M, M))

    tau248 += np.einsum("ikcb,kjac->ijab", tau28, tau30, optimize=True)

    tau249 = np.zeros((N, N, M, M))

    tau249 += np.einsum("ijab->ijab", tau248, optimize=True)

    tau248 = None

    tau267 = np.zeros((N, N, M, M))

    tau267 += np.einsum("kjac,ikcb->ijab", tau28, tau30, optimize=True)

    tau268 = np.zeros((N, N, M, M))

    tau268 += np.einsum("ijab->ijab", tau267, optimize=True)

    tau267 = None

    tau271 = np.zeros((N, N, M, M))

    tau271 += np.einsum("ikcb,kjac->ijab", tau30, tau30, optimize=True)

    tau272 = np.zeros((N, N, M, M))

    tau272 += np.einsum("ijab->ijab", tau271, optimize=True)

    tau271 = None

    tau292 += np.einsum("ijab->ijab", tau30, optimize=True)

    tau293 = np.zeros((M, M, M, M))

    tau293 += np.einsum("ijcd,ijab->abcd", tau13, tau292, optimize=True)

    tau294 += np.einsum("acbd->abcd", tau293, optimize=True)

    tau293 = None

    tau301 = np.zeros((M, M, M, M))

    tau301 += np.einsum("abcd->abcd", tau294, optimize=True)

    tau301 += np.einsum("dcba->abcd", tau294, optimize=True)

    tau294 = None

    tau299 = np.zeros((M, M, M, M))

    tau299 += np.einsum("jicd,iajb->abcd", tau292, u[o, v, o, v], optimize=True)

    tau300 = np.zeros((M, M, M, M))

    tau300 -= np.einsum("cdab->abcd", tau299, optimize=True)

    tau299 = None

    tau337 = np.zeros((N, N, M, M))

    tau337 += np.einsum("klab,jlki->ijab", tau30, u[o, o, o, o], optimize=True)

    tau343 = np.zeros((N, N, N, N))

    tau343 += np.einsum("ijab,kabl->ijkl", tau30, u[o, v, v, o], optimize=True)

    tau344 = np.zeros((N, N, N, N))

    tau344 -= np.einsum("ijkl->ijkl", tau343, optimize=True)

    tau343 = None

    tau362 = np.zeros((N, N, M, M))

    tau362 -= 3 * np.einsum("ikcb,kjac->ijab", tau24, tau30, optimize=True)

    tau363 = np.zeros((N, N, M, M))

    tau363 -= 3 * np.einsum("ikcb,kjac->ijab", tau19, tau30, optimize=True)

    tau363 -= 3 * np.einsum("kjac,ikcb->ijab", tau24, tau30, optimize=True)

    tau34 = np.zeros((N, N, M, M))

    tau34 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau34 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau37 = np.zeros((M, M))

    tau37 += np.einsum("ijca,ijbc->ab", tau34, tau36, optimize=True)

    tau36 = None

    tau62 = np.zeros((M, M))

    tau62 -= np.einsum("ba->ab", tau37, optimize=True)

    tau37 = None

    tau177 = np.zeros((N, N, M, M))

    tau177 += np.einsum("bcjk,kica->ijab", t2, tau34, optimize=True)

    tau180 = np.zeros((N, N, M, M))

    tau180 += 3 * np.einsum("jiba->ijab", tau177, optimize=True)

    tau373 = np.zeros((N, N, N, N))

    tau373 += np.einsum("jiba,klab->ijkl", tau177, tau28, optimize=True)

    tau374 = np.zeros((N, N, N, N))

    tau374 += np.einsum("klij->ijkl", tau373, optimize=True)

    tau373 = None

    tau396 -= np.einsum("ikba,kjab->ij", tau22, tau34, optimize=True)

    tau34 = None

    tau22 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum("acik,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau40 = np.zeros((N, N, M, M))

    tau40 += 4 * np.einsum("ijab->ijab", tau39, optimize=True)

    tau39 = None

    tau40 += 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau40 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau41 = np.zeros((N, N, M, M))

    tau41 += np.einsum("bcjk,kica->ijab", l2, tau40, optimize=True)

    tau40 = None

    tau42 += np.einsum("jiba->ijab", tau41, optimize=True)

    tau41 = None

    tau43 = np.zeros((M, M))

    tau43 += np.einsum("cbij,ijca->ab", t2, tau42, optimize=True)

    tau42 = None

    tau62 += np.einsum("ba->ab", tau43, optimize=True)

    tau43 = None

    tau44 = np.zeros((N, N, M, M))

    tau44 += np.einsum("acki,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau49 += np.einsum("ijab->ijab", tau44, optimize=True)

    tau44 = None

    tau46 = np.zeros((N, N, M, M))

    tau46 += np.einsum("acki,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau47 = np.zeros((N, N, M, M))

    tau47 += np.einsum("ijab->ijab", tau46, optimize=True)

    tau297 = np.zeros((N, N, M, M))

    tau297 += np.einsum("ijab->ijab", tau46, optimize=True)

    tau46 = None

    tau47 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau48 = np.zeros((N, N, M, M))

    tau48 += np.einsum("bcjk,kica->ijab", l2, tau47, optimize=True)

    tau49 -= np.einsum("jiba->ijab", tau48, optimize=True)

    tau48 = None

    tau50 = np.zeros((M, M))

    tau50 += np.einsum("cbji,ijca->ab", t2, tau49, optimize=True)

    tau49 = None

    tau62 -= np.einsum("ba->ab", tau50, optimize=True)

    tau50 = None

    tau75 = np.zeros((N, N, M, M))

    tau75 += np.einsum("bc,ijac->ijab", tau0, tau47, optimize=True)

    tau0 = None

    tau76 += np.einsum("ijba->ijab", tau75, optimize=True)

    tau75 = None

    tau114 = np.zeros((N, N, M, M))

    tau114 += np.einsum("jk,ikab->ijab", tau4, tau47, optimize=True)

    tau115 = np.zeros((N, N, M, M))

    tau115 += np.einsum("jiab->ijab", tau114, optimize=True)

    tau114 = None

    tau51 = np.zeros((N, N, M, M))

    tau51 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau51 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau53 = np.zeros((M, M))

    tau53 += np.einsum("ijca,ijbc->ab", tau51, tau52, optimize=True)

    tau52 = None

    tau62 -= np.einsum("ba->ab", tau53, optimize=True)

    tau53 = None

    tau296 = np.zeros((N, N, M, M))

    tau296 += np.einsum("bcjk,kiac->ijab", t2, tau51, optimize=True)

    tau51 = None

    tau297 -= np.einsum("jiba->ijab", tau296, optimize=True)

    tau298 = np.zeros((M, M, M, M))

    tau298 += np.einsum("ijcd,ijab->abcd", tau24, tau297, optimize=True)

    tau300 += np.einsum("bdac->abcd", tau298, optimize=True)

    tau298 = None

    tau350 = np.zeros((N, N, N, N))

    tau350 += np.einsum("ijab,klab->ijkl", tau297, tau30, optimize=True)

    tau352 += np.einsum("klij->ijkl", tau350, optimize=True)

    tau350 = None

    tau360 += np.einsum("kjli->ijkl", tau352, optimize=True)

    tau360 += 2 * np.einsum("klji->ijkl", tau352, optimize=True)

    tau352 = None

    tau356 = np.zeros((N, N, M, M))

    tau356 -= 3 * np.einsum("jiba->ijab", tau296, optimize=True)

    tau296 = None

    tau54 = np.zeros((N, N, N, N))

    tau54 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau55 = np.zeros((N, N, M, M))

    tau55 += np.einsum("ijlk,lkab->ijab", tau54, u[o, o, v, v], optimize=True)

    tau56 += np.einsum("ijab->ijab", tau55, optimize=True)

    tau57 = np.zeros((M, M))

    tau57 += np.einsum("bcij,ijac->ab", t2, tau56, optimize=True)

    tau62 += np.einsum("ba->ab", tau57, optimize=True)

    tau57 = None

    tau305 += 2 * np.einsum("ijab->ijab", tau56, optimize=True)

    tau305 += np.einsum("ijba->ijab", tau56, optimize=True)

    tau315 += np.einsum("jiab->ijab", tau56, optimize=True)

    tau315 += 2 * np.einsum("jiba->ijab", tau56, optimize=True)

    tau56 = None

    tau108 += np.einsum("jiab->ijab", tau55, optimize=True)

    tau109 = np.zeros((N, N, M, M))

    tau109 += np.einsum("bcjk,ikca->ijab", t2, tau108, optimize=True)

    tau108 = None

    tau115 += np.einsum("ijba->ijab", tau109, optimize=True)

    tau109 = None

    tau141 += np.einsum("ijba->ijab", tau55, optimize=True)

    tau162 += np.einsum("jiab->ijab", tau55, optimize=True)

    tau163 = np.zeros((N, N, M, M))

    tau163 += np.einsum("bckj,ikca->ijab", t2, tau162, optimize=True)

    tau162 = None

    tau167 = np.zeros((N, N, M, M))

    tau167 += np.einsum("ijba->ijab", tau163, optimize=True)

    tau163 = None

    tau215 += np.einsum("ijba->ijab", tau55, optimize=True)

    tau216 = np.zeros((N, N, M, M))

    tau216 += np.einsum("bckj,ikca->ijab", t2, tau215, optimize=True)

    tau215 = None

    tau220 = np.zeros((N, N, M, M))

    tau220 += np.einsum("ijba->ijab", tau216, optimize=True)

    tau216 = None

    tau331 += np.einsum("ijab->ijab", tau55, optimize=True)

    tau335 += np.einsum("jiba->ijab", tau55, optimize=True)

    tau55 = None

    tau337 += np.einsum("bcki,jkca->ijab", t2, tau335, optimize=True)

    tau335 = None

    tau70 = np.zeros((N, N, M, M))

    tau70 += np.einsum("ablk,lkij->ijab", t2, tau54, optimize=True)

    tau72 += np.einsum("ijab->ijab", tau70, optimize=True)

    tau73 = np.zeros((N, N, M, M))

    tau73 += np.einsum("ikac,kjcb->ijab", tau72, u[o, o, v, v], optimize=True)

    tau72 = None

    tau76 += np.einsum("ijba->ijab", tau73, optimize=True)

    tau73 = None

    tau77 = np.zeros((N, N, M, M))

    tau77 += np.einsum("bcjk,kiac->ijab", l2, tau76, optimize=True)

    tau76 = None

    r2 += 4 * np.einsum("jiba->abij", tau77, optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", tau77, optimize=True)

    r2 -= 2 * np.einsum("ijba->abij", tau77, optimize=True)

    r2 += 4 * np.einsum("ijab->abij", tau77, optimize=True)

    tau77 = None

    tau152 += np.einsum("ijab->ijab", tau70, optimize=True)

    tau70 = None

    tau153 = np.zeros((N, N, M, M))

    tau153 += np.einsum("ikac,kjbc->ijab", tau152, u[o, o, v, v], optimize=True)

    tau152 = None

    tau156 += np.einsum("ijba->ijab", tau153, optimize=True)

    tau153 = None

    tau166 = np.zeros((N, N, M, M))

    tau166 += np.einsum("kijl,klab->ijab", tau54, tau9, optimize=True)

    tau167 -= np.einsum("ijab->ijab", tau166, optimize=True)

    tau166 = None

    tau170 = np.zeros((N, N, N, N))

    tau170 += np.einsum("imkn,njml->ijkl", tau54, tau54, optimize=True)

    tau171 = np.zeros((N, N, N, N))

    tau171 += np.einsum("ijkl->ijkl", tau170, optimize=True)

    tau170 = None

    tau194 = np.zeros((N, N, M, M))

    tau194 += np.einsum("abkl,lkij->ijab", t2, tau54, optimize=True)

    tau196 = np.zeros((N, N, M, M))

    tau196 += np.einsum("ijab->ijab", tau194, optimize=True)

    tau194 = None

    tau219 = np.zeros((N, N, M, M))

    tau219 += np.einsum("iklj,klab->ijab", tau54, tau9, optimize=True)

    tau220 -= np.einsum("ijab->ijab", tau219, optimize=True)

    tau219 = None

    tau228 = np.zeros((N, N, M, M))

    tau228 += np.einsum("klab,likj->ijab", tau30, tau54, optimize=True)

    tau229 = np.zeros((N, N, M, M))

    tau229 += np.einsum("ijab->ijab", tau228, optimize=True)

    tau228 = None

    tau231 = np.zeros((N, N, M, M))

    tau231 += np.einsum("klab,iljk->ijab", tau30, tau54, optimize=True)

    tau233 = np.zeros((N, N, M, M))

    tau233 += np.einsum("ijab->ijab", tau231, optimize=True)

    tau231 = None

    tau251 = np.zeros((N, N, M, M))

    tau251 += np.einsum("lkba,ijkl->ijab", tau198, tau54, optimize=True)

    tau253 = np.zeros((N, N, M, M))

    tau253 += np.einsum("ijab->ijab", tau251, optimize=True)

    tau251 = None

    tau252 = np.zeros((N, N, M, M))

    tau252 += np.einsum("ijlk,lkba->ijab", tau54, tau67, optimize=True)

    tau253 += np.einsum("ijab->ijab", tau252, optimize=True)

    tau252 = None

    tau259 = np.zeros((N, N, M, M))

    tau259 += np.einsum("ijkl,lkba->ijab", tau54, tau67, optimize=True)

    tau67 = None

    tau261 = np.zeros((N, N, M, M))

    tau261 += np.einsum("ijab->ijab", tau259, optimize=True)

    tau259 = None

    tau260 = np.zeros((N, N, M, M))

    tau260 += np.einsum("lkba,ijlk->ijab", tau198, tau54, optimize=True)

    tau198 = None

    tau261 += np.einsum("ijab->ijab", tau260, optimize=True)

    tau260 = None

    tau307 = np.zeros((N, N, N, N))

    tau307 += np.einsum("ijkl->ijkl", tau54, optimize=True)

    tau307 += 2 * np.einsum("ijlk->ijkl", tau54, optimize=True)

    tau312 = np.zeros((N, N, M, M))

    tau312 -= np.einsum("iklj,klba->ijab", tau307, tau9, optimize=True)

    tau320 -= np.einsum("kjil,klba->ijab", tau307, tau9, optimize=True)

    tau376 = np.zeros((N, N, N, N))

    tau376 += np.einsum("injm,mkln->ijkl", tau307, u[o, o, o, o], optimize=True)

    tau311 = np.zeros((N, N, M, M))

    tau311 += np.einsum("klab,ikjl->ijab", tau14, tau54, optimize=True)

    tau312 -= 3 * np.einsum("ijba->ijab", tau311, optimize=True)

    tau325 = np.zeros((N, N, M, M))

    tau325 += np.einsum("ijba->ijab", tau311, optimize=True)

    tau311 = None

    tau319 = np.zeros((N, N, M, M))

    tau319 += np.einsum("klab,kilj->ijab", tau14, tau54, optimize=True)

    tau320 -= 3 * np.einsum("jiba->ijab", tau319, optimize=True)

    tau329 += np.einsum("ijba->ijab", tau319, optimize=True)

    tau319 = None

    tau333 = np.zeros((N, N, M, M))

    tau333 -= np.einsum("ikjl,klba->ijab", tau54, tau9, optimize=True)

    tau337 -= np.einsum("kjli,klba->ijab", tau54, tau9, optimize=True)

    tau339 = np.zeros((N, N, N, N))

    tau339 += np.einsum("mijn,knml->ijkl", tau54, u[o, o, o, o], optimize=True)

    tau341 = np.zeros((N, N, N, N))

    tau341 += np.einsum("ijkl->ijkl", tau339, optimize=True)

    tau339 = None

    tau342 = np.zeros((N, N, N, N))

    tau342 += np.einsum("minj,knml->ijkl", tau54, u[o, o, o, o], optimize=True)

    tau344 += np.einsum("ijkl->ijkl", tau342, optimize=True)

    tau342 = None

    tau360 += np.einsum("kjil->ijkl", tau344, optimize=True)

    tau360 += 2 * np.einsum("klij->ijkl", tau344, optimize=True)

    tau344 = None

    tau360 += np.einsum("imnj,knml->ijkl", tau54, u[o, o, o, o], optimize=True)

    tau362 += np.einsum("klab,iljk->ijab", tau28, tau54, optimize=True)

    tau363 += np.einsum("klab,likj->ijab", tau28, tau54, optimize=True)

    tau377 += np.einsum("klab,likj->ijab", tau24, tau54, optimize=True)

    tau378 += np.einsum("klab,iljk->ijab", tau24, tau54, optimize=True)

    tau380 = np.zeros((N, N, N, N))

    tau380 -= 3 * np.einsum("ml,ijkm->ijkl", tau4, tau54, optimize=True)

    tau381 += np.einsum("imnl,njmk->ijkl", tau54, tau54, optimize=True)

    tau381 -= 3 * np.einsum("mk,ijml->ijkl", tau4, tau54, optimize=True)

    tau381 += np.einsum("imln,njkm->ijkl", tau307, tau54, optimize=True)

    tau307 = None

    r2 -= 2 * np.einsum("klba,ijlk->abij", tau382, tau54, optimize=True)

    tau382 = None

    r2 += 2 * np.einsum("lkba,ijkl->abij", tau10, tau54, optimize=True)

    tau10 = None

    tau58 = np.zeros((M, M, M, M))

    tau58 += 2 * np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau58 -= np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau59 = np.zeros((M, M))

    tau59 += np.einsum("cd,cabd->ab", tau17, tau58, optimize=True)

    tau58 = None

    tau62 -= np.einsum("ab->ab", tau59, optimize=True)

    tau59 = None

    tau60 = np.zeros((N, N, M, M))

    tau60 -= np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau60 += 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau61 = np.zeros((M, M))

    tau61 += np.einsum("ji,ijab->ab", tau4, tau60, optimize=True)

    tau62 += np.einsum("ab->ab", tau61, optimize=True)

    tau61 = None

    tau63 = np.zeros((N, N, M, M))

    tau63 += np.einsum("ca,bcij->ijab", tau62, l2, optimize=True)

    tau62 = None

    tau64 += 3 * np.einsum("ijba->ijab", tau63, optimize=True)

    tau63 = None

    r2 -= 2 * np.einsum("ijab->abij", tau64, optimize=True) / 3

    r2 -= 2 * np.einsum("jiba->abij", tau64, optimize=True) / 3

    tau64 = None

    tau388 = np.zeros((N, N))

    tau388 += np.einsum("ab,ijab->ij", tau17, tau60, optimize=True)

    tau17 = None

    tau60 = None

    tau391 = np.zeros((N, N))

    tau391 += np.einsum("ij->ij", tau388, optimize=True)

    tau396 += np.einsum("ji->ij", tau388, optimize=True)

    tau388 = None

    tau78 = np.zeros((N, N, M, M))

    tau78 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau79 = np.zeros((N, N, N, N))

    tau79 += np.einsum("ilab,jkba->ijkl", tau24, tau78, optimize=True)

    tau80 = np.zeros((N, N, M, M))

    tau80 += np.einsum("ijkl,lkba->ijab", tau79, u[o, o, v, v], optimize=True)

    tau79 = None

    tau120 = np.zeros((N, N, M, M))

    tau120 += 3 * np.einsum("ijab->ijab", tau80, optimize=True)

    tau80 = None

    tau113 = np.zeros((N, N, M, M))

    tau113 += np.einsum("ikca,jkcb->ijab", tau14, tau78, optimize=True)

    tau115 += np.einsum("jiba->ijab", tau113, optimize=True)

    tau113 = None

    tau124 = np.zeros((N, N, M, M))

    tau124 -= 3 * np.einsum("ijab->ijab", tau78, optimize=True)

    tau329 -= np.einsum("ikcb,jkca->ijab", tau78, tau9, optimize=True)

    tau81 = np.zeros((N, N, M, M))

    tau81 += 2 * np.einsum("abij->ijab", l2, optimize=True)

    tau81 += np.einsum("baij->ijab", l2, optimize=True)

    tau82 = np.zeros((N, N, M, M))

    tau82 += np.einsum("bckj,ikca->ijab", t2, tau81, optimize=True)

    tau312 += np.einsum("jkca,ikcb->ijab", tau14, tau82, optimize=True)

    tau93 = np.zeros((N, N, M, M))

    tau93 += np.einsum("cbjk,ikca->ijab", t2, tau81, optimize=True)

    tau94 = np.zeros((N, N, M, M))

    tau94 += np.einsum("kjbc,ikca->ijab", tau28, tau93, optimize=True)

    tau105 = np.zeros((N, N, M, M))

    tau105 += np.einsum("ijba->ijab", tau94, optimize=True)

    tau94 = None

    tau360 += np.einsum("lkab,ijab->ijkl", tau47, tau93, optimize=True)

    tau123 = np.zeros((N, N, M, M))

    tau123 += np.einsum("bckj,kica->ijab", t2, tau81, optimize=True)

    tau124 += np.einsum("ijab->ijab", tau123, optimize=True)

    tau123 = None

    tau125 = np.zeros((N, N, M, M))

    tau125 += np.einsum("ijcd,adcb->ijab", tau124, tau15, optimize=True)

    tau137 += np.einsum("ijab->ijab", tau125, optimize=True)

    tau363 += np.einsum("ijab->ijab", tau125, optimize=True)

    tau125 = None

    tau320 -= np.einsum("jkcb,ikca->ijab", tau124, tau9, optimize=True)

    tau380 += np.einsum("jkab,ilba->ijkl", tau124, tau28, optimize=True)

    tau124 = None

    tau126 = np.zeros((N, N, M, M))

    tau126 += np.einsum("bckj,kiac->ijab", t2, tau81, optimize=True)

    tau320 += np.einsum("jkcb,ikca->ijab", tau126, tau14, optimize=True)

    tau128 = np.zeros((N, N, M, M))

    tau128 += np.einsum("cbjk,kica->ijab", t2, tau81, optimize=True)

    tau129 += np.einsum("ijab->ijab", tau128, optimize=True)

    tau128 = None

    tau130 = np.zeros((N, N, M, M))

    tau130 += np.einsum("ikca,kjbc->ijab", tau129, tau30, optimize=True)

    tau129 = None

    tau137 += np.einsum("ijba->ijab", tau130, optimize=True)

    tau130 = None

    tau131 = np.zeros((N, N, M, M))

    tau131 += np.einsum("cbjk,kiac->ijab", t2, tau81, optimize=True)

    tau132 = np.zeros((N, N, M, M))

    tau132 += np.einsum("ikca,kjbc->ijab", tau131, tau28, optimize=True)

    tau137 += np.einsum("ijba->ijab", tau132, optimize=True)

    tau132 = None

    tau375 = np.zeros((N, N, M, M))

    tau375 += np.einsum("kjcb,ikca->ijab", tau13, tau81, optimize=True)

    tau81 = None

    tau13 = None

    tau376 += np.einsum("abjl,ikab->ijkl", t2, tau375, optimize=True)

    tau375 = None

    tau83 = np.zeros((M, M, M, M))

    tau83 += np.einsum("abji,cdij->abcd", l2, t2, optimize=True)

    tau84 = np.zeros((N, N, M, M))

    tau84 += np.einsum("ijcd,adcb->ijab", tau82, tau83, optimize=True)

    tau82 = None

    tau105 += np.einsum("ijab->ijab", tau84, optimize=True)

    tau84 = None

    tau127 = np.zeros((N, N, M, M))

    tau127 += np.einsum("ijcd,adcb->ijab", tau126, tau83, optimize=True)

    tau126 = None

    tau137 += np.einsum("ijab->ijab", tau127, optimize=True)

    tau127 = None

    tau150 = np.zeros((N, N, M, M))

    tau150 += np.einsum("acbd,ijcd->ijab", tau83, tau9, optimize=True)

    tau156 -= np.einsum("ijab->ijab", tau150, optimize=True)

    tau150 = None

    tau157 = np.zeros((N, N, M, M))

    tau157 += np.einsum("bcjk,kiac->ijab", l2, tau156, optimize=True)

    tau156 = None

    r2 -= 2 * np.einsum("jiba->abij", tau157, optimize=True)

    r2 += 4 * np.einsum("jiab->abij", tau157, optimize=True) / 3

    r2 += 4 * np.einsum("ijba->abij", tau157, optimize=True) / 3

    r2 -= 2 * np.einsum("ijab->abij", tau157, optimize=True)

    tau157 = None

    tau204 = np.zeros((M, M, M, M))

    tau204 += np.einsum("aefc,bfed->abcd", tau15, tau83, optimize=True)

    tau205 -= np.einsum("abcd->abcd", tau204, optimize=True)

    tau204 = None

    tau206 = np.zeros((N, N, M, M))

    tau206 += np.einsum("abcd,ijdc->ijab", tau205, u[o, o, v, v], optimize=True)

    tau205 = None

    tau207 = np.zeros((N, N, M, M))

    tau207 -= np.einsum("jiab->ijab", tau206, optimize=True)

    tau206 = None

    tau210 = np.zeros((M, M, M, M))

    tau210 += np.einsum("aefc,bfed->abcd", tau83, tau83, optimize=True)

    tau211 = np.zeros((N, N, M, M))

    tau211 += np.einsum("abdc,ijdc->ijab", tau210, u[o, o, v, v], optimize=True)

    tau210 = None

    tau222 += np.einsum("ijab->ijab", tau211, optimize=True)

    tau211 = None

    tau377 += np.einsum("ijdc,acdb->ijab", tau78, tau83, optimize=True)

    tau85 = np.zeros((N, N, M, M))

    tau85 += np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    tau88 = np.zeros((N, N, M, M))

    tau88 -= 3 * np.einsum("ijab->ijab", tau85, optimize=True)

    tau144 = np.zeros((N, N, M, M))

    tau144 += np.einsum("ikca,jkcb->ijab", tau14, tau85, optimize=True)

    tau146 = np.zeros((N, N, M, M))

    tau146 += np.einsum("jiba->ijab", tau144, optimize=True)

    tau144 = None

    tau280 = np.zeros((N, N, M, M))

    tau280 += np.einsum("klab,ilkj->ijab", tau85, u[o, o, o, o], optimize=True)

    tau281 = np.zeros((N, N, M, M))

    tau281 += np.einsum("jiab->ijab", tau280, optimize=True)

    tau280 = None

    tau310 = np.zeros((N, N, M, M))

    tau310 += np.einsum("ijab->ijab", tau85, optimize=True)

    tau325 -= np.einsum("ikcb,jkca->ijab", tau85, tau9, optimize=True)

    tau332 = np.zeros((N, N, M, M))

    tau332 -= 3 * np.einsum("ijab->ijab", tau85, optimize=True)

    tau361 = np.zeros((N, N, M, M))

    tau361 += 3 * np.einsum("ijab->ijab", tau85, optimize=True)

    tau378 += np.einsum("acdb,ijdc->ijab", tau83, tau85, optimize=True)

    tau85 = None

    r2 -= 2 * np.einsum("ikbc,kjac->abij", tau378, u[o, o, v, v], optimize=True)

    tau378 = None

    tau86 = np.zeros((N, N, M, M))

    tau86 += np.einsum("abij->ijab", l2, optimize=True)

    tau86 += 2 * np.einsum("baij->ijab", l2, optimize=True)

    tau87 = np.zeros((N, N, M, M))

    tau87 += np.einsum("bckj,ikca->ijab", t2, tau86, optimize=True)

    tau88 += np.einsum("ijab->ijab", tau87, optimize=True)

    tau87 = None

    tau89 = np.zeros((N, N, M, M))

    tau89 += np.einsum("adcb,ijcd->ijab", tau15, tau88, optimize=True)

    tau105 += np.einsum("ijab->ijab", tau89, optimize=True)

    tau89 = None

    tau312 -= np.einsum("ikcb,jkca->ijab", tau88, tau9, optimize=True)

    tau88 = None

    tau90 = np.zeros((N, N, M, M))

    tau90 += np.einsum("cbjk,ikca->ijab", t2, tau86, optimize=True)

    tau86 = None

    tau91 += np.einsum("ijab->ijab", tau90, optimize=True)

    tau92 = np.zeros((N, N, M, M))

    tau92 += np.einsum("kjbc,ikca->ijab", tau30, tau91, optimize=True)

    tau91 = None

    tau105 += np.einsum("ijba->ijab", tau92, optimize=True)

    tau92 = None

    tau376 += np.einsum("lkab,ijab->ijkl", tau47, tau90, optimize=True)

    tau90 = None

    tau95 = np.zeros((N, N, M, M))

    tau95 += np.einsum("caik,cbjk->ijab", l2, t2, optimize=True)

    tau96 += np.einsum("ijab->ijab", tau95, optimize=True)

    tau97 = np.zeros((N, N, M, M))

    tau97 += np.einsum("kjbc,ikca->ijab", tau24, tau96, optimize=True)

    tau105 -= 3 * np.einsum("ijba->ijab", tau97, optimize=True)

    tau97 = None

    tau360 -= 3 * np.einsum("lkab,ijab->ijkl", tau14, tau96, optimize=True)

    tau247 = np.zeros((N, N, M, M))

    tau247 += np.einsum("kjac,ikcb->ijab", tau28, tau95, optimize=True)

    tau249 += np.einsum("ijab->ijab", tau247, optimize=True)

    tau247 = None

    tau250 = np.zeros((N, N, M, M))

    tau250 += np.einsum("ikac,kjbc->ijab", tau249, u[o, o, v, v], optimize=True)

    tau249 = None

    tau253 += np.einsum("ijab->ijab", tau250, optimize=True)

    tau250 = None

    tau255 = np.zeros((N, N, M, M))

    tau255 += np.einsum("kjac,ikcb->ijab", tau30, tau95, optimize=True)

    tau257 += np.einsum("ijab->ijab", tau255, optimize=True)

    tau255 = None

    tau258 = np.zeros((N, N, M, M))

    tau258 += np.einsum("ikac,kjbc->ijab", tau257, u[o, o, v, v], optimize=True)

    tau257 = None

    tau261 += np.einsum("ijab->ijab", tau258, optimize=True)

    tau258 = None

    tau362 -= 3 * np.einsum("kjac,ikcb->ijab", tau24, tau95, optimize=True)

    tau368 = np.zeros((N, N, N, N))

    tau368 += np.einsum("jiba,klab->ijkl", tau177, tau95, optimize=True)

    tau177 = None

    tau369 = np.zeros((N, N, N, N))

    tau369 += np.einsum("klij->ijkl", tau368, optimize=True)

    tau368 = None

    tau380 -= 3 * np.einsum("jlba,ikab->ijkl", tau78, tau95, optimize=True)

    tau78 = None

    tau387 += np.einsum("ijab->ijab", tau95, optimize=True)

    tau391 -= np.einsum("kjab,iakb->ij", tau387, u[o, v, o, v], optimize=True)

    tau387 = None

    tau98 = np.zeros((N, N, M, M))

    tau98 += 2 * np.einsum("abij->ijab", l2, optimize=True)

    tau98 += np.einsum("abji->ijab", l2, optimize=True)

    tau99 = np.zeros((N, N, M, M))

    tau99 += np.einsum("cbjk,ikac->ijab", t2, tau98, optimize=True)

    tau100 += np.einsum("ijab->ijab", tau99, optimize=True)

    tau101 = np.zeros((N, N, M, M))

    tau101 += np.einsum("klab,ilkj->ijab", tau100, tau54, optimize=True)

    tau105 += np.einsum("ijab->ijab", tau101, optimize=True)

    tau362 += np.einsum("ijab->ijab", tau101, optimize=True)

    tau101 = None

    tau135 = np.zeros((N, N, M, M))

    tau135 += np.einsum("klab,lijk->ijab", tau100, tau54, optimize=True)

    tau137 += np.einsum("ijab->ijab", tau135, optimize=True)

    tau363 += np.einsum("ijab->ijab", tau135, optimize=True)

    tau135 = None

    tau176 = np.zeros((N, N, M, M))

    tau176 += np.einsum("kjbc,kiac->ijab", tau100, tau9, optimize=True)

    tau190 -= np.einsum("jiba->ijab", tau176, optimize=True)

    tau176 = None

    tau184 = np.zeros((N, N, M, M))

    tau184 += np.einsum("cbjk,kica->ijab", t2, tau99, optimize=True)

    tau99 = None

    tau187 += np.einsum("jiab->ijab", tau184, optimize=True)

    tau184 = None

    tau174 = np.zeros((M, M, M, M))

    tau174 += np.einsum("cdij,ijab->abcd", t2, tau98, optimize=True)

    tau175 = np.zeros((N, N, M, M))

    tau175 += np.einsum("acbd,ijcd->ijab", tau174, tau9, optimize=True)

    tau190 -= np.einsum("ijab->ijab", tau175, optimize=True)

    tau175 = None

    tau332 += np.einsum("bckj,ikac->ijab", t2, tau98, optimize=True)

    tau98 = None

    tau102 = np.zeros((N, N, M, M))

    tau102 += np.einsum("abij->ijab", l2, optimize=True)

    tau102 += 2 * np.einsum("abji->ijab", l2, optimize=True)

    tau103 = np.zeros((N, N, M, M))

    tau103 += np.einsum("cbjk,ikac->ijab", t2, tau102, optimize=True)

    tau104 = np.zeros((N, N, M, M))

    tau104 += np.einsum("klab,iljk->ijab", tau103, tau54, optimize=True)

    tau105 += np.einsum("ijab->ijab", tau104, optimize=True)

    tau104 = None

    tau106 = np.zeros((N, N, M, M))

    tau106 += np.einsum("ikac,kjcb->ijab", tau105, u[o, o, v, v], optimize=True)

    tau105 = None

    tau120 -= np.einsum("ijab->ijab", tau106, optimize=True)

    tau106 = None

    tau136 = np.zeros((N, N, M, M))

    tau136 += np.einsum("klab,likj->ijab", tau103, tau54, optimize=True)

    tau137 += np.einsum("ijab->ijab", tau136, optimize=True)

    tau136 = None

    tau138 = np.zeros((N, N, M, M))

    tau138 += np.einsum("ikac,kjcb->ijab", tau137, u[o, o, v, v], optimize=True)

    tau137 = None

    tau149 -= np.einsum("ijab->ijab", tau138, optimize=True)

    tau138 = None

    tau189 = np.zeros((N, N, M, M))

    tau189 += np.einsum("kjbc,kiac->ijab", tau103, tau14, optimize=True)

    tau190 += np.einsum("jiba->ijab", tau189, optimize=True)

    tau189 = None

    tau309 = np.zeros((N, N, M, M))

    tau309 += np.einsum("bckj,ikac->ijab", t2, tau102, optimize=True)

    tau102 = None

    tau110 = np.zeros((M, M, M, M))

    tau110 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau111 = np.zeros((M, M, M, M))

    tau111 += np.einsum("badc->abcd", tau110, optimize=True)

    tau239 = np.zeros((M, M, M, M))

    tau239 += np.einsum("abcd->abcd", tau110, optimize=True)

    tau110 = None

    tau111 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau112 = np.zeros((N, N, M, M))

    tau112 += np.einsum("cabd,ijcd->ijab", tau111, tau19, optimize=True)

    tau115 += np.einsum("ijab->ijab", tau112, optimize=True)

    tau112 = None

    tau116 = np.zeros((N, N, M, M))

    tau116 += np.einsum("bcjk,ikca->ijab", l2, tau115, optimize=True)

    tau115 = None

    tau120 += 3 * np.einsum("jiba->ijab", tau116, optimize=True)

    tau116 = None

    tau143 = np.zeros((N, N, M, M))

    tau143 += np.einsum("cabd,ijcd->ijab", tau111, tau24, optimize=True)

    tau146 += np.einsum("ijab->ijab", tau143, optimize=True)

    tau143 = None

    tau212 = np.zeros((N, N, M, M))

    tau212 += np.einsum("cabd,ijcd->ijab", tau111, tau28, optimize=True)

    tau220 += np.einsum("ijab->ijab", tau212, optimize=True)

    tau212 = None

    tau306 = np.zeros((N, N, M, M))

    tau306 += np.einsum("cadb,ijcd->ijab", tau111, tau96, optimize=True)

    tau96 = None

    tau312 -= 3 * np.einsum("ijba->ijab", tau306, optimize=True)

    tau325 += np.einsum("ijba->ijab", tau306, optimize=True)

    tau306 = None

    tau312 += np.einsum("cbad,ijcd->ijab", tau111, tau93, optimize=True)

    tau93 = None

    tau314 = np.zeros((N, N, M, M))

    tau314 += np.einsum("cadb,ijcd->ijab", tau111, tau133, optimize=True)

    tau320 -= 3 * np.einsum("jiba->ijab", tau314, optimize=True)

    tau329 += np.einsum("ijba->ijab", tau314, optimize=True)

    tau314 = None

    tau320 += np.einsum("cbad,jicd->ijab", tau111, tau131, optimize=True)

    tau131 = None

    tau333 += np.einsum("cbad,ijcd->ijab", tau111, tau95, optimize=True)

    tau337 += np.einsum("cbad,jicd->ijab", tau111, tau30, optimize=True)

    tau117 = np.zeros((N, N))

    tau117 += np.einsum("baik,bajk->ij", l2, t2, optimize=True)

    tau119 = np.zeros((N, N, M, M))

    tau119 += np.einsum("jk,ikab->ijab", tau117, tau118, optimize=True)

    tau118 = None

    tau120 += 3 * np.einsum("jiab->ijab", tau119, optimize=True)

    tau119 = None

    r2 += 4 * np.einsum("ijab->abij", tau120, optimize=True) / 3

    r2 -= 2 * np.einsum("ijba->abij", tau120, optimize=True) / 3

    tau120 = None

    tau139 = np.zeros((N, N, M, M))

    tau139 += np.einsum("ik,jkab->ijab", tau117, u[o, o, v, v], optimize=True)

    tau141 -= 2 * np.einsum("ijab->ijab", tau139, optimize=True)

    tau141 += np.einsum("ijba->ijab", tau139, optimize=True)

    tau142 = np.zeros((N, N, M, M))

    tau142 += np.einsum("bcjk,ikca->ijab", t2, tau141, optimize=True)

    tau141 = None

    tau146 += np.einsum("ijba->ijab", tau142, optimize=True)

    tau142 = None

    tau305 -= 3 * np.einsum("ijba->ijab", tau139, optimize=True)

    tau321 += np.einsum("ijba->ijab", tau139, optimize=True)

    tau139 = None

    tau325 += np.einsum("bckj,ikca->ijab", t2, tau321, optimize=True)

    tau321 = None

    tau145 = np.zeros((N, N, M, M))

    tau145 += np.einsum("jk,ikab->ijab", tau117, tau47, optimize=True)

    tau47 = None

    tau146 += np.einsum("jiab->ijab", tau145, optimize=True)

    tau145 = None

    tau147 = np.zeros((N, N, M, M))

    tau147 += np.einsum("bcjk,ikca->ijab", l2, tau146, optimize=True)

    tau146 = None

    tau149 += 3 * np.einsum("jiba->ijab", tau147, optimize=True)

    tau147 = None

    r2 -= 2 * np.einsum("jiab->abij", tau149, optimize=True) / 3

    r2 += 4 * np.einsum("jiba->abij", tau149, optimize=True) / 3

    tau149 = None

    tau223 = np.zeros((N, N, M, M))

    tau223 += np.einsum("ik,kjab->ijab", tau117, tau3, optimize=True)

    tau3 = None

    tau226 += np.einsum("ijab->ijab", tau223, optimize=True)

    tau223 = None

    r2 -= 2 * np.einsum("ijab->abij", tau226, optimize=True)

    r2 += 4 * np.einsum("ijba->abij", tau226, optimize=True)

    tau226 = None

    tau302 = np.zeros((N, N, M, M))

    tau302 += np.einsum("ik,kajb->ijab", tau117, u[o, v, o, v], optimize=True)

    tau312 += 3 * np.einsum("ijba->ijab", tau302, optimize=True)

    tau325 -= np.einsum("ijba->ijab", tau302, optimize=True)

    tau302 = None

    tau158 = np.zeros((N, N, M, M))

    tau158 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau159 = np.zeros((N, N, M, M))

    tau159 += np.einsum("cabd,ijcd->ijab", tau111, tau158, optimize=True)

    tau111 = None

    tau167 += np.einsum("ijab->ijab", tau159, optimize=True)

    tau159 = None

    tau195 = np.zeros((N, N, M, M))

    tau195 += np.einsum("cbjk,kica->ijab", t2, tau158, optimize=True)

    tau196 += np.einsum("ijba->ijab", tau195, optimize=True)

    tau197 = np.zeros((N, N, M, M))

    tau197 += np.einsum("ikac,kjbc->ijab", tau196, u[o, o, v, v], optimize=True)

    tau196 = None

    tau201 += np.einsum("ijba->ijab", tau197, optimize=True)

    tau197 = None

    tau202 = np.zeros((N, N, M, M))

    tau202 += np.einsum("bcjk,kiac->ijab", l2, tau201, optimize=True)

    tau201 = None

    tau207 += np.einsum("jiba->ijab", tau202, optimize=True)

    tau202 = None

    tau264 = np.zeros((N, N, N, N))

    tau264 += np.einsum("ijab,klab->ijkl", tau195, u[o, o, v, v], optimize=True)

    tau265 = np.zeros((N, N, M, M))

    tau265 += np.einsum("ablk,lkij->ijab", l2, tau264, optimize=True)

    tau264 = None

    r2 += 4 * np.einsum("ijab->abij", tau265, optimize=True) / 3

    r2 += 2 * np.einsum("jiab->abij", tau265, optimize=True) / 3

    tau265 = None

    tau379 += 2 * np.einsum("ijab->ijab", tau195, optimize=True)

    tau379 += np.einsum("ijba->ijab", tau195, optimize=True)

    tau385 -= np.einsum("ijab->ijab", tau195, optimize=True)

    tau391 -= np.einsum("jkab,kiba->ij", tau385, u[o, o, v, v], optimize=True)

    tau385 = None

    tau392 += np.einsum("ijab->ijab", tau195, optimize=True)

    tau195 = None

    tau396 += np.einsum("ikab,kjba->ij", tau392, u[o, o, v, v], optimize=True)

    tau392 = None

    tau266 = np.zeros((N, N, M, M))

    tau266 += np.einsum("ikcb,kjac->ijab", tau158, tau30, optimize=True)

    tau30 = None

    tau268 += np.einsum("ijab->ijab", tau266, optimize=True)

    tau266 = None

    tau269 = np.zeros((N, N, M, M))

    tau269 += np.einsum("ikac,kjbc->ijab", tau268, u[o, o, v, v], optimize=True)

    tau268 = None

    r2 += 2 * np.einsum("jiab->abij", tau269, optimize=True) / 3

    r2 += 4 * np.einsum("jiba->abij", tau269, optimize=True) / 3

    tau269 = None

    tau270 = np.zeros((N, N, M, M))

    tau270 += np.einsum("ikcb,kjac->ijab", tau158, tau28, optimize=True)

    tau272 += np.einsum("ijab->ijab", tau270, optimize=True)

    tau270 = None

    tau273 = np.zeros((N, N, M, M))

    tau273 += np.einsum("ikac,kjbc->ijab", tau272, u[o, o, v, v], optimize=True)

    tau272 = None

    r2 += 4 * np.einsum("jiab->abij", tau273, optimize=True) / 3

    r2 += 2 * np.einsum("jiba->abij", tau273, optimize=True) / 3

    tau273 = None

    tau340 = np.zeros((N, N, N, N))

    tau340 += np.einsum("ijab,kabl->ijkl", tau158, u[o, v, v, o], optimize=True)

    tau341 -= np.einsum("ijkl->ijkl", tau340, optimize=True)

    tau340 = None

    tau360 += 2 * np.einsum("kjil->ijkl", tau341, optimize=True)

    tau360 += np.einsum("klij->ijkl", tau341, optimize=True)

    tau341 = None

    tau347 = np.zeros((N, N, N, N))

    tau347 += np.einsum("klab,ijab->ijkl", tau158, tau297, optimize=True)

    tau297 = None

    tau349 += np.einsum("klij->ijkl", tau347, optimize=True)

    tau347 = None

    tau377 += np.einsum("ikcb,kjac->ijab", tau158, tau24, optimize=True)

    tau158 = None

    r2 -= 2 * np.einsum("jkac,kibc->abij", tau377, u[o, o, v, v], optimize=True)

    tau377 = None

    tau160 = np.zeros((N, N, M, M))

    tau160 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau161 = np.zeros((N, N, M, M))

    tau161 += np.einsum("jkcb,ikca->ijab", tau160, tau9, optimize=True)

    tau167 -= np.einsum("jiba->ijab", tau161, optimize=True)

    tau161 = None

    tau169 = np.zeros((N, N, N, N))

    tau169 += np.einsum("jlba,ikab->ijkl", tau160, tau95, optimize=True)

    tau171 += np.einsum("ijkl->ijkl", tau169, optimize=True)

    tau169 = None

    tau172 = np.zeros((N, N, M, M))

    tau172 += np.einsum("ijkl,lkab->ijab", tau171, u[o, o, v, v], optimize=True)

    tau171 = None

    tau173 = np.zeros((N, N, M, M))

    tau173 += np.einsum("ijba->ijab", tau172, optimize=True)

    tau172 = None

    tau227 = np.zeros((N, N, M, M))

    tau227 += np.einsum("ijdc,acdb->ijab", tau160, tau83, optimize=True)

    tau229 += np.einsum("ijab->ijab", tau227, optimize=True)

    tau227 = None

    tau230 = np.zeros((N, N, M, M))

    tau230 += np.einsum("ikac,kjbc->ijab", tau229, u[o, o, v, v], optimize=True)

    tau229 = None

    r2 -= 2 * np.einsum("jiab->abij", tau230, optimize=True)

    r2 += 4 * np.einsum("jiba->abij", tau230, optimize=True) / 3

    tau230 = None

    tau310 += np.einsum("ijab->ijab", tau160, optimize=True)

    tau333 += np.einsum("klab,ilkj->ijab", tau160, u[o, o, o, o], optimize=True)

    tau337 += np.einsum("ikca,jkcb->ijab", tau14, tau160, optimize=True)

    tau381 -= 3 * np.einsum("jlba,ikab->ijkl", tau160, tau24, optimize=True)

    tau160 = None

    tau164 = np.zeros((N, N, M, M))

    tau164 += np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau165 = np.zeros((N, N, M, M))

    tau165 += np.einsum("ikca,jkcb->ijab", tau14, tau164, optimize=True)

    tau167 += np.einsum("jiba->ijab", tau165, optimize=True)

    tau165 = None

    tau168 = np.zeros((N, N, M, M))

    tau168 += np.einsum("bcjk,ikca->ijab", l2, tau167, optimize=True)

    tau167 = None

    tau173 += np.einsum("jiba->ijab", tau168, optimize=True)

    tau168 = None

    r2 -= 2 * np.einsum("ijab->abij", tau173, optimize=True)

    r2 += 4 * np.einsum("ijba->abij", tau173, optimize=True) / 3

    tau173 = None

    tau337 -= np.einsum("jkcb,ikca->ijab", tau164, tau9, optimize=True)

    tau363 += np.einsum("ijdc,acdb->ijab", tau164, tau83, optimize=True)

    r2 += 2 * np.einsum("jkbc,kiac->abij", tau363, u[o, o, v, v], optimize=True) / 3

    tau363 = None

    tau380 -= 3 * np.einsum("jkba,ilab->ijkl", tau164, tau24, optimize=True)

    tau24 = None

    tau381 += np.einsum("jkba,ilab->ijkl", tau164, tau95, optimize=True)

    tau164 = None

    tau95 = None

    r2 += 2 * np.einsum("ijkl,klab->abij", tau381, u[o, o, v, v], optimize=True) / 3

    tau381 = None

    tau178 = np.zeros((N, N, M, M))

    tau178 -= 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau178 += 3 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau179 = np.zeros((N, N, M, M))

    tau179 += np.einsum("bckj,kica->ijab", t2, tau178, optimize=True)

    tau178 = None

    tau180 -= np.einsum("jiba->ijab", tau179, optimize=True)

    tau179 = None

    tau180 += 3 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau180 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau181 = np.zeros((N, N, M, M))

    tau181 += np.einsum("bcjk,kica->ijab", l2, tau180, optimize=True)

    tau182 -= np.einsum("jiba->ijab", tau181, optimize=True)

    tau183 = np.zeros((N, N, M, M))

    tau183 += np.einsum("bckj,kiac->ijab", t2, tau182, optimize=True)

    tau182 = None

    tau190 -= np.einsum("jiab->ijab", tau183, optimize=True)

    tau183 = None

    tau305 += np.einsum("jiba->ijab", tau181, optimize=True)

    tau181 = None

    tau315 += np.einsum("caki,kjcb->ijab", l2, tau180, optimize=True)

    tau180 = None

    tau320 += np.einsum("bcki,jkca->ijab", t2, tau315, optimize=True)

    tau315 = None

    tau185 = np.zeros((N, N, M, M))

    tau185 += np.einsum("abij->ijab", t2, optimize=True)

    tau185 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau186 = np.zeros((N, N, M, M))

    tau186 += np.einsum("klab,klij->ijab", tau185, tau54, optimize=True)

    tau185 = None

    tau187 += np.einsum("ijab->ijab", tau186, optimize=True)

    tau186 = None

    tau188 = np.zeros((N, N, M, M))

    tau188 += np.einsum("ikac,kjbc->ijab", tau187, u[o, o, v, v], optimize=True)

    tau187 = None

    tau190 += np.einsum("ijba->ijab", tau188, optimize=True)

    tau188 = None

    tau191 = np.zeros((N, N, M, M))

    tau191 += np.einsum("bckj,kiac->ijab", l2, tau190, optimize=True)

    tau190 = None

    tau207 += np.einsum("jiba->ijab", tau191, optimize=True)

    tau191 = None

    r2 += 2 * np.einsum("ijba->abij", tau207, optimize=True) / 3

    r2 += 2 * np.einsum("jiab->abij", tau207, optimize=True) / 3

    tau207 = None

    tau213 = np.zeros((N, N, M, M))

    tau213 += np.einsum("caik,bckj->ijab", l2, t2, optimize=True)

    tau214 = np.zeros((N, N, M, M))

    tau214 += np.einsum("jkcb,ikca->ijab", tau213, tau9, optimize=True)

    tau220 -= np.einsum("jiba->ijab", tau214, optimize=True)

    tau214 = None

    tau232 = np.zeros((N, N, M, M))

    tau232 += np.einsum("ijdc,acdb->ijab", tau213, tau83, optimize=True)

    tau233 += np.einsum("ijab->ijab", tau232, optimize=True)

    tau232 = None

    tau234 = np.zeros((N, N, M, M))

    tau234 += np.einsum("ikac,kjbc->ijab", tau233, u[o, o, v, v], optimize=True)

    tau233 = None

    r2 += 4 * np.einsum("ijab->abij", tau234, optimize=True) / 3

    r2 -= 2 * np.einsum("ijba->abij", tau234, optimize=True)

    tau234 = None

    tau333 += np.einsum("jkca,ikcb->ijab", tau14, tau213, optimize=True)

    tau361 -= np.einsum("ijab->ijab", tau213, optimize=True)

    tau213 = None

    tau362 -= np.einsum("adcb,ijcd->ijab", tau15, tau361, optimize=True)

    tau361 = None

    tau217 = np.zeros((N, N, M, M))

    tau217 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau218 = np.zeros((N, N, M, M))

    tau218 += np.einsum("ikca,jkcb->ijab", tau14, tau217, optimize=True)

    tau14 = None

    tau220 += np.einsum("jiba->ijab", tau218, optimize=True)

    tau218 = None

    tau221 = np.zeros((N, N, M, M))

    tau221 += np.einsum("bcjk,ikca->ijab", l2, tau220, optimize=True)

    tau220 = None

    tau222 += np.einsum("jiba->ijab", tau221, optimize=True)

    tau221 = None

    r2 += 4 * np.einsum("jiab->abij", tau222, optimize=True) / 3

    r2 -= 2 * np.einsum("jiba->abij", tau222, optimize=True)

    tau222 = None

    tau274 = np.zeros((N, N, M, M))

    tau274 += np.einsum("klab,ilkj->ijab", tau217, u[o, o, o, o], optimize=True)

    tau277 = np.zeros((N, N, M, M))

    tau277 += np.einsum("ijab->ijab", tau274, optimize=True)

    tau274 = None

    tau333 -= np.einsum("ikcb,jkca->ijab", tau217, tau9, optimize=True)

    tau9 = None

    tau362 += np.einsum("acdb,ijdc->ijab", tau174, tau217, optimize=True)

    tau217 = None

    tau174 = None

    r2 += 2 * np.einsum("ikac,kjbc->abij", tau362, u[o, o, v, v], optimize=True) / 3

    tau362 = None

    tau239 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau240 = np.zeros((M, M, M, M))

    tau240 += np.einsum("cedf,eabf->abcd", tau15, tau239, optimize=True)

    tau245 += np.einsum("cdab->abcd", tau240, optimize=True)

    tau240 = None

    tau241 = np.zeros((M, M, M, M))

    tau241 += np.einsum("eabf,cedf->abcd", tau239, tau83, optimize=True)

    tau83 = None

    tau245 += np.einsum("bdac->abcd", tau241, optimize=True)

    tau241 = None

    tau295 = np.zeros((M, M, M, M))

    tau295 += np.einsum("cedf,eafb->abcd", tau15, tau239, optimize=True)

    tau239 = None

    tau15 = None

    tau300 += np.einsum("cdab->abcd", tau295, optimize=True)

    tau295 = None

    tau301 += np.einsum("acbd->abcd", tau300, optimize=True)

    tau301 += np.einsum("dbca->abcd", tau300, optimize=True)

    tau300 = None

    r2 -= 2 * np.einsum("cdij,bcda->abij", l2, tau301, optimize=True)

    tau301 = None

    tau243 += np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau244 = np.zeros((M, M, M, M))

    tau244 += np.einsum("ijab,jicd->abcd", tau243, tau28, optimize=True)

    tau28 = None

    tau245 -= np.einsum("bdac->abcd", tau244, optimize=True)

    tau244 = None

    tau246 = np.zeros((N, N, M, M))

    tau246 += np.einsum("cdij,acdb->ijab", l2, tau245, optimize=True)

    tau253 += np.einsum("ijab->ijab", tau246, optimize=True)

    tau246 = None

    r2 += 4 * np.einsum("ijab->abij", tau253, optimize=True) / 3

    r2 += 2 * np.einsum("ijba->abij", tau253, optimize=True) / 3

    tau253 = None

    tau254 = np.zeros((N, N, M, M))

    tau254 += np.einsum("dcij,acdb->ijab", l2, tau245, optimize=True)

    tau245 = None

    tau261 += np.einsum("ijab->ijab", tau254, optimize=True)

    tau254 = None

    r2 += 2 * np.einsum("ijab->abij", tau261, optimize=True) / 3

    r2 += 4 * np.einsum("ijba->abij", tau261, optimize=True) / 3

    tau261 = None

    tau360 += 3 * np.einsum("kjab,ilab->ijkl", tau19, tau243, optimize=True)

    tau243 = None

    tau275 = np.zeros((N, N, M, M))

    tau275 += np.einsum("ablk,jikl->ijab", l2, u[o, o, o, o], optimize=True)

    tau276 = np.zeros((N, N, M, M))

    tau276 += np.einsum("bcki,jkac->ijab", t2, tau275, optimize=True)

    tau277 += np.einsum("jiab->ijab", tau276, optimize=True)

    tau276 = None

    tau278 = np.zeros((N, N, M, M))

    tau278 += np.einsum("bcjk,ikac->ijab", l2, tau277, optimize=True)

    tau277 = None

    r2 -= 2 * np.einsum("jiba->abij", tau278, optimize=True)

    r2 += 4 * np.einsum("ijba->abij", tau278, optimize=True) / 3

    tau278 = None

    tau279 = np.zeros((N, N, M, M))

    tau279 += np.einsum("bcik,jkac->ijab", t2, tau275, optimize=True)

    tau281 += np.einsum("ijab->ijab", tau279, optimize=True)

    tau279 = None

    tau282 = np.zeros((N, N, M, M))

    tau282 += np.einsum("bcjk,kiac->ijab", l2, tau281, optimize=True)

    tau281 = None

    r2 += 4 * np.einsum("jiba->abij", tau282, optimize=True)

    r2 -= 2 * np.einsum("ijba->abij", tau282, optimize=True)

    tau282 = None

    tau284 = np.zeros((N, N, M, M))

    tau284 += np.einsum("cbki,jkac->ijab", t2, tau275, optimize=True)

    tau285 += np.einsum("jiab->ijab", tau284, optimize=True)

    tau284 = None

    tau286 = np.zeros((N, N, M, M))

    tau286 += np.einsum("bcjk,ikac->ijab", l2, tau285, optimize=True)

    tau285 = None

    r2 -= 2 * np.einsum("jiab->abij", tau286, optimize=True)

    r2 += 4 * np.einsum("ijab->abij", tau286, optimize=True)

    tau286 = None

    tau287 = np.zeros((N, N, M, M))

    tau287 += np.einsum("cbik,jkac->ijab", t2, tau275, optimize=True)

    tau275 = None

    tau289 += np.einsum("ijab->ijab", tau287, optimize=True)

    tau287 = None

    tau290 = np.zeros((N, N, M, M))

    tau290 += np.einsum("bcjk,kiac->ijab", l2, tau289, optimize=True)

    tau289 = None

    r2 += 4 * np.einsum("jiab->abij", tau290, optimize=True) / 3

    r2 -= 2 * np.einsum("ijab->abij", tau290, optimize=True)

    tau290 = None

    tau303 = np.zeros((N, N, N, N))

    tau303 += np.einsum("abij,lkba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau304 = np.zeros((N, N, N, N))

    tau304 += 2 * np.einsum("klji->ijkl", tau303, optimize=True)

    tau304 += np.einsum("lkji->ijkl", tau303, optimize=True)

    tau308 = np.zeros((N, N, N, N))

    tau308 += np.einsum("lkji->ijkl", tau303, optimize=True)

    tau316 = np.zeros((N, N, N, N))

    tau316 += np.einsum("klji->ijkl", tau303, optimize=True)

    tau316 += 2 * np.einsum("lkji->ijkl", tau303, optimize=True)

    tau318 = np.zeros((N, N, N, N))

    tau318 += np.einsum("klij->ijkl", tau303, optimize=True)

    tau323 = np.zeros((N, N, M, M))

    tau323 += np.einsum("ablk,lkij->ijab", l2, tau303, optimize=True)

    tau333 -= 3 * np.einsum("bcjk,ikac->ijab", t2, tau323, optimize=True)

    tau326 = np.zeros((N, N, M, M))

    tau326 += np.einsum("ablk,klij->ijab", l2, tau303, optimize=True)

    tau327 += np.einsum("jiba->ijab", tau326, optimize=True)

    tau329 += np.einsum("bckj,ikca->ijab", t2, tau327, optimize=True)

    tau327 = None

    tau329 -= 2 * np.einsum("bcjk,kiac->ijab", t2, tau326, optimize=True)

    tau337 -= 3 * np.einsum("cbki,kjac->ijab", t2, tau326, optimize=True)

    tau326 = None

    tau329 -= np.einsum("jkli,klab->ijab", tau303, tau328, optimize=True)

    tau328 = None

    tau330 = np.zeros((N, N, N, N))

    tau330 += 2 * np.einsum("klji->ijkl", tau303, optimize=True)

    tau330 += np.einsum("lkji->ijkl", tau303, optimize=True)

    tau333 += np.einsum("kjil,klab->ijab", tau303, tau332, optimize=True)

    tau332 = None

    tau337 += np.einsum("klab,iklj->ijab", tau100, tau303, optimize=True)

    tau100 = None

    tau338 = np.zeros((N, N, N, N))

    tau338 += np.einsum("jmnl,imnk->ijkl", tau303, tau54, optimize=True)

    tau360 += 2 * np.einsum("ijlk->ijkl", tau338, optimize=True)

    tau376 += np.einsum("ijlk->ijkl", tau338, optimize=True)

    tau338 = None

    tau345 = np.zeros((N, N, N, N))

    tau345 += np.einsum("mkln,mijn->ijkl", tau303, tau54, optimize=True)

    tau349 += np.einsum("ijkl->ijkl", tau345, optimize=True)

    tau345 = None

    tau346 = np.zeros((N, N, N, N))

    tau346 += np.einsum("mjln,mink->ijkl", tau303, tau54, optimize=True)

    tau349 += np.einsum("ijkl->ijkl", tau346, optimize=True)

    tau346 = None

    tau360 += 2 * np.einsum("kjli->ijkl", tau349, optimize=True)

    tau360 += np.einsum("klji->ijkl", tau349, optimize=True)

    tau349 = None

    tau359 = np.zeros((N, N, N, N))

    tau359 += 3 * np.einsum("ijkl->ijkl", tau303, optimize=True)

    tau372 = np.zeros((N, N, N, N))

    tau372 += np.einsum("jmnl,imkn->ijkl", tau303, tau54, optimize=True)

    tau374 -= np.einsum("ijkl->ijkl", tau372, optimize=True)

    tau372 = None

    tau391 += np.einsum("lmki,mljk->ij", tau303, tau54, optimize=True)

    tau304 += 2 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau304 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau305 += np.einsum("balk,jikl->ijab", l2, tau304, optimize=True)

    tau304 = None

    tau312 += np.einsum("bckj,ikca->ijab", t2, tau305, optimize=True)

    tau305 = None

    tau308 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau312 += np.einsum("kijl,lkab->ijab", tau308, tau309, optimize=True)

    tau309 = None

    tau312 -= 3 * np.einsum("kilj,lkab->ijab", tau308, tau310, optimize=True)

    r2 += 2 * np.einsum("ackj,ikbc->abij", l2, tau312, optimize=True) / 3

    tau312 = None

    tau325 += np.einsum("klab,likj->ijab", tau292, tau308, optimize=True)

    tau360 -= 3 * np.einsum("minj,nkml->ijkl", tau308, tau54, optimize=True)

    tau360 -= 3 * np.einsum("km,milj->ijkl", tau4, tau308, optimize=True)

    tau396 += np.einsum("kjlm,mlik->ij", tau308, tau54, optimize=True)

    tau308 = None

    tau316 += np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau316 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau317 = np.zeros((N, N, M, M))

    tau317 += np.einsum("ablk,jikl->ijab", l2, tau316, optimize=True)

    tau316 = None

    tau320 += np.einsum("cbik,kjac->ijab", t2, tau317, optimize=True)

    tau317 = None

    tau318 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau320 += np.einsum("lkab,kjil->ijab", tau103, tau318, optimize=True)

    tau103 = None

    tau320 -= 3 * np.einsum("klab,ljki->ijab", tau292, tau318, optimize=True)

    tau292 = None

    r2 += 2 * np.einsum("bcki,kjac->abij", l2, tau320, optimize=True) / 3

    tau320 = None

    tau329 += np.einsum("klab,likj->ijab", tau310, tau318, optimize=True)

    tau310 = None

    r2 -= 2 * np.einsum("acik,jkbc->abij", l2, tau329, optimize=True)

    tau329 = None

    tau360 -= 3 * np.einsum("im,mkjl->ijkl", tau117, tau318, optimize=True)

    tau117 = None

    tau318 = None

    tau322 = np.zeros((N, N, M, M))

    tau322 -= np.einsum("abij->ijab", t2, optimize=True)

    tau322 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau324 = np.zeros((N, N, M, M))

    tau324 += np.einsum("acik,jkcb->ijab", l2, tau322, optimize=True)

    tau325 -= np.einsum("kjil,klab->ijab", tau303, tau324, optimize=True)

    tau324 = None

    tau303 = None

    tau325 -= np.einsum("jkcb,ikac->ijab", tau322, tau323, optimize=True)

    tau323 = None

    r2 -= 2 * np.einsum("bcjk,ikac->abij", l2, tau325, optimize=True)

    tau325 = None

    tau379 += 3 * np.einsum("kica,jkcb->ijab", tau19, tau322, optimize=True)

    tau19 = None

    tau380 += np.einsum("abij,klab->ijkl", l2, tau379, optimize=True)

    tau379 = None

    r2 += 2 * np.einsum("ijkl,lkba->abij", tau380, u[o, o, v, v], optimize=True) / 3

    tau380 = None

    tau384 -= np.einsum("kj,ikab->ijab", tau4, tau322, optimize=True)

    tau391 += np.einsum("jkab,kiab->ij", tau384, u[o, o, v, v], optimize=True)

    tau384 = None

    tau386 = np.zeros((N, N, M, M))

    tau386 += np.einsum("caki,jkcb->ijab", l2, tau322, optimize=True)

    tau322 = None

    tau391 += np.einsum("kjab,iabk->ij", tau386, u[o, v, v, o], optimize=True)

    tau386 = None

    tau330 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau331 += np.einsum("bakl,jikl->ijab", l2, tau330, optimize=True)

    tau333 += np.einsum("bckj,ikca->ijab", t2, tau331, optimize=True)

    tau331 = None

    r2 += 2 * np.einsum("acjk,ikbc->abij", l2, tau333, optimize=True) / 3

    tau333 = None

    tau336 = np.zeros((N, N, M, M))

    tau336 += np.einsum("ablk,jikl->ijab", l2, tau330, optimize=True)

    tau330 = None

    tau337 += np.einsum("cbik,kjac->ijab", t2, tau336, optimize=True)

    tau336 = None

    r2 += 2 * np.einsum("bcik,kjac->abij", l2, tau337, optimize=True) / 3

    tau337 = None

    tau355 = np.zeros((N, N, M, M))

    tau355 += 3 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau355 -= 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau356 += np.einsum("acki,kjbc->ijab", t2, tau355, optimize=True)

    tau355 = None

    tau357 -= np.einsum("acik,kjcb->ijab", l2, tau356, optimize=True)

    tau356 = None

    tau360 += np.einsum("abjl,ikab->ijkl", t2, tau357, optimize=True)

    tau357 = None

    tau358 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau360 += 3 * np.einsum("klab,ijab->ijkl", tau133, tau358, optimize=True)

    tau133 = None

    tau358 = None

    tau359 -= 2 * np.einsum("lkij->ijkl", u[o, o, o, o], optimize=True)

    tau359 += 3 * np.einsum("lkji->ijkl", u[o, o, o, o], optimize=True)

    tau360 -= np.einsum("mlnk,imjn->ijkl", tau359, tau54, optimize=True)

    tau54 = None

    tau359 = None

    r2 += 2 * np.einsum("abkl,ikjl->abij", l2, tau360, optimize=True) / 3

    tau360 = None

    tau365 = np.zeros((N, N, M, M))

    tau365 += np.einsum("acik,jcbk->ijab", l2, u[o, v, v, o], optimize=True)

    tau376 += 3 * np.einsum("ablj,ikab->ijkl", t2, tau365, optimize=True)

    tau393 += np.einsum("ijab->ijab", tau365, optimize=True)

    tau365 = None

    tau396 -= np.einsum("abik,kjab->ij", t2, tau393, optimize=True)

    tau393 = None

    tau366 = np.zeros((N, N, M, M))

    tau366 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau367 = np.zeros((N, N, N, N))

    tau367 += np.einsum("abjk,ilab->ijkl", t2, tau366, optimize=True)

    tau366 = None

    tau369 += np.einsum("ijkl->ijkl", tau367, optimize=True)

    tau367 = None

    tau376 -= np.einsum("ijlk->ijkl", tau369, optimize=True)

    tau376 -= 2 * np.einsum("iljk->ijkl", tau369, optimize=True)

    tau369 = None

    tau370 = np.zeros((N, N, M, M))

    tau370 += np.einsum("caik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau371 = np.zeros((N, N, N, N))

    tau371 += np.einsum("abjk,ilab->ijkl", t2, tau370, optimize=True)

    tau370 = None

    tau374 += np.einsum("ijkl->ijkl", tau371, optimize=True)

    tau371 = None

    tau376 -= 2 * np.einsum("ijlk->ijkl", tau374, optimize=True)

    tau376 -= np.einsum("iljk->ijkl", tau374, optimize=True)

    tau374 = None

    r2 += 2 * np.einsum("ablk,ikjl->abij", l2, tau376, optimize=True) / 3

    tau376 = None

    tau383 = np.zeros((N, N, N, N))

    tau383 += np.einsum("baij,abkl->ijkl", l2, t2, optimize=True)

    tau391 += np.einsum("mljk,iklm->ij", tau383, u[o, o, o, o], optimize=True)

    tau383 = None

    tau389 = np.zeros((N, N, N, N))

    tau389 -= np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau389 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau390 = np.zeros((N, N))

    tau390 += np.einsum("lk,kilj->ij", tau4, tau389, optimize=True)

    tau4 = None

    tau389 = None

    tau391 -= np.einsum("ij->ij", tau390, optimize=True)

    r2 -= 2 * np.einsum("jk,abik->abij", tau391, l2, optimize=True)

    tau391 = None

    tau396 -= np.einsum("ji->ij", tau390, optimize=True)

    tau390 = None

    tau394 = np.zeros((N, N, M, M))

    tau394 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau394 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau395 = np.zeros((N, N, M, M))

    tau395 += np.einsum("acik,jkcb->ijab", l2, tau394, optimize=True)

    tau394 = None

    tau396 += np.einsum("abki,kjab->ij", t2, tau395, optimize=True)

    tau395 = None

    r2 -= 2 * np.einsum("ki,abkj->abij", tau396, l2, optimize=True)

    tau396 = None

    return r2
