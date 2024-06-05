import numpy as np
from quadratictheory.cc.rhs.l_inter_RCCD import lambda_amplitudes_intermediates_rccd


def l_intermediates_qccd_restricted(t2, l2, u, f, v, o):
    r2 = lambda_amplitudes_intermediates_rccd(t2, l2, u, f, v, o)
    r2 += l2_intermediate_qccd_addition_restricted(t2, l2, u, f, v, o)

    return r2


def l2_intermediate_qccd_addition_restricted(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N))

    tau0 += np.einsum("abik,abkj->ij", l2, t2, optimize=True)

    tau169 = zeros((N, N, M, M))

    tau169 += np.einsum("ik,jkab->ijab", tau0, u[o, o, v, v], optimize=True)

    tau170 = zeros((N, N, M, M))

    tau170 += np.einsum("ijab->ijab", tau169, optimize=True)

    tau206 = zeros((N, N, M, M))

    tau206 -= np.einsum("ijab->ijab", tau169, optimize=True)

    tau169 = None

    tau1 = zeros((M, M))

    tau1 += np.einsum("acji,cbij->ab", l2, t2, optimize=True)

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("ac,jicb->ijab", tau1, u[o, o, v, v], optimize=True)

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum("jiab->ijab", tau2, optimize=True)

    tau102 = zeros((N, N, M, M))

    tau102 += 2 * np.einsum("ijab->ijab", tau2, optimize=True)

    tau102 -= np.einsum("jiab->ijab", tau2, optimize=True)

    tau206 += 18 * np.einsum("jiab->ijab", tau2, optimize=True)

    tau510 = zeros((N, N, M, M))

    tau510 += 18 * np.einsum("ijba->ijab", tau2, optimize=True)

    tau510 += 18 * np.einsum("jiab->ijab", tau2, optimize=True)

    r2 = zeros((M, M, N, N))

    r2 -= np.einsum("ik,jkab->abij", tau0, tau2, optimize=True) / 3

    tau0 = None

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("ac,ibjc->ijab", tau1, u[o, v, o, v], optimize=True)

    tau48 = zeros((N, N, M, M))

    tau48 -= np.einsum("jiab->ijab", tau28, optimize=True)

    tau218 = zeros((N, N, M, M))

    tau218 += 18 * np.einsum("jiab->ijab", tau28, optimize=True)

    tau28 = None

    tau140 = zeros((N, N, M, M))

    tau140 += np.einsum("bc,jiac->ijab", tau1, tau2, optimize=True)

    tau165 = zeros((N, N, M, M))

    tau165 += 36 * np.einsum("ijab->ijab", tau140, optimize=True)

    tau140 = None

    tau3 = zeros((N, N))

    tau3 += np.einsum("abki,abjk->ij", l2, t2, optimize=True)

    r2 -= np.einsum("jk,ikba->abij", tau3, tau2, optimize=True) / 3

    tau3 = None

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("klab,ilkj->ijab", tau4, u[o, o, o, o], optimize=True)

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("ijab->ijab", tau5, optimize=True)

    tau5 = None

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("abkl,jilk->ijab", l2, u[o, o, o, o], optimize=True)

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("bcki,jkac->ijab", t2, tau6, optimize=True)

    tau8 += np.einsum("jiab->ijab", tau7, optimize=True)

    tau7 = None

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("bcjk,ikac->ijab", l2, tau8, optimize=True)

    tau8 = None

    r2 -= 2 * np.einsum("jiba->abij", tau9, optimize=True)

    r2 += 4 * np.einsum("ijba->abij", tau9, optimize=True) / 3

    tau9 = None

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("bcik,jkac->ijab", t2, tau6, optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("ijab->ijab", tau10, optimize=True)

    tau10 = None

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("cbki,jkac->ijab", t2, tau6, optimize=True)

    tau18 = zeros((N, N, M, M))

    tau18 += np.einsum("jiab->ijab", tau17, optimize=True)

    tau17 = None

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum("cbik,jkac->ijab", t2, tau6, optimize=True)

    tau6 = None

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("ijab->ijab", tau20, optimize=True)

    tau20 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("klab,ilkj->ijab", tau11, u[o, o, o, o], optimize=True)

    tau13 += np.einsum("jiab->ijab", tau12, optimize=True)

    tau12 = None

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("bcjk,kiac->ijab", l2, tau13, optimize=True)

    tau13 = None

    r2 += 4 * np.einsum("jiba->abij", tau14, optimize=True)

    r2 -= 2 * np.einsum("ijba->abij", tau14, optimize=True)

    tau14 = None

    tau476 = zeros((N, N, M, M))

    tau476 += np.einsum("ijab->ijab", tau11, optimize=True)

    tau482 = zeros((N, N, M, M))

    tau482 -= 3 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau489 = zeros((N, N, M, M))

    tau489 += 3 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("acik,cbkj->ijab", l2, t2, optimize=True)

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("klab,ilkj->ijab", tau15, u[o, o, o, o], optimize=True)

    tau18 += np.einsum("ijab->ijab", tau16, optimize=True)

    tau16 = None

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("bcjk,ikac->ijab", l2, tau18, optimize=True)

    tau18 = None

    r2 -= 2 * np.einsum("jiab->abij", tau19, optimize=True)

    r2 += 4 * np.einsum("ijab->abij", tau19, optimize=True)

    tau19 = None

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum("caki,kjcb->ijab", t2, tau15, optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("ikca,jkbc->ijab", tau25, u[o, o, v, v], optimize=True)

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("acki,kjcb->ijab", l2, tau26, optimize=True)

    tau26 = None

    tau92 = zeros((N, N, M, M))

    tau92 += 2 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau27 = None

    tau45 = zeros((N, N, M, M))

    tau45 -= 2 * np.einsum("ijab->ijab", tau25, optimize=True)

    tau211 = zeros((N, N, M, M))

    tau211 += 6 * np.einsum("ijba->ijab", tau25, optimize=True)

    tau469 = zeros((N, N, N, N))

    tau469 += 6 * np.einsum("jlab,kiba->ijkl", tau25, u[o, o, v, v], optimize=True)

    tau516 = zeros((N, N, M, M))

    tau516 -= 4 * np.einsum("ijba->ijab", tau25, optimize=True)

    tau517 = zeros((N, N, M, M))

    tau517 -= 2 * np.einsum("ijab->ijab", tau25, optimize=True)

    tau25 = None

    tau141 = zeros((M, M, M, M))

    tau141 += np.einsum("ijbc,jiad->abcd", tau15, tau15, optimize=True)

    tau142 = zeros((N, N, M, M))

    tau142 += np.einsum("abcd,ijcd->ijab", tau141, u[o, o, v, v], optimize=True)

    tau141 = None

    tau165 += 36 * np.einsum("ijab->ijab", tau142, optimize=True)

    tau142 = None

    tau198 = zeros((N, N, M, M))

    tau198 -= 3 * np.einsum("ijab->ijab", tau15, optimize=True)

    tau244 = zeros((N, N, M, M))

    tau244 += np.einsum("ijcd,cadb->ijab", tau15, u[v, v, v, v], optimize=True)

    tau247 = zeros((N, N, M, M))

    tau247 += np.einsum("ijab->ijab", tau244, optimize=True)

    tau408 = zeros((N, N, M, M))

    tau408 -= 28 * np.einsum("ijba->ijab", tau244, optimize=True)

    tau244 = None

    tau330 = zeros((N, N, M, M))

    tau330 += np.einsum("ikcb,kjac->ijab", tau15, tau15, optimize=True)

    tau331 = zeros((N, N, M, M))

    tau331 += np.einsum("ikac,jkcb->ijab", tau330, u[o, o, v, v], optimize=True)

    tau332 = zeros((N, N, M, M))

    tau332 += np.einsum("ijab->ijab", tau331, optimize=True)

    tau331 = None

    tau346 = zeros((N, N, M, M))

    tau346 += np.einsum("ikac,jkbc->ijab", tau330, u[o, o, v, v], optimize=True)

    tau330 = None

    r2 += 28 * np.einsum("ijab->abij", tau346, optimize=True) / 9

    r2 -= 2 * np.einsum("ijba->abij", tau346, optimize=True)

    tau346 = None

    tau410 = zeros((N, N, M, M))

    tau410 += np.einsum("ijab->ijab", tau15, optimize=True)

    tau465 = zeros((N, N, M, M))

    tau465 += np.einsum("ijab->ijab", tau15, optimize=True)

    tau495 = zeros((N, N, M, M))

    tau495 += np.einsum("ijcd,cabd->ijab", tau15, u[v, v, v, v], optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 += np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum("klab,ilkj->ijab", tau21, u[o, o, o, o], optimize=True)

    tau23 += np.einsum("jiab->ijab", tau22, optimize=True)

    tau22 = None

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("bcjk,kiac->ijab", l2, tau23, optimize=True)

    tau23 = None

    r2 += 4 * np.einsum("jiab->abij", tau24, optimize=True) / 3

    r2 -= 2 * np.einsum("ijab->abij", tau24, optimize=True)

    tau24 = None

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("cbkj,kica->ijab", t2, tau21, optimize=True)

    tau45 += np.einsum("ijab->ijab", tau44, optimize=True)

    tau45 += np.einsum("jiba->ijab", tau44, optimize=True)

    tau67 = zeros((N, N, M, M))

    tau67 -= 12 * np.einsum("ijba->ijab", tau44, optimize=True)

    tau67 -= 12 * np.einsum("jiab->ijab", tau44, optimize=True)

    tau76 = zeros((N, N, M, M))

    tau76 -= 6 * np.einsum("ijba->ijab", tau44, optimize=True)

    tau76 -= 6 * np.einsum("jiab->ijab", tau44, optimize=True)

    tau99 = zeros((N, N, M, M))

    tau99 += 14 * np.einsum("ijba->ijab", tau44, optimize=True)

    tau99 += 18 * np.einsum("jiab->ijab", tau44, optimize=True)

    tau211 -= 3 * np.einsum("ijba->ijab", tau44, optimize=True)

    tau211 -= 3 * np.einsum("jiab->ijab", tau44, optimize=True)

    tau463 = zeros((N, N, N, N))

    tau463 += np.einsum("ijba,klba->ijkl", tau44, u[o, o, v, v], optimize=True)

    tau469 -= 3 * np.einsum("jlik->ijkl", tau463, optimize=True)

    tau469 -= 3 * np.einsum("ljki->ijkl", tau463, optimize=True)

    tau463 = None

    tau514 = zeros((N, N, N, N))

    tau514 -= 3 * np.einsum("baij,klba->ijkl", l2, tau44, optimize=True)

    tau516 += 2 * np.einsum("ijba->ijab", tau44, optimize=True)

    tau516 += 2 * np.einsum("jiab->ijab", tau44, optimize=True)

    tau517 += np.einsum("ijab->ijab", tau44, optimize=True)

    tau517 += np.einsum("jiba->ijab", tau44, optimize=True)

    tau44 = None

    tau50 = zeros((M, M, M, M))

    tau50 += np.einsum("ijad,jibc->abcd", tau15, tau21, optimize=True)

    tau53 = zeros((M, M, M, M))

    tau53 += 3 * np.einsum("abcd->abcd", tau50, optimize=True)

    tau50 = None

    tau61 = zeros((N, N, M, M))

    tau61 += np.einsum("caik,kjcb->ijab", t2, tau21, optimize=True)

    tau67 += 6 * np.einsum("jiab->ijab", tau61, optimize=True)

    tau223 = zeros((N, N, M, M))

    tau223 += np.einsum("ijba->ijab", tau61, optimize=True)

    tau383 = zeros((N, N, N, N))

    tau383 += np.einsum("ijab,lkba->ijkl", tau61, u[o, o, v, v], optimize=True)

    tau384 = zeros((N, N, M, M))

    tau384 += np.einsum("abkl,klij->ijab", l2, tau383, optimize=True)

    tau383 = None

    r2 += 4 * np.einsum("ijab->abij", tau384, optimize=True) / 3

    r2 += 2 * np.einsum("jiab->abij", tau384, optimize=True) / 3

    tau384 = None

    tau506 = zeros((N, N, M, M))

    tau506 += 2 * np.einsum("ijab->ijab", tau61, optimize=True)

    tau506 += np.einsum("ijba->ijab", tau61, optimize=True)

    tau516 -= np.einsum("ijba->ijab", tau61, optimize=True)

    tau61 = None

    tau128 = zeros((N, N, M, M))

    tau128 += np.einsum("ikcb,kjac->ijab", tau21, tau21, optimize=True)

    tau137 = zeros((N, N, M, M))

    tau137 -= 10 * np.einsum("ijab->ijab", tau128, optimize=True)

    tau295 = zeros((N, N, M, M))

    tau295 += np.einsum("ijab->ijab", tau128, optimize=True)

    tau128 = None

    tau271 = zeros((N, N, M, M))

    tau271 += np.einsum("ijcd,cabd->ijab", tau21, u[v, v, v, v], optimize=True)

    tau274 = zeros((N, N, M, M))

    tau274 += np.einsum("ijab->ijab", tau271, optimize=True)

    tau271 = None

    tau435 = zeros((N, N, N, N))

    tau435 += np.einsum("ijab,kabl->ijkl", tau21, u[o, v, v, o], optimize=True)

    tau436 = zeros((N, N, N, N))

    tau436 -= np.einsum("ijkl->ijkl", tau435, optimize=True)

    tau435 = None

    tau483 = zeros((N, N, M, M))

    tau483 -= np.einsum("ijcd,cbda->ijab", tau21, u[v, v, v, v], optimize=True)

    tau496 = zeros((M, M, M, M))

    tau496 += np.einsum("ijad,jibc->abcd", tau21, tau21, optimize=True)

    tau497 = zeros((N, N, M, M))

    tau497 += np.einsum("kjac,ikcb->ijab", tau15, tau21, optimize=True)

    tau499 = zeros((N, N, M, M))

    tau499 += 2 * np.einsum("ijab->ijab", tau497, optimize=True)

    tau505 = zeros((N, N, M, M))

    tau505 += np.einsum("ijab->ijab", tau497, optimize=True)

    tau497 = None

    tau505 += np.einsum("ikcb,kjac->ijab", tau15, tau21, optimize=True)

    tau29 = zeros((N, N))

    tau29 += np.einsum("abki,abkj->ij", l2, t2, optimize=True)

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("ik,jkab->ijab", tau29, u[o, o, v, v], optimize=True)

    tau36 += np.einsum("ijba->ijab", tau30, optimize=True)

    tau117 = zeros((N, N, M, M))

    tau117 -= 2 * np.einsum("ijab->ijab", tau30, optimize=True)

    tau117 += np.einsum("ijba->ijab", tau30, optimize=True)

    tau206 -= 2 * np.einsum("ijab->ijab", tau30, optimize=True)

    tau206 += 18 * np.einsum("ijba->ijab", tau30, optimize=True)

    tau417 = zeros((N, N, M, M))

    tau417 -= 36 * np.einsum("ijba->ijab", tau30, optimize=True)

    tau475 = zeros((N, N, M, M))

    tau475 += np.einsum("ijba->ijab", tau30, optimize=True)

    tau524 = zeros((N, N, M, M))

    tau524 += 2 * np.einsum("ijab->ijab", tau30, optimize=True)

    tau525 = zeros((N, N, M, M))

    tau525 += np.einsum("ijab->ijab", tau30, optimize=True)

    tau327 = zeros((N, N, M, M))

    tau327 += np.einsum("kj,ikab->ijab", tau29, tau15, optimize=True)

    tau339 = zeros((N, N, M, M))

    tau339 += np.einsum("ik,jkab->ijab", tau29, tau2, optimize=True)

    tau345 = zeros((N, N, M, M))

    tau345 -= 3 * np.einsum("ijab->ijab", tau339, optimize=True)

    tau339 = None

    tau358 = zeros((N, N, M, M))

    tau358 += np.einsum("ik,kjab->ijab", tau29, tau2, optimize=True)

    tau360 = zeros((N, N, M, M))

    tau360 += np.einsum("ijab->ijab", tau358, optimize=True)

    tau358 = None

    tau371 = zeros((N, N, M, M))

    tau371 += np.einsum("kj,ikab->ijab", tau29, tau21, optimize=True)

    tau372 = zeros((N, N, M, M))

    tau372 += np.einsum("ikac,jkbc->ijab", tau371, u[o, o, v, v], optimize=True)

    tau371 = None

    r2 += 10 * np.einsum("ijab->abij", tau372, optimize=True) / 3

    r2 -= 14 * np.einsum("ijba->abij", tau372, optimize=True) / 9

    r2 -= 14 * np.einsum("jiab->abij", tau372, optimize=True) / 9

    r2 += 10 * np.einsum("jiba->abij", tau372, optimize=True) / 3

    tau372 = None

    tau416 = zeros((N, N, M, M))

    tau416 += np.einsum("ik,kajb->ijab", tau29, u[o, v, o, v], optimize=True)

    tau430 = zeros((N, N, M, M))

    tau430 += 36 * np.einsum("jiba->ijab", tau416, optimize=True)

    tau478 = zeros((N, N, M, M))

    tau478 -= np.einsum("ijba->ijab", tau416, optimize=True)

    tau416 = None

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("acik,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau34 = zeros((N, N, M, M))

    tau34 -= np.einsum("ijab->ijab", tau31, optimize=True)

    tau468 = zeros((N, N, M, M))

    tau468 -= np.einsum("jiab->ijab", tau31, optimize=True)

    tau31 = None

    tau32 = zeros((N, N, M, M))

    tau32 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau32 -= np.einsum("abji->ijab", t2, optimize=True)

    tau33 = zeros((N, N, M, M))

    tau33 += np.einsum("ikac,kjcb->ijab", tau32, u[o, o, v, v], optimize=True)

    tau34 += np.einsum("ijab->ijab", tau33, optimize=True)

    tau468 += np.einsum("jiab->ijab", tau33, optimize=True)

    tau33 = None

    tau66 = zeros((N, N, M, M))

    tau66 += np.einsum("kj,kiab->ijab", tau29, tau32, optimize=True)

    tau67 -= 6 * np.einsum("ijab->ijab", tau66, optimize=True)

    tau516 += np.einsum("ijab->ijab", tau66, optimize=True)

    tau66 = None

    tau477 = zeros((N, N, M, M))

    tau477 += np.einsum("acik,jkbc->ijab", l2, tau32, optimize=True)

    tau32 = None

    tau34 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau35 = zeros((N, N, M, M))

    tau35 += np.einsum("bcjk,kica->ijab", l2, tau34, optimize=True)

    tau36 -= np.einsum("jiba->ijab", tau35, optimize=True)

    tau37 = zeros((N, N, M, M))

    tau37 += np.einsum("bckj,kiac->ijab", t2, tau36, optimize=True)

    tau36 = None

    tau48 += np.einsum("jiab->ijab", tau37, optimize=True)

    tau37 = None

    tau102 += np.einsum("jiba->ijab", tau35, optimize=True)

    tau145 = zeros((N, N, M, M))

    tau145 -= np.einsum("jiba->ijab", tau35, optimize=True)

    tau469 += 3 * np.einsum("abjl,kiba->ijkl", t2, tau35, optimize=True)

    tau470 = zeros((N, N, M, M))

    tau470 -= np.einsum("jiba->ijab", tau35, optimize=True)

    tau510 -= 18 * np.einsum("jiba->ijab", tau35, optimize=True)

    tau35 = None

    tau47 = zeros((N, N, M, M))

    tau47 += np.einsum("kjbc,kiac->ijab", tau15, tau34, optimize=True)

    tau48 -= np.einsum("jiba->ijab", tau47, optimize=True)

    tau47 = None

    tau116 = zeros((N, N, M, M))

    tau116 += np.einsum("cbkj,kica->ijab", l2, tau34, optimize=True)

    tau117 -= np.einsum("jiba->ijab", tau116, optimize=True)

    tau469 += 3 * np.einsum("ablj,ikba->ijkl", t2, tau116, optimize=True)

    tau475 -= np.einsum("jiba->ijab", tau116, optimize=True)

    tau116 = None

    tau221 = zeros((N, N, M, M))

    tau221 += np.einsum("kjbc,kiac->ijab", tau21, tau34, optimize=True)

    tau228 = zeros((N, N, M, M))

    tau228 -= np.einsum("jiba->ijab", tau221, optimize=True)

    tau221 = None

    tau246 = zeros((N, N, M, M))

    tau246 += np.einsum("jkcb,ikca->ijab", tau11, tau34, optimize=True)

    tau247 -= np.einsum("jiba->ijab", tau246, optimize=True)

    tau246 = None

    tau278 = zeros((M, M, M, M))

    tau278 += np.einsum("ijcd,ijab->abcd", tau21, tau34, optimize=True)

    tau279 = zeros((M, M, M, M))

    tau279 -= np.einsum("cdab->abcd", tau278, optimize=True)

    tau278 = None

    tau310 = zeros((N, N, M, M))

    tau310 += np.einsum("ikca,jkcb->ijab", tau34, tau4, optimize=True)

    tau311 = zeros((N, N, M, M))

    tau311 -= np.einsum("jiba->ijab", tau310, optimize=True)

    tau310 = None

    tau320 = zeros((M, M, M, M))

    tau320 += np.einsum("ijcd,ijab->abcd", tau15, tau34, optimize=True)

    tau321 = zeros((M, M, M, M))

    tau321 -= np.einsum("cdab->abcd", tau320, optimize=True)

    tau320 = None

    tau469 += 3 * np.einsum("ilab,jkab->ijkl", tau15, tau34, optimize=True)

    tau481 = zeros((N, N, M, M))

    tau481 += np.einsum("caik,kjcb->ijab", l2, tau34, optimize=True)

    tau485 = zeros((N, N, M, M))

    tau485 += np.einsum("bckj,kica->ijab", l2, tau34, optimize=True)

    tau486 = zeros((N, N, M, M))

    tau486 += np.einsum("jiba->ijab", tau485, optimize=True)

    tau510 -= np.einsum("jiab->ijab", tau485, optimize=True)

    tau485 = None

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum("acki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("ijab->ijab", tau38, optimize=True)

    tau466 = zeros((N, N, M, M))

    tau466 -= np.einsum("jiab->ijab", tau38, optimize=True)

    tau39 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau101 = zeros((N, N, M, M))

    tau101 += np.einsum("bckj,kica->ijab", l2, tau39, optimize=True)

    tau102 -= np.einsum("jiba->ijab", tau101, optimize=True)

    tau103 = zeros((N, N, M, M))

    tau103 += np.einsum("bcjk,kiac->ijab", t2, tau102, optimize=True)

    tau102 = None

    tau108 = zeros((N, N, M, M))

    tau108 -= 18 * np.einsum("jiab->ijab", tau103, optimize=True)

    tau103 = None

    tau117 += np.einsum("jiba->ijab", tau101, optimize=True)

    tau170 += 6 * np.einsum("jiba->ijab", tau101, optimize=True)

    tau171 = zeros((N, N, M, M))

    tau171 += np.einsum("bckj,kiac->ijab", t2, tau170, optimize=True)

    tau170 = None

    tau175 = zeros((N, N, M, M))

    tau175 += np.einsum("jiab->ijab", tau171, optimize=True)

    tau171 = None

    tau179 = zeros((N, N, M, M))

    tau179 -= np.einsum("bckj,kiac->ijab", t2, tau101, optimize=True)

    tau180 = zeros((N, N, M, M))

    tau180 += np.einsum("ijba->ijab", tau179, optimize=True)

    tau179 = None

    tau206 -= 6 * np.einsum("jiba->ijab", tau101, optimize=True)

    tau457 = zeros((N, N, N, N))

    tau457 += np.einsum("abkl,jiba->ijkl", t2, tau101, optimize=True)

    tau458 = zeros((N, N, N, N))

    tau458 += np.einsum("iljk->ijkl", tau457, optimize=True)

    tau457 = None

    tau524 -= np.einsum("jiba->ijab", tau101, optimize=True)

    tau104 = zeros((N, N, M, M))

    tau104 += np.einsum("kjbc,kiac->ijab", tau15, tau39, optimize=True)

    tau108 += 18 * np.einsum("jiba->ijab", tau104, optimize=True)

    tau104 = None

    tau144 = zeros((N, N, M, M))

    tau144 += np.einsum("cbjk,kica->ijab", l2, tau39, optimize=True)

    tau145 += np.einsum("jiba->ijab", tau144, optimize=True)

    tau240 = zeros((N, N, M, M))

    tau240 -= np.einsum("bckj,kiac->ijab", t2, tau144, optimize=True)

    tau241 = zeros((N, N, M, M))

    tau241 += np.einsum("ijba->ijab", tau240, optimize=True)

    tau240 = None

    tau452 = zeros((N, N, N, N))

    tau452 += np.einsum("abkl,jiba->ijkl", t2, tau144, optimize=True)

    tau144 = None

    tau453 = zeros((N, N, N, N))

    tau453 += np.einsum("iljk->ijkl", tau452, optimize=True)

    tau452 = None

    tau148 = zeros((N, N, M, M))

    tau148 += np.einsum("jkcb,ikca->ijab", tau11, tau39, optimize=True)

    tau150 = zeros((N, N, M, M))

    tau150 += np.einsum("jiba->ijab", tau148, optimize=True)

    tau148 = None

    tau174 = zeros((N, N, M, M))

    tau174 += np.einsum("kjbc,kiac->ijab", tau21, tau39, optimize=True)

    tau175 += 6 * np.einsum("jiba->ijab", tau174, optimize=True)

    tau174 = None

    tau225 = zeros((N, N, M, M))

    tau225 += np.einsum("bcjk,kica->ijab", l2, tau39, optimize=True)

    tau226 = zeros((N, N, M, M))

    tau226 -= np.einsum("bckj,ikca->ijab", t2, tau225, optimize=True)

    tau228 -= np.einsum("jiab->ijab", tau226, optimize=True)

    tau226 = None

    tau448 = zeros((N, N, N, N))

    tau448 += np.einsum("abkl,jiba->ijkl", t2, tau225, optimize=True)

    tau449 = zeros((N, N, N, N))

    tau449 += np.einsum("iljk->ijkl", tau448, optimize=True)

    tau448 = None

    tau481 += 6 * np.einsum("jiba->ijab", tau225, optimize=True)

    tau239 = zeros((N, N, M, M))

    tau239 += np.einsum("ikca,jkcb->ijab", tau39, tau4, optimize=True)

    tau241 -= np.einsum("jiba->ijab", tau239, optimize=True)

    tau239 = None

    tau242 = zeros((N, N, M, M))

    tau242 += np.einsum("bcjk,ikca->ijab", l2, tau241, optimize=True)

    tau241 = None

    tau243 = zeros((N, N, M, M))

    tau243 -= np.einsum("jiba->ijab", tau242, optimize=True)

    tau242 = None

    tau460 = zeros((N, N, M, M))

    tau460 += np.einsum("cbkj,kica->ijab", l2, tau39, optimize=True)

    tau461 = zeros((N, N, N, N))

    tau461 += np.einsum("abkl,jiba->ijkl", t2, tau460, optimize=True)

    tau462 = zeros((N, N, N, N))

    tau462 += np.einsum("iljk->ijkl", tau461, optimize=True)

    tau461 = None

    tau486 += 6 * np.einsum("jiba->ijab", tau460, optimize=True)

    tau460 = None

    tau488 = zeros((N, N, M, M))

    tau488 += np.einsum("bcki,jkca->ijab", t2, tau486, optimize=True)

    tau486 = None

    tau40 = zeros((M, M, M, M))

    tau40 += np.einsum("abij,cdij->abcd", l2, t2, optimize=True)

    tau41 = zeros((N, N, M, M))

    tau41 += np.einsum("ijcd,acbd->ijab", tau39, tau40, optimize=True)

    tau48 += np.einsum("ijab->ijab", tau41, optimize=True)

    tau218 -= 18 * np.einsum("ijab->ijab", tau41, optimize=True)

    tau41 = None

    tau127 = zeros((N, N, M, M))

    tau127 += np.einsum("ijdc,acdb->ijab", tau11, tau40, optimize=True)

    tau137 += 36 * np.einsum("ijab->ijab", tau127, optimize=True)

    tau127 = None

    tau220 = zeros((N, N, M, M))

    tau220 += np.einsum("ijcd,acbd->ijab", tau34, tau40, optimize=True)

    tau228 -= np.einsum("ijab->ijab", tau220, optimize=True)

    tau220 = None

    tau249 = zeros((M, M, M, M))

    tau249 += np.einsum("aebf,cefd->abcd", tau40, u[v, v, v, v], optimize=True)

    tau252 = zeros((M, M, M, M))

    tau252 += np.einsum("abcd->abcd", tau249, optimize=True)

    tau249 = None

    tau361 = zeros((N, N, M, M))

    tau361 += np.einsum("ijdc,acdb->ijab", tau4, tau40, optimize=True)

    tau362 = zeros((N, N, M, M))

    tau362 += np.einsum("ikac,jkbc->ijab", tau361, u[o, o, v, v], optimize=True)

    tau361 = None

    r2 -= 23 * np.einsum("ijab->abij", tau362, optimize=True) / 9

    r2 += 4 * np.einsum("ijba->abij", tau362, optimize=True) / 3

    tau362 = None

    tau409 = zeros((M, M, M, M))

    tau409 += np.einsum("aebf,cedf->abcd", tau40, u[v, v, v, v], optimize=True)

    tau412 = zeros((M, M, M, M))

    tau412 -= np.einsum("abcd->abcd", tau409, optimize=True)

    tau409 = None

    tau496 += np.einsum("afed,befc->abcd", tau40, tau40, optimize=True)

    r2 += 4 * np.einsum("bacd,jicd->abij", tau496, u[o, o, v, v], optimize=True) / 3

    tau496 = None

    tau42 = zeros((M, M))

    tau42 += np.einsum("caij,cbij->ab", l2, t2, optimize=True)

    tau43 = zeros((N, N, M, M))

    tau43 += np.einsum("cb,caij->ijab", tau42, t2, optimize=True)

    tau45 += np.einsum("ijba->ijab", tau43, optimize=True)

    tau46 = zeros((N, N, M, M))

    tau46 += np.einsum("ikca,kjbc->ijab", tau45, u[o, o, v, v], optimize=True)

    tau48 += np.einsum("ijba->ijab", tau46, optimize=True)

    tau46 = None

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum("bcjk,kiac->ijab", l2, tau48, optimize=True)

    tau48 = None

    tau92 += 6 * np.einsum("jiba->ijab", tau49, optimize=True)

    tau49 = None

    tau67 += 6 * np.einsum("jiab->ijab", tau43, optimize=True)

    tau99 += 18 * np.einsum("ijab->ijab", tau43, optimize=True)

    tau99 -= 36 * np.einsum("jiab->ijab", tau43, optimize=True)

    tau211 -= 3 * np.einsum("ijab->ijab", tau43, optimize=True)

    tau464 = zeros((N, N, N, N))

    tau464 += np.einsum("ijba,klba->ijkl", tau43, u[o, o, v, v], optimize=True)

    tau469 -= 3 * np.einsum("jlki->ijkl", tau464, optimize=True)

    tau469 -= 3 * np.einsum("ljik->ijkl", tau464, optimize=True)

    tau464 = None

    tau516 += 2 * np.einsum("ijab->ijab", tau43, optimize=True)

    tau516 -= np.einsum("jiab->ijab", tau43, optimize=True)

    tau523 = zeros((N, N))

    tau523 -= np.einsum("jkab,kiab->ij", tau516, u[o, o, v, v], optimize=True)

    tau516 = None

    tau517 += np.einsum("ijba->ijab", tau43, optimize=True)

    tau517 -= 2 * np.einsum("jiba->ijab", tau43, optimize=True)

    tau93 = zeros((N, N, M, M))

    tau93 += np.einsum("ac,ijbc->ijab", tau42, u[o, o, v, v], optimize=True)

    tau94 = zeros((N, N, M, M))

    tau94 += np.einsum("caki,kjcb->ijab", t2, tau93, optimize=True)

    tau93 = None

    tau108 += 18 * np.einsum("ijba->ijab", tau94, optimize=True)

    tau94 = None

    tau51 = zeros((N, N, M, M))

    tau51 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    tau52 = zeros((M, M, M, M))

    tau52 += np.einsum("ijad,jibc->abcd", tau21, tau51, optimize=True)

    tau53 -= np.einsum("abcd->abcd", tau52, optimize=True)

    tau52 = None

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("abcd,ijdc->ijab", tau53, u[o, o, v, v], optimize=True)

    tau53 = None

    tau92 += 2 * np.einsum("jiab->ijab", tau54, optimize=True)

    tau54 = None

    tau64 = zeros((N, N, M, M))

    tau64 -= np.einsum("ijab->ijab", tau51, optimize=True)

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum("cbjk,kica->ijab", t2, tau51, optimize=True)

    tau76 -= 6 * np.einsum("jiab->ijab", tau69, optimize=True)

    tau99 += 18 * np.einsum("jiab->ijab", tau69, optimize=True)

    tau172 = zeros((N, N, M, M))

    tau172 += np.einsum("jiab->ijab", tau69, optimize=True)

    tau381 = zeros((N, N, N, N))

    tau381 += np.einsum("ijba,klba->ijkl", tau69, u[o, o, v, v], optimize=True)

    tau382 = zeros((N, N, M, M))

    tau382 += np.einsum("abkl,lkji->ijab", l2, tau381, optimize=True)

    tau381 = None

    r2 += 2 * np.einsum("ijab->abij", tau382, optimize=True) / 3

    r2 += 4 * np.einsum("jiab->abij", tau382, optimize=True) / 3

    tau382 = None

    tau506 += 2 * np.einsum("jiab->ijab", tau69, optimize=True)

    tau506 += np.einsum("jiba->ijab", tau69, optimize=True)

    tau517 += np.einsum("jiba->ijab", tau69, optimize=True)

    tau69 = None

    tau523 += np.einsum("jkab,kiab->ij", tau517, u[o, o, v, v], optimize=True)

    tau517 = None

    tau97 = zeros((N, N, M, M))

    tau97 += np.einsum("ijab->ijab", tau51, optimize=True)

    tau131 = zeros((N, N, M, M))

    tau131 += 10 * np.einsum("ijab->ijab", tau51, optimize=True)

    tau168 = zeros((N, N, M, M))

    tau168 += np.einsum("kiac,kjbc->ijab", tau34, tau51, optimize=True)

    tau175 -= 6 * np.einsum("jiba->ijab", tau168, optimize=True)

    tau168 = None

    tau227 = zeros((N, N, M, M))

    tau227 += np.einsum("kiac,kjbc->ijab", tau39, tau51, optimize=True)

    tau228 += np.einsum("jiba->ijab", tau227, optimize=True)

    tau227 = None

    tau230 = zeros((M, M, M, M))

    tau230 += np.einsum("ijad,jibc->abcd", tau15, tau51, optimize=True)

    tau232 = zeros((M, M, M, M))

    tau232 += 3 * np.einsum("abcd->abcd", tau230, optimize=True)

    tau230 = None

    tau235 = zeros((M, M, M, M))

    tau235 += np.einsum("ijbd,jiac->abcd", tau51, tau51, optimize=True)

    tau236 = zeros((N, N, M, M))

    tau236 += np.einsum("abdc,ijdc->ijab", tau235, u[o, o, v, v], optimize=True)

    tau235 = None

    tau243 += np.einsum("ijab->ijab", tau236, optimize=True)

    tau236 = None

    tau251 = zeros((M, M, M, M))

    tau251 += np.einsum("ijab,ijcd->abcd", tau34, tau51, optimize=True)

    tau252 -= np.einsum("cdab->abcd", tau251, optimize=True)

    tau251 = None

    tau254 = zeros((N, N, M, M))

    tau254 += np.einsum("ijcd,cabd->ijab", tau51, u[v, v, v, v], optimize=True)

    tau257 = zeros((N, N, M, M))

    tau257 += np.einsum("ijab->ijab", tau254, optimize=True)

    tau254 = None

    tau282 = zeros((N, N, M, M))

    tau282 += np.einsum("ikcb,kjac->ijab", tau21, tau51, optimize=True)

    tau283 = zeros((N, N, M, M))

    tau283 += np.einsum("ijab->ijab", tau282, optimize=True)

    tau282 = None

    tau386 = zeros((N, N, M, M))

    tau386 += np.einsum("kjac,ikcb->ijab", tau21, tau51, optimize=True)

    tau387 = zeros((N, N, M, M))

    tau387 += np.einsum("ijab->ijab", tau386, optimize=True)

    tau386 = None

    tau389 = zeros((N, N, M, M))

    tau389 += np.einsum("ikcb,kjac->ijab", tau51, tau51, optimize=True)

    tau390 = zeros((N, N, M, M))

    tau390 += np.einsum("ijab->ijab", tau389, optimize=True)

    tau389 = None

    tau410 += np.einsum("ijab->ijab", tau51, optimize=True)

    tau411 = zeros((M, M, M, M))

    tau411 += np.einsum("ijab,jcid->abcd", tau410, u[o, v, o, v], optimize=True)

    tau412 += np.einsum("abcd->abcd", tau411, optimize=True)

    tau411 = None

    tau415 = zeros((M, M, M, M))

    tau415 -= 18 * np.einsum("abcd->abcd", tau412, optimize=True)

    tau415 -= 18 * np.einsum("dcba->abcd", tau412, optimize=True)

    tau412 = None

    tau414 = zeros((M, M, M, M))

    tau414 += np.einsum("ijab,ijcd->abcd", tau38, tau410, optimize=True)

    tau38 = None

    tau415 += 18 * np.einsum("cdab->abcd", tau414, optimize=True)

    tau415 += 18 * np.einsum("badc->abcd", tau414, optimize=True)

    tau414 = None

    tau420 = zeros((N, N, M, M))

    tau420 += 7 * np.einsum("ijab->ijab", tau51, optimize=True)

    tau422 = zeros((N, N, M, M))

    tau422 += 9 * np.einsum("ijab->ijab", tau51, optimize=True)

    tau441 = zeros((N, N, N, N))

    tau441 += np.einsum("ijab,kabl->ijkl", tau51, u[o, v, v, o], optimize=True)

    tau442 = zeros((N, N, N, N))

    tau442 -= np.einsum("ijkl->ijkl", tau441, optimize=True)

    tau441 = None

    tau467 = zeros((N, N, M, M))

    tau467 += np.einsum("ijab->ijab", tau51, optimize=True)

    tau478 += np.einsum("ijcd,cbda->ijab", tau51, u[v, v, v, v], optimize=True)

    tau488 += 6 * np.einsum("klab,jlki->ijab", tau51, u[o, o, o, o], optimize=True)

    tau490 = zeros((N, N, M, M))

    tau490 -= 3 * np.einsum("ikcb,kjac->ijab", tau15, tau51, optimize=True)

    tau493 = zeros((N, N, M, M))

    tau493 -= 3 * np.einsum("kjac,ikcb->ijab", tau15, tau51, optimize=True)

    tau55 = zeros((N, N, M, M))

    tau55 += np.einsum("cdij,dcba->ijab", l2, u[v, v, v, v], optimize=True)

    tau56 = zeros((M, M))

    tau56 += np.einsum("acij,ijbc->ab", t2, tau55, optimize=True)

    tau90 = zeros((M, M))

    tau90 += 6 * np.einsum("ab->ab", tau56, optimize=True)

    tau56 = None

    tau117 += np.einsum("jiab->ijab", tau55, optimize=True)

    tau145 += np.einsum("ijba->ijab", tau55, optimize=True)

    tau260 = zeros((N, N, M, M))

    tau260 += np.einsum("ackj,ikbc->ijab", t2, tau55, optimize=True)

    tau264 = zeros((N, N, M, M))

    tau264 += np.einsum("ijab->ijab", tau260, optimize=True)

    tau260 = None

    tau266 = zeros((N, N, M, M))

    tau266 += np.einsum("ackj,kicb->ijab", t2, tau55, optimize=True)

    tau269 = zeros((N, N, M, M))

    tau269 += np.einsum("ijab->ijab", tau266, optimize=True)

    tau266 = None

    tau308 = zeros((N, N, M, M))

    tau308 += np.einsum("ackj,ikcb->ijab", t2, tau55, optimize=True)

    tau311 += np.einsum("ijab->ijab", tau308, optimize=True)

    tau308 = None

    tau313 = zeros((N, N, M, M))

    tau313 += np.einsum("ackj,kibc->ijab", t2, tau55, optimize=True)

    tau316 = zeros((N, N, M, M))

    tau316 += np.einsum("ijab->ijab", tau313, optimize=True)

    tau313 = None

    tau318 = zeros((N, N, M, M))

    tau318 += np.einsum("ac,ijbc->ijab", tau1, tau55, optimize=True)

    tau323 = zeros((N, N, M, M))

    tau323 += np.einsum("ijab->ijab", tau318, optimize=True)

    tau318 = None

    tau324 = zeros((N, N, M, M))

    tau324 += np.einsum("ac,ijcb->ijab", tau1, tau55, optimize=True)

    tau326 = zeros((N, N, M, M))

    tau326 += np.einsum("ijab->ijab", tau324, optimize=True)

    tau324 = None

    tau397 = zeros((N, N, M, M))

    tau397 += 19 * np.einsum("ijab->ijab", tau55, optimize=True)

    tau397 += 8 * np.einsum("ijba->ijab", tau55, optimize=True)

    tau417 += 8 * np.einsum("jiab->ijab", tau55, optimize=True)

    tau417 += 19 * np.einsum("jiba->ijab", tau55, optimize=True)

    tau55 = None

    tau57 = zeros((M, M))

    tau57 += np.einsum("caij,cbji->ab", l2, t2, optimize=True)

    tau58 = zeros((M, M))

    tau58 += np.einsum("cd,cadb->ab", tau57, u[v, v, v, v], optimize=True)

    tau90 -= np.einsum("ab->ab", tau58, optimize=True)

    tau58 = None

    tau95 = zeros((N, N, M, M))

    tau95 += np.einsum("cb,caij->ijab", tau57, t2, optimize=True)

    tau57 = None

    tau99 -= 3 * np.einsum("jiab->ijab", tau95, optimize=True)

    tau95 = None

    tau59 = zeros((N, N, N, N))

    tau59 += np.einsum("abij,abkl->ijkl", l2, t2, optimize=True)

    tau60 = zeros((N, N, M, M))

    tau60 += np.einsum("abkl,klij->ijab", t2, tau59, optimize=True)

    tau67 += 6 * np.einsum("jiab->ijab", tau60, optimize=True)

    tau99 += 18 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau172 += np.einsum("ijab->ijab", tau60, optimize=True)

    tau60 = None

    tau173 = zeros((N, N, M, M))

    tau173 += np.einsum("ikac,kjbc->ijab", tau172, u[o, o, v, v], optimize=True)

    tau172 = None

    tau175 += 6 * np.einsum("ijba->ijab", tau173, optimize=True)

    tau173 = None

    tau115 = zeros((N, N, M, M))

    tau115 += np.einsum("ijkl,lkba->ijab", tau59, u[o, o, v, v], optimize=True)

    tau117 += np.einsum("jiab->ijab", tau115, optimize=True)

    tau118 = zeros((N, N, M, M))

    tau118 += np.einsum("bcjk,ikca->ijab", t2, tau117, optimize=True)

    tau117 = None

    tau124 = zeros((N, N, M, M))

    tau124 += np.einsum("ijba->ijab", tau118, optimize=True)

    tau118 = None

    tau145 += np.einsum("ijba->ijab", tau115, optimize=True)

    tau255 = zeros((N, N, M, M))

    tau255 += np.einsum("ackj,kibc->ijab", t2, tau115, optimize=True)

    tau257 += np.einsum("ijab->ijab", tau255, optimize=True)

    tau255 = None

    tau272 = zeros((N, N, M, M))

    tau272 += np.einsum("ackj,ikbc->ijab", t2, tau115, optimize=True)

    tau274 += np.einsum("ijab->ijab", tau272, optimize=True)

    tau272 = None

    tau289 = zeros((N, N, M, M))

    tau289 += np.einsum("ackj,kicb->ijab", t2, tau115, optimize=True)

    tau291 = zeros((N, N, M, M))

    tau291 += np.einsum("ijab->ijab", tau289, optimize=True)

    tau289 = None

    tau301 = zeros((N, N, M, M))

    tau301 += np.einsum("ackj,ikcb->ijab", t2, tau115, optimize=True)

    tau303 = zeros((N, N, M, M))

    tau303 += np.einsum("ijab->ijab", tau301, optimize=True)

    tau301 = None

    tau397 += 23 * np.einsum("ijab->ijab", tau115, optimize=True)

    tau397 += 10 * np.einsum("ijba->ijab", tau115, optimize=True)

    tau494 = zeros((N, N, M, M))

    tau494 += np.einsum("ijab->ijab", tau115, optimize=True)

    tau115 = None

    tau126 = zeros((N, N, M, M))

    tau126 += np.einsum("klab,ilkj->ijab", tau15, tau59, optimize=True)

    tau137 += 36 * np.einsum("ijab->ijab", tau126, optimize=True)

    tau126 = None

    tau152 = zeros((N, N, M, M))

    tau152 += np.einsum("klab,lijk->ijab", tau15, tau59, optimize=True)

    tau163 = zeros((N, N, M, M))

    tau163 += 36 * np.einsum("ijab->ijab", tau152, optimize=True)

    tau152 = None

    tau184 = zeros((N, N, N, N))

    tau184 += np.einsum("imkn,njml->ijkl", tau59, tau59, optimize=True)

    tau185 = zeros((N, N, N, N))

    tau185 += np.einsum("ijkl->ijkl", tau184, optimize=True)

    tau184 = None

    tau222 = zeros((N, N, M, M))

    tau222 += np.einsum("ablk,klij->ijab", t2, tau59, optimize=True)

    tau223 += np.einsum("ijab->ijab", tau222, optimize=True)

    tau222 = None

    tau224 = zeros((N, N, M, M))

    tau224 += np.einsum("ikac,kjbc->ijab", tau223, u[o, o, v, v], optimize=True)

    tau223 = None

    tau228 += np.einsum("ijba->ijab", tau224, optimize=True)

    tau224 = None

    tau229 = zeros((N, N, M, M))

    tau229 += np.einsum("bcjk,kiac->ijab", l2, tau228, optimize=True)

    tau228 = None

    tau234 = zeros((N, N, M, M))

    tau234 += 6 * np.einsum("jiba->ijab", tau229, optimize=True)

    tau229 = None

    tau256 = zeros((N, N, M, M))

    tau256 += np.einsum("klab,kilj->ijab", tau34, tau59, optimize=True)

    tau257 -= np.einsum("ijab->ijab", tau256, optimize=True)

    tau256 = None

    tau258 = zeros((N, N, M, M))

    tau258 += np.einsum("bckj,ikca->ijab", l2, tau257, optimize=True)

    r2 += np.einsum("jiba->abij", tau258, optimize=True) / 9

    r2 += 23 * np.einsum("jiab->abij", tau258, optimize=True) / 18

    tau258 = None

    tau305 = zeros((N, N, M, M))

    tau305 += np.einsum("bcjk,ikca->ijab", l2, tau257, optimize=True)

    tau257 = None

    tau307 = zeros((N, N, M, M))

    tau307 += np.einsum("jiba->ijab", tau305, optimize=True)

    tau305 = None

    tau273 = zeros((N, N, M, M))

    tau273 += np.einsum("klab,iklj->ijab", tau34, tau59, optimize=True)

    tau274 -= np.einsum("ijab->ijab", tau273, optimize=True)

    tau275 = zeros((N, N, M, M))

    tau275 += np.einsum("bcjk,ikca->ijab", l2, tau274, optimize=True)

    tau274 = None

    r2 += 10 * np.einsum("ijba->abij", tau275, optimize=True) / 9

    r2 -= 14 * np.einsum("ijab->abij", tau275, optimize=True) / 9

    tau275 = None

    tau408 -= 10 * np.einsum("ijba->ijab", tau273, optimize=True)

    tau273 = None

    tau285 = zeros((N, N, M, M))

    tau285 += np.einsum("lkba,ijkl->ijab", tau225, tau59, optimize=True)

    tau287 = zeros((N, N, M, M))

    tau287 += 6 * np.einsum("ijab->ijab", tau285, optimize=True)

    tau285 = None

    tau286 = zeros((N, N, M, M))

    tau286 += np.einsum("lkba,ijlk->ijab", tau101, tau59, optimize=True)

    tau287 += 6 * np.einsum("ijab->ijab", tau286, optimize=True)

    tau286 = None

    tau290 = zeros((N, N, M, M))

    tau290 += np.einsum("klab,kijl->ijab", tau34, tau59, optimize=True)

    tau291 -= np.einsum("ijab->ijab", tau290, optimize=True)

    tau290 = None

    tau297 = zeros((N, N, M, M))

    tau297 += np.einsum("lkba,ijkl->ijab", tau101, tau59, optimize=True)

    tau101 = None

    tau299 = zeros((N, N, M, M))

    tau299 += 6 * np.einsum("ijab->ijab", tau297, optimize=True)

    tau297 = None

    tau298 = zeros((N, N, M, M))

    tau298 += np.einsum("lkba,ijlk->ijab", tau225, tau59, optimize=True)

    tau225 = None

    tau299 += 6 * np.einsum("ijab->ijab", tau298, optimize=True)

    tau298 = None

    tau302 = zeros((N, N, M, M))

    tau302 += np.einsum("klab,ikjl->ijab", tau34, tau59, optimize=True)

    tau303 -= np.einsum("ijab->ijab", tau302, optimize=True)

    tau302 = None

    tau336 = zeros((N, N, M, M))

    tau336 += np.einsum("klab,likj->ijab", tau51, tau59, optimize=True)

    tau337 = zeros((N, N, M, M))

    tau337 += np.einsum("ijab->ijab", tau336, optimize=True)

    tau368 = zeros((N, N, M, M))

    tau368 += np.einsum("ikac,jkbc->ijab", tau336, u[o, o, v, v], optimize=True)

    tau336 = None

    r2 += 23 * np.einsum("jiab->abij", tau368, optimize=True) / 18

    r2 -= 19 * np.einsum("jiba->abij", tau368, optimize=True) / 9

    tau368 = None

    tau347 = zeros((N, N, M, M))

    tau347 += np.einsum("klab,iljk->ijab", tau51, tau59, optimize=True)

    tau349 = zeros((N, N, M, M))

    tau349 += np.einsum("ijab->ijab", tau347, optimize=True)

    tau365 = zeros((N, N, M, M))

    tau365 += np.einsum("ikac,jkbc->ijab", tau347, u[o, o, v, v], optimize=True)

    tau347 = None

    r2 -= 19 * np.einsum("ijab->abij", tau365, optimize=True) / 9

    r2 += 23 * np.einsum("ijba->abij", tau365, optimize=True) / 18

    tau365 = None

    tau366 = zeros((N, N, M, M))

    tau366 += np.einsum("klab,ilkj->ijab", tau21, tau59, optimize=True)

    tau367 = zeros((N, N, M, M))

    tau367 += np.einsum("ikac,jkbc->ijab", tau366, u[o, o, v, v], optimize=True)

    tau366 = None

    r2 -= 19 * np.einsum("ijab->abij", tau367, optimize=True) / 9

    r2 += 10 * np.einsum("ijba->abij", tau367, optimize=True) / 9

    tau367 = None

    tau369 = zeros((N, N, M, M))

    tau369 += np.einsum("klab,lijk->ijab", tau21, tau59, optimize=True)

    tau370 = zeros((N, N, M, M))

    tau370 += np.einsum("ikac,jkbc->ijab", tau369, u[o, o, v, v], optimize=True)

    tau369 = None

    r2 += 10 * np.einsum("jiab->abij", tau370, optimize=True) / 9

    r2 -= 19 * np.einsum("jiba->abij", tau370, optimize=True) / 9

    tau370 = None

    tau373 = zeros((N, N, M, M))

    tau373 += np.einsum("klab,iljk->ijab", tau21, tau59, optimize=True)

    tau375 = zeros((N, N, M, M))

    tau375 += np.einsum("ijab->ijab", tau373, optimize=True)

    tau490 += np.einsum("ijab->ijab", tau373, optimize=True)

    tau373 = None

    tau374 = zeros((N, N, M, M))

    tau374 += np.einsum("klab,ilkj->ijab", tau51, tau59, optimize=True)

    tau375 += np.einsum("ijab->ijab", tau374, optimize=True)

    tau374 = None

    tau376 = zeros((N, N, M, M))

    tau376 += np.einsum("ikac,kjcb->ijab", tau375, u[o, o, v, v], optimize=True)

    tau375 = None

    r2 -= 8 * np.einsum("ijab->abij", tau376, optimize=True) / 9

    r2 += 5 * np.einsum("ijba->abij", tau376, optimize=True) / 9

    tau376 = None

    tau377 = zeros((N, N, M, M))

    tau377 += np.einsum("klab,lijk->ijab", tau51, tau59, optimize=True)

    tau379 = zeros((N, N, M, M))

    tau379 += np.einsum("ijab->ijab", tau377, optimize=True)

    tau377 = None

    tau378 = zeros((N, N, M, M))

    tau378 += np.einsum("klab,likj->ijab", tau21, tau59, optimize=True)

    tau379 += np.einsum("ijab->ijab", tau378, optimize=True)

    tau380 = zeros((N, N, M, M))

    tau380 += np.einsum("ikac,kjcb->ijab", tau379, u[o, o, v, v], optimize=True)

    tau379 = None

    r2 += 5 * np.einsum("jiab->abij", tau380, optimize=True) / 9

    r2 -= 8 * np.einsum("jiba->abij", tau380, optimize=True) / 9

    tau380 = None

    tau493 += np.einsum("ijab->ijab", tau378, optimize=True)

    tau378 = None

    tau429 = zeros((N, N, M, M))

    tau429 += np.einsum("klab,kilj->ijab", tau39, tau59, optimize=True)

    tau430 -= 36 * np.einsum("jiba->ijab", tau429, optimize=True)

    tau478 += np.einsum("ijba->ijab", tau429, optimize=True)

    tau429 = None

    tau431 = zeros((N, N, N, N))

    tau431 += np.einsum("imjn,knml->ijkl", tau59, u[o, o, o, o], optimize=True)

    tau433 = zeros((N, N, N, N))

    tau433 += np.einsum("ijkl->ijkl", tau431, optimize=True)

    tau431 = None

    tau434 = zeros((N, N, N, N))

    tau434 += np.einsum("imnj,knml->ijkl", tau59, u[o, o, o, o], optimize=True)

    tau436 += np.einsum("ijkl->ijkl", tau434, optimize=True)

    tau434 = None

    tau469 += np.einsum("ijkl->ijkl", tau436, optimize=True)

    tau469 += 2 * np.einsum("ilkj->ijkl", tau436, optimize=True)

    tau436 = None

    tau437 = zeros((N, N, N, N))

    tau437 += np.einsum("mijn,knml->ijkl", tau59, u[o, o, o, o], optimize=True)

    tau439 = zeros((N, N, N, N))

    tau439 += np.einsum("ijkl->ijkl", tau437, optimize=True)

    tau437 = None

    tau440 = zeros((N, N, N, N))

    tau440 += np.einsum("minj,knml->ijkl", tau59, u[o, o, o, o], optimize=True)

    tau442 += np.einsum("ijkl->ijkl", tau440, optimize=True)

    tau440 = None

    tau469 += np.einsum("kjil->ijkl", tau442, optimize=True)

    tau469 += 2 * np.einsum("klij->ijkl", tau442, optimize=True)

    tau442 = None

    tau473 = zeros((N, N, M, M))

    tau473 += np.einsum("klba,ikjl->ijab", tau39, tau59, optimize=True)

    tau504 = zeros((N, N, M, M))

    tau504 += np.einsum("klab,likj->ijab", tau15, tau59, optimize=True)

    tau505 += np.einsum("klab,iljk->ijab", tau15, tau59, optimize=True)

    tau507 = zeros((N, N, N, N))

    tau507 -= 3 * np.einsum("ml,ijkm->ijkl", tau29, tau59, optimize=True)

    tau513 = zeros((N, N, N, N))

    tau513 += np.einsum("ijkl->ijkl", tau59, optimize=True)

    tau513 += 2 * np.einsum("ijlk->ijkl", tau59, optimize=True)

    tau514 += np.einsum("imnl,njmk->ijkl", tau59, tau59, optimize=True)

    tau514 -= 3 * np.einsum("mk,ijml->ijkl", tau29, tau59, optimize=True)

    tau514 += np.einsum("imln,njkm->ijkl", tau513, tau59, optimize=True)

    tau513 = None

    tau62 = zeros((N, N, M, M))

    tau62 += 12 * np.einsum("abij->ijab", l2, optimize=True)

    tau62 += np.einsum("abji->ijab", l2, optimize=True)

    tau63 = zeros((N, N, M, M))

    tau63 += np.einsum("cbkj,ikac->ijab", t2, tau62, optimize=True)

    tau64 += 2 * np.einsum("ijab->ijab", tau63, optimize=True)

    tau65 = zeros((N, N, M, M))

    tau65 += np.einsum("cbkj,kica->ijab", t2, tau64, optimize=True)

    tau64 = None

    tau67 += np.einsum("ijba->ijab", tau65, optimize=True)

    tau65 = None

    tau68 = zeros((M, M))

    tau68 += np.einsum("ijac,ijcb->ab", tau67, u[o, o, v, v], optimize=True)

    tau67 = None

    tau90 += np.einsum("ab->ab", tau68, optimize=True)

    tau68 = None

    tau70 = zeros((N, N, M, M))

    tau70 += np.einsum("cbkj,kica->ijab", t2, tau63, optimize=True)

    tau63 = None

    tau76 += np.einsum("jiab->ijab", tau70, optimize=True)

    tau70 = None

    tau79 = zeros((N, N, M, M))

    tau79 += np.einsum("ikac,jcbk->ijab", tau62, u[o, v, v, o], optimize=True)

    tau62 = None

    tau80 = zeros((N, N, M, M))

    tau80 += np.einsum("ijab->ijab", tau79, optimize=True)

    tau79 = None

    tau71 = zeros((N, N, M, M))

    tau71 += np.einsum("abij->ijab", t2, optimize=True)

    tau71 += 12 * np.einsum("abji->ijab", t2, optimize=True)

    tau72 = zeros((M, M))

    tau72 += np.einsum("cbji,ijca->ab", l2, tau71, optimize=True)

    tau71 = None

    tau73 = zeros((N, N, M, M))

    tau73 += np.einsum("ac,cbij->ijab", tau72, t2, optimize=True)

    tau72 = None

    tau76 += np.einsum("jiba->ijab", tau73, optimize=True)

    tau73 = None

    tau74 = zeros((N, N, M, M))

    tau74 -= np.einsum("abij->ijab", t2, optimize=True)

    tau74 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau75 = zeros((N, N, M, M))

    tau75 += np.einsum("kj,kiab->ijab", tau29, tau74, optimize=True)

    tau76 += 6 * np.einsum("ijab->ijab", tau75, optimize=True)

    tau75 = None

    tau77 = zeros((M, M))

    tau77 += np.einsum("ijac,ijbc->ab", tau76, u[o, o, v, v], optimize=True)

    tau76 = None

    tau90 -= np.einsum("ab->ab", tau77, optimize=True)

    tau77 = None

    tau472 = zeros((N, N, M, M))

    tau472 += np.einsum("bcjk,ikca->ijab", l2, tau74, optimize=True)

    tau506 += 3 * np.einsum("caki,jkbc->ijab", t2, tau472, optimize=True)

    tau507 += np.einsum("abij,klab->ijkl", l2, tau506, optimize=True)

    tau506 = None

    tau502 = zeros((N, N, M, M))

    tau502 += np.einsum("caki,jkcb->ijab", l2, tau74, optimize=True)

    tau78 = zeros((N, N, M, M))

    tau78 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau80 -= 6 * np.einsum("ijab->ijab", tau78, optimize=True)

    tau78 = None

    tau81 = zeros((M, M))

    tau81 += np.einsum("cbij,ijca->ab", t2, tau80, optimize=True)

    tau80 = None

    tau90 += np.einsum("ba->ab", tau81, optimize=True)

    tau81 = None

    tau82 = zeros((N, N, M, M))

    tau82 += np.einsum("acik,jcbk->ijab", l2, u[o, v, v, o], optimize=True)

    tau84 = zeros((N, N, M, M))

    tau84 += np.einsum("ijab->ijab", tau82, optimize=True)

    tau82 = None

    tau83 = zeros((N, N, M, M))

    tau83 += np.einsum("acki,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau84 += np.einsum("ijab->ijab", tau83, optimize=True)

    tau83 = None

    tau85 = zeros((M, M))

    tau85 += np.einsum("cbji,ijca->ab", t2, tau84, optimize=True)

    tau90 -= 6 * np.einsum("ba->ab", tau85, optimize=True)

    tau85 = None

    tau523 -= np.einsum("abjk,kiab->ij", t2, tau84, optimize=True)

    tau84 = None

    tau86 = zeros((M, M, M, M))

    tau86 += 2 * np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau86 -= np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau87 = zeros((M, M))

    tau87 += np.einsum("cd,cabd->ab", tau42, tau86, optimize=True)

    tau86 = None

    tau90 -= 6 * np.einsum("ab->ab", tau87, optimize=True)

    tau87 = None

    tau88 = zeros((N, N, M, M))

    tau88 -= np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau88 += 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau89 = zeros((M, M))

    tau89 += np.einsum("ji,ijab->ab", tau29, tau88, optimize=True)

    tau90 += 6 * np.einsum("ab->ab", tau89, optimize=True)

    tau89 = None

    tau91 = zeros((N, N, M, M))

    tau91 += np.einsum("ca,bcij->ijab", tau90, l2, optimize=True)

    tau90 = None

    tau92 += np.einsum("ijba->ijab", tau91, optimize=True)

    tau91 = None

    r2 -= np.einsum("ijab->abij", tau92, optimize=True) / 3

    r2 -= np.einsum("jiba->abij", tau92, optimize=True) / 3

    tau92 = None

    tau520 = zeros((N, N))

    tau520 += np.einsum("ab,ijab->ij", tau42, tau88, optimize=True)

    tau88 = None

    tau42 = None

    tau523 += np.einsum("ij->ij", tau520, optimize=True)

    tau526 = zeros((N, N))

    tau526 += np.einsum("ji->ij", tau520, optimize=True)

    tau520 = None

    tau96 = zeros((N, N, M, M))

    tau96 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau97 += 18 * np.einsum("ijab->ijab", tau96, optimize=True)

    tau98 = zeros((N, N, M, M))

    tau98 += np.einsum("cbkj,kica->ijab", t2, tau97, optimize=True)

    tau97 = None

    tau99 -= 2 * np.einsum("ijba->ijab", tau98, optimize=True)

    tau98 = None

    tau100 = zeros((N, N, M, M))

    tau100 += np.einsum("ikac,kjcb->ijab", tau99, u[o, o, v, v], optimize=True)

    tau99 = None

    tau108 += np.einsum("ijba->ijab", tau100, optimize=True)

    tau100 = None

    tau158 = zeros((N, N, M, M))

    tau158 -= 18 * np.einsum("ijab->ijab", tau96, optimize=True)

    tau188 = zeros((N, N, M, M))

    tau188 += np.einsum("ijcd,cadb->ijab", tau96, u[v, v, v, v], optimize=True)

    tau191 = zeros((N, N, M, M))

    tau191 += np.einsum("ijab->ijab", tau188, optimize=True)

    tau188 = None

    tau333 = zeros((N, N, M, M))

    tau333 += np.einsum("kjac,ikcb->ijab", tau15, tau96, optimize=True)

    tau334 = zeros((N, N, M, M))

    tau334 += np.einsum("ikac,jkbc->ijab", tau333, u[o, o, v, v], optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", tau334, optimize=True)

    r2 += 28 * np.einsum("jiba->abij", tau334, optimize=True) / 9

    tau334 = None

    tau359 = zeros((N, N, M, M))

    tau359 += np.einsum("ikac,jkcb->ijab", tau333, u[o, o, v, v], optimize=True)

    tau333 = None

    tau360 += np.einsum("ijab->ijab", tau359, optimize=True)

    tau359 = None

    r2 += 4 * np.einsum("jiab->abij", tau360, optimize=True)

    r2 -= 2 * np.einsum("jiba->abij", tau360, optimize=True)

    tau360 = None

    tau467 += np.einsum("ijab->ijab", tau96, optimize=True)

    tau493 -= 3 * np.einsum("kjac,ikcb->ijab", tau51, tau96, optimize=True)

    tau504 += np.einsum("kjac,ikcb->ijab", tau21, tau96, optimize=True)

    tau105 = zeros((N, N, M, M))

    tau105 += np.einsum("acki,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau106 = zeros((N, N, M, M))

    tau106 += np.einsum("ijab->ijab", tau105, optimize=True)

    tau446 = zeros((N, N, M, M))

    tau446 += np.einsum("ijab->ijab", tau105, optimize=True)

    tau105 = None

    tau106 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau107 = zeros((N, N, M, M))

    tau107 += np.einsum("bc,ijac->ijab", tau1, tau106, optimize=True)

    tau1 = None

    tau108 += 18 * np.einsum("ijba->ijab", tau107, optimize=True)

    tau107 = None

    tau109 = zeros((N, N, M, M))

    tau109 += np.einsum("bcjk,kiac->ijab", l2, tau108, optimize=True)

    tau108 = None

    r2 += 2 * np.einsum("jiba->abij", tau109, optimize=True) / 9

    r2 -= np.einsum("jiab->abij", tau109, optimize=True) / 9

    r2 -= np.einsum("ijba->abij", tau109, optimize=True) / 9

    r2 += 2 * np.einsum("ijab->abij", tau109, optimize=True) / 9

    tau109 = None

    tau123 = zeros((N, N, M, M))

    tau123 += np.einsum("jk,ikab->ijab", tau29, tau106, optimize=True)

    tau124 += np.einsum("jiab->ijab", tau123, optimize=True)

    tau123 = None

    tau524 -= np.einsum("acik,kjcb->ijab", l2, tau106, optimize=True)

    tau526 -= np.einsum("abik,kjab->ij", t2, tau524, optimize=True)

    tau524 = None

    tau110 = zeros((N, N, M, M))

    tau110 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau111 = zeros((N, N, N, N))

    tau111 += np.einsum("jkba,ilab->ijkl", tau110, tau15, optimize=True)

    tau112 = zeros((N, N, M, M))

    tau112 += np.einsum("ijkl,lkba->ijab", tau111, u[o, o, v, v], optimize=True)

    tau111 = None

    tau139 = zeros((N, N, M, M))

    tau139 += 36 * np.einsum("ijab->ijab", tau112, optimize=True)

    tau112 = None

    tau122 = zeros((N, N, M, M))

    tau122 += np.einsum("jkcb,ikca->ijab", tau110, tau39, optimize=True)

    tau124 += np.einsum("jiba->ijab", tau122, optimize=True)

    tau122 = None

    tau153 = zeros((N, N, M, M))

    tau153 += np.einsum("ijdc,acdb->ijab", tau110, tau40, optimize=True)

    tau163 += 36 * np.einsum("ijab->ijab", tau153, optimize=True)

    tau153 = None

    tau190 = zeros((N, N, M, M))

    tau190 += np.einsum("jkcb,ikca->ijab", tau110, tau34, optimize=True)

    tau191 -= np.einsum("jiba->ijab", tau190, optimize=True)

    tau190 = None

    tau419 = zeros((N, N, M, M))

    tau419 -= 28 * np.einsum("ijab->ijab", tau110, optimize=True)

    tau492 = zeros((N, N, M, M))

    tau492 -= 3 * np.einsum("ijab->ijab", tau110, optimize=True)

    tau113 = zeros((N, N))

    tau113 += np.einsum("abik,abjk->ij", l2, t2, optimize=True)

    tau114 = zeros((N, N, M, M))

    tau114 += np.einsum("ik,jkab->ijab", tau113, tau30, optimize=True)

    tau30 = None

    tau139 += 36 * np.einsum("ijab->ijab", tau114, optimize=True)

    tau114 = None

    tau143 = zeros((N, N, M, M))

    tau143 += np.einsum("ik,jkab->ijab", tau113, u[o, o, v, v], optimize=True)

    tau145 -= 2 * np.einsum("ijab->ijab", tau143, optimize=True)

    tau145 += np.einsum("ijba->ijab", tau143, optimize=True)

    tau146 = zeros((N, N, M, M))

    tau146 += np.einsum("bcjk,ikca->ijab", t2, tau145, optimize=True)

    tau145 = None

    tau150 += np.einsum("ijba->ijab", tau146, optimize=True)

    tau146 = None

    tau397 -= 36 * np.einsum("ijba->ijab", tau143, optimize=True)

    tau470 += np.einsum("ijba->ijab", tau143, optimize=True)

    tau473 += np.einsum("bckj,ikca->ijab", t2, tau470, optimize=True)

    tau470 = None

    tau494 += np.einsum("ijab->ijab", tau143, optimize=True)

    tau494 -= 2 * np.einsum("ijba->ijab", tau143, optimize=True)

    tau143 = None

    tau495 += np.einsum("acjk,ikbc->ijab", t2, tau494, optimize=True)

    tau494 = None

    tau149 = zeros((N, N, M, M))

    tau149 += np.einsum("jk,ikab->ijab", tau113, tau106, optimize=True)

    tau106 = None

    tau150 += np.einsum("jiab->ijab", tau149, optimize=True)

    tau495 += np.einsum("jiab->ijab", tau149, optimize=True)

    tau149 = None

    r2 += np.einsum("bckj,ikca->abij", l2, tau495, optimize=True) / 3

    tau495 = None

    tau329 = zeros((N, N, M, M))

    tau329 += np.einsum("ik,kjab->ijab", tau113, tau2, optimize=True)

    tau332 += np.einsum("ijab->ijab", tau329, optimize=True)

    tau329 = None

    r2 -= 2 * np.einsum("ijab->abij", tau332, optimize=True)

    r2 += 4 * np.einsum("ijba->abij", tau332, optimize=True)

    tau332 = None

    tau351 = zeros((N, N, M, M))

    tau351 += np.einsum("ik,jkab->ijab", tau113, tau2, optimize=True)

    tau2 = None

    tau357 = zeros((N, N, M, M))

    tau357 -= 3 * np.einsum("ijab->ijab", tau351, optimize=True)

    tau351 = None

    tau392 = zeros((N, N, M, M))

    tau392 += np.einsum("ik,kajb->ijab", tau113, u[o, v, o, v], optimize=True)

    tau408 += 36 * np.einsum("ijba->ijab", tau392, optimize=True)

    tau473 -= np.einsum("ijba->ijab", tau392, optimize=True)

    tau392 = None

    tau119 = zeros((M, M, M, M))

    tau119 += np.einsum("abij,jidc->abcd", t2, u[o, o, v, v], optimize=True)

    tau120 = zeros((M, M, M, M))

    tau120 += np.einsum("badc->abcd", tau119, optimize=True)

    tau189 = zeros((N, N, M, M))

    tau189 += np.einsum("acbd,ijcd->ijab", tau119, tau51, optimize=True)

    tau191 += np.einsum("ijab->ijab", tau189, optimize=True)

    tau189 = None

    tau192 = zeros((N, N, M, M))

    tau192 += np.einsum("bcjk,ikca->ijab", l2, tau191, optimize=True)

    tau191 = None

    r2 -= 2 * np.einsum("jiba->abij", tau192, optimize=True)

    r2 += 2 * np.einsum("jiab->abij", tau192, optimize=True) / 9

    tau192 = None

    tau261 = zeros((N, N, M, M))

    tau261 += np.einsum("acdb,ijcd->ijab", tau119, tau21, optimize=True)

    tau264 += np.einsum("ijab->ijab", tau261, optimize=True)

    tau261 = None

    tau277 = zeros((M, M, M, M))

    tau277 += np.einsum("ecdf,aebf->abcd", tau119, tau40, optimize=True)

    tau279 += np.einsum("acbd->abcd", tau277, optimize=True)

    tau277 = None

    tau314 = zeros((N, N, M, M))

    tau314 += np.einsum("acdb,ijcd->ijab", tau119, tau51, optimize=True)

    tau316 += np.einsum("ijab->ijab", tau314, optimize=True)

    tau314 = None

    tau319 = zeros((M, M, M, M))

    tau319 += np.einsum("ecfd,aebf->abcd", tau119, tau40, optimize=True)

    tau321 += np.einsum("acbd->abcd", tau319, optimize=True)

    tau319 = None

    tau322 = zeros((N, N, M, M))

    tau322 += np.einsum("cdij,acdb->ijab", l2, tau321, optimize=True)

    tau323 += np.einsum("ijab->ijab", tau322, optimize=True)

    tau322 = None

    r2 += np.einsum("ijab->abij", tau323, optimize=True) / 6

    r2 -= 5 * np.einsum("ijba->abij", tau323, optimize=True) / 3

    tau323 = None

    tau325 = zeros((N, N, M, M))

    tau325 += np.einsum("dcij,acdb->ijab", l2, tau321, optimize=True)

    tau321 = None

    tau326 += np.einsum("ijab->ijab", tau325, optimize=True)

    tau325 = None

    r2 -= 5 * np.einsum("ijab->abij", tau326, optimize=True) / 3

    r2 += np.einsum("ijba->abij", tau326, optimize=True) / 6

    tau326 = None

    tau398 = zeros((M, M, M, M))

    tau398 += 28 * np.einsum("bacd->abcd", tau119, optimize=True)

    tau398 -= 19 * np.einsum("badc->abcd", tau119, optimize=True)

    tau401 = zeros((M, M, M, M))

    tau401 += 4 * np.einsum("abcd->abcd", tau119, optimize=True)

    tau473 += np.einsum("bcad,ijcd->ijab", tau119, tau15, optimize=True)

    tau478 += np.einsum("bcad,ijcd->ijab", tau119, tau96, optimize=True)

    tau120 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau121 = zeros((N, N, M, M))

    tau121 += np.einsum("cabd,ijcd->ijab", tau120, tau96, optimize=True)

    tau124 += np.einsum("ijab->ijab", tau121, optimize=True)

    tau121 = None

    tau125 = zeros((N, N, M, M))

    tau125 += np.einsum("bcjk,ikca->ijab", l2, tau124, optimize=True)

    tau124 = None

    tau139 += 36 * np.einsum("jiba->ijab", tau125, optimize=True)

    tau125 = None

    tau147 = zeros((N, N, M, M))

    tau147 += np.einsum("cabd,ijcd->ijab", tau120, tau15, optimize=True)

    tau120 = None

    tau150 += np.einsum("ijab->ijab", tau147, optimize=True)

    tau147 = None

    tau151 = zeros((N, N, M, M))

    tau151 += np.einsum("bcjk,ikca->ijab", l2, tau150, optimize=True)

    tau150 = None

    tau165 += 36 * np.einsum("jiba->ijab", tau151, optimize=True)

    tau151 = None

    tau129 = zeros((N, N, M, M))

    tau129 -= 23 * np.einsum("abij->ijab", t2, optimize=True)

    tau129 += 36 * np.einsum("abji->ijab", t2, optimize=True)

    tau130 = zeros((N, N, M, M))

    tau130 += np.einsum("bcjk,ikca->ijab", l2, tau129, optimize=True)

    tau129 = None

    tau131 -= np.einsum("jiba->ijab", tau130, optimize=True)

    tau130 = None

    tau132 = zeros((N, N, M, M))

    tau132 += np.einsum("caik,cbjk->ijab", l2, t2, optimize=True)

    tau133 = zeros((N, N, M, M))

    tau133 += np.einsum("kiac,jkcb->ijab", tau131, tau132, optimize=True)

    tau131 = None

    tau137 -= np.einsum("jiab->ijab", tau133, optimize=True)

    tau133 = None

    tau245 = zeros((N, N, M, M))

    tau245 += np.einsum("acbd,ijcd->ijab", tau119, tau132, optimize=True)

    tau247 += np.einsum("ijab->ijab", tau245, optimize=True)

    tau245 = None

    tau248 = zeros((N, N, M, M))

    tau248 += np.einsum("bcjk,ikca->ijab", l2, tau247, optimize=True)

    tau247 = None

    r2 += 2 * np.einsum("ijba->abij", tau248, optimize=True) / 9

    r2 -= 2 * np.einsum("ijab->abij", tau248, optimize=True)

    tau248 = None

    tau281 = zeros((N, N, M, M))

    tau281 += np.einsum("ikcb,kjac->ijab", tau132, tau21, optimize=True)

    tau283 += np.einsum("ijab->ijab", tau281, optimize=True)

    tau281 = None

    tau284 = zeros((N, N, M, M))

    tau284 += np.einsum("ikac,kjbc->ijab", tau283, u[o, o, v, v], optimize=True)

    tau283 = None

    tau287 += 6 * np.einsum("ijab->ijab", tau284, optimize=True)

    tau284 = None

    tau294 = zeros((N, N, M, M))

    tau294 += np.einsum("ikcb,kjac->ijab", tau132, tau51, optimize=True)

    tau295 += np.einsum("ijab->ijab", tau294, optimize=True)

    tau294 = None

    tau296 = zeros((N, N, M, M))

    tau296 += np.einsum("ikac,kjbc->ijab", tau295, u[o, o, v, v], optimize=True)

    tau295 = None

    tau299 += 6 * np.einsum("ijab->ijab", tau296, optimize=True)

    tau296 = None

    tau300 = zeros((N, N, M, M))

    tau300 += np.einsum("ijcd,cabd->ijab", tau132, u[v, v, v, v], optimize=True)

    tau303 += np.einsum("ijab->ijab", tau300, optimize=True)

    tau300 = None

    tau304 = zeros((N, N, M, M))

    tau304 += np.einsum("bcjk,ikca->ijab", l2, tau303, optimize=True)

    tau303 = None

    r2 += 5 * np.einsum("ijba->abij", tau304, optimize=True) / 9

    r2 += 2 * np.einsum("ijab->abij", tau304, optimize=True) / 9

    tau304 = None

    tau309 = zeros((N, N, M, M))

    tau309 += np.einsum("acdb,ijcd->ijab", tau119, tau132, optimize=True)

    tau311 += np.einsum("ijab->ijab", tau309, optimize=True)

    tau309 = None

    tau312 = zeros((N, N, M, M))

    tau312 += np.einsum("bcjk,ikca->ijab", l2, tau311, optimize=True)

    tau311 = None

    r2 += 4 * np.einsum("ijba->abij", tau312, optimize=True) / 9

    r2 += np.einsum("ijab->abij", tau312, optimize=True) / 6

    tau312 = None

    tau432 = zeros((N, N, N, N))

    tau432 += np.einsum("ijab,kabl->ijkl", tau132, u[o, v, v, o], optimize=True)

    tau433 -= np.einsum("ijkl->ijkl", tau432, optimize=True)

    tau432 = None

    tau469 += 2 * np.einsum("ijkl->ijkl", tau433, optimize=True)

    tau469 += np.einsum("ilkj->ijkl", tau433, optimize=True)

    tau433 = None

    tau465 += np.einsum("ijab->ijab", tau132, optimize=True)

    tau469 -= 3 * np.einsum("lkab,ijab->ijkl", tau39, tau465, optimize=True)

    tau465 = None

    tau473 += np.einsum("ijcd,cbda->ijab", tau132, u[v, v, v, v], optimize=True)

    tau490 -= 3 * np.einsum("ikcb,kjac->ijab", tau132, tau15, optimize=True)

    tau507 -= 3 * np.einsum("jlba,ikab->ijkl", tau110, tau132, optimize=True)

    tau134 = zeros((N, N, M, M))

    tau134 += 9 * np.einsum("abij->ijab", t2, optimize=True)

    tau134 -= 5 * np.einsum("abji->ijab", t2, optimize=True)

    tau135 = zeros((N, N, M, M))

    tau135 += np.einsum("bcjk,kica->ijab", l2, tau134, optimize=True)

    tau134 = None

    tau136 = zeros((N, N, M, M))

    tau136 += np.einsum("kiac,kjbc->ijab", tau135, tau51, optimize=True)

    tau135 = None

    tau137 += 4 * np.einsum("ijba->ijab", tau136, optimize=True)

    tau136 = None

    tau138 = zeros((N, N, M, M))

    tau138 += np.einsum("ikac,kjcb->ijab", tau137, u[o, o, v, v], optimize=True)

    tau137 = None

    tau139 += np.einsum("ijab->ijab", tau138, optimize=True)

    tau138 = None

    r2 += np.einsum("ijab->abij", tau139, optimize=True) / 9

    r2 -= np.einsum("ijba->abij", tau139, optimize=True) / 18

    tau139 = None

    tau154 = zeros((N, N, M, M))

    tau154 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau155 = zeros((N, N, M, M))

    tau155 += np.einsum("ikcb,kjac->ijab", tau154, tau21, optimize=True)

    tau163 -= 10 * np.einsum("ijab->ijab", tau155, optimize=True)

    tau390 += np.einsum("ijab->ijab", tau155, optimize=True)

    tau155 = None

    tau391 = zeros((N, N, M, M))

    tau391 += np.einsum("ikac,kjbc->ijab", tau390, u[o, o, v, v], optimize=True)

    tau390 = None

    r2 += 4 * np.einsum("jiab->abij", tau391, optimize=True) / 3

    r2 += 2 * np.einsum("jiba->abij", tau391, optimize=True) / 3

    tau391 = None

    tau267 = zeros((N, N, M, M))

    tau267 += np.einsum("acdb,ijcd->ijab", tau119, tau154, optimize=True)

    tau269 += np.einsum("ijab->ijab", tau267, optimize=True)

    tau267 = None

    tau288 = zeros((N, N, M, M))

    tau288 += np.einsum("ijcd,cabd->ijab", tau154, u[v, v, v, v], optimize=True)

    tau291 += np.einsum("ijab->ijab", tau288, optimize=True)

    tau288 = None

    tau292 = zeros((N, N, M, M))

    tau292 += np.einsum("bcjk,ikca->ijab", l2, tau291, optimize=True)

    r2 -= 14 * np.einsum("jiba->abij", tau292, optimize=True) / 9

    r2 += 10 * np.einsum("jiab->abij", tau292, optimize=True) / 9

    tau292 = None

    tau306 = zeros((N, N, M, M))

    tau306 += np.einsum("bckj,ikca->ijab", l2, tau291, optimize=True)

    tau291 = None

    tau307 += np.einsum("jiba->ijab", tau306, optimize=True)

    tau306 = None

    r2 += 2 * np.einsum("ijab->abij", tau307, optimize=True) / 9

    r2 += 5 * np.einsum("ijba->abij", tau307, optimize=True) / 9

    tau307 = None

    tau385 = zeros((N, N, M, M))

    tau385 += np.einsum("ikcb,kjac->ijab", tau154, tau51, optimize=True)

    tau387 += np.einsum("ijab->ijab", tau385, optimize=True)

    tau385 = None

    tau388 = zeros((N, N, M, M))

    tau388 += np.einsum("ikac,kjbc->ijab", tau387, u[o, o, v, v], optimize=True)

    tau387 = None

    r2 += 2 * np.einsum("jiab->abij", tau388, optimize=True) / 3

    r2 += 4 * np.einsum("jiba->abij", tau388, optimize=True) / 3

    tau388 = None

    tau438 = zeros((N, N, N, N))

    tau438 += np.einsum("ijab,kabl->ijkl", tau154, u[o, v, v, o], optimize=True)

    tau439 -= np.einsum("ijkl->ijkl", tau438, optimize=True)

    tau438 = None

    tau469 += 2 * np.einsum("kjil->ijkl", tau439, optimize=True)

    tau469 += np.einsum("klij->ijkl", tau439, optimize=True)

    tau439 = None

    tau488 -= np.einsum("jicd,cbda->ijab", tau154, u[v, v, v, v], optimize=True)

    tau500 = zeros((N, N, M, M))

    tau500 += np.einsum("kjac,ikcb->ijab", tau15, tau154, optimize=True)

    tau503 = zeros((N, N, M, M))

    tau503 += 2 * np.einsum("ijab->ijab", tau500, optimize=True)

    tau504 += np.einsum("ijab->ijab", tau500, optimize=True)

    tau500 = None

    tau156 = zeros((N, N, M, M))

    tau156 += np.einsum("abij->ijab", l2, optimize=True)

    tau156 += 2 * np.einsum("baij->ijab", l2, optimize=True)

    tau157 = zeros((N, N, M, M))

    tau157 += np.einsum("cbjk,kiac->ijab", t2, tau156, optimize=True)

    tau158 += 5 * np.einsum("ijab->ijab", tau157, optimize=True)

    tau157 = None

    tau159 = zeros((N, N, M, M))

    tau159 += np.einsum("ikca,kjbc->ijab", tau158, tau51, optimize=True)

    tau158 = None

    tau163 -= 2 * np.einsum("ijba->ijab", tau159, optimize=True)

    tau159 = None

    tau353 = zeros((N, N, M, M))

    tau353 += np.einsum("bckj,ikac->ijab", t2, tau156, optimize=True)

    tau408 += 12 * np.einsum("ikcb,jkca->ijab", tau353, tau39, optimize=True)

    tau417 += 12 * np.einsum("kiac,kjcb->ijab", tau156, tau39, optimize=True)

    tau491 = zeros((N, N, M, M))

    tau491 += np.einsum("bckj,kiac->ijab", t2, tau156, optimize=True)

    tau492 += np.einsum("ijab->ijab", tau491, optimize=True)

    tau491 = None

    tau493 += np.einsum("adcb,ijcd->ijab", tau40, tau492, optimize=True)

    tau507 += np.einsum("ilba,jkab->ijkl", tau21, tau492, optimize=True)

    tau492 = None

    tau498 = zeros((N, N, M, M))

    tau498 += np.einsum("bcjk,ikca->ijab", t2, tau156, optimize=True)

    tau501 = zeros((N, N, M, M))

    tau501 += np.einsum("bcjk,kiac->ijab", t2, tau156, optimize=True)

    tau156 = None

    tau160 = zeros((N, N, M, M))

    tau160 += 36 * np.einsum("abij->ijab", t2, optimize=True)

    tau160 -= 23 * np.einsum("abji->ijab", t2, optimize=True)

    tau161 = zeros((N, N, M, M))

    tau161 += np.einsum("bcjk,kica->ijab", l2, tau160, optimize=True)

    tau160 = None

    tau162 = zeros((N, N, M, M))

    tau162 += np.einsum("ikca,jkcb->ijab", tau161, tau51, optimize=True)

    tau161 = None

    tau163 += np.einsum("jiab->ijab", tau162, optimize=True)

    tau162 = None

    tau164 = zeros((N, N, M, M))

    tau164 += np.einsum("ikac,kjcb->ijab", tau163, u[o, o, v, v], optimize=True)

    tau163 = None

    tau165 += np.einsum("ijab->ijab", tau164, optimize=True)

    tau164 = None

    r2 -= np.einsum("jiab->abij", tau165, optimize=True) / 18

    r2 += np.einsum("jiba->abij", tau165, optimize=True) / 9

    tau165 = None

    tau166 = zeros((M, M, M, M))

    tau166 += np.einsum("abij,cdji->abcd", l2, t2, optimize=True)

    tau167 = zeros((N, N, M, M))

    tau167 += np.einsum("acbd,ijcd->ijab", tau166, tau34, optimize=True)

    tau175 -= 6 * np.einsum("ijab->ijab", tau167, optimize=True)

    tau167 = None

    tau176 = zeros((N, N, M, M))

    tau176 += np.einsum("bcjk,kiac->ijab", l2, tau175, optimize=True)

    tau175 = None

    r2 -= np.einsum("jiba->abij", tau176, optimize=True) / 3

    r2 += 2 * np.einsum("jiab->abij", tau176, optimize=True) / 9

    r2 += 2 * np.einsum("ijba->abij", tau176, optimize=True) / 9

    r2 -= np.einsum("ijab->abij", tau176, optimize=True) / 3

    tau176 = None

    tau231 = zeros((M, M, M, M))

    tau231 += np.einsum("bfed,aefc->abcd", tau166, tau40, optimize=True)

    tau232 -= np.einsum("abcd->abcd", tau231, optimize=True)

    tau231 = None

    tau233 = zeros((N, N, M, M))

    tau233 += np.einsum("abcd,ijdc->ijab", tau232, u[o, o, v, v], optimize=True)

    tau232 = None

    tau234 -= 6 * np.einsum("jiab->ijab", tau233, optimize=True)

    tau233 = None

    tau237 = zeros((M, M, M, M))

    tau237 += np.einsum("aefc,bfed->abcd", tau166, tau166, optimize=True)

    tau238 = zeros((N, N, M, M))

    tau238 += np.einsum("badc,ijcd->ijab", tau237, u[o, o, v, v], optimize=True)

    tau237 = None

    tau243 += np.einsum("ijab->ijab", tau238, optimize=True)

    tau238 = None

    r2 += 4 * np.einsum("jiab->abij", tau243, optimize=True) / 3

    r2 -= 2 * np.einsum("jiba->abij", tau243, optimize=True)

    tau243 = None

    tau250 = zeros((M, M, M, M))

    tau250 += np.einsum("ecdf,aebf->abcd", tau119, tau166, optimize=True)

    tau252 += np.einsum("acbd->abcd", tau250, optimize=True)

    tau250 = None

    tau253 = zeros((N, N, M, M))

    tau253 += np.einsum("cdij,acdb->ijab", l2, tau252, optimize=True)

    r2 += 23 * np.einsum("ijab->abij", tau253, optimize=True) / 18

    r2 += 5 * np.einsum("ijba->abij", tau253, optimize=True) / 9

    tau253 = None

    tau259 = zeros((N, N, M, M))

    tau259 += np.einsum("dcij,acdb->ijab", l2, tau252, optimize=True)

    tau252 = None

    r2 += 5 * np.einsum("ijab->abij", tau259, optimize=True) / 9

    r2 += 23 * np.einsum("ijba->abij", tau259, optimize=True) / 18

    tau259 = None

    tau276 = zeros((M, M, M, M))

    tau276 += np.einsum("aebf,cefd->abcd", tau166, u[v, v, v, v], optimize=True)

    tau279 += np.einsum("abcd->abcd", tau276, optimize=True)

    tau276 = None

    tau280 = zeros((N, N, M, M))

    tau280 += np.einsum("dcij,acdb->ijab", l2, tau279, optimize=True)

    tau287 += 5 * np.einsum("ijab->ijab", tau280, optimize=True)

    tau280 = None

    r2 += 2 * np.einsum("ijab->abij", tau287, optimize=True) / 9

    r2 += np.einsum("ijba->abij", tau287, optimize=True) / 9

    tau287 = None

    tau293 = zeros((N, N, M, M))

    tau293 += np.einsum("cdij,acdb->ijab", l2, tau279, optimize=True)

    tau279 = None

    tau299 += 5 * np.einsum("ijab->ijab", tau293, optimize=True)

    tau293 = None

    r2 += np.einsum("ijab->abij", tau299, optimize=True) / 9

    r2 += 2 * np.einsum("ijba->abij", tau299, optimize=True) / 9

    tau299 = None

    tau354 = zeros((N, N, M, M))

    tau354 += np.einsum("adcb,ijcd->ijab", tau166, tau353, optimize=True)

    tau353 = None

    tau355 = zeros((N, N, M, M))

    tau355 += np.einsum("ijab->ijab", tau354, optimize=True)

    tau354 = None

    tau413 = zeros((M, M, M, M))

    tau413 += np.einsum("aebf,cedf->abcd", tau166, u[v, v, v, v], optimize=True)

    tau415 += np.einsum("dbca->abcd", tau413, optimize=True)

    tau415 += 2 * np.einsum("dcba->abcd", tau413, optimize=True)

    tau413 = None

    r2 -= np.einsum("cdij,acdb->abij", l2, tau415, optimize=True) / 9

    tau415 = None

    tau499 += np.einsum("adcb,ijcd->ijab", tau166, tau498, optimize=True)

    tau498 = None

    tau503 += np.einsum("adcb,ijcd->ijab", tau166, tau501, optimize=True)

    tau501 = None

    tau504 += np.einsum("ijdc,acdb->ijab", tau110, tau166, optimize=True)

    tau110 = None

    r2 -= 2 * np.einsum("jkac,kibc->abij", tau504, u[o, o, v, v], optimize=True)

    tau504 = None

    tau505 += np.einsum("ijdc,acdb->ijab", tau11, tau166, optimize=True)

    r2 -= 2 * np.einsum("ikbc,kjac->abij", tau505, u[o, o, v, v], optimize=True)

    tau505 = None

    tau177 = zeros((N, N, M, M))

    tau177 += np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau178 = zeros((N, N, M, M))

    tau178 += np.einsum("jkcb,ikca->ijab", tau177, tau39, optimize=True)

    tau180 -= np.einsum("jiba->ijab", tau178, optimize=True)

    tau178 = None

    tau181 = zeros((N, N, M, M))

    tau181 += np.einsum("bcjk,ikca->ijab", l2, tau180, optimize=True)

    tau180 = None

    tau187 = zeros((N, N, M, M))

    tau187 -= np.einsum("jiba->ijab", tau181, optimize=True)

    tau181 = None

    tau315 = zeros((N, N, M, M))

    tau315 += np.einsum("jkcb,ikca->ijab", tau177, tau34, optimize=True)

    tau316 -= np.einsum("jiba->ijab", tau315, optimize=True)

    tau315 = None

    tau317 = zeros((N, N, M, M))

    tau317 += np.einsum("bcjk,ikca->ijab", l2, tau316, optimize=True)

    tau316 = None

    r2 += np.einsum("jiba->abij", tau317, optimize=True) / 6

    r2 += 4 * np.einsum("jiab->abij", tau317, optimize=True) / 9

    tau317 = None

    tau340 = zeros((N, N, M, M))

    tau340 += np.einsum("acdb,ijdc->ijab", tau166, tau177, optimize=True)

    tau343 = zeros((N, N, M, M))

    tau343 += np.einsum("ijab->ijab", tau340, optimize=True)

    tau493 += np.einsum("ijab->ijab", tau340, optimize=True)

    tau340 = None

    tau363 = zeros((N, N, M, M))

    tau363 += np.einsum("ijdc,acdb->ijab", tau177, tau40, optimize=True)

    tau364 = zeros((N, N, M, M))

    tau364 += np.einsum("ikac,jkbc->ijab", tau363, u[o, o, v, v], optimize=True)

    tau363 = None

    r2 += 4 * np.einsum("jiab->abij", tau364, optimize=True) / 3

    r2 -= 23 * np.einsum("jiba->abij", tau364, optimize=True) / 9

    tau364 = None

    tau507 -= 3 * np.einsum("ilab,jkba->ijkl", tau15, tau177, optimize=True)

    r2 += 2 * np.einsum("ijkl,lkba->abij", tau507, u[o, o, v, v], optimize=True) / 3

    tau507 = None

    tau514 += np.einsum("ilab,jkba->ijkl", tau132, tau177, optimize=True)

    tau177 = None

    tau182 = zeros((N, N, M, M))

    tau182 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau183 = zeros((N, N, N, N))

    tau183 += np.einsum("ikab,jlba->ijkl", tau132, tau182, optimize=True)

    tau185 += np.einsum("ijkl->ijkl", tau183, optimize=True)

    tau183 = None

    tau186 = zeros((N, N, M, M))

    tau186 += np.einsum("ijkl,lkab->ijab", tau185, u[o, o, v, v], optimize=True)

    tau185 = None

    tau187 += np.einsum("ijba->ijab", tau186, optimize=True)

    tau186 = None

    r2 -= 2 * np.einsum("ijab->abij", tau187, optimize=True)

    r2 += 4 * np.einsum("ijba->abij", tau187, optimize=True) / 3

    tau187 = None

    tau268 = zeros((N, N, M, M))

    tau268 += np.einsum("jkcb,ikca->ijab", tau182, tau34, optimize=True)

    tau269 -= np.einsum("jiba->ijab", tau268, optimize=True)

    tau268 = None

    tau270 = zeros((N, N, M, M))

    tau270 += np.einsum("bcjk,ikca->ijab", l2, tau269, optimize=True)

    tau269 = None

    r2 -= 5 * np.einsum("jiba->abij", tau270, optimize=True) / 3

    r2 += 19 * np.einsum("jiab->abij", tau270, optimize=True) / 18

    tau270 = None

    tau335 = zeros((N, N, M, M))

    tau335 += np.einsum("acdb,ijdc->ijab", tau166, tau182, optimize=True)

    tau337 += np.einsum("ijab->ijab", tau335, optimize=True)

    tau335 = None

    tau338 = zeros((N, N, M, M))

    tau338 += np.einsum("ikac,kjbc->ijab", tau337, u[o, o, v, v], optimize=True)

    tau337 = None

    r2 -= 2 * np.einsum("jiab->abij", tau338, optimize=True)

    r2 += 4 * np.einsum("jiba->abij", tau338, optimize=True) / 3

    tau338 = None

    tau476 += np.einsum("ijab->ijab", tau182, optimize=True)

    tau483 += 6 * np.einsum("klab,ilkj->ijab", tau182, u[o, o, o, o], optimize=True)

    tau488 += 6 * np.einsum("jkcb,ikca->ijab", tau182, tau39, optimize=True)

    tau514 -= 3 * np.einsum("ikab,jlba->ijkl", tau15, tau182, optimize=True)

    r2 += 2 * np.einsum("ijkl,klab->abij", tau514, u[o, o, v, v], optimize=True) / 3

    tau514 = None

    tau193 = zeros((N, N, M, M))

    tau193 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau193 += np.einsum("abji->ijab", t2, optimize=True)

    tau194 = zeros((M, M, M, M))

    tau194 += np.einsum("cdij,ijab->abcd", l2, tau193, optimize=True)

    tau193 = None

    tau195 = zeros((N, N, M, M))

    tau195 += np.einsum("bdac,ijcd->ijab", tau194, tau34, optimize=True)

    tau218 -= 6 * np.einsum("ijab->ijab", tau195, optimize=True)

    tau195 = None

    tau490 += np.einsum("dbac,ijdc->ijab", tau194, tau4, optimize=True)

    tau194 = None

    tau196 = zeros((N, N, M, M))

    tau196 += 2 * np.einsum("abij->ijab", l2, optimize=True)

    tau196 += np.einsum("abji->ijab", l2, optimize=True)

    tau197 = zeros((N, N, M, M))

    tau197 += np.einsum("cbjk,ikac->ijab", t2, tau196, optimize=True)

    tau198 += np.einsum("ijab->ijab", tau197, optimize=True)

    tau199 = zeros((N, N, M, M))

    tau199 += np.einsum("kjbc,kiac->ijab", tau198, tau34, optimize=True)

    tau218 -= 6 * np.einsum("jiba->ijab", tau199, optimize=True)

    tau199 = None

    tau490 += np.einsum("klab,ilkj->ijab", tau198, tau59, optimize=True)

    tau493 += np.einsum("klab,lijk->ijab", tau198, tau59, optimize=True)

    r2 += 2 * np.einsum("jkbc,kiac->abij", tau493, u[o, o, v, v], optimize=True) / 3

    tau493 = None

    tau208 = zeros((N, N, M, M))

    tau208 += np.einsum("cbjk,kica->ijab", t2, tau197, optimize=True)

    tau211 += np.einsum("jiab->ijab", tau208, optimize=True)

    tau208 = None

    tau216 = zeros((N, N, M, M))

    tau216 += np.einsum("cbkj,kica->ijab", t2, tau197, optimize=True)

    tau197 = None

    tau217 = zeros((N, N, M, M))

    tau217 += np.einsum("ikca,kjcb->ijab", tau216, u[o, o, v, v], optimize=True)

    tau216 = None

    tau218 += np.einsum("ijba->ijab", tau217, optimize=True)

    tau217 = None

    tau482 += np.einsum("bckj,ikac->ijab", t2, tau196, optimize=True)

    tau196 = None

    tau200 = zeros((N, N, M, M))

    tau200 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau200 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau201 = zeros((N, N, M, M))

    tau201 += np.einsum("bcjk,kica->ijab", t2, tau200, optimize=True)

    tau204 = zeros((N, N, M, M))

    tau204 += 3 * np.einsum("jiba->ijab", tau201, optimize=True)

    tau405 = zeros((N, N, M, M))

    tau405 += 23 * np.einsum("jiba->ijab", tau201, optimize=True)

    tau201 = None

    tau328 = zeros((N, N, M, M))

    tau328 += np.einsum("kica,jkbc->ijab", tau200, tau327, optimize=True)

    tau327 = None

    r2 -= 4 * np.einsum("jiba->abij", tau328, optimize=True)

    r2 += 2 * np.einsum("jiab->abij", tau328, optimize=True)

    r2 += 2 * np.einsum("ijba->abij", tau328, optimize=True)

    r2 -= 25 * np.einsum("ijab->abij", tau328, optimize=True) / 6

    tau328 = None

    tau526 -= np.einsum("kjab,ikba->ij", tau200, tau45, optimize=True)

    tau200 = None

    tau45 = None

    tau202 = zeros((N, N, M, M))

    tau202 -= 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau202 += 3 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau203 = zeros((N, N, M, M))

    tau203 += np.einsum("bckj,kica->ijab", t2, tau202, optimize=True)

    tau202 = None

    tau204 -= np.einsum("jiba->ijab", tau203, optimize=True)

    tau203 = None

    tau204 += 3 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau204 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau205 = zeros((N, N, M, M))

    tau205 += np.einsum("bcjk,kica->ijab", l2, tau204, optimize=True)

    tau204 = None

    tau206 -= 6 * np.einsum("jiba->ijab", tau205, optimize=True)

    tau205 = None

    tau207 = zeros((N, N, M, M))

    tau207 += np.einsum("bckj,kiac->ijab", t2, tau206, optimize=True)

    tau206 = None

    tau218 -= np.einsum("jiab->ijab", tau207, optimize=True)

    tau207 = None

    tau209 = zeros((N, N, M, M))

    tau209 += np.einsum("abij->ijab", t2, optimize=True)

    tau209 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau210 = zeros((N, N, M, M))

    tau210 += np.einsum("klab,klij->ijab", tau209, tau59, optimize=True)

    tau211 += np.einsum("ijab->ijab", tau210, optimize=True)

    tau210 = None

    tau212 = zeros((N, N, M, M))

    tau212 += np.einsum("ikac,kjbc->ijab", tau211, u[o, o, v, v], optimize=True)

    tau211 = None

    tau218 += 6 * np.einsum("ijba->ijab", tau212, optimize=True)

    tau212 = None

    tau341 = zeros((M, M, M, M))

    tau341 += np.einsum("cdij,ijab->abcd", l2, tau209, optimize=True)

    tau209 = None

    tau342 = zeros((N, N, M, M))

    tau342 += np.einsum("ijdc,dbac->ijab", tau182, tau341, optimize=True)

    tau341 = None

    tau343 += np.einsum("ijab->ijab", tau342, optimize=True)

    tau342 = None

    tau344 = zeros((N, N, M, M))

    tau344 += np.einsum("ikac,kjcb->ijab", tau343, u[o, o, v, v], optimize=True)

    tau343 = None

    tau345 += np.einsum("ijab->ijab", tau344, optimize=True)

    tau344 = None

    r2 += 2 * np.einsum("jiab->abij", tau345, optimize=True) / 3

    r2 -= 10 * np.einsum("jiba->abij", tau345, optimize=True) / 9

    tau345 = None

    tau213 = zeros((N, N, M, M))

    tau213 += np.einsum("abij->ijab", l2, optimize=True)

    tau213 += 2 * np.einsum("abji->ijab", l2, optimize=True)

    tau214 = zeros((N, N, M, M))

    tau214 += np.einsum("cbjk,ikac->ijab", t2, tau213, optimize=True)

    tau213 = None

    tau215 = zeros((N, N, M, M))

    tau215 += np.einsum("kjbc,kiac->ijab", tau214, tau39, optimize=True)

    tau214 = None

    tau218 += 6 * np.einsum("jiba->ijab", tau215, optimize=True)

    tau215 = None

    tau219 = zeros((N, N, M, M))

    tau219 += np.einsum("bckj,kiac->ijab", l2, tau218, optimize=True)

    tau218 = None

    tau234 += np.einsum("jiba->ijab", tau219, optimize=True)

    tau219 = None

    r2 += np.einsum("ijba->abij", tau234, optimize=True) / 9

    r2 += np.einsum("jiab->abij", tau234, optimize=True) / 9

    tau234 = None

    tau262 = zeros((N, N, M, M))

    tau262 += np.einsum("caik,bckj->ijab", l2, t2, optimize=True)

    tau263 = zeros((N, N, M, M))

    tau263 += np.einsum("jkcb,ikca->ijab", tau262, tau34, optimize=True)

    tau264 -= np.einsum("jiba->ijab", tau263, optimize=True)

    tau263 = None

    tau265 = zeros((N, N, M, M))

    tau265 += np.einsum("bcjk,ikca->ijab", l2, tau264, optimize=True)

    tau264 = None

    r2 += 19 * np.einsum("ijba->abij", tau265, optimize=True) / 18

    r2 -= 5 * np.einsum("ijab->abij", tau265, optimize=True) / 3

    tau265 = None

    tau348 = zeros((N, N, M, M))

    tau348 += np.einsum("acdb,ijdc->ijab", tau166, tau262, optimize=True)

    tau166 = None

    tau349 += np.einsum("ijab->ijab", tau348, optimize=True)

    tau348 = None

    tau350 = zeros((N, N, M, M))

    tau350 += np.einsum("ikac,kjbc->ijab", tau349, u[o, o, v, v], optimize=True)

    tau349 = None

    r2 += 4 * np.einsum("ijab->abij", tau350, optimize=True) / 3

    r2 -= 2 * np.einsum("ijba->abij", tau350, optimize=True)

    tau350 = None

    tau352 = zeros((N, N, M, M))

    tau352 += np.einsum("ijdc,acdb->ijab", tau262, tau40, optimize=True)

    tau355 += np.einsum("ijab->ijab", tau352, optimize=True)

    tau352 = None

    tau356 = zeros((N, N, M, M))

    tau356 += np.einsum("ikac,kjcb->ijab", tau355, u[o, o, v, v], optimize=True)

    tau355 = None

    tau357 += np.einsum("ijab->ijab", tau356, optimize=True)

    tau356 = None

    r2 -= 10 * np.einsum("ijab->abij", tau357, optimize=True) / 9

    r2 += 2 * np.einsum("ijba->abij", tau357, optimize=True) / 3

    tau357 = None

    tau400 = zeros((N, N, M, M))

    tau400 += 8 * np.einsum("ijab->ijab", tau262, optimize=True)

    tau483 += 6 * np.einsum("ikcb,jkca->ijab", tau262, tau39, optimize=True)

    tau489 -= np.einsum("ijab->ijab", tau262, optimize=True)

    tau262 = None

    tau490 -= np.einsum("adcb,ijcd->ijab", tau40, tau489, optimize=True)

    tau40 = None

    tau489 = None

    r2 += 2 * np.einsum("ikac,kjbc->abij", tau490, u[o, o, v, v], optimize=True) / 3

    tau490 = None

    tau393 = zeros((N, N, M, M))

    tau393 += 9 * np.einsum("abij->ijab", l2, optimize=True)

    tau393 += np.einsum("baij->ijab", l2, optimize=True)

    tau397 += 4 * np.einsum("kjcb,ikac->ijab", tau34, tau393, optimize=True)

    tau393 = None

    tau394 = zeros((N, N, M, M))

    tau394 += 2 * np.einsum("abij->ijab", l2, optimize=True)

    tau394 += np.einsum("baij->ijab", l2, optimize=True)

    tau397 += 12 * np.einsum("kjcb,ikac->ijab", tau39, tau394, optimize=True)

    tau425 = zeros((N, N, M, M))

    tau425 += np.einsum("bckj,kiac->ijab", t2, tau394, optimize=True)

    tau394 = None

    tau430 += 12 * np.einsum("ikca,jkcb->ijab", tau39, tau425, optimize=True)

    tau425 = None

    tau39 = None

    tau395 = zeros((N, N, N, N))

    tau395 += np.einsum("baij,lkab->ijkl", t2, u[o, o, v, v], optimize=True)

    tau396 = zeros((N, N, N, N))

    tau396 += 2 * np.einsum("klji->ijkl", tau395, optimize=True)

    tau396 += np.einsum("lkji->ijkl", tau395, optimize=True)

    tau406 = zeros((N, N, N, N))

    tau406 += 3 * np.einsum("klji->ijkl", tau395, optimize=True)

    tau406 -= 2 * np.einsum("lkji->ijkl", tau395, optimize=True)

    tau407 = zeros((N, N, N, N))

    tau407 += np.einsum("lkji->ijkl", tau395, optimize=True)

    tau427 = zeros((N, N, N, N))

    tau427 -= 2 * np.einsum("klij->ijkl", tau395, optimize=True)

    tau427 += 3 * np.einsum("lkij->ijkl", tau395, optimize=True)

    tau428 = zeros((N, N, N, N))

    tau428 += np.einsum("klij->ijkl", tau395, optimize=True)

    tau443 = zeros((N, N, N, N))

    tau443 += np.einsum("jmnl,imnk->ijkl", tau395, tau59, optimize=True)

    tau449 += np.einsum("ijkl->ijkl", tau443, optimize=True)

    tau443 = None

    tau450 = zeros((N, N, N, N))

    tau450 += np.einsum("jmnl,imkn->ijkl", tau395, tau59, optimize=True)

    tau453 += np.einsum("ijkl->ijkl", tau450, optimize=True)

    tau450 = None

    tau454 = zeros((N, N, N, N))

    tau454 += np.einsum("mkln,mijn->ijkl", tau395, tau59, optimize=True)

    tau458 += np.einsum("ijkl->ijkl", tau454, optimize=True)

    tau454 = None

    tau455 = zeros((N, N, N, N))

    tau455 += np.einsum("mjln,mink->ijkl", tau395, tau59, optimize=True)

    tau458 += np.einsum("ijkl->ijkl", tau455, optimize=True)

    tau455 = None

    tau471 = zeros((N, N, M, M))

    tau471 += np.einsum("abkl,klij->ijab", l2, tau395, optimize=True)

    tau473 -= np.einsum("ikac,jkcb->ijab", tau471, tau74, optimize=True)

    tau74 = None

    tau483 -= 18 * np.einsum("bcjk,ikac->ijab", t2, tau471, optimize=True)

    tau471 = None

    tau473 -= np.einsum("kjil,lkba->ijab", tau395, tau472, optimize=True)

    tau474 = zeros((N, N, M, M))

    tau474 += np.einsum("abkl,lkij->ijab", l2, tau395, optimize=True)

    tau475 += np.einsum("jiba->ijab", tau474, optimize=True)

    tau478 += np.einsum("bckj,ikca->ijab", t2, tau475, optimize=True)

    tau475 = None

    tau478 -= 2 * np.einsum("bcjk,kiac->ijab", t2, tau474, optimize=True)

    tau488 -= 18 * np.einsum("cbki,kjac->ijab", t2, tau474, optimize=True)

    tau474 = None

    tau478 -= np.einsum("jkli,klab->ijab", tau395, tau477, optimize=True)

    tau477 = None

    tau480 = zeros((N, N, N, N))

    tau480 += 2 * np.einsum("klji->ijkl", tau395, optimize=True)

    tau480 += np.einsum("lkji->ijkl", tau395, optimize=True)

    tau483 += 6 * np.einsum("kjil,klab->ijab", tau395, tau482, optimize=True)

    tau482 = None

    tau488 += 6 * np.einsum("klab,iklj->ijab", tau198, tau395, optimize=True)

    tau198 = None

    tau523 += np.einsum("mlki,lmjk->ij", tau395, tau59, optimize=True)

    tau395 = None

    tau396 += 2 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau396 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau397 += 12 * np.einsum("balk,jikl->ijab", l2, tau396, optimize=True)

    tau408 += np.einsum("bckj,ikca->ijab", t2, tau397, optimize=True)

    tau397 = None

    tau426 = zeros((N, N, M, M))

    tau426 += np.einsum("abkl,jikl->ijab", l2, tau396, optimize=True)

    tau396 = None

    tau430 += 12 * np.einsum("cbik,kjac->ijab", t2, tau426, optimize=True)

    tau426 = None

    tau398 += 36 * np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau398 -= 23 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau408 -= np.einsum("ijcd,cbad->ijab", tau132, tau398, optimize=True)

    tau398 = None

    tau399 = zeros((N, N, M, M))

    tau399 -= 19 * np.einsum("abij->ijab", t2, optimize=True)

    tau399 += 28 * np.einsum("abji->ijab", t2, optimize=True)

    tau400 -= np.einsum("acik,kjbc->ijab", l2, tau399, optimize=True)

    tau399 = None

    tau408 -= np.einsum("jkca,ikcb->ijab", tau34, tau400, optimize=True)

    tau400 = None

    tau401 -= 2 * np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau401 += 5 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau408 += 2 * np.einsum("ijcd,bcda->ijab", tau21, tau401, optimize=True)

    tau401 = None

    tau402 = zeros((N, N, M, M))

    tau402 += np.einsum("abij->ijab", l2, optimize=True)

    tau402 += 9 * np.einsum("baij->ijab", l2, optimize=True)

    tau403 = zeros((N, N, M, M))

    tau403 += np.einsum("cbkj,ikca->ijab", t2, tau402, optimize=True)

    tau408 -= 4 * np.einsum("bcad,ijcd->ijab", tau119, tau403, optimize=True)

    tau403 = None

    tau417 += 4 * np.einsum("kjcb,kiac->ijab", tau34, tau402, optimize=True)

    tau430 += np.einsum("bcki,jkca->ijab", t2, tau417, optimize=True)

    tau417 = None

    tau420 += np.einsum("cbkj,kiac->ijab", t2, tau402, optimize=True)

    tau402 = None

    tau430 -= 4 * np.einsum("bcad,jicd->ijab", tau119, tau420, optimize=True)

    tau420 = None

    tau404 = zeros((N, N, M, M))

    tau404 += 36 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau404 -= 23 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau405 += np.einsum("acki,kjcb->ijab", t2, tau404, optimize=True)

    tau404 = None

    tau405 += 23 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau405 -= 36 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau408 -= np.einsum("klba,ikjl->ijab", tau405, tau59, optimize=True)

    tau405 = None

    tau406 += 3 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau406 -= 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau408 -= 12 * np.einsum("lkab,kijl->ijab", tau182, tau406, optimize=True)

    tau406 = None

    tau182 = None

    tau407 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau408 += 12 * np.einsum("lkab,kijl->ijab", tau4, tau407, optimize=True)

    tau4 = None

    tau408 -= 36 * np.einsum("lkab,kilj->ijab", tau11, tau407, optimize=True)

    tau11 = None

    r2 += np.einsum("ackj,ikbc->abij", l2, tau408, optimize=True) / 18

    tau408 = None

    tau469 -= 3 * np.einsum("minj,nkml->ijkl", tau407, tau59, optimize=True)

    tau469 -= 3 * np.einsum("km,milj->ijkl", tau29, tau407, optimize=True)

    tau473 += np.einsum("kilj,lkab->ijab", tau407, tau410, optimize=True)

    tau410 = None

    r2 -= 2 * np.einsum("bcjk,ikac->abij", l2, tau473, optimize=True)

    tau473 = None

    tau526 += np.einsum("kjlm,mlik->ij", tau407, tau59, optimize=True)

    tau407 = None

    tau418 = zeros((N, N, M, M))

    tau418 += 8 * np.einsum("abij->ijab", l2, optimize=True)

    tau418 += 19 * np.einsum("baij->ijab", l2, optimize=True)

    tau419 += np.einsum("bckj,kiac->ijab", t2, tau418, optimize=True)

    tau418 = None

    tau430 -= np.einsum("ikca,jkcb->ijab", tau34, tau419, optimize=True)

    tau419 = None

    tau421 = zeros((N, N, M, M))

    tau421 += np.einsum("abij->ijab", t2, optimize=True)

    tau421 += 7 * np.einsum("abji->ijab", t2, optimize=True)

    tau422 += np.einsum("caki,jkcb->ijab", l2, tau421, optimize=True)

    tau421 = None

    tau430 -= 4 * np.einsum("jicd,cbda->ijab", tau422, u[v, v, v, v], optimize=True)

    tau422 = None

    tau423 = zeros((N, N, M, M))

    tau423 += 19 * np.einsum("abij->ijab", l2, optimize=True)

    tau423 += 8 * np.einsum("baij->ijab", l2, optimize=True)

    tau424 = zeros((N, N, M, M))

    tau424 += np.einsum("cbjk,kiac->ijab", t2, tau423, optimize=True)

    tau423 = None

    tau430 += np.einsum("bcda,jicd->ijab", tau119, tau424, optimize=True)

    tau424 = None

    tau427 += 3 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau427 -= 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau430 -= 12 * np.einsum("kjil,lkab->ijab", tau427, tau51, optimize=True)

    tau427 = None

    tau428 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau430 += 12 * np.einsum("lkab,kjil->ijab", tau21, tau428, optimize=True)

    tau430 -= 36 * np.einsum("lkab,kjli->ijab", tau15, tau428, optimize=True)

    tau15 = None

    r2 += np.einsum("bcki,kjac->abij", l2, tau430, optimize=True) / 18

    tau430 = None

    tau469 -= 3 * np.einsum("mknl,injm->ijkl", tau428, tau59, optimize=True)

    tau469 -= 3 * np.einsum("im,mkjl->ijkl", tau113, tau428, optimize=True)

    tau113 = None

    tau478 += np.einsum("kilj,lkab->ijab", tau428, tau476, optimize=True)

    tau428 = None

    tau476 = None

    r2 -= 2 * np.einsum("acik,jkbc->abij", l2, tau478, optimize=True)

    tau478 = None

    tau444 = zeros((N, N, M, M))

    tau444 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau444 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau445 = zeros((N, N, M, M))

    tau445 += np.einsum("bcjk,kiac->ijab", t2, tau444, optimize=True)

    tau446 -= np.einsum("jiba->ijab", tau445, optimize=True)

    tau445 = None

    tau447 = zeros((N, N, N, N))

    tau447 += np.einsum("klab,ijab->ijkl", tau132, tau446, optimize=True)

    tau132 = None

    tau449 += np.einsum("klij->ijkl", tau447, optimize=True)

    tau447 = None

    tau469 += 2 * np.einsum("ijlk->ijkl", tau449, optimize=True)

    tau469 += np.einsum("iljk->ijkl", tau449, optimize=True)

    tau449 = None

    tau451 = zeros((N, N, N, N))

    tau451 += np.einsum("klab,ijab->ijkl", tau21, tau446, optimize=True)

    tau21 = None

    tau453 += np.einsum("klij->ijkl", tau451, optimize=True)

    tau451 = None

    tau469 += np.einsum("ijlk->ijkl", tau453, optimize=True)

    tau469 += 2 * np.einsum("iljk->ijkl", tau453, optimize=True)

    tau453 = None

    tau456 = zeros((N, N, N, N))

    tau456 += np.einsum("klab,ijab->ijkl", tau154, tau446, optimize=True)

    tau154 = None

    tau458 += np.einsum("klij->ijkl", tau456, optimize=True)

    tau456 = None

    tau469 += 2 * np.einsum("kjli->ijkl", tau458, optimize=True)

    tau469 += np.einsum("klji->ijkl", tau458, optimize=True)

    tau458 = None

    tau459 = zeros((N, N, N, N))

    tau459 += np.einsum("ijab,klab->ijkl", tau446, tau51, optimize=True)

    tau51 = None

    tau446 = None

    tau462 += np.einsum("klij->ijkl", tau459, optimize=True)

    tau459 = None

    tau469 += np.einsum("kjli->ijkl", tau462, optimize=True)

    tau469 += 2 * np.einsum("klji->ijkl", tau462, optimize=True)

    tau462 = None

    tau526 -= np.einsum("kiab,kjab->ij", tau43, tau444, optimize=True)

    tau43 = None

    tau444 = None

    tau466 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau469 += 3 * np.einsum("ijab,klab->ijkl", tau466, tau467, optimize=True)

    tau466 = None

    tau467 = None

    tau468 += np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau469 += 3 * np.einsum("ilab,kjab->ijkl", tau468, tau96, optimize=True)

    tau96 = None

    tau468 = None

    r2 += 2 * np.einsum("abkl,ikjl->abij", l2, tau469, optimize=True) / 3

    tau469 = None

    tau479 = zeros((N, N, M, M))

    tau479 += np.einsum("caik,cbkj->ijab", l2, t2, optimize=True)

    tau483 -= np.einsum("bcad,ijcd->ijab", tau119, tau479, optimize=True)

    tau479 = None

    tau480 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau481 += 6 * np.einsum("bakl,jikl->ijab", l2, tau480, optimize=True)

    tau483 += np.einsum("bckj,ikca->ijab", t2, tau481, optimize=True)

    tau481 = None

    r2 += np.einsum("acjk,ikbc->abij", l2, tau483, optimize=True) / 9

    tau483 = None

    tau487 = zeros((N, N, M, M))

    tau487 += np.einsum("ablk,jikl->ijab", l2, tau480, optimize=True)

    tau480 = None

    tau488 += 6 * np.einsum("cbik,kjac->ijab", t2, tau487, optimize=True)

    tau487 = None

    tau484 = zeros((N, N, M, M))

    tau484 += np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau488 -= np.einsum("bcad,jicd->ijab", tau119, tau484, optimize=True)

    tau119 = None

    r2 += np.einsum("bcik,kjac->abij", l2, tau488, optimize=True) / 9

    tau488 = None

    tau499 -= np.einsum("kibc,kjac->ijab", tau472, tau484, optimize=True)

    tau472 = None

    r2 += 2 * np.einsum("ikac,kjcb->abij", tau499, u[o, o, v, v], optimize=True) / 9

    tau499 = None

    tau503 -= np.einsum("kjac,ikcb->ijab", tau484, tau502, optimize=True)

    tau484 = None

    tau502 = None

    r2 += 2 * np.einsum("jkbc,kica->abij", tau503, u[o, o, v, v], optimize=True) / 9

    tau503 = None

    tau508 = zeros((M, M))

    tau508 += np.einsum("acij,cbij->ab", l2, t2, optimize=True)

    tau509 = zeros((N, N, M, M))

    tau509 += np.einsum("ac,jicb->ijab", tau508, u[o, o, v, v], optimize=True)

    tau508 = None

    tau510 += 2 * np.einsum("ijba->ijab", tau509, optimize=True)

    tau510 += np.einsum("jiba->ijab", tau509, optimize=True)

    tau509 = None

    r2 -= np.einsum("klab,ijkl->abij", tau510, tau59, optimize=True) / 9

    tau510 = None

    tau511 = zeros((N, N, M, M))

    tau511 += 9 * np.einsum("abij->ijab", l2, optimize=True)

    tau511 += np.einsum("abji->ijab", l2, optimize=True)

    tau512 = zeros((N, N, M, M))

    tau512 += np.einsum("kjcb,ikac->ijab", tau34, tau511, optimize=True)

    tau34 = None

    tau511 = None

    r2 += 2 * np.einsum("klba,ijlk->abij", tau512, tau59, optimize=True) / 9

    tau512 = None

    tau59 = None

    tau515 = zeros((N, N, N, N))

    tau515 += np.einsum("abij,bakl->ijkl", l2, t2, optimize=True)

    tau523 += np.einsum("lmjk,ikml->ij", tau515, u[o, o, o, o], optimize=True)

    tau515 = None

    tau518 = zeros((N, N, M, M))

    tau518 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau518 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau519 = zeros((N, N, M, M))

    tau519 += np.einsum("bcjk,ikca->ijab", l2, tau518, optimize=True)

    tau518 = None

    tau523 += np.einsum("abkj,ikba->ij", t2, tau519, optimize=True)

    tau525 += np.einsum("jiba->ijab", tau519, optimize=True)

    tau519 = None

    tau526 += np.einsum("abki,kjab->ij", t2, tau525, optimize=True)

    tau525 = None

    tau521 = zeros((N, N, N, N))

    tau521 += 2 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau521 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau522 = zeros((N, N))

    tau522 += np.einsum("lk,kijl->ij", tau29, tau521, optimize=True)

    tau521 = None

    tau29 = None

    tau523 -= np.einsum("ij->ij", tau522, optimize=True)

    r2 -= 2 * np.einsum("jk,abik->abij", tau523, l2, optimize=True)

    tau523 = None

    tau526 -= np.einsum("ji->ij", tau522, optimize=True)

    tau522 = None

    r2 -= 2 * np.einsum("ki,abkj->abij", tau526, l2, optimize=True)

    tau526 = None

    return r2
