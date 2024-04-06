import numpy as np
from clusterfock.cc.rhs.t_inter_RCCD import amplitudes_intermediates_rccd


def t_intermediates_qccd_restricted(t2, l2, u, f, v, o):
    r2 = amplitudes_intermediates_rccd(t2, u, f, v, o)
    r2 += t2_intermediate_qccd_addition_restricted(t2, l2, u, f, v, o)

    return r2


def t2_intermediate_qccd_addition_restricted(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N))

    tau0 += np.einsum("abki,abkj->ij", l2, t2, optimize=True)

    tau1 = zeros((N, N))

    tau1 += np.einsum("kl,lijk->ij", tau0, u[o, o, o, o], optimize=True)

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum("kj,abik->ijab", tau1, t2, optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("ijab->ijab", tau2, optimize=True)

    tau2 = None

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("kj,abki->ijab", tau1, t2, optimize=True)

    tau1 = None

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("ijab->abij", tau10, optimize=True) / 9

    r2 -= 7 * np.einsum("jiab->abij", tau10, optimize=True) / 9

    tau10 = None

    tau49 = zeros((N, N, M, M))

    tau49 += np.einsum("ik,kabj->ijab", tau0, u[o, v, v, o], optimize=True)

    tau84 = zeros((N, N, M, M))

    tau84 -= 12 * np.einsum("jiba->ijab", tau49, optimize=True)

    tau49 = None

    tau61 = zeros((N, N, M, M))

    tau61 += np.einsum("ik,kjba->ijab", tau0, u[o, o, v, v], optimize=True)

    tau67 = zeros((N, N, M, M))

    tau67 += 12 * np.einsum("ijab->ijab", tau61, optimize=True)

    tau67 -= 4 * np.einsum("ijba->ijab", tau61, optimize=True)

    tau67 -= 4 * np.einsum("jiab->ijab", tau61, optimize=True)

    tau67 += 8 * np.einsum("jiba->ijab", tau61, optimize=True)

    tau72 = zeros((N, N, M, M))

    tau72 += 12 * np.einsum("ijab->ijab", tau61, optimize=True)

    tau72 -= 6 * np.einsum("ijba->ijab", tau61, optimize=True)

    tau103 = zeros((N, N, M, M))

    tau103 -= 4 * np.einsum("jiab->ijab", tau61, optimize=True)

    tau103 += 12 * np.einsum("jiba->ijab", tau61, optimize=True)

    tau429 = zeros((N, N, M, M))

    tau429 -= 14 * np.einsum("ijab->ijab", tau61, optimize=True)

    tau429 += 36 * np.einsum("ijba->ijab", tau61, optimize=True)

    tau438 = zeros((N, N, M, M))

    tau438 += 36 * np.einsum("ijab->ijab", tau61, optimize=True)

    tau438 -= 14 * np.einsum("ijba->ijab", tau61, optimize=True)

    tau61 = None

    tau94 = zeros((M, M))

    tau94 += np.einsum("ij,jaib->ab", tau0, u[o, v, o, v], optimize=True)

    tau112 = zeros((M, M))

    tau112 -= 24 * np.einsum("ab->ab", tau94, optimize=True)

    tau94 = None

    tau441 = zeros((N, N, M, M))

    tau441 += np.einsum("kj,abki->ijab", tau0, t2, optimize=True)

    tau442 = zeros((N, N))

    tau442 -= 2 * np.einsum("ikba,kjab->ij", tau441, u[o, o, v, v], optimize=True)

    tau441 = None

    tau443 = zeros((N, N, M, M))

    tau443 += np.einsum("kj,abik->ijab", tau0, t2, optimize=True)

    tau444 = zeros((N, N))

    tau444 -= 2 * np.einsum("ikba,kjba->ij", tau443, u[o, o, v, v], optimize=True)

    tau443 = None

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum("klab,lijk->ijab", tau3, u[o, o, o, o], optimize=True)

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("ijab->ijab", tau4, optimize=True)

    tau46 = zeros((N, N, M, M))

    tau46 += np.einsum("acki,kjcb->ijab", t2, tau4, optimize=True)

    tau4 = None

    r2 += 19 * np.einsum("ijab->abij", tau46, optimize=True) / 36

    r2 += 2 * np.einsum("jiab->abij", tau46, optimize=True) / 9

    tau46 = None

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("acki,kjcb->ijab", t2, tau3, optimize=True)

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("klab,lkji->ijab", tau27, u[o, o, o, o], optimize=True)

    tau35 = zeros((N, N, M, M))

    tau35 += 4 * np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau143 = zeros((N, N, M, M))

    tau143 += np.einsum("kiac,kbjc->ijab", tau27, u[o, v, o, v], optimize=True)

    r2 -= 11 * np.einsum("jiab->abij", tau143, optimize=True) / 18

    r2 -= 2 * np.einsum("jiba->abij", tau143, optimize=True) / 9

    tau143 = None

    tau149 = zeros((N, N, M, M))

    tau149 += np.einsum("kiac,jkcb->ijab", tau27, u[o, o, v, v], optimize=True)

    tau155 = zeros((N, N, M, M))

    tau155 += np.einsum("jiab->ijab", tau149, optimize=True)

    tau149 = None

    tau241 = zeros((N, N, M, M))

    tau241 += np.einsum("cbik,kjca->ijab", t2, tau3, optimize=True)

    tau242 = zeros((N, N, M, M))

    tau242 += np.einsum("ijdc,bacd->ijab", tau241, u[v, v, v, v], optimize=True)

    tau258 = zeros((N, N, M, M))

    tau258 += 6 * np.einsum("ijab->ijab", tau242, optimize=True)

    tau242 = None

    tau386 = zeros((N, N, N, N))

    tau386 += np.einsum("ijba,lkab->ijkl", tau241, u[o, o, v, v], optimize=True)

    tau241 = None

    tau407 = zeros((N, N, N, N))

    tau407 += 8 * np.einsum("jlik->ijkl", tau386, optimize=True)

    tau407 += 16 * np.einsum("jlki->ijkl", tau386, optimize=True)

    tau386 = None

    tau324 = zeros((N, N, M, M))

    tau324 -= 3 * np.einsum("ijab->ijab", tau3, optimize=True)

    tau368 = zeros((N, N, M, M))

    tau368 += np.einsum("acik,kjcb->ijab", t2, tau3, optimize=True)

    tau370 = zeros((N, N, M, M))

    tau370 += np.einsum("ijab->ijab", tau368, optimize=True)

    tau380 = zeros((N, N, M, M))

    tau380 += np.einsum("ijab->ijab", tau368, optimize=True)

    tau408 = zeros((N, N, M, M))

    tau408 += np.einsum("ijab->ijab", tau368, optimize=True)

    tau368 = None

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("cdij,dcba->ijab", l2, u[v, v, v, v], optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ackj,ikbc->ijab", t2, tau5, optimize=True)

    tau7 += np.einsum("ijba->ijab", tau6, optimize=True)

    tau6 = None

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum("bcjk,kica->ijab", t2, tau7, optimize=True)

    tau7 = None

    tau9 += np.einsum("jiba->ijab", tau8, optimize=True)

    tau8 = None

    r2 -= 7 * np.einsum("ijab->abij", tau9, optimize=True) / 9

    r2 += np.einsum("jiab->abij", tau9, optimize=True) / 9

    tau9 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("ackj,kicb->ijab", t2, tau5, optimize=True)

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("ijba->ijab", tau13, optimize=True)

    tau13 = None

    tau29 = zeros((N, N, M, M))

    tau29 += np.einsum("acjk,kibc->ijab", t2, tau5, optimize=True)

    tau33 = zeros((N, N, M, M))

    tau33 += np.einsum("jiba->ijab", tau29, optimize=True)

    tau29 = None

    tau31 = zeros((N, N, M, M))

    tau31 += 2 * np.einsum("ijab->ijab", tau5, optimize=True)

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum("ackj,kibc->ijab", t2, tau5, optimize=True)

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("ijba->ijab", tau38, optimize=True)

    tau38 = None

    tau101 = zeros((N, N, M, M))

    tau101 += 6 * np.einsum("jiba->ijab", tau5, optimize=True)

    tau314 = zeros((N, N, M, M))

    tau314 += np.einsum("ackj,ikcb->ijab", t2, tau5, optimize=True)

    tau315 = zeros((N, N, M, M))

    tau315 += np.einsum("acki,kjbc->ijab", t2, tau314, optimize=True)

    tau314 = None

    r2 += 2 * np.einsum("ijab->abij", tau315, optimize=True) / 9

    r2 += 11 * np.einsum("jiab->abij", tau315, optimize=True) / 18

    tau315 = None

    tau334 = zeros((N, N, M, M))

    tau334 -= 72 * np.einsum("bcik,jkac->ijab", t2, tau5, optimize=True)

    tau5 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("klab,lijk->ijab", tau11, u[o, o, o, o], optimize=True)

    tau14 += np.einsum("ijab->ijab", tau12, optimize=True)

    tau15 = zeros((N, N, M, M))

    tau15 += np.einsum("bcjk,kica->ijab", t2, tau14, optimize=True)

    tau14 = None

    r2 += np.einsum("jiab->abij", tau15, optimize=True) / 9

    r2 -= 7 * np.einsum("ijab->abij", tau15, optimize=True) / 9

    tau15 = None

    tau41 = zeros((N, N, M, M))

    tau41 += np.einsum("acki,kjcb->ijab", t2, tau12, optimize=True)

    tau12 = None

    r2 += 2 * np.einsum("ijba->abij", tau41, optimize=True) / 9

    r2 += 19 * np.einsum("jiba->abij", tau41, optimize=True) / 36

    tau41 = None

    tau123 = zeros((N, N, M, M))

    tau123 += np.einsum("ikca,kcbj->ijab", tau11, u[o, v, v, o], optimize=True)

    tau124 = zeros((N, N, M, M))

    tau124 += np.einsum("acki,kjbc->ijab", t2, tau123, optimize=True)

    tau123 = None

    r2 -= 2 * np.einsum("ijab->abij", tau124, optimize=True) / 9

    r2 -= 11 * np.einsum("ijba->abij", tau124, optimize=True) / 18

    r2 -= 11 * np.einsum("jiab->abij", tau124, optimize=True) / 18

    r2 -= 2 * np.einsum("jiba->abij", tau124, optimize=True) / 9

    tau124 = None

    tau125 = zeros((N, N, M, M))

    tau125 += np.einsum("ikca,kcjb->ijab", tau11, u[o, v, o, v], optimize=True)

    tau126 = zeros((N, N, M, M))

    tau126 += np.einsum("acki,kjbc->ijab", t2, tau125, optimize=True)

    tau125 = None

    r2 -= 11 * np.einsum("ijab->abij", tau126, optimize=True) / 18

    r2 -= 2 * np.einsum("ijba->abij", tau126, optimize=True) / 9

    r2 -= 2 * np.einsum("jiab->abij", tau126, optimize=True) / 9

    r2 -= 11 * np.einsum("jiba->abij", tau126, optimize=True) / 18

    tau126 = None

    tau231 = zeros((N, N, M, M))

    tau231 += np.einsum("ijcd,acbd->ijab", tau11, u[v, v, v, v], optimize=True)

    tau235 = zeros((N, N, M, M))

    tau235 -= np.einsum("jiba->ijab", tau231, optimize=True)

    tau231 = None

    tau293 = zeros((N, N, M, M))

    tau293 += np.einsum("ijcd,acdb->ijab", tau11, u[v, v, v, v], optimize=True)

    tau295 = zeros((N, N, M, M))

    tau295 += np.einsum("ijab->ijab", tau293, optimize=True)

    tau293 = None

    tau417 = zeros((N, N, N, N))

    tau417 -= 12 * np.einsum("ilab,kabj->ijkl", tau11, u[o, v, v, o], optimize=True)

    tau16 = zeros((N, N, N, N))

    tau16 += np.einsum("abij,abkl->ijkl", l2, t2, optimize=True)

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("ijkl,lkba->ijab", tau16, u[o, o, v, v], optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 += np.einsum("ijba->ijab", tau17, optimize=True)

    tau31 += 3 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau101 += 6 * np.einsum("jiba->ijab", tau17, optimize=True)

    tau322 = zeros((N, N, M, M))

    tau322 += np.einsum("ijab->ijab", tau17, optimize=True)

    tau17 = None

    tau158 = zeros((N, N, M, M))

    tau158 += np.einsum("abkl,klij->ijab", t2, tau16, optimize=True)

    tau159 = zeros((N, N, M, M))

    tau159 += np.einsum("ikac,kbjc->ijab", tau158, u[o, v, o, v], optimize=True)

    tau169 = zeros((N, N, M, M))

    tau169 -= np.einsum("ijab->ijab", tau159, optimize=True)

    tau159 = None

    tau170 = zeros((N, N, M, M))

    tau170 += np.einsum("kiac,kbjc->ijab", tau158, u[o, v, o, v], optimize=True)

    r2 -= 7 * np.einsum("jiab->abij", tau170, optimize=True) / 18

    r2 -= np.einsum("jiba->abij", tau170, optimize=True) / 9

    tau170 = None

    tau172 = zeros((N, N, M, M))

    tau172 += np.einsum("kiac,jkcb->ijab", tau158, u[o, o, v, v], optimize=True)

    tau174 = zeros((N, N, M, M))

    tau174 += np.einsum("jiab->ijab", tau172, optimize=True)

    tau172 = None

    tau176 = zeros((N, N, M, M))

    tau176 += np.einsum("ikac,jkcb->ijab", tau158, u[o, o, v, v], optimize=True)

    tau178 = zeros((N, N, M, M))

    tau178 += np.einsum("jiab->ijab", tau176, optimize=True)

    tau176 = None

    tau347 = zeros((N, N, M, M))

    tau347 += np.einsum("ijab->ijab", tau158, optimize=True)

    tau158 = None

    tau160 = zeros((N, N, M, M))

    tau160 += np.einsum("ablk,klij->ijab", t2, tau16, optimize=True)

    tau161 = zeros((N, N, M, M))

    tau161 += np.einsum("kiac,jkcb->ijab", tau160, u[o, o, v, v], optimize=True)

    tau163 = zeros((N, N, M, M))

    tau163 += np.einsum("jiab->ijab", tau161, optimize=True)

    tau161 = None

    tau165 = zeros((N, N, M, M))

    tau165 += np.einsum("ikac,jkcb->ijab", tau160, u[o, o, v, v], optimize=True)

    tau167 = zeros((N, N, M, M))

    tau167 += np.einsum("jiab->ijab", tau165, optimize=True)

    tau165 = None

    tau171 = zeros((N, N, M, M))

    tau171 += np.einsum("ikac,kbjc->ijab", tau160, u[o, v, o, v], optimize=True)

    tau180 = zeros((N, N, M, M))

    tau180 -= np.einsum("ijab->ijab", tau171, optimize=True)

    tau171 = None

    tau181 = zeros((N, N, M, M))

    tau181 += np.einsum("kiac,kbjc->ijab", tau160, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("jiab->abij", tau181, optimize=True) / 9

    r2 -= 7 * np.einsum("jiba->abij", tau181, optimize=True) / 18

    tau181 = None

    tau321 = zeros((N, N, M, M))

    tau321 += np.einsum("ijab->ijab", tau160, optimize=True)

    tau160 = None

    tau405 = zeros((N, N, N, N))

    tau405 += np.einsum("ijkl->ijkl", tau16, optimize=True)

    tau405 += 2 * np.einsum("jikl->ijkl", tau16, optimize=True)

    tau406 = zeros((N, N, N, N))

    tau406 += 10 * np.einsum("ijkl->ijkl", tau16, optimize=True)

    tau406 += 23 * np.einsum("jikl->ijkl", tau16, optimize=True)

    tau415 = zeros((N, N, N, N))

    tau415 += 23 * np.einsum("ijkl->ijkl", tau16, optimize=True)

    tau415 += 10 * np.einsum("jikl->ijkl", tau16, optimize=True)

    tau416 = zeros((N, N, N, N))

    tau416 += 2 * np.einsum("ijkl->ijkl", tau16, optimize=True)

    tau416 += np.einsum("jikl->ijkl", tau16, optimize=True)

    tau18 = zeros((N, N, N, N))

    tau18 += np.einsum("baij,lkab->ijkl", t2, u[o, o, v, v], optimize=True)

    tau19 = zeros((N, N, N, N))

    tau19 += np.einsum("lkji->ijkl", tau18, optimize=True)

    tau329 = zeros((N, N, N, N))

    tau329 += np.einsum("klij->ijkl", tau18, optimize=True)

    tau426 = zeros((N, N, N, N))

    tau426 += 4 * np.einsum("ijkl->ijkl", tau18, optimize=True)

    tau426 += 11 * np.einsum("ijlk->ijkl", tau18, optimize=True)

    tau427 = zeros((N, N, N, N))

    tau427 += np.einsum("ijkl->ijkl", tau18, optimize=True)

    tau427 += 2 * np.einsum("ijlk->ijkl", tau18, optimize=True)

    r2 += 2 * np.einsum("lkab,ijkl->abij", tau27, tau427, optimize=True) / 9

    tau27 = None

    tau427 = None

    tau431 = zeros((N, N, N, N))

    tau431 += 9 * np.einsum("klji->ijkl", tau18, optimize=True)

    tau431 += np.einsum("lkji->ijkl", tau18, optimize=True)

    tau19 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum("abkl,ijkl->ijab", l2, tau19, optimize=True)

    tau21 += np.einsum("jiab->ijab", tau20, optimize=True)

    tau20 = None

    tau22 = zeros((N, N, M, M))

    tau22 += np.einsum("bckj,ikac->ijab", t2, tau21, optimize=True)

    tau21 = None

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("bckj,kica->ijab", t2, tau22, optimize=True)

    tau22 = None

    r2 += np.einsum("jiba->abij", tau23, optimize=True) / 3

    r2 += 2 * np.einsum("ijba->abij", tau23, optimize=True) / 3

    tau23 = None

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("ablk,ijkl->ijab", l2, tau19, optimize=True)

    tau31 += 3 * np.einsum("jiab->ijab", tau30, optimize=True)

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("bckj,kica->ijab", t2, tau31, optimize=True)

    tau31 = None

    tau33 += 2 * np.einsum("jiab->ijab", tau32, optimize=True)

    tau32 = None

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum("bckj,ikca->ijab", t2, tau33, optimize=True)

    tau33 = None

    tau35 += np.einsum("ijab->ijab", tau34, optimize=True)

    tau34 = None

    tau322 += np.einsum("jiab->ijab", tau30, optimize=True)

    tau30 = None

    tau358 = zeros((N, N, M, M))

    tau358 += 36 * np.einsum("bckj,kica->ijab", t2, tau322, optimize=True)

    tau407 += 10 * np.einsum("mkln,nijm->ijkl", tau19, tau405, optimize=True)

    tau405 = None

    tau417 += np.einsum("mkln,nijm->ijkl", tau19, tau415, optimize=True)

    tau415 = None

    tau417 -= 36 * np.einsum("niml,mknj->ijkl", tau16, tau19, optimize=True)

    tau417 -= 36 * np.einsum("im,mklj->ijkl", tau0, tau19, optimize=True)

    tau442 += np.einsum("mlki,kjlm->ij", tau16, tau19, optimize=True)

    r2 -= np.einsum("jk,abki->abij", tau442, t2, optimize=True) / 18

    tau442 = None

    tau444 += np.einsum("lmik,kjlm->ij", tau16, tau19, optimize=True)

    r2 -= np.einsum("ik,abjk->abij", tau444, t2, optimize=True) / 18

    tau444 = None

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum("klab,lijk->ijab", tau24, u[o, o, o, o], optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("acik,kjcb->ijab", t2, tau25, optimize=True)

    tau35 += np.einsum("ijab->ijab", tau26, optimize=True)

    tau26 = None

    r2 += np.einsum("ijab->abij", tau35, optimize=True) / 9

    r2 += np.einsum("jiab->abij", tau35, optimize=True) / 18

    tau35 = None

    tau45 = zeros((N, N, M, M))

    tau45 += np.einsum("acki,kjcb->ijab", t2, tau25, optimize=True)

    tau25 = None

    r2 += 2 * np.einsum("ijab->abij", tau45, optimize=True) / 9

    r2 += 19 * np.einsum("jiab->abij", tau45, optimize=True) / 36

    tau45 = None

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum("bckj,kica->ijab", t2, tau24, optimize=True)

    tau43 = zeros((N, N, M, M))

    tau43 += np.einsum("klab,lkji->ijab", tau42, u[o, o, o, o], optimize=True)

    r2 += 11 * np.einsum("ijba->abij", tau43, optimize=True) / 18

    r2 += 2 * np.einsum("jiba->abij", tau43, optimize=True) / 9

    tau43 = None

    tau215 = zeros((N, N, M, M))

    tau215 += np.einsum("ikca,jkcb->ijab", tau42, u[o, o, v, v], optimize=True)

    tau221 = zeros((N, N, M, M))

    tau221 += 2 * np.einsum("ijba->ijab", tau215, optimize=True)

    tau215 = None

    tau259 = zeros((N, N, M, M))

    tau259 += np.einsum("ikca,kbjc->ijab", tau42, u[o, v, o, v], optimize=True)

    tau265 = zeros((N, N, M, M))

    tau265 += 4 * np.einsum("ijab->ijab", tau259, optimize=True)

    tau259 = None

    r2 += np.einsum("lkba,ijkl->abij", tau42, tau426, optimize=True) / 18

    tau426 = None

    tau42 = None

    tau206 = zeros((N, N, M, M))

    tau206 += np.einsum("cbik,kjca->ijab", t2, tau24, optimize=True)

    tau207 = zeros((N, N, M, M))

    tau207 += np.einsum("ijdc,bacd->ijab", tau206, u[v, v, v, v], optimize=True)

    tau240 = zeros((N, N, M, M))

    tau240 += 6 * np.einsum("ijab->ijab", tau207, optimize=True)

    tau207 = None

    tau387 = zeros((N, N, N, N))

    tau387 += np.einsum("ijba,klba->ijkl", tau206, u[o, o, v, v], optimize=True)

    tau206 = None

    tau407 += 22 * np.einsum("jlik->ijkl", tau387, optimize=True)

    tau407 += 8 * np.einsum("jlki->ijkl", tau387, optimize=True)

    tau387 = None

    tau319 = zeros((N, N, M, M))

    tau319 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau330 = zeros((N, N, M, M))

    tau330 += 12 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau363 = zeros((N, N, M, M))

    tau363 += np.einsum("ijab->ijab", tau24, optimize=True)

    tau376 = zeros((N, N, M, M))

    tau376 += 8 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau382 = zeros((N, N, M, M))

    tau382 += np.einsum("acik,kjcb->ijab", t2, tau24, optimize=True)

    tau384 = zeros((N, N, M, M))

    tau384 += np.einsum("ijab->ijab", tau382, optimize=True)

    tau382 = None

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum("caik,bckj->ijab", l2, t2, optimize=True)

    tau37 = zeros((N, N, M, M))

    tau37 += np.einsum("klab,lijk->ijab", tau36, u[o, o, o, o], optimize=True)

    tau39 += np.einsum("ijab->ijab", tau37, optimize=True)

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum("bcjk,kica->ijab", t2, tau39, optimize=True)

    tau39 = None

    r2 += np.einsum("jiab->abij", tau40, optimize=True) / 18

    r2 += np.einsum("ijab->abij", tau40, optimize=True) / 9

    tau40 = None

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("acki,kjcb->ijab", t2, tau37, optimize=True)

    tau37 = None

    r2 += 19 * np.einsum("ijba->abij", tau44, optimize=True) / 36

    r2 += 2 * np.einsum("jiba->abij", tau44, optimize=True) / 9

    tau44 = None

    tau75 = zeros((N, N, M, M))

    tau75 += np.einsum("ijab->ijab", tau36, optimize=True)

    tau183 = zeros((N, N, M, M))

    tau183 += np.einsum("ikca,kcjb->ijab", tau36, u[o, v, o, v], optimize=True)

    tau188 = zeros((N, N, M, M))

    tau188 += 2 * np.einsum("ijba->ijab", tau183, optimize=True)

    tau183 = None

    tau199 = zeros((N, N, M, M))

    tau199 += np.einsum("ikca,kcbj->ijab", tau36, u[o, v, v, o], optimize=True)

    tau200 = zeros((N, N, M, M))

    tau200 += 2 * np.einsum("ijba->ijab", tau199, optimize=True)

    tau199 = None

    tau298 = zeros((N, N, M, M))

    tau298 += np.einsum("ijcd,acdb->ijab", tau36, u[v, v, v, v], optimize=True)

    tau300 = zeros((N, N, M, M))

    tau300 += np.einsum("ijab->ijab", tau298, optimize=True)

    tau298 = None

    tau334 += 36 * np.einsum("jicd,bcad->ijab", tau36, u[v, v, v, v], optimize=True)

    tau354 = zeros((N, N, M, M))

    tau354 += 12 * np.einsum("ijab->ijab", tau36, optimize=True)

    tau367 = zeros((N, N, M, M))

    tau367 += np.einsum("ijab->ijab", tau36, optimize=True)

    tau47 = zeros((M, M, M, M))

    tau47 += np.einsum("abij,cdji->abcd", l2, t2, optimize=True)

    tau48 = zeros((N, N, M, M))

    tau48 += np.einsum("acbd,icdj->ijab", tau47, u[o, v, v, o], optimize=True)

    tau84 += 6 * np.einsum("jiab->ijab", tau48, optimize=True)

    tau200 += 3 * np.einsum("ijab->ijab", tau48, optimize=True)

    tau48 = None

    tau201 = zeros((N, N, M, M))

    tau201 += np.einsum("bckj,kica->ijab", t2, tau200, optimize=True)

    tau200 = None

    tau202 = zeros((N, N, M, M))

    tau202 += np.einsum("jiba->ijab", tau201, optimize=True)

    tau201 = None

    tau190 = zeros((N, N, M, M))

    tau190 += np.einsum("acbd,icjd->ijab", tau47, u[o, v, o, v], optimize=True)

    tau191 = zeros((N, N, M, M))

    tau191 += np.einsum("acik,kjcb->ijab", t2, tau190, optimize=True)

    tau190 = None

    tau202 += np.einsum("ijab->ijab", tau191, optimize=True)

    tau191 = None

    tau307 = zeros((M, M, M, M))

    tau307 += np.einsum("aebf,cefd->abcd", tau47, u[v, v, v, v], optimize=True)

    tau309 = zeros((M, M, M, M))

    tau309 += np.einsum("abcd->abcd", tau307, optimize=True)

    tau307 = None

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum("acik,jcbk->ijab", l2, u[o, v, v, o], optimize=True)

    tau67 += 6 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau72 += 6 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau103 += 6 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau119 = zeros((N, N, M, M))

    tau119 -= 6 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau266 = zeros((N, N, M, M))

    tau266 -= np.einsum("ijab->ijab", tau50, optimize=True)

    tau395 = zeros((N, N, M, M))

    tau395 -= 3 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau430 = zeros((N, N, M, M))

    tau430 += np.einsum("ijab->ijab", tau50, optimize=True)

    tau438 += 18 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau50 = None

    tau51 = zeros((M, M))

    tau51 += np.einsum("caij,cbji->ab", l2, t2, optimize=True)

    tau52 = zeros((N, N, M, M))

    tau52 += np.einsum("ac,ijbc->ijab", tau51, u[o, o, v, v], optimize=True)

    tau51 = None

    tau67 += 2 * np.einsum("jiba->ijab", tau52, optimize=True)

    tau101 -= np.einsum("jiab->ijab", tau52, optimize=True)

    tau52 = None

    tau53 = zeros((N, N, M, M))

    tau53 += np.einsum("acki,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau56 = zeros((N, N, M, M))

    tau56 += np.einsum("ijab->ijab", tau53, optimize=True)

    tau72 += 6 * np.einsum("ijab->ijab", tau53, optimize=True)

    tau103 += 6 * np.einsum("ijab->ijab", tau53, optimize=True)

    tau411 = zeros((N, N, M, M))

    tau411 -= np.einsum("ijab->ijab", tau53, optimize=True)

    tau430 += np.einsum("ijab->ijab", tau53, optimize=True)

    tau438 += 18 * np.einsum("ijab->ijab", tau53, optimize=True)

    tau53 = None

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("acki,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau55 = zeros((N, N, M, M))

    tau55 += np.einsum("caik,kjcb->ijab", l2, tau54, optimize=True)

    tau56 -= np.einsum("ijab->ijab", tau55, optimize=True)

    tau67 += 6 * np.einsum("ijab->ijab", tau56, optimize=True)

    tau67 += 6 * np.einsum("jiba->ijab", tau56, optimize=True)

    tau196 = zeros((N, N, M, M))

    tau196 += np.einsum("bckj,kiac->ijab", t2, tau56, optimize=True)

    tau56 = None

    tau197 = zeros((N, N, M, M))

    tau197 += 3 * np.einsum("jiab->ijab", tau196, optimize=True)

    tau196 = None

    tau72 -= 6 * np.einsum("ijab->ijab", tau55, optimize=True)

    tau103 -= 6 * np.einsum("ijab->ijab", tau55, optimize=True)

    tau395 -= np.einsum("ijab->ijab", tau55, optimize=True)

    tau411 += np.einsum("ijab->ijab", tau55, optimize=True)

    tau430 -= np.einsum("ijab->ijab", tau55, optimize=True)

    tau438 -= 18 * np.einsum("ijab->ijab", tau55, optimize=True)

    tau55 = None

    tau127 = zeros((N, N, M, M))

    tau127 += np.einsum("ikca,jkcb->ijab", tau11, tau54, optimize=True)

    tau128 = zeros((N, N, M, M))

    tau128 += np.einsum("acki,kjbc->ijab", t2, tau127, optimize=True)

    tau127 = None

    tau142 = zeros((N, N, M, M))

    tau142 += np.einsum("ijab->ijab", tau128, optimize=True)

    tau128 = None

    tau185 = zeros((N, N, M, M))

    tau185 += np.einsum("caki,kjcb->ijab", l2, tau54, optimize=True)

    tau186 = zeros((N, N, M, M))

    tau186 -= np.einsum("ijab->ijab", tau185, optimize=True)

    tau185 = None

    tau219 = zeros((N, N, M, M))

    tau219 += np.einsum("ijab->ijab", tau54, optimize=True)

    tau232 = zeros((N, N, M, M))

    tau232 += np.einsum("acbd,ijcd->ijab", tau47, tau54, optimize=True)

    tau235 += 2 * np.einsum("ijab->ijab", tau232, optimize=True)

    tau232 = None

    tau251 = zeros((N, N, M, M))

    tau251 += np.einsum("ikca,jkcb->ijab", tau36, tau54, optimize=True)

    tau255 = zeros((N, N, M, M))

    tau255 += 4 * np.einsum("jiba->ijab", tau251, optimize=True)

    tau251 = None

    tau57 = zeros((N, N))

    tau57 += np.einsum("abki,bakj->ij", l2, t2, optimize=True)

    tau58 = zeros((N, N, M, M))

    tau58 += np.einsum("ik,jkab->ijab", tau57, u[o, o, v, v], optimize=True)

    tau57 = None

    tau67 += np.einsum("ijba->ijab", tau58, optimize=True)

    tau67 += np.einsum("jiab->ijab", tau58, optimize=True)

    tau67 -= 2 * np.einsum("jiba->ijab", tau58, optimize=True)

    tau103 += np.einsum("jiab->ijab", tau58, optimize=True)

    tau429 += 3 * np.einsum("ijba->ijab", tau58, optimize=True)

    tau438 += 3 * np.einsum("ijab->ijab", tau58, optimize=True)

    tau58 = None

    tau59 = zeros((M, M))

    tau59 += np.einsum("caij,cbij->ab", l2, t2, optimize=True)

    tau60 = zeros((N, N, M, M))

    tau60 += np.einsum("ac,ijbc->ijab", tau59, u[o, o, v, v], optimize=True)

    tau67 += 12 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau67 -= 6 * np.einsum("jiab->ijab", tau60, optimize=True)

    tau67 += 12 * np.einsum("jiba->ijab", tau60, optimize=True)

    tau89 = zeros((N, N, M, M))

    tau89 += 6 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau101 += 6 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau101 -= 12 * np.einsum("jiab->ijab", tau60, optimize=True)

    tau119 += 6 * np.einsum("ijba->ijab", tau60, optimize=True)

    tau266 += np.einsum("jiab->ijab", tau60, optimize=True)

    tau395 += 3 * np.einsum("jiab->ijab", tau60, optimize=True)

    tau401 = zeros((N, N, M, M))

    tau401 += 3 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau429 += 36 * np.einsum("ijba->ijab", tau60, optimize=True)

    tau429 -= 18 * np.einsum("jiba->ijab", tau60, optimize=True)

    tau430 -= np.einsum("ijba->ijab", tau60, optimize=True)

    tau430 += 2 * np.einsum("jiba->ijab", tau60, optimize=True)

    tau438 -= 18 * np.einsum("ijba->ijab", tau60, optimize=True)

    tau438 += 36 * np.einsum("jiba->ijab", tau60, optimize=True)

    tau439 = zeros((N, N, M, M))

    tau439 += 2 * np.einsum("ijba->ijab", tau60, optimize=True)

    tau439 -= np.einsum("jiba->ijab", tau60, optimize=True)

    tau60 = None

    tau86 = zeros((N, N, M, M))

    tau86 += np.einsum("ac,ibcj->ijab", tau59, u[o, v, v, o], optimize=True)

    tau92 = zeros((N, N, M, M))

    tau92 += 6 * np.einsum("ijab->ijab", tau86, optimize=True)

    tau86 = None

    tau117 = zeros((N, N, M, M))

    tau117 += np.einsum("ac,ibjc->ijab", tau59, u[o, v, o, v], optimize=True)

    tau121 = zeros((N, N, M, M))

    tau121 += 6 * np.einsum("ijab->ijab", tau117, optimize=True)

    tau268 = zeros((N, N, M, M))

    tau268 += np.einsum("jiab->ijab", tau117, optimize=True)

    tau117 = None

    tau408 += np.einsum("ca,cbij->ijab", tau59, t2, optimize=True)

    tau418 = zeros((N, N, M, M))

    tau418 -= np.einsum("cb,acij->ijab", tau59, t2, optimize=True)

    tau62 = zeros((N, N, M, M))

    tau62 += np.einsum("acki,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau65 = zeros((N, N, M, M))

    tau65 += np.einsum("ijab->ijab", tau62, optimize=True)

    tau211 = zeros((N, N, M, M))

    tau211 += np.einsum("ijab->ijab", tau62, optimize=True)

    tau62 = None

    tau63 = zeros((N, N, M, M))

    tau63 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau63 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau64 = zeros((N, N, M, M))

    tau64 += np.einsum("bcjk,kiac->ijab", t2, tau63, optimize=True)

    tau65 -= np.einsum("jiba->ijab", tau64, optimize=True)

    tau66 = zeros((N, N, M, M))

    tau66 += np.einsum("cbkj,kica->ijab", l2, tau65, optimize=True)

    tau67 -= 6 * np.einsum("jiba->ijab", tau66, optimize=True)

    tau68 = zeros((N, N, M, M))

    tau68 += np.einsum("bckj,ikca->ijab", t2, tau67, optimize=True)

    tau67 = None

    tau84 += np.einsum("jiab->ijab", tau68, optimize=True)

    tau68 = None

    tau72 -= 6 * np.einsum("jiba->ijab", tau66, optimize=True)

    tau103 -= 6 * np.einsum("jiba->ijab", tau66, optimize=True)

    tau104 = zeros((M, M))

    tau104 += np.einsum("bcij,ijca->ab", t2, tau103, optimize=True)

    tau103 = None

    tau112 += 2 * np.einsum("ba->ab", tau104, optimize=True)

    tau104 = None

    tau119 += 6 * np.einsum("jiba->ijab", tau66, optimize=True)

    tau266 += np.einsum("jiba->ijab", tau66, optimize=True)

    tau267 = zeros((N, N, M, M))

    tau267 += np.einsum("bckj,kiac->ijab", t2, tau266, optimize=True)

    tau266 = None

    tau268 -= np.einsum("jiab->ijab", tau267, optimize=True)

    tau267 = None

    tau430 -= np.einsum("jiba->ijab", tau66, optimize=True)

    tau437 = zeros((N, N))

    tau437 -= 18 * np.einsum("bakj,kiab->ij", t2, tau430, optimize=True)

    tau430 = None

    tau438 -= 18 * np.einsum("jiba->ijab", tau66, optimize=True)

    tau66 = None

    tau440 = zeros((N, N))

    tau440 -= np.einsum("abik,kjab->ij", t2, tau438, optimize=True)

    tau438 = None

    tau193 = zeros((N, N, M, M))

    tau193 += np.einsum("cbjk,kica->ijab", l2, tau65, optimize=True)

    tau65 = None

    tau194 = zeros((N, N, M, M))

    tau194 -= np.einsum("jiba->ijab", tau193, optimize=True)

    tau193 = None

    tau138 = zeros((N, N, M, M))

    tau138 -= np.einsum("jiba->ijab", tau64, optimize=True)

    tau393 = zeros((N, N, M, M))

    tau393 -= 3 * np.einsum("jiba->ijab", tau64, optimize=True)

    tau64 = None

    tau397 = zeros((N, N, M, M))

    tau397 += 2 * np.einsum("caki,kjbc->ijab", t2, tau63, optimize=True)

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum("abij->ijab", t2, optimize=True)

    tau69 += 6 * np.einsum("abji->ijab", t2, optimize=True)

    tau70 = zeros((M, M))

    tau70 += np.einsum("cbji,ijca->ab", l2, tau69, optimize=True)

    tau69 = None

    tau71 = zeros((N, N, M, M))

    tau71 += np.einsum("ca,ijcb->ijab", tau70, u[o, o, v, v], optimize=True)

    tau72 -= np.einsum("jiba->ijab", tau71, optimize=True)

    tau72 += 2 * np.einsum("ijba->ijab", tau71, optimize=True)

    tau71 = None

    tau73 = zeros((N, N, M, M))

    tau73 += np.einsum("bcjk,ikca->ijab", t2, tau72, optimize=True)

    tau72 = None

    tau84 -= 2 * np.einsum("jiab->ijab", tau73, optimize=True)

    tau73 = None

    tau80 = zeros((N, N, M, M))

    tau80 += np.einsum("ca,ibcj->ijab", tau70, u[o, v, v, o], optimize=True)

    tau70 = None

    tau84 -= 2 * np.einsum("jiab->ijab", tau80, optimize=True)

    tau80 = None

    tau74 = zeros((N, N, M, M))

    tau74 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau75 += np.einsum("ijab->ijab", tau74, optimize=True)

    tau76 = zeros((N, N, M, M))

    tau76 += np.einsum("ikca,kcbj->ijab", tau75, u[o, v, v, o], optimize=True)

    tau84 += 6 * np.einsum("jiba->ijab", tau76, optimize=True)

    tau76 = None

    tau87 = zeros((N, N, M, M))

    tau87 += np.einsum("ikca,kcjb->ijab", tau74, u[o, v, o, v], optimize=True)

    tau92 += 6 * np.einsum("ijba->ijab", tau87, optimize=True)

    tau87 = None

    tau118 = zeros((N, N, M, M))

    tau118 += np.einsum("ikca,kcbj->ijab", tau74, u[o, v, v, o], optimize=True)

    tau121 += 6 * np.einsum("ijba->ijab", tau118, optimize=True)

    tau118 = None

    tau312 = zeros((N, N, M, M))

    tau312 += np.einsum("ijcd,acbd->ijab", tau74, u[v, v, v, v], optimize=True)

    tau313 = zeros((N, N, M, M))

    tau313 += np.einsum("acik,kjbc->ijab", t2, tau312, optimize=True)

    tau312 = None

    r2 += np.einsum("jiab->abij", tau313, optimize=True) / 9

    r2 -= 7 * np.einsum("jiba->abij", tau313, optimize=True) / 9

    tau313 = None

    tau359 = zeros((N, N, M, M))

    tau359 += np.einsum("ijab->ijab", tau74, optimize=True)

    tau377 = zeros((N, N, M, M))

    tau377 += 36 * np.einsum("jkca,ikcb->ijab", tau54, tau74, optimize=True)

    tau409 = zeros((N, N, M, M))

    tau409 -= 3 * np.einsum("ijab->ijab", tau74, optimize=True)

    tau421 = zeros((N, N, M, M))

    tau421 += 24 * np.einsum("ijab->ijab", tau74, optimize=True)

    tau77 = zeros((N, N, M, M))

    tau77 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau77 -= np.einsum("abji->ijab", t2, optimize=True)

    tau78 = zeros((N, N, M, M))

    tau78 += np.einsum("cbkj,ikac->ijab", l2, tau77, optimize=True)

    tau79 = zeros((N, N, M, M))

    tau79 += np.einsum("kiac,kcjb->ijab", tau78, u[o, v, o, v], optimize=True)

    tau84 -= 6 * np.einsum("jiba->ijab", tau79, optimize=True)

    tau79 = None

    tau334 -= 36 * np.einsum("ikca,kjbc->ijab", tau54, tau78, optimize=True)

    tau361 = zeros((N, N, M, M))

    tau361 += np.einsum("bcjk,ikac->ijab", t2, tau78, optimize=True)

    tau362 = zeros((N, N, M, M))

    tau362 -= np.einsum("ijab->ijab", tau361, optimize=True)

    tau361 = None

    tau369 = zeros((N, N, M, M))

    tau369 += np.einsum("cbkj,ikac->ijab", t2, tau78, optimize=True)

    tau78 = None

    tau370 -= np.einsum("ijab->ijab", tau369, optimize=True)

    tau369 = None

    tau373 = zeros((N, N, M, M))

    tau373 += 36 * np.einsum("kjbc,kiac->ijab", tau370, u[o, o, v, v], optimize=True)

    r2 += np.einsum("kjac,kbic->abij", tau370, u[o, v, o, v], optimize=True)

    tau370 = None

    tau130 = zeros((N, N, M, M))

    tau130 += np.einsum("ikac,kjcb->ijab", tau77, u[o, o, v, v], optimize=True)

    tau131 = zeros((N, N, M, M))

    tau131 += np.einsum("ijab->ijab", tau130, optimize=True)

    tau339 = zeros((N, N, M, M))

    tau339 += np.einsum("jiab->ijab", tau130, optimize=True)

    tau130 = None

    tau334 -= 36 * np.einsum("jkac,ikbc->ijab", tau322, tau77, optimize=True)

    tau322 = None

    tau77 = None

    tau81 = zeros((N, N, M, M))

    tau81 -= np.einsum("abij->ijab", t2, optimize=True)

    tau81 += 4 * np.einsum("baij->ijab", t2, optimize=True)

    tau82 = zeros((N, N))

    tau82 += np.einsum("bakj,kiab->ij", l2, tau81, optimize=True)

    tau81 = None

    tau83 = zeros((N, N, M, M))

    tau83 += np.einsum("ki,kajb->ijab", tau82, u[o, v, o, v], optimize=True)

    tau84 += np.einsum("jiba->ijab", tau83, optimize=True)

    tau85 = zeros((N, N, M, M))

    tau85 += np.einsum("bcjk,ikca->ijab", t2, tau84, optimize=True)

    tau84 = None

    tau114 = zeros((N, N, M, M))

    tau114 += 2 * np.einsum("jiba->ijab", tau85, optimize=True)

    tau85 = None

    tau121 += np.einsum("ijba->ijab", tau83, optimize=True)

    tau83 = None

    tau88 = zeros((N, N, M, M))

    tau88 += np.einsum("ki,kjab->ijab", tau82, u[o, o, v, v], optimize=True)

    tau89 += np.einsum("jiab->ijab", tau88, optimize=True)

    tau90 = zeros((N, N, M, M))

    tau90 += np.einsum("bckj,kiac->ijab", t2, tau89, optimize=True)

    tau89 = None

    tau92 -= np.einsum("ijab->ijab", tau90, optimize=True)

    tau90 = None

    tau101 -= 2 * np.einsum("jiba->ijab", tau88, optimize=True)

    tau101 += np.einsum("jiab->ijab", tau88, optimize=True)

    tau119 += np.einsum("ijab->ijab", tau88, optimize=True)

    tau88 = None

    tau120 = zeros((N, N, M, M))

    tau120 += np.einsum("bckj,ikca->ijab", t2, tau119, optimize=True)

    tau119 = None

    tau121 -= np.einsum("ijab->ijab", tau120, optimize=True)

    tau120 = None

    tau91 = zeros((N, N, M, M))

    tau91 += np.einsum("ki,kabj->ijab", tau82, u[o, v, v, o], optimize=True)

    tau92 += np.einsum("ijba->ijab", tau91, optimize=True)

    tau91 = None

    tau93 = zeros((N, N, M, M))

    tau93 += np.einsum("bckj,kica->ijab", t2, tau92, optimize=True)

    tau92 = None

    tau114 += 2 * np.einsum("jiba->ijab", tau93, optimize=True)

    tau93 = None

    tau111 = zeros((M, M))

    tau111 += np.einsum("ji,jabi->ab", tau82, u[o, v, v, o], optimize=True)

    tau82 = None

    tau112 += 2 * np.einsum("ab->ab", tau111, optimize=True)

    tau111 = None

    tau95 = zeros((N, N, M, M))

    tau95 -= np.einsum("abij->ijab", t2, optimize=True)

    tau95 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau96 = zeros((N, N, M, M))

    tau96 += np.einsum("cbkj,ikca->ijab", l2, tau95, optimize=True)

    tau428 = zeros((N, N, M, M))

    tau428 += np.einsum("kiac,kjcb->ijab", tau63, tau96, optimize=True)

    tau429 -= 18 * np.einsum("jiba->ijab", tau428, optimize=True)

    tau439 -= np.einsum("jiba->ijab", tau428, optimize=True)

    tau428 = None

    tau97 = zeros((N, N, M, M))

    tau97 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau97 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau98 = zeros((N, N, M, M))

    tau98 += np.einsum("kica,kjcb->ijab", tau96, tau97, optimize=True)

    tau96 = None

    tau101 += 6 * np.einsum("ijab->ijab", tau98, optimize=True)

    tau98 = None

    tau237 = zeros((N, N, M, M))

    tau237 += np.einsum("bcjk,kica->ijab", t2, tau97, optimize=True)

    tau97 = None

    tau238 = zeros((M, M, M, M))

    tau238 += np.einsum("jiba,ijcd->abcd", tau237, tau36, optimize=True)

    tau237 = None

    tau239 = zeros((N, N, M, M))

    tau239 += np.einsum("dcij,adcb->ijab", t2, tau238, optimize=True)

    tau240 -= 4 * np.einsum("ijab->ijab", tau239, optimize=True)

    tau239 = None

    tau257 = zeros((N, N, M, M))

    tau257 += np.einsum("cdij,adcb->ijab", t2, tau238, optimize=True)

    tau238 = None

    tau258 -= 4 * np.einsum("ijab->ijab", tau257, optimize=True)

    tau257 = None

    tau99 = zeros((N, N, M, M))

    tau99 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau99 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau100 = zeros((N, N, M, M))

    tau100 += np.einsum("bcjk,ikca->ijab", l2, tau99, optimize=True)

    tau99 = None

    tau101 += 6 * np.einsum("jiba->ijab", tau100, optimize=True)

    tau102 = zeros((M, M))

    tau102 += np.einsum("bcji,ijca->ab", t2, tau101, optimize=True)

    tau101 = None

    tau112 -= 2 * np.einsum("ba->ab", tau102, optimize=True)

    tau102 = None

    tau429 -= 18 * np.einsum("jiba->ijab", tau100, optimize=True)

    tau437 -= np.einsum("abkj,kiab->ij", t2, tau429, optimize=True)

    tau429 = None

    tau439 -= np.einsum("jiba->ijab", tau100, optimize=True)

    tau100 = None

    tau440 -= 18 * np.einsum("baik,kjab->ij", t2, tau439, optimize=True)

    tau439 = None

    tau105 = zeros((N, N, M, M))

    tau105 -= np.einsum("abij->ijab", t2, optimize=True)

    tau105 += 10 * np.einsum("abji->ijab", t2, optimize=True)

    tau106 = zeros((M, M))

    tau106 += np.einsum("cbji,ijca->ab", l2, tau105, optimize=True)

    tau107 = zeros((M, M))

    tau107 += np.einsum("dc,cabd->ab", tau106, u[v, v, v, v], optimize=True)

    tau106 = None

    tau112 -= np.einsum("ab->ab", tau107, optimize=True)

    tau107 = None

    tau336 = zeros((M, M, M, M))

    tau336 += np.einsum("cdij,ijab->abcd", l2, tau105, optimize=True)

    tau105 = None

    tau108 = zeros((N, N, M, M))

    tau108 += np.einsum("abij->ijab", t2, optimize=True)

    tau108 += 12 * np.einsum("abji->ijab", t2, optimize=True)

    tau109 = zeros((M, M))

    tau109 += np.einsum("cbji,ijca->ab", l2, tau108, optimize=True)

    tau108 = None

    tau110 = zeros((M, M))

    tau110 += np.einsum("dc,cadb->ab", tau109, u[v, v, v, v], optimize=True)

    tau109 = None

    tau112 += 2 * np.einsum("ab->ab", tau110, optimize=True)

    tau110 = None

    tau113 = zeros((N, N, M, M))

    tau113 += np.einsum("ac,bcij->ijab", tau112, t2, optimize=True)

    tau112 = None

    tau114 += np.einsum("ijba->ijab", tau113, optimize=True)

    tau113 = None

    r2 += np.einsum("ijab->abij", tau114, optimize=True) / 12

    r2 += np.einsum("jiba->abij", tau114, optimize=True) / 12

    tau114 = None

    tau115 = zeros((M, M, M, M))

    tau115 += np.einsum("abij,cdij->abcd", l2, t2, optimize=True)

    tau116 = zeros((N, N, M, M))

    tau116 += np.einsum("acbd,icjd->ijab", tau115, u[o, v, o, v], optimize=True)

    tau121 += 6 * np.einsum("ijab->ijab", tau116, optimize=True)

    tau122 = zeros((N, N, M, M))

    tau122 += np.einsum("bckj,kica->ijab", t2, tau121, optimize=True)

    tau121 = None

    r2 += np.einsum("jiab->abij", tau122, optimize=True) / 6

    r2 += np.einsum("ijba->abij", tau122, optimize=True) / 6

    tau122 = None

    tau268 += np.einsum("jiab->ijab", tau116, optimize=True)

    tau116 = None

    tau269 = zeros((N, N, M, M))

    tau269 += np.einsum("bcjk,ikca->ijab", t2, tau268, optimize=True)

    tau268 = None

    r2 += 5 * np.einsum("jiba->abij", tau269, optimize=True) / 9

    r2 -= 2 * np.einsum("jiab->abij", tau269, optimize=True) / 9

    r2 -= 2 * np.einsum("ijba->abij", tau269, optimize=True) / 9

    r2 += 5 * np.einsum("ijab->abij", tau269, optimize=True) / 9

    tau269 = None

    tau182 = zeros((N, N, M, M))

    tau182 += np.einsum("acbd,icdj->ijab", tau115, u[o, v, v, o], optimize=True)

    tau188 += 3 * np.einsum("ijab->ijab", tau182, optimize=True)

    tau182 = None

    tau208 = zeros((M, M, M, M))

    tau208 += np.einsum("aebf,cefd->abcd", tau115, u[v, v, v, v], optimize=True)

    tau213 = zeros((M, M, M, M))

    tau213 += 5 * np.einsum("abcd->abcd", tau208, optimize=True)

    tau208 = None

    tau275 = zeros((N, N, M, M))

    tau275 += np.einsum("acbd,ijcd->ijab", tau115, tau54, optimize=True)

    tau278 = zeros((N, N, M, M))

    tau278 += np.einsum("ijab->ijab", tau275, optimize=True)

    tau373 += 36 * np.einsum("jiab->ijab", tau275, optimize=True)

    tau275 = None

    tau335 = zeros((M, M, M, M))

    tau335 += np.einsum("aebf,cedf->abcd", tau115, u[v, v, v, v], optimize=True)

    tau341 = zeros((M, M, M, M))

    tau341 += 12 * np.einsum("abcd->abcd", tau335, optimize=True)

    tau335 = None

    tau129 = zeros((N, N, M, M))

    tau129 += np.einsum("acik,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau131 -= np.einsum("ijab->ijab", tau129, optimize=True)

    tau339 -= np.einsum("jiab->ijab", tau129, optimize=True)

    tau129 = None

    tau131 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau132 = zeros((M, M, M, M))

    tau132 += np.einsum("ijcd,ijab->abcd", tau11, tau131, optimize=True)

    tau133 = zeros((N, N, M, M))

    tau133 += np.einsum("dcij,bdca->ijab", t2, tau132, optimize=True)

    tau142 -= np.einsum("ijab->ijab", tau133, optimize=True)

    tau133 = None

    tau148 = zeros((N, N, M, M))

    tau148 += np.einsum("cdij,bdca->ijab", t2, tau132, optimize=True)

    tau132 = None

    tau157 = zeros((N, N, M, M))

    tau157 -= np.einsum("ijab->ijab", tau148, optimize=True)

    tau148 = None

    tau162 = zeros((N, N, M, M))

    tau162 += np.einsum("klab,iklj->ijab", tau131, tau16, optimize=True)

    tau163 -= np.einsum("ijab->ijab", tau162, optimize=True)

    tau164 = zeros((N, N, M, M))

    tau164 += np.einsum("bckj,kiac->ijab", t2, tau163, optimize=True)

    tau163 = None

    tau169 += np.einsum("jiba->ijab", tau164, optimize=True)

    tau164 = None

    tau334 -= 36 * np.einsum("jiba->ijab", tau162, optimize=True)

    tau162 = None

    tau166 = zeros((N, N, M, M))

    tau166 += np.einsum("klab,ikjl->ijab", tau131, tau16, optimize=True)

    tau167 -= np.einsum("ijab->ijab", tau166, optimize=True)

    tau166 = None

    tau168 = zeros((N, N, M, M))

    tau168 += np.einsum("bckj,kiac->ijab", t2, tau167, optimize=True)

    tau167 = None

    tau169 += np.einsum("ijba->ijab", tau168, optimize=True)

    tau168 = None

    r2 += 7 * np.einsum("ijab->abij", tau169, optimize=True) / 18

    r2 += np.einsum("ijba->abij", tau169, optimize=True) / 9

    tau169 = None

    tau173 = zeros((N, N, M, M))

    tau173 += np.einsum("klab,kilj->ijab", tau131, tau16, optimize=True)

    tau174 -= np.einsum("ijab->ijab", tau173, optimize=True)

    tau173 = None

    tau175 = zeros((N, N, M, M))

    tau175 += np.einsum("bckj,kiac->ijab", t2, tau174, optimize=True)

    tau174 = None

    tau180 += np.einsum("jiba->ijab", tau175, optimize=True)

    tau175 = None

    tau177 = zeros((N, N, M, M))

    tau177 += np.einsum("klab,kijl->ijab", tau131, tau16, optimize=True)

    tau178 -= np.einsum("ijab->ijab", tau177, optimize=True)

    tau179 = zeros((N, N, M, M))

    tau179 += np.einsum("bckj,kiac->ijab", t2, tau178, optimize=True)

    tau178 = None

    tau180 += np.einsum("ijba->ijab", tau179, optimize=True)

    tau179 = None

    r2 += np.einsum("ijab->abij", tau180, optimize=True) / 9

    r2 += 7 * np.einsum("ijba->abij", tau180, optimize=True) / 18

    tau180 = None

    tau358 -= 36 * np.einsum("ijba->ijab", tau177, optimize=True)

    tau177 = None

    tau217 = zeros((N, N, M, M))

    tau217 += np.einsum("kiac,kjbc->ijab", tau131, tau24, optimize=True)

    tau221 -= 3 * np.einsum("jiba->ijab", tau217, optimize=True)

    tau334 -= 36 * np.einsum("jiba->ijab", tau217, optimize=True)

    tau217 = None

    tau233 = zeros((N, N, M, M))

    tau233 += np.einsum("kiac,kjbc->ijab", tau131, tau3, optimize=True)

    tau235 += 2 * np.einsum("jiba->ijab", tau233, optimize=True)

    tau255 -= 6 * np.einsum("jiba->ijab", tau233, optimize=True)

    tau233 = None

    tau134 = zeros((N, N, M, M))

    tau134 += np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau135 = zeros((N, N, M, M))

    tau135 += np.einsum("ackj,kicb->ijab", t2, tau134, optimize=True)

    tau136 = zeros((N, N, M, M))

    tau136 += np.einsum("ikac,jkcb->ijab", tau135, u[o, o, v, v], optimize=True)

    tau140 = zeros((N, N, M, M))

    tau140 += np.einsum("jiab->ijab", tau136, optimize=True)

    tau136 = None

    tau144 = zeros((N, N, M, M))

    tau144 += np.einsum("ikac,kbjc->ijab", tau135, u[o, v, o, v], optimize=True)

    tau135 = None

    tau157 -= np.einsum("ijab->ijab", tau144, optimize=True)

    tau144 = None

    tau227 = zeros((N, N, M, M))

    tau227 += np.einsum("kiac,kjbc->ijab", tau131, tau134, optimize=True)

    tau229 = zeros((N, N, M, M))

    tau229 -= 6 * np.einsum("jiba->ijab", tau227, optimize=True)

    tau263 = zeros((N, N, M, M))

    tau263 += 2 * np.einsum("jiba->ijab", tau227, optimize=True)

    tau227 = None

    tau351 = zeros((N, N, M, M))

    tau351 += 3 * np.einsum("ijab->ijab", tau134, optimize=True)

    tau360 = zeros((N, N, M, M))

    tau360 += np.einsum("acik,kjcb->ijab", t2, tau134, optimize=True)

    tau362 += np.einsum("ijab->ijab", tau360, optimize=True)

    tau360 = None

    tau366 = zeros((N, N, M, M))

    tau366 += 36 * np.einsum("kjbc,kiac->ijab", tau362, u[o, o, v, v], optimize=True)

    r2 += np.einsum("kibc,kajc->abij", tau362, u[o, v, o, v], optimize=True)

    tau362 = None

    tau137 = zeros((N, N, M, M))

    tau137 += np.einsum("caik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau138 += np.einsum("ijab->ijab", tau137, optimize=True)

    tau137 = None

    tau139 = zeros((N, N, M, M))

    tau139 += np.einsum("jkcb,ikca->ijab", tau11, tau138, optimize=True)

    tau140 += np.einsum("jiba->ijab", tau139, optimize=True)

    tau139 = None

    tau141 = zeros((N, N, M, M))

    tau141 += np.einsum("bckj,kiac->ijab", t2, tau140, optimize=True)

    tau140 = None

    tau142 += np.einsum("ijba->ijab", tau141, optimize=True)

    tau141 = None

    r2 += 11 * np.einsum("ijab->abij", tau142, optimize=True) / 18

    r2 += 2 * np.einsum("ijba->abij", tau142, optimize=True) / 9

    tau142 = None

    tau226 = zeros((N, N, M, M))

    tau226 += np.einsum("acbd,ijcd->ijab", tau115, tau138, optimize=True)

    tau229 += 6 * np.einsum("ijab->ijab", tau226, optimize=True)

    tau226 = None

    tau245 = zeros((N, N, M, M))

    tau245 += np.einsum("ijcd,acbd->ijab", tau138, tau47, optimize=True)

    tau249 = zeros((N, N, M, M))

    tau249 += 3 * np.einsum("ijab->ijab", tau245, optimize=True)

    tau358 += 36 * np.einsum("jiab->ijab", tau245, optimize=True)

    tau245 = None

    tau247 = zeros((N, N, M, M))

    tau247 += np.einsum("ikca,jkcb->ijab", tau138, tau36, optimize=True)

    tau249 += 2 * np.einsum("ijab->ijab", tau247, optimize=True)

    tau247 = None

    tau358 += 36 * np.einsum("jkca,ikcb->ijab", tau138, tau75, optimize=True)

    tau366 += 36 * np.einsum("jkca,ikcb->ijab", tau138, tau74, optimize=True)

    tau145 = zeros((N, N, M, M))

    tau145 += np.einsum("caik,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau146 = zeros((N, N, M, M))

    tau146 += np.einsum("ikca,jkcb->ijab", tau11, tau145, optimize=True)

    tau147 = zeros((N, N, M, M))

    tau147 += np.einsum("ackj,kibc->ijab", t2, tau146, optimize=True)

    tau157 += np.einsum("ijab->ijab", tau147, optimize=True)

    tau147 = None

    tau358 += 36 * np.einsum("ijba->ijab", tau146, optimize=True)

    tau146 = None

    tau223 = zeros((N, N, M, M))

    tau223 += np.einsum("jkcb,ikca->ijab", tau145, tau36, optimize=True)

    tau229 += 4 * np.einsum("jiba->ijab", tau223, optimize=True)

    tau223 = None

    tau262 = zeros((N, N, M, M))

    tau262 += np.einsum("ijcd,acbd->ijab", tau145, tau47, optimize=True)

    tau263 += 2 * np.einsum("ijab->ijab", tau262, optimize=True)

    tau262 = None

    tau270 = zeros((N, N, M, M))

    tau270 += np.einsum("acbd,ijcd->ijab", tau115, tau145, optimize=True)

    tau273 = zeros((N, N, M, M))

    tau273 += np.einsum("ijab->ijab", tau270, optimize=True)

    tau366 += 36 * np.einsum("jiab->ijab", tau270, optimize=True)

    tau270 = None

    tau348 = zeros((N, N, M, M))

    tau348 += np.einsum("jkca,ikcb->ijab", tau145, tau74, optimize=True)

    tau150 = zeros((N, N, M, M))

    tau150 += np.einsum("caki,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau153 = zeros((N, N, M, M))

    tau153 += np.einsum("ijab->ijab", tau150, optimize=True)

    tau402 = zeros((N, N, M, M))

    tau402 -= np.einsum("ijab->ijab", tau150, optimize=True)

    tau150 = None

    tau151 = zeros((N, N, M, M))

    tau151 -= np.einsum("abij->ijab", t2, optimize=True)

    tau151 += 2 * np.einsum("baij->ijab", t2, optimize=True)

    tau152 = zeros((N, N, M, M))

    tau152 += np.einsum("kiac,kjcb->ijab", tau151, u[o, o, v, v], optimize=True)

    tau153 -= np.einsum("ijab->ijab", tau152, optimize=True)

    tau154 = zeros((N, N, M, M))

    tau154 += np.einsum("jkcb,ikca->ijab", tau11, tau153, optimize=True)

    tau11 = None

    tau155 += np.einsum("jiba->ijab", tau154, optimize=True)

    tau154 = None

    tau156 = zeros((N, N, M, M))

    tau156 += np.einsum("bckj,kiac->ijab", t2, tau155, optimize=True)

    tau155 = None

    tau157 += np.einsum("jiba->ijab", tau156, optimize=True)

    tau156 = None

    r2 += 2 * np.einsum("ijab->abij", tau157, optimize=True) / 9

    r2 += 11 * np.einsum("ijba->abij", tau157, optimize=True) / 18

    tau157 = None

    tau216 = zeros((N, N, M, M))

    tau216 += np.einsum("ijcd,acbd->ijab", tau153, tau47, optimize=True)

    tau221 += 3 * np.einsum("ijab->ijab", tau216, optimize=True)

    tau334 += 36 * np.einsum("ijab->ijab", tau216, optimize=True)

    tau216 = None

    tau218 = zeros((N, N, M, M))

    tau218 += np.einsum("ikca,jkcb->ijab", tau153, tau36, optimize=True)

    tau221 += 2 * np.einsum("ijab->ijab", tau218, optimize=True)

    tau218 = None

    tau254 = zeros((N, N, M, M))

    tau254 += np.einsum("acbd,ijcd->ijab", tau115, tau153, optimize=True)

    tau255 += 6 * np.einsum("ijab->ijab", tau254, optimize=True)

    tau254 = None

    tau334 += 36 * np.einsum("ikca,jkcb->ijab", tau153, tau75, optimize=True)

    tau373 += 36 * np.einsum("jkca,ikcb->ijab", tau153, tau74, optimize=True)

    tau153 = None

    tau402 += np.einsum("ijab->ijab", tau152, optimize=True)

    tau152 = None

    tau317 = zeros((N, N, M, M))

    tau317 += np.einsum("bcjk,kiac->ijab", l2, tau151, optimize=True)

    tau318 = zeros((N, N, M, M))

    tau318 += np.einsum("bcjk,ikac->ijab", t2, tau317, optimize=True)

    tau317 = None

    tau321 -= np.einsum("jiba->ijab", tau318, optimize=True)

    tau318 = None

    tau328 = zeros((N, N, M, M))

    tau328 += np.einsum("caki,kjbc->ijab", l2, tau151, optimize=True)

    tau151 = None

    tau334 -= 36 * np.einsum("jicd,cbad->ijab", tau328, u[v, v, v, v], optimize=True)

    tau328 = None

    tau184 = zeros((N, N, M, M))

    tau184 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau186 += np.einsum("ijab->ijab", tau184, optimize=True)

    tau184 = None

    tau187 = zeros((N, N, M, M))

    tau187 += np.einsum("bckj,ikca->ijab", t2, tau186, optimize=True)

    tau186 = None

    tau188 += 3 * np.einsum("ijab->ijab", tau187, optimize=True)

    tau187 = None

    tau189 = zeros((N, N, M, M))

    tau189 += np.einsum("bckj,kica->ijab", t2, tau188, optimize=True)

    tau188 = None

    r2 -= np.einsum("jiba->abij", tau189, optimize=True) / 9

    r2 -= 2 * np.einsum("jiab->abij", tau189, optimize=True) / 9

    r2 -= 2 * np.einsum("ijba->abij", tau189, optimize=True) / 9

    r2 -= np.einsum("ijab->abij", tau189, optimize=True) / 9

    tau189 = None

    tau192 = zeros((N, N, M, M))

    tau192 += np.einsum("acki,jcbk->ijab", l2, u[o, v, v, o], optimize=True)

    tau194 += np.einsum("ijab->ijab", tau192, optimize=True)

    tau192 = None

    tau195 = zeros((N, N, M, M))

    tau195 += np.einsum("bcjk,ikca->ijab", t2, tau194, optimize=True)

    tau194 = None

    tau197 += np.einsum("jiab->ijab", tau195, optimize=True)

    tau195 = None

    tau198 = zeros((N, N, M, M))

    tau198 += np.einsum("bckj,ikca->ijab", t2, tau197, optimize=True)

    tau197 = None

    tau202 += np.einsum("ijab->ijab", tau198, optimize=True)

    tau198 = None

    r2 -= 2 * np.einsum("ijab->abij", tau202, optimize=True) / 9

    r2 -= np.einsum("ijba->abij", tau202, optimize=True) / 9

    r2 -= np.einsum("jiab->abij", tau202, optimize=True) / 9

    r2 -= 2 * np.einsum("jiba->abij", tau202, optimize=True) / 9

    tau202 = None

    tau203 = zeros((N, N, M, M))

    tau203 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    tau204 = zeros((N, N, M, M))

    tau204 += np.einsum("ackj,kicb->ijab", t2, tau203, optimize=True)

    tau205 = zeros((N, N, M, M))

    tau205 += np.einsum("ikac,kbjc->ijab", tau204, u[o, v, o, v], optimize=True)

    tau240 -= 4 * np.einsum("ijab->ijab", tau205, optimize=True)

    tau205 = None

    tau244 = zeros((N, N, M, M))

    tau244 += np.einsum("ikac,jkcb->ijab", tau204, u[o, o, v, v], optimize=True)

    tau204 = None

    tau249 += 2 * np.einsum("ijba->ijab", tau244, optimize=True)

    tau244 = None

    tau246 = zeros((N, N, M, M))

    tau246 += np.einsum("kiac,kjbc->ijab", tau131, tau203, optimize=True)

    tau249 -= 3 * np.einsum("jiba->ijab", tau246, optimize=True)

    tau358 -= 36 * np.einsum("ijba->ijab", tau246, optimize=True)

    tau246 = None

    tau345 = zeros((N, N, M, M))

    tau345 += np.einsum("ijab->ijab", tau203, optimize=True)

    tau209 = zeros((M, M, M, M))

    tau209 += np.einsum("abij,jidc->abcd", t2, u[o, o, v, v], optimize=True)

    tau210 = zeros((M, M, M, M))

    tau210 += np.einsum("cefd,eabf->abcd", tau209, tau47, optimize=True)

    tau47 = None

    tau213 += 5 * np.einsum("abcd->abcd", tau210, optimize=True)

    tau210 = None

    tau280 = zeros((N, N, M, M))

    tau280 += np.einsum("acbd,ijcd->ijab", tau209, tau24, optimize=True)

    tau281 = zeros((N, N, M, M))

    tau281 += np.einsum("acki,kjbc->ijab", t2, tau280, optimize=True)

    tau280 = None

    r2 += np.einsum("ijab->abij", tau281, optimize=True) / 9

    r2 -= 7 * np.einsum("ijba->abij", tau281, optimize=True) / 9

    tau281 = None

    tau285 = zeros((N, N, M, M))

    tau285 += np.einsum("ijcd,acbd->ijab", tau203, tau209, optimize=True)

    tau286 = zeros((N, N, M, M))

    tau286 += np.einsum("ackj,kibc->ijab", t2, tau285, optimize=True)

    tau285 = None

    tau287 = zeros((N, N, M, M))

    tau287 += np.einsum("ijab->ijab", tau286, optimize=True)

    tau286 = None

    tau290 = zeros((N, N, M, M))

    tau290 += np.einsum("ijcd,acdb->ijab", tau203, tau209, optimize=True)

    tau291 = zeros((N, N, M, M))

    tau291 += np.einsum("ijab->ijab", tau290, optimize=True)

    tau290 = None

    tau294 = zeros((N, N, M, M))

    tau294 += np.einsum("acdb,ijcd->ijab", tau209, tau3, optimize=True)

    tau295 += np.einsum("ijab->ijab", tau294, optimize=True)

    tau294 = None

    tau296 = zeros((N, N, M, M))

    tau296 += np.einsum("bckj,kiac->ijab", t2, tau295, optimize=True)

    tau295 = None

    tau297 = zeros((N, N, M, M))

    tau297 += np.einsum("jiba->ijab", tau296, optimize=True)

    tau296 = None

    tau299 = zeros((N, N, M, M))

    tau299 += np.einsum("acdb,ijcd->ijab", tau209, tau24, optimize=True)

    tau300 += np.einsum("ijab->ijab", tau299, optimize=True)

    tau299 = None

    tau301 = zeros((N, N, M, M))

    tau301 += np.einsum("bckj,kiac->ijab", t2, tau300, optimize=True)

    tau300 = None

    tau306 = zeros((N, N, M, M))

    tau306 += np.einsum("jiba->ijab", tau301, optimize=True)

    tau301 = None

    tau303 = zeros((N, N, M, M))

    tau303 += np.einsum("ijcd,acdb->ijab", tau134, tau209, optimize=True)

    tau304 = zeros((N, N, M, M))

    tau304 += np.einsum("ijab->ijab", tau303, optimize=True)

    tau303 = None

    tau308 = zeros((M, M, M, M))

    tau308 += np.einsum("eabf,cefd->abcd", tau115, tau209, optimize=True)

    tau115 = None

    tau309 += np.einsum("abcd->abcd", tau308, optimize=True)

    tau308 = None

    tau310 = zeros((N, N, M, M))

    tau310 += np.einsum("dcij,cabd->ijab", t2, tau309, optimize=True)

    r2 += 23 * np.einsum("ijab->abij", tau310, optimize=True) / 36

    r2 += 5 * np.einsum("ijba->abij", tau310, optimize=True) / 18

    tau310 = None

    tau311 = zeros((N, N, M, M))

    tau311 += np.einsum("cdij,cabd->ijab", t2, tau309, optimize=True)

    tau309 = None

    r2 += 5 * np.einsum("ijab->abij", tau311, optimize=True) / 18

    r2 += 23 * np.einsum("ijba->abij", tau311, optimize=True) / 36

    tau311 = None

    tau337 = zeros((M, M, M, M))

    tau337 += np.einsum("aebf,dfec->abcd", tau209, tau336, optimize=True)

    tau336 = None

    tau341 += np.einsum("cdab->abcd", tau337, optimize=True)

    tau337 = None

    tau211 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau212 = zeros((M, M, M, M))

    tau212 += np.einsum("ijab,ijcd->abcd", tau211, tau36, optimize=True)

    tau211 = None

    tau213 += 4 * np.einsum("cdab->abcd", tau212, optimize=True)

    tau212 = None

    tau214 = zeros((N, N, M, M))

    tau214 += np.einsum("cdij,cabd->ijab", t2, tau213, optimize=True)

    tau240 += np.einsum("ijab->ijab", tau214, optimize=True)

    tau214 = None

    tau243 = zeros((N, N, M, M))

    tau243 += np.einsum("dcij,cabd->ijab", t2, tau213, optimize=True)

    tau213 = None

    tau258 += np.einsum("ijab->ijab", tau243, optimize=True)

    tau243 = None

    tau219 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau220 = zeros((N, N, M, M))

    tau220 += np.einsum("kiac,kjbc->ijab", tau219, tau3, optimize=True)

    tau3 = None

    tau221 += 3 * np.einsum("jiba->ijab", tau220, optimize=True)

    tau220 = None

    tau222 = zeros((N, N, M, M))

    tau222 += np.einsum("bckj,ikca->ijab", t2, tau221, optimize=True)

    tau221 = None

    tau240 += 2 * np.einsum("jiba->ijab", tau222, optimize=True)

    tau222 = None

    tau228 = zeros((N, N, M, M))

    tau228 += np.einsum("kjbc,kiac->ijab", tau203, tau219, optimize=True)

    tau203 = None

    tau229 += 6 * np.einsum("jiba->ijab", tau228, optimize=True)

    tau263 -= 2 * np.einsum("jiba->ijab", tau228, optimize=True)

    tau228 = None

    tau234 = zeros((N, N, M, M))

    tau234 += np.einsum("kiac,kjbc->ijab", tau219, tau24, optimize=True)

    tau24 = None

    tau235 -= 2 * np.einsum("jiba->ijab", tau234, optimize=True)

    tau236 = zeros((N, N, M, M))

    tau236 += np.einsum("bcjk,ikca->ijab", t2, tau235, optimize=True)

    tau235 = None

    tau240 += np.einsum("jiba->ijab", tau236, optimize=True)

    tau236 = None

    tau255 += 6 * np.einsum("jiba->ijab", tau234, optimize=True)

    tau234 = None

    tau248 = zeros((N, N, M, M))

    tau248 += np.einsum("kjbc,kiac->ijab", tau134, tau219, optimize=True)

    tau134 = None

    tau249 += 3 * np.einsum("jiba->ijab", tau248, optimize=True)

    tau248 = None

    tau250 = zeros((N, N, M, M))

    tau250 += np.einsum("bckj,ikca->ijab", t2, tau249, optimize=True)

    tau249 = None

    tau258 += 2 * np.einsum("ijba->ijab", tau250, optimize=True)

    tau250 = None

    tau333 = zeros((N, N, M, M))

    tau333 += np.einsum("kilj,klab->ijab", tau16, tau219, optimize=True)

    tau334 += 36 * np.einsum("jiba->ijab", tau333, optimize=True)

    tau373 += 36 * np.einsum("ijba->ijab", tau333, optimize=True)

    tau333 = None

    tau338 = zeros((M, M, M, M))

    tau338 += np.einsum("ijab,ijcd->abcd", tau219, tau75, optimize=True)

    tau341 += 12 * np.einsum("cdab->abcd", tau338, optimize=True)

    tau338 = None

    tau357 = zeros((N, N, M, M))

    tau357 += np.einsum("ikjl,klab->ijab", tau16, tau219, optimize=True)

    tau358 += 36 * np.einsum("ijba->ijab", tau357, optimize=True)

    tau366 += 36 * np.einsum("ijba->ijab", tau357, optimize=True)

    tau357 = None

    tau407 -= 36 * np.einsum("lkab,ijab->ijkl", tau219, tau74, optimize=True)

    tau224 = zeros((N, N, M, M))

    tau224 += np.einsum("acki,bcjk->ijab", l2, t2, optimize=True)

    tau225 = zeros((N, N, M, M))

    tau225 += np.einsum("acbd,ijcd->ijab", tau209, tau224, optimize=True)

    tau229 -= np.einsum("jiba->ijab", tau225, optimize=True)

    tau225 = None

    tau230 = zeros((N, N, M, M))

    tau230 += np.einsum("bckj,ikca->ijab", t2, tau229, optimize=True)

    tau229 = None

    tau240 += np.einsum("ijba->ijab", tau230, optimize=True)

    tau230 = None

    r2 += np.einsum("ijab->abij", tau240, optimize=True) / 9

    r2 += np.einsum("ijba->abij", tau240, optimize=True) / 18

    tau240 = None

    tau351 -= np.einsum("ijab->ijab", tau224, optimize=True)

    tau224 = None

    tau358 += 12 * np.einsum("kibc,kjac->ijab", tau219, tau351, optimize=True)

    tau351 = None

    tau252 = zeros((N, N, M, M))

    tau252 += np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau253 = zeros((N, N, M, M))

    tau253 += np.einsum("acbd,ijcd->ijab", tau209, tau252, optimize=True)

    tau255 -= np.einsum("jiba->ijab", tau253, optimize=True)

    tau253 = None

    tau256 = zeros((N, N, M, M))

    tau256 += np.einsum("bckj,ikca->ijab", t2, tau255, optimize=True)

    tau255 = None

    tau258 += np.einsum("jiba->ijab", tau256, optimize=True)

    tau256 = None

    r2 += np.einsum("ijab->abij", tau258, optimize=True) / 18

    r2 += np.einsum("ijba->abij", tau258, optimize=True) / 9

    tau258 = None

    tau383 = zeros((N, N, M, M))

    tau383 += np.einsum("caik,kjcb->ijab", t2, tau252, optimize=True)

    tau252 = None

    tau384 += np.einsum("ijab->ijab", tau383, optimize=True)

    tau383 = None

    tau385 = zeros((N, N, N, N))

    tau385 += np.einsum("ijab,klba->ijkl", tau384, u[o, o, v, v], optimize=True)

    tau384 = None

    tau407 += 4 * np.einsum("jlki->ijkl", tau385, optimize=True)

    tau407 += 2 * np.einsum("jlik->ijkl", tau385, optimize=True)

    tau385 = None

    tau260 = zeros((N, N, M, M))

    tau260 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau261 = zeros((N, N, M, M))

    tau261 += np.einsum("ijcd,acbd->ijab", tau260, u[v, v, v, v], optimize=True)

    tau263 -= np.einsum("jiba->ijab", tau261, optimize=True)

    tau261 = None

    tau264 = zeros((N, N, M, M))

    tau264 += np.einsum("bcjk,ikca->ijab", t2, tau263, optimize=True)

    tau263 = None

    tau265 -= np.einsum("jiba->ijab", tau264, optimize=True)

    tau264 = None

    r2 -= np.einsum("jiab->abij", tau265, optimize=True) / 18

    r2 -= np.einsum("jiba->abij", tau265, optimize=True) / 9

    tau265 = None

    tau302 = zeros((N, N, M, M))

    tau302 += np.einsum("ijcd,acdb->ijab", tau260, u[v, v, v, v], optimize=True)

    tau304 += np.einsum("ijab->ijab", tau302, optimize=True)

    tau302 = None

    tau305 = zeros((N, N, M, M))

    tau305 += np.einsum("bckj,kiac->ijab", t2, tau304, optimize=True)

    tau304 = None

    tau306 += np.einsum("ijba->ijab", tau305, optimize=True)

    tau305 = None

    r2 += 2 * np.einsum("ijab->abij", tau306, optimize=True) / 9

    r2 += 19 * np.einsum("ijba->abij", tau306, optimize=True) / 36

    tau306 = None

    tau271 = zeros((N, N, M, M))

    tau271 += np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    tau272 = zeros((N, N, M, M))

    tau272 += np.einsum("kiac,kjbc->ijab", tau131, tau271, optimize=True)

    tau273 -= np.einsum("jiba->ijab", tau272, optimize=True)

    tau274 = zeros((N, N, M, M))

    tau274 += np.einsum("bcjk,ikca->ijab", t2, tau273, optimize=True)

    tau273 = None

    r2 += 2 * np.einsum("ijba->abij", tau274, optimize=True) / 9

    r2 -= 5 * np.einsum("ijab->abij", tau274, optimize=True) / 9

    tau274 = None

    tau366 -= 36 * np.einsum("ijba->ijab", tau272, optimize=True)

    tau272 = None

    tau316 = zeros((N, N, M, M))

    tau316 += np.einsum("klab,lijk->ijab", tau271, u[o, o, o, o], optimize=True)

    tau334 -= 72 * np.einsum("jiab->ijab", tau316, optimize=True)

    tau377 += 36 * np.einsum("ijab->ijab", tau316, optimize=True)

    tau316 = None

    tau332 = zeros((N, N, M, M))

    tau332 += 18 * np.einsum("ijab->ijab", tau271, optimize=True)

    tau345 += np.einsum("ijab->ijab", tau271, optimize=True)

    tau346 = zeros((N, N, M, M))

    tau346 += np.einsum("bckj,kica->ijab", t2, tau345, optimize=True)

    tau347 += np.einsum("ijba->ijab", tau346, optimize=True)

    tau346 = None

    tau358 += 36 * np.einsum("bcad,ijcd->ijab", tau209, tau345, optimize=True)

    tau345 = None

    tau348 += np.einsum("bcda,ijcd->ijab", tau209, tau271, optimize=True)

    tau348 += np.einsum("kibc,kjac->ijab", tau219, tau271, optimize=True)

    tau363 += np.einsum("ijab->ijab", tau271, optimize=True)

    tau366 += 36 * np.einsum("kilj,lkab->ijab", tau19, tau363, optimize=True)

    tau363 = None

    tau366 += 36 * np.einsum("bcad,ijcd->ijab", tau209, tau271, optimize=True)

    tau271 = None

    tau276 = zeros((N, N, M, M))

    tau276 += np.einsum("acik,cbkj->ijab", l2, t2, optimize=True)

    tau277 = zeros((N, N, M, M))

    tau277 += np.einsum("kiac,kjbc->ijab", tau131, tau276, optimize=True)

    tau131 = None

    tau278 -= np.einsum("jiba->ijab", tau277, optimize=True)

    tau279 = zeros((N, N, M, M))

    tau279 += np.einsum("bcjk,ikca->ijab", t2, tau278, optimize=True)

    tau278 = None

    r2 -= 5 * np.einsum("jiba->abij", tau279, optimize=True) / 9

    r2 += 2 * np.einsum("jiab->abij", tau279, optimize=True) / 9

    tau279 = None

    tau373 -= 36 * np.einsum("ijba->ijab", tau277, optimize=True)

    tau277 = None

    tau319 += np.einsum("ijab->ijab", tau276, optimize=True)

    tau320 = zeros((N, N, M, M))

    tau320 += np.einsum("bckj,kica->ijab", t2, tau319, optimize=True)

    tau321 += np.einsum("jiba->ijab", tau320, optimize=True)

    tau320 = None

    tau334 -= 36 * np.einsum("kibc,kjac->ijab", tau321, tau63, optimize=True)

    tau63 = None

    tau377 += 36 * np.einsum("kjbc,kica->ijab", tau321, u[o, o, v, v], optimize=True)

    r2 += np.einsum("kjbc,kaci->abij", tau321, u[o, v, v, o], optimize=True)

    tau321 = None

    tau334 += 36 * np.einsum("bcad,jicd->ijab", tau209, tau319, optimize=True)

    tau417 -= 36 * np.einsum("jkab,ilab->ijkl", tau145, tau319, optimize=True)

    tau145 = None

    tau319 = None

    tau327 = zeros((N, N, M, M))

    tau327 += 24 * np.einsum("ijab->ijab", tau276, optimize=True)

    tau373 += 36 * np.einsum("bcad,ijcd->ijab", tau209, tau276, optimize=True)

    tau377 += 36 * np.einsum("bcda,ijcd->ijab", tau209, tau276, optimize=True)

    tau377 += 36 * np.einsum("kibc,kjac->ijab", tau219, tau276, optimize=True)

    tau378 = zeros((N, N, M, M))

    tau378 += np.einsum("acik,kjcb->ijab", t2, tau276, optimize=True)

    tau407 += 72 * np.einsum("jlba,kiab->ijkl", tau378, u[o, o, v, v], optimize=True)

    tau418 += 2 * np.einsum("ijab->ijab", tau378, optimize=True)

    tau378 = None

    r2 += np.einsum("ijdc,bacd->abij", tau418, u[v, v, v, v], optimize=True)

    tau418 = None

    tau379 = zeros((N, N, M, M))

    tau379 += np.einsum("caik,kjcb->ijab", t2, tau276, optimize=True)

    tau380 += np.einsum("ijab->ijab", tau379, optimize=True)

    tau381 = zeros((N, N, N, N))

    tau381 += np.einsum("ijab,klba->ijkl", tau380, u[o, o, v, v], optimize=True)

    tau380 = None

    tau407 -= 28 * np.einsum("jlki->ijkl", tau381, optimize=True)

    tau407 += 4 * np.einsum("jlik->ijkl", tau381, optimize=True)

    tau381 = None

    tau408 += np.einsum("ijab->ijab", tau379, optimize=True)

    tau379 = None

    r2 -= np.einsum("ijdc,bacd->abij", tau408, u[v, v, v, v], optimize=True)

    tau408 = None

    tau390 = zeros((N, N, M, M))

    tau390 -= 3 * np.einsum("ijab->ijab", tau276, optimize=True)

    tau276 = None

    tau282 = zeros((N, N, M, M))

    tau282 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau283 = zeros((N, N, M, M))

    tau283 += np.einsum("ijcd,acbd->ijab", tau282, u[v, v, v, v], optimize=True)

    tau284 = zeros((N, N, M, M))

    tau284 += np.einsum("acik,kjbc->ijab", t2, tau283, optimize=True)

    tau283 = None

    tau287 += np.einsum("ijab->ijab", tau284, optimize=True)

    tau284 = None

    r2 -= 7 * np.einsum("ijab->abij", tau287, optimize=True) / 9

    r2 += np.einsum("ijba->abij", tau287, optimize=True) / 9

    tau287 = None

    tau367 += np.einsum("ijab->ijab", tau282, optimize=True)

    tau373 += 36 * np.einsum("ijcd,cbda->ijab", tau367, u[v, v, v, v], optimize=True)

    tau367 = None

    tau404 = zeros((N, N, M, M))

    tau404 -= 3 * np.einsum("ijab->ijab", tau282, optimize=True)

    tau417 += 36 * np.einsum("ilab,kajb->ijkl", tau282, u[o, v, o, v], optimize=True)

    tau282 = None

    tau288 = zeros((N, N, M, M))

    tau288 += np.einsum("caik,cbjk->ijab", l2, t2, optimize=True)

    tau289 = zeros((N, N, M, M))

    tau289 += np.einsum("ijcd,acdb->ijab", tau288, u[v, v, v, v], optimize=True)

    tau291 += np.einsum("ijab->ijab", tau289, optimize=True)

    tau289 = None

    tau292 = zeros((N, N, M, M))

    tau292 += np.einsum("bckj,kiac->ijab", t2, tau291, optimize=True)

    tau291 = None

    tau297 += np.einsum("ijba->ijab", tau292, optimize=True)

    tau292 = None

    r2 += 19 * np.einsum("ijab->abij", tau297, optimize=True) / 36

    r2 += 2 * np.einsum("ijba->abij", tau297, optimize=True) / 9

    tau297 = None

    tau358 += 36 * np.einsum("ijcd,bcad->ijab", tau288, u[v, v, v, v], optimize=True)

    tau359 += np.einsum("ijab->ijab", tau288, optimize=True)

    tau366 += 36 * np.einsum("ijcd,cbda->ijab", tau359, u[v, v, v, v], optimize=True)

    tau359 = None

    tau323 = zeros((N, N, M, M))

    tau323 += 6 * np.einsum("abij->ijab", l2, optimize=True)

    tau323 += np.einsum("abji->ijab", l2, optimize=True)

    tau324 += np.einsum("cbkj,ikac->ijab", t2, tau323, optimize=True)

    tau323 = None

    tau334 -= 12 * np.einsum("kjbc,kiac->ijab", tau219, tau324, optimize=True)

    tau219 = None

    tau324 = None

    tau325 = zeros((N, N, M, M))

    tau325 += 10 * np.einsum("abij->ijab", l2, optimize=True)

    tau325 -= np.einsum("abji->ijab", l2, optimize=True)

    tau326 = zeros((N, N, M, M))

    tau326 += np.einsum("bckj,ikac->ijab", t2, tau325, optimize=True)

    tau327 -= np.einsum("ijab->ijab", tau326, optimize=True)

    tau334 -= 3 * np.einsum("bcda,jicd->ijab", tau209, tau327, optimize=True)

    tau327 = None

    tau425 = zeros((N, N, M, M))

    tau425 += np.einsum("acik,kjcb->ijab", t2, tau326, optimize=True)

    tau326 = None

    tau330 += np.einsum("bcjk,ikac->ijab", t2, tau325, optimize=True)

    tau349 = zeros((N, N, M, M))

    tau349 += np.einsum("cbjk,ikac->ijab", t2, tau325, optimize=True)

    tau325 = None

    tau358 += 3 * np.einsum("bcda,ijcd->ijab", tau209, tau349, optimize=True)

    tau209 = None

    tau349 = None

    tau329 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau334 += 3 * np.einsum("kjli,lkab->ijab", tau329, tau330, optimize=True)

    tau330 = None

    tau348 += np.einsum("kijl,lkab->ijab", tau329, tau74, optimize=True)

    tau373 += 36 * np.einsum("likj,klab->ijab", tau329, tau75, optimize=True)

    tau75 = None

    tau407 += np.einsum("mkjn,niml->ijkl", tau329, tau406, optimize=True)

    tau406 = None

    tau407 -= 36 * np.einsum("injm,mknl->ijkl", tau16, tau329, optimize=True)

    tau407 -= 36 * np.einsum("im,mkjl->ijkl", tau0, tau329, optimize=True)

    tau0 = None

    tau417 += 10 * np.einsum("mkjn,niml->ijkl", tau329, tau416, optimize=True)

    tau416 = None

    r2 -= np.einsum("klij,klab->abij", tau329, tau425, optimize=True) / 12

    tau329 = None

    tau425 = None

    tau331 = zeros((N, N, M, M))

    tau331 += 7 * np.einsum("abij->ijab", l2, optimize=True)

    tau331 -= np.einsum("abji->ijab", l2, optimize=True)

    tau332 -= np.einsum("bckj,ikac->ijab", t2, tau331, optimize=True)

    tau331 = None

    tau334 -= 4 * np.einsum("kijl,klab->ijab", tau18, tau332, optimize=True)

    tau332 = None

    r2 -= np.einsum("acik,jkcb->abij", t2, tau334, optimize=True) / 36

    tau334 = None

    tau339 += np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau340 = zeros((M, M, M, M))

    tau340 += np.einsum("ijab,jicd->abcd", tau339, tau74, optimize=True)

    tau74 = None

    tau339 = None

    tau341 -= 12 * np.einsum("bdac->abcd", tau340, optimize=True)

    tau340 = None

    r2 -= np.einsum("cdij,cabd->abij", t2, tau341, optimize=True) / 12

    r2 -= np.einsum("dcij,cbad->abij", t2, tau341, optimize=True) / 12

    tau341 = None

    tau342 = zeros((N, N, M, M))

    tau342 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau342 -= np.einsum("baij->ijab", t2, optimize=True)

    tau343 = zeros((N, N, M, M))

    tau343 += np.einsum("bcjk,ikac->ijab", l2, tau342, optimize=True)

    tau344 = zeros((N, N, M, M))

    tau344 += np.einsum("bcjk,ikac->ijab", t2, tau343, optimize=True)

    tau343 = None

    tau347 -= np.einsum("ijba->ijab", tau344, optimize=True)

    tau344 = None

    tau348 += np.einsum("jkbc,kica->ijab", tau347, u[o, o, v, v], optimize=True)

    r2 += np.einsum("kica,kjbc->abij", tau348, tau95, optimize=True)

    tau348 = None

    tau95 = None

    tau358 += 36 * np.einsum("jkbc,kiac->ijab", tau347, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ikac,kbcj->abij", tau347, u[o, v, v, o], optimize=True)

    tau347 = None

    tau350 = zeros((N, N, M, M))

    tau350 += np.einsum("caki,jkbc->ijab", l2, tau342, optimize=True)

    tau342 = None

    tau358 -= 36 * np.einsum("ijcd,cbad->ijab", tau350, u[v, v, v, v], optimize=True)

    tau350 = None

    tau352 = zeros((N, N, M, M))

    tau352 -= np.einsum("abij->ijab", l2, optimize=True)

    tau352 += 10 * np.einsum("abji->ijab", l2, optimize=True)

    tau353 = zeros((N, N, M, M))

    tau353 += np.einsum("bcjk,ikca->ijab", t2, tau352, optimize=True)

    tau354 += np.einsum("ijab->ijab", tau353, optimize=True)

    tau358 += 3 * np.einsum("kilj,lkab->ijab", tau19, tau354, optimize=True)

    tau354 = None

    tau420 = zeros((N, N, M, M))

    tau420 -= 3 * np.einsum("ijcd,cabd->ijab", tau353, u[v, v, v, v], optimize=True)

    tau353 = None

    tau374 = zeros((N, N, M, M))

    tau374 += np.einsum("cbkj,ikca->ijab", t2, tau352, optimize=True)

    tau377 += 3 * np.einsum("ijcd,cbad->ijab", tau374, u[v, v, v, v], optimize=True)

    tau374 = None

    tau421 -= np.einsum("bckj,ikca->ijab", t2, tau352, optimize=True)

    tau352 = None

    tau422 = zeros((N, N, M, M))

    tau422 += np.einsum("bcjk,kica->ijab", t2, tau421, optimize=True)

    tau421 = None

    r2 += np.einsum("klji,lkab->abij", tau19, tau422, optimize=True) / 12

    tau19 = None

    tau422 = None

    tau355 = zeros((N, N, M, M))

    tau355 -= np.einsum("abij->ijab", l2, optimize=True)

    tau355 += 7 * np.einsum("abji->ijab", l2, optimize=True)

    tau356 = zeros((N, N, M, M))

    tau356 += np.einsum("bckj,ikca->ijab", t2, tau355, optimize=True)

    tau355 = None

    tau358 += 4 * np.einsum("jkli,klab->ijab", tau18, tau356, optimize=True)

    tau356 = None

    r2 -= np.einsum("bcjk,kica->abij", t2, tau358, optimize=True) / 36

    tau358 = None

    tau364 = zeros((N, N, M, M))

    tau364 += 8 * np.einsum("abij->ijab", l2, optimize=True)

    tau364 += 19 * np.einsum("abji->ijab", l2, optimize=True)

    tau365 = zeros((N, N, M, M))

    tau365 += np.einsum("bckj,ikac->ijab", t2, tau364, optimize=True)

    tau366 -= np.einsum("jkli,klab->ijab", tau18, tau365, optimize=True)

    tau365 = None

    r2 -= np.einsum("ackj,kicb->abij", t2, tau366, optimize=True) / 36

    tau366 = None

    tau419 = zeros((N, N, M, M))

    tau419 += np.einsum("bckj,ikca->ijab", t2, tau364, optimize=True)

    tau364 = None

    tau420 += np.einsum("jkli,klba->ijab", tau18, tau419, optimize=True)

    tau419 = None

    r2 += np.einsum("bckj,kiac->abij", t2, tau420, optimize=True) / 36

    tau420 = None

    tau371 = zeros((N, N, M, M))

    tau371 += 19 * np.einsum("abij->ijab", l2, optimize=True)

    tau371 += 8 * np.einsum("abji->ijab", l2, optimize=True)

    tau372 = zeros((N, N, M, M))

    tau372 += np.einsum("bckj,ikca->ijab", t2, tau371, optimize=True)

    tau371 = None

    tau373 -= np.einsum("kjil,klab->ijab", tau18, tau372, optimize=True)

    tau372 = None

    r2 -= np.einsum("bcki,kjca->abij", t2, tau373, optimize=True) / 36

    tau373 = None

    tau375 = zeros((N, N, M, M))

    tau375 -= 19 * np.einsum("abij->ijab", t2, optimize=True)

    tau375 += 36 * np.einsum("abji->ijab", t2, optimize=True)

    tau376 -= np.einsum("acik,kjbc->ijab", l2, tau375, optimize=True)

    tau375 = None

    tau377 -= np.einsum("kjil,klab->ijab", tau18, tau376, optimize=True)

    tau376 = None

    r2 -= np.einsum("acki,kjcb->abij", t2, tau377, optimize=True) / 36

    tau377 = None

    tau388 = zeros((N, N, M, M))

    tau388 += 2 * np.einsum("abij->ijab", l2, optimize=True)

    tau388 += np.einsum("abji->ijab", l2, optimize=True)

    tau389 = zeros((N, N, M, M))

    tau389 += np.einsum("bckj,ikac->ijab", t2, tau388, optimize=True)

    tau390 += np.einsum("ijab->ijab", tau389, optimize=True)

    tau407 += 12 * np.einsum("jkab,ilab->ijkl", tau138, tau390, optimize=True)

    tau390 = None

    tau423 = zeros((N, N, M, M))

    tau423 += np.einsum("iklj,klab->ijab", tau18, tau389, optimize=True)

    tau389 = None

    r2 += np.einsum("acjk,ikcb->abij", t2, tau423, optimize=True) / 18

    tau423 = None

    tau395 += np.einsum("ikac,jckb->ijab", tau388, u[o, v, o, v], optimize=True)

    tau400 = zeros((N, N, M, M))

    tau400 += np.einsum("ikca,kjcb->ijab", tau388, tau54, optimize=True)

    tau388 = None

    tau54 = None

    tau401 -= np.einsum("ijab->ijab", tau400, optimize=True)

    tau413 = zeros((N, N, M, M))

    tau413 -= np.einsum("ijab->ijab", tau400, optimize=True)

    tau400 = None

    tau391 = zeros((N, N, M, M))

    tau391 += 3 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau391 -= 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau392 = zeros((N, N, M, M))

    tau392 += np.einsum("bckj,kiac->ijab", t2, tau391, optimize=True)

    tau391 = None

    tau393 += np.einsum("jiba->ijab", tau392, optimize=True)

    tau392 = None

    tau394 = zeros((N, N, M, M))

    tau394 += np.einsum("cbkj,kica->ijab", l2, tau393, optimize=True)

    tau393 = None

    tau395 += np.einsum("jiba->ijab", tau394, optimize=True)

    tau407 -= 12 * np.einsum("abjl,ikab->ijkl", t2, tau395, optimize=True)

    tau395 = None

    tau411 -= np.einsum("jiba->ijab", tau394, optimize=True)

    tau394 = None

    tau396 = zeros((N, N, M, M))

    tau396 -= 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau396 += 3 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau397 += np.einsum("acki,kjbc->ijab", t2, tau396, optimize=True)

    tau396 = None

    tau397 += 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau397 -= 3 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau407 -= 12 * np.einsum("ijab,lkab->ijkl", tau288, tau397, optimize=True)

    tau288 = None

    tau397 = None

    tau398 = zeros((N, N, M, M))

    tau398 += np.einsum("abij->ijab", l2, optimize=True)

    tau398 += 2 * np.einsum("abji->ijab", l2, optimize=True)

    tau399 = zeros((N, N, M, M))

    tau399 += np.einsum("ikac,jckb->ijab", tau398, u[o, v, o, v], optimize=True)

    tau401 += np.einsum("ijab->ijab", tau399, optimize=True)

    tau407 -= 12 * np.einsum("bajl,ikab->ijkl", t2, tau401, optimize=True)

    tau401 = None

    tau413 += np.einsum("ijab->ijab", tau399, optimize=True)

    tau399 = None

    tau417 -= 12 * np.einsum("abjl,ikab->ijkl", t2, tau413, optimize=True)

    tau413 = None

    tau403 = zeros((N, N, M, M))

    tau403 += np.einsum("bckj,ikca->ijab", t2, tau398, optimize=True)

    tau404 += np.einsum("ijab->ijab", tau403, optimize=True)

    tau407 -= 12 * np.einsum("ilab,kabj->ijkl", tau404, u[o, v, v, o], optimize=True)

    tau404 = None

    tau424 = zeros((N, N, M, M))

    tau424 += np.einsum("kijl,klab->ijab", tau18, tau403, optimize=True)

    tau18 = None

    tau403 = None

    r2 += np.einsum("bcik,jkca->abij", t2, tau424, optimize=True) / 18

    tau424 = None

    tau409 += np.einsum("cbjk,ikca->ijab", t2, tau398, optimize=True)

    tau412 = zeros((N, N, M, M))

    tau412 += np.einsum("bckj,ikac->ijab", t2, tau398, optimize=True)

    tau398 = None

    tau417 += 12 * np.einsum("jkab,ilab->ijkl", tau138, tau412, optimize=True)

    tau138 = None

    tau412 = None

    tau402 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau407 -= 12 * np.einsum("ijab,lkab->ijkl", tau260, tau402, optimize=True)

    tau260 = None

    r2 += np.einsum("abkl,kilj->abij", t2, tau407, optimize=True) / 36

    tau407 = None

    tau417 -= 12 * np.einsum("lkab,ijab->ijkl", tau402, tau409, optimize=True)

    tau409 = None

    tau402 = None

    tau410 = zeros((N, N, M, M))

    tau410 += 3 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau410 -= 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau411 += np.einsum("acik,jkcb->ijab", l2, tau410, optimize=True)

    tau410 = None

    tau417 += 12 * np.einsum("bajl,ikab->ijkl", t2, tau411, optimize=True)

    tau411 = None

    tau414 = zeros((N, N, M, M))

    tau414 += 3 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau414 -= 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau417 += 12 * np.einsum("ilab,kjab->ijkl", tau36, tau414, optimize=True)

    tau36 = None

    tau414 = None

    r2 += np.einsum("ablk,kilj->abij", t2, tau417, optimize=True) / 36

    tau417 = None

    tau431 += 9 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau431 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau437 += 2 * np.einsum("mlkj,kilm->ij", tau16, tau431, optimize=True)

    tau440 += 2 * np.einsum("lmik,kjlm->ij", tau16, tau431, optimize=True)

    tau16 = None

    tau431 = None

    tau432 = zeros((N, N, M, M))

    tau432 -= np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau432 += 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau433 = zeros((N, N))

    tau433 += np.einsum("ab,ijab->ij", tau59, tau432, optimize=True)

    tau432 = None

    tau59 = None

    tau437 += 18 * np.einsum("ij->ij", tau433, optimize=True)

    tau440 += 18 * np.einsum("ji->ij", tau433, optimize=True)

    tau433 = None

    tau434 = zeros((N, N, M, M))

    tau434 += 12 * np.einsum("abij->ijab", t2, optimize=True)

    tau434 += np.einsum("baij->ijab", t2, optimize=True)

    tau435 = zeros((N, N))

    tau435 += np.einsum("abkj,kiab->ij", l2, tau434, optimize=True)

    tau434 = None

    tau436 = zeros((N, N))

    tau436 += np.einsum("lk,likj->ij", tau435, u[o, o, o, o], optimize=True)

    tau435 = None

    tau437 -= 3 * np.einsum("ij->ij", tau436, optimize=True)

    r2 -= np.einsum("kj,abik->abij", tau437, t2, optimize=True) / 18

    tau437 = None

    tau440 -= 3 * np.einsum("ji->ij", tau436, optimize=True)

    tau436 = None

    r2 -= np.einsum("ik,abkj->abij", tau440, t2, optimize=True) / 18

    tau440 = None

    return r2
