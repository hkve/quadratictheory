import numpy as np
from clusterfock.cc.rhs.t_inter_RCCD import amplitudes_intermediates_rccd


def t_intermediates_qccd_restricted(t2, l2, u, f, v, o):
    r2 = amplitudes_intermediates_rccd(t2, u, f, v, o)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("acik,cbkj->ijab", l2, t2, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("acik,kjcb->ijab", t2, tau0, optimize=True)

    tau382 = np.zeros((N, N, N, N))

    tau382 += 72 * np.einsum("jlba,kiba->ijkl", tau1, u[o, o, v, v], optimize=True)

    r2 += 2 * np.einsum("ijdc,abdc->abij", tau1, u[v, v, v, v], optimize=True)

    tau1 = None

    tau306 = np.zeros((N, N, M, M))

    tau306 += np.einsum("ijab->ijab", tau0, optimize=True)

    tau354 = np.zeros((N, N, M, M))

    tau354 += np.einsum("caik,kjcb->ijab", t2, tau0, optimize=True)

    tau356 = np.zeros((N, N, M, M))

    tau356 += 18 * np.einsum("ijab->ijab", tau354, optimize=True)

    tau378 = np.zeros((N, N, M, M))

    tau378 += np.einsum("ijab->ijab", tau354, optimize=True)

    tau354 = None

    tau389 = np.zeros((N, N, M, M))

    tau389 -= 3 * np.einsum("ijab->ijab", tau0, optimize=True)

    tau2 = np.zeros((N, N, N, N))

    tau2 += np.einsum("abij,abkl->ijkl", l2, t2, optimize=True)

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("abkl,klij->ijab", t2, tau2, optimize=True)

    tau235 = np.zeros((N, N, M, M))

    tau235 += np.einsum("ikac,kbjc->ijab", tau3, u[o, v, o, v], optimize=True)

    tau242 = np.zeros((N, N, M, M))

    tau242 += np.einsum("ijab->ijab", tau235, optimize=True)

    tau235 = None

    tau261 = np.zeros((N, N, M, M))

    tau261 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau289 = np.zeros((N, N, M, M))

    tau289 += np.einsum("kiac,kbjc->ijab", tau3, u[o, v, o, v], optimize=True)

    r2 -= 11 * np.einsum("jiab->abij", tau289, optimize=True) / 18

    r2 -= np.einsum("jiba->abij", tau289, optimize=True) / 3

    tau289 = None

    tau353 = np.zeros((N, N, M, M))

    tau353 -= 2 * np.einsum("kibc,jkac->ijab", tau3, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("kjac,kbci->abij", tau3, u[o, v, v, o], optimize=True) / 9

    tau76 = np.zeros((N, N, M, M))

    tau76 += np.einsum("ijlk,lkab->ijab", tau2, u[o, o, v, v], optimize=True)

    tau81 = np.zeros((N, N, M, M))

    tau81 += 5 * np.einsum("ijab->ijab", tau76, optimize=True)

    tau190 = np.zeros((N, N, M, M))

    tau190 += np.einsum("ijab->ijab", tau76, optimize=True)

    tau209 = np.zeros((M, M))

    tau209 += np.einsum("acij,ijbc->ab", t2, tau76, optimize=True)

    tau210 = np.zeros((M, M))

    tau210 += np.einsum("ab->ab", tau209, optimize=True)

    tau209 = None

    tau303 = np.zeros((N, N, M, M))

    tau303 += np.einsum("ijab->ijab", tau76, optimize=True)

    tau76 = None

    tau195 = np.zeros((N, N, M, M))

    tau195 += np.einsum("ablk,klij->ijab", t2, tau2, optimize=True)

    tau243 = np.zeros((N, N, M, M))

    tau243 += np.einsum("ikac,kbjc->ijab", tau195, u[o, v, o, v], optimize=True)

    tau254 = np.zeros((N, N, M, M))

    tau254 += np.einsum("ijab->ijab", tau243, optimize=True)

    tau243 = None

    tau286 = np.zeros((N, N, M, M))

    tau286 += np.einsum("kiac,kbcj->ijab", tau195, u[o, v, v, o], optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", tau286, optimize=True) / 9

    r2 += np.einsum("jiba->abij", tau286, optimize=True)

    tau286 = None

    tau301 = np.zeros((N, N, M, M))

    tau301 += np.einsum("kiac,kbjc->ijab", tau195, u[o, v, o, v], optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", tau301, optimize=True) / 9

    r2 -= 2 * np.einsum("jiba->abij", tau301, optimize=True) / 3

    tau301 = None

    tau334 = np.zeros((N, N, N, N))

    tau334 += 5 * np.einsum("ijkl->ijkl", tau2, optimize=True)

    tau334 -= 2 * np.einsum("jikl->ijkl", tau2, optimize=True)

    tau380 = np.zeros((N, N, N, N))

    tau380 += np.einsum("ijkl->ijkl", tau2, optimize=True)

    tau380 += 2 * np.einsum("jikl->ijkl", tau2, optimize=True)

    tau381 = np.zeros((N, N, N, N))

    tau381 += 23 * np.einsum("ijkl->ijkl", tau2, optimize=True)

    tau381 += 10 * np.einsum("jikl->ijkl", tau2, optimize=True)

    tau396 = np.zeros((N, N, N, N))

    tau396 += 2 * np.einsum("ijkl->ijkl", tau2, optimize=True)

    tau396 += np.einsum("jikl->ijkl", tau2, optimize=True)

    tau397 = np.zeros((N, N, N, N))

    tau397 += 7 * np.einsum("ijkl->ijkl", tau2, optimize=True)

    tau397 -= np.einsum("jikl->ijkl", tau2, optimize=True)

    tau398 = np.zeros((N, N, N, N))

    tau398 += 10 * np.einsum("ijkl->ijkl", tau2, optimize=True)

    tau398 += 23 * np.einsum("jikl->ijkl", tau2, optimize=True)

    tau4 = np.zeros((M, M, M, M))

    tau4 += np.einsum("abij,cdij->abcd", l2, t2, optimize=True)

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("acbd,icjd->ijab", tau4, u[o, v, o, v], optimize=True)

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("jiab->ijab", tau5, optimize=True)

    tau285 = np.zeros((N, N, M, M))

    tau285 += np.einsum("acki,kjcb->ijab", t2, tau5, optimize=True)

    tau5 = None

    r2 += 5 * np.einsum("ijba->abij", tau285, optimize=True) / 9

    r2 += np.einsum("jiab->abij", tau285, optimize=True)

    r2 -= 2 * np.einsum("jiba->abij", tau285, optimize=True) / 9

    tau285 = None

    tau70 = np.zeros((M, M, M, M))

    tau70 += np.einsum("aebf,cedf->abcd", tau4, u[v, v, v, v], optimize=True)

    tau71 = np.zeros((N, N, M, M))

    tau71 += np.einsum("dcij,cabd->ijab", t2, tau70, optimize=True)

    tau70 = None

    tau73 = np.zeros((N, N, M, M))

    tau73 += np.einsum("ijab->ijab", tau71, optimize=True)

    tau71 = None

    tau123 = np.zeros((M, M, M, M))

    tau123 += np.einsum("aebf,cefd->abcd", tau4, u[v, v, v, v], optimize=True)

    tau125 = np.zeros((M, M, M, M))

    tau125 += 5 * np.einsum("abcd->abcd", tau123, optimize=True)

    tau168 = np.zeros((N, N, M, M))

    tau168 += np.einsum("dcij,cabd->ijab", t2, tau123, optimize=True)

    tau123 = None

    tau172 = np.zeros((N, N, M, M))

    tau172 += np.einsum("ijab->ijab", tau168, optimize=True)

    tau168 = None

    tau287 = np.zeros((N, N, M, M))

    tau287 += np.einsum("acbd,icdj->ijab", tau4, u[o, v, v, o], optimize=True)

    tau288 = np.zeros((N, N, M, M))

    tau288 += np.einsum("acki,kjcb->ijab", t2, tau287, optimize=True)

    tau287 = None

    r2 -= 2 * np.einsum("ijab->abij", tau288, optimize=True) / 9

    r2 -= 4 * np.einsum("ijba->abij", tau288, optimize=True) / 9

    r2 -= 11 * np.einsum("jiab->abij", tau288, optimize=True) / 18

    r2 -= 2 * np.einsum("jiba->abij", tau288, optimize=True) / 9

    tau288 = None

    tau6 = np.zeros((M, M, M, M))

    tau6 += np.einsum("abij,cdji->abcd", l2, t2, optimize=True)

    tau7 = np.zeros((N, N, M, M))

    tau7 += np.einsum("acbd,icdj->ijab", tau6, u[o, v, v, o], optimize=True)

    tau35 += np.einsum("jiab->ijab", tau7, optimize=True)

    tau290 = np.zeros((N, N, M, M))

    tau290 += np.einsum("acki,kjcb->ijab", t2, tau7, optimize=True)

    tau7 = None

    r2 -= 4 * np.einsum("ijab->abij", tau290, optimize=True) / 9

    r2 -= 2 * np.einsum("ijba->abij", tau290, optimize=True) / 9

    r2 -= 2 * np.einsum("jiab->abij", tau290, optimize=True) / 9

    r2 -= 11 * np.einsum("jiba->abij", tau290, optimize=True) / 18

    tau290 = None

    tau150 = np.zeros((M, M, M, M))

    tau150 += np.einsum("aebf,cefd->abcd", tau6, u[v, v, v, v], optimize=True)

    tau151 = np.zeros((M, M, M, M))

    tau151 += 5 * np.einsum("abcd->abcd", tau150, optimize=True)

    tau173 = np.zeros((N, N, M, M))

    tau173 += np.einsum("dcij,cabd->ijab", t2, tau150, optimize=True)

    tau150 = None

    tau175 = np.zeros((N, N, M, M))

    tau175 += np.einsum("ijab->ijab", tau173, optimize=True)

    tau173 = None

    tau180 = np.zeros((N, N, M, M))

    tau180 += np.einsum("acbd,icjd->ijab", tau6, u[o, v, o, v], optimize=True)

    tau184 = np.zeros((N, N, M, M))

    tau184 -= 2 * np.einsum("ijab->ijab", tau180, optimize=True)

    tau180 = None

    tau321 = np.zeros((M, M, M, M))

    tau321 -= np.einsum("aecf,bedf->abcd", tau6, u[v, v, v, v], optimize=True)

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("acik,jcbk->ijab", l2, u[o, v, v, o], optimize=True)

    tau21 = np.zeros((N, N, M, M))

    tau21 += np.einsum("ijab->ijab", tau8, optimize=True)

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("ijab->ijab", tau8, optimize=True)

    tau45 = np.zeros((N, N, M, M))

    tau45 -= np.einsum("ijab->ijab", tau8, optimize=True)

    tau198 = np.zeros((N, N, M, M))

    tau198 += np.einsum("ijab->ijab", tau8, optimize=True)

    tau212 = np.zeros((N, N, M, M))

    tau212 += np.einsum("ijab->ijab", tau8, optimize=True)

    tau374 = np.zeros((N, N, M, M))

    tau374 -= 3 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau404 = np.zeros((N, N, M, M))

    tau404 -= np.einsum("ijab->ijab", tau8, optimize=True)

    tau8 = None

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum("acki,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau21 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau23 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau212 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau282 = np.zeros((N, N, M, M))

    tau282 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau391 = np.zeros((N, N, M, M))

    tau391 -= np.einsum("ijab->ijab", tau9, optimize=True)

    tau406 = np.zeros((N, N, M, M))

    tau406 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau413 = np.zeros((N, N, M, M))

    tau413 += np.einsum("ijab->ijab", tau9, optimize=True)

    tau9 = None

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("acki,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("caik,kjcb->ijab", l2, tau10, optimize=True)

    tau21 -= np.einsum("ijab->ijab", tau11, optimize=True)

    tau23 -= np.einsum("ijab->ijab", tau11, optimize=True)

    tau212 -= np.einsum("ijab->ijab", tau11, optimize=True)

    tau282 -= np.einsum("ijab->ijab", tau11, optimize=True)

    tau283 = np.zeros((N, N, M, M))

    tau283 += np.einsum("bckj,kiac->ijab", t2, tau282, optimize=True)

    tau282 = None

    tau284 = np.zeros((N, N, M, M))

    tau284 += np.einsum("bckj,kica->ijab", t2, tau283, optimize=True)

    tau283 = None

    r2 -= 11 * np.einsum("ijab->abij", tau284, optimize=True) / 18

    r2 -= 2 * np.einsum("ijba->abij", tau284, optimize=True) / 9

    r2 -= 2 * np.einsum("jiab->abij", tau284, optimize=True) / 9

    r2 -= 4 * np.einsum("jiba->abij", tau284, optimize=True) / 9

    tau284 = None

    tau374 -= np.einsum("ijab->ijab", tau11, optimize=True)

    tau391 += np.einsum("ijab->ijab", tau11, optimize=True)

    tau406 -= np.einsum("ijab->ijab", tau11, optimize=True)

    tau11 = None

    tau217 = np.zeros((N, N, M, M))

    tau217 += np.einsum("ijab->ijab", tau10, optimize=True)

    tau256 = np.zeros((N, N, M, M))

    tau256 += np.einsum("caki,kjcb->ijab", l2, tau10, optimize=True)

    tau257 = np.zeros((N, N, M, M))

    tau257 -= np.einsum("ijab->ijab", tau256, optimize=True)

    tau279 = np.zeros((N, N, M, M))

    tau279 -= np.einsum("ijab->ijab", tau256, optimize=True)

    tau405 = np.zeros((N, N, M, M))

    tau405 += np.einsum("ijab->ijab", tau256, optimize=True)

    tau256 = None

    tau269 = np.zeros((N, N, M, M))

    tau269 += np.einsum("ikac,kjbc->ijab", tau10, tau3, optimize=True)

    r2 += np.einsum("ijab->abij", tau269, optimize=True) / 3

    r2 += 11 * np.einsum("ijba->abij", tau269, optimize=True) / 18

    tau269 = None

    tau270 = np.zeros((N, N, M, M))

    tau270 += np.einsum("jkac,ikbc->ijab", tau10, tau3, optimize=True)

    r2 += np.einsum("ijab->abij", tau270, optimize=True) / 3

    r2 += 4 * np.einsum("ijba->abij", tau270, optimize=True) / 9

    tau270 = None

    tau276 = np.zeros((N, N, M, M))

    tau276 += np.einsum("ikac,kjbc->ijab", tau10, tau195, optimize=True)

    tau278 = np.zeros((N, N, M, M))

    tau278 += np.einsum("ijab->ijab", tau276, optimize=True)

    tau276 = None

    tau277 = np.zeros((N, N, M, M))

    tau277 += np.einsum("jkac,ikbc->ijab", tau10, tau195, optimize=True)

    tau278 += np.einsum("ijab->ijab", tau277, optimize=True)

    tau277 = None

    r2 += 2 * np.einsum("ijab->abij", tau278, optimize=True) / 3

    r2 += 2 * np.einsum("ijba->abij", tau278, optimize=True) / 9

    tau278 = None

    tau312 = np.zeros((N, N, M, M))

    tau312 += np.einsum("ijcd,acbd->ijab", tau10, tau4, optimize=True)

    tau12 = np.zeros((M, M))

    tau12 += np.einsum("caij,cbij->ab", l2, t2, optimize=True)

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum("ac,ijbc->ijab", tau12, u[o, o, v, v], optimize=True)

    tau21 += 2 * np.einsum("ijab->ijab", tau13, optimize=True)

    tau21 -= np.einsum("jiab->ijab", tau13, optimize=True)

    tau23 -= np.einsum("ijba->ijab", tau13, optimize=True)

    tau23 += 2 * np.einsum("jiba->ijab", tau13, optimize=True)

    tau204 = np.zeros((N, N, M, M))

    tau204 += np.einsum("ijab->ijab", tau13, optimize=True)

    tau212 += 2 * np.einsum("ijab->ijab", tau13, optimize=True)

    tau212 -= np.einsum("jiab->ijab", tau13, optimize=True)

    tau374 += 3 * np.einsum("jiab->ijab", tau13, optimize=True)

    tau377 = np.zeros((N, N, M, M))

    tau377 += 3 * np.einsum("ijab->ijab", tau13, optimize=True)

    tau404 += np.einsum("ijba->ijab", tau13, optimize=True)

    tau405 += np.einsum("jiba->ijab", tau13, optimize=True)

    tau406 += 2 * np.einsum("jiba->ijab", tau13, optimize=True)

    tau13 = None

    tau411 = np.zeros((N, N))

    tau411 -= 9 * np.einsum("abik,kjab->ij", t2, tau406, optimize=True)

    tau406 = None

    tau47 = np.zeros((N, N, M, M))

    tau47 += np.einsum("cb,acij->ijab", tau12, t2, optimize=True)

    tau53 = np.zeros((N, N, M, M))

    tau53 -= 2 * np.einsum("ijab->ijab", tau47, optimize=True)

    tau63 = np.zeros((N, N, M, M))

    tau63 += np.einsum("ijdc,abdc->ijab", tau47, u[v, v, v, v], optimize=True)

    tau69 = np.zeros((N, N, M, M))

    tau69 += np.einsum("ijab->ijab", tau63, optimize=True)

    tau63 = None

    tau208 = np.zeros((M, M))

    tau208 += np.einsum("jiac,jicb->ab", tau47, u[o, o, v, v], optimize=True)

    tau47 = None

    tau210 += np.einsum("ab->ab", tau208, optimize=True)

    tau208 = None

    tau211 = np.zeros((N, N, M, M))

    tau211 += np.einsum("ac,bcij->ijab", tau210, t2, optimize=True)

    tau210 = None

    r2 -= np.einsum("ijba->abij", tau211, optimize=True)

    r2 += np.einsum("ijab->abij", tau211, optimize=True) / 9

    r2 -= 7 * np.einsum("jiab->abij", tau211, optimize=True) / 9

    tau211 = None

    tau201 = np.zeros((N, N, M, M))

    tau201 += np.einsum("ac,ibjc->ijab", tau12, u[o, v, o, v], optimize=True)

    tau206 = np.zeros((N, N, M, M))

    tau206 += np.einsum("ijab->ijab", tau201, optimize=True)

    tau201 = None

    tau356 += 14 * np.einsum("ca,cbij->ijab", tau12, t2, optimize=True)

    tau14 = np.zeros((N, N))

    tau14 += np.einsum("abki,abkj->ij", l2, t2, optimize=True)

    tau15 = np.zeros((N, N, M, M))

    tau15 += np.einsum("ik,kjba->ijab", tau14, u[o, o, v, v], optimize=True)

    tau21 += 2 * np.einsum("ijab->ijab", tau15, optimize=True)

    tau21 -= np.einsum("ijba->ijab", tau15, optimize=True)

    tau21 -= np.einsum("jiab->ijab", tau15, optimize=True)

    tau21 += 2 * np.einsum("jiba->ijab", tau15, optimize=True)

    tau23 += 2 * np.einsum("ijab->ijab", tau15, optimize=True)

    tau23 -= np.einsum("ijba->ijab", tau15, optimize=True)

    tau45 += np.einsum("jiab->ijab", tau15, optimize=True)

    tau204 += np.einsum("ijab->ijab", tau15, optimize=True)

    tau205 = np.zeros((N, N, M, M))

    tau205 += np.einsum("bckj,ikac->ijab", t2, tau204, optimize=True)

    tau204 = None

    tau206 -= np.einsum("ijab->ijab", tau205, optimize=True)

    tau205 = None

    tau405 += np.einsum("ijab->ijab", tau15, optimize=True)

    tau405 -= 2 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau413 -= np.einsum("ijba->ijab", tau15, optimize=True)

    tau15 = None

    tau414 = np.zeros((N, N))

    tau414 -= np.einsum("bakj,kiab->ij", t2, tau413, optimize=True)

    tau413 = None

    tau203 = np.zeros((N, N, M, M))

    tau203 += np.einsum("ik,kajb->ijab", tau14, u[o, v, o, v], optimize=True)

    tau206 += np.einsum("ijba->ijab", tau203, optimize=True)

    tau203 = None

    tau412 = np.zeros((N, N, M, M))

    tau412 -= 2 * np.einsum("kj,abki->ijab", tau14, t2, optimize=True)

    tau16 = np.zeros((N, N, M, M))

    tau16 += np.einsum("acki,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau19 = np.zeros((N, N, M, M))

    tau19 += np.einsum("ijab->ijab", tau16, optimize=True)

    tau38 = np.zeros((N, N, M, M))

    tau38 += np.einsum("ijab->ijab", tau16, optimize=True)

    tau16 = None

    tau17 = np.zeros((N, N, M, M))

    tau17 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau17 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum("bcjk,kica->ijab", t2, tau17, optimize=True)

    tau19 -= np.einsum("jiba->ijab", tau18, optimize=True)

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum("cbkj,kica->ijab", l2, tau19, optimize=True)

    tau21 -= np.einsum("jiba->ijab", tau20, optimize=True)

    tau22 = np.zeros((N, N, M, M))

    tau22 += np.einsum("bckj,kiac->ijab", t2, tau21, optimize=True)

    tau21 = None

    tau35 += np.einsum("jiab->ijab", tau22, optimize=True)

    tau22 = None

    tau23 -= np.einsum("jiba->ijab", tau20, optimize=True)

    tau24 = np.zeros((N, N, M, M))

    tau24 += np.einsum("bcjk,ikca->ijab", t2, tau23, optimize=True)

    tau23 = None

    tau35 -= 2 * np.einsum("jiab->ijab", tau24, optimize=True)

    tau24 = None

    tau45 += np.einsum("jiba->ijab", tau20, optimize=True)

    tau198 -= np.einsum("jiba->ijab", tau20, optimize=True)

    tau199 = np.zeros((N, N, M, M))

    tau199 += np.einsum("bckj,kiac->ijab", t2, tau198, optimize=True)

    tau198 = None

    tau200 = np.zeros((N, N, M, M))

    tau200 += np.einsum("bckj,kica->ijab", t2, tau199, optimize=True)

    tau199 = None

    r2 -= 2 * np.einsum("ijab->abij", tau200, optimize=True) / 9

    r2 += np.einsum("ijba->abij", tau200, optimize=True)

    r2 += 5 * np.einsum("jiab->abij", tau200, optimize=True) / 9

    tau200 = None

    tau212 -= np.einsum("jiba->ijab", tau20, optimize=True)

    tau213 = np.zeros((N, N, M, M))

    tau213 += np.einsum("bcjk,kiac->ijab", t2, tau212, optimize=True)

    tau212 = None

    tau214 = np.zeros((N, N, M, M))

    tau214 += np.einsum("bckj,kica->ijab", t2, tau213, optimize=True)

    tau213 = None

    r2 += np.einsum("ijab->abij", tau214, optimize=True)

    r2 -= 2 * np.einsum("ijba->abij", tau214, optimize=True) / 9

    r2 += 5 * np.einsum("jiba->abij", tau214, optimize=True) / 9

    tau214 = None

    tau404 += np.einsum("jiba->ijab", tau20, optimize=True)

    tau20 = None

    tau77 = np.zeros((N, N, M, M))

    tau77 += np.einsum("cbjk,kica->ijab", l2, tau19, optimize=True)

    tau81 += 2 * np.einsum("jiba->ijab", tau77, optimize=True)

    tau257 -= np.einsum("jiba->ijab", tau77, optimize=True)

    tau77 = None

    tau196 = np.zeros((N, N, M, M))

    tau196 += np.einsum("ikac,kjbc->ijab", tau19, tau195, optimize=True)

    tau19 = None

    tau195 = None

    tau197 = np.zeros((N, N, M, M))

    tau197 += np.einsum("ijab->ijab", tau196, optimize=True)

    tau196 = None

    tau144 = np.zeros((N, N, M, M))

    tau144 -= np.einsum("jiba->ijab", tau18, optimize=True)

    tau372 = np.zeros((N, N, M, M))

    tau372 -= 3 * np.einsum("jiba->ijab", tau18, optimize=True)

    tau18 = None

    tau343 = np.zeros((N, N, M, M))

    tau343 += np.einsum("acki,kjbc->ijab", t2, tau17, optimize=True)

    tau393 = np.zeros((N, N, M, M))

    tau393 += 2 * np.einsum("caki,kjcb->ijab", t2, tau17, optimize=True)

    tau400 = np.zeros((N, N, M, M))

    tau400 -= 2 * np.einsum("kicb,kjac->ijab", tau17, tau3, optimize=True)

    tau17 = None

    tau3 = None

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum("caik,bckj->ijab", l2, t2, optimize=True)

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("ijab->ijab", tau25, optimize=True)

    tau64 = np.zeros((N, N, M, M))

    tau64 += np.einsum("ijcd,acbd->ijab", tau25, u[v, v, v, v], optimize=True)

    tau67 = np.zeros((N, N, M, M))

    tau67 += np.einsum("ijab->ijab", tau64, optimize=True)

    tau64 = None

    tau103 = np.zeros((N, N, M, M))

    tau103 += np.einsum("ijcd,acdb->ijab", tau25, u[v, v, v, v], optimize=True)

    tau104 = np.zeros((N, N, M, M))

    tau104 += np.einsum("acki,kjbc->ijab", t2, tau103, optimize=True)

    tau103 = None

    tau108 = np.zeros((N, N, M, M))

    tau108 += np.einsum("ijab->ijab", tau104, optimize=True)

    tau104 = None

    tau109 = np.zeros((N, N, M, M))

    tau109 += np.einsum("cbik,kjca->ijab", t2, tau25, optimize=True)

    tau110 = np.zeros((N, N, M, M))

    tau110 += np.einsum("ikac,kbjc->ijab", tau109, u[o, v, o, v], optimize=True)

    tau148 = np.zeros((N, N, M, M))

    tau148 -= 6 * np.einsum("ijab->ijab", tau110, optimize=True)

    tau110 = None

    tau163 = np.zeros((N, N, M, M))

    tau163 += np.einsum("ikac,jkcb->ijab", tau109, u[o, o, v, v], optimize=True)

    tau109 = None

    tau165 = np.zeros((N, N, M, M))

    tau165 += np.einsum("jiab->ijab", tau163, optimize=True)

    tau163 = None

    tau128 = np.zeros((N, N, M, M))

    tau128 += np.einsum("acki,kjcb->ijab", t2, tau25, optimize=True)

    tau129 = np.zeros((N, N, M, M))

    tau129 += np.einsum("ikca,jkcb->ijab", tau128, u[o, o, v, v], optimize=True)

    tau136 = np.zeros((N, N, M, M))

    tau136 += 3 * np.einsum("ijba->ijab", tau129, optimize=True)

    tau129 = None

    tau178 = np.zeros((N, N, M, M))

    tau178 += np.einsum("klab,lkji->ijab", tau128, u[o, o, o, o], optimize=True)

    tau186 = np.zeros((N, N, M, M))

    tau186 += 6 * np.einsum("ijab->ijab", tau178, optimize=True)

    tau178 = None

    tau300 = np.zeros((N, N, M, M))

    tau300 += np.einsum("ikca,kbjc->ijab", tau128, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("jiab->abij", tau300, optimize=True) / 3

    r2 -= 2 * np.einsum("jiba->abij", tau300, optimize=True) / 3

    tau300 = None

    tau401 = np.zeros((N, N, M, M))

    tau401 += np.einsum("ijba->ijab", tau128, optimize=True)

    tau128 = None

    tau156 = np.zeros((N, N, M, M))

    tau156 += np.einsum("jkcb,ikca->ijab", tau10, tau25, optimize=True)

    tau160 = np.zeros((N, N, M, M))

    tau160 += 3 * np.einsum("jiba->ijab", tau156, optimize=True)

    tau156 = None

    tau179 = np.zeros((N, N, M, M))

    tau179 += np.einsum("klab,lijk->ijab", tau25, u[o, o, o, o], optimize=True)

    tau184 += 5 * np.einsum("ijab->ijab", tau179, optimize=True)

    tau179 = None

    tau291 = np.zeros((N, N, M, M))

    tau291 += np.einsum("ikca,kcjb->ijab", tau25, u[o, v, o, v], optimize=True)

    tau293 = np.zeros((N, N, M, M))

    tau293 += np.einsum("ijab->ijab", tau291, optimize=True)

    tau291 = None

    tau295 = np.zeros((N, N, M, M))

    tau295 += np.einsum("ikca,kcbj->ijab", tau25, u[o, v, v, o], optimize=True)

    tau297 = np.zeros((N, N, M, M))

    tau297 += np.einsum("ijab->ijab", tau295, optimize=True)

    tau295 = None

    tau26 = np.zeros((N, N, M, M))

    tau26 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau27 += np.einsum("ijab->ijab", tau26, optimize=True)

    tau28 = np.zeros((N, N, M, M))

    tau28 += np.einsum("jkcb,kcai->ijab", tau27, u[o, v, v, o], optimize=True)

    tau35 += np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau55 = np.zeros((M, M))

    tau55 += np.einsum("jicb,icja->ab", tau27, u[o, v, o, v], optimize=True)

    tau60 = np.zeros((M, M))

    tau60 += np.einsum("ba->ab", tau55, optimize=True)

    tau55 = None

    tau176 = np.zeros((N, N, M, M))

    tau176 += np.einsum("klab,likj->ijab", tau27, u[o, o, o, o], optimize=True)

    tau177 = np.zeros((N, N, M, M))

    tau177 += np.einsum("bckj,kica->ijab", t2, tau176, optimize=True)

    tau176 = None

    r2 -= 7 * np.einsum("jiab->abij", tau177, optimize=True) / 9

    r2 += np.einsum("ijab->abij", tau177, optimize=True) / 9

    tau177 = None

    tau37 = np.zeros((N, N, M, M))

    tau37 += np.einsum("ikca,kcjb->ijab", tau26, u[o, v, o, v], optimize=True)

    tau40 = np.zeros((N, N, M, M))

    tau40 += np.einsum("ijab->ijab", tau37, optimize=True)

    tau37 = None

    tau202 = np.zeros((N, N, M, M))

    tau202 += np.einsum("ikca,kcbj->ijab", tau26, u[o, v, v, o], optimize=True)

    tau206 += np.einsum("ijba->ijab", tau202, optimize=True)

    tau202 = None

    tau207 = np.zeros((N, N, M, M))

    tau207 += np.einsum("bckj,kica->ijab", t2, tau206, optimize=True)

    tau206 = None

    r2 += np.einsum("jiab->abij", tau207, optimize=True)

    r2 += np.einsum("ijba->abij", tau207, optimize=True)

    tau207 = None

    tau324 = np.zeros((N, N, M, M))

    tau324 += 9 * np.einsum("ijab->ijab", tau26, optimize=True)

    tau365 = np.zeros((N, N, M, M))

    tau365 += 18 * np.einsum("jkca,ikcb->ijab", tau10, tau26, optimize=True)

    tau368 = np.zeros((N, N, M, M))

    tau368 -= 3 * np.einsum("ijab->ijab", tau26, optimize=True)

    tau387 = np.zeros((N, N, M, M))

    tau387 -= 36 * np.einsum("ijab->ijab", tau26, optimize=True)

    tau388 = np.zeros((N, N, M, M))

    tau388 -= 36 * np.einsum("klab,lijk->ijab", tau26, u[o, o, o, o], optimize=True)

    tau388 -= 36 * np.einsum("ijcd,bcda->ijab", tau26, u[v, v, v, v], optimize=True)

    tau29 = np.zeros((N, N, M, M))

    tau29 -= np.einsum("abij->ijab", t2, optimize=True)

    tau29 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("cbkj,kiac->ijab", l2, tau29, optimize=True)

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum("kjbc,kcia->ijab", tau30, u[o, v, o, v], optimize=True)

    tau35 -= np.einsum("ijab->ijab", tau31, optimize=True)

    tau31 = None

    tau312 -= np.einsum("ikca,kjbc->ijab", tau10, tau30, optimize=True)

    tau331 = np.zeros((N, N, M, M))

    tau331 += np.einsum("bcjk,ikac->ijab", t2, tau30, optimize=True)

    tau332 = np.zeros((N, N, M, M))

    tau332 -= np.einsum("ijab->ijab", tau331, optimize=True)

    tau331 = None

    tau348 = np.zeros((N, N, M, M))

    tau348 += np.einsum("cbkj,ikac->ijab", t2, tau30, optimize=True)

    tau349 = np.zeros((N, N, M, M))

    tau349 -= np.einsum("ijab->ijab", tau348, optimize=True)

    tau348 = None

    tau311 = np.zeros((N, N, M, M))

    tau311 += np.einsum("acik,kjbc->ijab", l2, tau29, optimize=True)

    tau317 = np.zeros((N, N, M, M))

    tau317 += np.einsum("kiac,kjcb->ijab", tau29, u[o, o, v, v], optimize=True)

    tau318 = np.zeros((N, N, M, M))

    tau318 += np.einsum("jiab->ijab", tau317, optimize=True)

    tau317 = None

    tau32 = np.zeros((N, N, M, M))

    tau32 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau32 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("bc,ijac->ijab", tau12, tau32, optimize=True)

    tau35 -= np.einsum("jiba->ijab", tau33, optimize=True)

    tau33 = None

    tau34 = np.zeros((N, N, M, M))

    tau34 += np.einsum("jk,kiab->ijab", tau14, tau32, optimize=True)

    tau32 = None

    tau35 -= np.einsum("ijba->ijab", tau34, optimize=True)

    tau34 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum("bcjk,ikca->ijab", t2, tau35, optimize=True)

    tau35 = None

    tau62 = np.zeros((N, N, M, M))

    tau62 += np.einsum("jiba->ijab", tau36, optimize=True)

    tau36 = None

    tau38 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum("jk,ikab->ijab", tau14, tau38, optimize=True)

    tau40 -= np.einsum("jiab->ijab", tau39, optimize=True)

    tau39 = None

    tau41 = np.zeros((N, N, M, M))

    tau41 += np.einsum("bckj,kiac->ijab", t2, tau40, optimize=True)

    tau40 = None

    tau62 += np.einsum("jiba->ijab", tau41, optimize=True)

    tau41 = None

    tau124 = np.zeros((M, M, M, M))

    tau124 += np.einsum("ijcd,ijab->abcd", tau25, tau38, optimize=True)

    tau125 += 6 * np.einsum("cdab->abcd", tau124, optimize=True)

    tau126 = np.zeros((N, N, M, M))

    tau126 += np.einsum("cdij,cabd->ijab", t2, tau125, optimize=True)

    tau125 = None

    tau148 += np.einsum("ijab->ijab", tau126, optimize=True)

    tau126 = None

    tau154 = np.zeros((M, M, M, M))

    tau154 += 6 * np.einsum("cdab->abcd", tau124, optimize=True)

    tau124 = None

    tau215 = np.zeros((N, N, M, M))

    tau215 += np.einsum("bc,ijac->ijab", tau12, tau38, optimize=True)

    tau38 = None

    tau216 = np.zeros((N, N, M, M))

    tau216 -= np.einsum("bckj,ikac->ijab", t2, tau215, optimize=True)

    tau215 = None

    r2 += 5 * np.einsum("jiba->abij", tau216, optimize=True) / 9

    r2 -= 2 * np.einsum("ijba->abij", tau216, optimize=True) / 9

    r2 += np.einsum("ijab->abij", tau216, optimize=True)

    tau216 = None

    tau42 = np.zeros((N, N, M, M))

    tau42 += np.einsum("cdij,dcba->ijab", l2, u[v, v, v, v], optimize=True)

    tau43 = np.zeros((M, M))

    tau43 += np.einsum("acij,ijbc->ab", t2, tau42, optimize=True)

    tau60 -= np.einsum("ab->ab", tau43, optimize=True)

    tau43 = None

    tau81 += 6 * np.einsum("ijab->ijab", tau42, optimize=True)

    tau92 = np.zeros((N, N, M, M))

    tau92 += 3 * np.einsum("jiab->ijab", tau42, optimize=True)

    tau303 += np.einsum("ijab->ijab", tau42, optimize=True)

    tau42 = None

    tau44 = np.zeros((N, N, M, M))

    tau44 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau44 -= np.einsum("abji->ijab", t2, optimize=True)

    tau46 = np.zeros((M, M))

    tau46 += np.einsum("ijac,jicb->ab", tau44, tau45, optimize=True)

    tau45 = None

    tau60 += np.einsum("ab->ab", tau46, optimize=True)

    tau46 = None

    tau52 = np.zeros((N, N, M, M))

    tau52 += np.einsum("kj,ikab->ijab", tau14, tau44, optimize=True)

    tau53 -= np.einsum("ijab->ijab", tau52, optimize=True)

    tau52 = None

    tau115 = np.zeros((N, N, M, M))

    tau115 += np.einsum("ikac,kjcb->ijab", tau44, u[o, o, v, v], optimize=True)

    tau116 = np.zeros((N, N, M, M))

    tau116 += np.einsum("ijab->ijab", tau115, optimize=True)

    tau115 = None

    tau48 = np.zeros((N, N, M, M))

    tau48 += np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    tau50 = np.zeros((N, N, M, M))

    tau50 += np.einsum("ijab->ijab", tau48, optimize=True)

    tau340 = np.zeros((N, N, M, M))

    tau340 += np.einsum("ijab->ijab", tau48, optimize=True)

    tau364 = np.zeros((N, N, M, M))

    tau364 -= 3 * np.einsum("ijab->ijab", tau48, optimize=True)

    tau365 += 18 * np.einsum("klab,lijk->ijab", tau48, u[o, o, o, o], optimize=True)

    tau403 = np.zeros((N, N, M, M))

    tau403 += np.einsum("ikac,kjcb->ijab", tau44, tau48, optimize=True)

    tau44 = None

    tau49 = np.zeros((N, N, M, M))

    tau49 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau50 += np.einsum("ijab->ijab", tau49, optimize=True)

    tau51 = np.zeros((N, N, M, M))

    tau51 += np.einsum("cbjk,kica->ijab", t2, tau50, optimize=True)

    tau53 += np.einsum("jiab->ijab", tau51, optimize=True)

    tau51 = None

    tau54 = np.zeros((M, M))

    tau54 += np.einsum("ijac,ijbc->ab", tau53, u[o, o, v, v], optimize=True)

    tau53 = None

    tau60 -= np.einsum("ab->ab", tau54, optimize=True)

    tau54 = None

    tau74 = np.zeros((N, N, M, M))

    tau74 += np.einsum("klab,lijk->ijab", tau49, u[o, o, o, o], optimize=True)

    tau83 = np.zeros((N, N, M, M))

    tau83 += 6 * np.einsum("ijab->ijab", tau74, optimize=True)

    tau74 = None

    tau306 += np.einsum("ijab->ijab", tau49, optimize=True)

    tau307 = np.zeros((N, N, M, M))

    tau307 += np.einsum("bckj,kica->ijab", t2, tau306, optimize=True)

    tau308 = np.zeros((N, N, M, M))

    tau308 += np.einsum("jiba->ijab", tau307, optimize=True)

    tau307 = None

    tau412 += np.einsum("cbjk,kica->ijab", t2, tau306, optimize=True)

    tau56 = np.zeros((M, M, M, M))

    tau56 -= np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau56 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau57 = np.zeros((M, M))

    tau57 += np.einsum("cd,cadb->ab", tau12, tau56, optimize=True)

    tau60 += np.einsum("ab->ab", tau57, optimize=True)

    tau57 = None

    tau58 = np.zeros((N, N, M, M))

    tau58 -= np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau58 += 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau59 = np.zeros((M, M))

    tau59 += np.einsum("ji,ijab->ab", tau14, tau58, optimize=True)

    tau60 -= np.einsum("ab->ab", tau59, optimize=True)

    tau59 = None

    tau61 = np.zeros((N, N, M, M))

    tau61 += np.einsum("ac,bcij->ijab", tau60, t2, optimize=True)

    tau60 = None

    tau62 += np.einsum("ijba->ijab", tau61, optimize=True)

    tau61 = None

    r2 += np.einsum("ijab->abij", tau62, optimize=True)

    r2 += np.einsum("jiba->abij", tau62, optimize=True)

    tau62 = None

    tau408 = np.zeros((N, N))

    tau408 += np.einsum("ab,ijab->ij", tau12, tau58, optimize=True)

    tau58 = None

    tau411 += 9 * np.einsum("ji->ij", tau408, optimize=True)

    tau414 += np.einsum("ij->ij", tau408, optimize=True)

    tau408 = None

    tau65 = np.zeros((M, M, M, M))

    tau65 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau66 = np.zeros((N, N, M, M))

    tau66 += np.einsum("ijcd,acbd->ijab", tau0, tau65, optimize=True)

    tau67 += np.einsum("ijab->ijab", tau66, optimize=True)

    tau66 = None

    tau68 = np.zeros((N, N, M, M))

    tau68 += np.einsum("bcjk,kiac->ijab", t2, tau67, optimize=True)

    tau69 += np.einsum("jiba->ijab", tau68, optimize=True)

    tau68 = None

    r2 -= np.einsum("ijab->abij", tau69, optimize=True)

    r2 += np.einsum("ijba->abij", tau69, optimize=True) / 9

    tau69 = None

    tau72 = np.zeros((N, N, M, M))

    tau72 += np.einsum("bckj,kiac->ijab", t2, tau67, optimize=True)

    tau67 = None

    tau73 += np.einsum("jiba->ijab", tau72, optimize=True)

    tau72 = None

    r2 += np.einsum("ijab->abij", tau73, optimize=True) / 9

    r2 -= np.einsum("ijba->abij", tau73, optimize=True)

    tau73 = None

    tau113 = np.zeros((M, M, M, M))

    tau113 += np.einsum("eabf,cefd->abcd", tau4, tau65, optimize=True)

    tau121 = np.zeros((M, M, M, M))

    tau121 += 5 * np.einsum("abcd->abcd", tau113, optimize=True)

    tau171 = np.zeros((N, N, M, M))

    tau171 += np.einsum("cdij,cabd->ijab", t2, tau113, optimize=True)

    tau113 = None

    tau172 += np.einsum("ijab->ijab", tau171, optimize=True)

    tau171 = None

    tau153 = np.zeros((M, M, M, M))

    tau153 += np.einsum("eabf,cefd->abcd", tau6, tau65, optimize=True)

    tau154 += 5 * np.einsum("abcd->abcd", tau153, optimize=True)

    tau155 = np.zeros((N, N, M, M))

    tau155 += np.einsum("dcij,cabd->ijab", t2, tau154, optimize=True)

    tau154 = None

    tau167 = np.zeros((N, N, M, M))

    tau167 += np.einsum("ijab->ijab", tau155, optimize=True)

    tau155 = None

    tau174 = np.zeros((N, N, M, M))

    tau174 += np.einsum("cdij,cabd->ijab", t2, tau153, optimize=True)

    tau153 = None

    tau175 += np.einsum("ijab->ijab", tau174, optimize=True)

    tau174 = None

    r2 += 23 * np.einsum("ijab->abij", tau175, optimize=True) / 36

    r2 += 5 * np.einsum("ijba->abij", tau175, optimize=True) / 18

    tau175 = None

    tau267 = np.zeros((N, N, M, M))

    tau267 += np.einsum("ijcd,acdb->ijab", tau49, tau65, optimize=True)

    tau268 = np.zeros((N, N, M, M))

    tau268 += np.einsum("acki,kjbc->ijab", t2, tau267, optimize=True)

    tau267 = None

    r2 += np.einsum("ijab->abij", tau268, optimize=True) / 3

    r2 += 5 * np.einsum("ijba->abij", tau268, optimize=True) / 9

    tau268 = None

    tau302 = np.zeros((N, N, M, M))

    tau302 += np.einsum("ijcd,acbd->ijab", tau49, tau65, optimize=True)

    tau312 += np.einsum("jiba->ijab", tau302, optimize=True)

    tau353 += 18 * np.einsum("jiba->ijab", tau302, optimize=True)

    tau302 = None

    tau313 = np.zeros((M, M, M, M))

    tau313 += np.einsum("eabf,cedf->abcd", tau6, tau65, optimize=True)

    tau320 = np.zeros((M, M, M, M))

    tau320 += 9 * np.einsum("abcd->abcd", tau313, optimize=True)

    tau321 += 18 * np.einsum("abcd->abcd", tau313, optimize=True)

    tau313 = None

    tau365 += 18 * np.einsum("ijcd,bcda->ijab", tau0, tau65, optimize=True)

    tau388 -= 36 * np.einsum("ijcd,bcda->ijab", tau48, tau65, optimize=True)

    tau75 = np.zeros((N, N, M, M))

    tau75 += np.einsum("acki,jcbk->ijab", l2, u[o, v, v, o], optimize=True)

    tau81 -= 2 * np.einsum("ijab->ijab", tau75, optimize=True)

    tau257 += np.einsum("ijab->ijab", tau75, optimize=True)

    tau75 = None

    tau78 = np.zeros((N, N, N, N))

    tau78 += np.einsum("abij,lkba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau79 = np.zeros((N, N, N, N))

    tau79 += np.einsum("lkji->ijkl", tau78, optimize=True)

    tau310 = np.zeros((N, N, N, N))

    tau310 += np.einsum("klij->ijkl", tau78, optimize=True)

    tau402 = np.zeros((N, N, N, N))

    tau402 += 2 * np.einsum("ijkl->ijkl", tau78, optimize=True)

    tau402 += np.einsum("ijlk->ijkl", tau78, optimize=True)

    tau407 = np.zeros((N, N, N, N))

    tau407 += 7 * np.einsum("klji->ijkl", tau78, optimize=True)

    tau407 -= np.einsum("lkji->ijkl", tau78, optimize=True)

    tau415 = np.zeros((N, N, N, N))

    tau415 += np.einsum("klji->ijkl", tau78, optimize=True)

    tau415 += 2 * np.einsum("lkji->ijkl", tau78, optimize=True)

    tau79 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau80 = np.zeros((N, N, M, M))

    tau80 += np.einsum("abkl,ijkl->ijab", l2, tau79, optimize=True)

    tau81 += 5 * np.einsum("jiba->ijab", tau80, optimize=True)

    tau80 = None

    tau82 = np.zeros((N, N, M, M))

    tau82 += np.einsum("bckj,ikca->ijab", t2, tau81, optimize=True)

    tau81 = None

    tau83 += np.einsum("ijab->ijab", tau82, optimize=True)

    tau82 = None

    tau84 = np.zeros((N, N, M, M))

    tau84 += np.einsum("bckj,kica->ijab", t2, tau83, optimize=True)

    tau83 = None

    r2 += np.einsum("jiba->abij", tau84, optimize=True) / 18

    r2 += np.einsum("ijba->abij", tau84, optimize=True) / 9

    tau84 = None

    tau189 = np.zeros((N, N, M, M))

    tau189 += np.einsum("ablk,ijkl->ijab", l2, tau79, optimize=True)

    tau190 += np.einsum("jiab->ijab", tau189, optimize=True)

    tau191 = np.zeros((N, N, M, M))

    tau191 += np.einsum("bckj,ikac->ijab", t2, tau190, optimize=True)

    tau190 = None

    tau192 = np.zeros((N, N, M, M))

    tau192 += np.einsum("bckj,kica->ijab", t2, tau191, optimize=True)

    tau191 = None

    r2 += 23 * np.einsum("jiba->abij", tau192, optimize=True) / 36

    r2 += 5 * np.einsum("ijba->abij", tau192, optimize=True) / 18

    tau192 = None

    tau303 += np.einsum("jiab->ijab", tau189, optimize=True)

    tau189 = None

    tau312 -= np.einsum("kibc,jkac->ijab", tau29, tau303, optimize=True)

    tau29 = None

    tau345 = np.zeros((N, N, M, M))

    tau345 += 9 * np.einsum("bckj,kica->ijab", t2, tau303, optimize=True)

    tau303 = None

    tau312 -= np.einsum("lkab,kjil->ijab", tau311, tau79, optimize=True)

    tau311 = None

    tau337 = np.zeros((N, N, M, M))

    tau337 += 9 * np.einsum("lkab,kilj->ijab", tau50, tau79, optimize=True)

    tau345 += 9 * np.einsum("lkab,kilj->ijab", tau27, tau79, optimize=True)

    tau382 += 10 * np.einsum("injm,mkln->ijkl", tau380, tau79, optimize=True)

    tau380 = None

    tau382 -= 36 * np.einsum("niml,mknj->ijkl", tau2, tau79, optimize=True)

    tau382 -= 36 * np.einsum("im,mklj->ijkl", tau14, tau79, optimize=True)

    tau399 = np.zeros((N, N, N, N))

    tau399 += 10 * np.einsum("injm,mkln->ijkl", tau396, tau79, optimize=True)

    tau399 += 2 * np.einsum("niml,mknj->ijkl", tau396, tau79, optimize=True)

    tau396 = None

    tau414 += np.einsum("lmkj,kilm->ij", tau2, tau79, optimize=True)

    r2 += np.einsum("lkab,klji->abij", tau403, tau79, optimize=True)

    tau403 = None

    tau79 = None

    tau85 = np.zeros((N, N, M, M))

    tau85 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau86 = np.zeros((N, N, M, M))

    tau86 += np.einsum("acki,kjcb->ijab", t2, tau85, optimize=True)

    tau87 = np.zeros((N, N, M, M))

    tau87 += np.einsum("klab,lkji->ijab", tau86, u[o, o, o, o], optimize=True)

    tau96 = np.zeros((N, N, M, M))

    tau96 += 3 * np.einsum("ijab->ijab", tau87, optimize=True)

    tau87 = None

    tau157 = np.zeros((N, N, M, M))

    tau157 += np.einsum("kiac,jkcb->ijab", tau86, u[o, o, v, v], optimize=True)

    tau160 += 3 * np.einsum("ijba->ijab", tau157, optimize=True)

    tau157 = None

    tau299 = np.zeros((N, N, M, M))

    tau299 += np.einsum("kiac,kbjc->ijab", tau86, u[o, v, o, v], optimize=True)

    r2 -= 2 * np.einsum("jiab->abij", tau299, optimize=True) / 3

    r2 -= np.einsum("jiba->abij", tau299, optimize=True) / 3

    tau299 = None

    tau401 += np.einsum("ijab->ijab", tau86, optimize=True)

    tau86 = None

    r2 += np.einsum("klab,ijkl->abij", tau401, tau402, optimize=True) / 3

    tau401 = None

    tau402 = None

    tau88 = np.zeros((N, N, M, M))

    tau88 += np.einsum("klab,lijk->ijab", tau85, u[o, o, o, o], optimize=True)

    tau94 = np.zeros((N, N, M, M))

    tau94 += 3 * np.einsum("ijab->ijab", tau88, optimize=True)

    tau88 = None

    tau169 = np.zeros((N, N, M, M))

    tau169 += np.einsum("cbik,kjca->ijab", t2, tau85, optimize=True)

    tau170 = np.zeros((N, N, M, M))

    tau170 += np.einsum("ijdc,abdc->ijab", tau169, u[v, v, v, v], optimize=True)

    tau172 += np.einsum("ijab->ijab", tau170, optimize=True)

    tau170 = None

    r2 += 5 * np.einsum("ijab->abij", tau172, optimize=True) / 18

    r2 += 23 * np.einsum("ijba->abij", tau172, optimize=True) / 36

    tau172 = None

    tau366 = np.zeros((N, N, N, N))

    tau366 += np.einsum("ijba,klba->ijkl", tau169, u[o, o, v, v], optimize=True)

    tau169 = None

    tau382 += 24 * np.einsum("jlik->ijkl", tau366, optimize=True)

    tau382 += 12 * np.einsum("jlki->ijkl", tau366, optimize=True)

    tau366 = None

    tau271 = np.zeros((N, N, M, M))

    tau271 += np.einsum("acdb,ijcd->ijab", tau65, tau85, optimize=True)

    tau272 = np.zeros((N, N, M, M))

    tau272 += np.einsum("acki,kjbc->ijab", t2, tau271, optimize=True)

    tau271 = None

    tau275 = np.zeros((N, N, M, M))

    tau275 += np.einsum("ijab->ijab", tau272, optimize=True)

    tau272 = None

    tau347 = np.zeros((N, N, M, M))

    tau347 += np.einsum("acik,kjcb->ijab", t2, tau85, optimize=True)

    tau349 += np.einsum("ijab->ijab", tau347, optimize=True)

    tau353 += 18 * np.einsum("kibc,kjac->ijab", tau349, u[o, o, v, v], optimize=True)

    r2 += np.einsum("kjac,kbic->abij", tau349, u[o, v, o, v], optimize=True)

    tau349 = None

    tau356 += 18 * np.einsum("ijab->ijab", tau347, optimize=True)

    tau378 += np.einsum("ijab->ijab", tau347, optimize=True)

    tau382 -= 36 * np.einsum("jlab,kiab->ijkl", tau378, u[o, o, v, v], optimize=True)

    tau378 = None

    tau89 = np.zeros((M, M))

    tau89 += np.einsum("caij,cbji->ab", l2, t2, optimize=True)

    tau90 = np.zeros((N, N, M, M))

    tau90 += np.einsum("ac,ibcj->ijab", tau89, u[o, v, v, o], optimize=True)

    tau94 -= np.einsum("ijab->ijab", tau90, optimize=True)

    tau90 = None

    tau91 = np.zeros((N, N, M, M))

    tau91 += np.einsum("ac,ijbc->ijab", tau89, u[o, o, v, v], optimize=True)

    tau92 += np.einsum("ijab->ijab", tau91, optimize=True)

    tau93 = np.zeros((N, N, M, M))

    tau93 += np.einsum("bckj,kiac->ijab", t2, tau92, optimize=True)

    tau92 = None

    tau94 += np.einsum("ijab->ijab", tau93, optimize=True)

    tau93 = None

    tau95 = np.zeros((N, N, M, M))

    tau95 += np.einsum("bckj,kica->ijab", t2, tau94, optimize=True)

    tau94 = None

    tau96 += np.einsum("jiba->ijab", tau95, optimize=True)

    tau95 = None

    r2 += 2 * np.einsum("ijab->abij", tau96, optimize=True) / 9

    r2 += np.einsum("jiab->abij", tau96, optimize=True) / 9

    tau96 = None

    tau257 += 2 * np.einsum("ijab->ijab", tau91, optimize=True)

    tau257 -= np.einsum("jiab->ijab", tau91, optimize=True)

    tau91 = None

    tau260 = np.zeros((N, N, M, M))

    tau260 += np.einsum("cb,acij->ijab", tau89, t2, optimize=True)

    tau261 += np.einsum("ijab->ijab", tau260, optimize=True)

    tau260 = None

    tau262 = np.zeros((M, M))

    tau262 += np.einsum("ijac,ijcb->ab", tau261, u[o, o, v, v], optimize=True)

    tau261 = None

    tau263 = np.zeros((N, N, M, M))

    tau263 += np.einsum("ac,bcij->ijab", tau262, t2, optimize=True)

    tau262 = None

    tau264 = np.zeros((N, N, M, M))

    tau264 += np.einsum("ijba->ijab", tau263, optimize=True)

    tau263 = None

    tau97 = np.zeros((N, N, M, M))

    tau97 += np.einsum("caik,cbjk->ijab", l2, t2, optimize=True)

    tau98 = np.zeros((N, N, M, M))

    tau98 += np.einsum("ijcd,acdb->ijab", tau97, u[v, v, v, v], optimize=True)

    tau99 = np.zeros((N, N, M, M))

    tau99 += np.einsum("ackj,kibc->ijab", t2, tau98, optimize=True)

    tau98 = None

    r2 += 5 * np.einsum("ijab->abij", tau99, optimize=True) / 9

    r2 += np.einsum("ijba->abij", tau99, optimize=True) / 3

    tau99 = None

    tau111 = np.zeros((N, N, M, M))

    tau111 += np.einsum("ackj,kicb->ijab", t2, tau97, optimize=True)

    tau112 = np.zeros((N, N, M, M))

    tau112 += np.einsum("ijdc,abdc->ijab", tau111, u[v, v, v, v], optimize=True)

    tau148 += 5 * np.einsum("ijab->ijab", tau112, optimize=True)

    tau112 = None

    tau367 = np.zeros((N, N, N, N))

    tau367 += np.einsum("ijba,lkab->ijkl", tau111, u[o, o, v, v], optimize=True)

    tau111 = None

    tau382 += 12 * np.einsum("jlik->ijkl", tau367, optimize=True)

    tau382 += 24 * np.einsum("jlki->ijkl", tau367, optimize=True)

    tau367 = None

    tau100 = np.zeros((N, N, M, M))

    tau100 += np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau101 = np.zeros((N, N, M, M))

    tau101 += np.einsum("ijcd,acdb->ijab", tau100, u[v, v, v, v], optimize=True)

    tau102 = np.zeros((N, N, M, M))

    tau102 += np.einsum("acki,kjbc->ijab", t2, tau101, optimize=True)

    r2 += 23 * np.einsum("ijab->abij", tau102, optimize=True) / 36

    r2 += np.einsum("ijba->abij", tau102, optimize=True) / 3

    tau102 = None

    tau312 += np.einsum("jiba->ijab", tau101, optimize=True)

    tau101 = None

    tau127 = np.zeros((N, N, M, M))

    tau127 += np.einsum("jkcb,ikca->ijab", tau10, tau100, optimize=True)

    tau136 += 3 * np.einsum("jiba->ijab", tau127, optimize=True)

    tau127 = None

    tau182 = np.zeros((N, N, M, M))

    tau182 += np.einsum("ijab->ijab", tau100, optimize=True)

    tau187 = np.zeros((N, N, M, M))

    tau187 += np.einsum("klab,lijk->ijab", tau100, u[o, o, o, o], optimize=True)

    tau188 = np.zeros((N, N, M, M))

    tau188 += np.einsum("acki,kjcb->ijab", t2, tau187, optimize=True)

    tau187 = None

    r2 += 5 * np.einsum("ijba->abij", tau188, optimize=True) / 18

    r2 += 23 * np.einsum("jiba->abij", tau188, optimize=True) / 36

    tau188 = None

    tau292 = np.zeros((N, N, M, M))

    tau292 += np.einsum("ikca,kcbj->ijab", tau100, u[o, v, v, o], optimize=True)

    tau293 += np.einsum("ijab->ijab", tau292, optimize=True)

    tau292 = None

    tau294 = np.zeros((N, N, M, M))

    tau294 += np.einsum("bckj,kiac->ijab", t2, tau293, optimize=True)

    tau293 = None

    r2 -= np.einsum("jiba->abij", tau294, optimize=True) / 3

    r2 -= 2 * np.einsum("jiab->abij", tau294, optimize=True) / 3

    r2 -= 2 * np.einsum("ijba->abij", tau294, optimize=True) / 3

    r2 -= np.einsum("ijab->abij", tau294, optimize=True) / 3

    tau294 = None

    tau296 = np.zeros((N, N, M, M))

    tau296 += np.einsum("ikca,kcjb->ijab", tau100, u[o, v, o, v], optimize=True)

    tau297 += np.einsum("ijab->ijab", tau296, optimize=True)

    tau296 = None

    tau298 = np.zeros((N, N, M, M))

    tau298 += np.einsum("bckj,kiac->ijab", t2, tau297, optimize=True)

    tau297 = None

    r2 -= 2 * np.einsum("jiba->abij", tau298, optimize=True) / 3

    r2 -= np.einsum("jiab->abij", tau298, optimize=True) / 3

    r2 -= np.einsum("ijba->abij", tau298, optimize=True) / 3

    r2 -= 2 * np.einsum("ijab->abij", tau298, optimize=True) / 3

    tau298 = None

    tau357 = np.zeros((N, N, M, M))

    tau357 += np.einsum("ijcd,acbd->ijab", tau100, u[v, v, v, v], optimize=True)

    tau365 -= np.einsum("ijba->ijab", tau357, optimize=True)

    tau400 += np.einsum("ijab->ijab", tau357, optimize=True)

    tau357 = None

    tau382 -= 12 * np.einsum("ilab,kabj->ijkl", tau100, u[o, v, v, o], optimize=True)

    tau105 = np.zeros((N, N, M, M))

    tau105 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau106 = np.zeros((N, N, M, M))

    tau106 += np.einsum("ijcd,acdb->ijab", tau105, u[v, v, v, v], optimize=True)

    tau107 = np.zeros((N, N, M, M))

    tau107 += np.einsum("ackj,kibc->ijab", t2, tau106, optimize=True)

    tau106 = None

    tau108 += np.einsum("ijab->ijab", tau107, optimize=True)

    tau107 = None

    r2 += 5 * np.einsum("ijab->abij", tau108, optimize=True) / 18

    r2 += 2 * np.einsum("ijba->abij", tau108, optimize=True) / 3

    tau108 = None

    tau114 = np.zeros((N, N, M, M))

    tau114 += np.einsum("acik,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau116 -= np.einsum("ijab->ijab", tau114, optimize=True)

    tau318 -= np.einsum("jiab->ijab", tau114, optimize=True)

    tau114 = None

    tau116 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau117 = np.zeros((M, M, M, M))

    tau117 += np.einsum("ijcd,ijab->abcd", tau100, tau116, optimize=True)

    tau121 -= 6 * np.einsum("cdab->abcd", tau117, optimize=True)

    tau151 -= 6 * np.einsum("cdab->abcd", tau117, optimize=True)

    tau117 = None

    tau230 = np.zeros((N, N, M, M))

    tau230 += np.einsum("klab,iklj->ijab", tau116, tau2, optimize=True)

    tau231 = np.zeros((N, N, M, M))

    tau231 += np.einsum("bckj,kiac->ijab", t2, tau230, optimize=True)

    tau232 = np.zeros((N, N, M, M))

    tau232 += np.einsum("jiba->ijab", tau231, optimize=True)

    tau231 = None

    tau312 -= np.einsum("jiba->ijab", tau230, optimize=True)

    tau230 = None

    tau236 = np.zeros((N, N, M, M))

    tau236 += np.einsum("kiac,kjbc->ijab", tau116, tau49, optimize=True)

    tau238 = np.zeros((N, N, M, M))

    tau238 += np.einsum("jiba->ijab", tau236, optimize=True)

    tau236 = None

    tau240 = np.zeros((N, N, M, M))

    tau240 += np.einsum("klab,ikjl->ijab", tau116, tau2, optimize=True)

    tau241 = np.zeros((N, N, M, M))

    tau241 += np.einsum("bckj,kiac->ijab", t2, tau240, optimize=True)

    tau240 = None

    tau242 += np.einsum("ijba->ijab", tau241, optimize=True)

    tau241 = None

    tau244 = np.zeros((N, N, M, M))

    tau244 += np.einsum("kiac,kjbc->ijab", tau116, tau85, optimize=True)

    tau247 = np.zeros((N, N, M, M))

    tau247 += np.einsum("jiba->ijab", tau244, optimize=True)

    tau244 = None

    tau246 = np.zeros((N, N, M, M))

    tau246 += np.einsum("klab,kilj->ijab", tau116, tau2, optimize=True)

    tau247 += np.einsum("jiba->ijab", tau246, optimize=True)

    tau246 = None

    tau251 = np.zeros((N, N, M, M))

    tau251 += np.einsum("klab,kijl->ijab", tau116, tau2, optimize=True)

    tau252 = np.zeros((N, N, M, M))

    tau252 += np.einsum("jiba->ijab", tau251, optimize=True)

    tau345 -= 9 * np.einsum("ijba->ijab", tau251, optimize=True)

    tau251 = None

    tau312 -= np.einsum("kjbc,kiac->ijab", tau116, tau306, optimize=True)

    tau337 -= 9 * np.einsum("kibc,kjac->ijab", tau116, tau48, optimize=True)

    tau353 -= 18 * np.einsum("kiac,kjbc->ijab", tau0, tau116, optimize=True)

    tau0 = None

    tau118 = np.zeros((N, N, M, M))

    tau118 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau118 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau119 = np.zeros((N, N, M, M))

    tau119 += np.einsum("bcjk,kiac->ijab", t2, tau118, optimize=True)

    tau120 = np.zeros((M, M, M, M))

    tau120 += np.einsum("jiba,ijcd->abcd", tau119, tau25, optimize=True)

    tau121 -= 6 * np.einsum("bdac->abcd", tau120, optimize=True)

    tau122 = np.zeros((N, N, M, M))

    tau122 += np.einsum("dcij,cabd->ijab", t2, tau121, optimize=True)

    tau121 = None

    tau148 += np.einsum("ijab->ijab", tau122, optimize=True)

    tau122 = None

    tau151 -= 6 * np.einsum("bdac->abcd", tau120, optimize=True)

    tau120 = None

    tau152 = np.zeros((N, N, M, M))

    tau152 += np.einsum("cdij,cabd->ijab", t2, tau151, optimize=True)

    tau151 = None

    tau167 += np.einsum("ijab->ijab", tau152, optimize=True)

    tau152 = None

    tau343 += np.einsum("jiba->ijab", tau119, optimize=True)

    tau119 = None

    tau130 = np.zeros((N, N, M, M))

    tau130 += np.einsum("caki,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau133 = np.zeros((N, N, M, M))

    tau133 += np.einsum("ijab->ijab", tau130, optimize=True)

    tau369 = np.zeros((N, N, M, M))

    tau369 -= np.einsum("ijab->ijab", tau130, optimize=True)

    tau130 = None

    tau131 = np.zeros((N, N, M, M))

    tau131 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau131 -= np.einsum("baij->ijab", t2, optimize=True)

    tau132 = np.zeros((N, N, M, M))

    tau132 += np.einsum("kica,kjcb->ijab", tau131, u[o, o, v, v], optimize=True)

    tau133 -= np.einsum("ijab->ijab", tau132, optimize=True)

    tau134 = np.zeros((N, N, M, M))

    tau134 += np.einsum("ijcd,acbd->ijab", tau133, tau6, optimize=True)

    tau136 += 2 * np.einsum("ijab->ijab", tau134, optimize=True)

    tau312 += np.einsum("ijab->ijab", tau134, optimize=True)

    tau134 = None

    tau135 = np.zeros((N, N, M, M))

    tau135 += np.einsum("ikca,jkcb->ijab", tau133, tau25, optimize=True)

    tau136 += 3 * np.einsum("ijab->ijab", tau135, optimize=True)

    tau135 = None

    tau137 = np.zeros((N, N, M, M))

    tau137 += np.einsum("bckj,ikca->ijab", t2, tau136, optimize=True)

    tau136 = None

    tau148 += 2 * np.einsum("jiba->ijab", tau137, optimize=True)

    tau137 = None

    tau158 = np.zeros((N, N, M, M))

    tau158 += np.einsum("ijcd,acbd->ijab", tau133, tau4, optimize=True)

    tau160 += 2 * np.einsum("ijab->ijab", tau158, optimize=True)

    tau158 = None

    tau159 = np.zeros((N, N, M, M))

    tau159 += np.einsum("jkcb,ikca->ijab", tau100, tau133, optimize=True)

    tau160 += 3 * np.einsum("ijab->ijab", tau159, optimize=True)

    tau159 = None

    tau161 = np.zeros((N, N, M, M))

    tau161 += np.einsum("bckj,ikca->ijab", t2, tau160, optimize=True)

    tau160 = None

    tau167 += 2 * np.einsum("jiba->ijab", tau161, optimize=True)

    tau161 = None

    tau312 += np.einsum("ikca,jkcb->ijab", tau133, tau27, optimize=True)

    tau353 += 18 * np.einsum("ikca,jkcb->ijab", tau133, tau26, optimize=True)

    tau133 = None

    tau369 += np.einsum("ijab->ijab", tau132, optimize=True)

    tau132 = None

    tau304 = np.zeros((N, N, M, M))

    tau304 += np.einsum("bcjk,kica->ijab", l2, tau131, optimize=True)

    tau305 = np.zeros((N, N, M, M))

    tau305 += np.einsum("bcjk,ikac->ijab", t2, tau304, optimize=True)

    tau308 -= np.einsum("jiba->ijab", tau305, optimize=True)

    tau305 = None

    tau312 -= np.einsum("kjac,kibc->ijab", tau118, tau308, optimize=True)

    tau118 = None

    tau365 += 18 * np.einsum("kjbc,kica->ijab", tau308, u[o, o, v, v], optimize=True)

    r2 += np.einsum("kjbc,kaci->abij", tau308, u[o, v, v, o], optimize=True)

    tau308 = None

    tau312 -= np.einsum("ijdc,bcda->ijab", tau304, tau65, optimize=True)

    tau411 -= 9 * np.einsum("ikab,kjba->ij", tau131, tau404, optimize=True)

    tau131 = None

    tau138 = np.zeros((N, N, M, M))

    tau138 += np.einsum("caik,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau139 = np.zeros((N, N, M, M))

    tau139 += np.einsum("jkcb,ikca->ijab", tau138, tau25, optimize=True)

    tau146 = np.zeros((N, N, M, M))

    tau146 += np.einsum("ijab->ijab", tau139, optimize=True)

    tau139 = None

    tau162 = np.zeros((N, N, M, M))

    tau162 += np.einsum("ikca,jkcb->ijab", tau100, tau138, optimize=True)

    tau165 += np.einsum("ijab->ijab", tau162, optimize=True)

    tau162 = None

    tau193 = np.zeros((N, N, M, M))

    tau193 += np.einsum("ijcd,acbd->ijab", tau138, tau4, optimize=True)

    tau194 = np.zeros((N, N, M, M))

    tau194 += np.einsum("ackj,ikcb->ijab", t2, tau193, optimize=True)

    tau197 += np.einsum("ijab->ijab", tau194, optimize=True)

    tau194 = None

    r2 -= np.einsum("ijab->abij", tau197, optimize=True)

    r2 += 2 * np.einsum("ijba->abij", tau197, optimize=True) / 9

    tau197 = None

    tau345 += 9 * np.einsum("jiab->ijab", tau193, optimize=True)

    tau193 = None

    tau345 -= 9 * np.einsum("jkca,kibc->ijab", tau138, tau30, optimize=True)

    tau382 -= 36 * np.einsum("jkab,ilab->ijkl", tau138, tau306, optimize=True)

    tau306 = None

    tau388 += 4 * np.einsum("jicd,acbd->ijab", tau138, tau6, optimize=True)

    tau388 -= 36 * np.einsum("jkca,ikcb->ijab", tau138, tau26, optimize=True)

    tau138 = None

    tau140 = np.zeros((N, N, M, M))

    tau140 += np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau141 = np.zeros((N, N, M, M))

    tau141 += np.einsum("ackj,kicb->ijab", t2, tau140, optimize=True)

    tau142 = np.zeros((N, N, M, M))

    tau142 += np.einsum("ikac,jkcb->ijab", tau141, u[o, o, v, v], optimize=True)

    tau146 += np.einsum("jiab->ijab", tau142, optimize=True)

    tau142 = None

    tau149 = np.zeros((N, N, M, M))

    tau149 += np.einsum("ikac,kbjc->ijab", tau141, u[o, v, o, v], optimize=True)

    tau141 = None

    tau167 -= 6 * np.einsum("ijab->ijab", tau149, optimize=True)

    tau149 = None

    tau225 = np.zeros((N, N, M, M))

    tau225 += np.einsum("kiac,kjbc->ijab", tau116, tau140, optimize=True)

    tau228 = np.zeros((N, N, M, M))

    tau228 += np.einsum("jiba->ijab", tau225, optimize=True)

    tau225 = None

    tau265 = np.zeros((N, N, M, M))

    tau265 += np.einsum("ijcd,acdb->ijab", tau140, tau65, optimize=True)

    tau266 = np.zeros((N, N, M, M))

    tau266 += np.einsum("ackj,kibc->ijab", t2, tau265, optimize=True)

    tau265 = None

    r2 += np.einsum("ijab->abij", tau266, optimize=True) / 3

    r2 += 23 * np.einsum("ijba->abij", tau266, optimize=True) / 36

    tau266 = None

    tau330 = np.zeros((N, N, M, M))

    tau330 += np.einsum("acik,kjcb->ijab", t2, tau140, optimize=True)

    tau332 += np.einsum("ijab->ijab", tau330, optimize=True)

    tau330 = None

    tau337 += 9 * np.einsum("kjbc,kiac->ijab", tau332, u[o, o, v, v], optimize=True)

    r2 += np.einsum("kibc,kajc->abij", tau332, u[o, v, o, v], optimize=True)

    tau332 = None

    tau143 = np.zeros((N, N, M, M))

    tau143 += np.einsum("caik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau144 += np.einsum("ijab->ijab", tau143, optimize=True)

    tau143 = None

    tau145 = np.zeros((N, N, M, M))

    tau145 += np.einsum("jkcb,ikca->ijab", tau100, tau144, optimize=True)

    tau100 = None

    tau146 += np.einsum("jiba->ijab", tau145, optimize=True)

    tau145 = None

    tau147 = np.zeros((N, N, M, M))

    tau147 += np.einsum("bckj,kiac->ijab", t2, tau146, optimize=True)

    tau146 = None

    tau148 += 6 * np.einsum("ijba->ijab", tau147, optimize=True)

    tau147 = None

    r2 += np.einsum("ijab->abij", tau148, optimize=True) / 9

    r2 += np.einsum("ijba->abij", tau148, optimize=True) / 18

    tau148 = None

    tau164 = np.zeros((N, N, M, M))

    tau164 += np.einsum("ikca,jkcb->ijab", tau144, tau25, optimize=True)

    tau165 += np.einsum("jiba->ijab", tau164, optimize=True)

    tau164 = None

    tau166 = np.zeros((N, N, M, M))

    tau166 += np.einsum("bckj,kiac->ijab", t2, tau165, optimize=True)

    tau165 = None

    tau167 += 6 * np.einsum("ijba->ijab", tau166, optimize=True)

    tau166 = None

    r2 += np.einsum("ijab->abij", tau167, optimize=True) / 18

    r2 += np.einsum("ijba->abij", tau167, optimize=True) / 9

    tau167 = None

    tau223 = np.zeros((N, N, M, M))

    tau223 += np.einsum("ijcd,acbd->ijab", tau144, tau4, optimize=True)

    tau4 = None

    tau224 = np.zeros((N, N, M, M))

    tau224 += np.einsum("bckj,ikca->ijab", t2, tau223, optimize=True)

    tau223 = None

    r2 += 11 * np.einsum("ijba->abij", tau224, optimize=True) / 18

    r2 += 2 * np.einsum("ijab->abij", tau224, optimize=True) / 9

    tau224 = None

    tau233 = np.zeros((N, N, M, M))

    tau233 += np.einsum("ijcd,acbd->ijab", tau144, tau6, optimize=True)

    tau6 = None

    tau234 = np.zeros((N, N, M, M))

    tau234 += np.einsum("bckj,ikca->ijab", t2, tau233, optimize=True)

    r2 += 2 * np.einsum("ijba->abij", tau234, optimize=True) / 9

    r2 += 11 * np.einsum("ijab->abij", tau234, optimize=True) / 18

    tau234 = None

    tau345 += 9 * np.einsum("jiab->ijab", tau233, optimize=True)

    tau233 = None

    tau337 += 9 * np.einsum("jkca,ikcb->ijab", tau144, tau26, optimize=True)

    tau345 += 9 * np.einsum("jkca,ikcb->ijab", tau144, tau27, optimize=True)

    tau181 = np.zeros((N, N, M, M))

    tau181 += np.einsum("caik,bcjk->ijab", l2, t2, optimize=True)

    tau182 += np.einsum("ijab->ijab", tau181, optimize=True)

    tau181 = None

    tau183 = np.zeros((N, N, M, M))

    tau183 += np.einsum("klab,likj->ijab", tau182, u[o, o, o, o], optimize=True)

    tau182 = None

    tau184 += np.einsum("ijab->ijab", tau183, optimize=True)

    tau183 = None

    tau185 = np.zeros((N, N, M, M))

    tau185 += np.einsum("bckj,kica->ijab", t2, tau184, optimize=True)

    tau184 = None

    tau186 += np.einsum("jiba->ijab", tau185, optimize=True)

    tau185 = None

    r2 += np.einsum("ijba->abij", tau186, optimize=True) / 9

    r2 += np.einsum("jiba->abij", tau186, optimize=True) / 18

    tau186 = None

    tau217 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau218 = np.zeros((N, N, M, M))

    tau218 += np.einsum("kiac,kjbc->ijab", tau217, tau48, optimize=True)

    tau219 = np.zeros((N, N, M, M))

    tau219 -= np.einsum("bckj,kiac->ijab", t2, tau218, optimize=True)

    tau218 = None

    tau222 = np.zeros((N, N, M, M))

    tau222 += np.einsum("ijba->ijab", tau219, optimize=True)

    tau219 = None

    tau220 = np.zeros((N, N, M, M))

    tau220 += np.einsum("kilj,klab->ijab", tau2, tau217, optimize=True)

    tau221 = np.zeros((N, N, M, M))

    tau221 -= np.einsum("bckj,kiac->ijab", t2, tau220, optimize=True)

    tau222 += np.einsum("jiba->ijab", tau221, optimize=True)

    tau221 = None

    r2 -= 2 * np.einsum("ijab->abij", tau222, optimize=True) / 9

    r2 += np.einsum("ijba->abij", tau222, optimize=True)

    tau222 = None

    tau312 += np.einsum("jiba->ijab", tau220, optimize=True)

    tau220 = None

    tau237 = np.zeros((N, N, M, M))

    tau237 += np.einsum("kiac,kjbc->ijab", tau217, tau85, optimize=True)

    tau85 = None

    tau238 -= np.einsum("jiba->ijab", tau237, optimize=True)

    tau237 = None

    tau239 = np.zeros((N, N, M, M))

    tau239 += np.einsum("bckj,ikca->ijab", t2, tau238, optimize=True)

    tau238 = None

    tau242 += np.einsum("jiba->ijab", tau239, optimize=True)

    tau239 = None

    r2 -= 4 * np.einsum("ijab->abij", tau242, optimize=True) / 9

    r2 -= np.einsum("ijba->abij", tau242, optimize=True) / 3

    tau242 = None

    tau245 = np.zeros((N, N, M, M))

    tau245 += np.einsum("kiac,kjbc->ijab", tau217, tau49, optimize=True)

    tau49 = None

    tau247 -= np.einsum("jiba->ijab", tau245, optimize=True)

    tau245 = None

    tau248 = np.zeros((N, N, M, M))

    tau248 += np.einsum("bckj,ikca->ijab", t2, tau247, optimize=True)

    tau247 = None

    tau254 += np.einsum("jiba->ijab", tau248, optimize=True)

    tau248 = None

    tau250 = np.zeros((N, N, M, M))

    tau250 += np.einsum("kjbc,kiac->ijab", tau140, tau217, optimize=True)

    tau140 = None

    tau252 -= np.einsum("jiba->ijab", tau250, optimize=True)

    tau345 += 9 * np.einsum("ijba->ijab", tau250, optimize=True)

    tau250 = None

    tau312 -= np.einsum("kjbc,ikca->ijab", tau217, tau304, optimize=True)

    tau304 = None

    tau316 = np.zeros((M, M, M, M))

    tau316 += np.einsum("ijcd,ijab->abcd", tau217, tau27, optimize=True)

    tau27 = None

    tau320 += 9 * np.einsum("abcd->abcd", tau316, optimize=True)

    tau321 += 18 * np.einsum("abcd->abcd", tau316, optimize=True)

    tau316 = None

    tau337 += np.einsum("klba,ikjl->ijab", tau217, tau334, optimize=True)

    tau334 = None

    tau345 += 9 * np.einsum("ikjl,klba->ijab", tau2, tau217, optimize=True)

    tau365 -= 2 * np.einsum("iklj,klba->ijab", tau2, tau217, optimize=True)

    tau399 -= 36 * np.einsum("lkab,ijab->ijkl", tau217, tau26, optimize=True)

    tau226 = np.zeros((N, N, M, M))

    tau226 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    tau227 = np.zeros((N, N, M, M))

    tau227 += np.einsum("kiac,kjbc->ijab", tau217, tau226, optimize=True)

    tau228 -= np.einsum("jiba->ijab", tau227, optimize=True)

    tau227 = None

    tau229 = np.zeros((N, N, M, M))

    tau229 += np.einsum("bckj,ikca->ijab", t2, tau228, optimize=True)

    tau228 = None

    tau232 += np.einsum("ijba->ijab", tau229, optimize=True)

    tau229 = None

    r2 -= 11 * np.einsum("ijab->abij", tau232, optimize=True) / 18

    r2 -= np.einsum("ijba->abij", tau232, optimize=True) / 3

    tau232 = None

    tau249 = np.zeros((N, N, M, M))

    tau249 += np.einsum("kiac,kjbc->ijab", tau116, tau226, optimize=True)

    tau252 += np.einsum("jiba->ijab", tau249, optimize=True)

    tau253 = np.zeros((N, N, M, M))

    tau253 += np.einsum("bckj,ikca->ijab", t2, tau252, optimize=True)

    tau252 = None

    tau254 += np.einsum("ijba->ijab", tau253, optimize=True)

    tau253 = None

    r2 -= 2 * np.einsum("ijab->abij", tau254, optimize=True) / 9

    r2 -= 2 * np.einsum("ijba->abij", tau254, optimize=True) / 3

    tau254 = None

    tau345 -= 9 * np.einsum("ijba->ijab", tau249, optimize=True)

    tau249 = None

    tau273 = np.zeros((N, N, M, M))

    tau273 += np.einsum("ijcd,acdb->ijab", tau226, tau65, optimize=True)

    tau274 = np.zeros((N, N, M, M))

    tau274 += np.einsum("ackj,kibc->ijab", t2, tau273, optimize=True)

    tau273 = None

    tau275 += np.einsum("ijab->ijab", tau274, optimize=True)

    tau274 = None

    r2 += 2 * np.einsum("ijab->abij", tau275, optimize=True) / 3

    r2 += 5 * np.einsum("ijba->abij", tau275, optimize=True) / 18

    tau275 = None

    tau328 = np.zeros((N, N, M, M))

    tau328 += 9 * np.einsum("ijab->ijab", tau226, optimize=True)

    tau340 += np.einsum("ijab->ijab", tau226, optimize=True)

    tau226 = None

    tau342 = np.zeros((N, N, M, M))

    tau342 += 9 * np.einsum("ackj,kicb->ijab", t2, tau340, optimize=True)

    tau340 = None

    tau255 = np.zeros((N, N, M, M))

    tau255 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau257 += np.einsum("ijab->ijab", tau255, optimize=True)

    tau258 = np.zeros((N, N, M, M))

    tau258 += np.einsum("bcjk,kiac->ijab", t2, tau257, optimize=True)

    tau257 = None

    tau259 = np.zeros((N, N, M, M))

    tau259 += np.einsum("bckj,kica->ijab", t2, tau258, optimize=True)

    tau258 = None

    tau264 -= 2 * np.einsum("ijab->ijab", tau259, optimize=True)

    tau259 = None

    r2 += np.einsum("ijba->abij", tau264, optimize=True) / 18

    r2 += np.einsum("jiba->abij", tau264, optimize=True) / 9

    tau264 = None

    tau279 += np.einsum("ijab->ijab", tau255, optimize=True)

    tau280 = np.zeros((N, N, M, M))

    tau280 += np.einsum("bckj,kiac->ijab", t2, tau279, optimize=True)

    tau279 = None

    tau281 = np.zeros((N, N, M, M))

    tau281 += np.einsum("bckj,kica->ijab", t2, tau280, optimize=True)

    tau280 = None

    r2 -= 2 * np.einsum("ijab->abij", tau281, optimize=True) / 9

    r2 -= 11 * np.einsum("ijba->abij", tau281, optimize=True) / 18

    r2 -= 4 * np.einsum("jiab->abij", tau281, optimize=True) / 9

    r2 -= 2 * np.einsum("jiba->abij", tau281, optimize=True) / 9

    tau281 = None

    tau405 -= np.einsum("ijab->ijab", tau255, optimize=True)

    tau411 += 9 * np.einsum("baik,kjab->ij", t2, tau405, optimize=True)

    tau405 = None

    tau414 -= np.einsum("bakj,kiba->ij", t2, tau255, optimize=True)

    tau255 = None

    tau309 = np.zeros((N, N, M, M))

    tau309 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau312 -= np.einsum("jicd,cbad->ijab", tau309, tau56, optimize=True)

    tau56 = None

    tau353 += 18 * np.einsum("jicd,bcad->ijab", tau309, u[v, v, v, v], optimize=True)

    tau365 += 18 * np.einsum("ijcd,bcda->ijab", tau309, u[v, v, v, v], optimize=True)

    tau382 += 36 * np.einsum("ilab,kajb->ijkl", tau309, u[o, v, o, v], optimize=True)

    tau395 = np.zeros((N, N, M, M))

    tau395 -= 3 * np.einsum("ijab->ijab", tau309, optimize=True)

    tau309 = None

    tau310 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau312 += np.einsum("kjli,lkab->ijab", tau310, tau50, optimize=True)

    tau50 = None

    r2 -= np.einsum("acik,jkcb->abij", t2, tau312, optimize=True)

    tau312 = None

    tau345 -= 9 * np.einsum("lkba,lijk->ijab", tau30, tau310, optimize=True)

    tau30 = None

    tau382 += np.einsum("mkjn,niml->ijkl", tau310, tau381, optimize=True)

    tau381 = None

    tau399 -= 4 * np.einsum("mknl,injm->ijkl", tau310, tau397, optimize=True)

    tau397 = None

    tau399 += np.einsum("mkjn,niml->ijkl", tau310, tau398, optimize=True)

    tau398 = None

    tau399 -= 36 * np.einsum("im,mkjl->ijkl", tau14, tau310, optimize=True)

    r2 -= np.einsum("klij,klab->abij", tau310, tau347, optimize=True)

    tau347 = None

    tau310 = None

    tau314 = np.zeros((N, N, M, M))

    tau314 += 7 * np.einsum("abij->ijab", t2, optimize=True)

    tau314 -= np.einsum("abji->ijab", t2, optimize=True)

    tau315 = np.zeros((M, M, M, M))

    tau315 += np.einsum("abij,ijcd->abcd", l2, tau314, optimize=True)

    tau320 += np.einsum("aebf,ecfd->abcd", tau315, u[v, v, v, v], optimize=True)

    tau315 = None

    tau351 = np.zeros((N, N, M, M))

    tau351 += np.einsum("caik,kjbc->ijab", l2, tau314, optimize=True)

    tau314 = None

    tau318 += np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau319 = np.zeros((M, M, M, M))

    tau319 += np.einsum("jicd,ijab->abcd", tau26, tau318, optimize=True)

    tau26 = None

    tau318 = None

    tau320 -= 9 * np.einsum("bdac->abcd", tau319, optimize=True)

    r2 -= np.einsum("cdij,cabd->abij", t2, tau320, optimize=True) / 9

    tau320 = None

    tau321 -= 18 * np.einsum("bdac->abcd", tau319, optimize=True)

    tau319 = None

    r2 -= np.einsum("dcij,cbad->abij", t2, tau321, optimize=True) / 18

    tau321 = None

    tau322 = np.zeros((N, N, M, M))

    tau322 -= np.einsum("abij->ijab", l2, optimize=True)

    tau322 += 7 * np.einsum("abji->ijab", l2, optimize=True)

    tau323 = np.zeros((N, N, M, M))

    tau323 += np.einsum("cbjk,kica->ijab", t2, tau322, optimize=True)

    tau322 = None

    tau324 += np.einsum("ijab->ijab", tau323, optimize=True)

    tau323 = None

    tau325 = np.zeros((N, N, M, M))

    tau325 += np.einsum("ijcd,cadb->ijab", tau324, u[v, v, v, v], optimize=True)

    tau324 = None

    tau337 += np.einsum("ijba->ijab", tau325, optimize=True)

    tau345 += np.einsum("ijba->ijab", tau325, optimize=True)

    tau325 = None

    tau326 = np.zeros((N, N, M, M))

    tau326 += 7 * np.einsum("abij->ijab", l2, optimize=True)

    tau326 -= np.einsum("abji->ijab", l2, optimize=True)

    tau327 = np.zeros((N, N, M, M))

    tau327 += np.einsum("bcjk,ikac->ijab", t2, tau326, optimize=True)

    tau326 = None

    tau328 += np.einsum("ijab->ijab", tau327, optimize=True)

    tau327 = None

    tau329 = np.zeros((N, N, M, M))

    tau329 += np.einsum("ijcd,acbd->ijab", tau328, tau65, optimize=True)

    tau328 = None

    tau337 += np.einsum("ijba->ijab", tau329, optimize=True)

    tau345 += np.einsum("ijba->ijab", tau329, optimize=True)

    tau329 = None

    tau333 = np.zeros((N, N, M, M))

    tau333 += np.einsum("acki,bcjk->ijab", l2, t2, optimize=True)

    tau337 -= np.einsum("kibc,kjac->ijab", tau217, tau333, optimize=True)

    tau333 = None

    tau335 = np.zeros((N, N, M, M))

    tau335 += np.einsum("abij->ijab", l2, optimize=True)

    tau335 += 2 * np.einsum("abji->ijab", l2, optimize=True)

    tau336 = np.zeros((N, N, M, M))

    tau336 += np.einsum("bckj,ikac->ijab", t2, tau335, optimize=True)

    tau337 -= 3 * np.einsum("klab,jkli->ijab", tau336, tau78, optimize=True)

    r2 -= np.einsum("ackj,kicb->abij", t2, tau337, optimize=True) / 9

    tau337 = None

    tau382 += 12 * np.einsum("jkab,ilab->ijkl", tau144, tau336, optimize=True)

    tau336 = None

    tau352 = np.zeros((N, N, M, M))

    tau352 += np.einsum("bckj,kica->ijab", t2, tau335, optimize=True)

    tau353 -= 5 * np.einsum("klab,kijl->ijab", tau352, tau78, optimize=True)

    tau352 = None

    tau375 = np.zeros((N, N, M, M))

    tau375 += np.einsum("jkbc,icka->ijab", tau335, u[o, v, o, v], optimize=True)

    tau377 += np.einsum("jiba->ijab", tau375, optimize=True)

    tau394 = np.zeros((N, N, M, M))

    tau394 += np.einsum("jiba->ijab", tau375, optimize=True)

    tau375 = None

    tau376 = np.zeros((N, N, M, M))

    tau376 += np.einsum("kjcb,kica->ijab", tau10, tau335, optimize=True)

    tau335 = None

    tau377 -= np.einsum("ijab->ijab", tau376, optimize=True)

    tau382 -= 12 * np.einsum("abjl,ikab->ijkl", t2, tau377, optimize=True)

    tau377 = None

    tau394 -= np.einsum("ijab->ijab", tau376, optimize=True)

    tau376 = None

    tau399 -= 12 * np.einsum("bajl,ikab->ijkl", t2, tau394, optimize=True)

    tau394 = None

    tau338 = np.zeros((N, N, M, M))

    tau338 -= np.einsum("abij->ijab", t2, optimize=True)

    tau338 += 2 * np.einsum("baij->ijab", t2, optimize=True)

    tau339 = np.zeros((N, N, M, M))

    tau339 += np.einsum("bcjk,ikca->ijab", l2, tau338, optimize=True)

    tau342 -= 9 * np.einsum("acjk,ikbc->ijab", t2, tau339, optimize=True)

    tau345 -= 9 * np.einsum("jidc,bcda->ijab", tau339, tau65, optimize=True)

    tau339 = None

    tau344 = np.zeros((N, N, M, M))

    tau344 += np.einsum("caki,jkcb->ijab", l2, tau338, optimize=True)

    tau345 -= 9 * np.einsum("ijcd,cbad->ijab", tau344, u[v, v, v, v], optimize=True)

    tau344 = None

    tau412 -= np.einsum("ca,jibc->ijab", tau12, tau338, optimize=True)

    tau12 = None

    tau414 += np.einsum("jkab,kiab->ij", tau412, u[o, o, v, v], optimize=True)

    tau412 = None

    tau414 -= np.einsum("kjab,kiba->ij", tau338, tau404, optimize=True)

    tau338 = None

    tau404 = None

    tau341 = np.zeros((N, N, M, M))

    tau341 += 5 * np.einsum("abij->ijab", t2, optimize=True)

    tau341 -= 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau342 += np.einsum("klij,klab->ijab", tau2, tau341, optimize=True)

    r2 += np.einsum("jkbc,ikac->abij", tau116, tau342, optimize=True) / 9

    tau116 = None

    tau342 = None

    tau346 = np.zeros((M, M, M, M))

    tau346 += np.einsum("abij,ijcd->abcd", l2, tau341, optimize=True)

    tau341 = None

    tau353 += 2 * np.einsum("ijcd,acbd->ijab", tau10, tau346, optimize=True)

    tau10 = None

    tau346 = None

    tau343 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau343 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau345 -= 9 * np.einsum("kibc,kjac->ijab", tau343, tau48, optimize=True)

    tau48 = None

    tau343 = None

    r2 -= np.einsum("bcjk,kica->abij", t2, tau345, optimize=True) / 9

    tau345 = None

    tau350 = np.zeros((N, N, M, M))

    tau350 -= np.einsum("abij->ijab", t2, optimize=True)

    tau350 += 7 * np.einsum("abji->ijab", t2, optimize=True)

    tau351 += np.einsum("caki,kjbc->ijab", l2, tau350, optimize=True)

    tau350 = None

    tau353 += 2 * np.einsum("klab,kilj->ijab", tau351, tau78, optimize=True)

    tau351 = None

    r2 -= np.einsum("bcki,jkca->abij", t2, tau353, optimize=True) / 18

    tau353 = None

    tau355 = np.zeros((N, N, M, M))

    tau355 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau355 += np.einsum("baij->ijab", t2, optimize=True)

    tau356 -= np.einsum("ca,ijcb->ijab", tau89, tau355, optimize=True)

    tau89 = None

    tau355 = None

    r2 -= np.einsum("ijcd,badc->abij", tau356, u[v, v, v, v], optimize=True) / 18

    tau356 = None

    tau358 = np.zeros((N, N, M, M))

    tau358 += np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau359 = np.zeros((N, N, M, M))

    tau359 += np.einsum("ijcd,acbd->ijab", tau358, tau65, optimize=True)

    tau65 = None

    tau358 = None

    tau365 -= np.einsum("ijba->ijab", tau359, optimize=True)

    tau400 += np.einsum("ijab->ijab", tau359, optimize=True)

    tau359 = None

    r2 += np.einsum("bcik,kjac->abij", t2, tau400, optimize=True) / 18

    tau400 = None

    tau360 = np.zeros((N, N, M, M))

    tau360 += 5 * np.einsum("abij->ijab", l2, optimize=True)

    tau360 -= 2 * np.einsum("abji->ijab", l2, optimize=True)

    tau361 = np.zeros((N, N, M, M))

    tau361 += np.einsum("cbkj,ikac->ijab", t2, tau360, optimize=True)

    tau360 = None

    tau365 += 2 * np.einsum("kibc,kjac->ijab", tau217, tau361, optimize=True)

    tau217 = None

    tau361 = None

    tau362 = np.zeros((N, N, M, M))

    tau362 += 2 * np.einsum("abij->ijab", l2, optimize=True)

    tau362 += np.einsum("abji->ijab", l2, optimize=True)

    tau363 = np.zeros((N, N, M, M))

    tau363 += np.einsum("bckj,ikac->ijab", t2, tau362, optimize=True)

    tau364 += np.einsum("ijab->ijab", tau363, optimize=True)

    tau365 -= 6 * np.einsum("klab,kjil->ijab", tau364, tau78, optimize=True)

    tau364 = None

    r2 -= np.einsum("acki,kjcb->abij", t2, tau365, optimize=True) / 18

    tau365 = None

    tau389 += np.einsum("ijab->ijab", tau363, optimize=True)

    tau363 = None

    tau399 += 12 * np.einsum("jkab,ilab->ijkl", tau144, tau389, optimize=True)

    tau144 = None

    tau389 = None

    tau368 += np.einsum("cbjk,kica->ijab", t2, tau362, optimize=True)

    tau374 += np.einsum("ikac,jckb->ijab", tau362, u[o, v, o, v], optimize=True)

    tau395 += np.einsum("bckj,kica->ijab", t2, tau362, optimize=True)

    tau362 = None

    tau399 -= 12 * np.einsum("ilab,kabj->ijkl", tau395, u[o, v, v, o], optimize=True)

    tau395 = None

    tau369 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau382 -= 12 * np.einsum("ijab,lkab->ijkl", tau368, tau369, optimize=True)

    tau368 = None

    tau399 -= 12 * np.einsum("ijab,lkab->ijkl", tau105, tau369, optimize=True)

    tau105 = None

    tau369 = None

    tau370 = np.zeros((N, N, M, M))

    tau370 -= 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau370 += 3 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau371 = np.zeros((N, N, M, M))

    tau371 += np.einsum("bckj,kica->ijab", t2, tau370, optimize=True)

    tau370 = None

    tau372 += np.einsum("jiba->ijab", tau371, optimize=True)

    tau371 = None

    tau373 = np.zeros((N, N, M, M))

    tau373 += np.einsum("cbkj,kica->ijab", l2, tau372, optimize=True)

    tau372 = None

    tau374 += np.einsum("jiba->ijab", tau373, optimize=True)

    tau382 -= 12 * np.einsum("bajl,ikab->ijkl", t2, tau374, optimize=True)

    tau374 = None

    tau391 -= np.einsum("jiba->ijab", tau373, optimize=True)

    tau373 = None

    tau379 = np.zeros((N, N, M, M))

    tau379 += 3 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau379 -= 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau382 += 12 * np.einsum("ilab,kjab->ijkl", tau25, tau379, optimize=True)

    tau25 = None

    tau379 = None

    r2 += np.einsum("ablk,kilj->abij", t2, tau382, optimize=True) / 36

    tau382 = None

    tau383 = np.zeros((N, N, M, M))

    tau383 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau383 += np.einsum("abji->ijab", t2, optimize=True)

    tau385 = np.zeros((N, N, M, M))

    tau385 += np.einsum("caik,kjbc->ijab", l2, tau383, optimize=True)

    tau383 = None

    tau384 = np.zeros((N, N, M, M))

    tau384 += np.einsum("abij->ijab", t2, optimize=True)

    tau384 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau385 += np.einsum("caki,kjbc->ijab", l2, tau384, optimize=True)

    tau384 = None

    tau388 += 2 * np.einsum("klab,jkil->ijab", tau385, tau78, optimize=True)

    tau385 = None

    tau386 = np.zeros((N, N, M, M))

    tau386 += 23 * np.einsum("abij->ijab", l2, optimize=True)

    tau386 += 10 * np.einsum("abji->ijab", l2, optimize=True)

    tau387 += np.einsum("bckj,kica->ijab", t2, tau386, optimize=True)

    tau386 = None

    tau388 += np.einsum("klab,jkli->ijab", tau387, tau78, optimize=True)

    tau387 = None

    tau78 = None

    r2 += np.einsum("bckj,kica->abij", t2, tau388, optimize=True) / 36

    tau388 = None

    tau390 = np.zeros((N, N, M, M))

    tau390 += 3 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau390 -= 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau391 += np.einsum("acik,jkcb->ijab", l2, tau390, optimize=True)

    tau390 = None

    tau399 += 12 * np.einsum("abjl,ikab->ijkl", t2, tau391, optimize=True)

    tau391 = None

    tau392 = np.zeros((N, N, M, M))

    tau392 += 3 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau392 -= 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau393 += np.einsum("acki,kjcb->ijab", t2, tau392, optimize=True)

    tau392 = None

    tau393 += 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau393 -= 3 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau399 -= 12 * np.einsum("lkab,ijab->ijkl", tau393, tau97, optimize=True)

    tau97 = None

    tau393 = None

    r2 += np.einsum("abkl,kilj->abij", t2, tau399, optimize=True) / 36

    tau399 = None

    tau407 += 7 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau407 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau411 += np.einsum("lmik,kjlm->ij", tau2, tau407, optimize=True)

    tau407 = None

    tau409 = np.zeros((N, N, N, N))

    tau409 -= np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau409 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau410 = np.zeros((N, N))

    tau410 += np.einsum("lk,kilj->ij", tau14, tau409, optimize=True)

    tau409 = None

    tau14 = None

    tau411 -= 9 * np.einsum("ji->ij", tau410, optimize=True)

    r2 -= np.einsum("ik,abkj->abij", tau411, t2, optimize=True) / 9

    tau411 = None

    tau414 -= np.einsum("ij->ij", tau410, optimize=True)

    tau410 = None

    r2 -= np.einsum("kj,abik->abij", tau414, t2, optimize=True)

    tau414 = None

    tau415 += np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau415 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau416 = np.zeros((N, N))

    tau416 += np.einsum("lmki,kjlm->ij", tau2, tau415, optimize=True)

    tau2 = None

    tau415 = None

    r2 += np.einsum("jk,abki->abij", tau416, t2, optimize=True) / 18

    tau416 = None

    return r2
