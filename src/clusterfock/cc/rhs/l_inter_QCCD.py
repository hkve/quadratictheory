import numpy as np


def lambda_amplitudes_intermediates_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau28 = zeros((N, N, M, M))

    tau28 -= 2 * np.einsum("jiba->ijab", tau0, optimize=True)

    tau56 = zeros((N, N, M, M))

    tau56 -= 2 * np.einsum("jiba->ijab", tau0, optimize=True)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("jiba->abij", tau0, optimize=True) / 2

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau45 = zeros((N, N, M, M))

    tau45 -= 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau1 = None

    tau2 = zeros((M, M))

    tau2 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 -= np.einsum("ac,ibjc->ijab", tau2, u[o, v, o, v], optimize=True)

    tau18 = zeros((N, N, M, M))

    tau18 -= 2 * np.einsum("jiab->ijab", tau3, optimize=True)

    tau3 = None

    tau8 = zeros((N, N, M, M))

    tau8 -= np.einsum("ac,ijbc->ijab", tau2, u[o, o, v, v], optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 -= 2 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau52 = zeros((N, N, M, M))

    tau52 -= np.einsum("ijab->ijab", tau8, optimize=True)

    tau56 += 2 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau60 = zeros((N, N, M, M))

    tau60 -= 4 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau70 = zeros((N, N, M, M))

    tau70 -= 4 * np.einsum("ijba->ijab", tau8, optimize=True)

    tau8 = None

    tau40 = zeros((N, N, M, M))

    tau40 -= np.einsum("cb,acji->ijab", tau2, t2, optimize=True)

    tau41 = zeros((N, N, M, M))

    tau41 += 2 * np.einsum("ijab->ijab", tau40, optimize=True)

    tau79 = zeros((N, N, M, M))

    tau79 += np.einsum("ijab->ijab", tau40, optimize=True)

    tau40 = None

    tau46 = zeros((M, M, M, M))

    tau46 += np.einsum("ae,cbde->abcd", tau2, u[v, v, v, v], optimize=True)

    tau50 = zeros((M, M, M, M))

    tau50 += np.einsum("acbd->abcd", tau46, optimize=True)

    tau46 = None

    tau55 = zeros((M, M))

    tau55 -= np.einsum("cd,cabd->ab", tau2, u[v, v, v, v], optimize=True)

    tau58 = zeros((M, M))

    tau58 -= 4 * np.einsum("ab->ab", tau55, optimize=True)

    tau55 = None

    tau69 = zeros((N, N))

    tau69 += np.einsum("ab,iajb->ij", tau2, u[o, v, o, v], optimize=True)

    tau72 = zeros((N, N))

    tau72 += 4 * np.einsum("ji->ij", tau69, optimize=True)

    tau69 = None

    tau76 = zeros((M, M, M, M))

    tau76 += np.einsum("ac,bd->abcd", tau2, tau2, optimize=True)

    tau4 = zeros((M, M, M, M))

    tau4 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau47 = zeros((M, M, M, M))

    tau47 -= np.einsum("aefb,cedf->abcd", tau4, u[v, v, v, v], optimize=True)

    tau50 -= 2 * np.einsum("abcd->abcd", tau47, optimize=True)

    tau47 = None

    tau76 += np.einsum("afde,becf->abcd", tau4, tau4, optimize=True)

    tau5 = zeros((N, N, M, M))

    tau5 -= np.einsum("caki,jkcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ijab->ijab", tau5, optimize=True)

    tau5 = None

    tau6 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau7 = zeros((N, N, M, M))

    tau7 += np.einsum("cadb,ijcd->ijab", tau4, tau6, optimize=True)

    tau18 -= 2 * np.einsum("ijab->ijab", tau7, optimize=True)

    tau7 = None

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("cbkj,kica->ijab", l2, tau6, optimize=True)

    tau13 += 4 * np.einsum("jiba->ijab", tau9, optimize=True)

    tau28 -= 4 * np.einsum("jiba->ijab", tau9, optimize=True)

    tau52 += 4 * np.einsum("jiba->ijab", tau9, optimize=True)

    tau56 -= 8 * np.einsum("jiba->ijab", tau9, optimize=True)

    tau62 = zeros((N, N, M, M))

    tau62 += 4 * np.einsum("jiba->ijab", tau9, optimize=True)

    tau70 -= 8 * np.einsum("jiba->ijab", tau9, optimize=True)

    tau9 = None

    tau10 = zeros((N, N, N, N))

    tau10 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau11 = zeros((N, N, N, N))

    tau11 += np.einsum("lkji->ijkl", tau10, optimize=True)

    tau10 = None

    tau11 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("abkl,ijkl->ijab", l2, tau11, optimize=True)

    tau13 += np.einsum("jiba->ijab", tau12, optimize=True)

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("cbkj,kiac->ijab", t2, tau13, optimize=True)

    tau13 = None

    tau18 -= np.einsum("jiab->ijab", tau14, optimize=True)

    tau14 = None

    tau70 -= np.einsum("jiba->ijab", tau12, optimize=True)

    tau12 = None

    tau15 = zeros((N, N, M, M))

    tau15 -= np.einsum("caik,cbkj->ijab", l2, t2, optimize=True)

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("kjbc,kiac->ijab", tau15, tau6, optimize=True)

    tau18 += 4 * np.einsum("jiba->ijab", tau16, optimize=True)

    tau16 = None

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("kilj,lkab->ijab", tau11, tau15, optimize=True)

    tau18 -= 2 * np.einsum("jiab->ijab", tau17, optimize=True)

    tau17 = None

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("cbkj,kiac->ijab", l2, tau18, optimize=True)

    tau18 = None

    tau45 -= np.einsum("jiab->ijab", tau19, optimize=True)

    tau19 = None

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("jkcb,ikca->ijab", tau15, tau6, optimize=True)

    tau32 = zeros((N, N, M, M))

    tau32 -= 4 * np.einsum("jiba->ijab", tau30, optimize=True)

    tau30 = None

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum("ikcb,kjac->ijab", tau15, tau15, optimize=True)

    tau43 = zeros((N, N, M, M))

    tau43 += 4 * np.einsum("ijab->ijab", tau34, optimize=True)

    tau34 = None

    tau35 = zeros((N, N, M, M))

    tau35 += np.einsum("ijdc,acbd->ijab", tau15, tau4, optimize=True)

    tau43 -= 2 * np.einsum("ijab->ijab", tau35, optimize=True)

    tau35 = None

    tau38 = zeros((N, N, M, M))

    tau38 += np.einsum("caki,kjcb->ijab", t2, tau15, optimize=True)

    tau41 += 4 * np.einsum("ijba->ijab", tau38, optimize=True)

    tau77 = zeros((N, N, M, M))

    tau77 += 2 * np.einsum("ijba->ijab", tau38, optimize=True)

    tau79 -= 2 * np.einsum("ijab->ijab", tau38, optimize=True)

    tau38 = None

    tau49 = zeros((M, M, M, M))

    tau49 += np.einsum("ijcd,ijab->abcd", tau15, tau6, optimize=True)

    tau50 -= 4 * np.einsum("cdab->abcd", tau49, optimize=True)

    tau49 = None

    tau64 = zeros((N, N, N, N))

    tau64 += np.einsum("klab,ijab->ijkl", tau15, tau6, optimize=True)

    tau66 = zeros((N, N, N, N))

    tau66 -= 8 * np.einsum("jkil->ijkl", tau64, optimize=True)

    tau64 = None

    tau76 -= 4 * np.einsum("ijac,jibd->abcd", tau15, tau15, optimize=True)

    r2 -= np.einsum("abcd,jicd->abij", tau76, u[o, o, v, v], optimize=True) / 4

    tau76 = None

    tau78 = zeros((N, N, N, N))

    tau78 += 4 * np.einsum("ilba,jkab->ijkl", tau15, tau15, optimize=True)

    tau20 = zeros((N, N))

    tau20 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 -= np.einsum("ik,kajb->ijab", tau20, u[o, v, o, v], optimize=True)

    tau32 += 2 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau21 = None

    tau27 = zeros((N, N, M, M))

    tau27 -= np.einsum("ik,jkab->ijab", tau20, u[o, o, v, v], optimize=True)

    tau28 -= 2 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau56 += 4 * np.einsum("jiba->ijab", tau27, optimize=True)

    tau62 += np.einsum("ijba->ijab", tau27, optimize=True)

    tau63 = zeros((N, N, N, N))

    tau63 += np.einsum("bakl,ijab->ijkl", t2, tau62, optimize=True)

    tau62 = None

    tau66 -= np.einsum("iklj->ijkl", tau63, optimize=True)

    tau63 = None

    tau70 -= 2 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau74 = zeros((N, N, M, M))

    tau74 -= 4 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau27 = None

    tau39 = zeros((N, N, M, M))

    tau39 += np.einsum("kj,abik->ijab", tau20, t2, optimize=True)

    tau41 -= 2 * np.einsum("ijba->ijab", tau39, optimize=True)

    tau77 -= np.einsum("ijba->ijab", tau39, optimize=True)

    tau39 = None

    tau43 += np.einsum("ab,ij->ijab", tau2, tau20, optimize=True)

    tau2 = None

    tau54 = zeros((M, M))

    tau54 += np.einsum("ij,jaib->ab", tau20, u[o, v, o, v], optimize=True)

    tau58 += 4 * np.einsum("ab->ab", tau54, optimize=True)

    tau54 = None

    tau61 = zeros((N, N, N, N))

    tau61 += np.einsum("im,jmlk->ijkl", tau20, u[o, o, o, o], optimize=True)

    tau66 -= 2 * np.einsum("iklj->ijkl", tau61, optimize=True)

    tau61 = None

    tau68 = zeros((N, N))

    tau68 -= np.einsum("kl,ilkj->ij", tau20, u[o, o, o, o], optimize=True)

    tau72 -= 4 * np.einsum("ji->ij", tau68, optimize=True)

    tau68 = None

    tau78 -= np.einsum("il,jk->ijkl", tau20, tau20, optimize=True)

    tau20 = None

    tau22 = zeros((M, M, M, M))

    tau22 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau23 = zeros((M, M, M, M))

    tau23 += np.einsum("badc->abcd", tau22, optimize=True)

    tau48 = zeros((M, M, M, M))

    tau48 -= np.einsum("cedf,aefb->abcd", tau22, tau4, optimize=True)

    tau4 = None

    tau22 = None

    tau50 += np.einsum("acbd->abcd", tau48, optimize=True)

    tau48 = None

    tau51 = zeros((N, N, M, M))

    tau51 += np.einsum("dcij,acdb->ijab", l2, tau50, optimize=True)

    tau50 = None

    tau60 += 2 * np.einsum("jiab->ijab", tau51, optimize=True)

    tau51 = None

    tau23 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("ijcd,cadb->ijab", tau15, tau23, optimize=True)

    tau23 = None

    tau32 += 2 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau24 = None

    tau25 = zeros((N, N, N, N))

    tau25 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 -= np.einsum("jilk,lkab->ijab", tau25, u[o, o, v, v], optimize=True)

    tau28 += np.einsum("ijba->ijab", tau26, optimize=True)

    tau29 = zeros((N, N, M, M))

    tau29 += np.einsum("cbkj,ikca->ijab", t2, tau28, optimize=True)

    tau28 = None

    tau32 -= np.einsum("ijba->ijab", tau29, optimize=True)

    tau29 = None

    tau56 += np.einsum("ijba->ijab", tau26, optimize=True)

    tau26 = None

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("kilj,klab->ijab", tau25, tau6, optimize=True)

    tau6 = None

    tau32 += 2 * np.einsum("ijab->ijab", tau31, optimize=True)

    tau31 = None

    tau33 = zeros((N, N, M, M))

    tau33 += np.einsum("cbkj,ikca->ijab", l2, tau32, optimize=True)

    tau32 = None

    tau45 += np.einsum("ijba->ijab", tau33, optimize=True)

    tau33 = None

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum("klab,iljk->ijab", tau15, tau25, optimize=True)

    tau15 = None

    tau43 -= 2 * np.einsum("ijab->ijab", tau36, optimize=True)

    tau36 = None

    tau37 = zeros((N, N, M, M))

    tau37 -= np.einsum("ablk,lkji->ijab", t2, tau25, optimize=True)

    tau41 += np.einsum("ijba->ijab", tau37, optimize=True)

    tau37 = None

    tau53 = zeros((N, N, M, M))

    tau53 += np.einsum("ijkl,lkab->ijab", tau25, tau52, optimize=True)

    tau52 = None

    tau60 += np.einsum("jiab->ijab", tau53, optimize=True)

    tau53 = None

    tau65 = zeros((N, N, N, N))

    tau65 += np.einsum("nkml,minj->ijkl", tau11, tau25, optimize=True)

    tau11 = None

    tau66 += 2 * np.einsum("ijkl->ijkl", tau65, optimize=True)

    tau65 = None

    tau67 = zeros((N, N, M, M))

    tau67 += np.einsum("ablk,ikjl->ijab", l2, tau66, optimize=True)

    tau66 = None

    tau74 += np.einsum("ijba->ijab", tau67, optimize=True)

    tau67 = None

    tau78 += np.einsum("inlm,jmkn->ijkl", tau25, tau25, optimize=True)

    tau25 = None

    tau41 -= 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum("cbkj,kica->ijab", l2, tau41, optimize=True)

    tau41 = None

    tau43 -= np.einsum("jiba->ijab", tau42, optimize=True)

    tau42 = None

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("ikac,kjcb->ijab", tau43, u[o, o, v, v], optimize=True)

    tau43 = None

    tau45 += np.einsum("ijab->ijab", tau44, optimize=True)

    tau44 = None

    r2 += np.einsum("ijab->abij", tau45, optimize=True) / 4

    r2 -= np.einsum("ijba->abij", tau45, optimize=True) / 4

    r2 -= np.einsum("jiab->abij", tau45, optimize=True) / 4

    r2 += np.einsum("jiba->abij", tau45, optimize=True) / 4

    tau45 = None

    tau56 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau57 = zeros((M, M))

    tau57 += np.einsum("cbij,ijca->ab", t2, tau56, optimize=True)

    tau56 = None

    tau58 -= np.einsum("ba->ab", tau57, optimize=True)

    tau57 = None

    tau58 -= 8 * np.einsum("ab->ab", f[v, v], optimize=True)

    tau59 = zeros((N, N, M, M))

    tau59 += np.einsum("ca,cbij->ijab", tau58, l2, optimize=True)

    tau58 = None

    tau60 -= np.einsum("jiba->ijab", tau59, optimize=True)

    tau59 = None

    r2 += np.einsum("ijab->abij", tau60, optimize=True) / 8

    r2 -= np.einsum("ijba->abij", tau60, optimize=True) / 8

    tau60 = None

    tau70 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau71 = zeros((N, N))

    tau71 += np.einsum("abkj,kiab->ij", t2, tau70, optimize=True)

    tau70 = None

    tau72 -= np.einsum("ji->ij", tau71, optimize=True)

    tau71 = None

    tau73 = zeros((N, N, M, M))

    tau73 += np.einsum("ki,abkj->ijab", tau72, l2, optimize=True)

    tau72 = None

    tau74 += np.einsum("jiba->ijab", tau73, optimize=True)

    tau73 = None

    r2 += np.einsum("ijba->abij", tau74, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau74, optimize=True) / 8

    tau74 = None

    tau75 = zeros((N, N, M, M))

    tau75 += np.einsum("jk,abik->ijab", f[o, o], l2, optimize=True)

    r2 += np.einsum("ijba->abij", tau75, optimize=True)

    r2 -= np.einsum("jiba->abij", tau75, optimize=True)

    tau75 = None

    tau77 -= np.einsum("baji->ijab", t2, optimize=True)

    tau78 += np.einsum("abji,klab->ijkl", l2, tau77, optimize=True)

    tau77 = None

    r2 += np.einsum("jikl,klba->abij", tau78, u[o, o, v, v], optimize=True) / 4

    tau78 = None

    tau79 -= np.einsum("baji->ijab", t2, optimize=True)

    tau80 = zeros((N, N, N, N))

    tau80 += np.einsum("klab,jiab->ijkl", tau79, u[o, o, v, v], optimize=True)

    tau79 = None

    tau80 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 += np.einsum("bakl,jikl->abij", l2, tau80, optimize=True) / 4

    tau80 = None

    r2 += np.einsum("jiba->abij", u[o, o, v, v], optimize=True)

    return r2
