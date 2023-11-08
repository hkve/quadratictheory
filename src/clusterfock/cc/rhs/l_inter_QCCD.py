import numpy as np


def lambda_amplitudes_intermediates_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau23 = np.zeros((N, N, M, M))

    tau23 -= 2 * np.einsum("jiba->ijab", tau0, optimize=True)

    tau43 = np.zeros((N, N, M, M))

    tau43 -= 2 * np.einsum("jiba->ijab", tau0, optimize=True)

    r2 = np.zeros((M, M, N, N))

    r2 += np.einsum("jiba->abij", tau0, optimize=True) / 2

    tau0 = None

    tau1 = np.zeros((M, M))

    tau1 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau2 = np.zeros((N, N, M, M))

    tau2 -= np.einsum("ac,ijbc->ijab", tau1, u[o, o, v, v], optimize=True)

    tau16 = np.zeros((N, N, M, M))

    tau16 -= np.einsum("ijab->ijab", tau2, optimize=True)

    tau23 += 2 * np.einsum("ijab->ijab", tau2, optimize=True)

    tau27 = np.zeros((N, N, M, M))

    tau27 -= 4 * np.einsum("ijab->ijab", tau2, optimize=True)

    tau34 = np.zeros((N, N, M, M))

    tau34 -= 2 * np.einsum("ijab->ijab", tau2, optimize=True)

    tau71 = np.zeros((N, N, M, M))

    tau71 -= 4 * np.einsum("ijba->ijab", tau2, optimize=True)

    tau2 = None

    tau3 = np.zeros((M, M, M, M))

    tau3 += np.einsum("ae,cbde->abcd", tau1, u[v, v, v, v], optimize=True)

    tau12 = np.zeros((M, M, M, M))

    tau12 += np.einsum("acbd->abcd", tau3, optimize=True)

    tau3 = None

    tau20 = np.zeros((M, M))

    tau20 -= np.einsum("cd,cabd->ab", tau1, u[v, v, v, v], optimize=True)

    tau25 = np.zeros((M, M))

    tau25 -= 4 * np.einsum("ab->ab", tau20, optimize=True)

    tau20 = None

    tau29 = np.zeros((N, N, M, M))

    tau29 -= np.einsum("ac,ibjc->ijab", tau1, u[o, v, o, v], optimize=True)

    tau38 = np.zeros((N, N, M, M))

    tau38 -= 2 * np.einsum("jiab->ijab", tau29, optimize=True)

    tau29 = None

    tau55 = np.zeros((N, N, M, M))

    tau55 -= np.einsum("cb,acji->ijab", tau1, t2, optimize=True)

    tau56 = np.zeros((N, N, M, M))

    tau56 += 2 * np.einsum("ijab->ijab", tau55, optimize=True)

    tau80 = np.zeros((N, N, M, M))

    tau80 += np.einsum("ijab->ijab", tau55, optimize=True)

    tau55 = None

    tau70 = np.zeros((N, N))

    tau70 += np.einsum("ab,iajb->ij", tau1, u[o, v, o, v], optimize=True)

    tau73 = np.zeros((N, N))

    tau73 += 4 * np.einsum("ji->ij", tau70, optimize=True)

    tau70 = None

    tau77 = np.zeros((M, M, M, M))

    tau77 += np.einsum("ac,bd->abcd", tau1, tau1, optimize=True)

    tau4 = np.zeros((M, M, M, M))

    tau4 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau5 = np.zeros((M, M, M, M))

    tau5 -= np.einsum("aefb,cedf->abcd", tau4, u[v, v, v, v], optimize=True)

    tau12 -= 2 * np.einsum("abcd->abcd", tau5, optimize=True)

    tau5 = None

    tau77 += np.einsum("afde,becf->abcd", tau4, tau4, optimize=True)

    tau6 = np.zeros((M, M, M, M))

    tau6 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau7 = np.zeros((M, M, M, M))

    tau7 -= np.einsum("aefb,cedf->abcd", tau4, tau6, optimize=True)

    tau12 += np.einsum("acbd->abcd", tau7, optimize=True)

    tau7 = None

    tau41 = np.zeros((M, M, M, M))

    tau41 += np.einsum("badc->abcd", tau6, optimize=True)

    tau6 = None

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("caki,jkbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum("ijab->ijab", tau8, optimize=True)

    tau8 = None

    tau9 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau15 = np.zeros((N, N, M, M))

    tau15 += np.einsum("cbkj,kica->ijab", l2, tau9, optimize=True)

    tau16 += 4 * np.einsum("jiba->ijab", tau15, optimize=True)

    tau23 -= 8 * np.einsum("jiba->ijab", tau15, optimize=True)

    tau34 += 4 * np.einsum("jiba->ijab", tau15, optimize=True)

    tau43 -= 4 * np.einsum("jiba->ijab", tau15, optimize=True)

    tau64 = np.zeros((N, N, M, M))

    tau64 += 4 * np.einsum("jiba->ijab", tau15, optimize=True)

    tau71 -= 8 * np.einsum("jiba->ijab", tau15, optimize=True)

    tau15 = None

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("cadb,ijcd->ijab", tau4, tau9, optimize=True)

    tau38 -= 2 * np.einsum("ijab->ijab", tau30, optimize=True)

    tau30 = None

    tau10 = np.zeros((N, N, M, M))

    tau10 -= np.einsum("caik,cbkj->ijab", l2, t2, optimize=True)

    tau11 = np.zeros((M, M, M, M))

    tau11 += np.einsum("ijcd,ijab->abcd", tau10, tau9, optimize=True)

    tau12 -= 4 * np.einsum("cdab->abcd", tau11, optimize=True)

    tau11 = None

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum("dcij,acdb->ijab", l2, tau12, optimize=True)

    tau12 = None

    tau27 += 2 * np.einsum("jiab->ijab", tau13, optimize=True)

    tau13 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum("kjbc,kiac->ijab", tau10, tau9, optimize=True)

    tau38 += 4 * np.einsum("jiba->ijab", tau36, optimize=True)

    tau36 = None

    tau45 = np.zeros((N, N, M, M))

    tau45 += np.einsum("jkcb,ikca->ijab", tau10, tau9, optimize=True)

    tau47 = np.zeros((N, N, M, M))

    tau47 -= 4 * np.einsum("jiba->ijab", tau45, optimize=True)

    tau45 = None

    tau49 = np.zeros((N, N, M, M))

    tau49 += np.einsum("ikcb,kjac->ijab", tau10, tau10, optimize=True)

    tau58 = np.zeros((N, N, M, M))

    tau58 += 4 * np.einsum("ijab->ijab", tau49, optimize=True)

    tau49 = None

    tau50 = np.zeros((N, N, M, M))

    tau50 += np.einsum("ijdc,acbd->ijab", tau10, tau4, optimize=True)

    tau4 = None

    tau58 -= 2 * np.einsum("ijab->ijab", tau50, optimize=True)

    tau50 = None

    tau53 = np.zeros((N, N, M, M))

    tau53 += np.einsum("caki,kjcb->ijab", t2, tau10, optimize=True)

    tau56 += 4 * np.einsum("ijba->ijab", tau53, optimize=True)

    tau78 = np.zeros((N, N, M, M))

    tau78 += 2 * np.einsum("ijba->ijab", tau53, optimize=True)

    tau80 -= 2 * np.einsum("ijab->ijab", tau53, optimize=True)

    tau53 = None

    tau66 = np.zeros((N, N, N, N))

    tau66 += np.einsum("klab,ijab->ijkl", tau10, tau9, optimize=True)

    tau67 = np.zeros((N, N, N, N))

    tau67 += 8 * np.einsum("lkij->ijkl", tau66, optimize=True)

    tau66 = None

    tau77 -= 4 * np.einsum("ijac,jibd->abcd", tau10, tau10, optimize=True)

    r2 -= np.einsum("abcd,jicd->abij", tau77, u[o, o, v, v], optimize=True) / 4

    tau77 = None

    tau79 = np.zeros((N, N, N, N))

    tau79 += 4 * np.einsum("ilba,jkab->ijkl", tau10, tau10, optimize=True)

    tau14 = np.zeros((N, N, N, N))

    tau14 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau17 = np.zeros((N, N, M, M))

    tau17 += np.einsum("ijkl,lkab->ijab", tau14, tau16, optimize=True)

    tau16 = None

    tau27 += np.einsum("jiab->ijab", tau17, optimize=True)

    tau17 = None

    tau21 = np.zeros((N, N, M, M))

    tau21 -= np.einsum("jilk,lkab->ijab", tau14, u[o, o, v, v], optimize=True)

    tau23 += np.einsum("ijba->ijab", tau21, optimize=True)

    tau43 += np.einsum("ijba->ijab", tau21, optimize=True)

    tau21 = None

    tau46 = np.zeros((N, N, M, M))

    tau46 += np.einsum("kilj,klab->ijab", tau14, tau9, optimize=True)

    tau9 = None

    tau47 += 2 * np.einsum("ijab->ijab", tau46, optimize=True)

    tau46 = None

    tau51 = np.zeros((N, N, M, M))

    tau51 += np.einsum("klab,iljk->ijab", tau10, tau14, optimize=True)

    tau58 -= 2 * np.einsum("ijab->ijab", tau51, optimize=True)

    tau51 = None

    tau52 = np.zeros((N, N, M, M))

    tau52 -= np.einsum("ablk,lkji->ijab", t2, tau14, optimize=True)

    tau56 += np.einsum("ijba->ijab", tau52, optimize=True)

    tau52 = None

    tau61 = np.zeros((N, N, N, N))

    tau61 += np.einsum("imnj,knml->ijkl", tau14, u[o, o, o, o], optimize=True)

    tau67 += 4 * np.einsum("ijkl->ijkl", tau61, optimize=True)

    tau61 = None

    tau79 += np.einsum("inlm,jmkn->ijkl", tau14, tau14, optimize=True)

    tau18 = np.zeros((N, N))

    tau18 -= np.einsum("baik,bakj->ij", l2, t2, optimize=True)

    tau19 = np.zeros((M, M))

    tau19 += np.einsum("ij,jaib->ab", tau18, u[o, v, o, v], optimize=True)

    tau25 += 4 * np.einsum("ab->ab", tau19, optimize=True)

    tau19 = None

    tau22 = np.zeros((N, N, M, M))

    tau22 -= np.einsum("ik,jkab->ijab", tau18, u[o, o, v, v], optimize=True)

    tau23 += 4 * np.einsum("jiba->ijab", tau22, optimize=True)

    tau43 -= 2 * np.einsum("ijba->ijab", tau22, optimize=True)

    tau44 = np.zeros((N, N, M, M))

    tau44 += np.einsum("cbkj,ikca->ijab", t2, tau43, optimize=True)

    tau43 = None

    tau47 -= np.einsum("ijba->ijab", tau44, optimize=True)

    tau44 = None

    tau64 += np.einsum("ijba->ijab", tau22, optimize=True)

    tau65 = np.zeros((N, N, N, N))

    tau65 += np.einsum("bakl,ijab->ijkl", t2, tau64, optimize=True)

    tau64 = None

    tau67 -= np.einsum("iklj->ijkl", tau65, optimize=True)

    tau65 = None

    tau71 -= 2 * np.einsum("ijba->ijab", tau22, optimize=True)

    tau75 = np.zeros((N, N, M, M))

    tau75 -= 4 * np.einsum("ijba->ijab", tau22, optimize=True)

    tau22 = None

    tau40 = np.zeros((N, N, M, M))

    tau40 -= np.einsum("ik,kajb->ijab", tau18, u[o, v, o, v], optimize=True)

    tau47 += 2 * np.einsum("ijab->ijab", tau40, optimize=True)

    tau40 = None

    tau54 = np.zeros((N, N, M, M))

    tau54 += np.einsum("kj,abik->ijab", tau18, t2, optimize=True)

    tau56 -= 2 * np.einsum("ijba->ijab", tau54, optimize=True)

    tau78 -= np.einsum("ijba->ijab", tau54, optimize=True)

    tau54 = None

    tau58 += np.einsum("ab,ij->ijab", tau1, tau18, optimize=True)

    tau1 = None

    tau62 = np.zeros((N, N, N, N))

    tau62 += np.einsum("im,jmlk->ijkl", tau18, u[o, o, o, o], optimize=True)

    tau67 -= 2 * np.einsum("iklj->ijkl", tau62, optimize=True)

    tau62 = None

    tau69 = np.zeros((N, N))

    tau69 -= np.einsum("kl,ilkj->ij", tau18, u[o, o, o, o], optimize=True)

    tau73 -= 4 * np.einsum("ji->ij", tau69, optimize=True)

    tau69 = None

    tau79 -= np.einsum("il,jk->ijkl", tau18, tau18, optimize=True)

    tau18 = None

    tau23 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau24 = np.zeros((M, M))

    tau24 += np.einsum("cbij,ijca->ab", t2, tau23, optimize=True)

    tau23 = None

    tau25 -= np.einsum("ba->ab", tau24, optimize=True)

    tau24 = None

    tau25 -= 8 * np.einsum("ab->ab", f[v, v], optimize=True)

    tau26 = np.zeros((N, N, M, M))

    tau26 += np.einsum("ca,cbij->ijab", tau25, l2, optimize=True)

    tau25 = None

    tau27 -= np.einsum("jiba->ijab", tau26, optimize=True)

    tau26 = None

    r2 += np.einsum("ijab->abij", tau27, optimize=True) / 8

    r2 -= np.einsum("ijba->abij", tau27, optimize=True) / 8

    tau27 = None

    tau28 = np.zeros((N, N, M, M))

    tau28 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau60 = np.zeros((N, N, M, M))

    tau60 -= 4 * np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau31 = np.zeros((N, N, N, N))

    tau31 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau32 = np.zeros((N, N, N, N))

    tau32 += np.einsum("lkji->ijkl", tau31, optimize=True)

    tau63 = np.zeros((N, N, N, N))

    tau63 += np.einsum("imnk,mjln->ijkl", tau14, tau31, optimize=True)

    tau14 = None

    tau31 = None

    tau67 -= 2 * np.einsum("ijlk->ijkl", tau63, optimize=True)

    tau63 = None

    tau68 = np.zeros((N, N, M, M))

    tau68 += np.einsum("abkl,ikjl->ijab", l2, tau67, optimize=True)

    tau67 = None

    tau75 -= np.einsum("ijba->ijab", tau68, optimize=True)

    tau68 = None

    tau32 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("abkl,ijkl->ijab", l2, tau32, optimize=True)

    tau34 += np.einsum("jiba->ijab", tau33, optimize=True)

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("cbkj,kiac->ijab", t2, tau34, optimize=True)

    tau34 = None

    tau38 -= np.einsum("jiab->ijab", tau35, optimize=True)

    tau35 = None

    tau71 -= np.einsum("jiba->ijab", tau33, optimize=True)

    tau33 = None

    tau37 = np.zeros((N, N, M, M))

    tau37 += np.einsum("lkab,kilj->ijab", tau10, tau32, optimize=True)

    tau32 = None

    tau38 -= 2 * np.einsum("jiab->ijab", tau37, optimize=True)

    tau37 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum("cbkj,kiac->ijab", l2, tau38, optimize=True)

    tau38 = None

    tau60 -= np.einsum("jiab->ijab", tau39, optimize=True)

    tau39 = None

    tau41 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau42 = np.zeros((N, N, M, M))

    tau42 += np.einsum("ijcd,cadb->ijab", tau10, tau41, optimize=True)

    tau41 = None

    tau10 = None

    tau47 += 2 * np.einsum("ijab->ijab", tau42, optimize=True)

    tau42 = None

    tau48 = np.zeros((N, N, M, M))

    tau48 += np.einsum("cbkj,ikca->ijab", l2, tau47, optimize=True)

    tau47 = None

    tau60 += np.einsum("ijba->ijab", tau48, optimize=True)

    tau48 = None

    tau56 -= 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau57 = np.zeros((N, N, M, M))

    tau57 += np.einsum("cbkj,kica->ijab", l2, tau56, optimize=True)

    tau56 = None

    tau58 -= np.einsum("jiba->ijab", tau57, optimize=True)

    tau57 = None

    tau59 = np.zeros((N, N, M, M))

    tau59 += np.einsum("jkbc,kica->ijab", tau58, u[o, o, v, v], optimize=True)

    tau58 = None

    tau60 += np.einsum("jiba->ijab", tau59, optimize=True)

    tau59 = None

    r2 += np.einsum("ijab->abij", tau60, optimize=True) / 4

    r2 -= np.einsum("ijba->abij", tau60, optimize=True) / 4

    r2 -= np.einsum("jiab->abij", tau60, optimize=True) / 4

    r2 += np.einsum("jiba->abij", tau60, optimize=True) / 4

    tau60 = None

    tau71 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau72 = np.zeros((N, N))

    tau72 += np.einsum("abkj,kiab->ij", t2, tau71, optimize=True)

    tau71 = None

    tau73 -= np.einsum("ji->ij", tau72, optimize=True)

    tau72 = None

    tau74 = np.zeros((N, N, M, M))

    tau74 += np.einsum("ki,abkj->ijab", tau73, l2, optimize=True)

    tau73 = None

    tau75 += np.einsum("jiba->ijab", tau74, optimize=True)

    tau74 = None

    r2 += np.einsum("ijba->abij", tau75, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau75, optimize=True) / 8

    tau75 = None

    tau76 = np.zeros((N, N, M, M))

    tau76 += np.einsum("jk,abik->ijab", f[o, o], l2, optimize=True)

    r2 += np.einsum("ijba->abij", tau76, optimize=True)

    r2 -= np.einsum("jiba->abij", tau76, optimize=True)

    tau76 = None

    tau78 -= np.einsum("baji->ijab", t2, optimize=True)

    tau79 += np.einsum("abji,klab->ijkl", l2, tau78, optimize=True)

    tau78 = None

    r2 += np.einsum("jikl,klba->abij", tau79, u[o, o, v, v], optimize=True) / 4

    tau79 = None

    tau80 -= np.einsum("baji->ijab", t2, optimize=True)

    tau81 = np.zeros((N, N, N, N))

    tau81 += np.einsum("klab,jiab->ijkl", tau80, u[o, o, v, v], optimize=True)

    tau80 = None

    tau81 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 += np.einsum("bakl,jikl->abij", l2, tau81, optimize=True) / 4

    tau81 = None

    r2 += np.einsum("jiba->abij", u[o, o, v, v], optimize=True)

    return r2
