import numpy as np


def lambda_amplitudes_intermediates_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)
    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum(
        "dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True
    )

    tau29 = np.zeros((N, N, M, M))

    tau29 -= 2 * np.einsum(
        "jiba->ijab", tau0, optimize=True
    )

    tau72 = np.zeros((N, N, M, M))

    tau72 -= 2 * np.einsum(
        "jiba->ijab", tau0, optimize=True
    )

    r2 = np.zeros((M, M, N, N))

    r2 += np.einsum(
        "jiba->abij", tau0, optimize=True
    ) / 2

    tau0 = None

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum(
        "jk,abik->ijab", f[o, o], l2, optimize=True
    )

    r2 += np.einsum(
        "ijba->abij", tau1, optimize=True
    )

    r2 -= np.einsum(
        "jiba->abij", tau1, optimize=True
    )

    tau1 = None

    tau2 = np.zeros((N, N, M, M))

    tau2 += np.einsum(
        "acik,jckb->ijab", l2, u[o, v, o, v], optimize=True
    )

    tau46 = np.zeros((N, N, M, M))

    tau46 -= 4 * np.einsum(
        "ijab->ijab", tau2, optimize=True
    )

    tau2 = None

    tau3 = np.zeros((M, M))

    tau3 += np.einsum(
        "caji,cbji->ab", l2, t2, optimize=True
    )

    tau4 = np.zeros((N, N, M, M))

    tau4 -= np.einsum(
        "ac,ibjc->ijab", tau3, u[o, v, o, v], optimize=True
    )

    tau19 = np.zeros((N, N, M, M))

    tau19 -= 2 * np.einsum(
        "jiab->ijab", tau4, optimize=True
    )

    tau4 = None

    tau9 = np.zeros((N, N, M, M))

    tau9 -= np.einsum(
        "ac,ijbc->ijab", tau3, u[o, o, v, v], optimize=True
    )

    tau14 = np.zeros((N, N, M, M))

    tau14 -= 2 * np.einsum(
        "ijab->ijab", tau9, optimize=True
    )

    tau56 = np.zeros((N, N, M, M))

    tau56 -= 4 * np.einsum(
        "ijba->ijab", tau9, optimize=True
    )

    tau68 = np.zeros((N, N, M, M))

    tau68 -= np.einsum(
        "ijab->ijab", tau9, optimize=True
    )

    tau72 += 2 * np.einsum(
        "ijab->ijab", tau9, optimize=True
    )

    tau76 = np.zeros((N, N, M, M))

    tau76 -= 4 * np.einsum(
        "ijab->ijab", tau9, optimize=True
    )

    tau9 = None

    tau41 = np.zeros((N, N, M, M))

    tau41 -= np.einsum(
        "cb,acji->ijab", tau3, t2, optimize=True
    )

    tau42 = np.zeros((N, N, M, M))

    tau42 += 2 * np.einsum(
        "ijab->ijab", tau41, optimize=True
    )

    tau80 = np.zeros((N, N, M, M))

    tau80 += np.einsum(
        "ijab->ijab", tau41, optimize=True
    )

    tau41 = None

    tau55 = np.zeros((N, N))

    tau55 += np.einsum(
        "ab,iajb->ij", tau3, u[o, v, o, v], optimize=True
    )

    tau58 = np.zeros((N, N))

    tau58 += 4 * np.einsum(
        "ji->ij", tau55, optimize=True
    )

    tau55 = None

    tau63 = np.zeros((M, M, M, M))

    tau63 += np.einsum(
        "ae,cbde->abcd", tau3, u[v, v, v, v], optimize=True
    )

    tau66 = np.zeros((M, M, M, M))

    tau66 += np.einsum(
        "acbd->abcd", tau63, optimize=True
    )

    tau63 = None

    tau71 = np.zeros((M, M))

    tau71 -= np.einsum(
        "cd,cabd->ab", tau3, u[v, v, v, v], optimize=True
    )

    tau74 = np.zeros((M, M))

    tau74 -= 4 * np.einsum(
        "ab->ab", tau71, optimize=True
    )

    tau71 = None

    tau77 = np.zeros((M, M, M, M))

    tau77 += np.einsum(
        "ac,bd->abcd", tau3, tau3, optimize=True
    )

    tau5 = np.zeros((M, M, M, M))

    tau5 += np.einsum(
        "abji,cdji->abcd", l2, t2, optimize=True
    )

    tau62 = np.zeros((M, M, M, M))

    tau62 -= np.einsum(
        "aefb,cedf->abcd", tau5, u[v, v, v, v], optimize=True
    )

    tau66 += 2 * np.einsum(
        "acbd->abcd", tau62, optimize=True
    )

    tau62 = None

    tau77 += np.einsum(
        "afde,becf->abcd", tau5, tau5, optimize=True
    )

    tau6 = np.zeros((N, N, M, M))

    tau6 -= np.einsum(
        "caki,jkcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau7 = np.zeros((N, N, M, M))

    tau7 += np.einsum(
        "ijab->ijab", tau6, optimize=True
    )

    tau7 -= np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum(
        "cadb,ijcd->ijab", tau5, tau7, optimize=True
    )

    tau19 -= 2 * np.einsum(
        "ijab->ijab", tau8, optimize=True
    )

    tau8 = None

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum(
        "cbkj,kica->ijab", l2, tau7, optimize=True
    )

    tau14 += 4 * np.einsum(
        "jiba->ijab", tau10, optimize=True
    )

    tau29 -= 4 * np.einsum(
        "jiba->ijab", tau10, optimize=True
    )

    tau48 = np.zeros((N, N, M, M))

    tau48 += 4 * np.einsum(
        "jiba->ijab", tau10, optimize=True
    )

    tau56 -= 8 * np.einsum(
        "jiba->ijab", tau10, optimize=True
    )

    tau68 += 4 * np.einsum(
        "jiba->ijab", tau10, optimize=True
    )

    tau72 -= 8 * np.einsum(
        "jiba->ijab", tau10, optimize=True
    )

    tau10 = None

    tau11 = np.zeros((N, N, N, N))

    tau11 += np.einsum(
        "baij,klba->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau12 = np.zeros((N, N, N, N))

    tau12 += np.einsum(
        "lkji->ijkl", tau11, optimize=True
    )

    tau11 = None

    tau12 += 2 * np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum(
        "abkl,ijkl->ijab", l2, tau12, optimize=True
    )

    tau14 += np.einsum(
        "jiba->ijab", tau13, optimize=True
    )

    tau15 = np.zeros((N, N, M, M))

    tau15 += np.einsum(
        "cbkj,kiac->ijab", t2, tau14, optimize=True
    )

    tau14 = None

    tau19 -= np.einsum(
        "jiab->ijab", tau15, optimize=True
    )

    tau15 = None

    tau56 -= np.einsum(
        "jiba->ijab", tau13, optimize=True
    )

    tau13 = None

    tau16 = np.zeros((N, N, M, M))

    tau16 -= np.einsum(
        "acki,cbkj->ijab", l2, t2, optimize=True
    )

    tau17 = np.zeros((N, N, M, M))

    tau17 += np.einsum(
        "kjbc,kiac->ijab", tau16, tau7, optimize=True
    )

    tau19 += 4 * np.einsum(
        "jiba->ijab", tau17, optimize=True
    )

    tau17 = None

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum(
        "kilj,lkab->ijab", tau12, tau16, optimize=True
    )

    tau19 -= 2 * np.einsum(
        "jiab->ijab", tau18, optimize=True
    )

    tau18 = None

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum(
        "cbkj,kiac->ijab", l2, tau19, optimize=True
    )

    tau19 = None

    tau46 -= np.einsum(
        "jiab->ijab", tau20, optimize=True
    )

    tau20 = None

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum(
        "jkcb,ikca->ijab", tau16, tau7, optimize=True
    )

    tau33 = np.zeros((N, N, M, M))

    tau33 -= 4 * np.einsum(
        "jiba->ijab", tau31, optimize=True
    )

    tau31 = None

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum(
        "ikcb,kjac->ijab", tau16, tau16, optimize=True
    )

    tau44 = np.zeros((N, N, M, M))

    tau44 += 4 * np.einsum(
        "ijab->ijab", tau35, optimize=True
    )

    tau35 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum(
        "ijdc,acbd->ijab", tau16, tau5, optimize=True
    )

    tau44 -= 2 * np.einsum(
        "ijab->ijab", tau36, optimize=True
    )

    tau36 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum(
        "bcjk,kica->ijab", t2, tau16, optimize=True
    )

    tau42 += 4 * np.einsum(
        "ijba->ijab", tau39, optimize=True
    )

    tau78 = np.zeros((N, N, M, M))

    tau78 += 2 * np.einsum(
        "ijba->ijab", tau39, optimize=True
    )

    tau80 -= 2 * np.einsum(
        "ijab->ijab", tau39, optimize=True
    )

    tau39 = None

    tau50 = np.zeros((N, N, N, N))

    tau50 += np.einsum(
        "klab,ijab->ijkl", tau16, tau7, optimize=True
    )

    tau52 = np.zeros((N, N, N, N))

    tau52 -= 8 * np.einsum(
        "jkil->ijkl", tau50, optimize=True
    )

    tau50 = None

    tau61 = np.zeros((M, M, M, M))

    tau61 += np.einsum(
        "ijab,jcid->abcd", tau16, u[o, v, o, v], optimize=True
    )

    tau66 += 4 * np.einsum(
        "abcd->abcd", tau61, optimize=True
    )

    tau61 = None

    tau65 = np.zeros((M, M, M, M))

    tau65 += np.einsum(
        "ijab,ijcd->abcd", tau16, tau6, optimize=True
    )

    tau6 = None

    tau66 += 4 * np.einsum(
        "acbd->abcd", tau65, optimize=True
    )

    tau65 = None

    tau77 -= 4 * np.einsum(
        "ijac,jibd->abcd", tau16, tau16, optimize=True
    )

    r2 -= np.einsum(
        "abcd,jicd->abij", tau77, u[o, o, v, v], optimize=True
    ) / 4

    tau77 = None

    tau79 = np.zeros((N, N, N, N))

    tau79 += 4 * np.einsum(
        "ilba,jkab->ijkl", tau16, tau16, optimize=True
    )

    tau21 = np.zeros((N, N))

    tau21 += np.einsum(
        "baki,bakj->ij", l2, t2, optimize=True
    )

    tau22 = np.zeros((N, N, M, M))

    tau22 -= np.einsum(
        "ik,kajb->ijab", tau21, u[o, v, o, v], optimize=True
    )

    tau33 += 2 * np.einsum(
        "ijab->ijab", tau22, optimize=True
    )

    tau22 = None

    tau28 = np.zeros((N, N, M, M))

    tau28 -= np.einsum(
        "ik,jkab->ijab", tau21, u[o, o, v, v], optimize=True
    )

    tau29 -= 2 * np.einsum(
        "ijba->ijab", tau28, optimize=True
    )

    tau48 += np.einsum(
        "ijba->ijab", tau28, optimize=True
    )

    tau49 = np.zeros((N, N, N, N))

    tau49 += np.einsum(
        "bakl,ijab->ijkl", t2, tau48, optimize=True
    )

    tau48 = None

    tau52 -= np.einsum(
        "iklj->ijkl", tau49, optimize=True
    )

    tau49 = None

    tau56 -= 2 * np.einsum(
        "ijba->ijab", tau28, optimize=True
    )

    tau60 = np.zeros((N, N, M, M))

    tau60 -= 4 * np.einsum(
        "ijba->ijab", tau28, optimize=True
    )

    tau72 += 4 * np.einsum(
        "jiba->ijab", tau28, optimize=True
    )

    tau28 = None

    tau40 = np.zeros((N, N, M, M))

    tau40 += np.einsum(
        "kj,abik->ijab", tau21, t2, optimize=True
    )

    tau42 -= 2 * np.einsum(
        "ijba->ijab", tau40, optimize=True
    )

    tau78 -= np.einsum(
        "ijba->ijab", tau40, optimize=True
    )

    tau40 = None

    tau44 += np.einsum(
        "ij,ab->ijab", tau21, tau3, optimize=True
    )

    tau3 = None

    tau47 = np.zeros((N, N, N, N))

    tau47 += np.einsum(
        "im,jmlk->ijkl", tau21, u[o, o, o, o], optimize=True
    )

    tau52 -= 2 * np.einsum(
        "iklj->ijkl", tau47, optimize=True
    )

    tau47 = None

    tau54 = np.zeros((N, N))

    tau54 -= np.einsum(
        "kl,ilkj->ij", tau21, u[o, o, o, o], optimize=True
    )

    tau58 -= 4 * np.einsum(
        "ji->ij", tau54, optimize=True
    )

    tau54 = None

    tau70 = np.zeros((M, M))

    tau70 += np.einsum(
        "ij,jaib->ab", tau21, u[o, v, o, v], optimize=True
    )

    tau74 += 4 * np.einsum(
        "ab->ab", tau70, optimize=True
    )

    tau70 = None

    tau79 -= np.einsum(
        "il,jk->ijkl", tau21, tau21, optimize=True
    )

    tau21 = None

    tau23 = np.zeros((M, M, M, M))

    tau23 += np.einsum(
        "abji,jicd->abcd", t2, u[o, o, v, v], optimize=True
    )

    tau24 = np.zeros((M, M, M, M))

    tau24 += np.einsum(
        "badc->abcd", tau23, optimize=True
    )

    tau64 = np.zeros((M, M, M, M))

    tau64 -= np.einsum(
        "cedf,aefb->abcd", tau23, tau5, optimize=True
    )

    tau5 = None

    tau23 = None

    tau66 -= np.einsum(
        "abcd->abcd", tau64, optimize=True
    )

    tau64 = None

    tau67 = np.zeros((N, N, M, M))

    tau67 += np.einsum(
        "dcij,acdb->ijab", l2, tau66, optimize=True
    )

    tau66 = None

    tau76 += 2 * np.einsum(
        "jiab->ijab", tau67, optimize=True
    )

    tau67 = None

    tau24 += 2 * np.einsum(
        "badc->abcd", u[v, v, v, v], optimize=True
    )

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum(
        "ijcd,cadb->ijab", tau16, tau24, optimize=True
    )

    tau24 = None

    tau33 += 2 * np.einsum(
        "ijab->ijab", tau25, optimize=True
    )

    tau25 = None

    tau26 = np.zeros((N, N, N, N))

    tau26 += np.einsum(
        "baij,bakl->ijkl", l2, t2, optimize=True
    )

    tau27 = np.zeros((N, N, M, M))

    tau27 -= np.einsum(
        "jilk,lkab->ijab", tau26, u[o, o, v, v], optimize=True
    )

    tau29 += np.einsum(
        "ijba->ijab", tau27, optimize=True
    )

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum(
        "cbkj,ikca->ijab", t2, tau29, optimize=True
    )

    tau29 = None

    tau33 -= np.einsum(
        "ijba->ijab", tau30, optimize=True
    )

    tau30 = None

    tau72 += np.einsum(
        "ijba->ijab", tau27, optimize=True
    )

    tau27 = None

    tau32 = np.zeros((N, N, M, M))

    tau32 += np.einsum(
        "kilj,klab->ijab", tau26, tau7, optimize=True
    )

    tau7 = None

    tau33 += 2 * np.einsum(
        "ijab->ijab", tau32, optimize=True
    )

    tau32 = None

    tau34 = np.zeros((N, N, M, M))

    tau34 += np.einsum(
        "cbkj,ikca->ijab", l2, tau33, optimize=True
    )

    tau33 = None

    tau46 += np.einsum(
        "ijba->ijab", tau34, optimize=True
    )

    tau34 = None

    tau37 = np.zeros((N, N, M, M))

    tau37 += np.einsum(
        "klab,iljk->ijab", tau16, tau26, optimize=True
    )

    tau16 = None

    tau44 -= 2 * np.einsum(
        "ijab->ijab", tau37, optimize=True
    )

    tau37 = None

    tau38 = np.zeros((N, N, M, M))

    tau38 -= np.einsum(
        "ablk,lkji->ijab", t2, tau26, optimize=True
    )

    tau42 += np.einsum(
        "ijba->ijab", tau38, optimize=True
    )

    tau38 = None

    tau51 = np.zeros((N, N, N, N))

    tau51 += np.einsum(
        "nkml,minj->ijkl", tau12, tau26, optimize=True
    )

    tau12 = None

    tau52 += 2 * np.einsum(
        "ijkl->ijkl", tau51, optimize=True
    )

    tau51 = None

    tau53 = np.zeros((N, N, M, M))

    tau53 += np.einsum(
        "ablk,ikjl->ijab", l2, tau52, optimize=True
    )

    tau52 = None

    tau60 += np.einsum(
        "ijba->ijab", tau53, optimize=True
    )

    tau53 = None

    tau69 = np.zeros((N, N, M, M))

    tau69 += np.einsum(
        "ijkl,lkab->ijab", tau26, tau68, optimize=True
    )

    tau68 = None

    tau76 += np.einsum(
        "jiab->ijab", tau69, optimize=True
    )

    tau69 = None

    tau79 += np.einsum(
        "inlm,jmkn->ijkl", tau26, tau26, optimize=True
    )

    tau26 = None

    tau42 -= 4 * np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau43 = np.zeros((N, N, M, M))

    tau43 += np.einsum(
        "cbkj,kica->ijab", l2, tau42, optimize=True
    )

    tau42 = None

    tau44 -= np.einsum(
        "jiba->ijab", tau43, optimize=True
    )

    tau43 = None

    tau45 = np.zeros((N, N, M, M))

    tau45 += np.einsum(
        "jkbc,kica->ijab", tau44, u[o, o, v, v], optimize=True
    )

    tau44 = None

    tau46 += np.einsum(
        "jiba->ijab", tau45, optimize=True
    )

    tau45 = None

    r2 += np.einsum(
        "ijab->abij", tau46, optimize=True
    ) / 4

    r2 -= np.einsum(
        "ijba->abij", tau46, optimize=True
    ) / 4

    r2 -= np.einsum(
        "jiab->abij", tau46, optimize=True
    ) / 4

    r2 += np.einsum(
        "jiba->abij", tau46, optimize=True
    ) / 4

    tau46 = None

    tau56 -= 4 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau57 = np.zeros((N, N))

    tau57 += np.einsum(
        "abkj,kiab->ij", t2, tau56, optimize=True
    )

    tau56 = None

    tau58 -= np.einsum(
        "ji->ij", tau57, optimize=True
    )

    tau57 = None

    tau59 = np.zeros((N, N, M, M))

    tau59 += np.einsum(
        "ki,abkj->ijab", tau58, l2, optimize=True
    )

    tau58 = None

    tau60 += np.einsum(
        "jiba->ijab", tau59, optimize=True
    )

    tau59 = None

    r2 += np.einsum(
        "ijba->abij", tau60, optimize=True
    ) / 8

    r2 -= np.einsum(
        "jiba->abij", tau60, optimize=True
    ) / 8

    tau60 = None

    tau72 -= 4 * np.einsum(
        "jiba->ijab", u[o, o, v, v], optimize=True
    )

    tau73 = np.zeros((M, M))

    tau73 += np.einsum(
        "cbij,ijca->ab", t2, tau72, optimize=True
    )

    tau72 = None

    tau74 -= np.einsum(
        "ba->ab", tau73, optimize=True
    )

    tau73 = None

    tau74 -= 8 * np.einsum(
        "ab->ab", f[v, v], optimize=True
    )

    tau75 = np.zeros((N, N, M, M))

    tau75 += np.einsum(
        "ca,cbij->ijab", tau74, l2, optimize=True
    )

    tau74 = None

    tau76 -= np.einsum(
        "jiba->ijab", tau75, optimize=True
    )

    tau75 = None

    r2 += np.einsum(
        "ijab->abij", tau76, optimize=True
    ) / 8

    r2 -= np.einsum(
        "ijba->abij", tau76, optimize=True
    ) / 8

    tau76 = None

    tau78 -= np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau79 += np.einsum(
        "abji,klab->ijkl", l2, tau78, optimize=True
    )

    tau78 = None

    r2 += np.einsum(
        "jikl,klba->abij", tau79, u[o, o, v, v], optimize=True
    ) / 4

    tau79 = None

    tau80 -= np.einsum(
        "baji->ijab", t2, optimize=True
    )

    tau81 = np.zeros((N, N, N, N))

    tau81 += np.einsum(
        "klab,jiab->ijkl", tau80, u[o, o, v, v], optimize=True
    )

    tau80 = None

    tau81 += 2 * np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    r2 += np.einsum(
        "bakl,jikl->abij", l2, tau81, optimize=True
    ) / 4

    tau81 = None

    r2 += np.einsum(
        "jiba->abij", u[o, o, v, v], optimize=True
    )
    
    return r2
