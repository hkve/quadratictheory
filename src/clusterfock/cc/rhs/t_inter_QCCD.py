import numpy as np


def amplitudes_intermediates_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum("ki,abjk->ijab", f[o, o], t2, optimize=True)

    r2 = zeros((M, M, N, N))

    r2 -= np.einsum("ijba->abij", tau0, optimize=True)

    r2 += np.einsum("jiba->abij", tau0, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, M, M))

    tau1 -= np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau22 = zeros((N, N, M, M))

    tau22 -= np.einsum("bckj,kica->ijab", t2, tau1, optimize=True)

    tau23 = zeros((N, N, M, M))

    tau23 += 4 * np.einsum("ijab->ijab", tau22, optimize=True)

    tau29 = zeros((N, N, M, M))

    tau29 += 4 * np.einsum("ijab->ijab", tau22, optimize=True)

    tau74 = zeros((N, N, M, M))

    tau74 -= 2 * np.einsum("ijab->ijab", tau22, optimize=True)

    tau76 = zeros((N, N, N, N))

    tau76 -= 2 * np.einsum("lkba,jiba->ijkl", tau22, u[o, o, v, v], optimize=True)

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("klab,iljk->ijab", tau1, u[o, o, o, o], optimize=True)

    tau34 = zeros((N, N, M, M))

    tau34 += 2 * np.einsum("ijab->ijab", tau31, optimize=True)

    tau31 = None

    tau33 = zeros((N, N, M, M))

    tau33 += np.einsum("ikca,kcjb->ijab", tau1, u[o, v, o, v], optimize=True)

    tau34 += 2 * np.einsum("ijba->ijab", tau33, optimize=True)

    tau33 = None

    tau36 = zeros((N, N, M, M))

    tau36 += np.einsum("kiac,jbkc->ijab", tau1, u[o, v, o, v], optimize=True)

    tau39 = zeros((N, N, M, M))

    tau39 += 2 * np.einsum("ijab->ijab", tau36, optimize=True)

    tau36 = None

    tau38 = zeros((N, N, M, M))

    tau38 -= np.einsum("ijcd,acdb->ijab", tau1, u[v, v, v, v], optimize=True)

    tau39 += 2 * np.einsum("jiba->ijab", tau38, optimize=True)

    tau38 = None

    tau2 = zeros((M, M, M, M))

    tau2 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau3 = zeros((N, N, M, M))

    tau3 -= np.einsum("ijcd,acdb->ijab", tau1, tau2, optimize=True)

    tau25 = zeros((N, N, M, M))

    tau25 += 2 * np.einsum("jiba->ijab", tau3, optimize=True)

    tau3 = None

    tau57 = zeros((M, M, M, M))

    tau57 += np.einsum("badc->abcd", tau2, optimize=True)

    tau2 = None

    tau4 = zeros((N, N, N, N))

    tau4 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 -= np.einsum("ablk,lkji->ijab", t2, tau4, optimize=True)

    tau23 -= np.einsum("ijba->ijab", tau21, optimize=True)

    tau24 = zeros((N, N, M, M))

    tau24 += np.einsum("jkcb,kica->ijab", tau23, u[o, o, v, v], optimize=True)

    tau23 = None

    tau25 += np.einsum("jiab->ijab", tau24, optimize=True)

    tau24 = None

    tau29 -= np.einsum("ijba->ijab", tau21, optimize=True)

    tau21 = None

    tau37 = zeros((N, N, M, M))

    tau37 -= np.einsum("kijl,lakb->ijab", tau4, u[o, v, o, v], optimize=True)

    tau39 -= np.einsum("jiba->ijab", tau37, optimize=True)

    tau37 = None

    tau40 = zeros((N, N, M, M))

    tau40 += np.einsum("cbkj,ikca->ijab", t2, tau39, optimize=True)

    tau39 = None

    tau41 = zeros((N, N, M, M))

    tau41 -= 2 * np.einsum("ijba->ijab", tau40, optimize=True)

    tau40 = None

    tau42 = zeros((N, N, N, N))

    tau42 += np.einsum("mijn,nklm->ijkl", tau4, u[o, o, o, o], optimize=True)

    tau45 = zeros((N, N, N, N))

    tau45 += 2 * np.einsum("ijkl->ijkl", tau42, optimize=True)

    tau42 = None

    tau63 = zeros((N, N, M, M))

    tau63 -= np.einsum("jilk,lkab->ijab", tau4, u[o, o, v, v], optimize=True)

    tau64 = zeros((N, N, M, M))

    tau64 += np.einsum("ijba->ijab", tau63, optimize=True)

    tau69 = zeros((N, N, M, M))

    tau69 += np.einsum("ijba->ijab", tau63, optimize=True)

    tau63 = None

    tau5 = zeros((N, N, M, M))

    tau5 -= np.einsum("caik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau6 = zeros((N, N, M, M))

    tau6 += np.einsum("ikjl,klab->ijab", tau4, tau5, optimize=True)

    tau25 += 2 * np.einsum("jiba->ijab", tau6, optimize=True)

    tau6 = None

    tau11 = zeros((N, N, M, M))

    tau11 += np.einsum("kiac,kjbc->ijab", tau1, tau5, optimize=True)

    tau25 -= 4 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau11 = None

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("ikca,jkcb->ijab", tau1, tau5, optimize=True)

    tau25 -= 4 * np.einsum("jiba->ijab", tau12, optimize=True)

    tau12 = None

    tau17 = zeros((N, N, M, M))

    tau17 += np.einsum("ijab->ijab", tau5, optimize=True)

    tau7 = zeros((N, N, N, N))

    tau7 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau8 = zeros((N, N, M, M))

    tau8 -= np.einsum("klab,ikjl->ijab", tau1, tau7, optimize=True)

    tau25 -= 2 * np.einsum("ijab->ijab", tau8, optimize=True)

    tau8 = None

    tau43 = zeros((N, N, N, N))

    tau43 -= np.einsum("imkn,jmnl->ijkl", tau4, tau7, optimize=True)

    tau4 = None

    tau45 -= np.einsum("ijlk->ijkl", tau43, optimize=True)

    tau43 = None

    tau49 = zeros((N, N, N, N))

    tau49 += np.einsum("lkji->ijkl", tau7, optimize=True)

    tau7 = None

    tau9 = zeros((M, M, M, M))

    tau9 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 += np.einsum("ijcd,cabd->ijab", tau5, tau9, optimize=True)

    tau5 = None

    tau25 -= 2 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau10 = None

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("acbd,icjd->ijab", tau9, u[o, v, o, v], optimize=True)

    tau34 -= np.einsum("ijab->ijab", tau32, optimize=True)

    tau32 = None

    tau35 = zeros((N, N, M, M))

    tau35 += np.einsum("cbkj,kica->ijab", t2, tau34, optimize=True)

    tau34 = None

    tau41 -= 2 * np.einsum("jiab->ijab", tau35, optimize=True)

    tau35 = None

    tau13 = zeros((N, N))

    tau13 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau14 = zeros((N, N, M, M))

    tau14 -= np.einsum("ik,jkab->ijab", tau13, u[o, o, v, v], optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 -= np.einsum("ijba->ijab", tau14, optimize=True)

    tau51 = zeros((N, N, M, M))

    tau51 -= 2 * np.einsum("ijba->ijab", tau14, optimize=True)

    tau69 += 4 * np.einsum("jiba->ijab", tau14, optimize=True)

    tau75 = zeros((N, N, M, M))

    tau75 -= np.einsum("ijba->ijab", tau14, optimize=True)

    tau14 = None

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("kj,abik->ijab", tau13, t2, optimize=True)

    tau29 += 2 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau27 = None

    tau47 = zeros((N, N))

    tau47 -= np.einsum("kl,lijk->ij", tau13, u[o, o, o, o], optimize=True)

    tau53 = zeros((N, N))

    tau53 -= 4 * np.einsum("ji->ij", tau47, optimize=True)

    tau47 = None

    tau67 = zeros((M, M))

    tau67 += np.einsum("ij,jaib->ab", tau13, u[o, v, o, v], optimize=True)

    tau71 = zeros((M, M))

    tau71 += 4 * np.einsum("ab->ab", tau67, optimize=True)

    tau67 = None

    tau76 += 2 * np.einsum("jm,milk->ijkl", tau13, u[o, o, o, o], optimize=True)

    tau13 = None

    tau15 = zeros((M, M))

    tau15 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("ac,jibc->ijab", tau15, u[o, o, v, v], optimize=True)

    tau19 -= np.einsum("ijba->ijab", tau16, optimize=True)

    tau51 -= 4 * np.einsum("ijba->ijab", tau16, optimize=True)

    tau69 += 2 * np.einsum("ijab->ijab", tau16, optimize=True)

    tau75 -= np.einsum("ijba->ijab", tau16, optimize=True)

    tau16 = None

    tau28 = zeros((N, N, M, M))

    tau28 -= np.einsum("cb,acji->ijab", tau15, t2, optimize=True)

    tau29 += 2 * np.einsum("ijba->ijab", tau28, optimize=True)

    tau74 += np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau48 = zeros((N, N))

    tau48 += np.einsum("ab,iajb->ij", tau15, u[o, v, o, v], optimize=True)

    tau53 += 4 * np.einsum("ji->ij", tau48, optimize=True)

    tau48 = None

    tau68 = zeros((M, M))

    tau68 -= np.einsum("cd,acdb->ab", tau15, u[v, v, v, v], optimize=True)

    tau15 = None

    tau71 -= 4 * np.einsum("ab->ab", tau68, optimize=True)

    tau68 = None

    tau17 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau18 = zeros((N, N, M, M))

    tau18 += np.einsum("cbkj,kica->ijab", l2, tau17, optimize=True)

    tau19 -= 2 * np.einsum("jiba->ijab", tau18, optimize=True)

    tau20 = zeros((N, N, M, M))

    tau20 += np.einsum("cbkj,kiac->ijab", t2, tau19, optimize=True)

    tau19 = None

    tau25 -= 2 * np.einsum("jiab->ijab", tau20, optimize=True)

    tau20 = None

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("cbkj,ikca->ijab", t2, tau25, optimize=True)

    tau25 = None

    tau41 += np.einsum("ijab->ijab", tau26, optimize=True)

    tau26 = None

    tau51 -= 8 * np.einsum("jiba->ijab", tau18, optimize=True)

    tau69 -= 8 * np.einsum("jiba->ijab", tau18, optimize=True)

    tau75 += 4 * np.einsum("jiab->ijab", tau18, optimize=True)

    tau18 = None

    tau44 = zeros((N, N, N, N))

    tau44 += np.einsum("klab,ijab->ijkl", tau1, tau17, optimize=True)

    tau45 += 4 * np.einsum("lkij->ijkl", tau44, optimize=True)

    tau44 = None

    tau46 = zeros((N, N, M, M))

    tau46 += np.einsum("ablk,kilj->ijab", t2, tau45, optimize=True)

    tau45 = None

    tau55 = zeros((N, N, M, M))

    tau55 += 2 * np.einsum("ijba->ijab", tau46, optimize=True)

    tau46 = None

    tau59 = zeros((M, M, M, M))

    tau59 += np.einsum("ijcd,ijab->abcd", tau1, tau17, optimize=True)

    tau1 = None

    tau17 = None

    tau60 = zeros((M, M, M, M))

    tau60 += 4 * np.einsum("cdab->abcd", tau59, optimize=True)

    tau59 = None

    tau29 += 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("jkcb,kaic->ijab", tau29, u[o, v, o, v], optimize=True)

    tau29 = None

    tau41 -= np.einsum("jiba->ijab", tau30, optimize=True)

    tau30 = None

    r2 -= np.einsum("ijab->abij", tau41, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau41, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau41, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau41, optimize=True) / 4

    tau41 = None

    tau49 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum("abkl,ijkl->ijab", l2, tau49, optimize=True)

    tau51 -= np.einsum("jiba->ijab", tau50, optimize=True)

    tau64 -= np.einsum("jiba->ijab", tau50, optimize=True)

    tau50 = None

    r2 -= np.einsum("klab,klji->abij", tau22, tau49, optimize=True) / 2

    tau49 = None

    tau22 = None

    tau51 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau52 = zeros((N, N))

    tau52 += np.einsum("abkj,kiab->ij", t2, tau51, optimize=True)

    tau51 = None

    tau53 -= np.einsum("ji->ij", tau52, optimize=True)

    tau52 = None

    tau54 = zeros((N, N, M, M))

    tau54 += np.einsum("ik,abkj->ijab", tau53, t2, optimize=True)

    tau53 = None

    tau55 += np.einsum("jiba->ijab", tau54, optimize=True)

    tau54 = None

    r2 += np.einsum("ijba->abij", tau55, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau55, optimize=True) / 8

    tau55 = None

    tau56 = zeros((N, N, M, M))

    tau56 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    tau73 = zeros((N, N, M, M))

    tau73 += 8 * np.einsum("jiab->ijab", tau56, optimize=True)

    tau56 = None

    tau57 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau58 = zeros((M, M, M, M))

    tau58 += np.einsum("ecfd,eafb->abcd", tau57, tau9, optimize=True)

    tau9 = None

    tau57 = None

    tau60 += np.einsum("abcd->abcd", tau58, optimize=True)

    tau58 = None

    tau61 = zeros((N, N, M, M))

    tau61 += np.einsum("dcij,cabd->ijab", t2, tau60, optimize=True)

    tau60 = None

    tau73 -= 2 * np.einsum("jiab->ijab", tau61, optimize=True)

    tau61 = None

    tau62 = zeros((N, N, M, M))

    tau62 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau64 -= 2 * np.einsum("jiba->ijab", tau62, optimize=True)

    tau69 -= 2 * np.einsum("jiba->ijab", tau62, optimize=True)

    tau62 = None

    tau64 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau65 = zeros((N, N, M, M))

    tau65 += np.einsum("cbkj,ikca->ijab", t2, tau64, optimize=True)

    tau64 = None

    tau66 = zeros((N, N, M, M))

    tau66 += np.einsum("cbkj,kica->ijab", t2, tau65, optimize=True)

    tau65 = None

    tau73 -= 2 * np.einsum("jiab->ijab", tau66, optimize=True)

    tau66 = None

    tau69 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau70 = zeros((M, M))

    tau70 += np.einsum("cbij,ijca->ab", t2, tau69, optimize=True)

    tau69 = None

    tau71 -= np.einsum("ba->ab", tau70, optimize=True)

    tau70 = None

    tau72 = zeros((N, N, M, M))

    tau72 += np.einsum("ac,cbij->ijab", tau71, t2, optimize=True)

    tau71 = None

    tau73 -= np.einsum("jiba->ijab", tau72, optimize=True)

    tau72 = None

    r2 += np.einsum("ijab->abij", tau73, optimize=True) / 8

    r2 -= np.einsum("ijba->abij", tau73, optimize=True) / 8

    tau73 = None

    tau74 -= np.einsum("baji->ijab", t2, optimize=True)

    r2 += np.einsum("ijcd,bacd->abij", tau74, u[v, v, v, v], optimize=True) / 2

    tau74 = None

    tau75 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau76 += np.einsum("ablk,jiab->ijkl", t2, tau75, optimize=True)

    tau75 = None

    tau76 -= 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 -= np.einsum("bakl,klji->abij", t2, tau76, optimize=True) / 4

    tau76 = None

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    r2 = 0.25*(r2 - r2.transpose(1,0,2,3) - r2.transpose(0,1,3,2) + r2.transpose(1,0,3,2))

    return r2
