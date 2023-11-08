import numpy as np


def amplitudes_intermediates_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 -= np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau21 = np.zeros((N, N, M, M))

    tau21 -= np.einsum("cbjk,kica->ijab", t2, tau0, optimize=True)

    tau22 = np.zeros((N, N, M, M))

    tau22 += 4 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau28 = np.zeros((N, N, M, M))

    tau28 += 4 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau74 = np.zeros((N, N, M, M))

    tau74 -= 2 * np.einsum("ijab->ijab", tau21, optimize=True)

    tau76 = np.zeros((N, N, N, N))

    tau76 -= 2 * np.einsum("lkba,jiba->ijkl", tau21, u[o, o, v, v], optimize=True)

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("klab,iljk->ijab", tau0, u[o, o, o, o], optimize=True)

    tau33 = np.zeros((N, N, M, M))

    tau33 += 2 * np.einsum("ijab->ijab", tau30, optimize=True)

    tau30 = None

    tau32 = np.zeros((N, N, M, M))

    tau32 += np.einsum("ikca,kcjb->ijab", tau0, u[o, v, o, v], optimize=True)

    tau33 += 2 * np.einsum("ijba->ijab", tau32, optimize=True)

    tau32 = None

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("kiac,jbkc->ijab", tau0, u[o, v, o, v], optimize=True)

    tau38 = np.zeros((N, N, M, M))

    tau38 += 2 * np.einsum("ijab->ijab", tau35, optimize=True)

    tau35 = None

    tau37 = np.zeros((N, N, M, M))

    tau37 -= np.einsum("ijcd,acdb->ijab", tau0, u[v, v, v, v], optimize=True)

    tau38 += 2 * np.einsum("jiba->ijab", tau37, optimize=True)

    tau37 = None

    tau1 = np.zeros((M, M, M, M))

    tau1 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau2 = np.zeros((N, N, M, M))

    tau2 -= np.einsum("ijcd,acdb->ijab", tau0, tau1, optimize=True)

    tau24 = np.zeros((N, N, M, M))

    tau24 += 2 * np.einsum("jiba->ijab", tau2, optimize=True)

    tau2 = None

    tau42 = np.zeros((M, M, M, M))

    tau42 += np.einsum("badc->abcd", tau1, optimize=True)

    tau1 = None

    tau3 = np.zeros((N, N, N, N))

    tau3 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau20 = np.zeros((N, N, M, M))

    tau20 -= np.einsum("ablk,lkji->ijab", t2, tau3, optimize=True)

    tau22 -= np.einsum("ijba->ijab", tau20, optimize=True)

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("jkcb,kica->ijab", tau22, u[o, o, v, v], optimize=True)

    tau22 = None

    tau24 += np.einsum("jiab->ijab", tau23, optimize=True)

    tau23 = None

    tau28 -= np.einsum("ijba->ijab", tau20, optimize=True)

    tau20 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 -= np.einsum("kijl,lakb->ijab", tau3, u[o, v, o, v], optimize=True)

    tau38 -= np.einsum("jiba->ijab", tau36, optimize=True)

    tau36 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum("cbkj,ikca->ijab", t2, tau38, optimize=True)

    tau38 = None

    tau40 = np.zeros((N, N, M, M))

    tau40 -= 2 * np.einsum("ijba->ijab", tau39, optimize=True)

    tau39 = None

    tau48 = np.zeros((N, N, M, M))

    tau48 -= np.einsum("jilk,lkab->ijab", tau3, u[o, o, v, v], optimize=True)

    tau51 = np.zeros((N, N, M, M))

    tau51 += np.einsum("ijba->ijab", tau48, optimize=True)

    tau56 = np.zeros((N, N, M, M))

    tau56 += np.einsum("ijba->ijab", tau48, optimize=True)

    tau48 = None

    tau61 = np.zeros((N, N, N, N))

    tau61 += np.einsum("mijn,nklm->ijkl", tau3, u[o, o, o, o], optimize=True)

    tau64 = np.zeros((N, N, N, N))

    tau64 += 2 * np.einsum("ijkl->ijkl", tau61, optimize=True)

    tau61 = None

    tau4 = np.zeros((N, N, M, M))

    tau4 -= np.einsum("caik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("ikjl,klab->ijab", tau3, tau4, optimize=True)

    tau24 += 2 * np.einsum("jiba->ijab", tau5, optimize=True)

    tau5 = None

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("kiac,kjbc->ijab", tau0, tau4, optimize=True)

    tau24 -= 4 * np.einsum("ijab->ijab", tau10, optimize=True)

    tau10 = None

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("ikca,jkcb->ijab", tau0, tau4, optimize=True)

    tau24 -= 4 * np.einsum("jiba->ijab", tau11, optimize=True)

    tau11 = None

    tau16 = np.zeros((N, N, M, M))

    tau16 += np.einsum("ijab->ijab", tau4, optimize=True)

    tau6 = np.zeros((N, N, N, N))

    tau6 += np.einsum("baij,klba->ijkl", t2, u[o, o, v, v], optimize=True)

    tau7 = np.zeros((N, N, M, M))

    tau7 -= np.einsum("klab,ikjl->ijab", tau0, tau6, optimize=True)

    tau24 -= 2 * np.einsum("ijab->ijab", tau7, optimize=True)

    tau7 = None

    tau49 = np.zeros((N, N, N, N))

    tau49 += np.einsum("lkji->ijkl", tau6, optimize=True)

    tau62 = np.zeros((N, N, N, N))

    tau62 -= np.einsum("imkn,jmnl->ijkl", tau3, tau6, optimize=True)

    tau3 = None

    tau6 = None

    tau64 -= np.einsum("ijlk->ijkl", tau62, optimize=True)

    tau62 = None

    tau8 = np.zeros((M, M, M, M))

    tau8 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 -= np.einsum("ijcd,acbd->ijab", tau4, tau8, optimize=True)

    tau4 = None

    tau24 -= 2 * np.einsum("ijab->ijab", tau9, optimize=True)

    tau9 = None

    tau31 = np.zeros((N, N, M, M))

    tau31 -= np.einsum("cabd,icjd->ijab", tau8, u[o, v, o, v], optimize=True)

    tau33 -= np.einsum("ijab->ijab", tau31, optimize=True)

    tau31 = None

    tau34 = np.zeros((N, N, M, M))

    tau34 += np.einsum("cbkj,kica->ijab", t2, tau33, optimize=True)

    tau33 = None

    tau40 -= 2 * np.einsum("jiab->ijab", tau34, optimize=True)

    tau34 = None

    tau12 = np.zeros((N, N))

    tau12 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau13 = np.zeros((N, N, M, M))

    tau13 -= np.einsum("ik,jkab->ijab", tau12, u[o, o, v, v], optimize=True)

    tau18 = np.zeros((N, N, M, M))

    tau18 -= np.einsum("ijba->ijab", tau13, optimize=True)

    tau56 += 4 * np.einsum("jiba->ijab", tau13, optimize=True)

    tau68 = np.zeros((N, N, M, M))

    tau68 -= 2 * np.einsum("ijba->ijab", tau13, optimize=True)

    tau75 = np.zeros((N, N, M, M))

    tau75 -= np.einsum("ijba->ijab", tau13, optimize=True)

    tau13 = None

    tau26 = np.zeros((N, N, M, M))

    tau26 += np.einsum("kj,abik->ijab", tau12, t2, optimize=True)

    tau28 += 2 * np.einsum("ijba->ijab", tau26, optimize=True)

    tau26 = None

    tau54 = np.zeros((M, M))

    tau54 += np.einsum("ij,jaib->ab", tau12, u[o, v, o, v], optimize=True)

    tau58 = np.zeros((M, M))

    tau58 += 4 * np.einsum("ab->ab", tau54, optimize=True)

    tau54 = None

    tau66 = np.zeros((N, N))

    tau66 -= np.einsum("kl,lijk->ij", tau12, u[o, o, o, o], optimize=True)

    tau70 = np.zeros((N, N))

    tau70 -= 4 * np.einsum("ji->ij", tau66, optimize=True)

    tau66 = None

    tau76 += 2 * np.einsum("jm,milk->ijkl", tau12, u[o, o, o, o], optimize=True)

    tau12 = None

    tau14 = np.zeros((M, M))

    tau14 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau15 = np.zeros((N, N, M, M))

    tau15 += np.einsum("ac,jibc->ijab", tau14, u[o, o, v, v], optimize=True)

    tau18 -= np.einsum("ijba->ijab", tau15, optimize=True)

    tau56 += 2 * np.einsum("ijab->ijab", tau15, optimize=True)

    tau68 -= 4 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau75 -= np.einsum("ijba->ijab", tau15, optimize=True)

    tau15 = None

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("cb,acij->ijab", tau14, t2, optimize=True)

    tau28 += 2 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau74 += np.einsum("ijab->ijab", tau27, optimize=True)

    tau27 = None

    tau55 = np.zeros((M, M))

    tau55 -= np.einsum("cd,acdb->ab", tau14, u[v, v, v, v], optimize=True)

    tau58 -= 4 * np.einsum("ab->ab", tau55, optimize=True)

    tau55 = None

    tau67 = np.zeros((N, N))

    tau67 += np.einsum("ab,iajb->ij", tau14, u[o, v, o, v], optimize=True)

    tau14 = None

    tau70 += 4 * np.einsum("ji->ij", tau67, optimize=True)

    tau67 = None

    tau16 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau17 = np.zeros((N, N, M, M))

    tau17 += np.einsum("cbkj,kica->ijab", l2, tau16, optimize=True)

    tau18 -= 2 * np.einsum("jiba->ijab", tau17, optimize=True)

    tau19 = np.zeros((N, N, M, M))

    tau19 += np.einsum("cbkj,kiac->ijab", t2, tau18, optimize=True)

    tau18 = None

    tau24 -= 2 * np.einsum("jiab->ijab", tau19, optimize=True)

    tau19 = None

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum("cbkj,ikca->ijab", t2, tau24, optimize=True)

    tau24 = None

    tau40 += np.einsum("ijab->ijab", tau25, optimize=True)

    tau25 = None

    tau56 -= 8 * np.einsum("jiba->ijab", tau17, optimize=True)

    tau68 -= 8 * np.einsum("jiba->ijab", tau17, optimize=True)

    tau75 += 4 * np.einsum("jiab->ijab", tau17, optimize=True)

    tau17 = None

    tau44 = np.zeros((M, M, M, M))

    tau44 += np.einsum("ijcd,ijab->abcd", tau0, tau16, optimize=True)

    tau45 = np.zeros((M, M, M, M))

    tau45 += 4 * np.einsum("cdab->abcd", tau44, optimize=True)

    tau44 = None

    tau63 = np.zeros((N, N, N, N))

    tau63 += np.einsum("klab,ijab->ijkl", tau0, tau16, optimize=True)

    tau0 = None

    tau16 = None

    tau64 += 4 * np.einsum("lkij->ijkl", tau63, optimize=True)

    tau63 = None

    tau65 = np.zeros((N, N, M, M))

    tau65 += np.einsum("ablk,kilj->ijab", t2, tau64, optimize=True)

    tau64 = None

    tau72 = np.zeros((N, N, M, M))

    tau72 += 2 * np.einsum("ijba->ijab", tau65, optimize=True)

    tau65 = None

    tau28 += 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau29 = np.zeros((N, N, M, M))

    tau29 += np.einsum("jkcb,kaic->ijab", tau28, u[o, v, o, v], optimize=True)

    tau28 = None

    tau40 -= np.einsum("jiba->ijab", tau29, optimize=True)

    tau29 = None

    r2 = np.zeros((M, M, N, N))

    r2 -= np.einsum("ijab->abij", tau40, optimize=True) / 4

    r2 += np.einsum("ijba->abij", tau40, optimize=True) / 4

    r2 += np.einsum("jiab->abij", tau40, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau40, optimize=True) / 4

    tau40 = None

    tau41 = np.zeros((N, N, M, M))

    tau41 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    tau60 = np.zeros((N, N, M, M))

    tau60 += 8 * np.einsum("jiab->ijab", tau41, optimize=True)

    tau41 = None

    tau42 += 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau43 = np.zeros((M, M, M, M))

    tau43 += np.einsum("eafb,ecfd->abcd", tau42, tau8, optimize=True)

    tau8 = None

    tau42 = None

    tau45 += np.einsum("cdab->abcd", tau43, optimize=True)

    tau43 = None

    tau46 = np.zeros((N, N, M, M))

    tau46 += np.einsum("dcij,cabd->ijab", t2, tau45, optimize=True)

    tau45 = None

    tau60 -= 2 * np.einsum("jiab->ijab", tau46, optimize=True)

    tau46 = None

    tau47 = np.zeros((N, N, M, M))

    tau47 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau51 -= 2 * np.einsum("jiba->ijab", tau47, optimize=True)

    tau56 -= 2 * np.einsum("jiba->ijab", tau47, optimize=True)

    tau47 = None

    tau49 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau50 = np.zeros((N, N, M, M))

    tau50 += np.einsum("abkl,ijkl->ijab", l2, tau49, optimize=True)

    tau51 -= np.einsum("jiba->ijab", tau50, optimize=True)

    tau68 -= np.einsum("jiba->ijab", tau50, optimize=True)

    tau50 = None

    r2 += np.einsum("klba,klji->abij", tau21, tau49, optimize=True) / 2

    tau49 = None

    tau21 = None

    tau51 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau52 = np.zeros((N, N, M, M))

    tau52 += np.einsum("cbkj,ikca->ijab", t2, tau51, optimize=True)

    tau51 = None

    tau53 = np.zeros((N, N, M, M))

    tau53 += np.einsum("cbkj,kica->ijab", t2, tau52, optimize=True)

    tau52 = None

    tau60 -= 2 * np.einsum("ijba->ijab", tau53, optimize=True)

    tau53 = None

    tau56 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau57 = np.zeros((M, M))

    tau57 += np.einsum("cbij,ijca->ab", t2, tau56, optimize=True)

    tau56 = None

    tau58 -= np.einsum("ba->ab", tau57, optimize=True)

    tau57 = None

    tau59 = np.zeros((N, N, M, M))

    tau59 += np.einsum("ac,cbij->ijab", tau58, t2, optimize=True)

    tau58 = None

    tau60 -= np.einsum("jiba->ijab", tau59, optimize=True)

    tau59 = None

    r2 += np.einsum("ijab->abij", tau60, optimize=True) / 8

    r2 -= np.einsum("ijba->abij", tau60, optimize=True) / 8

    tau60 = None

    tau68 -= 4 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau69 = np.zeros((N, N))

    tau69 += np.einsum("abkj,kiab->ij", t2, tau68, optimize=True)

    tau68 = None

    tau70 -= np.einsum("ji->ij", tau69, optimize=True)

    tau69 = None

    tau71 = np.zeros((N, N, M, M))

    tau71 += np.einsum("ik,abkj->ijab", tau70, t2, optimize=True)

    tau70 = None

    tau72 += np.einsum("jiba->ijab", tau71, optimize=True)

    tau71 = None

    r2 += np.einsum("ijba->abij", tau72, optimize=True) / 8

    r2 -= np.einsum("jiba->abij", tau72, optimize=True) / 8

    tau72 = None

    tau73 = np.zeros((N, N, M, M))

    tau73 += np.einsum("ki,abjk->ijab", f[o, o], t2, optimize=True)

    r2 -= np.einsum("ijba->abij", tau73, optimize=True)

    r2 += np.einsum("jiba->abij", tau73, optimize=True)

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

    return r2
