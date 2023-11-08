import numpy as np


def amplitudes_intermediates_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 -= np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 -= np.einsum("cbjk,kica->ijab", t2, tau0, optimize=True)

    tau40 = np.zeros((N, N, M, M))

    tau40 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau54 = np.zeros((N, N, M, M))

    tau54 -= 2 * np.einsum("ijab->ijab", tau1, optimize=True)

    r2 = np.zeros((M, M, N, N))

    r2 += np.einsum("klab,lkji->abij", tau1, u[o, o, o, o], optimize=True)

    tau1 = None

    tau4 = np.zeros((N, N, N, N))

    tau4 += np.einsum("ijab,kalb->ijkl", tau0, u[o, v, o, v], optimize=True)

    tau5 = np.zeros((N, N, N, N))

    tau5 += 2 * np.einsum("ijkl->ijkl", tau4, optimize=True)

    tau4 = None

    tau19 = np.zeros((M, M, M, M))

    tau19 += np.einsum("ijab,jcid->abcd", tau0, u[o, v, o, v], optimize=True)

    tau22 = np.zeros((M, M, M, M))

    tau22 += 2 * np.einsum("abcd->abcd", tau19, optimize=True)

    tau19 = None

    tau42 = np.zeros((N, N, M, M))

    tau42 += np.einsum("klab,iljk->ijab", tau0, u[o, o, o, o], optimize=True)

    tau45 = np.zeros((N, N, M, M))

    tau45 += 2 * np.einsum("ijab->ijab", tau42, optimize=True)

    tau42 = None

    tau44 = np.zeros((N, N, M, M))

    tau44 += np.einsum("ikca,kcjb->ijab", tau0, u[o, v, o, v], optimize=True)

    tau45 += 2 * np.einsum("ijba->ijab", tau44, optimize=True)

    tau44 = None

    tau47 = np.zeros((N, N, M, M))

    tau47 += np.einsum("kiac,jbkc->ijab", tau0, u[o, v, o, v], optimize=True)

    tau50 = np.zeros((N, N, M, M))

    tau50 += 2 * np.einsum("ijab->ijab", tau47, optimize=True)

    tau47 = None

    tau49 = np.zeros((N, N, M, M))

    tau49 -= np.einsum("ijcd,acdb->ijab", tau0, u[v, v, v, v], optimize=True)

    tau0 = None

    tau50 += 2 * np.einsum("jiba->ijab", tau49, optimize=True)

    tau49 = None

    tau2 = np.zeros((N, N, N, N))

    tau2 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau3 = np.zeros((N, N, N, N))

    tau3 -= np.einsum("imjn,nklm->ijkl", tau2, u[o, o, o, o], optimize=True)

    tau5 -= np.einsum("ijkl->ijkl", tau3, optimize=True)

    tau3 = None

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("ablk,kilj->ijab", t2, tau5, optimize=True)

    tau5 = None

    tau17 = np.zeros((N, N, M, M))

    tau17 -= 2 * np.einsum("ijba->ijab", tau6, optimize=True)

    tau6 = None

    tau37 = np.zeros((N, N, M, M))

    tau37 -= np.einsum("ablk,lkji->ijab", t2, tau2, optimize=True)

    tau40 -= np.einsum("ijba->ijab", tau37, optimize=True)

    tau37 = None

    tau48 = np.zeros((N, N, M, M))

    tau48 += np.einsum("ikjl,lakb->ijab", tau2, u[o, v, o, v], optimize=True)

    tau2 = None

    tau50 -= np.einsum("jiba->ijab", tau48, optimize=True)

    tau48 = None

    tau51 = np.zeros((N, N, M, M))

    tau51 += np.einsum("cbkj,ikca->ijab", t2, tau50, optimize=True)

    tau50 = None

    tau52 = np.zeros((N, N, M, M))

    tau52 += 2 * np.einsum("ijba->ijab", tau51, optimize=True)

    tau51 = None

    tau7 = np.zeros((N, N))

    tau7 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau8 = np.zeros((N, N))

    tau8 -= np.einsum("kl,lijk->ij", tau7, u[o, o, o, o], optimize=True)

    tau15 = np.zeros((N, N))

    tau15 -= 2 * np.einsum("ji->ij", tau8, optimize=True)

    tau8 = None

    tau28 = np.zeros((M, M))

    tau28 += np.einsum("ij,jaib->ab", tau7, u[o, v, o, v], optimize=True)

    tau32 = np.zeros((M, M))

    tau32 += 2 * np.einsum("ab->ab", tau28, optimize=True)

    tau28 = None

    tau38 = np.zeros((N, N, M, M))

    tau38 += np.einsum("kj,abik->ijab", tau7, t2, optimize=True)

    tau40 += 2 * np.einsum("ijba->ijab", tau38, optimize=True)

    tau38 = None

    tau56 = np.zeros((N, N, N, N))

    tau56 += 2 * np.einsum("jm,milk->ijkl", tau7, u[o, o, o, o], optimize=True)

    tau7 = None

    tau9 = np.zeros((M, M))

    tau9 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau10 = np.zeros((N, N))

    tau10 += np.einsum("ab,iajb->ij", tau9, u[o, v, o, v], optimize=True)

    tau15 += 2 * np.einsum("ji->ij", tau10, optimize=True)

    tau10 = None

    tau29 = np.zeros((M, M))

    tau29 -= np.einsum("cd,acdb->ab", tau9, u[v, v, v, v], optimize=True)

    tau32 -= 2 * np.einsum("ab->ab", tau29, optimize=True)

    tau29 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 -= np.einsum("cb,acji->ijab", tau9, t2, optimize=True)

    tau9 = None

    tau40 += 2 * np.einsum("ijba->ijab", tau39, optimize=True)

    tau54 += np.einsum("ijab->ijab", tau39, optimize=True)

    tau39 = None

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("ablk,ijlk->ijab", l2, u[o, o, o, o], optimize=True)

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum("jiba->ijab", tau11, optimize=True)

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum("jiba->ijab", tau11, optimize=True)

    tau11 = None

    tau12 = np.zeros((N, N, M, M))

    tau12 += np.einsum("caki,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau13 -= 4 * np.einsum("ijab->ijab", tau12, optimize=True)

    tau30 = np.zeros((N, N, M, M))

    tau30 -= 4 * np.einsum("ijab->ijab", tau12, optimize=True)

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("bcik,kjac->ijab", t2, tau12, optimize=True)

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum("bcjk,ikca->ijab", t2, tau35, optimize=True)

    tau35 = None

    tau52 += 4 * np.einsum("ijab->ijab", tau36, optimize=True)

    tau36 = None

    tau55 = np.zeros((N, N, M, M))

    tau55 += 4 * np.einsum("ijba->ijab", tau12, optimize=True)

    tau12 = None

    tau13 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau14 = np.zeros((N, N))

    tau14 += np.einsum("abkj,kiab->ij", t2, tau13, optimize=True)

    tau13 = None

    tau15 += np.einsum("ji->ij", tau14, optimize=True)

    tau14 = None

    tau16 = np.zeros((N, N, M, M))

    tau16 += np.einsum("ik,abkj->ijab", tau15, t2, optimize=True)

    tau15 = None

    tau17 += np.einsum("jiba->ijab", tau16, optimize=True)

    tau16 = None

    r2 += np.einsum("ijba->abij", tau17, optimize=True) / 4

    r2 -= np.einsum("jiba->abij", tau17, optimize=True) / 4

    tau17 = None

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    tau34 = np.zeros((N, N, M, M))

    tau34 += 4 * np.einsum("jiab->ijab", tau18, optimize=True)

    tau18 = None

    tau20 = np.zeros((M, M, M, M))

    tau20 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau21 = np.zeros((M, M, M, M))

    tau21 -= np.einsum("eabf,cedf->abcd", tau20, u[v, v, v, v], optimize=True)

    tau22 -= np.einsum("abcd->abcd", tau21, optimize=True)

    tau21 = None

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("dcij,cabd->ijab", t2, tau22, optimize=True)

    tau22 = None

    tau34 += 2 * np.einsum("jiab->ijab", tau23, optimize=True)

    tau23 = None

    tau43 = np.zeros((N, N, M, M))

    tau43 -= np.einsum("cabd,icjd->ijab", tau20, u[o, v, o, v], optimize=True)

    tau20 = None

    tau45 -= np.einsum("ijab->ijab", tau43, optimize=True)

    tau43 = None

    tau46 = np.zeros((N, N, M, M))

    tau46 += np.einsum("cbkj,kica->ijab", t2, tau45, optimize=True)

    tau45 = None

    tau52 += 2 * np.einsum("jiab->ijab", tau46, optimize=True)

    tau46 = None

    tau24 = np.zeros((N, N, M, M))

    tau24 += np.einsum("dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    tau25 += np.einsum("jiba->ijab", tau24, optimize=True)

    tau30 += np.einsum("jiba->ijab", tau24, optimize=True)

    tau24 = None

    tau25 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau26 = np.zeros((N, N, M, M))

    tau26 += np.einsum("cbkj,kica->ijab", t2, tau25, optimize=True)

    tau25 = None

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("cbkj,kica->ijab", t2, tau26, optimize=True)

    tau26 = None

    tau34 -= 2 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau27 = None

    tau30 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau31 = np.zeros((M, M))

    tau31 += np.einsum("cbij,ijca->ab", t2, tau30, optimize=True)

    tau30 = None

    tau32 += np.einsum("ba->ab", tau31, optimize=True)

    tau31 = None

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("ac,cbij->ijab", tau32, t2, optimize=True)

    tau32 = None

    tau34 -= np.einsum("jiba->ijab", tau33, optimize=True)

    tau33 = None

    r2 += np.einsum("ijab->abij", tau34, optimize=True) / 4

    r2 -= np.einsum("ijba->abij", tau34, optimize=True) / 4

    tau34 = None

    tau40 += 4 * np.einsum("baji->ijab", t2, optimize=True)

    tau41 = np.zeros((N, N, M, M))

    tau41 += np.einsum("ikca,kbjc->ijab", tau40, u[o, v, o, v], optimize=True)

    tau40 = None

    tau52 += np.einsum("ijab->ijab", tau41, optimize=True)

    tau41 = None

    r2 += np.einsum("ijab->abij", tau52, optimize=True) / 4

    r2 -= np.einsum("ijba->abij", tau52, optimize=True) / 4

    r2 -= np.einsum("jiab->abij", tau52, optimize=True) / 4

    r2 += np.einsum("jiba->abij", tau52, optimize=True) / 4

    tau52 = None

    tau53 = np.zeros((N, N, M, M))

    tau53 += np.einsum("ki,abjk->ijab", f[o, o], t2, optimize=True)

    r2 -= np.einsum("ijba->abij", tau53, optimize=True)

    r2 += np.einsum("jiba->abij", tau53, optimize=True)

    tau53 = None

    tau54 -= np.einsum("baji->ijab", t2, optimize=True)

    r2 += np.einsum("ijcd,bacd->abij", tau54, u[v, v, v, v], optimize=True) / 2

    tau54 = None

    tau55 += np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau56 -= np.einsum("ablk,jiab->ijkl", t2, tau55, optimize=True)

    tau55 = None

    tau56 -= 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 -= np.einsum("bakl,klji->abij", t2, tau56, optimize=True) / 4

    tau56 = None

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    return r2
