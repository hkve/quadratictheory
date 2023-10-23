import numpy as np


def lambda_amplitudes_intermediates_ccsd(t1, t2, l1, l2, u, f, v, o):
    M, N = t1.shape

    tau0 = np.zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau17 = np.zeros((N, N, N, M))

    tau17 -= np.einsum("al,ilkj->ijka", t1, tau0, optimize=True)

    r1 = np.zeros((M, N))

    r1 -= np.einsum("ijlk,lkja->ai", tau0, u[o, o, o, v], optimize=True) / 4

    r2 = np.zeros((M, M, N, N))

    r2 += np.einsum("ijlk,lkab->abij", tau0, u[o, o, v, v], optimize=True) / 4

    tau0 = None

    tau1 = np.zeros((N, N, N, M))

    tau1 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau2 = np.zeros((N, N, N, N))

    tau2 += np.einsum("ak,ijla->ijkl", t1, tau1, optimize=True)

    r1 += np.einsum("iljk,kjla->ai", tau2, u[o, o, o, v], optimize=True) / 2

    tau2 = None

    tau24 = np.zeros((N, M))

    tau24 += np.einsum("bakj,kjib->ia", t2, tau1, optimize=True)

    tau3 = np.zeros((N, N, N, M))

    tau3 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau3 -= np.einsum("bi,kjab->ijka", t1, u[o, o, v, v], optimize=True)

    tau11 = np.zeros((N, N, N, M))

    tau11 += 4 * np.einsum("balj,klib->ijka", t2, tau3, optimize=True)

    tau3 = None

    tau4 = np.zeros((N, N, M, M))

    tau4 -= np.einsum("baji->ijab", t2, optimize=True)

    tau4 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau5 = np.zeros((N, M, M, M))

    tau5 += np.einsum("aj,ijbc->iabc", t1, u[o, o, v, v], optimize=True)

    tau6 = np.zeros((N, M, M, M))

    tau6 += np.einsum("iacb->iabc", tau5, optimize=True)

    tau51 = np.zeros((N, M, M, M))

    tau51 += np.einsum("iacb->iabc", tau5, optimize=True)

    tau5 = None

    tau6 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau11 -= np.einsum("kjbc,iabc->ijka", tau4, tau6, optimize=True)

    tau4 = None

    tau6 = None

    tau7 = np.zeros((N, M))

    tau7 -= np.einsum("bj,ijba->ia", t1, u[o, o, v, v], optimize=True)

    tau8 = np.zeros((N, M))

    tau8 += np.einsum("ia->ia", tau7, optimize=True)

    tau7 = None

    tau8 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau11 += 2 * np.einsum("ib,bakj->ijka", tau8, t2, optimize=True)

    tau46 = np.zeros((N, N, M, M))

    tau46 += np.einsum("ai,jb->ijab", l1, tau8, optimize=True)

    tau8 = None

    tau9 = np.zeros((N, N, N, N))

    tau9 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau10 = np.zeros((N, N, N, N))

    tau10 -= 2 * np.einsum("kjil->ijkl", tau9, optimize=True)

    tau53 = np.zeros((N, N, N, N))

    tau53 -= 4 * np.einsum("ljik->ijkl", tau9, optimize=True)

    tau9 = None

    tau10 -= np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau11 -= 2 * np.einsum("al,likj->ijka", t1, tau10, optimize=True)

    tau10 = None

    tau11 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau11 -= 4 * np.einsum("bk,iajb->ijka", t1, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bajk,ijkb->ai", l2, tau11, optimize=True) / 4

    tau11 = None

    tau12 = np.zeros((N, M, M, M))

    tau12 -= np.einsum("baic->iabc", u[v, v, o, v], optimize=True)

    tau12 += np.einsum("di,bacd->iabc", t1, u[v, v, v, v], optimize=True)

    r1 += np.einsum("bcji,jbca->ai", l2, tau12, optimize=True) / 2

    tau12 = None

    tau13 = np.zeros((N, N, M, M))

    tau13 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau13 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau17 += 2 * np.einsum("likb,ljba->ijka", tau1, tau13, optimize=True)

    tau13 = None

    tau14 = np.zeros((N, N))

    tau14 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau16 = np.zeros((N, N))

    tau16 += 2 * np.einsum("ij->ij", tau14, optimize=True)

    tau14 = None

    tau15 = np.zeros((N, N))

    tau15 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau16 += np.einsum("ij->ij", tau15, optimize=True)

    tau15 = None

    tau17 += 2 * np.einsum("aj,ik->ijka", t1, tau16, optimize=True)

    tau24 += np.einsum("aj,ji->ia", t1, tau16, optimize=True)

    tau49 = np.zeros((N, N, M, M))

    tau49 += np.einsum("ik,kjab->ijab", tau16, u[o, o, v, v], optimize=True)

    tau50 = np.zeros((N, N, M, M))

    tau50 -= np.einsum("ijba->ijab", tau49, optimize=True)

    tau49 = None

    r1 -= np.einsum("kj,jika->ai", tau16, u[o, o, o, v], optimize=True) / 2

    r1 -= np.einsum("ja,ij->ai", f[o, v], tau16, optimize=True) / 2

    tau16 = None

    tau17 -= 2 * np.einsum("bi,abkj->ijka", l1, t2, optimize=True)

    r1 -= np.einsum("ijkb,jkba->ai", tau17, u[o, o, v, v], optimize=True) / 4

    tau17 = None

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau19 = np.zeros((N, N, M, M))

    tau19 -= np.einsum("jiab->ijab", tau18, optimize=True)

    tau44 = np.zeros((N, N, M, M))

    tau44 += np.einsum("ijab->ijab", tau18, optimize=True)

    tau18 = None

    tau19 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    r1 += np.einsum("kijb,jkba->ai", tau1, tau19, optimize=True)

    tau1 = None

    tau19 = None

    tau20 = np.zeros((M, M))

    tau20 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau22 = np.zeros((M, M))

    tau22 += 2 * np.einsum("ab->ab", tau20, optimize=True)

    tau20 = None

    tau21 = np.zeros((M, M))

    tau21 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau22 += np.einsum("ab->ab", tau21, optimize=True)

    tau21 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum("ac,ijcb->ijab", tau22, u[o, o, v, v], optimize=True)

    tau37 = np.zeros((N, N, M, M))

    tau37 += np.einsum("jiab->ijab", tau36, optimize=True)

    tau36 = None

    r1 -= np.einsum("bc,ibca->ai", tau22, u[o, v, v, v], optimize=True) / 2

    tau22 = None

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("ai,bj->ijab", l1, t1, optimize=True)

    tau23 -= np.einsum("caik,bcjk->ijab", l2, t2, optimize=True)

    r1 += np.einsum("ijbc,jbca->ai", tau23, u[o, v, v, v], optimize=True)

    tau23 = None

    tau24 -= 2 * np.einsum("ai->ia", t1, optimize=True)

    tau24 -= 2 * np.einsum("bj,abij->ia", l1, t2, optimize=True)

    r1 -= np.einsum("jb,jiba->ai", tau24, u[o, o, v, v], optimize=True) / 2

    tau24 = None

    tau25 = np.zeros((N, N))

    tau25 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau29 = np.zeros((N, N))

    tau29 += 2 * np.einsum("ij->ij", tau25, optimize=True)

    tau47 = np.zeros((N, N))

    tau47 += 2 * np.einsum("ij->ij", tau25, optimize=True)

    tau25 = None

    tau26 = np.zeros((N, N))

    tau26 += np.einsum("ak,ikja->ij", t1, u[o, o, o, v], optimize=True)

    tau29 += 2 * np.einsum("ij->ij", tau26, optimize=True)

    tau47 += 2 * np.einsum("ij->ij", tau26, optimize=True)

    tau26 = None

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("baji->ijab", t2, optimize=True)

    tau27 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau28 = np.zeros((N, N))

    tau28 += np.einsum("kiab,kjab->ij", tau27, u[o, o, v, v], optimize=True)

    tau29 += np.einsum("ji->ij", tau28, optimize=True)

    tau47 += np.einsum("ji->ij", tau28, optimize=True)

    tau28 = None

    tau48 = np.zeros((N, N, M, M))

    tau48 += np.einsum("ik,abkj->ijab", tau47, l2, optimize=True)

    tau47 = None

    tau50 += np.einsum("jiba->ijab", tau48, optimize=True)

    tau48 = None

    r2 += np.einsum("ijba->abij", tau50, optimize=True) / 2

    r2 -= np.einsum("jiba->abij", tau50, optimize=True) / 2

    tau50 = None

    tau33 = np.zeros((M, M))

    tau33 += np.einsum("ijca,ijcb->ab", tau27, u[o, o, v, v], optimize=True)

    tau34 = np.zeros((M, M))

    tau34 += np.einsum("ab->ab", tau33, optimize=True)

    tau33 = None

    tau53 += np.einsum("lkab,jiab->ijkl", tau27, u[o, o, v, v], optimize=True)

    tau27 = None

    tau29 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    r1 -= np.einsum("aj,ij->ai", l1, tau29, optimize=True) / 2

    tau29 = None

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("ak,ijkb->ijab", l1, u[o, o, o, v], optimize=True)

    tau37 += 2 * np.einsum("jiab->ijab", tau30, optimize=True)

    tau30 = None

    tau31 = np.zeros((M, M))

    tau31 += np.einsum("ia,bi->ab", f[o, v], t1, optimize=True)

    tau34 += 2 * np.einsum("ba->ab", tau31, optimize=True)

    tau31 = None

    tau32 = np.zeros((M, M))

    tau32 += np.einsum("ci,iabc->ab", t1, u[o, v, v, v], optimize=True)

    tau34 += 2 * np.einsum("ab->ab", tau32, optimize=True)

    tau32 = None

    tau34 -= 2 * np.einsum("ab->ab", f[v, v], optimize=True)

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("ca,cbij->ijab", tau34, l2, optimize=True)

    tau34 = None

    tau37 -= np.einsum("jiba->ijab", tau35, optimize=True)

    tau35 = None

    r2 += np.einsum("ijab->abij", tau37, optimize=True) / 2

    r2 -= np.einsum("ijba->abij", tau37, optimize=True) / 2

    tau37 = None

    tau38 = np.zeros((N, N, M, M))

    tau38 += np.einsum("jk,abik->ijab", f[o, o], l2, optimize=True)

    tau40 = np.zeros((N, N, M, M))

    tau40 -= np.einsum("ijba->ijab", tau38, optimize=True)

    tau38 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum("ci,jcab->ijab", l1, u[o, v, v, v], optimize=True)

    tau40 -= np.einsum("ijba->ijab", tau39, optimize=True)

    tau39 = None

    r2 -= np.einsum("ijab->abij", tau40, optimize=True)

    r2 += np.einsum("jiab->abij", tau40, optimize=True)

    tau40 = None

    tau41 = np.zeros((N, N, M, M))

    tau41 += np.einsum("ak,ikjb->ijab", t1, u[o, o, o, v], optimize=True)

    tau44 += np.einsum("jiab->ijab", tau41, optimize=True)

    tau41 = None

    tau42 = np.zeros((N, N, M, M))

    tau42 += np.einsum("baji->ijab", t2, optimize=True)

    tau42 -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    tau43 = np.zeros((N, N, M, M))

    tau43 += np.einsum("kica,kjcb->ijab", tau42, u[o, o, v, v], optimize=True)

    tau42 = None

    tau44 += np.einsum("ijab->ijab", tau43, optimize=True)

    tau43 = None

    tau44 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau45 = np.zeros((N, N, M, M))

    tau45 += np.einsum("cbkj,kica->ijab", l2, tau44, optimize=True)

    tau44 = None

    tau46 += np.einsum("jiba->ijab", tau45, optimize=True)

    tau45 = None

    r2 += np.einsum("ijab->abij", tau46, optimize=True)

    r2 -= np.einsum("ijba->abij", tau46, optimize=True)

    r2 -= np.einsum("jiab->abij", tau46, optimize=True)

    r2 += np.einsum("jiba->abij", tau46, optimize=True)

    tau46 = None

    tau51 -= 2 * np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau52 = np.zeros((M, M, M, M))

    tau52 -= np.einsum("bi,iadc->abcd", t1, tau51, optimize=True)

    tau51 = None

    tau52 += np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    r2 += np.einsum("cdji,cdba->abij", l2, tau52, optimize=True) / 2

    tau52 = None

    tau53 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 += np.einsum("bakl,jikl->abij", l2, tau53, optimize=True) / 4

    tau53 = None

    r1 += np.einsum("ia->ai", f[o, v], optimize=True)

    r1 -= np.einsum("bj,ibja->ai", l1, u[o, v, o, v], optimize=True)

    r1 += np.einsum("bi,ba->ai", l1, f[v, v], optimize=True)

    r2 += np.einsum("jiba->abij", u[o, o, v, v], optimize=True)

    return r1, r2
