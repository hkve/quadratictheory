import numpy as np


def amplitudes_CCSD_t1_t2(t1, t2, u, f, v, o):
    M, N = t1.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("baji->ijab", t2, optimize=True)

    tau0 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau6 = np.zeros((N, N))

    tau6 += np.einsum("kjab,kiab->ij", tau0, u[o, o, v, v], optimize=True)

    tau14 = np.zeros((N, N, N, M))

    tau14 += np.einsum("ijbc,kabc->ijka", tau0, u[o, v, v, v], optimize=True)

    tau16 = np.zeros((N, N, N, M))

    tau16 += np.einsum("kjia->ijka", tau14, optimize=True)

    tau14 = None

    tau41 = np.zeros((N, N, N, N))

    tau41 += np.einsum("lkab,jiab->ijkl", tau0, u[o, o, v, v], optimize=True)

    r1 = np.zeros((M, N))

    r1 += np.einsum("jibc,jabc->ai", tau0, u[o, v, v, v], optimize=True) / 2

    tau1 = np.zeros((N, N, N, M))

    tau1 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau2 = np.zeros((N, N, N, M))

    tau2 -= np.einsum("ikja->ijka", tau1, optimize=True)

    tau37 = np.zeros((N, N, N, M))

    tau37 += np.einsum("abjl,ilkb->ijka", t2, tau1, optimize=True)

    tau1 = None

    tau38 = np.zeros((N, N, N, M))

    tau38 -= np.einsum("ikja->ijka", tau37, optimize=True)

    tau37 = None

    tau2 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bajk,ijkb->ai", t2, tau2, optimize=True) / 2

    tau2 = None

    tau3 = np.zeros((N, M))

    tau3 += np.einsum("bj,ijab->ia", t1, u[o, o, v, v], optimize=True)

    tau4 = np.zeros((N, M))

    tau4 += np.einsum("ia->ia", tau3, optimize=True)

    tau3 = None

    tau4 += np.einsum("ia->ia", f[o, v], optimize=True)

    tau15 = np.zeros((N, N, N, M))

    tau15 += np.einsum("kb,baij->ijka", tau4, t2, optimize=True)

    tau16 += 2 * np.einsum("kjia->ijka", tau15, optimize=True)

    tau15 = None

    tau26 = np.zeros((N, N))

    tau26 += np.einsum("ai,ja->ij", t1, tau4, optimize=True)

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("ik,abkj->ijab", tau26, t2, optimize=True)

    tau26 = None

    tau31 = np.zeros((N, N, M, M))

    tau31 -= 2 * np.einsum("ijba->ijab", tau27, optimize=True)

    tau27 = None

    r1 += np.einsum("jb,baji->ai", tau4, t2, optimize=True)

    tau4 = None

    tau5 = np.zeros((N, N))

    tau5 -= np.einsum("ak,kija->ij", t1, u[o, o, o, v], optimize=True)

    tau6 += 2 * np.einsum("ij->ij", tau5, optimize=True)

    tau29 = np.zeros((N, N))

    tau29 += 2 * np.einsum("ij->ij", tau5, optimize=True)

    tau5 = None

    tau6 += 2 * np.einsum("ij->ij", f[o, o], optimize=True)

    tau6 += 2 * np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    r1 -= np.einsum("aj,ji->ai", t1, tau6, optimize=True) / 2

    tau6 = None

    tau7 = np.zeros((N, N, M, M))

    tau7 += np.einsum("ac,bcij->ijab", f[v, v], t2, optimize=True)

    tau18 = np.zeros((N, N, M, M))

    tau18 -= 2 * np.einsum("jiab->ijab", tau7, optimize=True)

    tau7 = None

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("acik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum("acik,jkbc->ijab", t2, tau8, optimize=True)

    tau8 = None

    tau18 += 2 * np.einsum("ijba->ijab", tau9, optimize=True)

    tau9 = None

    tau10 = np.zeros((M, M))

    tau10 -= np.einsum("ci,iacb->ab", t1, u[o, v, v, v], optimize=True)

    tau12 = np.zeros((M, M))

    tau12 += 2 * np.einsum("ab->ab", tau10, optimize=True)

    tau10 = None

    tau11 = np.zeros((M, M))

    tau11 -= np.einsum("acji,jicb->ab", t2, u[o, o, v, v], optimize=True)

    tau12 += np.einsum("ab->ab", tau11, optimize=True)

    tau11 = None

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum("ac,cbij->ijab", tau12, t2, optimize=True)

    tau12 = None

    tau18 += np.einsum("jiba->ijab", tau13, optimize=True)

    tau13 = None

    tau16 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau17 = np.zeros((N, N, M, M))

    tau17 += np.einsum("bk,kija->ijab", t1, tau16, optimize=True)

    tau16 = None

    tau18 += np.einsum("jiba->ijab", tau17, optimize=True)

    tau17 = None

    r2 = np.zeros((M, M, N, N))

    r2 -= np.einsum("ijab->abij", tau18, optimize=True) / 2

    r2 += np.einsum("ijba->abij", tau18, optimize=True) / 2

    tau18 = None

    tau19 = np.zeros((N, N, M, M))

    tau19 += np.einsum("ki,abjk->ijab", f[o, o], t2, optimize=True)

    tau24 = np.zeros((N, N, M, M))

    tau24 += np.einsum("ijba->ijab", tau19, optimize=True)

    tau19 = None

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum("ci,abjc->ijab", t1, u[v, v, o, v], optimize=True)

    tau24 -= np.einsum("ijba->ijab", tau20, optimize=True)

    tau20 = None

    tau21 = np.zeros((N, N, N, N))

    tau21 += np.einsum("ai,jkla->ijkl", t1, u[o, o, o, v], optimize=True)

    tau22 = np.zeros((N, N, N, M))

    tau22 += np.einsum("al,ijlk->ijka", t1, tau21, optimize=True)

    tau23 = np.zeros((N, N, M, M))

    tau23 -= np.einsum("bk,ikja->ijab", t1, tau22, optimize=True)

    tau22 = None

    tau24 -= np.einsum("ijba->ijab", tau23, optimize=True)

    tau23 = None

    r2 -= np.einsum("ijab->abij", tau24, optimize=True)

    r2 += np.einsum("jiab->abij", tau24, optimize=True)

    tau24 = None

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum("ablk,ilkj->ijab", t2, tau21, optimize=True)

    tau21 = None

    tau31 -= np.einsum("ijba->ijab", tau25, optimize=True)

    tau25 = None

    tau28 = np.zeros((N, N))

    tau28 -= np.einsum("baik,kjba->ij", t2, u[o, o, v, v], optimize=True)

    tau29 += np.einsum("ji->ij", tau28, optimize=True)

    tau28 = None

    tau30 = np.zeros((N, N, M, M))

    tau30 += np.einsum("ki,abkj->ijab", tau29, t2, optimize=True)

    tau29 = None

    tau31 += np.einsum("jiba->ijab", tau30, optimize=True)

    tau30 = None

    r2 += np.einsum("ijba->abij", tau31, optimize=True) / 2

    r2 -= np.einsum("jiba->abij", tau31, optimize=True) / 2

    tau31 = None

    tau32 = np.zeros((N, N, M, M))

    tau32 += np.einsum("acik,kbjc->ijab", t2, u[o, v, o, v], optimize=True)

    tau40 = np.zeros((N, N, M, M))

    tau40 -= np.einsum("ijab->ijab", tau32, optimize=True)

    tau32 = None

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("ci,jabc->ijab", t1, u[o, v, v, v], optimize=True)

    tau34 = np.zeros((N, N, M, M))

    tau34 -= np.einsum("acjk,ikbc->ijab", t2, tau33, optimize=True)

    tau33 = None

    tau40 += np.einsum("ijab->ijab", tau34, optimize=True)

    tau34 = None

    tau35 = np.zeros((N, N, N, M))

    tau35 += np.einsum("bi,jakb->ijka", t1, u[o, v, o, v], optimize=True)

    tau38 += np.einsum("ijka->ijka", tau35, optimize=True)

    tau35 = None

    tau36 = np.zeros((N, N, N, M))

    tau36 -= np.einsum("abil,ljkb->ijka", t2, u[o, o, o, v], optimize=True)

    tau38 += np.einsum("ijka->ijka", tau36, optimize=True)

    tau36 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum("bk,ikja->ijab", t1, tau38, optimize=True)

    tau38 = None

    tau40 += np.einsum("ijba->ijab", tau39, optimize=True)

    tau39 = None

    r2 += np.einsum("ijab->abij", tau40, optimize=True)

    r2 -= np.einsum("ijba->abij", tau40, optimize=True)

    r2 -= np.einsum("jiab->abij", tau40, optimize=True)

    r2 += np.einsum("jiba->abij", tau40, optimize=True)

    tau40 = None

    tau41 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    r2 += np.einsum("klba,klji->abij", tau0, tau41, optimize=True) / 4

    tau0 = None

    tau41 = None

    tau42 = np.zeros((N, N, M, M))

    tau42 -= np.einsum("baji->ijab", t2, optimize=True)

    tau42 += 2 * np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    r2 -= np.einsum("jicd,bacd->abij", tau42, u[v, v, v, v], optimize=True) / 2

    tau42 = None

    r1 -= np.einsum("bj,jaib->ai", t1, u[o, v, o, v], optimize=True)

    r1 += np.einsum("ab,bi->ai", f[v, v], t1, optimize=True)

    r1 += np.einsum("ai->ai", f[v, o], optimize=True)

    r2 += np.einsum("baji->abij", u[v, v, o, o], optimize=True)

    return r1, r2
