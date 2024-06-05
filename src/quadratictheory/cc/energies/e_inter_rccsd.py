import numpy as np


def td_energy_addition_restricted(t1, t2, l1, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N))

    tau0 += np.einsum("ia,aj->ij", f[o, v], t1, optimize=True)

    tau1 = zeros((N, N))

    tau1 += np.einsum("abki,abkj->ij", l2, t2, optimize=True)

    tau12 = zeros((N, N))

    tau12 += 2 * np.einsum("ij->ij", tau1, optimize=True)

    e = 0

    e -= 2 * np.einsum("ij,ji->", tau0, tau1, optimize=True)

    tau1 = None

    tau2 = zeros((N, M))

    tau2 += np.einsum("bakj,kjib->ia", t2, u[o, o, o, v], optimize=True)

    e += np.einsum("ai,ia->", l1, tau2, optimize=True)

    tau2 = None

    tau3 = zeros((N, M))

    tau3 += np.einsum("abkj,kjib->ia", t2, u[o, o, o, v], optimize=True)

    e -= 2 * np.einsum("ai,ia->", l1, tau3, optimize=True)

    tau3 = None

    tau4 = zeros((N, M))

    tau4 += np.einsum("bcji,jacb->ia", t2, u[o, v, v, v], optimize=True)

    e -= np.einsum("ai,ia->", l1, tau4, optimize=True)

    tau4 = None

    tau5 = zeros((N, M))

    tau5 += np.einsum("cbji,jacb->ia", t2, u[o, v, v, v], optimize=True)

    e += 2 * np.einsum("ai,ia->", l1, tau5, optimize=True)

    tau5 = None

    tau6 = zeros((M, M))

    tau6 += np.einsum("acji,bcji->ab", l2, t2, optimize=True)

    tau37 = zeros((M, M))

    tau37 += 2 * np.einsum("ab->ab", tau6, optimize=True)

    e += 2 * np.einsum("ba,ba->", f[v, v], tau6, optimize=True)

    tau7 = zeros((N, N))

    tau7 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau12 += np.einsum("ij->ij", tau7, optimize=True)

    tau13 = zeros((N, M))

    tau13 += np.einsum("aj,ji->ia", t1, tau12, optimize=True)

    e -= np.einsum("ji,ij->", tau0, tau7, optimize=True)

    tau0 = None

    tau7 = None

    tau8 = zeros((N, N, M, M))

    tau8 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau8 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau43 = zeros((N, N))

    tau43 += np.einsum("abkj,kiab->ij", t2, tau8, optimize=True)

    tau9 = zeros((N, N, M, M))

    tau9 += np.einsum("abji->ijab", t2, optimize=True)

    tau9 -= np.einsum("baji->ijab", t2, optimize=True)

    tau10 = zeros((N, N, M, M))

    tau10 -= np.einsum("caki,kjcb->ijab", l2, tau9, optimize=True)

    tau9 = None

    tau16 = zeros((N, N, M, M))

    tau16 -= 2 * np.einsum("caki,kjcb->ijab", t2, tau10, optimize=True)

    tau10 = None

    tau11 = zeros((N, N, N, M))

    tau11 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau13 += 2 * np.einsum("bakj,kjib->ia", t2, tau11, optimize=True)

    tau16 += np.einsum("bj,ia->ijab", t1, tau13, optimize=True)

    tau13 = None

    tau26 = zeros((N, N, N, N))

    tau26 += np.einsum("al,jika->ijkl", t1, tau11, optimize=True)

    tau39 = zeros((N, M))

    tau39 += 2 * np.einsum("likj,kjla->ia", tau26, u[o, o, o, v], optimize=True)

    tau26 = None

    tau36 = zeros((N, N, N, M))

    tau36 += 2 * np.einsum("abkl,lijb->ijka", t2, tau11, optimize=True)

    tau39 -= 2 * np.einsum("kijb,jbak->ia", tau11, u[o, v, v, o], optimize=True)

    tau39 -= 2 * np.einsum("ikjb,jbka->ia", tau11, u[o, v, o, v], optimize=True)

    tau11 = None

    tau14 = zeros((N, N, M, M))

    tau14 -= np.einsum("abji->ijab", t2, optimize=True)

    tau14 += 2 * np.einsum("baji->ijab", t2, optimize=True)

    tau15 = zeros((N, M))

    tau15 += np.einsum("bj,jiba->ia", l1, tau14, optimize=True)

    tau16 -= np.einsum("ai,jb->ijab", t1, tau15, optimize=True)

    tau15 = None

    e -= np.einsum("jiba,ijab->", tau16, tau8, optimize=True)

    tau8 = None

    tau16 = None

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("caki,kjcb->ijab", l2, tau14, optimize=True)

    tau39 += 2 * np.einsum("ijbc,jbca->ia", tau31, u[o, v, v, v], optimize=True)

    tau31 = None

    tau17 = zeros((N, N, M, M))

    tau17 += 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau17 -= np.einsum("acki,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("caki,kjcb->ijab", l2, tau17, optimize=True)

    tau17 = None

    tau18 = zeros((N, N, M, M))

    tau18 += 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau18 -= np.einsum("acki,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau19 += np.einsum("acki,kjcb->ijab", l2, tau18, optimize=True)

    tau18 = None

    tau19 += np.einsum("ai,jb->ijab", l1, f[o, v], optimize=True)

    e -= np.einsum("abji,ijab->", t2, tau19, optimize=True)

    tau19 = None

    tau20 = zeros((N, N, M, M))

    tau20 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau20 -= np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau21 = zeros((N, N, M, M))

    tau21 += 2 * np.einsum("caki,jkcb->ijab", l2, tau20, optimize=True)

    tau39 += np.einsum("bj,ijba->ia", l1, tau20, optimize=True)

    tau20 = None

    tau21 += 2 * np.einsum("ai,jb->ijab", l1, f[o, v], optimize=True)

    tau21 += np.einsum("cdji,dcab->ijab", l2, u[v, v, v, v], optimize=True)

    e += np.einsum("baji,ijab->", t2, tau21, optimize=True)

    tau21 = None

    tau22 = zeros((N, N, N, N))

    tau22 += np.einsum("abij,abkl->ijkl", l2, t2, optimize=True)

    tau36 += np.einsum("al,ilkj->ijka", t1, tau22, optimize=True)

    tau39 += 2 * np.einsum("jilk,lkja->ia", tau22, u[o, o, o, v], optimize=True)

    tau23 = zeros((N, N, N, N))

    tau23 += np.einsum("baij,lkab->ijkl", t2, u[o, o, v, v], optimize=True)

    tau24 = zeros((N, N, N, N))

    tau24 += np.einsum("lkji->ijkl", tau23, optimize=True)

    tau30 = zeros((N, N, N, M))

    tau30 -= np.einsum("al,kjil->ijka", t1, tau23, optimize=True)

    tau23 = None

    tau24 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    e += np.einsum("ijkl,lkji->", tau22, tau24, optimize=True)

    tau24 = None

    tau22 = None

    tau25 = zeros((N, N, N, M))

    tau25 += np.einsum("bi,bakj->ijka", l1, t2, optimize=True)

    tau36 -= 2 * np.einsum("ijka->ijka", tau25, optimize=True)

    tau39 += np.einsum("ikjb,jkab->ia", tau36, u[o, o, v, v], optimize=True)

    tau36 = None

    tau39 += np.einsum("ijkb,kjba->ia", tau25, u[o, o, v, v], optimize=True)

    tau25 = None

    tau27 = zeros((N, N, N, M))

    tau27 += np.einsum("bi,jkab->ijka", t1, u[o, o, v, v], optimize=True)

    tau28 = zeros((N, N, N, M))

    tau28 += np.einsum("ijka->ijka", tau27, optimize=True)

    tau34 = zeros((N, N, N, N))

    tau34 += np.einsum("al,kjia->ijkl", t1, tau27, optimize=True)

    tau27 = None

    tau28 += np.einsum("kjia->ijka", u[o, o, o, v], optimize=True)

    tau30 += 2 * np.einsum("ljba,klib->ijka", tau14, tau28, optimize=True)

    tau14 = None

    tau35 = zeros((N, N, N, M))

    tau35 += 2 * np.einsum("abli,kjlb->ijka", t2, tau28, optimize=True)

    tau28 = None

    tau29 = zeros((N, N, M, M))

    tau29 += np.einsum("baji->ijab", t2, optimize=True)

    tau29 += np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau30 += 2 * np.einsum("kjbc,iabc->ijka", tau29, u[o, v, v, v], optimize=True)

    tau29 = None

    tau30 += 2 * np.einsum("iakj->ijka", u[o, v, o, o], optimize=True)

    tau30 += 2 * np.einsum("ib,abjk->ijka", f[o, v], t2, optimize=True)

    tau30 -= 2 * np.einsum("balj,likb->ijka", t2, u[o, o, o, v], optimize=True)

    tau39 -= np.einsum("bajk,ijkb->ia", l2, tau30, optimize=True)

    tau30 = None

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau32 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    tau39 -= 2 * np.einsum("ijbc,jbac->ia", tau32, u[o, v, v, v], optimize=True)

    tau32 = None

    tau33 = zeros((N, M, M, M))

    tau33 += 2 * np.einsum("abic->iabc", u[v, v, o, v], optimize=True)

    tau33 += np.einsum("di,bacd->iabc", t1, u[v, v, v, v], optimize=True)

    tau39 += np.einsum("bcji,jbca->ia", l2, tau33, optimize=True)

    tau33 = None

    tau34 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau35 += np.einsum("al,ljki->ijka", t1, tau34, optimize=True)

    tau34 = None

    tau39 += np.einsum("abjk,jikb->ia", l2, tau35, optimize=True)

    tau35 = None

    tau37 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau38 = zeros((N, M, M, M))

    tau38 += 2 * np.einsum("iabc->iabc", u[o, v, v, v], optimize=True)

    tau38 -= np.einsum("iacb->iabc", u[o, v, v, v], optimize=True)

    tau39 += np.einsum("bc,ibac->ia", tau37, tau38, optimize=True)

    tau37 = None

    tau38 = None

    tau39 += np.einsum("bi,ba->ia", l1, f[v, v], optimize=True)

    e += np.einsum("ai,ia->", t1, tau39, optimize=True)

    tau39 = None

    tau40 = zeros((N, N, M, M))

    tau40 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau40 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau41 = zeros((M, M))

    tau41 += np.einsum("acij,ijcb->ab", t2, tau40, optimize=True)

    tau40 = None

    e -= 2 * np.einsum("ab,ab->", tau41, tau6, optimize=True)

    tau41 = None

    tau6 = None

    tau42 = zeros((N, N, N, M))

    tau42 -= np.einsum("ijka->ijka", u[o, o, o, v], optimize=True)

    tau42 += 2 * np.einsum("jika->ijka", u[o, o, o, v], optimize=True)

    tau43 += np.einsum("ak,kija->ij", t1, tau42, optimize=True)

    tau42 = None

    tau43 += np.einsum("ij->ij", f[o, o], optimize=True)

    e -= np.einsum("ij,ji->", tau12, tau43, optimize=True)

    tau43 = None

    tau12 = None

    e += np.einsum("ai,ai->", l1, f[v, o], optimize=True)

    e += np.einsum("abji,abji->", l2, u[v, v, o, o], optimize=True)

    return e
