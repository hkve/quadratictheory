import numpy as np

from clusterfock.cc.energies.e_inter_ccsd_t1transformed import td_energy_addition

def t1transformed_qccsd_energy(t2, l1, l2, u, f, o, v):
    e = 0
    e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4
    e += td_energy_addition(t2, l1, l2, u, f, o, v)
    e += qccsd_energy_addition(t2, l1, l2, u, f, o, v)

    return e

def qccsd_energy_addition(t2, l1, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((M, M, M, M))

    tau0 += np.einsum(
        "abji,cdji->abcd", l2, t2, optimize=True
    )

    tau1 = zeros((M, M, M, M))

    tau1 += np.einsum(
        "afce,bedf->abcd", tau0, tau0, optimize=True
    )

    tau23 = zeros((N, N, M, M))

    tau23 -= np.einsum(
        "abcd,ijdc->ijab", tau1, u[o, o, v, v], optimize=True
    )

    tau1 = None

    tau6 = zeros((M, M, M, M))

    tau6 -= np.einsum(
        "eafc,befd->abcd", tau0, u[v, v, v, v], optimize=True
    )

    tau0 = None

    tau2 = zeros((N, N, M, M))

    tau2 += np.einsum(
        "caki,bcjk->ijab", l2, t2, optimize=True
    )

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum(
        "caki,kjcb->ijab", t2, tau2, optimize=True
    )

    tau4 = zeros((N, N, M, M))

    tau4 += np.einsum(
        "ikca,jkbc->ijab", tau3, u[o, o, v, v], optimize=True
    )

    tau23 += 16 * np.einsum(
        "bcjk,kica->ijab", l2, tau4, optimize=True
    )

    tau4 = None

    tau26 = zeros((N, N, N, N))

    tau26 -= 2 * np.einsum(
        "ljab,ikba->ijkl", tau3, u[o, o, v, v], optimize=True
    )

    tau38 = zeros((N, N, M, M))

    tau38 += 8 * np.einsum(
        "ijba->ijab", tau3, optimize=True
    )

    tau42 = zeros((N, N, N, N))

    tau42 += np.einsum(
        "baij,klba->ijkl", l2, tau3, optimize=True
    )

    tau3 = None

    tau6 += 4 * np.einsum(
        "ijab,jcid->abcd", tau2, u[o, v, o, v], optimize=True
    )

    tau10 = zeros((N, N, M, M))

    tau10 -= 4 * np.einsum(
        "ikca,kcjb->ijab", tau2, u[o, v, o, v], optimize=True
    )

    tau19 = zeros((N, M))

    tau19 -= 4 * np.einsum(
        "ijbc,jbca->ia", tau2, u[o, v, v, v], optimize=True
    )

    tau29 = zeros((N, N, N, N))

    tau29 += np.einsum(
        "ikab,jlba->ijkl", tau2, tau2, optimize=True
    )

    tau37 = zeros((N, N, N, M))

    tau37 += 4 * np.einsum(
        "lkab,ijlb->ijka", tau2, u[o, o, o, v], optimize=True
    )

    tau37 += 8 * np.einsum(
        "ikbc,jbac->ijka", tau2, u[o, v, v, v], optimize=True
    )

    tau39 = zeros((N, M))

    tau39 += 8 * np.einsum(
        "jicb,cajb->ia", tau2, u[v, v, o, v], optimize=True
    )

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum(
        "caki,kjcb->ijab", t2, u[o, o, v, v], optimize=True
    )

    tau6 += 4 * np.einsum(
        "ijac,ijbd->abcd", tau2, tau5, optimize=True
    )

    tau23 -= 2 * np.einsum(
        "cdji,bcda->ijab", l2, tau6, optimize=True
    )

    tau6 = None

    tau10 += 4 * np.einsum(
        "kjbc,kiac->ijab", tau2, tau5, optimize=True
    )

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum(
        "ijab->ijab", tau5, optimize=True
    )

    tau5 = None

    tau7 = zeros((M, M, M, M))

    tau7 += 2 * np.einsum(
        "badc->abcd", u[v, v, v, v], optimize=True
    )

    tau7 += np.einsum(
        "baji,jidc->abcd", t2, u[o, o, v, v], optimize=True
    )

    tau10 -= np.einsum(
        "ijcd,cadb->ijab", tau2, tau7, optimize=True
    )

    tau7 = None

    tau8 = zeros((N, N, M, M))

    tau8 += np.einsum(
        "dcij,dcab->ijab", l2, u[v, v, v, v], optimize=True
    )

    tau9 = zeros((N, N, M, M))

    tau9 -= np.einsum(
        "jiba->ijab", tau8, optimize=True
    )

    tau32 = zeros((N, N, M, M))

    tau32 -= 2 * np.einsum(
        "jiba->ijab", tau8, optimize=True
    )

    tau8 = None

    tau9 += 4 * np.einsum(
        "caki,jckb->ijab", l2, u[o, v, o, v], optimize=True
    )

    tau10 += np.einsum(
        "cakj,ikcb->ijab", t2, tau9, optimize=True
    )

    tau9 = None

    tau23 -= 4 * np.einsum(
        "caki,jkcb->ijab", l2, tau10, optimize=True
    )

    tau10 = None

    tau11 = zeros((N, N, N, M))

    tau11 -= np.einsum(
        "bi,bajk->ijka", l1, t2, optimize=True
    )

    tau19 += 2 * np.einsum(
        "ikjb,kjba->ia", tau11, u[o, o, v, v], optimize=True
    )

    tau37 += 4 * np.einsum(
        "ilkb,ljba->ijka", tau11, u[o, o, v, v], optimize=True
    )

    tau11 = None

    tau12 = zeros((N, N))

    tau12 += np.einsum(
        "baki,bakj->ij", l2, t2, optimize=True
    )

    tau19 += 2 * np.einsum(
        "ja,ij->ia", f[o, v], tau12, optimize=True
    )

    tau22 = zeros((N, M))

    tau22 += np.einsum(
        "jk,ikja->ia", tau12, u[o, o, o, v], optimize=True
    )

    tau40 = zeros((N, N, M, M))

    tau40 -= np.einsum(
        "ik,jkba->ijab", tau12, u[o, o, v, v], optimize=True
    )

    tau41 = zeros((N, N))

    tau41 -= 2 * np.einsum(
        "kl,likj->ij", tau12, u[o, o, o, o], optimize=True
    )

    tau13 = zeros((N, N, N, N))

    tau13 += np.einsum(
        "baij,bakl->ijkl", l2, t2, optimize=True
    )

    tau19 -= np.einsum(
        "jilk,lkja->ia", tau13, u[o, o, o, v], optimize=True
    )

    tau26 += 2 * np.einsum(
        "minj,nkml->ijkl", tau13, u[o, o, o, o], optimize=True
    )

    tau32 -= np.einsum(
        "ijlk,lkab->ijab", tau13, u[o, o, v, v], optimize=True
    )

    tau37 += np.einsum(
        "ijml,mlka->ijka", tau13, u[o, o, o, v], optimize=True
    )

    tau37 += 4 * np.einsum(
        "limk,jmla->ijka", tau13, u[o, o, o, v], optimize=True
    )

    tau38 -= np.einsum(
        "ablk,lkij->ijab", t2, tau13, optimize=True
    )

    tau39 += 2 * np.einsum(
        "lkji,jalk->ia", tau13, u[o, v, o, o], optimize=True
    )

    tau14 = zeros((N, N, N, M))

    tau14 += np.einsum(
        "ib,abjk->ijka", f[o, v], t2, optimize=True
    )

    tau17 = zeros((N, N, N, M))

    tau17 -= 2 * np.einsum(
        "ikja->ijka", tau14, optimize=True
    )

    tau36 = zeros((N, N, N, M))

    tau36 += 2 * np.einsum(
        "ikja->ijka", tau14, optimize=True
    )

    tau14 = None

    tau15 = zeros((N, N, N, M))

    tau15 -= np.einsum(
        "bali,ljkb->ijka", t2, u[o, o, o, v], optimize=True
    )

    tau17 += 4 * np.einsum(
        "jika->ijka", tau15, optimize=True
    )

    tau36 -= 2 * np.einsum(
        "jika->ijka", tau15, optimize=True
    )

    tau36 += 2 * np.einsum(
        "kija->ijka", tau15, optimize=True
    )

    tau15 = None

    tau16 = zeros((N, N, N, M))

    tau16 += np.einsum(
        "cbij,kacb->ijka", t2, u[o, v, v, v], optimize=True
    )

    tau17 += np.einsum(
        "kjia->ijka", tau16, optimize=True
    )

    tau36 -= np.einsum(
        "kjia->ijka", tau16, optimize=True
    )

    tau16 = None

    tau17 += 2 * np.einsum(
        "iakj->ijka", u[o, v, o, o], optimize=True
    )

    tau19 += np.einsum(
        "bajk,ijkb->ia", l2, tau17, optimize=True
    )

    tau17 = None

    tau18 = zeros((N, N))

    tau18 += 2 * np.einsum(
        "ij->ij", f[o, o], optimize=True
    )

    tau18 += np.einsum(
        "bakj,kiba->ij", t2, u[o, o, v, v], optimize=True
    )

    tau19 += 2 * np.einsum(
        "aj,ij->ia", l1, tau18, optimize=True
    )

    tau18 = None

    tau19 -= 4 * np.einsum(
        "bi,ba->ia", l1, f[v, v], optimize=True
    )

    tau19 -= 2 * np.einsum(
        "cbji,cbja->ia", l2, u[v, v, o, v], optimize=True
    )

    tau23 += 4 * np.einsum(
        "ai,jb->ijab", l1, tau19, optimize=True
    )

    tau19 = None

    tau20 = zeros((N, M))

    tau20 += np.einsum(
        "bj,baji->ia", l1, t2, optimize=True
    )

    tau22 += np.einsum(
        "jb,jiba->ia", tau20, u[o, o, v, v], optimize=True
    )

    tau20 = None

    tau21 = zeros((M, M))

    tau21 += np.einsum(
        "caji,cbji->ab", l2, t2, optimize=True
    )

    tau22 += np.einsum(
        "bc,ibac->ia", tau21, u[o, v, v, v], optimize=True
    )

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum(
        "ac,jibc->ijab", tau21, u[o, o, v, v], optimize=True
    )

    tau32 += np.einsum(
        "ijab->ijab", tau31, optimize=True
    )

    tau40 -= 4 * np.einsum(
        "ijab->ijab", tau31, optimize=True
    )

    tau31 = None

    tau33 = zeros((M, M))

    tau33 -= 2 * np.einsum(
        "cd,cadb->ab", tau21, u[v, v, v, v], optimize=True
    )

    tau37 -= 2 * np.einsum(
        "ab,ijkb->ijka", tau21, u[o, o, o, v], optimize=True
    )

    tau38 -= 4 * np.einsum(
        "cb,caij->ijab", tau21, t2, optimize=True
    )

    tau39 += np.einsum(
        "ijbc,jabc->ia", tau38, u[o, v, v, v], optimize=True
    )

    tau38 = None

    tau39 += 4 * np.einsum(
        "bc,abic->ia", tau21, u[v, v, o, v], optimize=True
    )

    tau41 += 4 * np.einsum(
        "ab,iajb->ij", tau21, u[o, v, o, v], optimize=True
    )

    tau22 -= 2 * np.einsum(
        "bj,ibja->ia", l1, u[o, v, o, v], optimize=True
    )

    tau23 -= 8 * np.einsum(
        "bj,ia->ijab", l1, tau22, optimize=True
    )

    tau22 = None

    e = 0

    e -= np.einsum(
        "abij,ijab->", t2, tau23, optimize=True
    ) / 16

    tau23 = None

    tau24 = zeros((N, N, N, N))

    tau24 += np.einsum(
        "baij,klba->ijkl", t2, u[o, o, v, v], optimize=True
    )

    tau26 += np.einsum(
        "imnl,mjnk->ijkl", tau13, tau24, optimize=True
    )

    tau30 = zeros((N, N, N, N))

    tau30 += np.einsum(
        "lkji->ijkl", tau24, optimize=True
    )

    tau24 = None

    tau25 -= np.einsum(
        "jaib->ijab", u[o, v, o, v], optimize=True
    )

    tau26 -= 8 * np.einsum(
        "ilab,jkab->ijkl", tau2, tau25, optimize=True
    )

    tau2 = None

    e -= np.einsum(
        "ijkl,ljki->", tau13, tau26, optimize=True
    ) / 16

    tau26 = None

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum(
        "caki,kjcb->ijab", l2, tau25, optimize=True
    )

    tau25 = None

    tau28 = zeros((N, N, N, N))

    tau28 -= np.einsum(
        "abkj,ilba->ijkl", t2, tau27, optimize=True
    )

    e += np.einsum(
        "ijkl,lijk->", tau13, tau28, optimize=True
    ) / 4

    tau28 = None

    tau13 = None

    tau32 -= 8 * np.einsum(
        "ijab->ijab", tau27, optimize=True
    )

    tau33 -= np.einsum(
        "caij,ijcb->ab", t2, tau32, optimize=True
    )

    tau32 = None

    e -= np.einsum(
        "ab,ab->", tau21, tau33, optimize=True
    ) / 16

    tau33 = None

    tau21 = None

    tau40 -= 8 * np.einsum(
        "ijba->ijab", tau27, optimize=True
    )

    tau27 = None

    tau30 += 2 * np.einsum(
        "jilk->ijkl", u[o, o, o, o], optimize=True
    )

    tau37 -= np.einsum(
        "al,jilk->ijka", l1, tau30, optimize=True
    )

    tau40 += np.einsum(
        "bakl,jikl->ijab", l2, tau30, optimize=True
    )

    e -= np.einsum(
        "ijkl,lkji->", tau29, tau30, optimize=True
    ) / 4

    tau29 = None

    e -= np.einsum(
        "lkij,ijkl->", tau30, tau42, optimize=True
    ) / 8

    tau42 = None

    tau30 = None

    tau34 = zeros((N, M, M, M))

    tau34 -= np.einsum(
        "di,dabc->iabc", l1, u[v, v, v, v], optimize=True
    )

    tau39 += 2 * np.einsum(
        "cbij,jacb->ia", t2, tau34, optimize=True
    )

    tau34 = None

    tau35 = zeros((N, M, M, M))

    tau35 -= np.einsum(
        "baic->iabc", u[v, v, o, v], optimize=True
    )

    tau35 += np.einsum(
        "jc,baij->iabc", f[o, v], t2, optimize=True
    )

    tau35 -= 2 * np.einsum(
        "dbji,jadc->iabc", t2, u[o, v, v, v], optimize=True
    )

    tau37 -= 2 * np.einsum(
        "bcji,kbca->ijka", l2, tau35, optimize=True
    )

    tau35 = None

    tau36 -= 2 * np.einsum(
        "iakj->ijka", u[o, v, o, o], optimize=True
    )

    tau37 -= 4 * np.einsum(
        "balj,ilkb->ijka", l2, tau36, optimize=True
    )

    tau36 = None

    tau37 -= 8 * np.einsum(
        "bi,jbka->ijka", l1, u[o, v, o, v], optimize=True
    )

    tau39 -= np.einsum(
        "bajk,kjib->ia", t2, tau37, optimize=True
    )

    tau37 = None

    tau39 += 4 * np.einsum(
        "bj,abij->ia", l1, u[v, v, o, o], optimize=True
    )

    e += np.einsum(
        "ai,ia->", l1, tau39, optimize=True
    ) / 8

    tau39 = None

    tau40 += 8 * np.einsum(
        "ak,jikb->ijab", l1, u[o, o, o, v], optimize=True
    )

    tau40 += 4 * np.einsum(
        "ci,jcba->ijab", l1, u[o, v, v, v], optimize=True
    )

    tau41 -= np.einsum(
        "abkj,kiba->ij", t2, tau40, optimize=True
    )

    tau40 = None

    tau41 -= 8 * np.einsum(
        "ak,iakj->ij", l1, u[o, v, o, o], optimize=True
    )

    e -= np.einsum(
        "ij,ji->", tau12, tau41, optimize=True
    ) / 16

    tau41 = None

    tau12 = None

    return e