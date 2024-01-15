import numpy as np


def two_body_density_addition(rho, t2, l2, o, v):
    rho = two_body_density_oooo(rho, t2, l2, o, v)
    rho = two_body_density_oovv(rho, t2, l2, o, v)
    rho = two_body_density_ovov(rho, t2, l2, o, v)
    rho = two_body_density_vvvv(rho, t2, l2, o, v)

    return rho


def two_body_density_oooo(rho, t2, l2, o, v):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau10 = np.zeros((N, N, M, M))

    tau10 -= 4 * np.einsum("cbkj,kica->ijab", t2, tau1, optimize=True)

    rho[o, o, o, o] += np.einsum("ljba,kiab->ijkl", tau0, tau1, optimize=True)

    tau0 = None

    tau1 = None

    tau2 = np.zeros((N, N, M, M))

    tau2 += np.einsum("acki,cbkj->ijab", l2, t2, optimize=True)

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    rho[o, o, o, o] -= np.einsum("kjab,liba->ijkl", tau2, tau3, optimize=True)

    tau2 = None

    tau3 = None

    tau4 = np.zeros((N, N, N, N))

    tau4 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau5 = np.zeros((N, N, N, N))

    tau5 += np.einsum("mikn,njml->ijkl", tau4, tau4, optimize=True)

    tau4 = None

    tau8 = np.zeros((N, N, N, N))

    tau8 += np.einsum("ijkl->ijkl", tau5, optimize=True)

    tau5 = None

    tau6 = np.zeros((N, N))

    tau6 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau7 = np.zeros((N, N))

    tau7 += np.einsum("baki,bajk->ij", l2, t2, optimize=True)

    tau8 += np.einsum("il,jk->ijkl", tau6, tau7, optimize=True)

    tau6 = None

    rho[o, o, o, o] += np.einsum("klij->ijkl", tau8, optimize=True) / 4

    rho[o, o, o, o] -= np.einsum("lkij->ijkl", tau8, optimize=True) / 4

    tau8 = None

    tau10 += np.einsum("ki,abkj->ijab", tau7, t2, optimize=True)

    tau7 = None

    tau9 = np.zeros((N, N))

    tau9 += np.einsum("baik,bakj->ij", l2, t2, optimize=True)

    tau10 += np.einsum("kj,abik->ijab", tau9, t2, optimize=True)

    tau9 = None

    rho[o, o, o, o] += np.einsum("abkl,ijab->ijkl", l2, tau10, optimize=True) / 4

    tau10 = None

    return rho


def two_body_density_oovv(rho, t2, l2, o, v):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, N, N))

    tau0 += np.einsum("abij,abkl->ijkl", l2, t2, optimize=True)

    tau1 = np.zeros((N, N, N, N))

    tau1 += np.einsum("mikn,njml->ijkl", tau0, tau0, optimize=True)

    rho[o, o, v, v] -= np.einsum("ablk,klij->ijab", t2, tau1, optimize=True) / 4

    tau1 = None

    tau12 = np.zeros((N, N, M, M))

    tau12 += np.einsum("ablk,lkij->ijab", t2, tau0, optimize=True)

    tau18 = np.zeros((N, N, M, M))

    tau18 -= np.einsum("ijab->ijab", tau12, optimize=True)

    tau30 = np.zeros((N, N, M, M))

    tau30 -= np.einsum("ijab->ijab", tau12, optimize=True)

    tau12 = None

    tau2 = np.zeros((M, M))

    tau2 += np.einsum("caij,bcij->ab", l2, t2, optimize=True)

    tau41 = np.zeros((N, N, M, M))

    tau41 += np.einsum("ca,bcij->ijab", tau2, t2, optimize=True)

    tau3 = np.zeros((N, N))

    tau3 += np.einsum("abki,abkj->ij", l2, t2, optimize=True)

    tau17 = np.zeros((N, N, M, M))

    tau17 += np.einsum("kj,abki->ijab", tau3, t2, optimize=True)

    tau18 -= 2 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau17 = None

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum("ab,ij->ijab", tau2, tau3, optimize=True)

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum("acki,bcjk->ijab", l2, t2, optimize=True)

    tau5 = np.zeros((N, N, M, M))

    tau5 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("ikcb,kjac->ijab", tau4, tau5, optimize=True)

    tau5 = None

    tau20 += 4 * np.einsum("ijab->ijab", tau6, optimize=True)

    tau6 = None

    tau7 = np.zeros((M, M, M, M))

    tau7 += np.einsum("abij,cdij->abcd", l2, t2, optimize=True)

    tau37 = np.zeros((M, M, M, M))

    tau37 += np.einsum("afce,bedf->abcd", tau7, tau7, optimize=True)

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 += np.einsum("cabd,ijdc->ijab", tau7, tau8, optimize=True)

    tau8 = None

    tau20 += 2 * np.einsum("ijab->ijab", tau9, optimize=True)

    tau9 = None

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("caik,bcjk->ijab", l2, t2, optimize=True)

    tau11 = np.zeros((N, N, M, M))

    tau11 += np.einsum("kilj,lkab->ijab", tau0, tau10, optimize=True)

    tau20 -= 2 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau11 = None

    tau27 = np.zeros((N, N, M, M))

    tau27 += np.einsum("kijl,lkab->ijab", tau0, tau10, optimize=True)

    tau10 = None

    tau32 = np.zeros((N, N, M, M))

    tau32 -= 2 * np.einsum("ijab->ijab", tau27, optimize=True)

    tau27 = None

    tau13 = np.zeros((M, M))

    tau13 += np.einsum("acij,bcij->ab", l2, t2, optimize=True)

    tau14 = np.zeros((N, N, M, M))

    tau14 += np.einsum("ca,cbij->ijab", tau13, t2, optimize=True)

    tau18 += 2 * np.einsum("ijab->ijab", tau14, optimize=True)

    tau30 += 2 * np.einsum("ijab->ijab", tau14, optimize=True)

    tau14 = None

    tau37 -= np.einsum("ac,bd->abcd", tau13, tau13, optimize=True)

    tau41 += np.einsum("cb,acij->ijab", tau13, t2, optimize=True)

    tau13 = None

    tau15 = np.zeros((N, N, M, M))

    tau15 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau16 = np.zeros((N, N, M, M))

    tau16 += np.einsum("cbkj,kica->ijab", t2, tau15, optimize=True)

    tau18 += 4 * np.einsum("ijab->ijab", tau16, optimize=True)

    tau16 = None

    tau19 = np.zeros((N, N, M, M))

    tau19 += np.einsum("cbkj,kiac->ijab", l2, tau18, optimize=True)

    tau18 = None

    tau20 -= np.einsum("jiba->ijab", tau19, optimize=True)

    tau19 = None

    tau21 = np.zeros((N, N, M, M))

    tau21 += np.einsum("acik,kjcb->ijab", t2, tau20, optimize=True)

    tau20 = None

    tau34 = np.zeros((N, N, M, M))

    tau34 += np.einsum("ijab->ijab", tau21, optimize=True)

    tau21 = None

    tau28 = np.zeros((N, N, M, M))

    tau28 += np.einsum("cbik,kjca->ijab", t2, tau15, optimize=True)

    tau30 += 4 * np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau39 = np.zeros((N, N, M, M))

    tau39 -= 4 * np.einsum("acik,kjcb->ijab", t2, tau15, optimize=True)

    tau15 = None

    tau22 = np.zeros((N, N))

    tau22 += np.einsum("abki,abjk->ij", l2, t2, optimize=True)

    tau29 = np.zeros((N, N, M, M))

    tau29 += np.einsum("ki,abkj->ijab", tau22, t2, optimize=True)

    tau30 -= 2 * np.einsum("ijab->ijab", tau29, optimize=True)

    tau31 = np.zeros((N, N, M, M))

    tau31 += np.einsum("cbkj,ikac->ijab", l2, tau30, optimize=True)

    tau30 = None

    tau32 -= np.einsum("jiba->ijab", tau31, optimize=True)

    tau31 = None

    tau39 += np.einsum("ijab->ijab", tau29, optimize=True)

    tau29 = None

    tau32 += np.einsum("ab,ij->ijab", tau2, tau22, optimize=True)

    tau2 = None

    tau40 = np.zeros((N, N, N, N))

    tau40 -= 2 * np.einsum("ik,jl->ijkl", tau22, tau3, optimize=True)

    tau3 = None

    tau22 = None

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau24 = np.zeros((N, N, M, M))

    tau24 += np.einsum("ijdc,cabd->ijab", tau23, tau7, optimize=True)

    tau23 = None

    tau7 = None

    tau32 += 2 * np.einsum("ijab->ijab", tau24, optimize=True)

    tau24 = None

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau26 = np.zeros((N, N, M, M))

    tau26 += np.einsum("kjac,ikcb->ijab", tau25, tau4, optimize=True)

    tau4 = None

    tau32 += 4 * np.einsum("ijab->ijab", tau26, optimize=True)

    tau26 = None

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("acki,kjcb->ijab", t2, tau32, optimize=True)

    tau32 = None

    tau34 -= np.einsum("jiab->ijab", tau33, optimize=True)

    tau33 = None

    rho[o, o, v, v] -= np.einsum("ijab->ijab", tau34, optimize=True) / 4

    rho[o, o, v, v] += np.einsum("ijba->ijab", tau34, optimize=True) / 4

    tau34 = None

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau40 += 8 * np.einsum("ikab,jlba->ijkl", tau25, tau35, optimize=True)

    tau25 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 += np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    tau37 += 4 * np.einsum("ijac,jibd->abcd", tau35, tau36, optimize=True)

    tau35 = None

    rho[o, o, v, v] -= np.einsum("cdij,cdab->ijab", t2, tau37, optimize=True) / 4

    tau37 = None

    tau41 -= 4 * np.einsum("acik,kjcb->ijab", t2, tau36, optimize=True)

    tau36 = None

    rho[o, o, v, v] -= np.einsum("klij,klab->ijab", tau0, tau41, optimize=True) / 8

    tau0 = None

    tau41 = None

    tau38 = np.zeros((N, N))

    tau38 += np.einsum("abik,abkj->ij", l2, t2, optimize=True)

    tau39 += np.einsum("kj,abik->ijab", tau38, t2, optimize=True)

    tau38 = None

    tau40 += np.einsum("abij,klab->ijkl", l2, tau39, optimize=True)

    tau39 = None

    rho[o, o, v, v] += np.einsum("abkl,klij->ijab", t2, tau40, optimize=True) / 8

    tau40 = None

    return rho


def two_body_density_ovov(rho, t2, l2, o, v):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("acki,bcjk->ijab", l2, t2, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    rho[o, v, o, v] += np.einsum("kiac,jkcb->iajb", tau0, tau1, optimize=True)

    tau0 = None

    tau1 = None

    tau2 = np.zeros((N, N, M, M))

    tau2 += np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    tau3 = np.zeros((N, N, N, N))

    tau3 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau9 = np.zeros((N, N, M, M))

    tau9 -= np.einsum("ablk,lkij->ijab", t2, tau3, optimize=True)

    rho[o, v, o, v] -= np.einsum("klab,ljik->iajb", tau2, tau3, optimize=True) / 2

    tau2 = None

    tau3 = None

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau5 = np.zeros((M, M, M, M))

    tau5 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    rho[o, v, o, v] -= np.einsum("jidc,acbd->iajb", tau4, tau5, optimize=True) / 2

    tau4 = None

    tau5 = None

    tau6 = np.zeros((N, N))

    tau6 += np.einsum("abki,abjk->ij", l2, t2, optimize=True)

    tau9 -= 2 * np.einsum("ki,abkj->ijab", tau6, t2, optimize=True)

    tau7 = np.zeros((M, M))

    tau7 += np.einsum("acji,bcji->ab", l2, t2, optimize=True)

    tau9 += 2 * np.einsum("ca,cbij->ijab", tau7, t2, optimize=True)

    rho[o, v, o, v] += np.einsum("ji,ab->iajb", tau6, tau7, optimize=True) / 4

    tau6 = None

    tau7 = None

    tau8 = np.zeros((N, N, M, M))

    tau8 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau9 += 4 * np.einsum("cbik,kjca->ijab", t2, tau8, optimize=True)

    tau8 = None

    rho[o, v, o, v] -= np.einsum("ackj,ikbc->iajb", l2, tau9, optimize=True) / 4

    tau9 = None

    rho[o, v, v, o] = -rho[o, v, o, v].transpose(0, 1, 3, 2)
    rho[v, o, o, v] = -rho[o, v, o, v].transpose(1, 0, 2, 3)
    rho[v, o, v, o] = rho[o, v, o, v].transpose(1, 0, 3, 2)

    return rho


def two_body_density_vvvv(rho, t2, l2, o, v):
    M, _, N, _ = t2.shape

    tau0 = np.zeros((N, N, M, M))

    tau0 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("acki,kjcb->ijab", t2, tau0, optimize=True)

    rho[v, v, v, v] += np.einsum("abij,ijcd->abcd", l2, tau1, optimize=True)

    tau1 = None

    tau2 = np.zeros((N, N, M, M))

    tau2 += np.einsum("acki,bcjk->ijab", l2, t2, optimize=True)

    rho[v, v, v, v] += np.einsum("ijad,jibc->abcd", tau0, tau2, optimize=True)

    tau0 = None

    tau2 = None

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum("acik,bcjk->ijab", l2, t2, optimize=True)

    rho[v, v, v, v] -= np.einsum("ijac,jibd->abcd", tau3, tau4, optimize=True)

    tau3 = None

    tau4 = None

    tau5 = np.zeros((M, M))

    tau5 += np.einsum("acij,bcij->ab", l2, t2, optimize=True)

    tau8 = np.zeros((M, M, M, M))

    tau8 += np.einsum("ac,bd->abcd", tau5, tau5, optimize=True)

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("cb,acij->ijab", tau5, t2, optimize=True)

    tau5 = None

    tau6 = np.zeros((M, M, M, M))

    tau6 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau7 = np.zeros((M, M, M, M))

    tau7 += np.einsum("aecf,bfde->abcd", tau6, tau6, optimize=True)

    tau6 = None

    tau8 -= np.einsum("badc->abcd", tau7, optimize=True)

    tau7 = None

    rho[v, v, v, v] -= np.einsum("abdc->abcd", tau8, optimize=True) / 4

    rho[v, v, v, v] += np.einsum("abcd->abcd", tau8, optimize=True) / 4

    tau8 = None

    tau9 = np.zeros((M, M))

    tau9 += np.einsum("caij,bcij->ab", l2, t2, optimize=True)

    tau10 += np.einsum("ca,bcij->ijab", tau9, t2, optimize=True)

    tau9 = None

    rho[v, v, v, v] -= np.einsum("abij,ijcd->abcd", l2, tau10, optimize=True) / 4

    tau10 = None

    return rho
