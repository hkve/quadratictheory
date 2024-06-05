import numpy as np


def one_body_density_addition(rho, t1, t2, l1, l2, o, v):
    rho = one_body_density_oo(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_ov(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_vv(rho, t1, t2, l1, l2, o, v)

    return rho


def two_body_density_addition(rho, t1, t2, l1, l2, o, v):
    rho = two_body_density_oooo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_ooov(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_oovv(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_ovoo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_ovov(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vovv(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vvoo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vvvo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vvvv(rho, t1, t2, l1, l2, o, v)

    return rho


def one_body_density_oo(rho, t1, t2, l1, l2, o, v):
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    M, N = t1.shape

    tau0 = zeros((N, M))

    tau0 += np.einsum("bj,baij->ia", l1, t2, optimize=True)

    rho[o, o] += np.einsum("aj,ia->ij", l1, tau0, optimize=True)

    return rho


def one_body_density_ov(rho, t1, t2, l1, l2, o, v):
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    M, N = t1.shape

    tau0 = zeros((N, M))

    tau0 += np.einsum("bj,abij->ia", l1, t2, optimize=True)

    tau1 = zeros((N, N))

    tau1 += np.einsum("ai,ja->ij", l1, tau0, optimize=True)

    tau0 = None

    rho[o, v] -= np.einsum("aj,ji->ia", t1, tau1, optimize=True)

    tau1 = None

    tau2 = zeros((N, N, N, N))

    tau2 += np.einsum("abij,abkl->ijkl", l2, t2, optimize=True)

    tau5 = zeros((N, N, N, M))

    tau5 += np.einsum("al,ijkl->ijka", l1, tau2, optimize=True)

    tau2 = None

    tau3 = zeros((N, M))

    tau3 += np.einsum("bj,baij->ia", l1, t2, optimize=True)

    tau5 += 2 * np.einsum("kb,baij->ijka", tau3, l2, optimize=True)

    tau3 = None

    tau4 = zeros((N, N))

    tau4 += np.einsum("abki,abjk->ij", l2, t2, optimize=True)

    tau5 -= 2 * np.einsum("ai,jk->ijka", l1, tau4, optimize=True)

    tau4 = None

    rho[o, v] += np.einsum("abjk,jkib->ia", t2, tau5, optimize=True) / 4

    tau5 = None

    tau6 = zeros((N, N))

    tau6 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau8 = zeros((N, N, N, M))

    tau8 += np.einsum("ai,jk->ijka", l1, tau6, optimize=True)

    tau6 = None

    tau7 = zeros((N, N, N, M))

    tau7 += np.einsum("bi,bajk->ijka", l1, t2, optimize=True)

    tau8 += np.einsum("bali,jklb->ijka", l2, tau7, optimize=True)

    tau7 = None

    rho[o, v] -= np.einsum("abkj,jkib->ia", t2, tau8, optimize=True)

    tau8 = None

    return rho


def one_body_density_vv(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M))

    tau0 += np.einsum("bj,abij->ia", l1, t2, optimize=True)

    rho[v, v] += np.einsum("ai,ib->ab", l1, tau0, optimize=True)

    return rho


def two_body_density_oooo(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, M))

    tau0 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    rho[o, o, o, o] -= np.einsum("al,kija->ijkl", l1, tau0, optimize=True)

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum("caik,cbjk->ijab", l2, t2, optimize=True)

    tau2 = zeros((N, N, M, M))

    tau2 -= np.einsum("cbjk,kica->ijab", t2, tau1, optimize=True)

    rho[o, o, o, o] += np.einsum("balk,jiba->ijkl", l2, tau2, optimize=True)

    tau2 = None

    tau10 = zeros((N, N, N, N))

    tau10 += np.einsum("ikab,jlba->ijkl", tau1, tau1, optimize=True)

    tau1 = None

    tau16 = zeros((N, N, N, N))

    tau16 -= 4 * np.einsum("ijkl->ijkl", tau10, optimize=True)

    tau10 = None

    tau3 = zeros((N, N))

    tau3 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 -= np.einsum("ki,bajk->ijab", tau3, t2, optimize=True)

    tau14 = zeros((N, N, M, M))

    tau14 += 2 * np.einsum("ijba->ijab", tau12, optimize=True)

    tau12 = None

    tau16 += 4 * np.einsum("ik,jl->ijkl", tau3, tau3, optimize=True)

    tau4 = zeros((N, N))

    tau4 += np.einsum("baik,bajk->ij", l2, t2, optimize=True)

    tau7 = zeros((N, N, N, N))

    tau7 -= np.einsum("ik,jl->ijkl", tau3, tau4, optimize=True)

    tau3 = None

    tau13 = zeros((N, N, M, M))

    tau13 -= np.einsum("kj,baik->ijab", tau4, t2, optimize=True)

    tau14 -= np.einsum("ijba->ijab", tau13, optimize=True)

    tau13 = None

    tau16 += np.einsum("ik,jl->ijkl", tau4, tau4, optimize=True)

    tau4 = None

    tau5 = zeros((N, N, N, M))

    tau5 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau6 = zeros((N, N, N, N))

    tau6 += np.einsum("ilma,jmka->ijkl", tau0, tau5, optimize=True)

    tau0 = None

    tau5 = None

    tau7 += 2 * np.einsum("ijkl->ijkl", tau6, optimize=True)

    tau6 = None

    rho[o, o, o, o] -= np.einsum("klij->ijkl", tau7, optimize=True) / 2

    rho[o, o, o, o] += np.einsum("klji->ijkl", tau7, optimize=True) / 2

    rho[o, o, o, o] += np.einsum("lkij->ijkl", tau7, optimize=True) / 2

    rho[o, o, o, o] -= np.einsum("lkji->ijkl", tau7, optimize=True) / 2

    tau7 = None

    tau8 = zeros((N, N, N, N))

    tau8 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau9 = zeros((N, N, N, N))

    tau9 += np.einsum("imln,jnkm->ijkl", tau8, tau8, optimize=True)

    tau8 = None

    tau16 += np.einsum("ijkl->ijkl", tau9, optimize=True)

    tau9 = None

    tau11 = zeros((N, M))

    tau11 -= np.einsum("bj,baij->ia", l1, t2, optimize=True)

    tau14 += 4 * np.einsum("bi,ja->ijab", t1, tau11, optimize=True)

    tau15 = zeros((N, N, N, N))

    tau15 += np.einsum("abij,klab->ijkl", l2, tau14, optimize=True)

    tau14 = None

    tau16 += np.einsum("jikl->ijkl", tau15, optimize=True)

    tau15 = None

    rho[o, o, o, o] -= np.einsum("lkij->ijkl", tau16, optimize=True) / 4

    rho[o, o, o, o] += np.einsum("lkji->ijkl", tau16, optimize=True) / 4

    tau16 = None

    tau17 = zeros((N, N))

    tau17 += np.einsum("ai,ja->ij", l1, tau11, optimize=True)

    tau11 = None

    I = np.eye(N, dtype=dtype)

    rho[o, o, o, o] -= np.einsum("ik,lj->ijkl", I, tau17, optimize=True)

    rho[o, o, o, o] += np.einsum("il,kj->ijkl", I, tau17, optimize=True)

    rho[o, o, o, o] += np.einsum("jk,li->ijkl", I, tau17, optimize=True)

    rho[o, o, o, o] -= np.einsum("jl,ki->ijkl", I, tau17, optimize=True)

    tau17 = None

    return rho


def two_body_density_ooov(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, M))

    tau0 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau24 = zeros((N, N, N, M))

    tau24 -= np.einsum("abjl,iklb->ijka", l2, tau0, optimize=True)

    tau26 = zeros((N, N, N, M))

    tau26 -= 2 * np.einsum("ijka->ijka", tau24, optimize=True)

    tau26 += 2 * np.einsum("jika->ijka", tau24, optimize=True)

    tau44 = zeros((N, N, N, M))

    tau44 += 4 * np.einsum("jika->ijka", tau24, optimize=True)

    tau24 = None

    tau1 = zeros((N, N, M, M))

    tau1 += np.einsum("caki,bcjk->ijab", l2, t2, optimize=True)

    tau2 = zeros((N, N, M, M))

    tau2 -= np.einsum("cbjk,kica->ijab", t2, tau1, optimize=True)

    tau3 = zeros((N, N, N, N))

    tau3 += np.einsum("baji,klab->ijkl", l2, tau2, optimize=True)

    rho[o, o, o, v] -= np.einsum("al,lkij->ijka", t1, tau3, optimize=True)

    tau3 = None

    tau20 = zeros((N, N, M, M))

    tau20 += 4 * np.einsum("ijba->ijab", tau2, optimize=True)

    tau2 = None

    tau28 = zeros((N, M, M, M))

    tau28 += np.einsum("bj,jiac->iabc", t1, tau1, optimize=True)

    tau30 = zeros((N, M, M, M))

    tau30 += 2 * np.einsum("iabc->iabc", tau28, optimize=True)

    tau28 = None

    rho[o, o, o, v] += np.einsum("lijb,klba->ijka", tau0, tau1, optimize=True)

    tau4 = zeros((N, M))

    tau4 -= np.einsum("bj,baij->ia", l1, t2, optimize=True)

    tau5 = zeros((N, N))

    tau5 += np.einsum("ai,ja->ij", l1, tau4, optimize=True)

    tau42 = zeros((N, N, N, M))

    tau42 -= 4 * np.einsum("aj,ik->ijka", t1, tau5, optimize=True)

    tau43 = zeros((N, M))

    tau43 += np.einsum("aj,ji->ia", t1, tau5, optimize=True)

    tau5 = None

    tau46 = zeros((N, M))

    tau46 += 4 * np.einsum("ia->ia", tau43, optimize=True)

    tau43 = None

    tau20 += 4 * np.einsum("bi,ja->ijab", t1, tau4, optimize=True)

    tau23 = zeros((N, N, N, M))

    tau23 += np.einsum("kb,abji->ijka", tau4, l2, optimize=True)

    tau26 -= 2 * np.einsum("ijka->ijka", tau23, optimize=True)

    tau44 -= 2 * np.einsum("ijka->ijka", tau23, optimize=True)

    tau23 = None

    tau40 = zeros((N, M))

    tau40 -= 2 * np.einsum("ia->ia", tau4, optimize=True)

    tau6 = zeros((N, N))

    tau6 -= np.einsum("baki,bajk->ij", l2, t2, optimize=True)

    tau15 = zeros((N, N, M, M))

    tau15 -= np.einsum("kj,baik->ijab", tau6, t2, optimize=True)

    tau20 -= 2 * np.einsum("ijba->ijab", tau15, optimize=True)

    tau15 = None

    tau25 = zeros((N, N))

    tau25 += np.einsum("ij->ij", tau6, optimize=True)

    tau42 += 2 * np.einsum("ka,ij->ijka", tau4, tau6, optimize=True)

    tau7 = zeros((N, N))

    tau7 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau25 += 2 * np.einsum("ij->ij", tau7, optimize=True)

    tau26 += np.einsum("ai,jk->ijka", l1, tau25, optimize=True)

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum("ki,abkj->ijab", tau25, t2, optimize=True)

    tau35 = zeros((N, N, N, N))

    tau35 += np.einsum("abkl,jiba->ijkl", l2, tau34, optimize=True)

    tau34 = None

    tau36 = zeros((N, N, N, N))

    tau36 += np.einsum("lkji->ijkl", tau35, optimize=True)

    tau35 = None

    tau39 = zeros((N, M))

    tau39 += np.einsum("aj,ji->ia", t1, tau25, optimize=True)

    tau40 += np.einsum("ia->ia", tau39, optimize=True)

    tau41 = zeros((N, M))

    tau41 += np.einsum("ia->ia", tau39, optimize=True)

    tau39 = None

    tau44 += 2 * np.einsum("ai,jk->ijka", l1, tau25, optimize=True)

    tau25 = None

    tau8 = zeros((N, N, N, M))

    tau8 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau9 = zeros((N, N, N, M))

    tau9 -= np.einsum("bakl,lijb->ijka", t2, tau8, optimize=True)

    tau10 = zeros((N, N, N, M))

    tau10 += np.einsum("lj,ikla->ijka", tau7, tau9, optimize=True)

    tau42 += 4 * np.einsum("ijka->ijka", tau10, optimize=True)

    tau10 = None

    tau13 = zeros((N, N, N, M))

    tau13 += np.einsum("ilba,ljkb->ijka", tau1, tau9, optimize=True)

    tau42 += 4 * np.einsum("ijka->ijka", tau13, optimize=True)

    tau13 = None

    tau29 = zeros((N, M, M, M))

    tau29 -= np.einsum("cbkj,kjia->iabc", t2, tau8, optimize=True)

    tau30 += np.einsum("iacb->iabc", tau29, optimize=True)

    tau29 = None

    tau31 = zeros((N, N, N, M))

    tau31 += np.einsum("jkcb,ibac->ijka", tau1, tau30, optimize=True)

    tau1 = None

    tau30 = None

    tau42 -= 2 * np.einsum("jika->ijka", tau31, optimize=True)

    tau31 = None

    tau32 = zeros((N, N, N, N))

    tau32 -= np.einsum("ilma,mjka->ijkl", tau0, tau8, optimize=True)

    tau36 += 4 * np.einsum("ijkl->ijkl", tau32, optimize=True)

    tau32 = None

    tau38 = zeros((N, M))

    tau38 -= np.einsum("abkj,kjib->ia", t2, tau8, optimize=True)

    tau40 += np.einsum("ia->ia", tau38, optimize=True)

    tau42 -= 2 * np.einsum("ka,ij->ijka", tau40, tau7, optimize=True)

    tau40 = None

    tau7 = None

    tau41 += np.einsum("ia->ia", tau38, optimize=True)

    tau38 = None

    tau42 += np.einsum("ja,ik->ijka", tau41, tau6, optimize=True)

    tau41 = None

    tau6 = None

    tau11 = zeros((N, N, N, N))

    tau11 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau12 = zeros((N, N, N, M))

    tau12 -= np.einsum("imkl,ljma->ijka", tau11, tau9, optimize=True)

    tau9 = None

    tau42 += 2 * np.einsum("ijka->ijka", tau12, optimize=True)

    tau12 = None

    tau14 = zeros((N, N, M, M))

    tau14 -= np.einsum("balk,lkij->ijab", t2, tau11, optimize=True)

    tau20 += np.einsum("ijba->ijab", tau14, optimize=True)

    tau14 = None

    tau22 = zeros((N, N, N, M))

    tau22 -= np.einsum("al,jikl->ijka", l1, tau11, optimize=True)

    tau26 += np.einsum("ijka->ijka", tau22, optimize=True)

    tau27 = zeros((N, N, N, M))

    tau27 += np.einsum("balk,iljb->ijka", t2, tau26, optimize=True)

    tau26 = None

    tau42 -= 2 * np.einsum("ijka->ijka", tau27, optimize=True)

    tau27 = None

    tau44 += np.einsum("ijka->ijka", tau22, optimize=True)

    tau22 = None

    tau45 = zeros((N, M))

    tau45 += np.einsum("bajk,jkib->ia", t2, tau44, optimize=True)

    tau44 = None

    tau46 += np.einsum("ia->ia", tau45, optimize=True)

    tau45 = None

    I = np.eye(N, dtype=dtype)

    rho[o, o, o, v] -= np.einsum("ik,ja->ijka", I, tau46, optimize=True) / 4

    rho[o, o, o, v] += np.einsum("jk,ia->ijka", I, tau46, optimize=True) / 4

    tau46 = None

    tau33 = zeros((N, N, N, N))

    tau33 -= np.einsum("jnkm,miln->ijkl", tau11, tau11, optimize=True)

    tau11 = None

    tau36 += np.einsum("ijlk->ijkl", tau33, optimize=True)

    tau33 = None

    tau37 = zeros((N, N, N, M))

    tau37 += np.einsum("al,iljk->ijka", t1, tau36, optimize=True)

    tau36 = None

    tau42 += np.einsum("ijka->ijka", tau37, optimize=True)

    tau37 = None

    tau16 = zeros((M, M))

    tau16 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau18 = zeros((M, M))

    tau18 += 2 * np.einsum("ab->ab", tau16, optimize=True)

    tau16 = None

    tau17 = zeros((M, M))

    tau17 -= np.einsum("caji,bcji->ab", l2, t2, optimize=True)

    tau18 += np.einsum("ab->ab", tau17, optimize=True)

    tau17 = None

    tau19 = zeros((N, N, M, M))

    tau19 += np.einsum("ca,cbij->ijab", tau18, t2, optimize=True)

    tau20 += 2 * np.einsum("jiba->ijab", tau19, optimize=True)

    tau19 = None

    tau21 = zeros((N, N, N, M))

    tau21 += np.einsum("liba,ljkb->ijka", tau20, tau8, optimize=True)

    tau8 = None

    tau20 = None

    tau42 -= np.einsum("kija->ijka", tau21, optimize=True)

    tau21 = None

    rho[o, o, o, v] -= np.einsum("kija->ijka", tau42, optimize=True) / 4

    rho[o, o, o, v] += np.einsum("kjia->ijka", tau42, optimize=True) / 4

    tau42 = None

    rho[o, o, o, v] += np.einsum("ba,kjib->ijka", tau18, tau0, optimize=True) / 2

    tau18 = None

    tau47 = zeros((N, N, M, M))

    tau47 += np.einsum("baji->ijab", t2, optimize=True)

    tau47 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau48 = zeros((N, N, N, N))

    tau48 += np.einsum("abkl,ijab->ijkl", l2, tau47, optimize=True)

    tau47 = None

    rho[o, o, o, v] -= np.einsum("klma,ijml->ijka", tau0, tau48, optimize=True) / 4

    tau0 = None

    rho[o, o, o, v] += np.einsum("la,ijkl->ijka", tau4, tau48, optimize=True) / 2

    tau48 = None

    tau4 = None

    rho[o, o, v, o] = -rho[o, o, o, v].transpose(0, 1, 3, 2)

    return rho


def two_body_density_oovv(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau18 = zeros((N, N, M, M))

    tau18 -= np.einsum("ablk,lkji->ijab", t2, tau0, optimize=True)

    tau27 = zeros((N, N, M, M))

    tau27 += np.einsum("ijba->ijab", tau18, optimize=True)

    tau44 = zeros((N, N, M, M))

    tau44 += np.einsum("ijba->ijab", tau18, optimize=True)

    tau18 = None

    tau29 = zeros((N, N, N, M))

    tau29 -= np.einsum("al,ilkj->ijka", t1, tau0, optimize=True)

    tau31 = zeros((N, N, N, M))

    tau31 -= np.einsum("ikja->ijka", tau29, optimize=True)

    tau29 = None

    tau50 = zeros((N, N, N, M))

    tau50 -= np.einsum("al,jikl->ijka", l1, tau0, optimize=True)

    tau52 = zeros((N, N, N, M))

    tau52 += np.einsum("ijka->ijka", tau50, optimize=True)

    tau50 = None

    tau71 = zeros((N, N, N, N))

    tau71 += np.einsum("miln,njkm->ijkl", tau0, tau0, optimize=True)

    tau73 = zeros((N, N, N, N))

    tau73 -= np.einsum("ijkl->ijkl", tau71, optimize=True)

    tau89 = zeros((N, N, N, N))

    tau89 += np.einsum("ijlk->ijkl", tau71, optimize=True)

    tau71 = None

    tau1 = zeros((N, N, M, M))

    tau1 -= np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau2 = zeros((N, N, M, M))

    tau2 -= np.einsum("cbjk,kica->ijab", t2, tau1, optimize=True)

    tau5 = zeros((N, N, N, N))

    tau5 -= np.einsum("baji,klba->ijkl", l2, tau2, optimize=True)

    tau6 = zeros((N, N, N, M))

    tau6 += np.einsum("al,iljk->ijka", t1, tau5, optimize=True)

    rho[o, o, v, v] -= np.einsum("ak,kjib->ijab", t1, tau6, optimize=True)

    tau6 = None

    tau89 += 2 * np.einsum("ijlk->ijkl", tau5, optimize=True)

    tau5 = None

    tau27 += 4 * np.einsum("ijba->ijab", tau2, optimize=True)

    tau44 += 4 * np.einsum("ijba->ijab", tau2, optimize=True)

    rho[o, o, v, v] -= np.einsum("lkij,klab->ijab", tau0, tau2, optimize=True) / 2

    tau14 = zeros((N, N, M, M))

    tau14 += np.einsum("ikcb,kjac->ijab", tau1, tau1, optimize=True)

    tau33 = zeros((N, N, M, M))

    tau33 += 4 * np.einsum("ijab->ijab", tau14, optimize=True)

    tau14 = None

    tau17 = zeros((N, N, M, M))

    tau17 -= np.einsum("kijl,lkab->ijab", tau0, tau1, optimize=True)

    tau33 -= 2 * np.einsum("ijab->ijab", tau17, optimize=True)

    tau17 = None

    tau72 = zeros((N, N, N, N))

    tau72 += np.einsum("ikab,jlba->ijkl", tau1, tau1, optimize=True)

    tau73 += 4 * np.einsum("ijkl->ijkl", tau72, optimize=True)

    tau74 = zeros((N, N, N, M))

    tau74 += np.einsum("al,iljk->ijka", t1, tau73, optimize=True)

    tau73 = None

    tau75 = zeros((N, N, N, M))

    tau75 += np.einsum("ijka->ijka", tau74, optimize=True)

    tau74 = None

    tau89 += 4 * np.einsum("ijkl->ijkl", tau72, optimize=True)

    tau72 = None

    tau88 = zeros((M, M, M, M))

    tau88 += 4 * np.einsum("ijac,jibd->abcd", tau1, tau1, optimize=True)

    tau3 = zeros((N, N, N, M))

    tau3 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau4 = zeros((N, N, N, N))

    tau4 += np.einsum("al,jika->ijkl", t1, tau3, optimize=True)

    rho[o, o, v, v] += np.einsum("klba,klji->ijab", tau2, tau4, optimize=True)

    tau2 = None

    tau4 = None

    tau7 = zeros((N, M))

    tau7 -= np.einsum("abkj,kjib->ia", t2, tau3, optimize=True)

    tau56 = zeros((N, M))

    tau56 += np.einsum("ia->ia", tau7, optimize=True)

    tau58 = zeros((N, M))

    tau58 += np.einsum("ia->ia", tau7, optimize=True)

    tau77 = zeros((N, N, M, M))

    tau77 -= 2 * np.einsum("ia,jb->ijab", tau7, tau7, optimize=True)

    tau30 = zeros((N, N, N, M))

    tau30 -= np.einsum("bakl,lijb->ijka", t2, tau3, optimize=True)

    tau31 += 2 * np.einsum("ikja->ijka", tau30, optimize=True)

    tau32 = zeros((N, N, M, M))

    tau32 += np.einsum("ak,ikjb->ijab", l1, tau31, optimize=True)

    tau31 = None

    tau33 += 2 * np.einsum("ijab->ijab", tau32, optimize=True)

    tau32 = None

    tau38 = zeros((N, N, N, M))

    tau38 += np.einsum("mikl,ljma->ijka", tau0, tau30, optimize=True)

    tau0 = None

    tau46 = zeros((N, N, N, M))

    tau46 += 2 * np.einsum("ijka->ijka", tau38, optimize=True)

    tau38 = None

    tau41 = zeros((N, N, N, M))

    tau41 += np.einsum("ilba,ljkb->ijka", tau1, tau30, optimize=True)

    tau46 += 4 * np.einsum("ijka->ijka", tau41, optimize=True)

    tau41 = None

    tau62 = zeros((N, N, M, M))

    tau62 += np.einsum("kila,ljkb->ijab", tau30, tau30, optimize=True)

    tau77 += 8 * np.einsum("ijab->ijab", tau62, optimize=True)

    tau62 = None

    tau39 = zeros((N, M, M, M))

    tau39 += np.einsum("bckj,kjia->iabc", t2, tau3, optimize=True)

    tau40 = zeros((N, N, N, M))

    tau40 += np.einsum("ikcb,jbac->ijka", tau1, tau39, optimize=True)

    tau46 += 2 * np.einsum("ijka->ijka", tau40, optimize=True)

    tau40 = None

    tau63 = zeros((N, N, M, M))

    tau63 -= np.einsum("icad,jdcb->ijab", tau39, tau39, optimize=True)

    tau39 = None

    tau77 += 2 * np.einsum("ijab->ijab", tau63, optimize=True)

    tau63 = None

    tau8 = zeros((N, N))

    tau8 -= np.einsum("baki,bajk->ij", l2, t2, optimize=True)

    tau9 = zeros((N, M))

    tau9 += np.einsum("aj,ji->ia", t1, tau8, optimize=True)

    tau58 += np.einsum("ia->ia", tau9, optimize=True)

    tau59 = zeros((N, N, M, M))

    tau59 += np.einsum("ib,ja->ijab", tau7, tau9, optimize=True)

    tau7 = None

    tau77 -= 2 * np.einsum("ia,jb->ijab", tau9, tau9, optimize=True)

    tau9 = None

    tau25 = zeros((N, N))

    tau25 += np.einsum("ij->ij", tau8, optimize=True)

    tau42 = zeros((N, N, M, M))

    tau42 += np.einsum("kj,abik->ijab", tau8, t2, optimize=True)

    tau44 -= 2 * np.einsum("ijba->ijab", tau42, optimize=True)

    tau80 = zeros((N, N, M, M))

    tau80 -= np.einsum("ijba->ijab", tau42, optimize=True)

    tau42 = None

    tau89 -= np.einsum("ik,jl->ijkl", tau8, tau8, optimize=True)

    tau10 = zeros((N, N, N, M))

    tau10 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau11 = zeros((N, N, N, M))

    tau11 -= np.einsum("balj,iklb->ijka", l2, tau10, optimize=True)

    tau12 = zeros((N, N, M, M))

    tau12 += np.einsum("bk,ikja->ijab", t1, tau11, optimize=True)

    tau33 += 4 * np.einsum("ijab->ijab", tau12, optimize=True)

    tau12 = None

    tau52 += 4 * np.einsum("jika->ijka", tau11, optimize=True)

    tau11 = None

    tau13 = zeros((N, N, M, M))

    tau13 += np.einsum("ilkb,lkja->ijab", tau10, tau3, optimize=True)

    tau33 -= 2 * np.einsum("ijab->ijab", tau13, optimize=True)

    tau13 = None

    tau35 = zeros((N, N, N, N))

    tau35 -= np.einsum("ilma,mjka->ijkl", tau10, tau3, optimize=True)

    tau36 = zeros((N, N, N, M))

    tau36 += np.einsum("al,iljk->ijka", t1, tau35, optimize=True)

    tau46 += 4 * np.einsum("ijka->ijka", tau36, optimize=True)

    tau36 = None

    tau82 = zeros((N, N, N, N))

    tau82 -= 8 * np.einsum("ijkl->ijkl", tau35, optimize=True)

    tau35 = None

    tau60 = zeros((N, N, M, M))

    tau60 += np.einsum("kjlb,lika->ijab", tau10, tau10, optimize=True)

    tau77 += 8 * np.einsum("ijab->ijab", tau60, optimize=True)

    tau60 = None

    tau66 = zeros((N, N, M, M))

    tau66 += np.einsum("ak,kijb->ijab", t1, tau10, optimize=True)

    tau67 = zeros((N, N, M, M))

    tau67 -= 2 * np.einsum("ijab->ijab", tau66, optimize=True)

    tau66 = None

    tau69 = zeros((N, N, N, M))

    tau69 -= np.einsum("ilba,ljkb->ijka", tau1, tau10, optimize=True)

    tau75 -= 4 * np.einsum("ikja->ijka", tau69, optimize=True)

    tau69 = None

    tau15 = zeros((M, M, M, M))

    tau15 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau16 = zeros((N, N, M, M))

    tau16 += np.einsum("ijdc,acbd->ijab", tau1, tau15, optimize=True)

    tau1 = None

    tau33 -= 2 * np.einsum("ijab->ijab", tau16, optimize=True)

    tau16 = None

    tau88 -= np.einsum("afce,ebdf->abcd", tau15, tau15, optimize=True)

    tau15 = None

    tau19 = zeros((N, M))

    tau19 -= np.einsum("bj,baij->ia", l1, t2, optimize=True)

    tau27 += 4 * np.einsum("bi,ja->ijab", t1, tau19, optimize=True)

    tau27 += 4 * np.einsum("aj,ib->ijab", t1, tau19, optimize=True)

    tau48 = zeros((N, N))

    tau48 += np.einsum("ai,ja->ij", l1, tau19, optimize=True)

    tau49 = zeros((N, M))

    tau49 += np.einsum("aj,ji->ia", t1, tau48, optimize=True)

    tau54 = zeros((N, M))

    tau54 += 4 * np.einsum("ia->ia", tau49, optimize=True)

    tau49 = None

    tau78 = zeros((N, N, M, M))

    tau78 += np.einsum("kj,abik->ijab", tau48, t2, optimize=True)

    tau48 = None

    tau86 = zeros((N, N, M, M))

    tau86 -= 8 * np.einsum("ijba->ijab", tau78, optimize=True)

    tau78 = None

    tau51 = zeros((N, N, N, M))

    tau51 += np.einsum("kb,abji->ijka", tau19, l2, optimize=True)

    tau52 -= 2 * np.einsum("ijka->ijka", tau51, optimize=True)

    tau51 = None

    tau61 = zeros((N, N, M, M))

    tau61 += np.einsum("kb,kija->ijab", tau19, tau10, optimize=True)

    tau77 += 8 * np.einsum("ijab->ijab", tau61, optimize=True)

    tau61 = None

    tau67 += 4 * np.einsum("aj,ib->ijab", t1, tau19, optimize=True)

    tau77 -= 8 * np.einsum("ia,jb->ijab", tau19, tau19, optimize=True)

    tau80 += 4 * np.einsum("bi,ja->ijab", t1, tau19, optimize=True)

    tau20 = zeros((M, M))

    tau20 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau22 = zeros((M, M))

    tau22 += 2 * np.einsum("ab->ab", tau20, optimize=True)

    tau88 += 4 * np.einsum("ad,bc->abcd", tau20, tau20, optimize=True)

    tau20 = None

    tau21 = zeros((M, M))

    tau21 -= np.einsum("caji,bcji->ab", l2, t2, optimize=True)

    tau22 += np.einsum("ab->ab", tau21, optimize=True)

    tau23 = zeros((N, N, M, M))

    tau23 += np.einsum("ca,cbij->ijab", tau22, t2, optimize=True)

    tau27 -= 2 * np.einsum("jiab->ijab", tau23, optimize=True)

    tau23 = None

    tau43 = zeros((N, N, M, M))

    tau43 -= np.einsum("cb,acji->ijab", tau21, t2, optimize=True)

    tau44 += 2 * np.einsum("ijab->ijab", tau43, optimize=True)

    tau45 = zeros((N, N, N, M))

    tau45 += np.einsum("lijb,lkba->ijka", tau3, tau44, optimize=True)

    tau3 = None

    tau44 = None

    tau46 -= np.einsum("ijka->ijka", tau45, optimize=True)

    tau45 = None

    tau67 += np.einsum("ijab->ijab", tau43, optimize=True)

    tau43 = None

    tau70 = zeros((N, N, N, M))

    tau70 -= np.einsum("ba,ijkb->ijka", tau21, tau10, optimize=True)

    tau10 = None

    tau75 += 2 * np.einsum("ikja->ijka", tau70, optimize=True)

    tau70 = None

    tau76 = zeros((N, N, M, M))

    tau76 += np.einsum("ak,kijb->ijab", t1, tau75, optimize=True)

    tau75 = None

    tau77 += 2 * np.einsum("ijab->ijab", tau76, optimize=True)

    tau76 = None

    tau88 -= np.einsum("ac,bd->abcd", tau21, tau21, optimize=True)

    tau21 = None

    tau24 = zeros((N, N))

    tau24 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau25 += 2 * np.einsum("ij->ij", tau24, optimize=True)

    tau26 = zeros((N, N, M, M))

    tau26 += np.einsum("ki,abkj->ijab", tau25, t2, optimize=True)

    tau27 -= 2 * np.einsum("ijba->ijab", tau26, optimize=True)

    tau26 = None

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("cbkj,ikac->ijab", l2, tau27, optimize=True)

    tau27 = None

    tau33 -= np.einsum("jiba->ijab", tau28, optimize=True)

    tau28 = None

    tau33 += np.einsum("ab,ij->ijab", tau22, tau25, optimize=True)

    tau22 = None

    tau34 = zeros((N, N, M, M))

    tau34 += np.einsum("cbkj,kica->ijab", t2, tau33, optimize=True)

    tau33 = None

    tau59 -= np.einsum("ijab->ijab", tau34, optimize=True)

    tau34 = None

    tau52 += 2 * np.einsum("ai,jk->ijka", l1, tau25, optimize=True)

    tau53 = zeros((N, M))

    tau53 += np.einsum("bajk,jkib->ia", t2, tau52, optimize=True)

    tau52 = None

    tau54 += np.einsum("ia->ia", tau53, optimize=True)

    tau53 = None

    tau59 += np.einsum("ai,jb->ijab", t1, tau54, optimize=True)

    tau54 = None

    tau55 = zeros((N, M))

    tau55 += np.einsum("aj,ji->ia", t1, tau25, optimize=True)

    tau25 = None

    tau56 += np.einsum("ia->ia", tau55, optimize=True)

    tau55 = None

    tau59 += 2 * np.einsum("jb,ia->ijab", tau19, tau56, optimize=True)

    tau56 = None

    tau19 = None

    tau37 = zeros((N, N, N, M))

    tau37 += np.einsum("lj,ikla->ijka", tau24, tau30, optimize=True)

    tau30 = None

    tau46 += 4 * np.einsum("ijka->ijka", tau37, optimize=True)

    tau37 = None

    tau47 = zeros((N, N, M, M))

    tau47 += np.einsum("ak,kijb->ijab", t1, tau46, optimize=True)

    tau46 = None

    tau59 += np.einsum("ijab->ijab", tau47, optimize=True)

    tau47 = None

    tau57 = zeros((N, M))

    tau57 += np.einsum("aj,ji->ia", t1, tau24, optimize=True)

    tau59 -= 2 * np.einsum("ia,jb->ijab", tau57, tau58, optimize=True)

    tau58 = None

    rho[o, o, v, v] -= np.einsum("ijab->ijab", tau59, optimize=True) / 4

    rho[o, o, v, v] += np.einsum("ijba->ijab", tau59, optimize=True) / 4

    rho[o, o, v, v] += np.einsum("jiab->ijab", tau59, optimize=True) / 4

    rho[o, o, v, v] -= np.einsum("jiba->ijab", tau59, optimize=True) / 4

    tau59 = None

    tau77 -= 8 * np.einsum("ia,jb->ijab", tau57, tau57, optimize=True)

    tau57 = None

    tau79 = zeros((N, N, M, M))

    tau79 -= np.einsum("ki,bajk->ijab", tau24, t2, optimize=True)

    tau80 += 2 * np.einsum("ijba->ijab", tau79, optimize=True)

    tau79 = None

    tau81 = zeros((N, N, N, N))

    tau81 += np.einsum("abkl,ijab->ijkl", l2, tau80, optimize=True)

    tau80 = None

    tau82 += np.einsum("klji->ijkl", tau81, optimize=True)

    tau84 = zeros((N, N, N, M))

    tau84 -= np.einsum("al,jkli->ijka", t1, tau81, optimize=True)

    tau81 = None

    tau85 = zeros((N, N, M, M))

    tau85 -= np.einsum("ak,kijb->ijab", t1, tau84, optimize=True)

    tau84 = None

    tau86 -= 2 * np.einsum("ijab->ijab", tau85, optimize=True)

    tau85 = None

    tau82 += 4 * np.einsum("ik,jl->ijkl", tau24, tau8, optimize=True)

    tau8 = None

    tau83 = zeros((N, N, M, M))

    tau83 += np.einsum("ablk,klij->ijab", t2, tau82, optimize=True)

    tau82 = None

    tau86 -= np.einsum("ijba->ijab", tau83, optimize=True)

    tau83 = None

    rho[o, o, v, v] -= np.einsum("ijab->ijab", tau86, optimize=True) / 8

    rho[o, o, v, v] += np.einsum("jiab->ijab", tau86, optimize=True) / 8

    tau86 = None

    tau89 -= 4 * np.einsum("ik,jl->ijkl", tau24, tau24, optimize=True)

    tau24 = None

    rho[o, o, v, v] -= np.einsum("bakl,klji->ijab", t2, tau89, optimize=True) / 4

    tau89 = None

    tau64 = zeros((N, N, M, M))

    tau64 += np.einsum("baji->ijab", t2, optimize=True)

    tau64 += 2 * np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    tau65 = zeros((N, N, N, N))

    tau65 += np.einsum("abkl,ijab->ijkl", l2, tau64, optimize=True)

    tau64 = None

    tau68 = zeros((N, N, M, M))

    tau68 -= np.einsum("jilk,klab->ijab", tau65, tau67, optimize=True)

    tau65 = None

    tau67 = None

    tau77 += np.einsum("jiab->ijab", tau68, optimize=True)

    tau68 = None

    rho[o, o, v, v] -= np.einsum("ijab->ijab", tau77, optimize=True) / 8

    rho[o, o, v, v] += np.einsum("ijba->ijab", tau77, optimize=True) / 8

    tau77 = None

    tau87 = zeros((N, M, M, M))

    tau87 += np.einsum("aj,bcij->iabc", l1, t2, optimize=True)

    tau88 += 2 * np.einsum("ai,ibdc->abcd", l1, tau87, optimize=True)

    tau87 = None

    rho[o, o, v, v] += np.einsum("cdji,cdab->ijab", t2, tau88, optimize=True) / 4

    tau88 = None

    return rho


def two_body_density_ovoo(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    rho[o, v, o, o] -= np.einsum("al,kjil->iajk", l1, tau0, optimize=True) / 2

    tau0 = None

    tau1 = zeros((N, M))

    tau1 -= np.einsum("bj,baij->ia", l1, t2, optimize=True)

    rho[o, v, o, o] -= np.einsum("ib,abkj->iajk", tau1, l2, optimize=True)

    tau1 = None

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau3 = zeros((N, N, N, M))

    tau3 -= np.einsum("abjl,iklb->ijka", l2, tau2, optimize=True)

    tau2 = None

    tau7 = zeros((N, N, N, M))

    tau7 -= 2 * np.einsum("ijka->ijka", tau3, optimize=True)

    tau3 = None

    tau4 = zeros((N, N))

    tau4 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau6 = zeros((N, N))

    tau6 += 2 * np.einsum("ij->ij", tau4, optimize=True)

    tau4 = None

    tau5 = zeros((N, N))

    tau5 += np.einsum("baik,bajk->ij", l2, t2, optimize=True)

    tau6 += np.einsum("ij->ij", tau5, optimize=True)

    tau5 = None

    tau7 += np.einsum("ai,jk->ijka", l1, tau6, optimize=True)

    tau6 = None

    rho[o, v, o, o] += np.einsum("jkia->iajk", tau7, optimize=True) / 2

    rho[o, v, o, o] -= np.einsum("kjia->iajk", tau7, optimize=True) / 2

    tau7 = None

    rho[v, o, o, o] = -rho[o, v, o, o].transpose(1, 0, 2, 3)

    return rho


def two_body_density_ovov(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M))

    tau0 -= np.einsum("bj,abji->ia", l1, t2, optimize=True)

    tau1 = zeros((M, M))

    tau1 += np.einsum("ai,ib->ab", l1, tau0, optimize=True)

    I = np.eye(N, dtype=dtype)

    rho[o, v, o, v] += np.einsum("ij,ab->iajb", I, tau1, optimize=True)

    tau1 = None

    tau14 = zeros((N, N, M, M))

    tau14 += 4 * np.einsum("bi,ja->ijab", t1, tau0, optimize=True)

    tau14 += 4 * np.einsum("aj,ib->ijab", t1, tau0, optimize=True)

    tau16 = zeros((N, M))

    tau16 -= 2 * np.einsum("ia->ia", tau0, optimize=True)

    tau0 = None

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau4 = zeros((N, N, N, M))

    tau4 += np.einsum("ablj,iklb->ijka", l2, tau2, optimize=True)

    rho[o, v, o, v] -= np.einsum("bk,jkia->iajb", t1, tau4, optimize=True)

    tau4 = None

    tau15 = zeros((N, N, N, M))

    tau15 -= 2 * np.einsum("ikja->ijka", tau2, optimize=True)

    tau3 = zeros((N, N, N, M))

    tau3 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau15 += 2 * np.einsum("ablj,ilkb->ijka", t2, tau3, optimize=True)

    tau16 -= np.einsum("abkj,kjib->ia", t2, tau3, optimize=True)

    rho[o, v, o, v] += np.einsum("jlkb,lkia->iajb", tau2, tau3, optimize=True) / 2

    tau2 = None

    tau3 = None

    tau5 = zeros((N, N, M, M))

    tau5 += np.einsum("caik,bckj->ijab", l2, t2, optimize=True)

    tau14 -= 4 * np.einsum("cajk,kicb->ijab", t2, tau5, optimize=True)

    rho[o, v, o, v] -= np.einsum("jkcb,kiac->iajb", tau5, tau5, optimize=True)

    tau6 = zeros((M, M, M, M))

    tau6 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    rho[o, v, o, v] += np.einsum("jidc,acbd->iajb", tau5, tau6, optimize=True) / 2

    tau6 = None

    tau7 = zeros((N, N, N, N))

    tau7 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau14 -= np.einsum("balk,lkji->ijab", t2, tau7, optimize=True)

    tau15 += np.einsum("al,iljk->ijka", t1, tau7, optimize=True)

    rho[o, v, o, v] -= np.einsum("ak,jkib->iajb", l1, tau15, optimize=True) / 2

    tau15 = None

    rho[o, v, o, v] += np.einsum("klab,jlik->iajb", tau5, tau7, optimize=True) / 2

    tau5 = None

    tau7 = None

    tau8 = zeros((M, M))

    tau8 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau10 = zeros((M, M))

    tau10 += 2 * np.einsum("ab->ab", tau8, optimize=True)

    tau8 = None

    tau9 = zeros((M, M))

    tau9 -= np.einsum("caji,bcji->ab", l2, t2, optimize=True)

    tau10 += np.einsum("ab->ab", tau9, optimize=True)

    tau9 = None

    tau14 -= 2 * np.einsum("ca,cbji->ijab", tau10, t2, optimize=True)

    tau11 = zeros((N, N))

    tau11 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau13 = zeros((N, N))

    tau13 += 2 * np.einsum("ij->ij", tau11, optimize=True)

    tau11 = None

    tau12 = zeros((N, N))

    tau12 -= np.einsum("baki,bajk->ij", l2, t2, optimize=True)

    tau13 += np.einsum("ij->ij", tau12, optimize=True)

    tau12 = None

    tau14 -= 2 * np.einsum("ki,bakj->ijab", tau13, t2, optimize=True)

    rho[o, v, o, v] += np.einsum("cakj,ikbc->iajb", l2, tau14, optimize=True) / 4

    tau14 = None

    tau16 += np.einsum("aj,ji->ia", t1, tau13, optimize=True)

    rho[o, v, o, v] += np.einsum("aj,ib->iajb", l1, tau16, optimize=True) / 2

    tau16 = None

    rho[o, v, o, v] -= np.einsum("ab,ji->iajb", tau10, tau13, optimize=True) / 4

    tau10 = None

    tau13 = None

    rho[v, o, o, v] = -rho[o, v, o, v].transpose(1, 0, 2, 3)
    rho[o, v, v, o] = -rho[o, v, o, v].transpose(0, 1, 3, 2)
    rho[v, o, v, o] = rho[o, v, o, v].transpose(1, 0, 3, 2)

    return rho


def two_body_density_vovv(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, N, M, M))

    tau0 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    tau1 = zeros((N, N, M, M))

    tau1 -= np.einsum("cbjk,kica->ijab", t2, tau0, optimize=True)

    tau36 = zeros((N, N, M, M))

    tau36 += 4 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau6 = zeros((N, M, M, M))

    tau6 += np.einsum("bj,jiac->iabc", t1, tau0, optimize=True)

    tau8 = zeros((N, M, M, M))

    tau8 += 2 * np.einsum("iabc->iabc", tau6, optimize=True)

    tau6 = None

    tau14 = zeros((N, M, M, M))

    tau14 += np.einsum("aj,ijbc->iabc", l1, tau0, optimize=True)

    tau17 = zeros((N, M, M, M))

    tau17 -= 2 * np.einsum("iabc->iabc", tau14, optimize=True)

    tau17 += 2 * np.einsum("ibac->iabc", tau14, optimize=True)

    tau14 = None

    tau31 = zeros((N, N, M, M))

    tau31 += np.einsum("ikcb,kjac->ijab", tau0, tau0, optimize=True)

    tau38 = zeros((N, N, M, M))

    tau38 += 4 * np.einsum("ijab->ijab", tau31, optimize=True)

    tau31 = None

    tau2 = zeros((N, N, N, M))

    tau2 += np.einsum("bk,abij->ijka", t1, l2, optimize=True)

    tau7 = zeros((N, M, M, M))

    tau7 -= np.einsum("cbkj,kjia->iabc", t2, tau2, optimize=True)

    tau8 += np.einsum("iacb->iabc", tau7, optimize=True)

    tau7 = None

    tau22 = zeros((N, N, N, M))

    tau22 -= np.einsum("ablk,lijb->ijka", t2, tau2, optimize=True)

    tau23 = zeros((N, N, N, M))

    tau23 += 2 * np.einsum("ikja->ijka", tau22, optimize=True)

    tau28 = zeros((N, N, M, M))

    tau28 += np.einsum("ak,ijkb->ijab", l1, tau22, optimize=True)

    tau22 = None

    tau38 += 4 * np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau40 = zeros((N, M))

    tau40 -= np.einsum("abkj,kjib->ia", t2, tau2, optimize=True)

    tau43 = zeros((N, M))

    tau43 += np.einsum("ia->ia", tau40, optimize=True)

    tau44 = zeros((N, M))

    tau44 += np.einsum("ia->ia", tau40, optimize=True)

    tau40 = None

    rho[v, o, v, v] -= np.einsum("jkcb,kjia->aibc", tau1, tau2, optimize=True)

    tau1 = None

    tau3 = zeros((N, M))

    tau3 -= np.einsum("bj,abji->ia", l1, t2, optimize=True)

    tau4 = zeros((M, M))

    tau4 += np.einsum("ai,ib->ab", l1, tau3, optimize=True)

    tau45 = zeros((N, M, M, M))

    tau45 += 4 * np.einsum("bi,ac->iabc", t1, tau4, optimize=True)

    tau4 = None

    tau13 = zeros((N, M, M, M))

    tau13 += np.einsum("jc,baij->iabc", tau3, l2, optimize=True)

    tau17 += 2 * np.einsum("ibac->iabc", tau13, optimize=True)

    tau13 = None

    tau26 = zeros((N, N, M, M))

    tau26 -= 4 * np.einsum("ai,jb->ijab", t1, tau3, optimize=True)

    tau43 -= 2 * np.einsum("ia->ia", tau3, optimize=True)

    tau47 = zeros((N, N, N, M))

    tau47 += np.einsum("kb,abji->ijka", tau3, l2, optimize=True)

    tau48 = zeros((N, N, N, M))

    tau48 -= 2 * np.einsum("ijka->ijka", tau47, optimize=True)

    tau49 = zeros((N, N, N, M))

    tau49 += 2 * np.einsum("ijka->ijka", tau47, optimize=True)

    tau47 = None

    tau5 = zeros((M, M))

    tau5 -= np.einsum("caji,bcji->ab", l2, t2, optimize=True)

    tau16 = zeros((M, M))

    tau16 += np.einsum("ab->ab", tau5, optimize=True)

    tau35 = zeros((N, N, M, M))

    tau35 -= np.einsum("cb,acji->ijab", tau5, t2, optimize=True)

    tau36 -= 2 * np.einsum("ijab->ijab", tau35, optimize=True)

    tau35 = None

    tau45 -= 2 * np.einsum("ic,ab->iabc", tau3, tau5, optimize=True)

    tau3 = None

    tau9 = zeros((M, M, M, M))

    tau9 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau10 = zeros((N, M, M, M))

    tau10 += np.einsum("idae,ebdc->iabc", tau8, tau9, optimize=True)

    tau8 = None

    tau9 = None

    tau45 += np.einsum("ibac->iabc", tau10, optimize=True)

    tau10 = None

    tau11 = zeros((N, N, N, M))

    tau11 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau12 = zeros((N, M, M, M))

    tau12 += np.einsum("abkj,ikjc->iabc", l2, tau11, optimize=True)

    tau17 -= np.einsum("ibac->iabc", tau12, optimize=True)

    tau12 = None

    tau29 = zeros((N, N, N, M))

    tau29 += np.einsum("ablj,iklb->ijka", l2, tau11, optimize=True)

    tau11 = None

    tau30 = zeros((N, N, M, M))

    tau30 += np.einsum("bk,ikja->ijab", t1, tau29, optimize=True)

    tau38 += 4 * np.einsum("ijab->ijab", tau30, optimize=True)

    tau30 = None

    tau48 -= 4 * np.einsum("ijka->ijka", tau29, optimize=True)

    tau29 = None

    tau15 = zeros((M, M))

    tau15 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau16 += 2 * np.einsum("ab->ab", tau15, optimize=True)

    tau17 += np.einsum("ai,bc->iabc", l1, tau16, optimize=True)

    tau18 = zeros((N, M, M, M))

    tau18 += np.einsum("dcji,jadb->iabc", t2, tau17, optimize=True)

    tau17 = None

    tau45 += 2 * np.einsum("iabc->iabc", tau18, optimize=True)

    tau18 = None

    tau25 = zeros((N, N, M, M))

    tau25 += np.einsum("ca,cbij->ijab", tau16, t2, optimize=True)

    tau16 = None

    tau26 -= np.einsum("jiab->ijab", tau25, optimize=True)

    tau25 = None

    tau27 = zeros((N, M, M, M))

    tau27 += np.einsum("kjic,jkab->iabc", tau2, tau26, optimize=True)

    tau2 = None

    tau26 = None

    tau45 += np.einsum("ibca->iabc", tau27, optimize=True)

    tau27 = None

    tau19 = zeros((N, N))

    tau19 -= np.einsum("baki,bajk->ij", l2, t2, optimize=True)

    tau23 += np.einsum("aj,ik->ijka", t1, tau19, optimize=True)

    tau41 = zeros((N, N))

    tau41 += np.einsum("ij->ij", tau19, optimize=True)

    tau19 = None

    tau20 = zeros((N, N, N, N))

    tau20 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau21 = zeros((N, N, N, M))

    tau21 -= np.einsum("al,ilkj->ijka", t1, tau20, optimize=True)

    tau23 -= np.einsum("ikja->ijka", tau21, optimize=True)

    tau21 = None

    tau24 = zeros((N, M, M, M))

    tau24 += np.einsum("kjbc,jkia->iabc", tau0, tau23, optimize=True)

    tau0 = None

    tau23 = None

    tau45 += 2 * np.einsum("ibac->iabc", tau24, optimize=True)

    tau24 = None

    tau34 = zeros((N, N, M, M))

    tau34 -= np.einsum("ablk,lkji->ijab", t2, tau20, optimize=True)

    tau36 -= np.einsum("ijba->ijab", tau34, optimize=True)

    tau34 = None

    tau46 = zeros((N, N, N, M))

    tau46 -= np.einsum("al,jikl->ijka", l1, tau20, optimize=True)

    tau20 = None

    tau48 += np.einsum("ijka->ijka", tau46, optimize=True)

    tau49 -= np.einsum("ijka->ijka", tau46, optimize=True)

    tau46 = None

    tau50 = zeros((N, N, M, M))

    tau50 += np.einsum("bk,ikja->ijab", t1, tau49, optimize=True)

    tau49 = None

    rho[v, o, v, v] += np.einsum("bj,jiac->aibc", t1, tau50, optimize=True) / 2

    tau50 = None

    tau32 = zeros((N, N))

    tau32 += np.einsum("ai,aj->ij", l1, t1, optimize=True)

    tau33 = zeros((N, N, M, M))

    tau33 -= np.einsum("ki,bajk->ijab", tau32, t2, optimize=True)

    tau36 -= 4 * np.einsum("ijba->ijab", tau33, optimize=True)

    tau33 = None

    tau37 = zeros((N, N, M, M))

    tau37 += np.einsum("cbkj,ikca->ijab", l2, tau36, optimize=True)

    tau36 = None

    tau38 -= np.einsum("jiba->ijab", tau37, optimize=True)

    tau37 = None

    tau39 = zeros((N, M, M, M))

    tau39 += np.einsum("cj,jiab->iabc", t1, tau38, optimize=True)

    tau38 = None

    tau45 -= np.einsum("iacb->iabc", tau39, optimize=True)

    tau39 = None

    tau41 += 2 * np.einsum("ij->ij", tau32, optimize=True)

    tau32 = None

    tau42 = zeros((N, M))

    tau42 += np.einsum("aj,ji->ia", t1, tau41, optimize=True)

    tau43 += np.einsum("ia->ia", tau42, optimize=True)

    tau45 += 2 * np.einsum("ab,ic->iabc", tau15, tau43, optimize=True)

    tau43 = None

    tau15 = None

    tau44 += np.einsum("ia->ia", tau42, optimize=True)

    tau42 = None

    tau45 -= np.einsum("ib,ac->iabc", tau44, tau5, optimize=True)

    tau44 = None

    tau5 = None

    rho[v, o, v, v] -= np.einsum("iabc->aibc", tau45, optimize=True) / 4

    rho[v, o, v, v] += np.einsum("iacb->aibc", tau45, optimize=True) / 4

    tau45 = None

    tau48 += 2 * np.einsum("ai,jk->ijka", l1, tau41, optimize=True)

    tau41 = None

    rho[v, o, v, v] -= np.einsum("cbkj,jkia->aibc", t2, tau48, optimize=True) / 4

    tau48 = None

    rho[o, v, v, v] = -rho[v, o, v, v].transpose(1, 0, 2, 3)

    return rho


def two_body_density_vvoo(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    rho[v, v, o, o] -= np.einsum("aj,bi->abij", l1, l1, optimize=True)

    rho[v, v, o, o] += np.einsum("ai,bj->abij", l1, l1, optimize=True)

    return rho


def two_body_density_vvvo(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M))

    tau0 -= np.einsum("bj,abji->ia", l1, t2, optimize=True)

    rho[v, v, v, o] -= np.einsum("jc,abij->abci", tau0, l2, optimize=True)

    tau0 = None

    tau1 = zeros((N, N, N, M))

    tau1 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    rho[v, v, v, o] -= np.einsum("abkj,ikjc->abci", l2, tau1, optimize=True) / 2

    tau1 = None

    tau2 = zeros((N, N, M, M))

    tau2 -= np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau3 = zeros((N, M, M, M))

    tau3 += np.einsum("aj,ijbc->iabc", l1, tau2, optimize=True)

    tau2 = None

    tau7 = zeros((N, M, M, M))

    tau7 -= 2 * np.einsum("iabc->iabc", tau3, optimize=True)

    tau3 = None

    tau4 = zeros((M, M))

    tau4 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    tau6 = zeros((M, M))

    tau6 += 2 * np.einsum("ab->ab", tau4, optimize=True)

    tau4 = None

    tau5 = zeros((M, M))

    tau5 += np.einsum("acji,bcji->ab", l2, t2, optimize=True)

    tau6 += np.einsum("ab->ab", tau5, optimize=True)

    tau5 = None

    tau7 += np.einsum("ai,bc->iabc", l1, tau6, optimize=True)

    tau6 = None

    rho[v, v, v, o] -= np.einsum("iabc->abci", tau7, optimize=True) / 2

    rho[v, v, v, o] += np.einsum("ibac->abci", tau7, optimize=True) / 2

    tau7 = None

    rho[v, v, o, v] = -rho[v, v, v, o].transpose(0, 1, 3, 2)

    return rho


def two_body_density_vvvv(rho, t1, t2, l1, l2, o, v):
    M, N = t1.shape
    dtype = t1.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = zeros((N, M, M, M))

    tau0 += np.einsum("aj,bcij->iabc", l1, t2, optimize=True)

    rho[v, v, v, v] -= np.einsum("ai,ibdc->abcd", l1, tau0, optimize=True)

    tau0 = None

    tau1 = zeros((M, M))

    tau1 += np.einsum("ai,bi->ab", l1, t1, optimize=True)

    rho[v, v, v, v] -= np.einsum("ad,bc->abcd", tau1, tau1, optimize=True)

    rho[v, v, v, v] += np.einsum("ac,bd->abcd", tau1, tau1, optimize=True)

    tau2 = zeros((M, M))

    tau2 += np.einsum("acji,bcji->ab", l2, t2, optimize=True)

    tau6 = zeros((M, M, M, M))

    tau6 -= np.einsum("ac,bd->abcd", tau1, tau2, optimize=True)

    tau1 = None

    tau12 = zeros((N, N, M, M))

    tau12 -= np.einsum("cb,acji->ijab", tau2, t2, optimize=True)

    tau13 = zeros((N, N, M, M))

    tau13 -= np.einsum("ijab->ijab", tau12, optimize=True)

    tau12 = None

    tau15 = zeros((M, M, M, M))

    tau15 -= np.einsum("ac,bd->abcd", tau2, tau2, optimize=True)

    tau2 = None

    tau3 = zeros((N, N, M, M))

    tau3 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau4 = zeros((N, M, M, M))

    tau4 += np.einsum("bj,jiac->iabc", t1, tau3, optimize=True)

    tau5 = zeros((M, M, M, M))

    tau5 += np.einsum("ai,ibcd->abcd", l1, tau4, optimize=True)

    tau4 = None

    tau6 += 2 * np.einsum("abcd->abcd", tau5, optimize=True)

    tau5 = None

    rho[v, v, v, v] -= np.einsum("abcd->abcd", tau6, optimize=True) / 2

    rho[v, v, v, v] += np.einsum("abdc->abcd", tau6, optimize=True) / 2

    rho[v, v, v, v] += np.einsum("bacd->abcd", tau6, optimize=True) / 2

    rho[v, v, v, v] -= np.einsum("badc->abcd", tau6, optimize=True) / 2

    tau6 = None

    tau9 = zeros((M, M, M, M))

    tau9 += np.einsum("ijac,jibd->abcd", tau3, tau3, optimize=True)

    tau15 += 4 * np.einsum("abcd->abcd", tau9, optimize=True)

    tau9 = None

    tau17 = zeros((N, N, M, M))

    tau17 -= np.einsum("bckj,kica->ijab", t2, tau3, optimize=True)

    tau3 = None

    tau7 = zeros((M, M, M, M))

    tau7 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau8 = zeros((M, M, M, M))

    tau8 += np.einsum("afce,bedf->abcd", tau7, tau7, optimize=True)

    tau7 = None

    tau15 += np.einsum("abcd->abcd", tau8, optimize=True)

    tau8 = None

    tau10 = zeros((N, N, N, M))

    tau10 += np.einsum("bi,abjk->ijka", l1, t2, optimize=True)

    tau11 = zeros((N, N, M, M))

    tau11 -= np.einsum("ak,kjib->ijab", t1, tau10, optimize=True)

    tau10 = None

    tau13 += 2 * np.einsum("ijab->ijab", tau11, optimize=True)

    tau11 = None

    tau14 = zeros((M, M, M, M))

    tau14 += np.einsum("abij,ijcd->abcd", l2, tau13, optimize=True)

    tau13 = None

    tau15 += np.einsum("bacd->abcd", tau14, optimize=True)

    tau14 = None

    rho[v, v, v, v] -= np.einsum("abcd->abcd", tau15, optimize=True) / 4

    rho[v, v, v, v] += np.einsum("abdc->abcd", tau15, optimize=True) / 4

    tau15 = None

    tau16 = zeros((N, M))

    tau16 -= np.einsum("bj,abji->ia", l1, t2, optimize=True)

    tau17 -= np.einsum("aj,ib->ijab", t1, tau16, optimize=True)

    tau17 += np.einsum("bj,ia->ijab", t1, tau16, optimize=True)

    tau16 = None

    rho[v, v, v, v] += np.einsum("baij,ijdc->abcd", l2, tau17, optimize=True)

    tau17 = None

    return rho
