import numpy as np
from quadratictheory.cc.energies.e_inter_rccd import td_energy_addition_opti_restricted


def energy_intermediates_qccd_restricted(t2, l2, u, f, o, v):
    D = np.einsum("ijab,abij", u[o, o, v, v], t2, optimize=True)
    E = np.einsum("ijba,abij", u[o, o, v, v], t2, optimize=True)
    e = 2 * D - E

    e += td_energy_addition_opti_restricted(t2, l2, u, f, o, v)
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    tau0 = np.zeros((N, N, N, N))

    tau0 += np.einsum("baij,bakl->ijkl", l2, t2, optimize=True)

    tau155 = np.zeros((N, N, M, M))

    tau155 += np.einsum("ijlk,lkab->ijab", tau0, u[o, o, v, v], optimize=True)

    tau1 = np.zeros((N, N, M, M))

    tau1 += np.einsum("acki,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau2 = np.zeros((N, N, N, N))

    tau2 += np.einsum("abjk,ilab->ijkl", t2, tau1, optimize=True)

    e -= 2 * np.einsum("klij,iklj->", tau0, tau2, optimize=True) / 3

    tau2 = None

    tau68 = np.zeros((N, N, M, M))

    tau68 -= 4 * np.einsum("cajk,ikcb->ijab", t2, tau1, optimize=True)

    tau127 = np.zeros((N, N, M, M))

    tau127 += 2 * np.einsum("ijab->ijab", tau1, optimize=True)

    tau1 = None

    tau3 = np.zeros((N, N, M, M))

    tau3 += np.einsum("caki,cbkj->ijab", l2, t2, optimize=True)

    tau10 = np.zeros((N, N, M, M))

    tau10 += np.einsum("caik,kjcb->ijab", t2, tau3, optimize=True)

    tau11 = np.zeros((N, N, N, N))

    tau11 += np.einsum("ijba,klba->ijkl", tau10, u[o, o, v, v], optimize=True)

    tau10 = None

    e -= 2 * np.einsum("ijlk,ijlk->", tau0, tau11, optimize=True)

    tau11 = None

    tau18 = np.zeros((N, N, M, M))

    tau18 += np.einsum("cbkj,kica->ijab", t2, tau3, optimize=True)

    tau19 = np.zeros((N, N, N, N))

    tau19 += np.einsum("ijab,klab->ijkl", tau18, u[o, o, v, v], optimize=True)

    e += 2 * np.einsum("ijlk,ijlk->", tau0, tau19, optimize=True)

    tau19 = None

    tau132 = np.zeros((N, N, N, N))

    tau132 += 3 * np.einsum("abij,klab->ijkl", l2, tau18, optimize=True)

    tau18 = None

    tau42 = np.zeros((N, N, M, M))

    tau42 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau43 = np.zeros((N, N, M, M))

    tau43 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau49 = np.zeros((N, N, M, M))

    tau49 += 2 * np.einsum("ijcd,bcda->ijab", tau3, u[v, v, v, v], optimize=True)

    tau61 = np.zeros((N, N, M, M))

    tau61 -= 3 * np.einsum("ijab->ijab", tau3, optimize=True)

    tau110 = np.zeros((N, N, M, M))

    tau110 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau123 = np.zeros((N, N, M, M))

    tau123 += np.einsum("ijab->ijab", tau3, optimize=True)

    tau133 = np.zeros((N, N, N, N))

    tau133 += np.einsum("ikab,jlba->ijkl", tau3, tau3, optimize=True)

    tau135 = np.zeros((N, N, N, N))

    tau135 += 2 * np.einsum("ijkl->ijkl", tau133, optimize=True)

    tau138 = np.zeros((N, N, N, N))

    tau138 -= 3 * np.einsum("ijkl->ijkl", tau133, optimize=True)

    tau133 = None

    tau139 = np.zeros((N, N, M, M))

    tau139 += np.einsum("acki,kjcb->ijab", t2, tau3, optimize=True)

    tau140 = np.zeros((N, N, N, N))

    tau140 += 6 * np.einsum("abij,lkab->ijkl", l2, tau139, optimize=True)

    tau139 = None

    tau147 = np.zeros((N, N, M, M))

    tau147 -= 3 * np.einsum("ijab->ijab", tau3, optimize=True)

    tau4 = np.zeros((N, N, M, M))

    tau4 += np.einsum("caki,bckj->ijab", l2, t2, optimize=True)

    tau5 = np.zeros((N, N, N, N))

    tau5 += np.einsum("ikab,jlba->ijkl", tau3, tau4, optimize=True)

    e -= 2 * np.einsum("ijkl,lkij->", tau5, u[o, o, o, o], optimize=True)

    tau5 = None

    tau59 = np.zeros((M, M, M, M))

    tau59 -= 4 * np.einsum("ijab,jcdi->abcd", tau4, u[o, v, v, o], optimize=True)

    tau124 = np.zeros((N, N, N, N))

    tau124 += 2 * np.einsum("ilab,jkba->ijkl", tau4, tau4, optimize=True)

    tau6 = np.zeros((N, N, M, M))

    tau6 += np.einsum("caik,bckj->ijab", l2, t2, optimize=True)

    tau42 += np.einsum("ijab->ijab", tau6, optimize=True)

    tau7 = np.zeros((N, N, M, M))

    tau7 += np.einsum("acik,cbkj->ijab", l2, t2, optimize=True)

    tau8 = np.zeros((N, N, N, N))

    tau8 += np.einsum("jkba,ilab->ijkl", tau6, tau7, optimize=True)

    tau76 = np.zeros((M, M, M, M))

    tau76 += np.einsum("ijbc,jiad->abcd", tau7, tau7, optimize=True)

    tau77 = np.zeros((N, N, M, M))

    tau77 += np.einsum("abcd,ijcd->ijab", tau76, u[o, o, v, v], optimize=True)

    tau76 = None

    tau105 = np.zeros((N, N, M, M))

    tau105 -= 3 * np.einsum("ijab->ijab", tau77, optimize=True)

    tau105 += 6 * np.einsum("ijba->ijab", tau77, optimize=True)

    tau77 = None

    tau80 = np.zeros((N, N, M, M))

    tau80 += 3 * np.einsum("ijab->ijab", tau7, optimize=True)

    tau9 = np.zeros((N, N, N, N))

    tau9 += np.einsum("abij,klab->ijkl", t2, u[o, o, v, v], optimize=True)

    tau109 = np.zeros((N, N, N, N))

    tau109 += np.einsum("mijn,mkln->ijkl", tau0, tau9, optimize=True)

    tau115 = np.zeros((N, N, N, N))

    tau115 += 2 * np.einsum("ijlk->ijkl", tau109, optimize=True)

    tau115 += 2 * np.einsum("iljk->ijkl", tau109, optimize=True)

    tau109 = None

    tau114 = np.zeros((N, N, N, N))

    tau114 += 3 * np.einsum("klij->ijkl", tau9, optimize=True)

    tau114 -= 2 * np.einsum("klji->ijkl", tau9, optimize=True)

    tau129 = np.zeros((N, N, N, N))

    tau129 += np.einsum("lkji->ijkl", tau9, optimize=True)

    tau136 = np.zeros((N, N, N, N))

    tau136 += np.einsum("klij->ijkl", tau9, optimize=True)

    tau154 = np.zeros((N, N, N, N))

    tau154 += 2 * np.einsum("ijkl->ijkl", tau9, optimize=True)

    tau154 += np.einsum("ijlk->ijkl", tau9, optimize=True)

    e -= 2 * np.einsum("ijkl,ijlk->", tau8, tau9, optimize=True)

    tau8 = None

    tau12 = np.zeros((N, N, M, M))

    tau12 += np.einsum("acik,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau13 = np.zeros((N, N, M, M))

    tau13 += np.einsum("acik,kjcb->ijab", l2, tau12, optimize=True)

    tau14 = np.zeros((N, N, N, N))

    tau14 += np.einsum("abjk,ilab->ijkl", t2, tau13, optimize=True)

    tau13 = None

    e -= 2 * np.einsum("klij,iklj->", tau0, tau14, optimize=True)

    tau14 = None

    tau36 = np.zeros((N, N, M, M))

    tau36 -= np.einsum("ijab->ijab", tau12, optimize=True)

    tau67 = np.zeros((N, N, M, M))

    tau67 -= np.einsum("ijab->ijab", tau12, optimize=True)

    tau15 = np.zeros((N, N, M, M))

    tau15 += np.einsum("acki,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau16 = np.zeros((N, N, M, M))

    tau16 += np.einsum("acik,kjcb->ijab", l2, tau15, optimize=True)

    tau17 = np.zeros((N, N, N, N))

    tau17 += np.einsum("abjk,ilab->ijkl", t2, tau16, optimize=True)

    tau16 = None

    e -= 2 * np.einsum("klij,iklj->", tau0, tau17, optimize=True)

    tau17 = None

    tau48 = np.zeros((N, N, M, M))

    tau48 += np.einsum("ijab->ijab", tau15, optimize=True)

    tau111 = np.zeros((N, N, M, M))

    tau111 += np.einsum("ijab->ijab", tau15, optimize=True)

    tau20 = np.zeros((N, N, M, M))

    tau20 += np.einsum("acik,kjcb->ijab", t2, u[o, o, v, v], optimize=True)

    tau21 = np.zeros((N, N, M, M))

    tau21 += np.einsum("acik,kjcb->ijab", l2, tau20, optimize=True)

    tau22 = np.zeros((N, N, N, N))

    tau22 += np.einsum("abjk,ilab->ijkl", t2, tau21, optimize=True)

    tau21 = None

    e += 4 * np.einsum("klij,iklj->", tau0, tau22, optimize=True)

    tau22 = None

    tau81 = np.zeros((M, M, M, M))

    tau81 += 6 * np.einsum("ijcd,ijab->abcd", tau20, tau7, optimize=True)

    tau88 = np.zeros((N, N, M, M))

    tau88 += 2 * np.einsum("ijab->ijab", tau20, optimize=True)

    tau20 = None

    tau23 = np.zeros((N, N, M, M))

    tau23 += np.einsum("caik,cbjk->ijab", l2, t2, optimize=True)

    tau24 = np.zeros((N, N, N, N))

    tau24 += np.einsum("klab,ijab->ijkl", tau15, tau23, optimize=True)

    e += 2 * np.einsum("kjil,ijkl->", tau0, tau24, optimize=True) / 3

    tau24 = None

    tau43 += 2 * np.einsum("ijab->ijab", tau23, optimize=True)

    tau49 -= np.einsum("ijcd,cbda->ijab", tau43, u[v, v, v, v], optimize=True)

    tau43 = None

    tau110 += np.einsum("ijab->ijab", tau23, optimize=True)

    tau164 = np.zeros((N, N, M, M))

    tau164 += 2 * np.einsum("cbjk,kica->ijab", t2, tau110, optimize=True)

    tau168 = np.zeros((N, N))

    tau168 -= 2 * np.einsum("kiab,jakb->ij", tau110, u[o, v, o, v], optimize=True)

    tau143 = np.zeros((N, N, M, M))

    tau143 += np.einsum("caik,kjcb->ijab", t2, tau23, optimize=True)

    tau144 = np.zeros((N, N, N, N))

    tau144 += np.einsum("ijba,lkab->ijkl", tau143, u[o, o, v, v], optimize=True)

    tau143 = None

    tau145 = np.zeros((N, N, N, N))

    tau145 += np.einsum("jikl->ijkl", tau144, optimize=True)

    tau145 += 2 * np.einsum("jilk->ijkl", tau144, optimize=True)

    tau144 = None

    tau25 = np.zeros((N, N, M, M))

    tau25 += np.einsum("acki,kjbc->ijab", t2, u[o, o, v, v], optimize=True)

    tau26 = np.zeros((N, N, M, M))

    tau26 += np.einsum("acki,kjcb->ijab", l2, tau25, optimize=True)

    tau27 = np.zeros((N, N, N, N))

    tau27 += np.einsum("abjk,ilab->ijkl", t2, tau26, optimize=True)

    e += 2 * np.einsum("klij,iklj->", tau0, tau27, optimize=True) / 3

    e += 4 * np.einsum("klij,ilkj->", tau0, tau27, optimize=True) / 3

    tau27 = None

    tau93 = np.zeros((N, N, M, M))

    tau93 += 2 * np.einsum("ijab->ijab", tau26, optimize=True)

    tau26 = None

    tau28 = np.zeros((N, N, M, M))

    tau28 += np.einsum("acik,kjcb->ijab", l2, tau25, optimize=True)

    tau29 = np.zeros((N, N, N, N))

    tau29 += np.einsum("abjk,ilab->ijkl", t2, tau28, optimize=True)

    e += 4 * np.einsum("klij,iklj->", tau0, tau29, optimize=True) / 3

    tau29 = None

    tau127 -= np.einsum("ijab->ijab", tau28, optimize=True)

    tau28 = None

    tau128 = np.zeros((N, N, N, N))

    tau128 += np.einsum("abkj,ilab->ijkl", t2, tau127, optimize=True)

    tau127 = None

    tau38 = np.zeros((N, N, M, M))

    tau38 += np.einsum("ijab->ijab", tau25, optimize=True)

    tau69 = np.zeros((N, N, M, M))

    tau69 -= 2 * np.einsum("ijab->ijab", tau25, optimize=True)

    tau25 = None

    tau30 = np.zeros((M, M, M, M))

    tau30 += np.einsum("abji,cdji->abcd", l2, t2, optimize=True)

    tau49 += 2 * np.einsum("acbd,icjd->ijab", tau30, u[o, v, o, v], optimize=True)

    tau74 = np.zeros((N, N, M, M))

    tau74 += np.einsum("acbd,icdj->ijab", tau30, u[o, v, v, o], optimize=True)

    tau104 = np.zeros((N, N, M, M))

    tau104 -= 3 * np.einsum("dacb,ijcd->ijab", tau30, tau7, optimize=True)

    tau31 = np.zeros((M, M, M, M))

    tau31 += np.einsum("abji,cdij->abcd", l2, t2, optimize=True)

    tau49 += 2 * np.einsum("acbd,icdj->ijab", tau31, u[o, v, v, o], optimize=True)

    tau56 = np.zeros((M, M, M, M))

    tau56 += np.einsum("aebf,cefd->abcd", tau31, u[v, v, v, v], optimize=True)

    tau59 += 2 * np.einsum("abcd->abcd", tau56, optimize=True)

    tau108 = np.zeros((N, N, M, M))

    tau108 += 2 * np.einsum("cdij,cbad->ijab", t2, tau56, optimize=True)

    tau56 = None

    tau98 = np.zeros((M, M, M, M))

    tau98 -= np.einsum("aefc,bfed->abcd", tau30, tau31, optimize=True)

    tau32 = np.zeros((N, N, M, M))

    tau32 += np.einsum("acki,cbjk->ijab", l2, t2, optimize=True)

    tau49 += 2 * np.einsum("kjac,ibck->ijab", tau32, u[o, v, v, o], optimize=True)

    tau78 = np.zeros((M, M, M, M))

    tau78 += np.einsum("ijac,jibd->abcd", tau32, tau32, optimize=True)

    tau79 = np.zeros((N, N, M, M))

    tau79 += np.einsum("abcd,ijcd->ijab", tau78, u[o, o, v, v], optimize=True)

    tau78 = None

    tau105 += 2 * np.einsum("ijab->ijab", tau79, optimize=True)

    tau105 -= 3 * np.einsum("ijba->ijab", tau79, optimize=True)

    tau79 = None

    tau98 += 3 * np.einsum("jibc,ijad->abcd", tau32, tau7, optimize=True)

    tau105 -= 2 * np.einsum("bacd,jicd->ijab", tau98, u[o, o, v, v], optimize=True)

    tau98 = None

    tau33 = np.zeros((N, N, M, M))

    tau33 += np.einsum("cdij,dcba->ijab", l2, u[v, v, v, v], optimize=True)

    tau49 += 2 * np.einsum("bcjk,ikac->ijab", t2, tau33, optimize=True)

    tau49 -= 2 * np.einsum("bckj,kica->ijab", t2, tau33, optimize=True)

    tau66 = np.zeros((N, N, M, M))

    tau66 += 2 * np.einsum("ijab->ijab", tau33, optimize=True)

    tau66 += np.einsum("ijba->ijab", tau33, optimize=True)

    tau68 += np.einsum("ackj,ikbc->ijab", t2, tau66, optimize=True)

    tau107 = np.zeros((N, N, M, M))

    tau107 += np.einsum("ackj,kibc->ijab", t2, tau66, optimize=True)

    tau66 = None

    tau155 += np.einsum("ijab->ijab", tau33, optimize=True)

    tau33 = None

    tau158 = np.zeros((M, M))

    tau158 += 2 * np.einsum("acij,ijbc->ab", t2, tau155, optimize=True)

    tau155 = None

    tau34 = np.zeros((N, N, M, M))

    tau34 += 2 * np.einsum("abij->ijab", t2, optimize=True)

    tau34 -= np.einsum("abji->ijab", t2, optimize=True)

    tau35 = np.zeros((N, N, M, M))

    tau35 += np.einsum("ikac,kjcb->ijab", tau34, u[o, o, v, v], optimize=True)

    tau36 += np.einsum("ijab->ijab", tau35, optimize=True)

    tau69 += np.einsum("ijab->ijab", tau35, optimize=True)

    tau35 = None

    tau36 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau37 = np.zeros((N, N, M, M))

    tau37 += np.einsum("bcjk,kica->ijab", l2, tau36, optimize=True)

    tau40 = np.zeros((N, N, M, M))

    tau40 += np.einsum("jiba->ijab", tau37, optimize=True)

    tau49 += 2 * np.einsum("jkca,ikcb->ijab", tau36, tau42, optimize=True)

    tau36 = None

    tau38 -= np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau39 = np.zeros((N, N, M, M))

    tau39 += np.einsum("bckj,kica->ijab", l2, tau38, optimize=True)

    tau40 -= np.einsum("jiba->ijab", tau39, optimize=True)

    tau68 += 6 * np.einsum("acjk,ikcb->ijab", t2, tau40, optimize=True)

    tau87 = np.zeros((N, N, M, M))

    tau87 += np.einsum("jiba->ijab", tau39, optimize=True)

    tau39 = None

    tau49 -= 2 * np.einsum("jkca,ikcb->ijab", tau38, tau4, optimize=True)

    tau115 -= 6 * np.einsum("ijab,lkab->ijkl", tau110, tau38, optimize=True)

    tau110 = None

    tau158 += 2 * np.einsum("ijcb,ijca->ab", tau38, tau42, optimize=True)

    tau42 = None

    tau41 = np.zeros((N, N, M, M))

    tau41 -= np.einsum("abij->ijab", t2, optimize=True)

    tau41 += 2 * np.einsum("baij->ijab", t2, optimize=True)

    tau49 -= 2 * np.einsum("ikca,jkcb->ijab", tau40, tau41, optimize=True)

    tau40 = None

    tau158 += 2 * np.einsum("jibc,ijac->ab", tau37, tau41, optimize=True)

    tau44 = np.zeros((N, N, M, M))

    tau44 -= np.einsum("abij->ijab", t2, optimize=True)

    tau44 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau45 = np.zeros((N, N, M, M))

    tau45 += np.einsum("caki,jkcb->ijab", l2, tau44, optimize=True)

    tau49 += 2 * np.einsum("kibc,kjac->ijab", tau38, tau45, optimize=True)

    tau45 = None

    tau38 = None

    tau168 += 2 * np.einsum("jkba,ikab->ij", tau37, tau44, optimize=True)

    tau37 = None

    tau46 = np.zeros((N, N, M, M))

    tau46 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau46 -= np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau47 = np.zeros((N, N, M, M))

    tau47 += np.einsum("bcjk,kiac->ijab", t2, tau46, optimize=True)

    tau46 = None

    tau48 -= np.einsum("jiba->ijab", tau47, optimize=True)

    tau49 -= 2 * np.einsum("kjac,kibc->ijab", tau23, tau48, optimize=True)

    tau48 = None

    tau75 = np.zeros((N, N, M, M))

    tau75 += 3 * np.einsum("bcjk,kica->ijab", t2, tau49, optimize=True)

    tau49 = None

    tau72 = np.zeros((N, N, M, M))

    tau72 -= 2 * np.einsum("jiba->ijab", tau47, optimize=True)

    tau95 = np.zeros((N, N, M, M))

    tau95 -= np.einsum("jiba->ijab", tau47, optimize=True)

    tau47 = None

    tau50 = np.zeros((M, M, M, M))

    tau50 += np.einsum("abji,jicd->abcd", t2, u[o, o, v, v], optimize=True)

    tau55 = np.zeros((M, M, M, M))

    tau55 += 2 * np.einsum("eacf,befd->abcd", tau30, tau50, optimize=True)

    tau58 = np.zeros((M, M, M, M))

    tau58 -= 2 * np.einsum("abcd->abcd", tau50, optimize=True)

    tau58 += 3 * np.einsum("abdc->abcd", tau50, optimize=True)

    tau50 = None

    tau59 -= np.einsum("eacf,befd->abcd", tau31, tau58, optimize=True)

    tau58 = None

    tau51 = np.zeros((M, M, M, M))

    tau51 -= 2 * np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau51 += 3 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau55 -= np.einsum("aebf,ecfd->abcd", tau30, tau51, optimize=True)

    tau51 = None

    tau52 = np.zeros((N, N, M, M))

    tau52 += 3 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau52 -= 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau53 = np.zeros((N, N, M, M))

    tau53 -= np.einsum("acki,kjcb->ijab", t2, tau52, optimize=True)

    tau52 = None

    tau53 += 3 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau53 -= 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau54 = np.zeros((N, N, M, M))

    tau54 += np.einsum("acki,bckj->ijab", l2, t2, optimize=True)

    tau55 += 2 * np.einsum("ijcd,ijab->abcd", tau53, tau54, optimize=True)

    tau53 = None

    tau75 += np.einsum("cdji,cbad->ijab", t2, tau55, optimize=True)

    tau55 = None

    tau80 -= np.einsum("ijab->ijab", tau54, optimize=True)

    tau81 -= np.einsum("ijcd,ijab->abcd", tau12, tau80, optimize=True)

    tau12 = None

    tau105 += 2 * np.einsum("cdji,bcda->ijab", l2, tau81, optimize=True)

    tau81 = None

    tau100 = np.zeros((N, N, M, M))

    tau100 += np.einsum("dacb,ijcd->ijab", tau31, tau80, optimize=True)

    tau80 = None

    tau99 = np.zeros((N, N, M, M))

    tau99 += np.einsum("cbik,kjca->ijab", t2, tau54, optimize=True)

    tau100 += 3 * np.einsum("caki,kjcb->ijab", l2, tau99, optimize=True)

    tau99 = None

    tau123 += 2 * np.einsum("ijab->ijab", tau54, optimize=True)

    tau124 -= 3 * np.einsum("jlab,ikba->ijkl", tau123, tau3, optimize=True)

    tau123 = None

    tau134 = np.zeros((N, N, N, N))

    tau134 += np.einsum("ikab,jlba->ijkl", tau54, tau54, optimize=True)

    tau135 -= np.einsum("ijlk->ijkl", tau134, optimize=True)

    tau138 += 2 * np.einsum("ijlk->ijkl", tau134, optimize=True)

    tau134 = None

    tau150 = np.zeros((N, N, M, M))

    tau150 += np.einsum("bckj,kica->ijab", t2, tau54, optimize=True)

    tau151 = np.zeros((N, N, N, N))

    tau151 += np.einsum("baij,klba->ijkl", l2, tau150, optimize=True)

    tau153 = np.zeros((N, N, N, N))

    tau153 += np.einsum("baij,klab->ijkl", l2, tau150, optimize=True)

    tau150 = None

    e += np.einsum("ijkl,ijkl->", tau153, tau154, optimize=True) / 3

    tau153 = None

    tau154 = None

    tau57 = np.zeros((N, N, M, M))

    tau57 += np.einsum("acik,bckj->ijab", l2, t2, optimize=True)

    tau59 += 4 * np.einsum("ijcd,ijab->abcd", tau15, tau57, optimize=True)

    tau75 += np.einsum("cdij,cbad->ijab", t2, tau59, optimize=True)

    tau59 = None

    tau100 -= np.einsum("dacb,ijcd->ijab", tau30, tau57, optimize=True)

    tau105 -= 2 * np.einsum("jkbc,kica->ijab", tau100, u[o, o, v, v], optimize=True)

    tau100 = None

    tau130 = np.zeros((N, N, M, M))

    tau130 += np.einsum("acki,kjcb->ijab", t2, tau57, optimize=True)

    tau131 = np.zeros((N, N, N, N))

    tau131 += np.einsum("baij,klba->ijkl", l2, tau130, optimize=True)

    tau130 = None

    tau132 += np.einsum("ijkl->ijkl", tau131, optimize=True)

    tau140 -= np.einsum("ijkl->ijkl", tau131, optimize=True)

    tau131 = None

    tau138 -= 6 * np.einsum("jkba,ilab->ijkl", tau57, tau7, optimize=True)

    tau7 = None

    tau60 = np.zeros((N, N, M, M))

    tau60 += np.einsum("abij->ijab", l2, optimize=True)

    tau60 += np.einsum("abji->ijab", l2, optimize=True)

    tau61 += np.einsum("cbjk,ikca->ijab", t2, tau60, optimize=True)

    tau68 += 2 * np.einsum("ijcd,cabd->ijab", tau61, u[v, v, v, v], optimize=True)

    tau61 = None

    tau137 = np.zeros((N, N, M, M))

    tau137 += np.einsum("bckj,ikac->ijab", t2, tau60, optimize=True)

    tau60 = None

    tau138 += 2 * np.einsum("jkab,ilba->ijkl", tau137, tau57, optimize=True)

    tau57 = None

    tau137 = None

    e += np.einsum("ijkl,ijkl->", tau138, tau9, optimize=True) / 3

    tau9 = None

    tau138 = None

    tau62 = np.zeros((N, N, M, M))

    tau62 -= np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau62 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau63 = np.zeros((N, N, M, M))

    tau63 += np.einsum("bcjk,kica->ijab", t2, tau62, optimize=True)

    tau62 = None

    tau65 = np.zeros((N, N, M, M))

    tau65 += 2 * np.einsum("jiba->ijab", tau63, optimize=True)

    tau86 = np.zeros((N, N, M, M))

    tau86 += 3 * np.einsum("jiba->ijab", tau63, optimize=True)

    tau89 = np.zeros((N, N, M, M))

    tau89 += 2 * np.einsum("acbd,jidc->ijab", tau30, tau63, optimize=True)

    tau30 = None

    tau115 += 6 * np.einsum("ilab,kjba->ijkl", tau3, tau63, optimize=True)

    tau3 = None

    tau118 = np.zeros((N, N, N, N))

    tau118 += np.einsum("klab,jiba->ijkl", tau23, tau63, optimize=True)

    tau120 = np.zeros((N, N, N, N))

    tau120 -= 2 * np.einsum("lkij->ijkl", tau118, optimize=True)

    tau120 -= np.einsum("jkil->ijkl", tau118, optimize=True)

    tau118 = None

    tau64 = np.zeros((N, N, M, M))

    tau64 += np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau64 += 2 * np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau65 -= np.einsum("acki,kjcb->ijab", t2, tau64, optimize=True)

    tau95 += np.einsum("acki,kjbc->ijab", t2, tau64, optimize=True)

    tau64 = None

    tau97 = np.zeros((N, N, M, M))

    tau97 -= np.einsum("kiac,kjbc->ijab", tau23, tau95, optimize=True)

    tau95 = None

    tau65 += np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau65 += 2 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau68 -= 2 * np.einsum("ikca,jkcb->ijab", tau6, tau65, optimize=True)

    tau65 = None

    tau6 = None

    tau67 += 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau68 -= 2 * np.einsum("ikca,jkcb->ijab", tau4, tau67, optimize=True)

    tau67 = None

    tau4 = None

    tau75 += np.einsum("bckj,kiac->ijab", t2, tau68, optimize=True)

    tau68 = None

    tau69 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau70 = np.zeros((N, N, M, M))

    tau70 += np.einsum("caki,cbjk->ijab", l2, t2, optimize=True)

    tau74 += np.einsum("kibc,kjac->ijab", tau69, tau70, optimize=True)

    tau69 = None

    tau119 = np.zeros((N, N, N, N))

    tau119 += np.einsum("jiba,klab->ijkl", tau63, tau70, optimize=True)

    tau120 -= np.einsum("lkij->ijkl", tau119, optimize=True)

    tau120 -= 2 * np.einsum("jkil->ijkl", tau119, optimize=True)

    tau119 = None

    tau141 = np.zeros((N, N, M, M))

    tau141 += np.einsum("cbjk,kica->ijab", t2, tau70, optimize=True)

    tau142 = np.zeros((N, N, N, N))

    tau142 += np.einsum("ijab,lkba->ijkl", tau141, u[o, o, v, v], optimize=True)

    tau141 = None

    tau145 += np.einsum("ijkl->ijkl", tau142, optimize=True)

    tau145 += 2 * np.einsum("ijlk->ijkl", tau142, optimize=True)

    tau142 = None

    e += np.einsum("ijlk,ijkl->", tau0, tau145, optimize=True) / 3

    tau145 = None

    tau71 = np.zeros((N, N, M, M))

    tau71 += 2 * np.einsum("jiab->ijab", u[o, o, v, v], optimize=True)

    tau71 += np.einsum("jiba->ijab", u[o, o, v, v], optimize=True)

    tau72 += np.einsum("acki,kjbc->ijab", t2, tau71, optimize=True)

    tau71 = None

    tau74 -= np.einsum("kjac,kibc->ijab", tau23, tau72, optimize=True)

    tau72 = None

    tau73 = np.zeros((N, N, M, M))

    tau73 += np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau73 += 2 * np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau74 += np.einsum("kjac,ikbc->ijab", tau32, tau73, optimize=True)

    tau73 = None

    tau75 -= 2 * np.einsum("cbjk,kica->ijab", t2, tau74, optimize=True)

    tau74 = None

    e += np.einsum("baji,ijab->", l2, tau75, optimize=True) / 3

    tau75 = None

    tau82 = np.zeros((N, N, M, M))

    tau82 += np.einsum("acik,cbjk->ijab", l2, t2, optimize=True)

    tau85 = np.zeros((M, M, M, M))

    tau85 += np.einsum("jibc,ijad->abcd", tau32, tau82, optimize=True)

    tau83 = np.zeros((N, N, M, M))

    tau83 -= np.einsum("abij->ijab", t2, optimize=True)

    tau83 += 3 * np.einsum("abji->ijab", t2, optimize=True)

    tau84 = np.zeros((N, N, M, M))

    tau84 += np.einsum("acik,jkcb->ijab", l2, tau83, optimize=True)

    tau83 = None

    tau85 -= np.einsum("jibc,ijad->abcd", tau82, tau84, optimize=True)

    tau82 = None

    tau84 = None

    tau105 += 2 * np.einsum("bacd,jidc->ijab", tau85, u[o, o, v, v], optimize=True)

    tau85 = None

    tau86 += 3 * np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau86 -= 2 * np.einsum("jaib->ijab", u[o, v, o, v], optimize=True)

    tau87 += np.einsum("acik,kjcb->ijab", l2, tau86, optimize=True)

    tau86 = None

    tau89 -= np.einsum("bcki,kjac->ijab", t2, tau87, optimize=True)

    tau87 = None

    tau88 += np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau89 += np.einsum("acbd,ijcd->ijab", tau31, tau88, optimize=True)

    tau88 = None

    tau31 = None

    tau105 -= 2 * np.einsum("caik,kjbc->ijab", l2, tau89, optimize=True)

    tau89 = None

    tau90 = np.zeros((N, N, M, M))

    tau90 += np.einsum("abij->ijab", t2, optimize=True)

    tau90 += 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau91 = np.zeros((M, M, M, M))

    tau91 += np.einsum("abij,ijcd->abcd", l2, tau90, optimize=True)

    tau90 = None

    tau94 = np.zeros((N, N, M, M))

    tau94 += np.einsum("jidc,acbd->ijab", tau63, tau91, optimize=True)

    tau91 = None

    tau63 = None

    tau92 = np.zeros((N, N, M, M))

    tau92 += np.einsum("acik,jckb->ijab", l2, u[o, v, o, v], optimize=True)

    tau93 -= np.einsum("ijab->ijab", tau92, optimize=True)

    tau94 -= np.einsum("bcki,kjac->ijab", t2, tau93, optimize=True)

    tau93 = None

    tau105 -= 2 * np.einsum("caki,kjbc->ijab", l2, tau94, optimize=True)

    tau94 = None

    tau126 = np.zeros((N, N, N, N))

    tau126 += np.einsum("abjk,ilab->ijkl", t2, tau92, optimize=True)

    tau92 = None

    tau128 += 2 * np.einsum("ijkl->ijkl", tau126, optimize=True)

    tau128 += np.einsum("ikjl->ijkl", tau126, optimize=True)

    tau126 = None

    tau96 = np.zeros((N, N, M, M))

    tau96 += np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau96 += 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau97 += np.einsum("kiac,jkbc->ijab", tau32, tau96, optimize=True)

    tau32 = None

    tau96 = None

    tau105 -= 2 * np.einsum("acki,kjbc->ijab", l2, tau97, optimize=True)

    tau97 = None

    tau101 = np.zeros((N, N, M, M))

    tau101 += 2 * np.einsum("abij->ijab", l2, optimize=True)

    tau101 += np.einsum("abji->ijab", l2, optimize=True)

    tau102 = np.zeros((N, N, M, M))

    tau102 += np.einsum("cbjk,ikca->ijab", t2, tau101, optimize=True)

    tau101 = None

    tau103 = np.zeros((N, N, M, M))

    tau103 += np.einsum("ackj,kicb->ijab", t2, tau102, optimize=True)

    tau102 = None

    tau104 += np.einsum("caki,kjcb->ijab", l2, tau103, optimize=True)

    tau103 = None

    tau105 += 2 * np.einsum("jkbc,kiac->ijab", tau104, u[o, o, v, v], optimize=True)

    tau104 = None

    e += np.einsum("baij,ijab->", t2, tau105, optimize=True) / 3

    tau105 = None

    tau106 = np.zeros((M, M, M, M))

    tau106 += 3 * np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau106 -= 2 * np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau107 -= np.einsum("cabd,ijcd->ijab", tau106, tau23, optimize=True)

    tau106 = None

    tau108 += np.einsum("bckj,kiac->ijab", t2, tau107, optimize=True)

    tau107 = None

    e += np.einsum("baij,ijab->", l2, tau108, optimize=True) / 3

    tau108 = None

    tau111 -= np.einsum("jabi->ijab", u[o, v, v, o], optimize=True)

    tau115 += 4 * np.einsum("lkab,ijab->ijkl", tau111, tau23, optimize=True)

    tau23 = None

    tau152 = np.zeros((N, N, N, N))

    tau152 -= np.einsum("lkab,ijab->ijkl", tau111, tau70, optimize=True)

    tau70 = None

    tau111 = None

    e -= 2 * np.einsum("jlik,ijkl->", tau0, tau152, optimize=True) / 3

    tau152 = None

    tau112 = np.zeros((N, N, M, M))

    tau112 += 3 * np.einsum("abij->ijab", t2, optimize=True)

    tau112 -= 2 * np.einsum("abji->ijab", t2, optimize=True)

    tau113 = np.zeros((N, N, M, M))

    tau113 += np.einsum("caki,kjcb->ijab", l2, tau112, optimize=True)

    tau112 = None

    tau115 -= 2 * np.einsum("ilab,jkab->ijkl", tau113, tau15, optimize=True)

    tau113 = None

    tau15 = None

    tau114 -= 2 * np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau114 += 3 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau115 -= np.einsum("injm,mknl->ijkl", tau0, tau114, optimize=True)

    tau114 = None

    e += np.einsum("jlik,ijkl->", tau0, tau115, optimize=True) / 3

    tau115 = None

    tau116 = np.zeros((N, N, N, N))

    tau116 += np.einsum("abij,bakl->ijkl", l2, t2, optimize=True)

    tau117 = np.zeros((N, N, N, N))

    tau117 += np.einsum("imjn,nklm->ijkl", tau116, u[o, o, o, o], optimize=True)

    tau120 += np.einsum("ijkl->ijkl", tau117, optimize=True)

    e += 2 * np.einsum("jlik,ijkl->", tau0, tau120, optimize=True) / 3

    tau120 = None

    tau148 = np.zeros((N, N, N, N))

    tau148 += np.einsum("ijkl->ijkl", tau117, optimize=True)

    tau117 = None

    tau121 = np.zeros((N, N, M, M))

    tau121 += np.einsum("abij->ijab", l2, optimize=True)

    tau121 += np.einsum("baij->ijab", l2, optimize=True)

    tau122 = np.zeros((N, N, M, M))

    tau122 += np.einsum("bckj,kica->ijab", t2, tau121, optimize=True)

    tau121 = None

    tau124 += 2 * np.einsum("jkab,ilba->ijkl", tau122, tau54, optimize=True)

    tau122 = None

    tau54 = None

    e += np.einsum("ijkl,lkji->", tau124, u[o, o, o, o], optimize=True) / 3

    tau124 = None

    tau125 = np.zeros((N, N, M, M))

    tau125 += np.einsum("acik,jcbk->ijab", l2, u[o, v, v, o], optimize=True)

    tau128 -= 3 * np.einsum("abjk,ilab->ijkl", t2, tau125, optimize=True)

    tau125 = None

    e -= 2 * np.einsum("jkil,ijkl->", tau0, tau128, optimize=True) / 3

    tau128 = None

    tau129 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau168 += 2 * np.einsum("mlik,kjlm->ij", tau0, tau129, optimize=True)

    tau0 = None

    e += 2 * np.einsum("ijkl,lkji->", tau129, tau132, optimize=True) / 3

    tau129 = None

    tau132 = None

    tau136 += np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    e += np.einsum("ijkl,lkij->", tau135, tau136, optimize=True)

    tau135 = None

    e -= np.einsum("ijkl,klji->", tau136, tau140, optimize=True) / 3

    tau136 = None

    tau140 = None

    tau146 = np.zeros((N, N, M, M))

    tau146 += np.einsum("abij->ijab", l2, optimize=True)

    tau146 += 2 * np.einsum("abji->ijab", l2, optimize=True)

    tau147 += np.einsum("cbjk,ikca->ijab", t2, tau146, optimize=True)

    tau146 = None

    tau148 -= np.einsum("ijab,kabl->ijkl", tau147, u[o, v, v, o], optimize=True)

    tau147 = None

    e += 2 * np.einsum("jlik,ijkl->", tau116, tau148, optimize=True) / 3

    tau148 = None

    tau116 = None

    tau149 = np.zeros((N, N, N, N))

    tau149 += np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau149 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    e += np.einsum("ijkl,klji->", tau149, tau151, optimize=True) / 3

    tau149 = None

    tau151 = None

    tau156 = np.zeros((M, M))

    tau156 += np.einsum("caji,cbji->ab", l2, t2, optimize=True)

    tau162 = np.zeros((N, N, M, M))

    tau162 += np.einsum("cb,caij->ijab", tau156, t2, optimize=True)

    tau164 += 2 * np.einsum("ijba->ijab", tau162, optimize=True)

    tau164 -= 4 * np.einsum("jiba->ijab", tau162, optimize=True)

    tau162 = None

    tau165 = np.zeros((N, N, M, M))

    tau165 += np.einsum("ac,ijbc->ijab", tau156, u[o, o, v, v], optimize=True)

    tau168 -= 2 * np.einsum("kjab,ikab->ij", tau165, tau34, optimize=True)

    tau34 = None

    tau165 = None

    tau157 = np.zeros((M, M, M, M))

    tau157 += 2 * np.einsum("bacd->abcd", u[v, v, v, v], optimize=True)

    tau157 -= np.einsum("badc->abcd", u[v, v, v, v], optimize=True)

    tau158 -= np.einsum("cd,cabd->ab", tau156, tau157, optimize=True)

    tau157 = None

    e -= np.einsum("ab,ab->", tau156, tau158, optimize=True)

    tau158 = None

    tau159 = np.zeros((M, M))

    tau159 += np.einsum("acji,cbij->ab", l2, t2, optimize=True)

    tau160 = np.zeros((N, N, M, M))

    tau160 += np.einsum("cb,ijac->ijab", tau159, tau41, optimize=True)

    tau41 = None

    tau161 = np.zeros((M, M))

    tau161 += np.einsum("ijac,ijcb->ab", tau160, u[o, o, v, v], optimize=True)

    tau160 = None

    e += np.einsum("ab,ab->", tau159, tau161, optimize=True)

    tau161 = None

    tau159 = None

    tau163 = np.zeros((N, N))

    tau163 += np.einsum("baki,bakj->ij", l2, t2, optimize=True)

    tau164 -= np.einsum("kj,ikab->ijab", tau163, tau44, optimize=True)

    tau44 = None

    tau168 += np.einsum("ikab,kjab->ij", tau164, u[o, o, v, v], optimize=True)

    tau164 = None

    tau166 = np.zeros((N, N, M, M))

    tau166 -= np.einsum("iabj->ijab", u[o, v, v, o], optimize=True)

    tau166 += 2 * np.einsum("iajb->ijab", u[o, v, o, v], optimize=True)

    tau168 += 2 * np.einsum("ab,jiab->ij", tau156, tau166, optimize=True)

    tau156 = None

    tau166 = None

    tau167 = np.zeros((N, N, N, N))

    tau167 -= np.einsum("jikl->ijkl", u[o, o, o, o], optimize=True)

    tau167 += 2 * np.einsum("jilk->ijkl", u[o, o, o, o], optimize=True)

    tau168 -= np.einsum("lk,kjli->ij", tau163, tau167, optimize=True)

    tau167 = None

    e -= np.einsum("ij,ij->", tau163, tau168, optimize=True)

    tau168 = None

    tau163 = None

    return e
