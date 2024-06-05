import numpy as np


def lambda1_amplitudes_ccsd(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    r1 = np.zeros((M, N), dtype=u.dtype)

    r1 += np.einsum("bcij,bdjk,kcad->ai", l2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bcjk,bdjk,icad->ai", l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("abjk,cdjk,ibcd->ai", l2, t2, u[o, v, v, v], optimize=True) / 4

    r1 -= np.einsum("bk,dj,bcij,kcad->ai", t1, t1, l2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("cj,dk,abjk,ibcd->ai", t1, t1, l2, u[o, v, v, v], optimize=True) / 2

    r1 += np.einsum("bcij,bcaj->ai", l2, u[v, v, v, o], optimize=True) / 2

    r1 += np.einsum("bj,cj,ibac->ai", l1, t1, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,jbac->ai", l1, t1, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bl,abjk,iljk->ai", t1, l2, u[o, o, o, o], optimize=True) / 2

    r1 += np.einsum("bl,bcjk,cdjk,ilad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("dj,bcjk,bckl,ilad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bk,bcij,cdjl,klad->ai", t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("cj,abjk,bdkl,ilcd->ai", t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("cl,abjk,bdjk,ilcd->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("dk,bcij,bcjl,klad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bl,abjk,cdjk,ilcd->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("dj,bcij,bckl,klad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("bk,cl,dj,bcij,klad->ai", t1, t1, t1, l2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bl,cj,dk,abjk,ilcd->ai", t1, t1, t1, l2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bj,ibaj->ai", l1, u[o, v, v, o], optimize=True)

    r1 += np.einsum("bj,ijab->ai", t1, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("aj,ib,bj->ai", l1, f[o, v], t1, optimize=True)

    r1 -= np.einsum("bi,ja,bj->ai", l1, f[o, v], t1, optimize=True)

    r1 += np.einsum("abjk,bcjl,ilkc->ai", l2, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bcjk,bcjl,ilak->ai", l2, t2, u[o, o, v, o], optimize=True) / 2

    r1 += np.einsum("bcij,bckl,klaj->ai", l2, t2, u[o, o, v, o], optimize=True) / 4

    r1 += np.einsum("bk,cl,bcij,klaj->ai", t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r1 -= np.einsum("bl,cj,abjk,ilkc->ai", t1, t1, l2, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bj,bcjk,ikac->ai", l1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("cj,abjk,ibkc->ai", t1, l2, u[o, v, o, v], optimize=True)

    r1 += np.einsum("ic,abjk,bcjk->ai", f[o, v], l2, t2, optimize=True) / 2

    r1 += np.einsum("ka,bcij,bcjk->ai", f[o, v], l2, t2, optimize=True) / 2

    r1 -= np.einsum("bk,bcij,kcaj->ai", t1, l2, u[o, v, v, o], optimize=True)

    r1 -= np.einsum("aj,bcjk,ikbc->ai", l1, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bi,bcjk,jkac->ai", l1, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("aj,bj,ck,ikbc->ai", l1, t1, t1, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bi,bj,ck,jkac->ai", l1, t1, t1, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bj,bk,cj,ikac->ai", l1, t1, t1, u[o, o, v, v], optimize=True)

    r1 += np.einsum("ia->ai", f[o, v], optimize=True)

    r1 -= np.einsum("abjk,ibjk->ai", l2, u[o, v, o, o], optimize=True) / 2

    r1 -= np.einsum("aj,bk,ikjb->ai", l1, t1, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bj,bk,ikaj->ai", l1, t1, u[o, o, v, o], optimize=True)

    r1 += np.einsum("dj,bcij,bcad->ai", t1, l2, u[v, v, v, v], optimize=True) / 2

    r1 -= np.einsum("aj,ij->ai", l1, f[o, o], optimize=True)

    r1 += np.einsum("bi,ba->ai", l1, f[v, v], optimize=True)

    return r1


def lambda2_amplitudes_ccsd(t1, t2, l1, l2, u, f, v, o):
    M, N = t1.shape

    r2 = np.zeros((M, M, N, N), dtype=u.dtype)

    r2 += np.einsum("ak,ijbk->abij", l1, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("bk,ijak->abij", l1, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ik,abkj->abij", f[o, o], l2, optimize=True)

    r2 -= np.einsum("jk,abik->abij", f[o, o], l2, optimize=True)

    r2 += np.einsum("abkl,ijkl->abij", l2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("cdij,cdab->abij", l2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("ackj,cdkl,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bcik,cdkl,ljad->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("abik,cdkl,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bcij,cdkl,klad->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bckl,cdkl,ijad->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("cdik,cdkl,ljab->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,cdkl,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bckj,cdkl,ilad->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abkj,cdkl,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acij,cdkl,klbd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ackl,cdkl,ijbd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("cdkj,cdkl,ilab->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("abkl,cdkl,ijcd->abij", l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("cdij,cdkl,klab->abij", l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("ck,dl,abik,ljcd->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,bcij,klad->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cl,dk,acik,ljbd->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cl,dk,bckj,ilad->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,abkl,ijcd->abij", t1, t1, l2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,dl,cdij,klab->abij", t1, t1, l2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,dl,abkj,ilcd->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,acij,klbd->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cl,dk,ackj,ilbd->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cl,dk,bcik,ljad->abij", t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ackj,icbk->abij", l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("bcik,cjak->abij", l2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("acik,cjbk->abij", l2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bckj,icak->abij", l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ak,ck,ijbc->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bi,ck,kjac->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bj,ck,ikac->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ka,ck,bcij->abij", f[o, v], t1, l2, optimize=True)

    r2 -= np.einsum("ai,ck,kjbc->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aj,ck,ikbc->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,ck,ijac->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ci,ck,kjab->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cj,ck,ikab->abij", l1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ic,ck,abkj->abij", f[o, v], t1, l2, optimize=True)

    r2 -= np.einsum("jc,ck,abik->abij", f[o, v], t1, l2, optimize=True)

    r2 -= np.einsum("kb,ck,acij->abij", f[o, v], t1, l2, optimize=True)

    r2 += np.einsum("ai,jb->abij", l1, f[o, v], optimize=True)

    r2 += np.einsum("bj,ia->abij", l1, f[o, v], optimize=True)

    r2 -= np.einsum("aj,ib->abij", l1, f[o, v], optimize=True)

    r2 -= np.einsum("bi,ja->abij", l1, f[o, v], optimize=True)

    r2 += np.einsum("ijab->abij", u[o, o, v, v], optimize=True)

    r2 += np.einsum("dk,ackj,icbd->abij", t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("dk,bcij,kcad->abij", t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("dk,bcik,cjad->abij", t1, l2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cdij,kdab->abij", t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("dk,acij,kcbd->abij", t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("dk,acik,cjbd->abij", t1, l2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("dk,bckj,icad->abij", t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ci,cjab->abij", l1, u[v, o, v, v], optimize=True)

    r2 += np.einsum("cj,icab->abij", l1, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cb,acij->abij", f[v, v], l2, optimize=True)

    r2 -= np.einsum("ca,bcij->abij", f[v, v], l2, optimize=True)

    r2 += np.einsum("cl,abik,ljkc->abij", t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cl,acik,ljbk->abij", t1, l2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cl,bckj,ilak->abij", t1, l2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,abkl,ijlc->abij", t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cl,abkj,ilkc->abij", t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cl,ackj,ilbk->abij", t1, l2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cl,bcik,ljak->abij", t1, l2, u[o, o, v, o], optimize=True)

    return r2


def lambda_amplitudes_ccsd(t1, t2, l1, l2, u, f, v, o):
    r1 = lambda1_amplitudes_ccsd(t1, t2, l1, l2, u, f, v, o)
    r2 = lambda2_amplitudes_ccsd(t1, t2, l1, l2, u, f, v, o)

    return r1, r2
