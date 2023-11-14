import numpy as np


def amplitudes_CCSD_t1(t1, t2, u, f, v, o):
    M, N = t1.shape

    r1 = np.zeros((M, N), dtype=u.dtype)

    r1 += np.einsum("jb,abij->ai", f[o, v], t2, optimize=True)

    r1 += np.einsum("bj,ajib->ai", t1, u[v, o, o, v], optimize=True)

    r1 -= np.einsum("jb,aj,bi->ai", f[o, v], t1, t1, optimize=True)

    r1 += np.einsum("bj,acik,jkbc->ai", t1, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("aj,bcik,jkbc->ai", t1, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bi,acjk,jkbc->ai", t1, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("aj,bi,ck,jkbc->ai", t1, t1, t1, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bcij,ajbc->ai", t2, u[v, o, v, v], optimize=True) / 2

    r1 += np.einsum("bi,cj,ajbc->ai", t1, t1, u[v, o, v, v], optimize=True)

    r1 -= np.einsum("ji,aj->ai", f[o, o], t1, optimize=True)

    r1 += np.einsum("ab,bi->ai", f[v, v], t1, optimize=True)

    r1 += np.einsum("ai->ai", f[v, o], optimize=True)

    r1 -= np.einsum("abjk,jkib->ai", t2, u[o, o, o, v], optimize=True) / 2

    r1 -= np.einsum("aj,bk,jkib->ai", t1, t1, u[o, o, o, v], optimize=True)

    return r1


def amplitudes_CCSD_t2(t1, t2, u, f, v, o):
    M, N = t1.shape

    r2 = np.zeros((M, M, N, N), u.dtype)

    r2 += np.einsum("ak,bkij->abij", t1, u[v, o, o, o], optimize=True)

    r2 -= np.einsum("ki,abkj->abij", f[o, o], t2, optimize=True)

    r2 -= np.einsum("kj,abik->abij", f[o, o], t2, optimize=True)

    r2 -= np.einsum("bk,akij->abij", t1, u[v, o, o, o], optimize=True)

    r2 += np.einsum("ackj,bdil,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ablj,cdik,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("adij,bckl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,bdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abil,cdkj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ackl,bdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("abkl,cdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("ak,ci,bdlj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,cj,bdil,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,cl,adij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,dk,ablj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,bl,cdij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dj,abkl,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,cl,bdij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,ci,adlj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cj,adil,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dj,abil,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,bl,ci,dj,klcd->abij", t1, t1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ackj,bkic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("bcik,akcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("acik,bkcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bckj,akic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("kc,ak,bcij->abij", f[o, v], t1, t2, optimize=True)

    r2 += np.einsum("ak,ci,bkcj->abij", t1, t1, u[v, o, v, o], optimize=True)

    r2 += np.einsum("ak,cj,bkic->abij", t1, t1, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("kc,bk,acij->abij", f[o, v], t1, t2, optimize=True)

    r2 -= np.einsum("kc,ci,abkj->abij", f[o, v], t1, t2, optimize=True)

    r2 -= np.einsum("kc,cj,abik->abij", f[o, v], t1, t2, optimize=True)

    r2 -= np.einsum("bk,ci,akcj->abij", t1, t1, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bk,cj,akic->abij", t1, t1, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ci,adkj,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("cj,adik,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,bdij,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ak,cdij,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ci,bdkj,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("cj,bdik,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,adij,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cdij,akcd->abij", t1, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ak,ci,dj,bkcd->abij", t1, t1, t1, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("bk,ci,dj,akcd->abij", t1, t1, t1, u[v, o, v, v], optimize=True)

    r2 += np.einsum("bc,acij->abij", f[v, v], t2, optimize=True)

    r2 += np.einsum("ci,abcj->abij", t1, u[v, v, v, o], optimize=True)

    r2 += np.einsum("cj,abic->abij", t1, u[v, v, o, v], optimize=True)

    r2 -= np.einsum("ac,bcij->abij", f[v, v], t2, optimize=True)

    r2 += np.einsum("abij->abij", u[v, v, o, o], optimize=True)

    r2 += np.einsum("cdij,abcd->abij", t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dj,abcd->abij", t1, t1, u[v, v, v, v], optimize=True)

    r2 += np.einsum("abkl,klij->abij", t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("ak,bl,klij->abij", t1, t1, u[o, o, o, o], optimize=True)

    r2 += np.einsum("ak,bclj,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,acil,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,ablj,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ci,abkl,klcj->abij", t1, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("cj,abkl,klic->abij", t1, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ak,bcil,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("bk,aclj,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,abil,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ak,bl,ci,klcj->abij", t1, t1, t1, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ak,bl,cj,klic->abij", t1, t1, t1, u[o, o, o, v], optimize=True)

    return r2


def amplitudes_ccsd(t1, t2, u, f, v, o):
    r1 = amplitudes_CCSD_t1(t1, t2, u, f, v, o)
    r2 = amplitudes_CCSD_t2(t1, t2, u, f, v, o)

    return r1, r2
