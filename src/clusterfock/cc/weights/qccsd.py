import numpy as np


def reference_addition_qccsd(t1, t2, l1, l2):
    det = np.einsum("ai,bj,bcjk,acik->", l1, t1, l2, t2, optimize=True)

    det -= np.einsum("ai,aj,bcjk,bcik->", l1, t1, l2, t2, optimize=True) / 2

    det -= np.einsum("ai,bi,bcjk,acjk->", l1, t1, l2, t2, optimize=True) / 2

    det += np.einsum("ai,ai,bcjk,bcjk->", l1, t1, l2, t2, optimize=True) / 4

    det += np.einsum("ai,ai,bj,ck,bcjk->", l1, t1, t1, t1, l2, optimize=True) / 2

    det -= np.einsum("ai,aj,bi,ck,bcjk->", l1, t1, t1, t1, l2, optimize=True)

    det -= np.einsum("ai,bj,abij->", l1, l1, t2, optimize=True) / 2

    det += np.einsum("ai,bj,ai,bj->", l1, l1, t1, t1, optimize=True) / 2

    det -= np.einsum("ai,bj,aj,bi->", l1, l1, t1, t1, optimize=True) / 2

    det -= np.einsum("abjk,cdil,abjl,cdik->", l2, l2, t2, t2, optimize=True) / 8

    det -= np.einsum("abjk,cdil,acjk,bdil->", l2, l2, t2, t2, optimize=True) / 8

    det += np.einsum("abjk,cdil,acjl,bdik->", l2, l2, t2, t2, optimize=True) / 4

    det += np.einsum("abjk,cdil,abil,cdjk->", l2, l2, t2, t2, optimize=True) / 32

    det += np.einsum("abjk,cdil,abjk,cdil->", l2, l2, t2, t2, optimize=True) / 32

    det += np.einsum("ai,bj,ck,dl,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 2

    det += np.einsum("ai,bl,cj,dk,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 8

    det += np.einsum("aj,bk,ci,dl,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 8

    return det


def bra_singles_addition_qccsd(t1, t2, l1, l2):
    bra = np.einsum("bj,adil,bcjk,cdkl->ai", t1, l2, l2, t2, optimize=True)

    bra += np.einsum("bj,adkl,bcij,cdkl->ai", t1, l2, l2, t2, optimize=True) / 2

    bra += np.einsum("bl,adil,bcjk,cdjk->ai", t1, l2, l2, t2, optimize=True) / 2

    bra += np.einsum("dj,adil,bcjk,bckl->ai", t1, l2, l2, t2, optimize=True) / 2

    bra -= np.einsum("bk,adkl,bcij,cdjl->ai", t1, l2, l2, t2, optimize=True)

    bra -= np.einsum("dk,adkl,bcij,bcjl->ai", t1, l2, l2, t2, optimize=True) / 2

    bra += np.einsum("dj,adkl,bcij,bckl->ai", t1, l2, l2, t2, optimize=True) / 4

    bra += np.einsum("dl,adil,bcjk,bcjk->ai", t1, l2, l2, t2, optimize=True) / 4

    bra += np.einsum("bj,ck,dl,adkl,bcij->ai", t1, t1, t1, l2, l2, optimize=True)

    bra += np.einsum("bj,ck,dl,adil,bcjk->ai", t1, t1, t1, l2, l2, optimize=True) / 2

    bra += np.einsum("bk,cl,dj,adkl,bcij->ai", t1, t1, t1, l2, l2, optimize=True) / 2

    bra -= np.einsum("bj,cl,dk,adil,bcjk->ai", t1, t1, t1, l2, l2, optimize=True)

    bra += np.einsum("aj,bcik,bcjk->ai", l1, l2, t2, optimize=True) / 2

    bra += np.einsum("bi,acjk,bcjk->ai", l1, l2, t2, optimize=True) / 2

    bra -= np.einsum("bj,acik,bcjk->ai", l1, l2, t2, optimize=True)

    bra -= np.einsum("ai,bcjk,bcjk->ai", l1, l2, t2, optimize=True) / 4

    bra += np.einsum("aj,bj,ck,bcik->ai", l1, t1, t1, l2, optimize=True)

    bra += np.einsum("bi,bj,ck,acjk->ai", l1, t1, t1, l2, optimize=True)

    bra += np.einsum("bj,bk,cj,acik->ai", l1, t1, t1, l2, optimize=True)

    bra -= np.einsum("bj,bj,ck,acik->ai", l1, t1, t1, l2, optimize=True)

    bra -= np.einsum("ai,bj,ck,bcjk->ai", l1, t1, t1, l2, optimize=True) / 2

    bra += np.einsum("aj,bi,bj->ai", l1, l1, t1, optimize=True)

    bra -= np.einsum("ai,bj,bj->ai", l1, l1, t1, optimize=True)

    return bra


def bra_doubles_addition_qccsd(t1, t2, l1, l2):
    bra = np.einsum("ai,bj->abij", l1, l1, optimize=True)

    bra -= np.einsum("aj,bi->abij", l1, l1, optimize=True)

    bra += np.einsum("ai,ck,bckj->abij", l1, t1, l2, optimize=True)

    bra += np.einsum("aj,ck,bcik->abij", l1, t1, l2, optimize=True)

    bra += np.einsum("bk,ck,acij->abij", l1, t1, l2, optimize=True)

    bra += np.einsum("ci,ck,abkj->abij", l1, t1, l2, optimize=True)

    bra += np.einsum("cj,ck,abik->abij", l1, t1, l2, optimize=True)

    bra -= np.einsum("ak,ck,bcij->abij", l1, t1, l2, optimize=True)

    bra -= np.einsum("bi,ck,ackj->abij", l1, t1, l2, optimize=True)

    bra -= np.einsum("bj,ck,acik->abij", l1, t1, l2, optimize=True)

    bra -= np.einsum("ck,ck,abij->abij", l1, t1, l2, optimize=True)

    bra += np.einsum("acik,bdlj,cdkl->abij", l2, l2, t2, optimize=True)

    bra += np.einsum("abil,cdkj,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra += np.einsum("ackl,bdij,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra -= np.einsum("ackj,bdil,cdkl->abij", l2, l2, t2, optimize=True)

    bra -= np.einsum("ablj,cdik,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra -= np.einsum("adij,bckl,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra -= np.einsum("abij,cdkl,cdkl->abij", l2, l2, t2, optimize=True) / 4

    bra -= np.einsum("abkl,cdij,cdkl->abij", l2, l2, t2, optimize=True) / 4

    bra += np.einsum("ck,dl,ablj,cdik->abij", t1, t1, l2, l2, optimize=True)

    bra += np.einsum("ck,dl,ackj,bdil->abij", t1, t1, l2, l2, optimize=True)

    bra += np.einsum("ck,dl,adij,bckl->abij", t1, t1, l2, l2, optimize=True)

    bra += np.einsum("cl,dk,acik,bdlj->abij", t1, t1, l2, l2, optimize=True)

    bra += np.einsum("ck,dl,abij,cdkl->abij", t1, t1, l2, l2, optimize=True) / 2

    bra += np.einsum("ck,dl,abkl,cdij->abij", t1, t1, l2, l2, optimize=True) / 2

    bra -= np.einsum("ck,dl,abil,cdkj->abij", t1, t1, l2, l2, optimize=True)

    bra -= np.einsum("ck,dl,acik,bdlj->abij", t1, t1, l2, l2, optimize=True)

    bra -= np.einsum("ck,dl,ackl,bdij->abij", t1, t1, l2, l2, optimize=True)

    bra -= np.einsum("cl,dk,ackj,bdil->abij", t1, t1, l2, l2, optimize=True)

    return bra


def triples_weigth_qccsd(t1, t2, l1, l2):
    WT = 0

    WT += np.einsum("ai,bj,bcjk,acik->", l1, t1, l2, t2, optimize=True)

    WT -= np.einsum("ai,aj,bcjk,bcik->", l1, t1, l2, t2, optimize=True) / 2

    WT -= np.einsum("ai,bi,bcjk,acjk->", l1, t1, l2, t2, optimize=True) / 2

    WT += np.einsum("ai,ai,bcjk,bcjk->", l1, t1, l2, t2, optimize=True) / 4

    WT += np.einsum("ai,ai,bj,ck,bcjk->", l1, t1, t1, t1, l2, optimize=True) / 2

    WT -= np.einsum("ai,aj,bi,ck,bcjk->", l1, t1, t1, t1, l2, optimize=True)

    WT += np.einsum("ai,bj,abik,cdjl,cdkl->", t1, t1, l2, l2, t2, optimize=True)

    WT += np.einsum("ai,bj,acij,bdkl,cdkl->", t1, t1, l2, l2, t2, optimize=True)

    WT += np.einsum("aj,bi,acik,bdjl,cdkl->", t1, t1, l2, l2, t2, optimize=True)

    WT -= np.einsum("ai,bj,acik,bdjl,cdkl->", t1, t1, l2, l2, t2, optimize=True)

    WT -= np.einsum("ai,bj,abij,cdkl,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 4

    WT -= np.einsum("ai,bj,abkl,cdij,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 4

    WT -= 2 * np.einsum("ai,bj,ck,dl,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True)

    WT -= np.einsum("ai,bl,cj,dk,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 2

    WT -= np.einsum("aj,bk,ci,dl,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 2

    return WT


def quadruple_weigth_qccsd(t1, t2, l1, l2):
    WQ = 0

    WQ -= np.einsum("abjk,cdil,abjl,cdik->", l2, l2, t2, t2, optimize=True) / 8

    WQ -= np.einsum("abjk,cdil,acjk,bdil->", l2, l2, t2, t2, optimize=True) / 8

    WQ += np.einsum("abjk,cdil,acjl,bdik->", l2, l2, t2, t2, optimize=True) / 4

    WQ += np.einsum("abjk,cdil,abil,cdjk->", l2, l2, t2, t2, optimize=True) / 32

    WQ += np.einsum("abjk,cdil,abjk,cdil->", l2, l2, t2, t2, optimize=True) / 32

    WQ -= np.einsum("ai,bj,abik,cdjl,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 4

    WQ -= np.einsum("ai,bj,acij,bdkl,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 4

    WQ -= np.einsum("aj,bi,acik,bdjl,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 4

    WQ += np.einsum("ai,bj,acik,bdjl,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 4

    WQ += np.einsum("ai,bj,abij,cdkl,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 16

    WQ += np.einsum("ai,bj,abkl,cdij,cdkl->", t1, t1, l2, l2, t2, optimize=True) / 16

    return WQ
