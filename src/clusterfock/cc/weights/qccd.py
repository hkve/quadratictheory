import numpy as np


def reference_addition_qccd(t2, l2):
    det = -np.einsum("abjk,cdil,abjl,cdik->", l2, l2, t2, t2, optimize=True) / 8

    det -= np.einsum("abjk,cdil,acjk,bdil->", l2, l2, t2, t2, optimize=True) / 8

    det += np.einsum("abjk,cdil,acjl,bdik->", l2, l2, t2, t2, optimize=True) / 4

    det += np.einsum("abjk,cdil,abil,cdjk->", l2, l2, t2, t2, optimize=True) / 32

    det += np.einsum("abjk,cdil,abjk,cdil->", l2, l2, t2, t2, optimize=True) / 32

    return det


def bra_doubles_addition_qccd(t2, l2):
    bra = np.einsum("acik,bdlj,cdkl->abij", l2, l2, t2, optimize=True)

    bra += np.einsum("abil,cdkj,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra += np.einsum("ackl,bdij,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra -= np.einsum("ackj,bdil,cdkl->abij", l2, l2, t2, optimize=True)

    bra -= np.einsum("ablj,cdik,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra -= np.einsum("adij,bckl,cdkl->abij", l2, l2, t2, optimize=True) / 2

    bra -= np.einsum("abij,cdkl,cdkl->abij", l2, l2, t2, optimize=True) / 4

    bra -= np.einsum("abkl,cdij,cdkl->abij", l2, l2, t2, optimize=True) / 4

    return bra


def quadruple_weigth_qccd(t2, l2):
    WQ = 0

    WQ -= np.einsum(
        "abjk,cdil,abjl,cdik->", l2, l2, t2, t2, optimize=True
    ) / 8

    WQ -= np.einsum(
        "abjk,cdil,acjk,bdil->", l2, l2, t2, t2, optimize=True
    ) / 8

    WQ += np.einsum(
        "abjk,cdil,acjl,bdik->", l2, l2, t2, t2, optimize=True
    ) / 4

    WQ += np.einsum(
        "abjk,cdil,abil,cdjk->", l2, l2, t2, t2, optimize=True
    ) / 32

    WQ += np.einsum(
        "abjk,cdil,abjk,cdil->", l2, l2, t2, t2, optimize=True
    ) / 32

    return WQ