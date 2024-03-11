import numpy as np


def reference_ccsd(t1, t2, l1, l2):
    overlap = 1 - np.einsum("ai,ai->", l1, t1)
    overlap -= 0.25 * np.einsum("abij,abij->", l2, t2)
    overlap += 0.25 * np.einsum("abij,ai,bj->", l2, t1, t1)
    overlap -= 0.25 * np.einsum("abij,aj,bi->", l2, t1, t1)

    return overlap


def ket_singles_ccsd(t1, t2, l1, l2):
    return t1.copy()


def bra_singles_ccsd(t1, t2, l1, l2):
    return l1 - np.einsum("aeim,em->ai", l2, t1)


def ket_doubles_ccsd(t1, t2, l1, l2):
    ket = np.einsum("ai,bj->abij", t1, t1)
    ket = ket - ket.transpose(1, 0, 2, 3) + t2
    return ket


def bra_doubles_ccsd(t1, t2, l1, l2):
    return l2.copy()
