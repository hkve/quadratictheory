import numpy as np

def reference_ccsd(t1, t2, l1, l2):
    overlap = 1 - np.einsum("ai,ai->", l1, t1)
    overlap -= 0.25*np.einsum("abij,abij->", l2, t2)
    overlap += 0.25*np.einsum("abij,ai,bj->", l2, t1, t1)
    overlap -= 0.25*np.einsum("abij,aj,bi->", l2, t1, t1)

    return overlap

def singles_ccsd(t1, t2, l1, l2):
    bra = l1 - np.einsum("aeim,em->ai", l2, t1)
    ket = t1

    return np.multiply(bra, ket)

def doubles_ccsd(t1, t2, l1, l2):
    ket = np.einsum("ai,bj->abij", t1, t1)
    ket = ket - ket.transpose(1,0,2,3) + t2

    bra = l2
    return np.multiply(bra, ket)