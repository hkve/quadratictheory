import numpy as np


def reference_ccd(t2, l2):
    return 1 - 0.25 * np.einsum("abij,abij->", l2, t2)


def ket_doubles_ccd(t2, l2):
    return t2.copy()


def bra_doubles_ccd(t2, l2):
    return l2.copy()
