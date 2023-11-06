import numpy as np


def energy_uccd2(t2, u, f, o, v):
    e = 0

    e += np.einsum("abij,ijkl,abkl->", t2, u[o, o, o, o], np.conjugate(t2), optimize=True) / 8

    e += np.einsum("ab,bcij,acij->", f[v, v], t2, np.conjugate(t2), optimize=True) / 8

    e += np.einsum("abij,abij->", u[v, v, o, o], np.conjugate(t2), optimize=True) / 8

    e += np.einsum("ab,bcij,acij->", f[v, v], t2, np.conjugate(t2), optimize=True) / 8

    e -= np.einsum("abij,jckb,acik->", t2, u[o, v, o, v], np.conjugate(t2), optimize=True) / 4

    e += np.einsum("ab,bcij,acij->", f[v, v], t2, np.conjugate(t2), optimize=True) / 8

    e -= np.einsum("ij,abik,abjk->", f[o, o], t2, np.conjugate(t2), optimize=True) / 4

    e += np.einsum("ab,bcij,acij->", f[v, v], t2, np.conjugate(t2), optimize=True) / 8

    e += np.einsum("abij,cdab,cdij->", t2, u[v, v, v, v], np.conjugate(t2), optimize=True) / 16

    e -= np.einsum("abij,jckb,acik->", t2, u[o, v, o, v], np.conjugate(t2), optimize=True) / 4

    e += np.einsum("abij,cdab,cdij->", t2, u[v, v, v, v], np.conjugate(t2), optimize=True) / 16

    e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4

    e -= np.einsum("abij,jckb,acik->", t2, u[o, v, o, v], np.conjugate(t2), optimize=True) / 4

    e += np.einsum("abij,abij->", u[v, v, o, o], np.conjugate(t2), optimize=True) / 8

    e -= np.einsum("abij,jckb,acik->", t2, u[o, v, o, v], np.conjugate(t2), optimize=True) / 4

    e -= np.einsum("ij,abik,abjk->", f[o, o], t2, np.conjugate(t2), optimize=True) / 4

    return e
