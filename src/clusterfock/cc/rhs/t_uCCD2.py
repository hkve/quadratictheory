import numpy as np


def amplitudes_uccd2(t2, u, f, v, o):
    r2 = np.zeros_like(t2)

    r2 += np.einsum("cdij,abcd->abij", t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,bdlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("bckj,adil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("cdkj,abil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("ackj,bdil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("bcik,adlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("cdik,ablj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("abkj,cdil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("abik,cdlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("ackj,bkic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("bcik,akcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("acik,bkcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bckj,akic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("abkl,cdij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("ackl,bdij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("bckl,adij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("ki,abkj->abij", f[o, o], t2, optimize=True)

    r2 -= np.einsum("kj,abik->abij", f[o, o], t2, optimize=True)

    r2 += np.einsum("bc,acij->abij", f[v, v], t2, optimize=True)

    r2 -= np.einsum("ac,bcij->abij", f[v, v], t2, optimize=True)

    r2 -= 3 * np.einsum("acik,bdlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= 3 * np.einsum("bckj,adil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= 3 * np.einsum("cdkj,abil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("acij,bdkl,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 8

    r2 -= np.einsum("ackl,bdij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("bcij,adkl,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 8

    r2 += np.einsum("cdij,abkl,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 8

    r2 += np.einsum("bckl,adij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += 3 * np.einsum("ackj,bdil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += 3 * np.einsum("bcik,adlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += 3 * np.einsum("cdik,ablj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("abkl,klij->abij", t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("ackj,bdil,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ablj,cdik,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("adij,bckl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,bdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abil,cdkj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ackl,bdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("abkl,cdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 4

    r2 -= 3 * np.einsum("acik,bdlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= 3 * np.einsum("bckj,adil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("acij,bdkl,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 8

    r2 -= np.einsum("ackl,bdij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("bcij,adkl,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 8

    r2 += np.einsum("bckl,adij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += 3 * np.einsum("ackj,bdil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += 3 * np.einsum("bcik,adlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("abij->abij", u[v, v, o, o], optimize=True)

    r2 -= np.einsum("ackl,bdij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("bckl,adij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= 3 * np.einsum("abkj,cdil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("abkl,cdij,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += 3 * np.einsum("abik,cdlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("acik,bdlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 -= np.einsum("bckj,adil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("ackj,bdil,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    r2 += np.einsum("bcik,adlj,cdkl->abij", t2, u[v, v, o, o], t2.conj(), optimize=True) / 16

    return r2
