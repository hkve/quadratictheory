import numpy as np


def energy_qccd(t2, l2, u, f, o, v):
    e = 0

    e += np.einsum(
        "acij,bdkl,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += (
        np.einsum(
            "abjl,cdik,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e += (
        np.einsum(
            "acjk,bdil,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= np.einsum(
        "adij,bckl,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= (
        np.einsum(
            "abij,cdkl,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "adjk,bcil,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "abil,cdjk,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e -= (
        np.einsum(
            "abjk,cdil,abim,cejk,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e -= (
        np.einsum(
            "abij,cdkl,abmn,ceik,dfjl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e -= (
        np.einsum(
            "abij,cdkl,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e -= (
        np.einsum(
            "abik,cdjl,abkm,cdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e -= (
        np.einsum(
            "abik,cdjl,abln,cdkm,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e -= (
        np.einsum(
            "acjk,bdil,abmn,cejk,dfil,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e += (
        np.einsum(
            "abik,cdjl,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e += (
        np.einsum(
            "acjk,bdil,abmn,cejl,dfik,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e += (
        np.einsum(
            "abij,cdkl,abkm,cdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e += (
        np.einsum(
            "abij,cdkl,abmn,ceij,dfkl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e += (
        np.einsum(
            "acjk,bdil,abmn,cfil,dejk,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e += np.einsum("abil,cdjk,acjm,beik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += np.einsum("abij,cdkl,abim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("abij,cdkl,acim,bejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("abij,cdkl,ackm,beij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("abil,cdjk,abjm,ceik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("abil,cdjk,acim,bejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e += np.einsum("abij,cdkl,abkm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 4

    e += np.einsum("abil,cdjk,abim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 4

    e += np.einsum("abik,cdjl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    e -= np.einsum("abij,cdkl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 4

    e -= np.einsum("abik,cdjl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e -= np.einsum("abik,cdjl,abjn,cdim,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum("abij,cdkl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4

    e += np.einsum("acjk,bdil,aejl,bfik,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    e -= np.einsum("abij,cdkl,aeik,bfjl,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 4

    e -= np.einsum("acjk,bdil,aejk,bfil,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e += np.einsum("abij,cdkl,aeij,bfkl,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e += np.einsum("acjk,bdil,afil,bejk,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    return e
