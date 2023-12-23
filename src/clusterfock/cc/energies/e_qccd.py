import numpy as np


def energy_qccd(t2, l2, u, f, o, v):
    e = 0

    e += np.einsum("abij,abij->", l2, u[v, v, o, o], optimize=True) / 4

    e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4

    e += np.einsum("abij,abkl,klij->", l2, t2, u[o, o, o, o], optimize=True) / 8

    e -= np.einsum("ac,abij,bcij->", f[v, v], l2, t2, optimize=True) / 2

    e += np.einsum("abij,cdij,abcd->", l2, t2, u[v, v, v, v], optimize=True) / 8

    e += np.einsum("ki,abij,abjk->", f[o, o], l2, t2, optimize=True) / 2

    e += np.einsum("abij,acik,bdjl,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    e -= np.einsum("abij,abik,cdjl,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True) / 4

    e -= np.einsum("abij,acij,bdkl,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True) / 4

    e += np.einsum("abij,abkl,cdij,klcd->", l2, t2, t2, u[o, o, v, v], optimize=True) / 16

    e -= np.einsum("acjk,bdil,aejl,bgik,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    e -= np.einsum("abij,cdkl,aeij,bgkl,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e -= np.einsum("acjk,bdil,agil,bejk,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e += np.einsum("abij,cdkl,aeik,bgjl,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 4

    e += np.einsum("acjk,bdil,aejk,bgil,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e += np.einsum("abij,cdkl,acim,bejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += np.einsum("abij,cdkl,ackm,beij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e += np.einsum("abil,cdjk,abjm,ceik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e += np.einsum("abil,cdjk,acim,bejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("abil,cdjk,acjm,beik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("abij,cdkl,abim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("abij,cdkl,abkm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 4

    e -= np.einsum("abil,cdjk,abim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 4

    e -= np.einsum("abij,acik,kbjc->", l2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("abik,cdjl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    e -= np.einsum("abij,cdkl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum("abij,cdkl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 4

    e += np.einsum("abik,cdjl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum("abik,cdjl,abjn,cdim,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum(
        "adij,bckl,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += (
        np.einsum(
            "abij,cdkl,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e += (
        np.einsum(
            "adjk,bcil,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= np.einsum(
        "acij,bdkl,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= (
        np.einsum(
            "abjl,cdik,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "acjk,bdil,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "abik,cdjl,ackm,bdln,egij,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e -= (
        np.einsum(
            "acjk,bdil,abmn,cejl,dgik,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e -= (
        np.einsum(
            "abij,cdkl,abkm,cdln,egij,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e -= (
        np.einsum(
            "abij,cdkl,abmn,ceij,dgkl,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e -= (
        np.einsum(
            "acjk,bdil,abmn,cgil,dejk,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e += (
        np.einsum(
            "abil,cdjk,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e += (
        np.einsum(
            "abjk,cdil,abim,cejk,dgln,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e += (
        np.einsum(
            "abij,cdkl,abmn,ceik,dgjl,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e += (
        np.einsum(
            "abij,cdkl,ackm,bdln,egij,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e += (
        np.einsum(
            "abik,cdjl,abkm,cdln,egij,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e += (
        np.einsum(
            "abik,cdjl,abln,cdkm,egij,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    e += (
        np.einsum(
            "acjk,bdil,abmn,cejk,dgil,mneg->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 16
    )

    return e
