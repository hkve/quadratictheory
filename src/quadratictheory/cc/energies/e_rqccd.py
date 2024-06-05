import numpy as np
from quadratictheory.cc.energies.e_inter_rccd import td_energy_addition_opti_restricted


def energy_qccd_restricted(t2, l2, u, f, o, v):
    D = np.einsum("ijab,abij", u[o, o, v, v], t2, optimize=True)
    E = np.einsum("ijba,abij", u[o, o, v, v], t2, optimize=True)
    e = 2 * D - E

    e += td_energy_addition_opti_restricted(t2, l2, u, f, o, v)
    e += energy_addition_qccd_restricted(t2, l2, u, f, o, v)

    return e


def energy_addition_qccd_restricted(t2, l2, u, f, o, v):
    e = 0

    e -= np.einsum("acik,bdjl,acjn,bdim,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)

    e -= np.einsum("acik,bdjl,adjn,bcim,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)

    e -= 2 * np.einsum("abij,cdkl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)

    e -= 2 * np.einsum("acik,bdjl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)

    e += 2 * np.einsum("abij,cdkl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)

    e += 2 * np.einsum("acik,bdjl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)

    e += 2 * np.einsum("acik,bdjl,adim,bcjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)

    e -= (
        16
        * np.einsum("acik,bdjl,adim,cbjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 9
    )

    e -= (
        11
        * np.einsum("abij,cdkl,adim,cbjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 6
    )

    e -= (
        7
        * np.einsum("acik,bdjl,acim,bdjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("acik,bdjl,adim,bcjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 6
    )

    e -= np.einsum("abij,cdkl,abim,cdjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    e -= np.einsum("abij,cdkl,abjm,cdin,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    e += np.einsum("adik,bcjl,adim,cbjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 6

    e += np.einsum("acik,bdjl,abim,cdjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    e += np.einsum("acik,bdjl,abjm,cdin,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    e += np.einsum("adik,bcjl,adim,cbjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    e += np.einsum("abij,cdkl,adjm,cbin,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 12

    e += np.einsum("adik,bcjl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 12

    e += np.einsum("acik,bdjl,adjm,cbin,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    e += (
        4
        * np.einsum("adik,bcjl,abjm,cdin,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("abij,cdkl,caim,dbjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("acik,bdjl,acjn,bdim,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("adik,bcjl,adjm,cbin,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("abij,cdkl,caim,dbjn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 18
    )

    e += (
        5
        * np.einsum("abij,cdkl,cbim,dajn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 18
    )

    e += (
        19
        * np.einsum("acik,bdjl,adjn,bcim,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 36
    )

    e += (
        19
        * np.einsum("acik,bdjl,cbjn,daim,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 36
    )

    e += (
        23
        * np.einsum("abij,cdkl,cbim,dajn,mnlk->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 36
    )

    e += (
        23
        * np.einsum("acik,bdjl,cajn,dbim,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True)
        / 36
    )

    e -= np.einsum(
        "acki,bdlj,abmn,ceik,dfjl,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= np.einsum(
        "acki,bdlj,abmn,ceil,dfjk,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= np.einsum(
        "acki,bdlj,acln,bdkm,efij,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= np.einsum(
        "acki,bdlj,adln,bckm,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 8 * np.einsum(
        "abik,cdjl,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 8 * np.einsum(
        "acij,bdkl,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 8 * np.einsum(
        "adil,bckj,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abij,cdkl,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abij,cdkl,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abij,cdlk,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abik,cdjl,abmn,cfjl,deik,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abik,cdjl,acln,bdkm,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abil,cdjk,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abil,cdjk,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abil,cdkj,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "abjk,cdli,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "ablk,cdji,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "ablk,cdji,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "acij,bdkl,abmn,cfjl,deik,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "acij,bdkl,acln,bdkm,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "acik,bdjl,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "acik,bdjl,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "acik,bdlj,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "acjl,bdki,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "ackj,bdil,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "ackj,bdil,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "ackl,bdij,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "aclj,bdki,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "aclj,bdki,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "adik,bcjl,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "adik,bclj,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "adik,bclj,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "adkj,bcil,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "adkl,bcij,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= 2 * np.einsum(
        "adkl,bcij,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 2 * np.einsum(
        "abkl,cdij,abmn,ceik,dfjl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 2 * np.einsum(
        "abkl,cdij,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 2 * np.einsum(
        "acki,bdlj,abmn,ceik,dfjl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 2 * np.einsum(
        "acki,bdlj,abmn,ceil,dfjk,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 2 * np.einsum(
        "acki,bdlj,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 2 * np.einsum(
        "acki,bdlj,adkm,bcln,efij,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "abij,cdkl,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "abik,cdjl,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "abik,cdjl,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "abik,cdlj,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "abil,cdjk,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "ablk,cdji,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "acij,bdkl,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "acij,bdkl,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "acik,bdjl,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "acil,bdkj,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "ackj,bdil,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "aclj,bdki,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "adij,bckl,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "adik,bclj,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "adil,bckj,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "adil,bckj,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += 4 * np.einsum(
        "adkl,bcij,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= (
        23
        * np.einsum(
            "ablj,cdki,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        23
        * np.einsum(
            "aclk,bdji,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        20
        * np.einsum(
            "abjl,cdik,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        20
        * np.einsum(
            "ackl,bdij,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        19
        * np.einsum(
            "adjk,bcli,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        19
        * np.einsum(
            "adkj,bcil,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        16
        * np.einsum(
            "abik,cdjl,abmn,cfjk,deil,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        16
        * np.einsum(
            "abik,cdjl,adln,bckm,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        14
        * np.einsum(
            "abjk,cdil,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        14
        * np.einsum(
            "abjk,cdil,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        14
        * np.einsum(
            "ablk,cdij,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        14
        * np.einsum(
            "acij,bdkl,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        14
        * np.einsum(
            "acil,bdkj,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        14
        * np.einsum(
            "acil,bdkj,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        11
        * np.einsum(
            "acij,bdkl,abmn,cfjk,deil,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e -= (
        11
        * np.einsum(
            "acij,bdkl,adln,bckm,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e -= (
        10
        * np.einsum(
            "abjl,cdki,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        10
        * np.einsum(
            "ablj,cdik,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        10
        * np.einsum(
            "ackl,bdji,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        10
        * np.einsum(
            "aclk,bdij,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        8
        * np.einsum(
            "adjk,bcil,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        8
        * np.einsum(
            "adkj,bcli,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        7
        * np.einsum(
            "acki,bdlj,abmn,cfjk,deil,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        7
        * np.einsum(
            "acki,bdlj,ackm,bdln,efij,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        5
        * np.einsum(
            "adij,bckl,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        5
        * np.einsum(
            "adij,bckl,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        5
        * np.einsum(
            "adil,bckj,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        5
        * np.einsum(
            "adjl,bcki,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        5
        * np.einsum(
            "adjl,bcki,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        5
        * np.einsum(
            "adlj,bcki,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        5
        * np.einsum(
            "acki,bdlj,abmn,cfjl,deik,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e -= (
        5
        * np.einsum(
            "acki,bdlj,adkm,bcln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e -= (
        4
        * np.einsum(
            "abjk,cdli,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        4
        * np.einsum(
            "abkj,cdil,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        4
        * np.einsum(
            "acil,bdjk,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        2
        * np.einsum(
            "acji,bdkl,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        2
        * np.einsum(
            "abkj,cdli,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        2
        * np.einsum(
            "acjl,bdki,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        2
        * np.einsum(
            "acjl,bdki,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        2
        * np.einsum(
            "aclj,bdki,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        np.einsum(
            "adij,bclk,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        np.einsum(
            "adjl,bcik,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e -= (
        np.einsum(
            "acij,bdkl,abmn,ceil,dfjk,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        np.einsum(
            "acij,bdkl,acln,bdkm,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        np.einsum(
            "acjl,bdik,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        np.einsum(
            "acjl,bdik,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        np.einsum(
            "aclj,bdik,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= (
        np.einsum(
            "acij,bdkl,abmn,ceik,dfjl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e -= (
        np.einsum(
            "acij,bdkl,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        np.einsum(
            "acji,bdkl,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e += (
        np.einsum(
            "acji,bdkl,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e += (
        np.einsum(
            "acli,bdkj,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e += (
        np.einsum(
            "acik,bdlj,abmn,ceik,dfjl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "acik,bdlj,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "adij,bclk,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "adij,bclk,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "adil,bcjk,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "adjl,bcik,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "adjl,bcik,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "adlj,bcik,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 6
    )

    e += (
        np.einsum(
            "abik,cdjl,abmn,ceil,dfjk,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "abik,cdjl,abmn,cfjk,deil,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "abik,cdjl,ackm,bdln,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "abik,cdjl,acln,bdkm,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "abkj,cdli,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "abkj,cdli,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "abkl,cdji,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "acik,bdlj,abmn,cfjk,deil,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "acik,bdlj,ackm,bdln,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        np.einsum(
            "acij,bdkl,abmn,cfjl,deik,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 12
    )

    e += (
        np.einsum(
            "acij,bdkl,adkm,bcln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 12
    )

    e += (
        np.einsum(
            "acik,bdlj,abmn,cfjl,deik,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 12
    )

    e += (
        np.einsum(
            "acik,bdlj,adkm,bcln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 12
    )

    e += (
        np.einsum(
            "abik,cdjl,abmn,ceil,dfjk,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        np.einsum(
            "abik,cdjl,adkm,bcln,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        2
        * np.einsum(
            "abjk,cdli,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "abjk,cdli,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "abkj,cdil,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "abkj,cdil,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "abkl,cdij,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "ablk,cdji,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "acij,bdlk,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "acil,bdjk,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "acil,bdjk,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        2
        * np.einsum(
            "acjl,bdik,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "acik,bdlj,abmn,cfjk,deil,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "acik,bdlj,adln,bckm,efij,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "acjl,bdki,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "adjk,bcil,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "adjk,bcil,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "adkj,bcli,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "adkj,bcli,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "adkl,bcji,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        4
        * np.einsum(
            "adlk,bcij,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "abij,cdkl,abmn,ceik,dfjl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "abij,cdkl,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "abjl,cdik,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "abjl,cdki,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "abjl,cdki,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "ablj,cdik,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "ablj,cdik,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "ablj,cdki,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "acik,bdlj,abmn,cfjl,deik,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "acik,bdlj,acln,bdkm,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "acjk,bdil,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "acki,bdlj,abmn,cfjl,deik,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "acki,bdlj,acln,bdkm,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "ackj,bdli,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "ackl,bdji,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "ackl,bdji,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "aclk,bdij,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "aclk,bdij,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        5
        * np.einsum(
            "abij,cdkl,abmn,ceik,dfjl,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        5
        * np.einsum(
            "abij,cdkl,ackm,bdln,efij,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        5
        * np.einsum(
            "abij,cdlk,abmn,ceik,dfjl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        5
        * np.einsum(
            "abij,cdlk,ackm,bdln,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        10
        * np.einsum(
            "adij,bckl,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e += (
        10
        * np.einsum(
            "adjl,bcki,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 3
    )

    e += (
        10
        * np.einsum(
            "abjl,cdik,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        10
        * np.einsum(
            "abjl,cdik,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        10
        * np.einsum(
            "ablj,cdik,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        10
        * np.einsum(
            "ackj,bdil,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        10
        * np.einsum(
            "ackl,bdij,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        10
        * np.einsum(
            "ackl,bdij,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        19
        * np.einsum(
            "adjk,bcli,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        19
        * np.einsum(
            "adjk,bcli,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        19
        * np.einsum(
            "adkj,bcil,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        19
        * np.einsum(
            "adkj,bcil,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        19
        * np.einsum(
            "adkl,bcij,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        19
        * np.einsum(
            "adlk,bcji,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        19
        * np.einsum(
            "acik,bdjl,abmn,cfjk,deil,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        19
        * np.einsum(
            "acik,bdjl,adln,bckm,efij,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        19
        * np.einsum(
            "acki,bdlj,abmn,cfjk,deil,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        19
        * np.einsum(
            "acki,bdlj,adln,bckm,efij,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        23
        * np.einsum(
            "abjl,cdki,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        23
        * np.einsum(
            "ablj,cdki,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        23
        * np.einsum(
            "ablj,cdki,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        23
        * np.einsum(
            "acjk,bdli,aeij,bckm,dfnl,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        23
        * np.einsum(
            "aclk,bdji,aeij,bckm,dfln,mnfe->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        23
        * np.einsum(
            "aclk,bdji,aeij,bckm,dfnl,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 18
    )

    e += (
        23
        * np.einsum(
            "abij,cdlk,abmn,ceik,dfjl,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        23
        * np.einsum(
            "abij,cdlk,ackm,bdln,efij,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        23
        * np.einsum(
            "acik,bdjl,abmn,cfjl,deik,nmef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        23
        * np.einsum(
            "acik,bdjl,acln,bdkm,efij,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 36
    )

    e += (
        28
        * np.einsum(
            "abjk,cdil,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e += (
        28
        * np.einsum(
            "acil,bdkj,aeij,bckm,dfln,mnef->", l2, l2, t2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 9
    )

    e -= 4 * np.einsum("abij,cdkl,aeij,cbkm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e -= 4 * np.einsum("abij,cdkl,aeik,bcjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e -= 4 * np.einsum("abij,cdlk,aeik,bcjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e -= 4 * np.einsum("abji,cdkl,abjm,ceki,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e -= 4 * np.einsum("adij,bckl,acim,bekj,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e -= 4 * np.einsum("adij,bckl,aeij,bckm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("abij,cdkl,abkm,ceij,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("abij,cdkl,aeij,bckm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("abij,cdkl,aeik,cbjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("abij,cdlk,aeij,bckm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("abij,cdlk,aeik,cbjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("abij,cdlk,aeki,bcjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("abji,cdkl,aeki,cbjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("adij,bckl,abim,cejk,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("adij,bckl,acim,bejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("adij,bckl,ackm,beij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("adij,bckl,aeik,bcjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("adij,bckl,aeik,cbjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("adij,bckl,aekj,bcim,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += 2 * np.einsum("adij,bckl,bakm,ceji,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("adij,bckl,beki,cajm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)

    e += 2 * np.einsum("adij,bckl,bekj,caim,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e -= (
        23
        * np.einsum("abij,cdlk,aeki,cbjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 18
    )

    e -= (
        23
        * np.einsum("adij,bckl,abkm,ceji,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 18
    )

    e -= (
        23
        * np.einsum("adij,bckl,baim,cejk,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 18
    )

    e -= (
        23
        * np.einsum("adij,bckl,bejk,caim,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 18
    )

    e -= (
        19
        * np.einsum("abij,cdkl,abkm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 18
    )

    e -= (
        19
        * np.einsum("abji,cdlk,abkm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 18
    )

    e -= (
        19
        * np.einsum("adij,bckl,aeki,cbjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 18
    )

    e -= (
        19
        * np.einsum("adij,bckl,aekj,bcim,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 18
    )

    e -= (
        10
        * np.einsum("abji,cdkl,aeki,cbjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        10
        * np.einsum("adij,bckl,ackm,beij,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        10
        * np.einsum("adij,bckl,bajm,ceik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        10
        * np.einsum("adij,bckl,beik,cajm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("abij,cdkl,aeki,cbjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("abji,cdlk,aeki,cbjm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("adij,bckl,abkm,ceij,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("adij,bckl,ackm,beji,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("adij,bckl,baim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("adij,bckl,bajm,ceik,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("adij,bckl,beik,cajm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("adij,bckl,bejk,caim,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        4
        * np.einsum("abij,cdlk,abkm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        4
        * np.einsum("abji,cdkl,abkm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        4
        * np.einsum("adij,bckl,aeki,bcjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        4
        * np.einsum("adij,bckl,aekj,cbim,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        2
        * np.einsum("abij,cdlk,aeij,cbkm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        2
        * np.einsum("abij,cdlk,aeki,bcjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        2
        * np.einsum("abji,cdkl,aeki,bcjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        2
        * np.einsum("adij,bckl,acjm,beki,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e -= (
        2
        * np.einsum("adij,bckl,bakm,ceji,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= (
        2
        * np.einsum("adij,bckl,beij,cakm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= np.einsum("abji,cdkl,aeij,cbkm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    e -= np.einsum("adij,bckl,abim,cekj,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    e -= np.einsum("abij,cdkl,abjm,ceik,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True) / 6

    e -= np.einsum("abij,cdkl,abjm,ceki,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 6

    e -= np.einsum("abij,cdlk,abjm,ceik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 6

    e -= np.einsum("adij,bckl,aeij,cbkm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True) / 6

    e -= np.einsum("abji,cdlk,aeki,bcjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    e -= np.einsum("adij,bckl,beji,cakm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    e += np.einsum("adij,bckl,abjm,ceik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    e += np.einsum("adij,bckl,acjm,beik,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    e += (
        2
        * np.einsum("adij,bckl,abim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e += (
        2
        * np.einsum("adij,bckl,acim,bejk,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("abji,cdkl,abjm,ceik,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 3
    )

    e += (
        5
        * np.einsum("abji,cdkl,abjm,ceki,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 3
    )

    e += (
        5
        * np.einsum("abji,cdlk,abjm,ceik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 3
    )

    e += (
        5
        * np.einsum("adij,bckl,aeij,bckm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 3
    )

    e += (
        14
        * np.einsum("abij,cdkl,aeij,cbkm,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e += (
        14
        * np.einsum("abij,cdkl,aeki,bcjm,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e += (
        14
        * np.einsum("adij,bckl,acim,bekj,mdel->", l2, l2, t2, t2, u[o, v, v, o], optimize=True)
        / 9
    )

    e += (
        14
        * np.einsum("adij,bckl,bakm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)
        / 9
    )

    e -= np.einsum("acik,bdjl,afjk,beil,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)

    e -= np.einsum("acik,bdjl,afjl,beik,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)

    e -= 2 * np.einsum("abij,cdkl,aeij,bfkl,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)

    e -= 2 * np.einsum("acik,bdjl,aeij,bfkl,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)

    e += 2 * np.einsum("abij,cdkl,aeik,bfjl,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)

    e += 2 * np.einsum("acik,bdjl,aeik,bfjl,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)

    e += 2 * np.einsum("acik,bdjl,aeil,bfjk,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)

    e -= (
        16
        * np.einsum("abij,cdkl,aeil,bfkj,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 9
    )

    e -= (
        11
        * np.einsum("acik,bdjl,aeil,bfkj,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 6
    )

    e -= (
        7
        * np.einsum("acik,bdjl,aeil,bfjk,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 9
    )

    e -= (
        5
        * np.einsum("acik,bdjl,aeik,bfjl,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 6
    )

    e -= np.einsum("acik,bdjl,aeil,bfkj,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    e -= np.einsum("acik,bdjl,afkl,beij,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    e += np.einsum("acil,bdjk,aeil,bfkj,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 6

    e += np.einsum("abij,cdkl,aeil,bfkj,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    e += np.einsum("abij,cdkl,afkj,beil,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    e += np.einsum("acil,bdjk,aeij,bfkl,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    e += np.einsum("acik,bdjl,afkl,beij,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 12

    e += np.einsum("acil,bdjk,aeil,bfkj,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 12

    e += np.einsum("abij,cdkl,afkj,beil,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    e += (
        4
        * np.einsum("acil,bdjk,afkl,beij,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("abij,cdkl,aeki,bflj,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("acik,bdjl,afjl,beik,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("acil,bdjk,afkj,beil,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 9
    )

    e += (
        5
        * np.einsum("abij,cdkl,aeki,bflj,dcef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 18
    )

    e += (
        5
        * np.einsum("abij,cdkl,aflj,beki,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 18
    )

    e += (
        19
        * np.einsum("acik,bdjl,aeli,bfkj,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 36
    )

    e += (
        19
        * np.einsum("acik,bdjl,afjk,beil,cdef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 36
    )

    e += (
        23
        * np.einsum("abij,cdkl,aflj,beki,dcef->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 36
    )

    e += (
        23
        * np.einsum("acik,bdjl,aflj,beki,cdfe->", l2, l2, t2, t2, u[v, v, v, v], optimize=True)
        / 36
    )

    return e
