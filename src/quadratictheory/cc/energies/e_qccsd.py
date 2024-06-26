import numpy as np
from quadratictheory.cc.energies.e_inter_ccsd import td_energy_addition


def energy_qccsd(t1, t2, l1, l2, u, f, o, v):
    e = ccsd_gs_energy(t1, t2, u, f, o, v)
    e += td_energy_addition(t1, t2, l1, l2, u, f, o, v)
    e += qccsd_energy_addition(t1, t2, l1, l2, u, f, o, v)

    return e


def ccsd_gs_energy(t1, t2, u, f, o, v):
    e = 0
    e += np.einsum("ia,ai->", f[o, v], t1, optimize=True)
    e += np.einsum("abij,ijab->", t2, u[o, o, v, v], optimize=True) / 4
    e += np.einsum("ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True) / 2

    return e


def qccsd_energy_addition(t1, t2, l1, l2, u, f, o, v):
    e = 0

    e += np.einsum("ai,bj,abij->", l1, l1, u[v, v, o, o], optimize=True) / 2

    e += np.einsum("ai,bj,cdij,abcd->", l1, l1, t2, u[v, v, v, v], optimize=True) / 4

    e += np.einsum("ai,bj,ci,dj,abcd->", l1, l1, t1, t1, u[v, v, v, v], optimize=True) / 2

    e -= np.einsum("ai,bj,ac,bcij->", l1, l1, f[v, v], t2, optimize=True)

    e -= np.einsum("ai,bj,ci,abjc->", l1, l1, t1, u[v, v, o, v], optimize=True)

    e += np.einsum("ai,bcjk,abil,cdjm,lmkd->", l1, l2, t2, t2, u[o, o, o, v], optimize=True)

    e += np.einsum("ai,bcjk,abjl,cdkm,lmid->", l1, l2, t2, t2, u[o, o, o, v], optimize=True)

    e += np.einsum("ai,bcjk,adjm,bcil,lmkd->", l1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2

    e += np.einsum("ai,bcjk,adkm,bcjl,lmid->", l1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,acjm,bdil,lmkd->", l1, l2, t2, t2, u[o, o, o, v], optimize=True)

    e -= np.einsum("ai,bcjk,ablm,cdij,lmkd->", l1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,adim,bcjl,lmkd->", l1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,adij,bclm,lmkd->", l1, l2, t2, t2, u[o, o, o, v], optimize=True) / 4

    e -= np.einsum("ai,bcjk,adjk,bclm,lmid->", l1, l2, t2, t2, u[o, o, o, v], optimize=True) / 8

    e += np.einsum("ai,bcjk,aclm,bdjk,lmid->", l1, l2, t2, t2, u[o, o, o, v], optimize=True) / 4

    e += np.einsum("ai,bl,di,bcjk,acjm,lmkd->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True)

    e += np.einsum("ai,bl,dj,bcjk,ackm,lmid->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True)

    e += (
        np.einsum("ai,al,dj,bcjk,bcim,lmkd->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    e -= np.einsum("ai,al,bm,bcjk,cdij,lmkd->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True)

    e -= np.einsum("ai,bl,dj,bcjk,acim,lmkd->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True)

    e -= (
        np.einsum("ai,al,bm,bcjk,cdjk,lmid->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,al,di,bcjk,bcjm,lmkd->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,al,dj,bcjk,bckm,lmid->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,bl,cm,bcjk,adij,lmkd->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,bl,cm,bcjk,adjk,lmid->", l1, t1, t1, l2, t2, u[o, o, o, v], optimize=True) / 4
    )

    e += np.einsum("ai,bcjk,adij,bekl,lcde->", l1, l2, t2, t2, u[o, v, v, v], optimize=True)

    e += np.einsum("ai,bcjk,bdij,cekl,lade->", l1, l2, t2, t2, u[o, v, v, v], optimize=True)

    e += np.einsum("ai,bcjk,adjk,beil,lcde->", l1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2

    e += np.einsum("ai,bcjk,bdjk,ceil,lade->", l1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,aekl,bdij,lcde->", l1, l2, t2, t2, u[o, v, v, v], optimize=True)

    e -= np.einsum("ai,bcjk,abkl,deij,lcde->", l1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,aeil,bdjk,lcde->", l1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,abil,dejk,lcde->", l1, l2, t2, t2, u[o, v, v, v], optimize=True) / 4

    e -= np.einsum("ai,bcjk,bckl,deij,lade->", l1, l2, t2, t2, u[o, v, v, v], optimize=True) / 4

    e -= np.einsum("ai,bcjk,bcil,dejk,lade->", l1, l2, t2, t2, u[o, v, v, v], optimize=True) / 8

    e += np.einsum("ai,al,dj,bcjk,beik,lcde->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True)

    e += np.einsum("ai,bl,dj,bcjk,ceik,lade->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True)

    e += (
        np.einsum("ai,bl,di,bcjk,aejk,lcde->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    e -= np.einsum("ai,bl,dj,bcjk,aeik,lcde->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True)

    e -= np.einsum("ai,di,ej,bcjk,abkl,lcde->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True)

    e -= (
        np.einsum("ai,al,di,bcjk,bejk,lcde->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,bl,di,bcjk,cejk,lade->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,di,ej,bcjk,bckl,lade->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,dj,ek,bcjk,abil,lcde->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    e -= (
        np.einsum("ai,dj,ek,bcjk,bcil,lade->", l1, t1, t1, l2, t2, u[o, v, v, v], optimize=True) / 4
    )

    e += np.einsum("ai,bj,ki,abjk->", l1, l1, f[o, o], t2, optimize=True)

    e -= np.einsum("ai,bj,ak,kbij->", l1, l1, t1, u[o, v, o, o], optimize=True)

    e += np.einsum("abij,cdkl,acim,bejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e += np.einsum("abij,cdkl,ackm,beij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e += np.einsum("abil,cdjk,abjm,ceik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e += np.einsum("abil,cdjk,acim,bejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("abil,cdjk,acjm,beik,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("abij,cdkl,abim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("abij,cdkl,abkm,ceij,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 4

    e -= np.einsum("abil,cdjk,abim,cejk,mdle->", l2, l2, t2, t2, u[o, v, o, v], optimize=True) / 4

    e += np.einsum("ai,al,bcjk,bdij,cekm,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)

    e += np.einsum("ai,bl,bcjk,aekm,cdij,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)

    e += np.einsum("ai,di,bcjk,abjl,cekm,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)

    e += np.einsum("ai,dj,bcjk,ackm,beil,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)

    e += (
        np.einsum("ai,al,bcjk,bdjk,ceim,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e += (
        np.einsum("ai,bl,bcjk,ackm,deij,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e += (
        np.einsum("ai,bl,bcjk,aeim,cdjk,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e += (
        np.einsum("ai,di,bcjk,aekm,bcjl,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e += (
        np.einsum("ai,dj,bcjk,ablm,ceik,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e += (
        np.einsum("ai,dj,bcjk,aeim,bckl,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e -= np.einsum("ai,bl,bcjk,adij,cekm,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)

    e -= np.einsum("ai,dj,bcjk,abil,cekm,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)

    e -= np.einsum("ai,dl,bcjk,ackm,beij,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)

    e -= (
        np.einsum("ai,bl,bcjk,adjk,ceim,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ai,dj,bcjk,aekm,bcil,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ai,dl,bcjk,acim,bejk,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ai,dl,bcjk,aeik,bcjm,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ai,al,bcjk,bckm,deij,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    e -= (
        np.einsum("ai,al,bcjk,bcim,dejk,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    e -= (
        np.einsum("ai,di,bcjk,aejk,bclm,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    e += (
        np.einsum("ai,bl,bcjk,acim,dejk,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ai,di,bcjk,aclm,bejk,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ai,dj,bcjk,aeik,bclm,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ai,dl,bcjk,aejk,bcim,lmde->", l1, t1, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    e += np.einsum(
        "ai,al,bm,dj,bcjk,ceik,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
    )

    e += np.einsum(
        "ai,bl,di,ej,bcjk,ackm,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
    )

    e += (
        np.einsum(
            "ai,bl,cm,dj,bcjk,aeik,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e += (
        np.einsum(
            "ai,bl,dj,ek,bcjk,acim,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "ai,al,bm,di,bcjk,cejk,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "ai,al,di,ej,bcjk,bckm,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "ai,al,dj,ek,bcjk,bcim,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e -= (
        np.einsum(
            "ai,bl,cm,di,bcjk,aejk,lmde->", l1, t1, t1, t1, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e += np.einsum("ai,bj,abkl,klij->", l1, l1, t2, u[o, o, o, o], optimize=True) / 4

    e += np.einsum("ai,bj,ak,bl,klij->", l1, l1, t1, t1, u[o, o, o, o], optimize=True) / 2

    e += np.einsum("ai,bcjk,bdij,ackd->", l1, l2, t2, u[v, v, o, v], optimize=True)

    e += np.einsum("ai,bcjk,bdjk,acid->", l1, l2, t2, u[v, v, o, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,adij,bckd->", l1, l2, t2, u[v, v, o, v], optimize=True) / 2

    e -= np.einsum("ai,bcjk,adjk,bcid->", l1, l2, t2, u[v, v, o, v], optimize=True) / 4

    e += np.einsum("ai,bj,ci,adjk,kbcd->", l1, l1, t1, t2, u[o, v, v, v], optimize=True)

    e += np.einsum("ai,bj,ck,adij,kbcd->", l1, l1, t1, t2, u[o, v, v, v], optimize=True)

    e -= np.einsum("aj,bi,ci,adjk,kbcd->", l1, l1, t1, t2, u[o, v, v, v], optimize=True)

    e -= np.einsum("ai,bj,ak,cdij,kbcd->", l1, l1, t1, t2, u[o, v, v, v], optimize=True) / 2

    e -= np.einsum("ai,bj,ak,ci,dj,kbcd->", l1, l1, t1, t1, t1, u[o, v, v, v], optimize=True)

    e += np.einsum(
        "am,adjk,bcil,bdjn,ceik,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True
    )

    e += (
        np.einsum("am,adkl,bcij,bcin,dejk,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 2
    )

    e -= np.einsum(
        "am,adkl,bcij,bdin,cejk,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True
    )

    e -= np.einsum(
        "ei,abij,cdkl,acjm,bdkn,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True
    )

    e -= (
        np.einsum("am,adjk,bcil,bcjn,deik,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("am,adjk,bcil,bdin,cejk,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("am,adkl,bcij,bdkn,ceij,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ei,abjk,cdil,acjm,bdkn,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 2
    )

    e += (
        np.einsum("am,adjk,bcil,bcin,dejk,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("am,adkl,bcij,bckn,deij,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ei,abij,cdkl,abjm,cdkn,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ei,abij,cdkl,abkn,cdjm,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ei,abjk,cdil,abjm,cdkn,mnle->", t1, l2, l2, t2, t2, u[o, o, o, v], optimize=True)
        / 4
    )

    e += np.einsum("ai,bl,bcjk,adij,lckd->", l1, t1, l2, t2, u[o, v, o, v], optimize=True)

    e += np.einsum("ai,dj,bcjk,abil,lckd->", l1, t1, l2, t2, u[o, v, o, v], optimize=True)

    e += np.einsum("ai,bj,acik,bdjl,klcd->", l1, l1, t2, t2, u[o, o, v, v], optimize=True) / 2

    e += np.einsum("ai,bl,bcjk,adjk,lcid->", l1, t1, l2, t2, u[o, v, o, v], optimize=True) / 2

    e += np.einsum("ai,dj,bcjk,bcil,lakd->", l1, t1, l2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("ai,ld,bcjk,ackl,bdij->", l1, f[o, v], l2, t2, t2, optimize=True)

    e -= np.einsum("ai,al,bcjk,bdij,lckd->", l1, t1, l2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("ai,bl,bcjk,cdij,lakd->", l1, t1, l2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("ai,di,bcjk,abjl,lckd->", l1, t1, l2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("ai,dj,bcjk,abkl,lcid->", l1, t1, l2, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("ai,bj,abik,cdjl,klcd->", l1, l1, t2, t2, u[o, o, v, v], optimize=True) / 2

    e -= np.einsum("ai,bj,acij,bdkl,klcd->", l1, l1, t2, t2, u[o, o, v, v], optimize=True) / 2

    e -= np.einsum("ai,bj,adjl,bcik,klcd->", l1, l1, t2, t2, u[o, o, v, v], optimize=True) / 2

    e -= np.einsum("ai,ld,bcjk,acil,bdjk->", l1, f[o, v], l2, t2, t2, optimize=True) / 2

    e -= np.einsum("ai,ld,bcjk,adik,bcjl->", l1, f[o, v], l2, t2, t2, optimize=True) / 2

    e -= np.einsum("ai,al,bcjk,bdjk,lcid->", l1, t1, l2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("ai,bl,bcjk,cdjk,laid->", l1, t1, l2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("ai,di,bcjk,bcjl,lakd->", l1, t1, l2, t2, u[o, v, o, v], optimize=True) / 2

    e -= np.einsum("ai,dj,bcjk,bckl,laid->", l1, t1, l2, t2, u[o, v, o, v], optimize=True) / 2

    e += np.einsum("ai,ld,bcjk,adjk,bcil->", l1, f[o, v], l2, t2, t2, optimize=True) / 4

    e += np.einsum("ai,bj,abkl,cdij,klcd->", l1, l1, t2, t2, u[o, o, v, v], optimize=True) / 8

    e += np.einsum("aj,bi,ak,ci,bdjl,klcd->", l1, l1, t1, t1, t2, u[o, o, v, v], optimize=True)

    e -= np.einsum("ai,bj,ak,ci,bdjl,klcd->", l1, l1, t1, t1, t2, u[o, o, v, v], optimize=True)

    e -= np.einsum("ai,bj,ak,cl,bdij,klcd->", l1, l1, t1, t1, t2, u[o, o, v, v], optimize=True)

    e -= np.einsum("ai,bj,ci,dk,abjl,klcd->", l1, l1, t1, t1, t2, u[o, o, v, v], optimize=True)

    e += np.einsum("ai,bj,ak,bl,cdij,klcd->", l1, l1, t1, t1, t2, u[o, o, v, v], optimize=True) / 4

    e += np.einsum("ai,bj,ci,dj,abkl,klcd->", l1, l1, t1, t1, t2, u[o, o, v, v], optimize=True) / 4

    e += (
        np.einsum("ai,bj,ak,bl,ci,dj,klcd->", l1, l1, t1, t1, t1, t1, u[o, o, v, v], optimize=True)
        / 2
    )

    e += np.einsum("ai,bcjk,abjl,lcik->", l1, l2, t2, u[o, v, o, o], optimize=True)

    e += np.einsum("ai,bcjk,bcjl,laik->", l1, l2, t2, u[o, v, o, o], optimize=True) / 2

    e -= np.einsum("ai,bcjk,abil,lcjk->", l1, l2, t2, u[o, v, o, o], optimize=True) / 2

    e -= np.einsum("ai,bcjk,bcil,lajk->", l1, l2, t2, u[o, v, o, o], optimize=True) / 4

    e += np.einsum("ai,bj,ak,bcil,kljc->", l1, l1, t1, t2, u[o, o, o, v], optimize=True)

    e += np.einsum("ai,bj,ck,abil,kljc->", l1, l1, t1, t2, u[o, o, o, v], optimize=True)

    e -= np.einsum("aj,bi,ak,bcil,kljc->", l1, l1, t1, t2, u[o, o, o, v], optimize=True)

    e -= np.einsum("ai,bj,ci,abkl,kljc->", l1, l1, t1, t2, u[o, o, o, v], optimize=True) / 2

    e -= np.einsum("ai,bj,ak,bl,ci,kljc->", l1, l1, t1, t1, t1, u[o, o, o, v], optimize=True)

    e -= np.einsum("abik,cdjl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    e -= np.einsum("abij,cdkl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum("abij,cdkl,acim,bdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 4

    e += np.einsum("abik,cdjl,abim,cdjn,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum("abik,cdjl,abjn,cdim,mnkl->", l2, l2, t2, t2, u[o, o, o, o], optimize=True) / 8

    e += np.einsum("ai,di,bcjk,bejk,acde->", l1, t1, l2, t2, u[v, v, v, v], optimize=True) / 2

    e += np.einsum("ai,dj,bcjk,aeik,bcde->", l1, t1, l2, t2, u[v, v, v, v], optimize=True) / 2

    e -= np.einsum("ai,dj,bcjk,beik,acde->", l1, t1, l2, t2, u[v, v, v, v], optimize=True)

    e -= np.einsum("ai,di,bcjk,aejk,bcde->", l1, t1, l2, t2, u[v, v, v, v], optimize=True) / 4

    e += np.einsum("ai,al,bcjk,bcjm,lmik->", l1, t1, l2, t2, u[o, o, o, o], optimize=True) / 2

    e += np.einsum("ai,bl,bcjk,acim,lmjk->", l1, t1, l2, t2, u[o, o, o, o], optimize=True) / 2

    e -= np.einsum("ai,bl,bcjk,acjm,lmik->", l1, t1, l2, t2, u[o, o, o, o], optimize=True)

    e -= np.einsum("ai,al,bcjk,bcim,lmjk->", l1, t1, l2, t2, u[o, o, o, o], optimize=True) / 4

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

    e += np.einsum(
        "am,ei,adil,bcjk,bdjn,cgkl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    e += (
        np.einsum(
            "am,ei,adil,bcjk,bdln,cgjk,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e += (
        np.einsum(
            "am,ei,adkl,bcij,bckn,dgjl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e += (
        np.einsum(
            "am,ei,adkl,bcij,bdjn,cgkl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= np.einsum(
        "am,ei,adkl,bcij,bdkn,cgjl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    e -= (
        np.einsum(
            "am,bn,acjk,bdil,cejl,dgik,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "am,ei,adil,bcjk,bcjn,dgkl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "ei,gj,abik,cdjl,ackm,bdln,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    e -= (
        np.einsum(
            "am,ei,adil,bcjk,bcln,dgjk,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e -= (
        np.einsum(
            "am,ei,adkl,bcij,bcjn,dgkl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e -= (
        np.einsum(
            "am,bn,abij,cdkl,ceij,dgkl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e -= (
        np.einsum(
            "am,bn,acjk,bdil,cgil,dejk,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e -= (
        np.einsum(
            "ei,gj,abij,cdkl,abkm,cdln,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e += (
        np.einsum(
            "am,bn,abij,cdkl,ceik,dgjl,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e += (
        np.einsum(
            "ei,gj,abij,cdkl,ackm,bdln,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    e += (
        np.einsum(
            "am,bn,acjk,bdil,cejk,dgil,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e += (
        np.einsum(
            "ei,gj,abik,cdjl,abkm,cdln,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e += (
        np.einsum(
            "ei,gj,abik,cdjl,abln,cdkm,mneg->", t1, t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 8
    )

    e -= np.einsum("acjk,bdil,aejl,bgik,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    e -= np.einsum("abij,cdkl,aeij,bgkl,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e -= np.einsum("acjk,bdil,agil,bejk,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e += np.einsum("abij,cdkl,aeik,bgjl,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 4

    e += np.einsum("acjk,bdil,aejk,bgil,cdeg->", l2, l2, t2, t2, u[v, v, v, v], optimize=True) / 8

    e += np.einsum("aj,bi,acik,kbjc->", l1, l1, t2, u[o, v, o, v], optimize=True)

    e -= np.einsum("ai,bj,acik,kbjc->", l1, l1, t2, u[o, v, o, v], optimize=True)

    e += np.einsum("ai,bj,kc,ak,bcij->", l1, l1, f[o, v], t1, t2, optimize=True)

    e += np.einsum("ai,bj,kc,ci,abjk->", l1, l1, f[o, v], t1, t2, optimize=True)

    e += np.einsum("ai,bj,ak,ci,kbjc->", l1, l1, t1, t1, u[o, v, o, v], optimize=True)

    e -= np.einsum("aj,bi,ak,ci,kbjc->", l1, l1, t1, t1, u[o, v, o, v], optimize=True)

    e += np.einsum(
        "ei,abij,cdkl,ackm,bgjl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True
    )

    e += (
        np.einsum("ei,abjk,cdil,abjm,cgkl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 2
    )

    e -= np.einsum(
        "am,abij,cdkl,beik,cgjl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True
    )

    e -= np.einsum(
        "ei,abjk,cdil,acjm,bgkl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True
    )

    e -= (
        np.einsum("am,adkl,bcij,beik,cgjl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ei,abij,cdkl,abkm,cgjl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ei,abij,cdkl,acjm,bgkl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("ei,abjk,cdil,aclm,bgjk,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 2
    )

    e -= (
        np.einsum("am,abij,cdkl,bgkl,ceij,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("am,abij,cdkl,beij,cgkl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("am,adkl,bcij,beij,cgkl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ei,abij,cdkl,abjm,cgkl,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 4
    )

    e += (
        np.einsum("ei,abjk,cdil,ablm,cgjk,mdeg->", t1, l2, l2, t2, t2, u[o, v, v, v], optimize=True)
        / 4
    )

    return e
