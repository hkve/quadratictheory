import numpy as np


def amplitudes_t1_qccsd(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r1 = zeros((M, N))

    r1 += np.einsum("bl,abjk,iljk->ai", t1, l2, u[o, o, o, o], optimize=True) / 2

    r1 += np.einsum("aj,bk,bl,iljk->ai", l1, l1, t1, u[o, o, o, o], optimize=True)

    r1 += np.einsum("bj,bcjk,ikac->ai", l1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("cj,abjk,ibkc->ai", t1, l2, u[o, v, o, v], optimize=True)

    r1 += np.einsum("ic,abjk,bcjk->ai", f[o, v], l2, t2, optimize=True) / 2

    r1 += np.einsum("ka,bcij,bcjk->ai", f[o, v], l2, t2, optimize=True) / 2

    r1 -= np.einsum("bk,bcij,kcaj->ai", t1, l2, u[o, v, v, o], optimize=True)

    r1 -= np.einsum("aj,bcjk,ikbc->ai", l1, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bi,bcjk,jkac->ai", l1, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("aj,bk,ic,bcjk->ai", l1, l1, f[o, v], t2, optimize=True)

    r1 += np.einsum("aj,bk,cj,ibkc->ai", l1, l1, t1, u[o, v, o, v], optimize=True)

    r1 += np.einsum("bi,cj,ka,bcjk->ai", l1, l1, f[o, v], t2, optimize=True)

    r1 += np.einsum("bi,cj,ck,kbaj->ai", l1, l1, t1, u[o, v, v, o], optimize=True)

    r1 -= np.einsum("aj,bk,ck,ibjc->ai", l1, l1, t1, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("aj,bj,ck,ikbc->ai", l1, t1, t1, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,bk,kcaj->ai", l1, l1, t1, u[o, v, v, o], optimize=True)

    r1 -= np.einsum("bi,bj,ck,jkac->ai", l1, t1, t1, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bj,bk,cj,ikac->ai", l1, t1, t1, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,bm,ackl,cdjk,imld->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bj,cl,cdik,bdjm,lmak->ai", l1, t1, l2, t2, u[o, o, v, o], optimize=True)

    r1 += np.einsum("bj,dj,ackl,bckm,imld->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bj,dk,ackl,bclm,imjd->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True)

    r1 += np.einsum("aj,dk,bckl,bcjm,imld->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r1 += np.einsum("bi,bl,cdjk,cdjm,lmak->ai", l1, t1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r1 += np.einsum("bj,bl,cdik,cdkm,lmaj->ai", l1, t1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r1 += np.einsum("bj,bm,ackl,cdkl,imjd->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r1 -= np.einsum("aj,bm,bckl,cdjk,imld->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bi,cl,cdjk,bdjm,lmak->ai", l1, t1, l2, t2, u[o, o, v, o], optimize=True)

    r1 -= np.einsum("bj,cl,cdik,bdkm,lmaj->ai", l1, t1, l2, t2, u[o, o, v, o], optimize=True)

    r1 -= np.einsum("bj,cm,ackl,bdjk,imld->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bj,dk,ackl,bcjm,imld->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("aj,bm,bckl,cdkl,imjd->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r1 -= np.einsum("aj,dj,bckl,bckm,imld->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r1 -= np.einsum("aj,dk,bckl,bclm,imjd->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r1 -= np.einsum("bj,bl,cdik,cdjm,lmak->ai", l1, t1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r1 -= np.einsum("bj,cm,ackl,bdkl,imjd->ai", l1, t1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r1 += np.einsum("aj,bckl,bckm,imjl->ai", l1, l2, t2, u[o, o, o, o], optimize=True) / 2

    r1 += np.einsum("bj,ackl,bcjm,imkl->ai", l1, l2, t2, u[o, o, o, o], optimize=True) / 2

    r1 -= np.einsum("bj,ackl,bckm,imjl->ai", l1, l2, t2, u[o, o, o, o], optimize=True)

    r1 -= np.einsum("aj,bckl,bcjm,imkl->ai", l1, l2, t2, u[o, o, o, o], optimize=True) / 4

    r1 -= np.einsum("aj,ij->ai", l1, f[o, o], optimize=True)

    r1 += np.einsum("bi,cdjk,bcjl,ldak->ai", l1, l2, t2, u[o, v, v, o], optimize=True)

    r1 += np.einsum("bj,ackl,bdjk,icld->ai", l1, l2, t2, u[o, v, o, v], optimize=True)

    r1 += np.einsum("bj,cdik,bckl,ldaj->ai", l1, l2, t2, u[o, v, v, o], optimize=True)

    r1 += np.einsum("bi,cdjk,cdjl,lbak->ai", l1, l2, t2, u[o, v, v, o], optimize=True) / 2

    r1 += np.einsum("bj,ackl,bdkl,icjd->ai", l1, l2, t2, u[o, v, o, v], optimize=True) / 2

    r1 += np.einsum("bj,cdik,cdkl,lbaj->ai", l1, l2, t2, u[o, v, v, o], optimize=True) / 2

    r1 += np.einsum("bl,bcjk,cdjk,ilad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("dj,bcjk,bckl,ilad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("aj,bckl,bdjk,icld->ai", l1, l2, t2, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bj,ackl,cdjk,ibld->ai", l1, l2, t2, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bj,cdik,bcjl,ldak->ai", l1, l2, t2, u[o, v, v, o], optimize=True)

    r1 -= np.einsum("bk,bcij,cdjl,klad->ai", t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("cj,abjk,bdkl,ilcd->ai", t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("aj,bckl,bdkl,icjd->ai", l1, l2, t2, u[o, v, o, v], optimize=True) / 2

    r1 -= np.einsum("bj,ackl,cdkl,ibjd->ai", l1, l2, t2, u[o, v, o, v], optimize=True) / 2

    r1 -= np.einsum("bj,cdik,cdjl,lbak->ai", l1, l2, t2, u[o, v, v, o], optimize=True) / 2

    r1 -= np.einsum("cl,abjk,bdjk,ilcd->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("dk,bcij,bcjl,klad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bl,abjk,cdjk,ilcd->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("dj,bcij,bckl,klad->ai", t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("aj,bk,ck,bdjl,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bi,cj,ck,bdjl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,ck,bl,cdjk,ilad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,ck,dj,bckl,ilad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("aj,bk,bl,cdjk,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bi,cj,dj,bckl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bk,cl,dj,bcij,klad->ai", t1, t1, t1, l2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bl,cj,dk,abjk,ilcd->ai", t1, t1, t1, l2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("aj,bk,cj,bdkl,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("aj,bk,cl,bdjk,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,bk,cdjl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,dk,bcjl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("aj,bk,bl,cj,dk,ilcd->ai", l1, l1, t1, t1, t1, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bi,cj,bk,cl,dj,klad->ai", l1, l1, t1, t1, t1, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bcij,bcaj->ai", l2, u[v, v, v, o], optimize=True) / 2

    r1 += np.einsum("bi,cj,bcaj->ai", l1, l1, u[v, v, v, o], optimize=True)

    r1 += np.einsum("bj,cj,ibac->ai", l1, t1, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,jbac->ai", l1, t1, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bj,ibaj->ai", l1, u[o, v, v, o], optimize=True)

    r1 += np.einsum("bj,ijab->ai", t1, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("aj,ib,bj->ai", l1, f[o, v], t1, optimize=True)

    r1 -= np.einsum("bi,ja,bj->ai", l1, f[o, v], t1, optimize=True)

    r1 += np.einsum("aj,dk,bckl,bejl,icde->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bj,bl,cdik,cejk,ldae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bj,cl,cdik,dejk,lbae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bj,dk,ackl,cejl,ibde->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bj,ej,cdik,bckl,ldae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bi,cl,cdjk,bejk,ldae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 += np.einsum("bj,dj,ackl,bekl,icde->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 += np.einsum("bj,ej,cdik,cdkl,lbae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bi,ej,cdjk,bckl,ldae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bj,cl,cdik,bejk,ldae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bj,dk,ackl,bejl,icde->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bj,ek,cdik,bcjl,ldae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("aj,dj,bckl,bekl,icde->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bi,bl,cdjk,cejk,ldae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bi,cl,cdjk,dejk,lbae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bi,ej,cdjk,cdkl,lbae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,dj,ackl,cekl,ibde->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,ek,cdik,cdjl,lbae->ai", l1, t1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 += np.einsum("ia->ai", f[o, v], optimize=True)

    r1 += np.einsum("aj,bckl,bdjk,celm,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bi,cdjk,bcjl,dekm,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,ackl,belm,cdjk,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,cdik,bdkm,cejl,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("aj,bckl,bdkl,cejm,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bi,cdjk,bekm,cdjl,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bj,ackl,bclm,dejk,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bj,ackl,bejm,cdkl,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bj,cdik,bclm,dejk,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bj,cdik,bejm,cdkl,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,ackl,bdjk,celm,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bj,cdik,bcjl,dekm,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bj,cdkl,bdlm,cejk,imae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bj,ackl,bdkl,cejm,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,cdik,bekm,cdjl,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,cdkl,bdjm,cekl,imae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,cdkl,bejl,cdkm,imae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("aj,bckl,bclm,dejk,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 4

    r1 -= np.einsum("aj,bckl,bcjm,dekl,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 8

    r1 -= np.einsum("bi,cdjk,bejk,cdlm,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 8

    r1 += np.einsum("bi,cdjk,bdlm,cejk,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("bj,ackl,bcjm,dekl,imde->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("bj,cdik,bejk,cdlm,lmae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("bj,cdkl,bekl,cdjm,imae->ai", l1, l2, t2, t2, u[o, o, v, v], optimize=True) / 4

    r1 += np.einsum("aj,bm,dk,bckl,cejl,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bi,cl,ej,cdjk,bdkm,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,bl,cm,cdik,dejk,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,cl,ek,cdik,bdjm,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,cm,dk,ackl,bejl,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,dj,ek,ackl,bclm,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 += (
        np.einsum("bj,bl,ej,cdik,cdkm,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 += (
        np.einsum("bj,bm,dj,ackl,cekl,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 += (
        np.einsum("bj,cl,dm,cdik,bejk,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 += (
        np.einsum("bj,dk,el,ackl,bcjm,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 -= np.einsum("bj,bm,dk,ackl,cejl,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bj,cl,ej,cdik,bdkm,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r1 -= (
        np.einsum("aj,bm,dj,bckl,cekl,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 -= (
        np.einsum("aj,dj,ek,bckl,bclm,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 -= (
        np.einsum("bi,bl,cm,cdjk,dejk,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 -= (
        np.einsum("bi,bl,ej,cdjk,cdkm,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 -= (
        np.einsum("bj,bl,ek,cdik,cdjm,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 -= (
        np.einsum("bj,cm,dj,ackl,bekl,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r1 -= (
        np.einsum("aj,dk,el,bckl,bcjm,imde->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r1 -= (
        np.einsum("bi,cl,dm,cdjk,bejk,lmae->ai", l1, t1, t1, l2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r1 += np.einsum(
        "bn,adlm,bcjk,cejl,dgkm,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r1 += np.einsum(
        "dm,bcjk,deil,bgjl,cekn,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r1 += np.einsum(
        "el,adlm,bcjk,bgjm,cdkn,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r1 += np.einsum(
        "gj,bcjk,deil,bdkm,celn,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r1 += (
        np.einsum(
            "bm,bcjk,deil,celn,dgjk,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 += (
        np.einsum(
            "dn,adlm,bcjk,bejl,cgkm,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 += (
        np.einsum(
            "ej,adlm,bcjk,bcln,dgkm,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 += (
        np.einsum(
            "gl,bcjk,deil,bdjm,cekn,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 -= np.einsum(
        "bm,bcjk,deil,cekn,dgjl,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r1 -= np.einsum(
        "ej,adlm,bcjk,bgkl,cdmn,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r1 -= (
        np.einsum(
            "bm,bcjk,deil,cgkl,dejn,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 -= (
        np.einsum(
            "dm,bcjk,deil,bcjn,egkl,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 -= (
        np.einsum(
            "dm,bcjk,deil,bgjk,celn,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 -= (
        np.einsum(
            "ej,adlm,bcjk,bglm,cdkn,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 -= (
        np.einsum(
            "el,adlm,bcjk,bcjn,dgkm,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 -= (
        np.einsum(
            "el,adlm,bcjk,bgjk,cdmn,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r1 -= (
        np.einsum(
            "bm,bcjk,deil,cgjk,deln,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "bn,adlm,bcjk,cejk,dglm,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "dm,bcjk,deil,bcln,egjk,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "dn,adlm,bcjk,bejk,cglm,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "ej,adlm,bcjk,bckn,dglm,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "el,adlm,bcjk,bcmn,dgjk,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "gj,bcjk,deil,bckm,deln,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "gj,bcjk,deil,bcln,dekm,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 -= (
        np.einsum(
            "gl,bcjk,deil,bcjm,dekn,mnag->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 += (
        np.einsum(
            "bn,adlm,bcjk,cglm,dejk,ineg->ai", t1, l2, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r1 += np.einsum("bi,ba->ai", l1, f[v, v], optimize=True)

    r1 += np.einsum("abjk,bcjl,ilkc->ai", l2, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bcjk,bcjl,ilak->ai", l2, t2, u[o, o, v, o], optimize=True) / 2

    r1 += np.einsum("bcij,bckl,klaj->ai", l2, t2, u[o, o, v, o], optimize=True) / 4

    r1 += np.einsum("aj,bk,bcjl,ilkc->ai", l1, l1, t2, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bi,cj,bckl,klaj->ai", l1, l1, t2, u[o, o, v, o], optimize=True) / 2

    r1 += np.einsum("bk,cl,bcij,klaj->ai", t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r1 -= np.einsum("aj,bk,bckl,iljc->ai", l1, l1, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bj,ck,bcjl,ilak->ai", l1, l1, t2, u[o, o, v, o], optimize=True)

    r1 -= np.einsum("bl,cj,abjk,ilkc->ai", t1, t1, l2, u[o, o, o, v], optimize=True)

    r1 += np.einsum("aj,bk,bl,ck,iljc->ai", l1, l1, t1, t1, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bi,cj,bk,cl,klaj->ai", l1, l1, t1, t1, u[o, o, v, o], optimize=True)

    r1 -= np.einsum("aj,bk,bl,cj,ilkc->ai", l1, l1, t1, t1, u[o, o, o, v], optimize=True)

    r1 += np.einsum("dj,bcij,bcad->ai", t1, l2, u[v, v, v, v], optimize=True) / 2

    r1 += np.einsum("bi,cj,dj,bcad->ai", l1, l1, t1, u[v, v, v, v], optimize=True)

    r1 += np.einsum("bcij,bdjk,kcad->ai", l2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bcjk,bdjk,icad->ai", l2, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("abjk,cdjk,ibcd->ai", l2, t2, u[o, v, v, v], optimize=True) / 4

    r1 += np.einsum("bi,cj,bdjk,kcad->ai", l1, l1, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bj,ck,bdjk,icad->ai", l1, l1, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,cdjk,kbad->ai", l1, l1, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bk,dj,bcij,kcad->ai", t1, t1, l2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("aj,bk,cdjk,ibcd->ai", l1, l1, t2, u[o, v, v, v], optimize=True) / 2

    r1 -= np.einsum("cj,dk,abjk,ibcd->ai", t1, t1, l2, u[o, v, v, v], optimize=True) / 2

    r1 += np.einsum("bi,cj,ck,dj,kbad->ai", l1, l1, t1, t1, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("aj,bk,cj,dk,ibcd->ai", l1, l1, t1, t1, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,bk,dj,kcad->ai", l1, l1, t1, t1, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bi,cdjk,cejk,bdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True) / 2

    r1 += np.einsum("bj,cdik,bejk,cdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,cdik,cejk,bdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cdjk,bejk,cdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True) / 4

    r1 += np.einsum("adlm,bcjk,bejl,dgkm,iceg->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bcjk,deil,bekm,dgjl,mcag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True)

    r1 += (
        np.einsum("bcjk,deil,bcjm,dgkl,meag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r1 += (
        np.einsum("bcjk,deil,bgjk,cdlm,meag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r1 += (
        np.einsum("bcjk,deil,bgkl,dejm,mcag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r1 -= np.einsum("bcjk,deil,bgjl,cdkm,meag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True)

    r1 -= (
        np.einsum("adlm,bcjk,bejl,cgkm,ideg->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r1 -= (
        np.einsum("bcjk,deil,belm,dgjk,mcag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r1 -= (
        np.einsum("adlm,bcjk,bejk,dglm,iceg->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 4
    )

    r1 += (
        np.einsum("adlm,bcjk,bejk,cglm,ideg->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 4
    )

    r1 += (
        np.einsum("adlm,bcjk,bglm,dejk,iceg->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 4
    )

    r1 += (
        np.einsum("bcjk,deil,bclm,dgjk,meag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 4
    )

    r1 += (
        np.einsum("bcjk,deil,bgjk,delm,mcag->ai", l2, l2, t2, t2, u[o, v, v, v], optimize=True) / 4
    )

    r1 += np.einsum("adlm,bcjk,bejl,cdmn,inke->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True)

    r1 += (
        np.einsum("adlm,bcjk,bcjn,dekl,inme->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r1 += (
        np.einsum("adlm,bcjk,bejk,cdln,inme->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r1 += (
        np.einsum("adlm,bcjk,belm,cdjn,inke->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r1 += (
        np.einsum("bcjk,deil,bdjm,cekn,mnal->ai", l2, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r1 -= np.einsum("adlm,bcjk,bejl,cdkn,inme->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bcjk,deil,bdjm,celn,mnak->ai", l2, l2, t2, t2, u[o, o, v, o], optimize=True)

    r1 -= (
        np.einsum("adlm,bcjk,bcln,dejm,inke->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r1 -= (
        np.einsum("bcjk,deil,bcjm,dekn,mnal->ai", l2, l2, t2, t2, u[o, o, v, o], optimize=True) / 4
    )

    r1 += (
        np.einsum("adlm,bcjk,bcjn,delm,inke->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True) / 4
    )

    r1 += (
        np.einsum("adlm,bcjk,bcln,dejk,inme->ai", l2, l2, t2, t2, u[o, o, o, v], optimize=True) / 4
    )

    r1 += (
        np.einsum("bcjk,deil,bcjm,deln,mnak->ai", l2, l2, t2, t2, u[o, o, v, o], optimize=True) / 4
    )

    r1 += (
        np.einsum("bcjk,deil,bcln,dejm,mnak->ai", l2, l2, t2, t2, u[o, o, v, o], optimize=True) / 4
    )

    r1 -= np.einsum("abjk,ibjk->ai", l2, u[o, v, o, o], optimize=True) / 2

    r1 -= np.einsum("aj,bk,ibjk->ai", l1, l1, u[o, v, o, o], optimize=True)

    r1 -= np.einsum("aj,bk,ikjb->ai", l1, t1, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bj,bk,ikaj->ai", l1, t1, u[o, o, v, o], optimize=True)

    return r1


def amplitudes_t2_qccsd(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum("ck,adkj,bcid->abij", l1, t2, u[v, v, o, v], optimize=True)

    r2 += np.einsum("ck,bdij,ackd->abij", l1, t2, u[v, v, o, v], optimize=True)

    r2 += np.einsum("ck,bdik,acdj->abij", l1, t2, u[v, v, v, o], optimize=True)

    r2 += np.einsum("ck,cdkj,abid->abij", l1, t2, u[v, v, o, v], optimize=True)

    r2 += np.einsum("ci,adkj,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("cj,adik,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,bdij,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ak,cdij,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,adij,bckd->abij", l1, t2, u[v, v, o, v], optimize=True)

    r2 -= np.einsum("ck,adik,bcdj->abij", l1, t2, u[v, v, v, o], optimize=True)

    r2 -= np.einsum("ck,bdkj,acid->abij", l1, t2, u[v, v, o, v], optimize=True)

    r2 -= np.einsum("ck,cdij,abkd->abij", l1, t2, u[v, v, o, v], optimize=True)

    r2 -= np.einsum("ck,cdik,abdj->abij", l1, t2, u[v, v, v, o], optimize=True)

    r2 -= np.einsum("ci,bdkj,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("cj,bdik,akcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,adij,bkcd->abij", t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cdij,akcd->abij", t1, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ak,ci,dj,bkcd->abij", t1, t1, t1, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("bk,ci,dj,akcd->abij", t1, t1, t1, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,adik,belj,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,adkl,beij,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,adlj,beik,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,aeil,cdkj,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,aekj,cdil,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,beij,cdkl,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,beik,cdlj,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,bekl,cdij,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,belj,cdik,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,abil,dekj,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("ck,acil,dekj,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,bckl,deij,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,bclj,deik,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,adil,bekj,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,adkj,beil,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,aeij,bdkl,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,aeij,cdkl,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,aeik,cdlj,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,aekl,cdij,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,aelj,cdik,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,beil,cdkj,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bekj,cdil,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,abkl,deij,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,ablj,deik,lcde->abij", l1, t2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,ackl,deij,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,aclj,deik,blde->abij", l1, t2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,bcil,dekj,alde->abij", l1, t2, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,al,di,bekj,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,al,di,cekj,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,al,dj,beik,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,al,dj,ceik,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,dk,aeij,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,bl,dk,ceij,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,di,bekj,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,dj,beik,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,dk,aeij,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,di,ej,bckl,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,di,ek,bclj,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,dk,ej,abil,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,dk,ej,acil,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,dk,beij,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,al,dk,ceij,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,di,aekj,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,di,cekj,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,dj,aeik,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,dj,ceik,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,di,aekj,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,dj,aeik,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,dk,beij,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,di,ej,abkl,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,di,ej,ackl,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,di,ek,ablj,lcde->abij", l1, t1, t1, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,di,ek,aclj,blde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,ej,bcil,alde->abij", l1, t1, t1, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,di,aekj,bcde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ck,di,cekj,abde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ck,dj,aeik,bcde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ck,dj,ceik,abde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ck,dk,beij,acde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("ck,di,bekj,acde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("ck,dj,beik,acde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,aeij,bcde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,ceij,abde->abij", l1, t1, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ackj,bdil,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ablj,cdik,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("adij,bckl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,bdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abil,cdkj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ackl,bdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("abkl,cdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("ck,ld,abkl,cdij->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 += np.einsum("ck,ld,ablj,cdik->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 += np.einsum("ck,ld,acil,bdkj->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 += np.einsum("ck,ld,adij,bckl->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 += np.einsum("ck,ld,adik,bclj->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 += np.einsum("ck,al,bdkj,lcid->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("ck,al,cdkj,blid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,bl,adij,lckd->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("ck,bl,adik,lcdj->abij", l1, t1, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ck,bl,cdij,alkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,bl,cdik,aldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("ck,cl,adij,blkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,cl,adik,bldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("ck,cl,bdkj,alid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,di,ablj,lckd->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("ck,di,aclj,blkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,di,bckl,aldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("ck,dj,abil,lckd->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("ck,dj,acil,blkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,dj,bckl,alid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,dk,abil,lcdj->abij", l1, t1, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ck,dk,acil,bldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("ck,dk,bclj,alid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ak,ci,bdlj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,cj,bdil,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,cl,adij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,dk,ablj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,bl,cdij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dj,abkl,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,ld,abil,cdkj->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 -= np.einsum("ck,ld,ackl,bdij->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 -= np.einsum("ck,ld,aclj,bdik->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 -= np.einsum("ck,ld,adkj,bcil->abij", l1, f[o, v], t2, t2, optimize=True)

    r2 -= np.einsum("ck,al,bdij,lckd->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ck,al,bdik,lcdj->abij", l1, t1, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ck,al,cdij,blkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,al,cdik,bldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("ck,bl,adkj,lcid->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ck,bl,cdkj,alid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,cl,adkj,blid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,cl,bdij,alkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,cl,bdik,aldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("ck,di,abkl,lcdj->abij", l1, t1, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ck,di,ackl,bldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("ck,di,bclj,alkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,dj,abkl,lcid->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ck,dj,ackl,blid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,dj,bcil,alkd->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,dk,ablj,lcid->abij", l1, t1, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ck,dk,aclj,blid->abij", l1, t1, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,dk,bcil,aldj->abij", l1, t1, t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("ak,cl,bdij,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,ci,adlj,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cj,adil,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dj,abil,klcd->abij", t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,bl,ci,dj,klcd->abij", t1, t1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,bkij->abij", t1, u[v, o, o, o], optimize=True)

    r2 -= np.einsum("ki,abkj->abij", f[o, o], t2, optimize=True)

    r2 -= np.einsum("kj,abik->abij", f[o, o], t2, optimize=True)

    r2 -= np.einsum("bk,akij->abij", t1, u[v, o, o, o], optimize=True)

    r2 += np.einsum("am,cdkl,bgil,cekj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bm,cdkl,aglj,ceik,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bm,cdkl,ceik,dglj,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("cm,cdkl,aeik,bglj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cm,cdkl,agil,dekj,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("cm,cdkl,bglj,deik,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ei,cdkl,aglj,bckm,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ei,cdkl,bdlm,cgkj,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ej,cdkl,agil,bckm,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ej,cdkl,bdlm,cgik,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,ablm,cgij,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,abmj,cgil,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,acim,bglj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,adlm,cgij,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,admj,cgil,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,agij,bclm,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,agil,bcmj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ek,cdkl,bdim,cglj,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 += (
        np.einsum("bm,cdkl,agij,cekl,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("bm,cdkl,agkl,ceij,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("bm,cdkl,cekl,dgij,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,cdkl,aekl,bgij,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,cdkl,bgij,dekl,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,cdkl,bgkl,deij,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ei,cdkl,acmj,bgkl,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ei,cdkl,bdmj,cgkl,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ei,cdkl,bglj,cdkm,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ej,cdkl,acim,bgkl,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ej,cdkl,bdim,cgkl,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ej,cdkl,bgil,cdkm,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,cdkl,aglj,cdim,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,cdkl,bgij,cdlm,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,cdkl,bgil,cdmj,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= np.einsum("am,cdkl,bglj,ceik,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("am,cdkl,ceik,dglj,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("bm,cdkl,agil,cekj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cm,cdkl,aekj,bgil,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cm,cdkl,aglj,deik,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("cm,cdkl,bgil,dekj,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ei,cdkl,ablm,cgkj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ei,cdkl,ackm,bglj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ei,cdkl,adlm,cgkj,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ej,cdkl,ablm,cgik,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ej,cdkl,ackm,bgil,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ej,cdkl,adlm,cgik,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,abim,cglj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,aclm,bgij,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,acmj,bgil,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,adim,cglj,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,aglj,bcim,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,bdlm,cgij,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,bdmj,cgil,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True)

    r2 -= (
        np.einsum("am,cdkl,bgij,cekl,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("am,cdkl,bgkl,ceij,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("am,cdkl,cekl,dgij,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,cdkl,agij,bekl,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,cdkl,agij,dekl,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,cdkl,agkl,deij,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ei,cdkl,abmj,cgkl,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ei,cdkl,admj,cgkl,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ei,cdkl,agkl,bcmj,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ei,cdkl,aglj,cdkm,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ej,cdkl,abim,cgkl,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ej,cdkl,adim,cgkl,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ej,cdkl,agil,cdkm,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ej,cdkl,agkl,bcim,mdeg->abij", t1, l2, t2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,cdkl,agij,cdlm,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,cdkl,agil,cdmj,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,cdkl,bglj,cdim,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ei,cdkl,bgkl,cdmj,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 4
    )

    r2 -= (
        np.einsum("ej,cdkl,bgkl,cdim,ameg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 4
    )

    r2 += (
        np.einsum("ei,cdkl,agkl,cdmj,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 4
    )

    r2 += (
        np.einsum("ej,cdkl,agkl,cdim,bmeg->abij", t1, l2, t2, t2, u[v, o, v, v], optimize=True) / 4
    )

    r2 += np.einsum("abij->abij", u[v, v, o, o], optimize=True)

    r2 += np.einsum("cdkl,agil,cekj,bdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("cdkl,bglj,ceik,adeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("cdkl,aekj,bgil,cdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("cdkl,bgij,cekl,adeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("cdkl,bgkl,ceij,adeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,aglj,ceik,bdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,bgil,cekj,adeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,ceik,dglj,abeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bglj,cdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,agij,cekl,bdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,agkl,ceij,bdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,cekl,dgij,abeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,aekl,bgij,cdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 4

    r2 += np.einsum("cdkl,agij,bekl,cdeg->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 4

    r2 += np.einsum("cdkl,abim,cekj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,ablm,ceik,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,ackm,beij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,ackm,beil,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,acmj,beik,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,adim,cekj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,adlm,ceik,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,aekj,bcim,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,aelj,bckm,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,bdkm,ceij,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bdlm,cekj,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bdmj,ceik,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,abim,cekl,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("cdkl,acmj,bekl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    r2 += np.einsum("cdkl,adim,cekl,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 2

    r2 += np.einsum("cdkl,aeij,cdkm,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("cdkl,aeik,cdmj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("cdkl,aeil,cdkm,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 2

    r2 += np.einsum("cdkl,aekl,bcim,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("cdkl,bdmj,cekl,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("cdkl,bekj,cdim,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("cdkl,belj,cdkm,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,abkm,ceij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cdkl,ablm,cekj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cdkl,abmj,ceik,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cdkl,acim,bekj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cdkl,ackm,belj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cdkl,adkm,ceij,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("cdkl,adlm,cekj,bmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("cdkl,admj,ceik,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("cdkl,aeij,bckm,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bcmj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cdkl,aeil,bckm,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("cdkl,bdim,cekj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("cdkl,bdlm,ceik,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("cdkl,abmj,cekl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,acim,bekl,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,admj,cekl,bmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,aekj,cdim,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,aekl,bcmj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,aelj,cdkm,bmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,bdim,cekl,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,beij,cdkm,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,beik,cdmj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("cdkl,beil,cdkm,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,aekl,cdim,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 4

    r2 -= np.einsum("cdkl,bekl,cdmj,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 4

    r2 += np.einsum("cdkl,aekl,cdmj,bmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 4

    r2 += np.einsum("cdkl,bekl,cdim,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 4

    r2 += np.einsum("ck,al,beim,cdkj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,al,bekj,cdim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,aeij,cdkm,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,aeik,cdmj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,aekm,cdij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,aemj,cdik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,adik,bemj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,adkm,beij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,admj,beik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,di,abmj,cekl,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,di,aekl,bcmj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,di,aemj,bckl,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dj,abim,cekl,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dj,aeim,bckl,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dj,aekl,bcim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dk,abim,celj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dk,acil,bemj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dk,aelj,bcim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,abkm,ceij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,abmj,ceik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,acim,bekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,aeij,bckm,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,aeik,bcmj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,al,bcim,dekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,bl,ackm,deij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,bl,acmj,deik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,cl,abim,dekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,di,ablm,cekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,di,aekj,bclm,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,dj,ablm,ceik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,dj,aeik,bclm,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,dk,aclm,beij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,al,beij,cdkm,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,beik,cdmj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,bekm,cdij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,bemj,cdik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,aeim,cdkj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,aekj,cdim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,adim,bekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,adkj,beim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,aeij,bdkm,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,di,abkm,celj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,di,ackl,bemj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,di,acmj,bekl,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dj,abkm,ceil,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dj,acim,bekl,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dj,ackl,beim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,abmj,ceil,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,aclj,beim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,aeil,bcmj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,abim,cekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,ackm,beij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,acmj,beik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,aekj,bcim,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,bckm,deij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,al,bcmj,deik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,bl,acim,dekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,cl,abkm,deij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,cl,abmj,deik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,di,aclm,bekj,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,dj,aclm,beik,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,dk,ablm,ceij,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,dk,aeij,bclm,lmde->abij", l1, t1, t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,al,bm,di,cekj,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,al,bm,dj,ceik,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,al,cm,dk,beij,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,al,dk,ej,bcim,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,cm,di,aekj,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,cm,dj,aeik,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,di,ej,ackm,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,bl,di,ek,acmj,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,dk,ej,abim,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,bm,dk,ceij,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,cm,di,bekj,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,cm,dj,beik,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,di,ej,bckm,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,al,di,ek,bcmj,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,cm,dk,aeij,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bl,dk,ej,acim,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,di,ej,abkm,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,di,ek,abmj,lmde->abij", l1, t1, t1, t1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdij,abcd->abij", t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dj,abcd->abij", t1, t1, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ck,abil,lckj->abij", l1, t2, u[o, v, o, o], optimize=True)

    r2 += np.einsum("ck,acil,blkj->abij", l1, t2, u[v, o, o, o], optimize=True)

    r2 += np.einsum("ck,bckl,alij->abij", l1, t2, u[v, o, o, o], optimize=True)

    r2 += np.einsum("ck,bclj,alik->abij", l1, t2, u[v, o, o, o], optimize=True)

    r2 += np.einsum("ak,bclj,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,acil,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,ablj,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ci,abkl,klcj->abij", t1, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("cj,abkl,klic->abij", t1, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ck,abkl,lcij->abij", l1, t2, u[o, v, o, o], optimize=True)

    r2 -= np.einsum("ck,ablj,lcik->abij", l1, t2, u[o, v, o, o], optimize=True)

    r2 -= np.einsum("ck,ackl,blij->abij", l1, t2, u[v, o, o, o], optimize=True)

    r2 -= np.einsum("ck,aclj,blik->abij", l1, t2, u[v, o, o, o], optimize=True)

    r2 -= np.einsum("ck,bcil,alkj->abij", l1, t2, u[v, o, o, o], optimize=True)

    r2 -= np.einsum("ak,bcil,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("bk,aclj,klic->abij", t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,abil,klcj->abij", t1, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ak,bl,ci,klcj->abij", t1, t1, t1, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ak,bl,cj,klic->abij", t1, t1, t1, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,ackm,bdln,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("cdkl,ackm,bdnj,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("cdkl,adin,bckm,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("cdkl,abin,cdkm,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("cdkl,abkn,cdmj,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("cdkl,acmj,bdin,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,ackm,bdin,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,adnj,bckm,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,abkn,cdim,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,abln,cdkm,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,abnj,cdkm,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,acim,bdnj,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("cdkl,abin,cdmj,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 4

    r2 += np.einsum("cdkl,abnj,cdim,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 4

    r2 += np.einsum("ck,al,bcim,lmkj->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("ck,bl,ackm,lmij->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("ck,bl,acmj,lmik->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("ck,cl,abim,lmkj->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ck,al,bckm,lmij->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ck,al,bcmj,lmik->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ck,bl,acim,lmkj->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ck,cl,abkm,lmij->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ck,cl,abmj,lmik->abij", l1, t1, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("cdkl,abln,ceik,dgmj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,abln,cekm,dgij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,abnj,cekm,dgil,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,adin,bglj,cekm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,adkn,bgil,cemj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,adlm,bgnj,ceik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,adnj,bglm,ceik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,agij,bdln,cekm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,agil,bdnj,cekm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,agin,bdlm,cekj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,agkm,bdln,ceij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,aglj,bdkn,ceim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdkl,aglm,bdin,cekj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += (
        np.einsum("cdkl,abin,cekl,dgmj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,ackm,bdln,egij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,aclm,bdin,egkj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,adim,bgnj,cekl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,admn,bglj,ceik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,adnj,bclm,egik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,adnj,bgkl,ceim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,aeil,bgnj,cdkm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,aekn,bglj,cdim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,aeln,bgij,cdkm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,aenj,bgil,cdkm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,agil,bdmn,cekj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,agil,bekn,cdmj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,agkl,bdin,cemj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,agmj,bdin,cekl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= np.einsum("cdkl,abin,cekm,dglj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abln,ceim,dgkj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adin,bglm,cekj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adkm,bgln,ceij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adkn,bglj,ceim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adlm,bgin,cekj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adln,bgij,cekm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adnj,bgil,cekm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,agil,bdkn,cemj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aglj,bdin,cekm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aglm,bdnj,ceik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,agnj,bdlm,ceik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= (
        np.einsum("cdkl,abmn,ceik,dglj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,abnj,cekl,dgim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,aclm,bdnj,egik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,adin,bclm,egkj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,adin,bgkl,cemj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,admj,bgin,cekl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,admn,bgil,cekj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,aein,bglj,cdkm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,aekn,bgil,cdmj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,aelj,bgin,cdkm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,agij,beln,cdkm,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,agim,bdnj,cekl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,agkl,bdnj,ceim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,aglj,bdmn,ceik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,aglj,bekn,cdim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cdkl,abln,cdkm,egij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,abln,cdmj,egik,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,abmn,cekl,dgij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,abnj,cdkm,egil,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,acim,bdnj,egkl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,aeik,bglj,cdmn,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,aekl,bgnj,cdim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,agij,bdmn,cekl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,agin,bekl,cdmj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,agkl,bdmn,ceij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,abin,cdmj,egkl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 -= (
        np.einsum("cdkl,aekl,bgij,cdmn,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += (
        np.einsum("cdkl,abin,cdkm,eglj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,abln,cdim,egkj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,acmj,bdin,egkl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,admn,bgij,cekl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,admn,bgkl,ceij,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,aekj,bgil,cdmn,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,aekl,bgin,cdmj,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,agnj,bekl,cdim,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("cdkl,abnj,cdim,egkl,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += (
        np.einsum("cdkl,agij,bekl,cdmn,mneg->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += np.einsum(
        "am,cn,cdkl,bgil,dekj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "am,ek,cdkl,bdln,cgij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "am,ek,cdkl,bdnj,cgil,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bm,cn,cdkl,aglj,deik,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bm,ei,cdkl,adln,cgkj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bm,ej,cdkl,adln,cgik,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bm,ek,cdkl,adin,cglj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ei,cdkl,abln,dgkj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ei,cdkl,adkn,bglj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ej,cdkl,abln,dgik,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ej,cdkl,adkn,bgil,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ek,cdkl,abin,dglj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ek,cdkl,adln,bgij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ek,cdkl,adnj,bgil,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,ek,cdkl,aglj,bdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ei,gj,cdkl,ackm,bdln,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ei,gk,cdkl,adnj,bclm,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ek,gj,cdkl,aclm,bdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += (
        np.einsum(
            "am,ek,cdkl,bglj,cdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,cn,cdkl,agij,dekl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,cn,cdkl,agkl,deij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,ei,cdkl,adnj,cgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,ei,cdkl,aglj,cdkn,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,ej,cdkl,adin,cgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,ej,cdkl,agil,cdkn,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,ek,cdkl,agij,cdln,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "bm,ek,cdkl,agil,cdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,dn,cdkl,aekj,bgil,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,ei,cdkl,abnj,dgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,ei,cdkl,agkl,bdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,ej,cdkl,abin,dgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,ej,cdkl,agkl,bdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "ei,gk,cdkl,abnj,cdlm,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "ek,gj,cdkl,abln,cdim,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "ek,gl,cdkl,acmj,bdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= np.einsum(
        "am,bn,cdkl,ceik,dglj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "am,cn,cdkl,bglj,deik,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "am,ei,cdkl,bdln,cgkj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "am,ej,cdkl,bdln,cgik,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "am,ek,cdkl,bdin,cglj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "bm,cn,cdkl,agil,dekj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "bm,ek,cdkl,adln,cgij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "bm,ek,cdkl,adnj,cgil,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,ei,cdkl,aglj,bdkn,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,ej,cdkl,agil,bdkn,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,ek,cdkl,abln,dgij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,ek,cdkl,abnj,dgil,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,ek,cdkl,adin,bglj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,ek,cdkl,agij,bdln,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,ek,cdkl,agil,bdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "ei,gk,cdkl,aclm,bdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "ek,gj,cdkl,adin,bclm,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= (
        np.einsum(
            "am,bn,cdkl,cekl,dgij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,cn,cdkl,bgij,dekl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,cn,cdkl,bgkl,deij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,ei,cdkl,bdnj,cgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,ei,cdkl,bglj,cdkn,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,ej,cdkl,bdin,cgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,ej,cdkl,bgil,cdkn,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,ek,cdkl,bgij,cdln,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "am,ek,cdkl,bgil,cdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "bm,ek,cdkl,aglj,cdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cm,dn,cdkl,aeik,bglj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cm,ei,cdkl,adnj,bgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cm,ej,cdkl,adin,bgkl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ei,gj,cdkl,abln,cdkm,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ei,gk,cdkl,abln,cdmj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ek,gj,cdkl,abin,cdlm,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ek,gl,cdkl,acim,bdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "bm,ei,cdkl,agkl,cdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 -= (
        np.einsum(
            "bm,ej,cdkl,agkl,cdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 -= (
        np.einsum(
            "cm,dn,cdkl,aekl,bgij,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 -= (
        np.einsum(
            "ek,gl,cdkl,abin,cdmj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "am,ei,cdkl,bgkl,cdnj,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "am,ej,cdkl,bgkl,cdin,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "cm,dn,cdkl,agij,bekl,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "ek,gl,cdkl,abnj,cdim,mneg->abij", t1, t1, l2, t2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += np.einsum("ackj,bkic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("bcik,akcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("acik,bkcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bckj,akic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("kc,ak,bcij->abij", f[o, v], t1, t2, optimize=True)

    r2 += np.einsum("ak,ci,bkcj->abij", t1, t1, u[v, o, v, o], optimize=True)

    r2 += np.einsum("ak,cj,bkic->abij", t1, t1, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("kc,bk,acij->abij", f[o, v], t1, t2, optimize=True)

    r2 -= np.einsum("kc,ci,abkj->abij", f[o, v], t1, t2, optimize=True)

    r2 -= np.einsum("kc,cj,abik->abij", f[o, v], t1, t2, optimize=True)

    r2 -= np.einsum("bk,ci,akcj->abij", t1, t1, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bk,cj,akic->abij", t1, t1, u[v, o, o, v], optimize=True)

    r2 += np.einsum("am,cdkl,bdin,cekj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("am,cdkl,bdln,ceik,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("bm,cdkl,adkn,ceij,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bm,cdkl,adln,cekj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bm,cdkl,adnj,ceik,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,abkn,deij,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,abln,dekj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,abnj,deik,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,adin,bekj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,adkn,belj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,aeij,bdkn,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,aeik,bdnj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cm,cdkl,aeil,bdkn,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ei,cdkl,ackm,bdln,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ei,cdkl,adnj,bckm,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ej,cdkl,ackm,bdln,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ej,cdkl,adin,bckm,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ek,cdkl,acim,bdnj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ek,cdkl,aclm,bdin,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ek,cdkl,adnj,bclm,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 += (
        np.einsum("am,cdkl,bdin,cekl,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("am,cdkl,beij,cdkn,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("am,cdkl,beik,cdnj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("am,cdkl,beil,cdkn,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("bm,cdkl,adnj,cekl,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("bm,cdkl,aekj,cdin,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("bm,cdkl,aelj,cdkn,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,cdkl,abnj,dekl,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,cdkl,adin,bekl,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,cdkl,aekl,bdnj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ei,cdkl,abnj,cdkm,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ej,cdkl,abin,cdkm,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,cdkl,abin,cdmj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,cdkl,abln,cdim,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,cdkl,abnj,cdlm,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= np.einsum("am,cdkl,bdkn,ceij,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("am,cdkl,bdln,cekj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("am,cdkl,bdnj,ceik,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bm,cdkl,adin,cekj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bm,cdkl,adln,ceik,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,cdkl,abin,dekj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cm,cdkl,abln,deik,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,cdkl,adkn,beij,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cm,cdkl,adkn,beil,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,cdkl,adnj,beik,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cm,cdkl,aekj,bdin,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cm,cdkl,aelj,bdkn,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ei,cdkl,ackm,bdnj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ej,cdkl,ackm,bdin,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,aclm,bdnj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,acmj,bdin,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ek,cdkl,adin,bclm,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True)

    r2 -= (
        np.einsum("am,cdkl,bdnj,cekl,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("am,cdkl,bekj,cdin,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("am,cdkl,belj,cdkn,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("bm,cdkl,adin,cekl,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("bm,cdkl,aeij,cdkn,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("bm,cdkl,aeik,cdnj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("bm,cdkl,aeil,cdkn,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,cdkl,abin,dekl,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,cdkl,adnj,bekl,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,cdkl,aekl,bdin,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ei,cdkl,abkn,cdmj,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ei,cdkl,abln,cdkm,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ej,cdkl,abkn,cdim,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ej,cdkl,abln,cdkm,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,cdkl,abin,cdlm,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,cdkl,abln,cdmj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,cdkl,abnj,cdim,mnle->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("am,cdkl,bekl,cdin,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 4
    )

    r2 -= (
        np.einsum("bm,cdkl,aekl,cdnj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 4
    )

    r2 += (
        np.einsum("am,cdkl,bekl,cdnj,mnie->abij", t1, l2, t2, t2, u[o, o, o, v], optimize=True) / 4
    )

    r2 += (
        np.einsum("bm,cdkl,aekl,cdin,mnej->abij", t1, l2, t2, t2, u[o, o, v, o], optimize=True) / 4
    )

    r2 += np.einsum("ck,abim,cdlj,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,abkm,cdil,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,abmj,cdkl,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,acil,bdmj,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,acim,bdkl,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,ackl,bdim,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,adkl,bcmj,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,adlj,bcim,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,admj,bckl,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,ablm,cdkj,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True) / 2

    r2 += np.einsum("ck,aclm,bdij,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True) / 2

    r2 += np.einsum("ck,aclm,bdik,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("ck,adkj,bclm,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ck,abim,cdkl,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,abkm,cdlj,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,abmj,cdil,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,ackl,bdmj,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,aclj,bdim,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,acmj,bdkl,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,adil,bcmj,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,adim,bckl,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,adkl,bcim,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,ablm,cdij,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ck,ablm,cdik,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("ck,aclm,bdkj,lmid->abij", l1, t2, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ck,adij,bclm,lmkd->abij", l1, t2, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ck,adik,bclm,lmdj->abij", l1, t2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("ck,al,bm,cdkj,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,al,cm,bdij,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,al,cm,bdik,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,al,di,bcmj,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,al,dj,bcim,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,al,dk,bcim,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,bl,cm,adkj,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,bl,di,ackm,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,bl,dj,ackm,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,bl,dk,acmj,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,cl,di,abmj,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,cl,dj,abim,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,cl,dk,abim,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,al,bm,cdij,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,al,bm,cdik,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,al,cm,bdkj,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,al,di,bckm,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,al,dj,bckm,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,al,dk,bcmj,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,bl,cm,adij,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,bl,cm,adik,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,bl,di,acmj,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,bl,dj,acim,lmkd->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,bl,dk,acim,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,cl,di,abkm,lmdj->abij", l1, t1, t1, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,cl,dj,abkm,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,cl,dk,abmj,lmid->abij", l1, t1, t1, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bc,acij->abij", f[v, v], t2, optimize=True)

    r2 += np.einsum("ci,abcj->abij", t1, u[v, v, v, o], optimize=True)

    r2 += np.einsum("cj,abic->abij", t1, u[v, v, o, v], optimize=True)

    r2 -= np.einsum("ac,bcij->abij", f[v, v], t2, optimize=True)

    r2 += np.einsum("abkl,klij->abij", t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("ak,bl,klij->abij", t1, t1, u[o, o, o, o], optimize=True)

    return r2


def amplitudes_qccsd(t1, t2, l1, l2, u, f, v, o):
    t1 = amplitudes_t1_qccsd(t1, t2, l1, l2, u, f, v, o)
    t2 = amplitudes_t2_qccsd(t1, t2, l1, l2, u, f, v, o)

    return t1, t2
