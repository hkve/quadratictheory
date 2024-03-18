import numpy as np
from clusterfock.cc.rhs.l_CCSD import lambda_amplitudes_ccsd


def lambda_amplitudes_qccsd(t1, t2, l1, l2, u, f, v, o):
    r1, r2 = lambda_amplitudes_ccsd(t1, t2, l1, l2, u, f, v, o)

    r1 += lambda_amplitudes_l1_qccsd(t1, t2, l1, l2, u, f, v, o)
    r2 += lambda_amplitudes_l2_qccsd(t1, t2, l1, l2, u, f, v, o)

    return r1, r2


def lambda_amplitudes_l1_qccsd(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r1 = np.zeros((M, N))

    r1 += np.einsum("aj,bk,ic,bcjk->ai", l1, l1, f[o, v], t2, optimize=True)

    r1 += np.einsum("aj,bk,cj,ibkc->ai", l1, l1, t1, u[o, v, o, v], optimize=True)

    r1 += np.einsum("bi,cj,ka,bcjk->ai", l1, l1, f[o, v], t2, optimize=True)

    r1 += np.einsum("bi,cj,ck,kbaj->ai", l1, l1, t1, u[o, v, v, o], optimize=True)

    r1 -= np.einsum("aj,bk,ck,ibjc->ai", l1, l1, t1, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bi,cj,bk,kcaj->ai", l1, l1, t1, u[o, v, v, o], optimize=True)

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

    r1 -= np.einsum("aj,bk,ibjk->ai", l1, l1, u[o, v, o, o], optimize=True)

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

    r1 += np.einsum("bi,cdjk,cejk,bdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True) / 2

    r1 += np.einsum("bj,cdik,bejk,cdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True) / 2

    r1 -= np.einsum("bj,cdik,cejk,bdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cdjk,bejk,cdae->ai", l1, l2, t2, u[v, v, v, v], optimize=True) / 4

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

    r1 += np.einsum("aj,bckl,bckm,imjl->ai", l1, l2, t2, u[o, o, o, o], optimize=True) / 2

    r1 += np.einsum("bj,ackl,bcjm,imkl->ai", l1, l2, t2, u[o, o, o, o], optimize=True) / 2

    r1 -= np.einsum("bj,ackl,bckm,imjl->ai", l1, l2, t2, u[o, o, o, o], optimize=True)

    r1 -= np.einsum("aj,bckl,bcjm,imkl->ai", l1, l2, t2, u[o, o, o, o], optimize=True) / 4

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

    r1 += np.einsum("bi,cj,bcaj->ai", l1, l1, u[v, v, v, o], optimize=True)

    r1 += np.einsum("bi,cj,bdjk,kcad->ai", l1, l1, t2, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bj,ck,bdjk,icad->ai", l1, l1, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,cdjk,kbad->ai", l1, l1, t2, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("aj,bk,cdjk,ibcd->ai", l1, l1, t2, u[o, v, v, v], optimize=True) / 2

    r1 += np.einsum("bi,cj,ck,dj,kbad->ai", l1, l1, t1, t1, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("aj,bk,cj,dk,ibcd->ai", l1, l1, t1, t1, u[o, v, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,bk,dj,kcad->ai", l1, l1, t1, t1, u[o, v, v, v], optimize=True)

    r1 += np.einsum("bi,cdjk,bcjl,ldak->ai", l1, l2, t2, u[o, v, v, o], optimize=True)

    r1 += np.einsum("bj,ackl,bdjk,icld->ai", l1, l2, t2, u[o, v, o, v], optimize=True)

    r1 += np.einsum("bj,cdik,bckl,ldaj->ai", l1, l2, t2, u[o, v, v, o], optimize=True)

    r1 += np.einsum("bi,cdjk,cdjl,lbak->ai", l1, l2, t2, u[o, v, v, o], optimize=True) / 2

    r1 += np.einsum("bj,ackl,bdkl,icjd->ai", l1, l2, t2, u[o, v, o, v], optimize=True) / 2

    r1 += np.einsum("bj,cdik,cdkl,lbaj->ai", l1, l2, t2, u[o, v, v, o], optimize=True) / 2

    r1 -= np.einsum("aj,bckl,bdjk,icld->ai", l1, l2, t2, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bj,ackl,cdjk,ibld->ai", l1, l2, t2, u[o, v, o, v], optimize=True)

    r1 -= np.einsum("bj,cdik,bcjl,ldak->ai", l1, l2, t2, u[o, v, v, o], optimize=True)

    r1 -= np.einsum("aj,bckl,bdkl,icjd->ai", l1, l2, t2, u[o, v, o, v], optimize=True) / 2

    r1 -= np.einsum("bj,ackl,cdkl,ibjd->ai", l1, l2, t2, u[o, v, o, v], optimize=True) / 2

    r1 -= np.einsum("bj,cdik,cdjl,lbak->ai", l1, l2, t2, u[o, v, v, o], optimize=True) / 2

    r1 += np.einsum("aj,bk,ck,bdjl,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bi,cj,ck,bdjl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,ck,bl,cdjk,ilad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bj,ck,dj,bckl,ilad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("aj,bk,bl,cdjk,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r1 += np.einsum("bi,cj,dj,bckl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True) / 2

    r1 -= np.einsum("aj,bk,cj,bdkl,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("aj,bk,cl,bdjk,ilcd->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,bk,cdjl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 -= np.einsum("bi,cj,dk,bcjl,klad->ai", l1, l1, t1, t2, u[o, o, v, v], optimize=True)

    r1 += np.einsum("aj,bk,bl,cj,dk,ilcd->ai", l1, l1, t1, t1, t1, u[o, o, v, v], optimize=True)

    r1 += np.einsum("bi,cj,bk,cl,dj,klad->ai", l1, l1, t1, t1, t1, u[o, o, v, v], optimize=True)

    r1 += np.einsum("aj,bk,bl,iljk->ai", l1, l1, t1, u[o, o, o, o], optimize=True)

    r1 += np.einsum("aj,bk,bcjl,ilkc->ai", l1, l1, t2, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bi,cj,bckl,klaj->ai", l1, l1, t2, u[o, o, v, o], optimize=True) / 2

    r1 -= np.einsum("aj,bk,bckl,iljc->ai", l1, l1, t2, u[o, o, o, v], optimize=True)

    r1 -= np.einsum("bj,ck,bcjl,ilak->ai", l1, l1, t2, u[o, o, v, o], optimize=True)

    r1 += np.einsum("aj,bk,bl,ck,iljc->ai", l1, l1, t1, t1, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bi,cj,bk,cl,klaj->ai", l1, l1, t1, t1, u[o, o, v, o], optimize=True)

    r1 -= np.einsum("aj,bk,bl,cj,ilkc->ai", l1, l1, t1, t1, u[o, o, o, v], optimize=True)

    r1 += np.einsum("bi,cj,dj,bcad->ai", l1, l1, t1, u[v, v, v, v], optimize=True)

    return r1


def lambda_amplitudes_l2_qccsd(t1, t2, l1, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r2 = np.zeros((M, M, N, N))

    r2 += np.einsum("cm,aeij,cdkl,dekn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cm,aeil,cdkj,deln,mnbk->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cm,aekl,cdij,dekn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cm,aelj,cdik,dekn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cm,beil,cdkj,dekn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cm,belj,cdik,deln,mnak->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cn,ablm,cdik,dekl,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cn,abmj,cdkl,dekm,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cn,ackl,bdim,dekm,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cn,admj,bckl,dekm,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("dn,ackl,bdmj,cekm,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("dn,adim,bckl,cekm,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ek,ackl,bdim,cdmn,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ek,ackl,bdmj,cdln,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ek,adim,bckl,cdln,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ek,admj,bckl,cdmn,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("em,ackl,bdmj,cdkn,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("em,adim,bckl,cdkn,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 += (
        np.einsum("cn,abim,cdkl,dekl,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cn,ablm,cdik,delm,njke->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cn,ackl,bdmj,dekl,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cn,adim,bckl,dekl,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("dn,ackl,bdim,cekl,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("dn,admj,bckl,cekl,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,abim,cdkl,cdln,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,ablm,cdik,cdln,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,abmj,cdkl,cdmn,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("el,ablm,cdik,cdmn,njke->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("el,ablm,cdkj,cdkn,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,abim,cdkl,cdkn,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,aeij,cdkl,cdkn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,aeil,cdkj,cdln,mnbk->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,aekl,cdij,cdkn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,aelj,cdik,cdkn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,beil,cdkj,cdkn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,belj,cdik,cdln,mnak->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= np.einsum("cm,aeil,cdkj,dekn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,aelj,cdik,deln,mnbk->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,beij,cdkl,dekn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,beil,cdkj,deln,mnak->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,bekl,cdij,dekn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cm,belj,cdik,dekn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cn,abim,cdkl,dekm,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cn,ablm,cdkj,dekl,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cn,ackl,bdmj,dekm,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cn,adim,bckl,dekm,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("dn,ackl,bdim,cekm,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("dn,admj,bckl,cekm,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ek,ackl,bdim,cdln,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ek,ackl,bdmj,cdmn,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ek,adim,bckl,cdmn,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ek,admj,bckl,cdln,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("em,ackl,bdim,cdkn,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("em,admj,bckl,cdkn,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= (
        np.einsum("cn,ablm,cdkj,delm,inke->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cn,abmj,cdkl,dekl,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cn,ackl,bdim,dekl,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cn,admj,bckl,dekl,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("dn,ackl,bdmj,cekl,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("dn,adim,bckl,cekl,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,abim,cdkl,cdmn,njle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,ablm,cdkj,cdln,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,abmj,cdkl,cdln,inme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("el,ablm,cdik,cdkn,njme->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("el,ablm,cdkj,cdmn,inke->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,abmj,cdkl,cdkn,inle->abij", t1, l2, l2, t2, u[o, o, o, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,aeil,cdkj,cdkn,mnbl->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,aelj,cdik,cdln,mnbk->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,beij,cdkl,cdkn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,beil,cdkj,cdln,mnak->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,bekl,cdij,cdkn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,belj,cdik,cdkn,mnal->abij", t1, l2, l2, t2, u[o, o, v, o], optimize=True) / 2
    )

    r2 += np.einsum("aeil,cdkj,cgkl,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("belj,cdik,cgkl,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("aeil,cdkj,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("beij,cdkl,cgkl,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("bekl,cdij,cgkl,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("belj,cdik,egkl,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("aelj,cdik,cgkl,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("beil,cdkj,cgkl,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("aeij,cdkl,cgkl,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("aekl,cdij,cgkl,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("aelj,cdik,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("beil,cdkj,egkl,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("aeij,cdkl,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 4

    r2 -= np.einsum("aekl,cdij,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 4

    r2 += np.einsum("beij,cdkl,egkl,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 4

    r2 += np.einsum("bekl,cdij,egkl,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 4

    r2 += np.einsum("ackl,bdmj,cdkn,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("adim,bckl,cdkn,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("abim,cdkl,cdkn,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("ablm,cdkj,cdln,inkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("ackl,bdmj,cdmn,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("adim,bckl,cdmn,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("ackl,bdim,cdkn,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("admj,bckl,cdkn,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ablm,cdik,cdln,njkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("abmj,cdkl,cdkn,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("ackl,bdim,cdmn,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("admj,bckl,cdmn,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("ablm,cdkj,cdkn,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 4

    r2 -= np.einsum("abmj,cdkl,cdmn,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 4

    r2 += np.einsum("abim,cdkl,cdmn,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 4

    r2 += np.einsum("ablm,cdik,cdkn,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 4

    r2 += np.einsum("ak,ci,cjbk->abij", l1, l1, u[v, o, v, o], optimize=True)

    r2 += np.einsum("ak,cj,icbk->abij", l1, l1, u[o, v, v, o], optimize=True)

    r2 += np.einsum("bi,ck,cjak->abij", l1, l1, u[v, o, v, o], optimize=True)

    r2 += np.einsum("bj,ck,icak->abij", l1, l1, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ai,ck,cjbk->abij", l1, l1, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("aj,ck,icbk->abij", l1, l1, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("bk,ci,cjak->abij", l1, l1, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bk,cj,icak->abij", l1, l1, u[o, v, v, o], optimize=True)

    r2 += np.einsum("aj,bk,ic,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 += np.einsum("aj,ci,kb,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 += np.einsum("ak,bi,jc,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 += np.einsum("bi,cj,ka,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 -= np.einsum("ai,bk,jc,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 -= np.einsum("ai,cj,kb,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 -= np.einsum("ak,bj,ic,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 -= np.einsum("bj,ci,ka,ck->abij", l1, l1, f[o, v], t1, optimize=True)

    r2 += np.einsum("ci,ek,adkj,cdbe->abij", l1, t1, l2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("cj,ek,adik,cdbe->abij", l1, t1, l2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ck,ek,bdij,cdae->abij", l1, t1, l2, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ak,ek,cdij,cdbe->abij", l1, t1, l2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("bi,ek,cdkj,cdae->abij", l1, t1, l2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("bj,ek,cdik,cdae->abij", l1, t1, l2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ci,ek,bdkj,cdae->abij", l1, t1, l2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cj,ek,bdik,cdae->abij", l1, t1, l2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("ck,ek,adij,cdbe->abij", l1, t1, l2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("ai,ek,cdkj,cdbe->abij", l1, t1, l2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("aj,ek,cdik,cdbe->abij", l1, t1, l2, u[v, v, v, v], optimize=True) / 2

    r2 -= np.einsum("bk,ek,cdij,cdae->abij", l1, t1, l2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("ak,bclj,ickl->abij", l1, l2, u[o, v, o, o], optimize=True)

    r2 += np.einsum("bk,acil,cjkl->abij", l1, l2, u[v, o, o, o], optimize=True)

    r2 += np.einsum("ck,ablj,ickl->abij", l1, l2, u[o, v, o, o], optimize=True)

    r2 += np.einsum("ai,bckl,cjkl->abij", l1, l2, u[v, o, o, o], optimize=True) / 2

    r2 += np.einsum("aj,bckl,ickl->abij", l1, l2, u[o, v, o, o], optimize=True) / 2

    r2 += np.einsum("ci,abkl,cjkl->abij", l1, l2, u[v, o, o, o], optimize=True) / 2

    r2 += np.einsum("cj,abkl,ickl->abij", l1, l2, u[o, v, o, o], optimize=True) / 2

    r2 -= np.einsum("ak,bcil,cjkl->abij", l1, l2, u[v, o, o, o], optimize=True)

    r2 -= np.einsum("bk,aclj,ickl->abij", l1, l2, u[o, v, o, o], optimize=True)

    r2 -= np.einsum("ck,abil,cjkl->abij", l1, l2, u[v, o, o, o], optimize=True)

    r2 -= np.einsum("bi,ackl,cjkl->abij", l1, l2, u[v, o, o, o], optimize=True) / 2

    r2 -= np.einsum("bj,ackl,ickl->abij", l1, l2, u[o, v, o, o], optimize=True) / 2

    r2 += np.einsum("ai,bk,cl,ljkc->abij", l1, l1, t1, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ai,ck,cl,ljbk->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 += np.einsum("aj,bk,cl,ilkc->abij", l1, l1, t1, u[o, o, o, v], optimize=True)

    r2 += np.einsum("aj,ck,cl,ilbk->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ak,bl,cl,ijkc->abij", l1, l1, t1, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,ci,cl,ljak->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 += np.einsum("bk,cj,cl,ilak->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ak,bi,cl,ljkc->abij", l1, l1, t1, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,bj,cl,ilkc->abij", l1, l1, t1, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,bl,ck,ijlc->abij", l1, l1, t1, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,ci,cl,ljbk->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ak,cj,cl,ilbk->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("bi,ck,cl,ljak->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("bj,ck,cl,ilak->abij", l1, l1, t1, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ai,cl,cdkj,ldbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("aj,cl,cdik,ldbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ak,ci,cdkl,ljbd->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,cj,cdkl,ilbd->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,dk,bcil,cjld->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ak,dl,bclj,ickd->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("bi,ck,cdkl,ljad->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bi,dk,ackl,cjld->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("bj,ck,cdkl,ilad->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bj,dk,ackl,icld->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("bk,cl,cdkl,ijad->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,id,aclj,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 += np.einsum("bk,jd,acil,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 += np.einsum("bk,cl,cdij,ldak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("bk,dk,aclj,icld->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("bk,dl,acil,cjkd->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ci,dk,cdkl,ljab->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,lb,adkj,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 += np.einsum("ci,cl,bdkj,ldak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ci,dl,adkj,lcbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cj,lb,adik,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 += np.einsum("cj,cl,bdik,ldak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cj,dl,adik,lcbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ck,la,bdij,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 += np.einsum("ck,cl,adij,ldbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ck,dk,abil,cjld->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ck,dl,ablj,ickd->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("ck,dl,bdij,lcak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("ai,bk,cdkl,ljcd->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ai,jd,bckl,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 += np.einsum("aj,bk,cdkl,ilcd->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("aj,ci,cdkl,klbd->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ak,bl,cdkl,ijcd->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ak,lb,cdij,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 += np.einsum("bi,cj,cdkl,klad->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bi,la,cdkj,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 += np.einsum("bj,id,ackl,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 += np.einsum("bj,la,cdik,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 += np.einsum("ci,dj,cdkl,klab->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,jd,abkl,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 -= np.einsum("ai,ck,cdkl,ljbd->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ai,dk,bckl,cjld->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("aj,ck,cdkl,ilbd->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aj,dk,bckl,icld->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ak,cl,cdkl,ijbd->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,id,bclj,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 -= np.einsum("ak,jd,bcil,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 -= np.einsum("ak,cl,cdij,ldbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ak,dk,bclj,icld->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ak,dl,bcil,cjkd->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("bi,cl,cdkj,ldak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("bj,cl,cdik,ldak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("bk,ci,cdkl,ljad->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cj,cdkl,ilad->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,dk,acil,cjld->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("bk,dl,aclj,ickd->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ci,la,bdkj,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 -= np.einsum("ci,cl,adkj,ldbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ci,dk,abkl,cjld->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ci,dl,bdkj,lcak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("cj,la,bdik,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 -= np.einsum("cj,cl,adik,ldbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("cj,dk,abkl,icld->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("cj,dl,bdik,lcak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ck,dj,cdkl,ilab->abij", l1, l1, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,id,ablj,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 -= np.einsum("ck,jd,abil,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 -= np.einsum("ck,lb,adij,cdkl->abij", l1, f[o, v], l2, t2, optimize=True)

    r2 -= np.einsum("ck,cl,bdij,ldak->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ck,dk,ablj,icld->abij", l1, t1, l2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ck,dl,abil,cjkd->abij", l1, t1, l2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ck,dl,adij,lcbk->abij", l1, t1, l2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ai,cj,cdkl,klbd->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ai,lb,cdkj,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 -= np.einsum("aj,id,bckl,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 -= np.einsum("aj,lb,cdik,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 -= np.einsum("ak,bi,cdkl,ljcd->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,bj,cdkl,ilcd->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bi,jd,ackl,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 -= np.einsum("bj,ci,cdkl,klad->abij", l1, l1, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bk,la,cdij,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 -= np.einsum("cj,id,abkl,cdkl->abij", l1, f[o, v], l2, t2, optimize=True) / 2

    r2 += np.einsum("ai,bk,ck,dl,ljcd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ai,ck,cl,dk,ljbd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aj,bk,ck,dl,ilcd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aj,ci,ck,dl,klbd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aj,ck,cl,dk,ilbd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,bl,ck,dl,ijcd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bi,cj,ck,dl,klad->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,ci,cl,dk,ljad->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,cj,cl,dk,ilad->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,dj,ck,dl,klab->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ai,cj,ck,dl,klbd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,bi,ck,dl,ljcd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,bj,ck,dl,ilcd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,ci,cl,dk,ljbd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,cj,cl,dk,ilbd->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bi,ck,cl,dk,ljad->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bj,ci,ck,dl,klad->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bj,ck,cl,dk,ilad->abij", l1, l1, t1, t1, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,cm,bcil,mjkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("bk,cm,aclj,imkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("ck,cm,abil,mjkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True)

    r2 += np.einsum("bi,cm,ackl,mjkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("bj,cm,ackl,imkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("ak,cm,bclj,imkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("bk,cm,acil,mjkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ck,cm,ablj,imkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("ai,cm,bckl,mjkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("aj,cm,bckl,imkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("ci,cm,abkl,mjkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True) / 2

    r2 -= np.einsum("cj,cm,abkl,imkl->abij", l1, t1, l2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("ablm,cdik,cekl,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("abmj,cdkl,cekm,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("ackl,bdim,cekm,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("ackl,bdmj,dekm,icle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("adim,bckl,dekm,cjle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("admj,bckl,cekm,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("aeil,cdkj,cekm,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("aelj,cdik,celm,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("beij,cdkl,cekm,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("beil,cdkj,celm,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("bekl,cdij,cekm,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("belj,cdik,cekm,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("abim,cdkl,cekl,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("ablm,cdik,celm,djke->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("ackl,bdim,dekl,cjme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("ackl,bdmj,cekl,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 2

    r2 += np.einsum("adim,bckl,cekl,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 2

    r2 += np.einsum("admj,bckl,dekl,icme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 2

    r2 += np.einsum("aeij,cdkl,cdkm,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("aeil,cdkj,cdlm,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("aekl,cdij,cdkm,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("aelj,cdik,cdkm,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("beil,cdkj,cdkm,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("belj,cdik,cdlm,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 -= np.einsum("abim,cdkl,cekm,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ablm,cdkj,cekl,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("ackl,bdim,dekm,cjle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("ackl,bdmj,cekm,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("adim,bckl,cekm,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 -= np.einsum("admj,bckl,dekm,icle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 -= np.einsum("aeij,cdkl,cekm,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("aeil,cdkj,celm,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("aekl,cdij,cekm,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("aelj,cdik,cekm,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("beil,cdkj,cekm,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("belj,cdik,celm,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= np.einsum("ablm,cdkj,celm,idke->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 2

    r2 -= np.einsum("abmj,cdkl,cekl,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 2

    r2 -= np.einsum("ackl,bdim,cekl,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ackl,bdmj,dekl,icme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 2

    r2 -= np.einsum("adim,bckl,dekl,cjme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 2

    r2 -= np.einsum("admj,bckl,cekl,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 2

    r2 -= np.einsum("aeil,cdkj,cdkm,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 -= np.einsum("aelj,cdik,cdlm,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 -= np.einsum("beij,cdkl,cdkm,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 -= np.einsum("beil,cdkj,cdlm,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 -= np.einsum("bekl,cdij,cdkm,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 -= np.einsum("belj,cdik,cdkm,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 2

    r2 += np.einsum("ai,cl,cdkj,dekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ai,dk,bckl,celm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aj,cl,cdik,dekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aj,dk,bckl,celm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,cm,cdil,dekl,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,dk,bclj,celm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,dl,bcil,cekm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,dl,bclm,cekm,ijde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ak,dm,bclj,cekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,cl,cdij,dekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,cm,cdlj,dekl,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,dk,acil,celm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,dl,aclj,cekm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,dm,acil,cekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,cl,bdkj,dekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,dk,abkl,celm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,dl,adkj,cekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,ek,adkl,cdlm,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,el,bdkj,cdkm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cj,cl,bdik,dekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cj,dk,abkl,celm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cj,dl,adik,cekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cj,ek,adkl,cdlm,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cj,el,bdik,cdkm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,adij,dekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cm,adlj,dekl,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cm,bdil,dekl,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dk,ablj,celm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,abil,cekm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,ablm,cekm,ijde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,bdij,cekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dl,deij,cekm,lmab->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dm,ablj,cekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dm,adil,cekl,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,dm,bdlj,cekl,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,ek,adlj,cdlm,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,ek,bdil,cdlm,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,el,adij,cdkm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,el,adil,cdkm,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,el,bdlj,cdkm,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ai,dm,bckl,cekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ai,el,cdkj,cdkm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("aj,dm,bckl,cekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("aj,el,cdik,cdkm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ak,cm,bcil,dekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ak,ek,cdil,cdlm,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ak,el,cdlj,cdkm,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bi,cm,cdkl,dekl,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bi,ek,cdkl,cdlm,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bj,cm,cdkl,dekl,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bj,ek,cdkl,cdlm,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bk,cm,aclj,dekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bk,dk,aclm,celm,ijde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bk,ek,cdlj,cdlm,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bk,el,cdij,cdkm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bk,el,cdil,cdkm,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,cl,dekj,dekm,lmab->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,cm,adkl,dekl,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dm,abkl,cekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dm,bdkl,cekl,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,ek,adkj,cdlm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("cj,cl,deik,dekm,lmab->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("cj,cm,adkl,dekl,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("cj,dm,abkl,cekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("cj,dm,bdkl,cekl,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("cj,ek,adik,cdlm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,cm,abil,dekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("ck,ek,bdij,cdlm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,cl,cdij,dekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,cm,cdlj,dekl,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,dk,bcil,celm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,dl,bclj,cekm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ak,dm,bcil,cekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bi,cl,cdkj,dekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bi,dk,ackl,celm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bj,cl,cdik,dekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bj,dk,ackl,celm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cm,cdil,dekl,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,dk,aclj,celm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,dl,acil,cekm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,dl,aclm,cekm,ijde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,dm,aclj,cekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ci,cl,adkj,dekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ci,dl,bdkj,cekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ci,dl,dekj,cekm,lmab->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ci,ek,bdkl,cdlm,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ci,el,adkj,cdkm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cj,cl,adik,dekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cj,dl,bdik,cekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cj,dl,deik,cekm,lmab->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cj,ek,bdkl,cdlm,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cj,el,adik,cdkm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,bdij,dekm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cm,adil,dekl,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cm,bdlj,dekl,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,abil,celm,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,ablj,cekm,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,adij,cekm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dm,abil,cekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dm,adlj,cekl,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dm,bdil,cekl,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,ek,adil,cdlm,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,ek,bdlj,cdlm,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,el,adlj,cdkm,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,el,bdij,cdkm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,el,bdil,cdkm,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ai,cm,cdkl,dekl,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ai,ek,cdkl,cdlm,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("aj,cm,cdkl,dekl,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("aj,ek,cdkl,cdlm,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,cm,bclj,dekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,dk,bclm,celm,ijde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,ek,cdlj,cdlm,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,el,cdij,cdkm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,el,cdil,cdkm,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bi,dm,ackl,cekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bi,el,cdkj,cdkm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bj,dm,ackl,cekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bj,el,cdik,cdkm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bk,cm,acil,dekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bk,ek,cdil,cdlm,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bk,el,cdlj,cdkm,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ci,cm,bdkl,dekl,mjae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ci,dm,adkl,cekl,mjbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ci,ek,bdkj,cdlm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("cj,cm,bdkl,dekl,imae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("cj,dm,adkl,cekl,imbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("cj,ek,bdik,cdlm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,cl,deij,dekm,lmab->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,cm,ablj,dekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,dk,ablm,celm,ijde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,ek,adij,cdlm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ai,cm,bckl,dekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 -= np.einsum("ai,ek,cdkj,cdlm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 -= np.einsum("aj,cm,bckl,dekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 -= np.einsum("aj,ek,cdik,cdlm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 -= np.einsum("bk,ek,cdij,cdlm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 -= np.einsum("ci,cm,abkl,dekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 -= np.einsum("cj,cm,abkl,dekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("ak,ek,cdij,cdlm,lmbe->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("bi,cm,ackl,dekl,mjde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("bi,ek,cdkj,cdlm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("bj,cm,ackl,dekl,imde->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("bj,ek,cdik,cdlm,lmae->abij", l1, t1, l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("ak,cm,dk,el,bcil,mjde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bk,cm,dk,el,aclj,imde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ci,cl,dm,ek,adkj,lmbe->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cj,cl,dm,ek,adik,lmbe->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cl,dm,ek,bdij,lmae->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ck,cm,dk,el,abil,mjde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 += (
        np.einsum("ak,cl,dm,ek,cdij,lmbe->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("bi,cl,dm,ek,cdkj,lmae->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("bi,cm,dk,el,ackl,mjde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("bj,cl,dm,ek,cdik,lmae->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("bj,cm,dk,el,ackl,imde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= np.einsum("ak,cm,dk,el,bclj,imde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cm,dk,el,acil,mjde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ci,cl,dm,ek,bdkj,lmae->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cj,cl,dm,ek,bdik,lmae->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,dm,ek,adij,lmbe->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ck,cm,dk,el,ablj,imde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)

    r2 -= (
        np.einsum("ai,cl,dm,ek,cdkj,lmbe->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("ai,cm,dk,el,bckl,mjde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aj,cl,dm,ek,cdik,lmbe->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aj,cm,dk,el,bckl,imde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("bk,cl,dm,ek,cdij,lmae->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("ci,cm,dk,el,abkl,mjde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("cj,cm,dk,el,abkl,imde->abij", l1, t1, t1, t1, l2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += np.einsum("ak,bcil,cdkm,mjld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ak,bclj,cdlm,imkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bi,ackl,cdkm,mjld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bj,ackl,cdkm,imld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,acil,cdlm,mjkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,aclj,cdkm,imld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,aclm,cdkl,ijmd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ci,bdkl,cdkm,mjal->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cj,bdkl,cdkm,imal->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,abil,cdkm,mjld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,ablj,cdlm,imkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,adil,cdkm,mjbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,adlj,cdlm,imbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,bdil,cdlm,mjak->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,bdlj,cdkm,imal->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ai,cdkl,cdkm,mjbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("aj,cdkl,cdkm,imbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("ak,cdil,cdlm,mjbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("ak,cdlj,cdkm,imbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("bk,aclm,cdlm,ijkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r2 += np.einsum("bk,cdil,cdkm,mjal->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("bk,cdlj,cdlm,imak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("ci,adkj,cdlm,lmbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("cj,adik,cdlm,lmbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("ck,bdij,cdlm,lmak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("ai,bckl,cdkm,mjld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("aj,bckl,cdkm,imld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,bcil,cdlm,mjkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,bclj,cdkm,imld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,bclm,cdkl,ijmd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bk,acil,cdkm,mjld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bk,aclj,cdlm,imkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ci,abkl,cdkm,mjld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ci,adkl,cdkm,mjbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cj,abkl,cdkm,imld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("cj,adkl,cdkm,imbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,abil,cdlm,mjkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,ablj,cdkm,imld->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,ablm,cdkl,ijmd->abij", l1, l2, t2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,adil,cdlm,mjbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,adlj,cdkm,imbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,bdil,cdkm,mjal->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,bdlj,cdlm,imak->abij", l1, l2, t2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ak,bclm,cdlm,ijkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ak,cdil,cdkm,mjbl->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("ak,cdlj,cdlm,imbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("bi,cdkl,cdkm,mjal->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("bj,cdkl,cdkm,imal->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("bk,cdil,cdlm,mjak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("bk,cdlj,cdkm,imal->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("ci,bdkj,cdlm,lmak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("cj,bdik,cdlm,lmak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("ck,ablm,cdlm,ijkd->abij", l1, l2, t2, u[o, o, o, v], optimize=True) / 2

    r2 -= np.einsum("ck,adij,cdlm,lmbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("ai,cdkj,cdlm,lmbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 4

    r2 -= np.einsum("aj,cdik,cdlm,lmbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 4

    r2 -= np.einsum("bk,cdij,cdlm,lmak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 4

    r2 += np.einsum("ak,cdij,cdlm,lmbk->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 4

    r2 += np.einsum("bi,cdkj,cdlm,lmak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 4

    r2 += np.einsum("bj,cdik,cdlm,lmak->abij", l1, l2, t2, u[o, o, v, o], optimize=True) / 4

    r2 += np.einsum("ai,cm,dk,bckl,mjld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("aj,cm,dk,bckl,imld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ak,cm,dk,bclj,imld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ak,cm,dl,bcil,mjkd->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,cm,dk,acil,mjld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("bk,cm,dl,aclj,imkd->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ci,cl,dm,adkj,lmbk->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ci,cm,dk,abkl,mjld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("cj,cl,dm,adik,lmbk->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("cj,cm,dk,abkl,imld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,cl,dm,bdij,lmak->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True)

    r2 += np.einsum("ck,cm,dk,ablj,imld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ck,cm,dl,abil,mjkd->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 += np.einsum("ak,cl,dm,cdij,lmbk->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("bi,cl,dm,cdkj,lmak->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("bj,cl,dm,cdik,lmak->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("ak,cm,dk,bcil,mjld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ak,cm,dl,bclj,imkd->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bi,cm,dk,ackl,mjld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bj,cm,dk,ackl,imld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bk,cm,dk,aclj,imld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("bk,cm,dl,acil,mjkd->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ci,cl,dm,bdkj,lmak->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("cj,cl,dm,bdik,lmak->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,cl,dm,adij,lmbk->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True)

    r2 -= np.einsum("ck,cm,dk,abil,mjld->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ck,cm,dl,ablj,imkd->abij", l1, t1, t1, l2, u[o, o, o, v], optimize=True)

    r2 -= np.einsum("ai,cl,dm,cdkj,lmbk->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("aj,cl,dm,cdik,lmbk->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r2 -= np.einsum("bk,cl,dm,cdij,lmak->abij", l1, t1, t1, l2, u[o, o, v, o], optimize=True) / 2

    r2 += np.einsum("abim,cdkl,cekm,dgln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ablm,cdkj,cekl,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ackl,bdim,cgln,dekm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ackl,bdmj,cekm,dgln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("adim,bckl,cekm,dgln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("admj,bckl,cgln,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aeil,cdkj,cekm,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aeim,cdkl,cgkm,deln,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aelj,cdik,cgkm,deln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("aelm,cdkj,cgkl,demn,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("beij,cdkl,cekm,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("beil,cdkj,cgkm,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bekl,cdij,cekm,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("belj,cdik,cekm,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("belm,cdik,cgkl,demn,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bemj,cdkl,cgkm,deln,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("cdik,eglj,cekm,dgln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 += (
        np.einsum("ablm,cdkj,celm,dgkn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("abmj,cdkl,cekl,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("abmn,cdkl,cekm,dgln,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("ackl,bdim,cdln,egkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("ackl,bdim,cekl,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("ackl,bdmj,cgmn,dekl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("adim,bckl,cgmn,dekl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("admj,bckl,cdln,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("admj,bckl,cekl,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aeij,cdkl,cdkm,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aeil,cdkj,cdlm,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aekl,cdij,cdkm,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aelj,cdik,cdkm,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aelj,cdik,cemn,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aelm,cdik,cdln,egkm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aelm,cdkj,cglm,dekn,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aemj,cdkl,cdkn,eglm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("aemj,cdkl,cgkl,demn,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("beil,cdkj,cdkm,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("beil,cdkj,cemn,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("beim,cdkl,cdkn,eglm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("beim,cdkl,cgkl,demn,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("belj,cdik,cdlm,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("belm,cdik,cglm,dekn,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("belm,cdkj,cdln,egkm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 += (
        np.einsum("cdkl,egij,cekm,dgln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= np.einsum("ablm,cdik,cekl,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abmj,cdkl,cekm,dgln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ackl,bdim,cekm,dgln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ackl,bdmj,cgln,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("ackl,bdmn,cekm,dgln,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("adim,bckl,cgln,dekm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("admj,bckl,cekm,dgln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aeij,cdkl,cekm,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aeil,cdkj,cgkm,deln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aekl,cdij,cekm,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aelj,cdik,cekm,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aelm,cdik,cgkl,demn,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("aemj,cdkl,cgkm,deln,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("beil,cdkj,cekm,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("beim,cdkl,cgkm,deln,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("belj,cdik,cgkm,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("belm,cdkj,cgkl,demn,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= (
        np.einsum("abim,cdkl,cekl,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("ablm,cdik,celm,dgkn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("ackl,bdim,cgmn,dekl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("ackl,bdmj,cdln,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("ackl,bdmj,cekl,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("adim,bckl,cdln,egkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("adim,bckl,cekl,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("admj,bckl,cgmn,dekl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aeil,cdkj,cdkm,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aeil,cdkj,cemn,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aeim,cdkl,cdkn,eglm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aeim,cdkl,cgkl,demn,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aelj,cdik,cdlm,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aelm,cdik,cglm,dekn,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("aelm,cdkj,cdln,egkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("beij,cdkl,cdkm,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("beil,cdkj,cdlm,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("bekl,cdij,cdkm,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("belj,cdik,cdkm,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("belj,cdik,cemn,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("belm,cdik,cdln,egkm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("belm,cdkj,cglm,dekn,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("bemj,cdkl,cdkn,eglm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("bemj,cdkl,cgkl,demn,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 2
    )

    r2 -= (
        np.einsum("abim,cdkl,cdln,egkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("ablm,cdkj,cdmn,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("abmn,cdkl,cekl,dgmn,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("ackl,bdim,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("ackl,bdmn,cgmn,dekl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("admj,bckl,cdmn,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("aeij,cdkl,cgkl,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("aeim,cdkl,cdmn,egkl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("aekl,cdij,cgkl,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("aelj,cdik,cdmn,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("aelm,cdik,cdkn,eglm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("beil,cdkj,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("belm,cdkj,cdkn,eglm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("bemj,cdkl,cdmn,egkl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdik,eglj,cdkm,egln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdik,eglj,cdln,egkm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("cdkl,egij,cdkm,egln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 -= (
        np.einsum("ablm,cdkj,cdkn,eglm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 -= (
        np.einsum("abmj,cdkl,cdmn,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 -= (
        np.einsum("aeij,cdkl,cdmn,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 -= (
        np.einsum("aekl,cdij,cdmn,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += (
        np.einsum("ablm,cdik,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("abmj,cdkl,cdln,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("ackl,bdmj,cdmn,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("ackl,bdmn,cekl,dgmn,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("adim,bckl,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("aeil,cdkj,cdmn,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("aelm,cdkj,cdkn,eglm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("aemj,cdkl,cdmn,egkl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("beij,cdkl,cgkl,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("beim,cdkl,cdmn,egkl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("bekl,cdij,cgkl,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("belj,cdik,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("belm,cdik,cdkn,eglm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 4
    )

    r2 += (
        np.einsum("abim,cdkl,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += (
        np.einsum("ablm,cdik,cdkn,eglm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += (
        np.einsum("beij,cdkl,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += (
        np.einsum("bekl,cdij,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 8
    )

    r2 += np.einsum(
        "cm,en,aelj,cdik,dgkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,en,beil,cdkj,dgkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,gk,aeil,cdkj,deln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,gk,beij,cdkl,deln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,gk,bekl,cdij,deln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,gk,belj,cdik,deln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,gl,aelj,cdik,dekn,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cm,gl,beil,cdkj,dekn,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cn,ek,abim,cdkl,dglm,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cn,ek,ackl,bdmj,dglm,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cn,ek,adim,bckl,dglm,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cn,el,ablm,cdkj,dgkm,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "dn,ek,ackl,bdim,cglm,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "dn,ek,admj,bckl,cglm,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ek,gm,ackl,bdim,cdln,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ek,gm,admj,bckl,cdln,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += (
        np.einsum(
            "cm,dn,aeil,cdkj,egkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,dn,belj,cdik,egkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,en,aeij,cdkl,dgkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cm,en,aekl,cdij,dgkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cn,ek,ablm,cdik,dglm,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cn,em,abim,cdkl,dgkl,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cn,em,ackl,bdmj,dgkl,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "cn,em,adim,bckl,dgkl,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "dn,em,ackl,bdim,cgkl,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "dn,em,admj,bckl,cgkl,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "ek,gl,ablm,cdik,cdmn,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "ek,gl,ackl,bdmj,cdmn,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "ek,gl,adim,bckl,cdmn,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "ek,gm,abmj,cdkl,cdln,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "em,gk,aeil,cdkj,cdln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "em,gk,beij,cdkl,cdln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "em,gk,bekl,cdij,cdln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "em,gk,belj,cdik,cdln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "em,gl,aelj,cdik,cdkn,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 += (
        np.einsum(
            "em,gl,beil,cdkj,cdkn,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= np.einsum(
        "cm,en,aeil,cdkj,dgkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,en,belj,cdik,dgkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,gk,aeij,cdkl,deln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,gk,aekl,cdij,deln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,gk,aelj,cdik,deln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,gk,beil,cdkj,deln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,gl,aeil,cdkj,dekn,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cm,gl,belj,cdik,dekn,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cn,ek,abmj,cdkl,dglm,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cn,ek,ackl,bdim,dglm,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cn,ek,admj,bckl,dglm,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "cn,el,ablm,cdik,dgkm,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "dn,ek,ackl,bdmj,cglm,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "dn,ek,adim,bckl,cglm,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "ek,gm,ackl,bdmj,cdln,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "ek,gm,adim,bckl,cdln,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= (
        np.einsum(
            "cm,dn,aelj,cdik,egkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cm,dn,beil,cdkj,egkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cm,en,beij,cdkl,dgkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cm,en,bekl,cdij,dgkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cn,ek,ablm,cdkj,dglm,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cn,em,abmj,cdkl,dgkl,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cn,em,ackl,bdim,dgkl,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cn,em,admj,bckl,dgkl,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "dn,em,ackl,bdmj,cgkl,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "dn,em,adim,bckl,cgkl,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ek,gl,ablm,cdkj,cdmn,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ek,gl,ackl,bdim,cdmn,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ek,gl,admj,bckl,cdmn,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "ek,gm,abim,cdkl,cdln,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "em,gk,aeij,cdkl,cdln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "em,gk,aekl,cdij,cdln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "em,gk,aelj,cdik,cdln,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "em,gk,beil,cdkj,cdln,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "em,gl,aeil,cdkj,cdkn,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "em,gl,belj,cdik,cdkn,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 2
    )

    r2 -= (
        np.einsum(
            "cm,dn,aeij,cdkl,egkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 -= (
        np.einsum(
            "cm,dn,aekl,cdij,egkl,mnbg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 -= (
        np.einsum(
            "ek,gl,abmj,cdkl,cdmn,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 -= (
        np.einsum(
            "el,gm,ablm,cdkj,cdkn,ineg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "cm,dn,beij,cdkl,egkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "cm,dn,bekl,cdij,egkl,mnag->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "ek,gl,abim,cdkl,cdmn,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += (
        np.einsum(
            "el,gm,ablm,cdik,cdkn,njeg->abij", t1, t1, l2, l2, t2, u[o, o, v, v], optimize=True
        )
        / 4
    )

    r2 += np.einsum("ak,bl,ijkl->abij", l1, l1, u[o, o, o, o], optimize=True)

    r2 += np.einsum("cm,aeil,cdkj,dgkl,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cm,aelj,cdik,egkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cm,beil,cdkj,egkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cm,belj,cdik,dgkl,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ek,abim,cdkl,cglm,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ek,ackl,bdim,dglm,cjeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ek,ackl,bdmj,cglm,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ek,adim,bckl,cglm,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ek,admj,bckl,dglm,iceg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("el,ablm,cdkj,cgkm,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("em,aeil,cdkj,cgkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("em,belj,cdik,cgkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("gk,aeij,cdkl,celm,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("gk,aekl,cdij,celm,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("gk,aelj,cdik,celm,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("gk,beil,cdkj,celm,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("gl,aeil,cdkj,cekm,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("gl,belj,cdik,cekm,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 += (
        np.einsum("cm,aeij,cdkl,egkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,aekl,cdij,egkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,beij,cdkl,dgkl,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("cm,bekl,cdij,dgkl,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("ek,ablm,cdik,cglm,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,abim,cdkl,cgkl,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,ackl,bdim,dgkl,cjeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,ackl,bdmj,cgkl,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,adim,bckl,cgkl,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,admj,bckl,dgkl,iceg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,beij,cdkl,cgkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("em,bekl,cdij,cgkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("gk,aeil,cdkj,cdlm,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("gk,beij,cdkl,cdlm,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("gk,bekl,cdij,cdlm,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("gk,belj,cdik,cdlm,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("gl,aelj,cdik,cdkm,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += (
        np.einsum("gl,beil,cdkj,cdkm,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= np.einsum("cm,aeil,cdkj,egkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cm,aelj,cdik,dgkl,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cm,beil,cdkj,dgkl,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cm,belj,cdik,egkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ek,abmj,cdkl,cglm,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ek,ackl,bdim,cglm,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ek,ackl,bdmj,dglm,iceg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ek,adim,bckl,dglm,cjeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ek,admj,bckl,cglm,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("el,ablm,cdik,cgkm,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("em,aelj,cdik,cgkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("em,beil,cdkj,cgkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("gk,aeil,cdkj,celm,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("gk,beij,cdkl,celm,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("gk,bekl,cdij,celm,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("gk,belj,cdik,celm,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("gl,aelj,cdik,cekm,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("gl,beil,cdkj,cekm,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= (
        np.einsum("cm,aeij,cdkl,dgkl,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,aekl,cdij,dgkl,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,beij,cdkl,egkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("cm,bekl,cdij,egkl,mdag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("ek,ablm,cdkj,cglm,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,abmj,cdkl,cgkl,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,ackl,bdim,cgkl,djeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,ackl,bdmj,dgkl,iceg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,adim,bckl,dgkl,cjeg->abij", t1, l2, l2, t2, u[v, o, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,admj,bckl,cgkl,ideg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,aeij,cdkl,cgkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("em,aekl,cdij,cgkl,mdbg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("gk,aeij,cdkl,cdlm,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("gk,aekl,cdij,cdlm,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("gk,aelj,cdik,cdlm,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("gk,beil,cdkj,cdlm,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("gl,aeil,cdkj,cdkm,mebg->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 -= (
        np.einsum("gl,belj,cdik,cdkm,meag->abij", t1, l2, l2, t2, u[o, v, v, v], optimize=True) / 2
    )

    r2 += np.einsum("ak,cdij,cekl,ldbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ak,cdil,cekl,djbe->abij", l1, l2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("bi,cdkj,cekl,ldae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bj,cdik,cekl,ldae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bk,cdlj,cekl,idae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ci,adkj,cekl,ldbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ci,bdkj,dekl,lcae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ci,dekj,cdkl,leab->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cj,adik,cekl,ldbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cj,bdik,dekl,lcae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cj,deik,cdkl,leab->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,adij,dekl,lcbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,adil,dekl,cjbe->abij", l1, l2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,adlj,cekl,idbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,bdij,cekl,ldae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,bdil,cekl,djae->abij", l1, l2, t2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ck,bdlj,dekl,icae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ak,bclj,dekl,icde->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("bi,cdkl,cekl,djae->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("bj,cdkl,cekl,idae->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("bk,acil,dekl,cjde->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,adkl,cekl,djbe->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,bdkl,dekl,cjae->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dekj,dekl,lcab->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("cj,adkl,cekl,idbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("cj,bdkl,dekl,icae->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("cj,deik,dekl,lcab->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("ck,ablj,dekl,icde->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ai,cdkj,cekl,ldbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("aj,cdik,cekl,ldbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ak,cdlj,cekl,idbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("bk,cdij,cekl,ldae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("bk,cdil,cekl,djae->abij", l1, l2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ci,adkj,dekl,lcbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ci,bdkj,cekl,ldae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cj,adik,dekl,lcbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cj,bdik,cekl,ldae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,adij,cekl,ldbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,adil,cekl,djbe->abij", l1, l2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,adlj,dekl,icbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,bdij,dekl,lcae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,bdil,dekl,cjae->abij", l1, l2, t2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,bdlj,cekl,idae->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,deij,cdkl,leab->abij", l1, l2, t2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ai,cdkl,cekl,djbe->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("aj,cdkl,cekl,idbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,bcil,dekl,cjde->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bk,aclj,dekl,icde->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ci,adkl,dekl,cjbe->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ci,bdkl,cekl,djae->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("cj,adkl,dekl,icbe->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("cj,bdkl,cekl,idae->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,abil,dekl,cjde->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ck,deij,dekl,lcab->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("bi,ackl,dekl,cjde->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 4

    r2 -= np.einsum("bj,ackl,dekl,icde->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 4

    r2 += np.einsum("ai,bckl,dekl,cjde->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 4

    r2 += np.einsum("aj,bckl,dekl,icde->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 4

    r2 += np.einsum("ci,abkl,dekl,cjde->abij", l1, l2, t2, u[v, o, v, v], optimize=True) / 4

    r2 += np.einsum("cj,abkl,dekl,icde->abij", l1, l2, t2, u[o, v, v, v], optimize=True) / 4

    r2 += np.einsum("ai,cl,ek,cdkj,ldbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("aj,cl,ek,cdik,ldbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ak,dk,el,bclj,icde->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bk,cl,ek,cdij,ldae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bk,dk,el,acil,cjde->abij", l1, t1, t1, l2, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ci,cl,ek,bdkj,ldae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ci,dl,ek,adkj,lcbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cj,cl,ek,bdik,ldae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("cj,dl,ek,adik,lcbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,cl,ek,adij,ldbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,dk,el,ablj,icde->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ck,dl,ek,bdij,lcae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ai,dk,el,bckl,cjde->abij", l1, t1, t1, l2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("aj,dk,el,bckl,icde->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dk,el,abkl,cjde->abij", l1, t1, t1, l2, u[v, o, v, v], optimize=True) / 2

    r2 += np.einsum("cj,dk,el,abkl,icde->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True) / 2

    r2 -= np.einsum("ak,cl,ek,cdij,ldbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ak,dk,el,bcil,cjde->abij", l1, t1, t1, l2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("bi,cl,ek,cdkj,ldae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("bj,cl,ek,cdik,ldae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("bk,dk,el,aclj,icde->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ci,cl,ek,adkj,ldbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ci,dl,ek,bdkj,lcae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cj,cl,ek,adik,ldbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("cj,dl,ek,bdik,lcae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,cl,ek,bdij,ldae->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ck,dk,el,abil,cjde->abij", l1, t1, t1, l2, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("ck,dl,ek,adij,lcbe->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("bi,dk,el,ackl,cjde->abij", l1, t1, t1, l2, u[v, o, v, v], optimize=True) / 2

    r2 -= np.einsum("bj,dk,el,ackl,icde->abij", l1, t1, t1, l2, u[o, v, v, v], optimize=True) / 2

    r2 += np.einsum("ci,dj,cdab->abij", l1, l1, u[v, v, v, v], optimize=True)

    r2 += np.einsum("ci,adkj,cdbk->abij", l1, l2, u[v, v, v, o], optimize=True)

    r2 += np.einsum("cj,adik,cdbk->abij", l1, l2, u[v, v, v, o], optimize=True)

    r2 += np.einsum("ck,bdij,cdak->abij", l1, l2, u[v, v, v, o], optimize=True)

    r2 += np.einsum("ak,cdij,cdbk->abij", l1, l2, u[v, v, v, o], optimize=True) / 2

    r2 += np.einsum("bi,cdkj,cdak->abij", l1, l2, u[v, v, v, o], optimize=True) / 2

    r2 += np.einsum("bj,cdik,cdak->abij", l1, l2, u[v, v, v, o], optimize=True) / 2

    r2 -= np.einsum("ci,bdkj,cdak->abij", l1, l2, u[v, v, v, o], optimize=True)

    r2 -= np.einsum("cj,bdik,cdak->abij", l1, l2, u[v, v, v, o], optimize=True)

    r2 -= np.einsum("ck,adij,cdbk->abij", l1, l2, u[v, v, v, o], optimize=True)

    r2 -= np.einsum("ai,cdkj,cdbk->abij", l1, l2, u[v, v, v, o], optimize=True) / 2

    r2 -= np.einsum("aj,cdik,cdbk->abij", l1, l2, u[v, v, v, o], optimize=True) / 2

    r2 -= np.einsum("bk,cdij,cdak->abij", l1, l2, u[v, v, v, o], optimize=True) / 2

    r2 += np.einsum("aj,ci,dk,kcbd->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ak,ci,dk,cjbd->abij", l1, l1, t1, u[v, o, v, v], optimize=True)

    r2 += np.einsum("ak,cj,dk,icbd->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bi,cj,dk,kcad->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 += np.einsum("bi,ck,dk,cjad->abij", l1, l1, t1, u[v, o, v, v], optimize=True)

    r2 += np.einsum("bj,ck,dk,icad->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 += np.einsum("ci,dj,dk,kcab->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ai,cj,dk,kcbd->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ai,ck,dk,cjbd->abij", l1, l1, t1, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("aj,ck,dk,icbd->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("bj,ci,dk,kcad->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("bk,ci,dk,cjad->abij", l1, l1, t1, u[v, o, v, v], optimize=True)

    r2 -= np.einsum("bk,cj,dk,icad->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 -= np.einsum("ci,dj,ck,kdab->abij", l1, l1, t1, u[o, v, v, v], optimize=True)

    r2 += np.einsum("aj,bk,ik->abij", l1, l1, f[o, o], optimize=True)

    r2 += np.einsum("ak,bi,jk->abij", l1, l1, f[o, o], optimize=True)

    r2 -= np.einsum("ai,bk,jk->abij", l1, l1, f[o, o], optimize=True)

    r2 -= np.einsum("ak,bj,ik->abij", l1, l1, f[o, o], optimize=True)

    r2 += np.einsum("ai,cj,cb->abij", l1, l1, f[v, v], optimize=True)

    r2 += np.einsum("bj,ci,ca->abij", l1, l1, f[v, v], optimize=True)

    r2 -= np.einsum("aj,ci,cb->abij", l1, l1, f[v, v], optimize=True)

    r2 -= np.einsum("bi,cj,ca->abij", l1, l1, f[v, v], optimize=True)

    return r2
