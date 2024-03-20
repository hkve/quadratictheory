import numpy as np
from clusterfock.cc.rhs.t_inter_RCCD import amplitudes_intermediates_rccd


def t_qccd_restricted(t2, l2, u, f, o, v):
    r2 = amplitudes_intermediates_rccd(t2, u, f, v, o)

    r2 += t_addition_qccd_restricted(t2, l2, u, f, o, v)

    return r2


def t_addition_qccd_restricted(t2, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    r2 = np.zeros((M, M, N, N), dtype=u.dtype)

    r2 += np.einsum("cdkl,abim,cekj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,abim,cekl,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,abim,eckj,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,abkm,ceij,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,abkm,ceil,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,abmj,ceik,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,abmj,cekl,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,abmj,ecik,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,abmk,ecij,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,abmk,eclj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,acim,edkj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,ackm,beji,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,ackm,bejl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,ackm,delj,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,ackm,edil,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,acmi,bejk,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,acmj,beki,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,acmj,bekl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,acmk,beji,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,acmk,bejl,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,acmk,beli,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,acmk,belj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,acmk,deij,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,acmk,deil,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,acmk,edij,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,acmk,edlj,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,admi,cekl,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,admj,cekl,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,aeij,bckm,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,aeij,bcmk,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdkl,aeik,bclm,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,aeik,bcmj,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,aeik,cdmj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,aekj,bcmi,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdkl,aekj,cdim,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,aekl,bcmi,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,aekl,cdim,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bcjm,deik,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bckm,delj,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bckm,edil,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,bcmk,deij,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bcmk,deil,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,bcmk,edij,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,bcmk,edlj,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bdmi,cekl,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdkl,bdmj,cekl,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdkl,bejk,cdim,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,abim,eckj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,abkm,celj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,abkm,ecil,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,abmj,ceik,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,abmk,celj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,abmk,ecil,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdlk,acim,bekj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,acim,bekl,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdlk,acim,edkj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,ackm,beji,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,ackm,bejl,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,ackm,deij,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,ackm,deil,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,acmi,bejk,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,acmi,dekj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,acmj,edik,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,aeij,bckm,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,aeik,bclm,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,aeik,bcmj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,aeik,bcml,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("cdlk,aeik,cdmj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,aeki,bcjm,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 += np.einsum("cdlk,aeki,bcml,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,aekj,bcml,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,aekl,bcjm,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,bcjm,deik,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,bckm,edij,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,bckm,edlj,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,bcmi,dekj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,bcmj,edik,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,bejk,cdim,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("cdlk,beki,cdmj,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 += np.einsum("cdlk,bekl,cdmj,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,abim,cekj,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,abim,cekl,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,abmj,cekl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,abmj,ecik,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,acim,bejk,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,acmk,beji,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,acmk,bejl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,acmk,delj,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,acmk,edil,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,adim,cekl,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,aeij,bcmk,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,aeij,cdkm,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,aeik,bcjm,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,aeil,cdkm,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,bcmk,delj,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,bcmk,edil,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,bdjm,cekl,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdkl,beji,cdkm,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdkl,bejl,cdkm,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdlk,acim,bejk,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdlk,acim,dekj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdlk,aeik,bcjm,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True)

    r2 -= 2 * np.einsum("cdlk,aeik,bcml,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 2 * np.einsum("cdlk,bcjm,edik,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True)

    r2 -= 11 * np.einsum("cdkl,ackm,beli,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 11 * np.einsum("cdkl,ackm,belj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 18

    r2 -= 11 * np.einsum("cdkl,ackm,edij,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 18

    r2 -= 11 * np.einsum("cdkl,ackm,edlj,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 18

    r2 -= 11 * np.einsum("cdkl,bckm,deij,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 18

    r2 -= 11 * np.einsum("cdkl,bckm,deil,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 18

    r2 -= 11 * np.einsum("cdlk,aeki,bclm,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 18

    r2 -= 11 * np.einsum("cdlk,aekj,bclm,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 18

    r2 -= 7 * np.einsum("cdkl,aeki,cdmj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 18

    r2 -= 7 * np.einsum("cdkl,aekj,cdim,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 18

    r2 -= 7 * np.einsum("cdkl,aekl,cdim,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 18

    r2 -= 7 * np.einsum("cdkl,aekl,cdmj,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 18

    r2 -= 7 * np.einsum("cdlk,beki,cdmj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 18

    r2 -= 7 * np.einsum("cdlk,bekj,cdim,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 18

    r2 -= 7 * np.einsum("cdlk,bekl,cdim,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 18

    r2 -= 7 * np.einsum("cdlk,bekl,cdmj,amei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 18

    r2 -= 4 * np.einsum("cdkl,aeki,bclm,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 4 * np.einsum("cdkl,aekj,bclm,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    r2 -= 4 * np.einsum("cdlk,ackm,beli,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 4 * np.einsum("cdlk,ackm,belj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 4 * np.einsum("cdlk,ackm,deij,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 4 * np.einsum("cdlk,ackm,deil,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 4 * np.einsum("cdlk,bckm,edij,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 4 * np.einsum("cdlk,bckm,edlj,amei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,abkm,ceij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,abkm,ceil,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,abmk,ecij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,abmk,eclj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,acmi,edkj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,acmj,beki,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,acmj,bekl,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,acmj,deik,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,aekj,bcmi,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,aekl,bcmi,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,bcmi,edkj,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,bcmj,deik,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,abkm,ecij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,abkm,eclj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,abmk,ceij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,abmk,ceil,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,acmi,bekj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,acmi,bekl,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,acmi,edkj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,acmj,deik,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,aeki,bcmj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,aekl,bcmj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,bcmi,edkj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("cdlk,bcmj,deik,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("cdkl,acjm,beki,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,acjm,bekl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,ackm,beli,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,ackm,belj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,ackm,deij,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,ackm,deil,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,adjm,cekl,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,aeki,bclm,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,aekj,bcim,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,aekj,bclm,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,aekl,bcim,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,bckm,edij,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,bckm,edlj,amei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdkl,bdim,cekl,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,acim,bekj,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,acim,bekl,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,acjm,edik,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,ackm,beli,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,ackm,belj,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,ackm,edij,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,ackm,edlj,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,aeki,bcjm,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,aeki,bclm,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,aekj,bclm,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,aekl,bcjm,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,bcim,dekj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,bckm,deij,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= 2 * np.einsum("cdlk,bckm,deil,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= np.einsum("cdkl,abkm,ecij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,abkm,eclj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,abmk,ceij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,abmk,ceil,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= np.einsum("cdkl,acim,dekj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= np.einsum("cdkl,acmi,bekj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,acmi,bekl,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= np.einsum("cdkl,acmi,edkj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= np.einsum("cdkl,acmj,deik,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,adim,celk,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,aeki,bcmj,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,aekl,bcmj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,bcjm,edik,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= np.einsum("cdkl,bcmi,edkj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,bcmj,deik,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= np.einsum("cdkl,bdjm,celk,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,abkm,ceij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,abkm,ceil,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= np.einsum("cdlk,abmk,ecij,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,abmk,eclj,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,acmi,edkj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,acmj,beki,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,acmj,bekl,dmie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,acmj,deik,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= np.einsum("cdlk,aekj,bcmi,mdle->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= np.einsum("cdlk,aekl,bcmi,mdej->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= np.einsum("cdlk,bcmi,edkj,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= np.einsum("cdlk,bcmj,deik,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= np.einsum("cdkl,aeij,dckm,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 6

    r2 -= np.einsum("cdkl,aeil,dckm,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 6

    r2 -= np.einsum("cdkl,aeli,dckm,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 6

    r2 -= np.einsum("cdkl,aelj,dckm,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 6

    r2 -= np.einsum("cdkl,beji,dckm,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 6

    r2 -= np.einsum("cdkl,bejl,dckm,amei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 6

    r2 -= np.einsum("cdkl,beli,dckm,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 6

    r2 -= np.einsum("cdkl,belj,dckm,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 6

    r2 -= np.einsum("cdkl,beki,cdmj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= np.einsum("cdkl,bekj,cdim,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= np.einsum("cdkl,bekl,cdim,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= np.einsum("cdkl,bekl,cdmj,amei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= np.einsum("cdlk,acjm,beki,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= np.einsum("cdlk,acjm,bekl,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    r2 -= np.einsum("cdlk,aeki,cdmj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= np.einsum("cdlk,aekj,bcim,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= np.einsum("cdlk,aekj,cdim,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= np.einsum("cdlk,aekl,bcim,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= np.einsum("cdlk,aekl,cdim,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 -= np.einsum("cdlk,aekl,cdmj,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += np.einsum("cdkl,acjm,deik,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += np.einsum("cdkl,bcim,edkj,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += np.einsum("cdlk,acjm,deik,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 += np.einsum("cdlk,bcim,edkj,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 += 2 * np.einsum("cdkl,aeij,cdkm,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,aeil,cdkm,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,aeli,cdkm,bmje->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,aelj,cdkm,bmei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,beji,cdkm,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,bejl,cdkm,amei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,beli,cdkm,amej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,belj,cdkm,amie->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,acim,edkj,bmel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += 2 * np.einsum("cdkl,bcjm,deik,amel->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += 2 * np.einsum("cdlk,acim,edkj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 += 2 * np.einsum("cdlk,bcjm,deik,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,acim,bekj,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,acim,bekl,dmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,adim,cekl,bmej->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,aeki,bcjm,mdel->abij", l2, t2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,aekl,bcjm,mdie->abij", l2, t2, t2, u[o, v, o, v], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,bdjm,cekl,amei->abij", l2, t2, t2, u[v, o, v, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdlk,acim,dekj,bmle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 += 5 * np.einsum("cdlk,bcjm,edik,amle->abij", l2, t2, t2, u[v, o, o, v], optimize=True) / 9

    r2 -= np.einsum("cdkl,abim,cdnj,mnlk->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,abkm,cdin,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,abln,cdkm,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,abmj,cdin,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,abnl,cdkm,nmij->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,acim,bdkn,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,acim,bdnj,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,acmi,bdjn,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,acmj,bdkn,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,acmk,bdni,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdkl,acmk,bdnj,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdlk,abmk,cdnj,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdlk,ackm,bdjn,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdlk,ackm,bdni,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdlk,acmi,bdnk,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= np.einsum("cdlk,acmj,bdnk,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("cdkl,abin,cdkm,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("cdkl,abnj,cdkm,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("cdkl,acim,bdjn,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("cdkl,acmk,bdjn,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("cdkl,acmk,bdnl,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("cdlk,acim,bdnk,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True)

    r2 -= 7 * np.einsum("cdkl,abin,cdkm,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 -= 7 * np.einsum("cdkl,abnj,cdkm,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 -= 7 * np.einsum("cdkl,ackm,bdjn,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 -= 7 * np.einsum("cdlk,acim,bdkn,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 -= 5 * np.einsum("cdkl,ackm,bdnl,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 6

    r2 -= 5 * np.einsum("cdkl,acmk,bdjn,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 6

    r2 -= 5 * np.einsum("cdkl,acmk,bdln,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 6

    r2 -= 5 * np.einsum("cdlk,acim,bdnk,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 6

    r2 -= np.einsum("cdkl,abim,cdnj,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 -= np.einsum("cdkl,abmj,cdin,mnlk->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 -= np.einsum("cdkl,abjm,cdin,mnlk->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 -= np.einsum("cdkl,abmi,cdnj,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += np.einsum("cdkl,acmi,bdnj,mnlk->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += np.einsum("cdkl,acmj,bdni,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += np.einsum("cdkl,abin,dckm,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 6

    r2 += np.einsum("cdkl,abnj,dckm,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 6

    r2 += np.einsum("cdkl,abjn,cdkm,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += np.einsum("cdkl,abni,cdkm,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += np.einsum("cdkl,acim,bdkn,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += np.einsum("cdkl,ackm,bdin,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += np.einsum("cdlk,acjm,bdkn,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += np.einsum("cdlk,ackm,bdjn,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += np.einsum("cdkl,acim,bdnk,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 12

    r2 += np.einsum("cdlk,ackm,bdnl,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 12

    r2 += np.einsum("cdlk,acmk,bdjn,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 12

    r2 += np.einsum("cdlk,acmk,bdln,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 12

    r2 += np.einsum("cdkl,acjm,bdkn,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += np.einsum("cdlk,ackm,bdin,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += 2 * np.einsum("cdkl,acmi,bdnj,mnkl->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,acmj,bdni,mnlk->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,ackm,bdln,nmij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 2 * np.einsum("cdkl,ackm,bdni,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 2 * np.einsum("cdkl,acmi,bdkn,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 2 * np.einsum("cdlk,ackm,bdln,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 2 * np.einsum("cdlk,ackm,bdnj,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 2 * np.einsum("cdlk,acmj,bdkn,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 4 * np.einsum("cdkl,ackm,bdln,mnij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,abkm,cdin,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdlk,abmk,cdnj,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,abmk,cdin,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += 5 * np.einsum("cdkl,abmk,cdnj,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += 5 * np.einsum("cdlk,abkm,cdin,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += 5 * np.einsum("cdlk,abkm,cdnj,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += 11 * np.einsum("cdlk,ackm,bdln,nmij->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 18

    r2 += 19 * np.einsum("cdkl,ackm,bdnj,mnil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 36

    r2 += 19 * np.einsum("cdkl,acmj,bdkn,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 36

    r2 += 19 * np.einsum("cdlk,ackm,bdni,nmlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 36

    r2 += 19 * np.einsum("cdlk,acmi,bdkn,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 36

    r2 += 23 * np.einsum("cdkl,abkm,cdnj,nmil->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 36

    r2 += 23 * np.einsum("cdlk,abmk,cdin,mnlj->abij", l2, t2, t2, u[o, o, o, o], optimize=True) / 36

    r2 -= np.einsum("cdkl,aeij,bfkl,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,fclj,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekj,cfil,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekl,bfji,dcef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekl,cfij,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,bejk,cfil,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,beki,fclj,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,bekl,fcij,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,ceik,dflj,abef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,cekl,dfij,abef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,cekl,fdij,abfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdkl,ecik,fdlj,abef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeik,fclj,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aekj,fcil,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdlk,bejk,cfil,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= np.einsum("cdlk,beki,cflj,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += 2 * np.einsum("cdkl,aeik,bfjl,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += 2 * np.einsum("cdkl,afij,cekl,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += 2 * np.einsum("cdkl,bfji,cekl,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += 2 * np.einsum("cdkl,dflj,ecik,abef->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += 2 * np.einsum("cdlk,aeik,cflj,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 += 2 * np.einsum("cdlk,bejk,fcil,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True)

    r2 -= 7 * np.einsum("cdkl,aeik,bflj,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 7 * np.einsum("cdkl,aeki,bfjl,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 7 * np.einsum("cdlk,aeik,cflj,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 7 * np.einsum("cdlk,bejk,fcil,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 5 * np.einsum("cdkl,afij,cekl,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 6

    r2 -= 5 * np.einsum("cdkl,bfji,cekl,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 6

    r2 -= 5 * np.einsum("cdlk,aeki,cflj,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 6

    r2 -= 5 * np.einsum("cdlk,bekj,fcil,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 6

    r2 -= np.einsum("cdlk,aeik,fclj,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= np.einsum("cdlk,bejk,cfil,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= np.einsum("cdlk,aejk,cfil,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 -= np.einsum("cdlk,beik,fclj,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += np.einsum("cdkl,ceik,fdlj,abfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 3

    r2 += np.einsum("cdlk,ceik,fdlj,abef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 3

    r2 += np.einsum("cdkl,afij,celk,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 6

    r2 += np.einsum("cdkl,bfji,celk,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 6

    r2 += np.einsum("cdkl,aeik,bflj,dcef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += np.einsum("cdkl,aejk,bfli,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += np.einsum("cdkl,aeki,bfjl,dcef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += np.einsum("cdkl,aekj,bfil,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += np.einsum("cdlk,aejk,fcil,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += np.einsum("cdlk,beik,cflj,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += np.einsum("cdkl,aeki,cflj,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 12

    r2 += np.einsum("cdkl,afij,celk,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 12

    r2 += np.einsum("cdkl,bekj,fcil,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 12

    r2 += np.einsum("cdkl,bfji,celk,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 12

    r2 += np.einsum("cdkl,aejk,bfli,dcef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += np.einsum("cdkl,aekj,bfil,dcef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 2 * np.einsum("cdkl,ceik,fdlj,abef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 3

    r2 += 2 * np.einsum("cdlk,ceik,fdlj,abfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 3

    r2 += 2 * np.einsum("cdkl,aeki,bflj,dcef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("cdkl,aeki,fclj,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("cdkl,aekj,bfli,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("cdkl,bekj,cfil,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("cdlk,aekj,cfil,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("cdlk,beki,fclj,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 4 * np.einsum("cdkl,aeki,bflj,cdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,aekl,cfij,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,bekl,fcij,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("cdkl,aekl,fcij,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 5 * np.einsum("cdkl,bekl,cfij,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 5 * np.einsum("cdlk,aekl,cfij,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 5 * np.einsum("cdlk,bekl,fcij,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 11 * np.einsum("cdkl,aekj,bfli,dcef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 19 * np.einsum("cdkl,aekj,cfil,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 36

    r2 += 19 * np.einsum("cdkl,beki,fclj,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 36

    r2 += 19 * np.einsum("cdlk,aeki,fclj,bdfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 36

    r2 += 19 * np.einsum("cdlk,bekj,cfil,adfe->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 36

    r2 += 23 * np.einsum("cdlk,aekl,fcij,bdef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 36

    r2 += 23 * np.einsum("cdlk,bekl,cfij,adef->abij", l2, t2, t2, u[v, v, v, v], optimize=True) / 36

    r2 -= np.einsum("cdkl,abim,cdnj,eflk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abim,cekl,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abim,cekl,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abin,cekm,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abin,cemk,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abin,cemk,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abkm,cdin,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abkm,ceil,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abkm,cein,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abkm,dflj,ecin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abln,cdkm,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abln,cekm,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abln,cemk,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abmj,cdin,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abmj,cekl,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abmj,cekl,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abmk,dflj,ecin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abmn,cekl,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abmn,cekl,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abnj,cekm,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abnj,cemk,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abnj,cemk,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abnl,cdkm,efij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abnl,cekm,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,abnl,cemk,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acim,bdkn,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acim,bdnj,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acim,benk,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,ackm,bejl,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,ackm,bejn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,ackm,beni,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,ackm,benj,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmi,bdjn,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmi,benk,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmj,bdkn,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmj,bekl,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmj,bekn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmj,benk,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,bdni,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,bdnj,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,bejl,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,bejl,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,bejn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,beli,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,beli,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,belj,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,beln,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,beni,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,benj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,benl,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmk,benl,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmn,bejk,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmn,beki,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,acmn,bekj,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adln,bfji,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adln,bfji,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,admi,bfjn,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,admi,bfnj,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,admj,bfni,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,admn,bfji,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adni,bfjl,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adni,bfjl,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adnj,bfli,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adnj,bfli,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,adnl,bfji,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeij,bfkl,cdmn,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bclm,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bclm,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bcmn,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bflj,cdmn,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bflm,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeik,bfml,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeim,bckn,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeim,bcnk,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeki,bcmn,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aeki,bfjl,cdmn,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekj,bcmn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekj,bfml,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekl,bcmi,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekl,bfji,cdnm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekl,bfjm,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekl,bfmj,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekm,bcni,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aekm,bfjl,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemi,bckn,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemi,bcnk,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemj,bckn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemj,bcnk,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemk,bcjn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemk,bcln,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemk,bcni,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemk,bcnj,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemk,bfjl,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aemk,bfli,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afij,bdln,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afij,bdln,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afij,bdmn,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afij,bdnl,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afil,bdnj,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afil,bdnj,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afim,bdnj,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aflj,bdni,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,aflj,bdni,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afmi,bdnj,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdkl,afmj,bdni,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,abin,cemk,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,abkm,dfnj,ecil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,abmk,cdnj,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,abmk,dfnj,ecil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,abmk,ecil,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,abmk,ecin,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,abnj,cemk,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,acim,bekl,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,acim,bekl,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,acim,bekn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,acim,benk,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,ackm,bdjn,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,ackm,bdni,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,ackm,bejl,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,ackm,bejl,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,ackm,bejn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,ackm,benj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,ackm,benl,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,acmi,bdnk,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,acmj,bdnk,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,acmn,bejk,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,adin,bflj,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,adln,bfji,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,adni,bfjl,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeik,bclm,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeik,bcml,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeik,bcml,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeik,bcmn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeik,bfml,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeim,bckn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeim,bfkl,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aeki,bcml,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aekj,bcml,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aekj,bcml,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aekl,bcjm,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aekl,bcjm,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aekm,bcjn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aekm,bcnl,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aemi,bckn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aemi,bfkl,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aemk,bcjn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aemk,bcnl,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aemk,bcnl,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,aemk,bfjl,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,afij,bdln,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,afil,bdnj,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("cdlk,afli,bdjn,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)

    r2 -= 4 * np.einsum(
        "cdkl,abin,cekm,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,abnj,cekm,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,acmk,bejn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,adim,bfjn,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,adin,bfjl,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,adnl,bfji,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,aeil,bfjn,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,aeim,bcnk,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,aein,bfjl,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,afij,bdnl,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,afil,bdjn,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "cdkl,afim,bdjn,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abim,cekl,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abim,cekl,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abin,cdkm,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abin,cekm,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abin,cekm,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abin,cemk,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abkm,cein,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abln,cekm,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abmj,cekl,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abmj,cekl,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abmn,dflj,ecik,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abnj,cdkm,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abnj,cekm,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abnj,cekm,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abnj,cemk,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,abnl,cekm,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acim,bdjn,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acim,benk,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,ackm,bejn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmj,bekn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,bdjn,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,bdnl,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,bejl,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,bejl,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,bejn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,bejn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,beli,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,beln,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,beni,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmk,benj,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,acmn,bejk,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adim,bfjn,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adim,bfnj,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adin,bfjl,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adin,bfjl,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adln,bfji,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,admi,bfjn,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,admn,bfji,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adni,bfjl,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adnj,bfli,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adnl,bfji,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,adnl,bfji,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeij,bfnl,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeik,bclm,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeik,bcmn,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeik,bfjl,cdmn,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeik,bflm,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeil,bfjn,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeil,bfnj,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeim,bckn,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeim,bcnk,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeim,bcnk,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aein,bfjl,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aekl,bfjm,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aekm,bcni,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aekm,bfjl,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aemi,bcnk,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aemj,bcnk,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aemk,bcjn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aeni,bfjl,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aenl,bfji,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afij,bdln,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afij,bdmn,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afij,bdnl,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afij,bdnl,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afil,bdjn,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afil,bdjn,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afil,bdnj,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afim,bdjn,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afim,bdnj,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,aflj,bdni,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdkl,afmi,bdjn,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,abmk,ecil,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,acim,bdnk,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,acim,bekl,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,acim,bekn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,ackm,bejl,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,ackm,bejn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,adin,bfjl,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aeik,bcml,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aeik,bcml,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aeim,bckn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aeim,bfkl,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aekj,bcml,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aekl,bcjm,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aekm,bcjn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,aekm,bcnl,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "cdlk,afil,bdjn,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= (
        11
        * np.einsum("cdkl,ackm,beli,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        11
        * np.einsum("cdkl,ackm,beln,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        11
        * np.einsum("cdlk,aekj,bclm,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        11
        * np.einsum("cdlk,aekm,bcln,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("cdkl,aeki,bclm,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("cdkl,aekm,bcln,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("cdlk,ackm,belj,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("cdlk,ackm,beln,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,abin,cdkm,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,abmn,ceik,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,abmn,ecik,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,abnj,cdkm,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,ackm,bdjn,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,aeki,bflm,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,aekj,bflm,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,aekm,bfli,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdkl,aekm,bflj,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdlk,acim,bdkn,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdlk,acmn,beki,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        7
        * np.einsum("cdlk,aekj,bcmn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,ackm,bdnl,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdkl,acmk,bdjn,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdkl,acmk,bdln,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdkl,acmn,bejk,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdkl,aeik,bcmn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdlk,acim,bdnk,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdlk,acmn,bekl,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdlk,aekl,bcmn,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 -= (
        5
        * np.einsum("cdkl,acim,bekl,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,acim,bekn,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,acim,benk,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,adim,bfnj,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,adin,bflj,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,adin,bflj,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,aekl,bcjm,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,aekm,bcjn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,aemk,bcjn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,afli,bdjn,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,afli,bdjn,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("cdkl,afmi,bdjn,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,abkm,ceil,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdkl,abkm,cein,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdkl,acmj,bekl,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdkl,acmj,bekn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdkl,aekl,bcmi,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdkl,aekm,bcni,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdlk,abmk,ceil,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdlk,abmk,cein,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdlk,acmi,bekl,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdlk,acmi,bekn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdlk,aekl,bcmj,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdlk,aekm,bcnj,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("cdkl,acjm,bekn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,ackm,belj,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,ackm,beln,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,adjn,bfli,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,aekj,bclm,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,aekm,bcin,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,aekm,bcln,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdkl,aflj,bdin,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdlk,ackm,beli,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdlk,ackm,beln,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdlk,adin,bflj,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdlk,aeki,bclm,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdlk,aekm,bcln,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("cdlk,afli,bdjn,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdkl,abmk,ceil,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,abmk,cein,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,acmi,bekl,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,acmi,bekn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,adim,bfjn,celk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeij,bfln,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeij,bfnl,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeil,bfnj,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aein,bflj,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aekl,bcmj,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aekm,bcnj,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeli,bfjn,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeli,bfnj,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aelj,bfni,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeln,bfji,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeni,bfjl,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aeni,bflj,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aenj,bfli,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,aenl,bfji,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,afim,bdjn,celk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdlk,abkm,ceil,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdlk,abkm,cein,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdlk,acmj,bekl,dfin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdlk,acmj,bekn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdlk,aekl,bcmi,fdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdlk,aekm,bcni,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("cdkl,acim,bekn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdkl,acim,benk,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdkl,aekm,bcjn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdkl,aemk,bcjn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,acim,benk,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,adjn,bfli,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,aeki,bflm,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,aekj,bflm,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,aekm,bfli,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,aekm,bflj,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,aemk,bcjn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("cdlk,aflj,bdin,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdkl,aeij,bfln,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("cdkl,aein,bflj,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("cdkl,aeli,bfjn,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("cdkl,aeln,bfji,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("cdkl,abim,cdnj,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdkl,abmj,cdin,eflk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdkl,acjm,bekn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdkl,acjm,benk,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdkl,aekm,bcin,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdkl,aemk,bcin,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdlk,acjm,benk,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdlk,acmn,beki,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdlk,aekj,bcmn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdlk,aemk,bcin,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("cdkl,abjm,cdin,eflk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 -= (
        np.einsum("cdkl,abmi,cdnj,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 -= (
        np.einsum("cdlk,acmn,bekj,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 -= (
        np.einsum("cdlk,aeki,bcmn,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        np.einsum("cdkl,abln,cemk,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,abmk,ceil,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,abmk,ceil,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,abmk,cein,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,abmk,ecin,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,abnl,cemk,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,acmi,bdnj,eflk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,acmi,bekl,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,acmi,bekl,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,acmi,bekn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,acmi,benk,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,acmj,bdni,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,acmj,benk,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,adim,bfjn,celk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,adim,bfnj,celk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,adni,bflj,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,aeki,bflj,cdnm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,aekj,bfli,cdmn,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,aekl,bcmj,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,aekl,bcmj,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,aekm,bcnj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,aemk,bcni,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,aemk,bcnj,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,afim,bdjn,celk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,afli,bdnj,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,afmi,bdjn,celk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,abkm,ceil,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,abkm,ceil,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,abkm,cein,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,abkm,ecin,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,abln,cemk,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,abnl,cemk,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,acim,benk,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,acmi,benk,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,acmj,bekl,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,acmj,bekl,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,acmj,bekn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,acmj,benk,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,adnj,bfli,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,aekl,bcmi,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,aekl,bcmi,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,aekm,bcni,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,aemk,bcjn,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,aemk,bcni,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,aemk,bcnj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdlk,aflj,bdni,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("cdkl,abin,dckm,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,abnj,dckm,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,admn,bfji,celk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeij,bfln,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeij,bfnl,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeil,bfnj,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aein,bflj,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeli,bfjn,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeli,bfnj,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aelj,bfni,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeln,bfji,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeni,bfjl,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aeni,bflj,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aenj,bfli,dckm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,aenl,bfji,dckm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,afij,bdmn,celk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("cdkl,abjn,cdkm,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,abmn,ceik,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,abmn,ecik,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,abni,cdkm,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,acim,bdkn,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,ackm,bdin,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,aemi,bfkl,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,aemj,bfkl,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,abmn,ceik,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,abmn,ecik,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,acjm,bdkn,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,acjm,bekl,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,ackm,bdjn,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,acmn,bekj,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,adjn,bfli,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,adjn,bfli,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aeki,bcmn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aeki,bflm,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aeki,bfml,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aekj,bflm,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aekj,bfml,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aekl,bcim,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aekl,bfmi,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aekl,bfmj,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aekm,bfli,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aekm,bflj,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aemk,bfli,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aemk,bflj,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aflj,bdin,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdlk,aflj,bdin,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("cdkl,acim,bdnk,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdkl,acmn,bekl,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdkl,aekl,bcmn,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdlk,ackm,bdnl,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdlk,acmk,bdjn,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdlk,acmk,bdln,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdlk,acmn,bejk,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdlk,aeik,bcmn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 12
    )

    r2 += (
        np.einsum("cdkl,acjm,bdkn,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        np.einsum("cdlk,abmn,ceik,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        np.einsum("cdlk,abmn,ecik,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        np.einsum("cdlk,ackm,bdin,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        2
        * np.einsum("cdkl,abkm,ceil,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,abkm,ceil,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,abkm,cein,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,abkm,ecin,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,abln,cemk,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,abnl,cemk,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,acmi,bdnj,efkl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,acmi,benk,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,acmj,bdni,eflk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,acmj,bekl,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,acmj,bekl,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,acmj,bekn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,acmj,benk,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,adnj,bfli,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aeki,bflj,cdmn,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aekj,bfli,cdnm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aekl,bcmi,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aekl,bcmi,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aekm,bcni,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aemk,bcni,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aemk,bcnj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,aflj,bdni,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,abln,cemk,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,abmk,ceil,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,abmk,ceil,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,abmk,cein,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,abmk,ecin,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,abnl,cemk,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,acmi,bekl,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,acmi,bekl,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,acmi,bekn,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,acmi,benk,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,acmj,benk,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,adni,bflj,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,aekl,bcmj,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,aekl,bcmj,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,aekm,bcnj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,aemk,bcni,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,aemk,bcnj,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,afli,bdnj,cemk,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,abmn,ceik,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,acjm,bekl,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,acjm,bekn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,acjm,bekn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,acjm,benk,fdil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,bdln,efij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,bdni,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,beli,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,belj,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,belj,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,beln,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,benj,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,ackm,benl,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,acmi,bdkn,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,acmn,beki,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,adjm,bfni,cekl,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,adjn,bfli,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,adjn,bfli,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aeki,bclm,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aekj,bclm,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aekj,bclm,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aekj,bcmn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aekl,bcim,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aekm,bcin,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aekm,bcin,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aekm,bcln,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aemi,bckn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aemk,bcin,dflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aemk,bcln,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aflj,bdin,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,aflj,bdin,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdkl,afmj,bdin,cekl,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,abmn,ceik,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,acim,bekl,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,bdln,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,bdnj,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,beli,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,beli,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,belj,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,beln,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,beni,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,ackm,benl,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,acmj,bdkn,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,acmn,bekj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,adin,bflj,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,adin,bflj,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aeki,bclm,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aeki,bclm,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aeki,bcmn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aekj,bclm,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aekl,bcjm,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aekm,bcln,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aemj,bckn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,aemk,bcln,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,afli,bdjn,cekm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("cdlk,afli,bdjn,cemk,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,aeij,bfln,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdkl,aein,bflj,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdkl,aeli,bfjn,cdkm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdkl,aeln,bfji,cdkm,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdkl,abmn,ceik,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,acim,bekn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,ackm,bdln,efij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,aeki,bclm,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,aeki,bclm,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,aekj,bclm,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,aekm,bcjn,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,aekm,bcln,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdkl,aemk,bcln,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdlk,ackm,beli,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdlk,ackm,belj,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdlk,ackm,belj,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdlk,ackm,beln,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdlk,ackm,benj,dfil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdlk,ackm,benl,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("cdlk,aemi,bckn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("cdkl,abkm,cdin,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("cdlk,abmk,cdnj,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("cdlk,acmn,bekl,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("cdlk,aekl,bcmn,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("cdkl,abmk,cdin,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        5
        * np.einsum("cdkl,abmk,cdnj,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        5
        * np.einsum("cdkl,acmn,bekl,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        5
        * np.einsum("cdkl,aekl,bcmn,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        5
        * np.einsum("cdlk,abkm,cdin,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        5
        * np.einsum("cdlk,abkm,cdnj,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        5
        * np.einsum("cdlk,acmn,bekl,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        5
        * np.einsum("cdlk,aekl,bcmn,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aeki,bflm,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aeki,bfml,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aekj,bflm,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aekj,bfml,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aekl,bfmi,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aekl,bfmj,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aekm,bfli,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aekm,bflj,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aemk,bfli,cdnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdkl,aemk,bflj,cdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdlk,aemi,bfkl,cdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        7
        * np.einsum("cdlk,aemj,bfkl,cdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        10
        * np.einsum("cdkl,acim,bekn,dflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("cdkl,adin,bflj,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("cdkl,aekm,bcjn,fdil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("cdkl,afli,bdjn,cekm,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        11
        * np.einsum("cdkl,ackm,beli,dfnj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdkl,ackm,beli,fdnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdkl,ackm,belj,fdin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdkl,ackm,beln,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdkl,ackm,beni,fdlj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdkl,ackm,benl,fdij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdkl,aemj,bckn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdlk,abmn,ceik,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdlk,ackm,bdln,efij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdlk,aeki,bclm,dfnj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdlk,aekj,bclm,dfin,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdlk,aekj,bclm,fdin,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdlk,aekm,bcln,dfij,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        11
        * np.einsum("cdlk,aemk,bcln,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("cdkl,ackm,bdnj,efil,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        19
        * np.einsum("cdkl,acmj,bdkn,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        19
        * np.einsum("cdkl,acmn,bekj,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        19
        * np.einsum("cdkl,aeki,bcmn,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        19
        * np.einsum("cdlk,ackm,bdni,eflj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        19
        * np.einsum("cdlk,acmi,bdkn,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        19
        * np.einsum("cdlk,acmn,beki,fdlj,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        19
        * np.einsum("cdlk,aekj,bcmn,dfil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        23
        * np.einsum("cdkl,abkm,cdnj,efil,mnfe->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        23
        * np.einsum("cdkl,acmn,bekl,dfij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        23
        * np.einsum("cdkl,aekl,bcmn,fdij,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    r2 += (
        23
        * np.einsum("cdlk,abmk,cdin,eflj,mnef->abij", l2, t2, t2, t2, u[o, o, v, v], optimize=True)
        / 36
    )

    return r2
