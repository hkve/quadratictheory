import numpy as np


def amplitudes_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r2 = zeros((M, M, N, N), dtype=u.dtype)

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

    r2 += np.einsum("ackj,bkic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("bcik,akcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("acik,bkcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bckj,akic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("abkl,klij->abij", t2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("bc,acij->abij", f[v, v], t2, optimize=True)

    r2 -= np.einsum("ac,bcij->abij", f[v, v], t2, optimize=True)

    r2 += np.einsum("cdij,abcd->abij", t2, u[v, v, v, v], optimize=True) / 2

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

    r2 -= np.einsum("ki,abkj->abij", f[o, o], t2, optimize=True)

    r2 -= np.einsum("kj,abik->abij", f[o, o], t2, optimize=True)

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

    r2 += np.einsum("abij->abij", u[v, v, o, o], optimize=True)

    r2 += np.einsum("ackj,bdil,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ablj,cdik,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("adij,bckl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,bdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abil,cdkj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ackl,bdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("abkl,cdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 4

    return r2
