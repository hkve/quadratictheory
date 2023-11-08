import numpy as np


def amplitudes_qccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    r2 = np.zeros((M, M, N, N))

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

    r2 += np.einsum("ackj,bkic->abij", t2, u[v, o, o, v], optimize=True)

    r2 += np.einsum("bcik,akcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("acik,bkcj->abij", t2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bckj,akic->abij", t2, u[v, o, o, v], optimize=True)

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

    r2 += np.einsum("ackj,bdil,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("ablj,cdik,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("adij,bckl,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,bdlj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abil,cdkj,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ackl,bdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("abkl,cdij,klcd->abij", t2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("abij->abij", u[v, v, o, o], optimize=True)

    r2 += np.einsum("cdij,abcd->abij", t2, u[v, v, v, v], optimize=True) / 2

    r2 += np.einsum("bc,acij->abij", f[v, v], t2, optimize=True)

    r2 -= np.einsum("ac,bcij->abij", f[v, v], t2, optimize=True)

    r2 -= np.einsum("ki,abkj->abij", f[o, o], t2, optimize=True)

    r2 -= np.einsum("kj,abik->abij", f[o, o], t2, optimize=True)

    r2 += np.einsum("abkl,klij->abij", t2, u[o, o, o, o], optimize=True) / 2

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

    return r2
