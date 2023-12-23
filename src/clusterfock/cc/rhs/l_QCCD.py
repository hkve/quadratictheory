import numpy as np


def lambda_amplitudes_qccd(t2, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r2 = zeros((M, M, N, N), dtype=u.dtype)

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

    r2 += np.einsum("ackj,icbk->abij", l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("bcik,cjak->abij", l2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("acik,cjbk->abij", l2, u[v, o, v, o], optimize=True)

    r2 -= np.einsum("bckj,icak->abij", l2, u[o, v, v, o], optimize=True)

    r2 += np.einsum("abkl,ijkl->abij", l2, u[o, o, o, o], optimize=True) / 2

    r2 += np.einsum("cb,acij->abij", f[v, v], l2, optimize=True)

    r2 -= np.einsum("ca,bcij->abij", f[v, v], l2, optimize=True)

    r2 += np.einsum("cdij,cdab->abij", l2, u[v, v, v, v], optimize=True) / 2

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

    r2 -= np.einsum("ik,abkj->abij", f[o, o], l2, optimize=True)

    r2 -= np.einsum("jk,abik->abij", f[o, o], l2, optimize=True)

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

    r2 += np.einsum("ijab->abij", u[o, o, v, v], optimize=True)

    r2 += np.einsum("ackj,cdkl,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("bcik,cdkl,ljad->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 += np.einsum("abik,cdkl,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bcij,cdkl,klad->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("bckl,cdkl,ijad->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("cdik,cdkl,ljab->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acik,cdkl,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("bckj,cdkl,ilad->abij", l2, t2, u[o, o, v, v], optimize=True)

    r2 -= np.einsum("abkj,cdkl,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("acij,cdkl,klbd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("ackl,cdkl,ijbd->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 -= np.einsum("cdkj,cdkl,ilab->abij", l2, t2, u[o, o, v, v], optimize=True) / 2

    r2 += np.einsum("abkl,cdkl,ijcd->abij", l2, t2, u[o, o, v, v], optimize=True) / 4

    r2 += np.einsum("cdij,cdkl,klab->abij", l2, t2, u[o, o, v, v], optimize=True) / 4

    return r2
