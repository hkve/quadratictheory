import numpy as np
import numpy as np
from clusterfock.cc.rhs.l_inter_RCCD import lambda_amplitudes_intermediates_rccd


def l_qccd_restricted(t2, l2, u, f, o, v):
    r2 = lambda_amplitudes_intermediates_rccd(t2, l2, u, f, v, o)

    r2 += l_addition_qccd_restricted(t2, l2, u, f, o, v)

    return r2


def l_addition_qccd_restricted(t2, l2, u, f, o, v):
    M, _, N, _ = t2.shape
    r2 = np.zeros((M, M, N, N), dtype=u.dtype)

    r2 = np.zeros((M, M, N, N), dtype=u.dtype)

    r2 -= 2 * np.einsum("acik,delj,dglk,cebg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("acik,delj,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("acjk,deil,cgkl,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("acjk,deil,eglk,dcbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("ackj,deil,dgkl,cebg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("ackl,deij,dgkl,ecbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("aeij,cdkl,cgkl,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("aeij,cdkl,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("bcik,delj,cgkl,edag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("bcik,delj,dglk,ecag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("bcjk,deil,dgkl,ceag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("bcjk,deil,eglk,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("bcki,delj,egkl,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("bckl,deij,egkl,dcag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("beji,cdkl,cgkl,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 2 * np.einsum("beji,cdkl,egkl,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += 4 * np.einsum("acik,delj,cgkl,edbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += 4 * np.einsum("acik,delj,dglk,ecbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += 4 * np.einsum("aeij,cdkl,cgkl,edbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += 4 * np.einsum("bcjk,deil,cgkl,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += 4 * np.einsum("bcjk,deil,eglk,dcag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 += 4 * np.einsum("beji,cdkl,cgkl,edag->abij", l2, l2, t2, u[v, v, v, v], optimize=True)

    r2 -= 14 * np.einsum("acik,delj,dgkl,ecbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 14 * np.einsum("ackj,deil,eglk,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 14 * np.einsum("bcjk,deil,egkl,dcag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 14 * np.einsum("bcki,delj,dglk,ceag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 5 * np.einsum("acik,delj,cglk,edbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 3

    r2 -= 5 * np.einsum("ackl,deij,cglk,edbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 3

    r2 -= 5 * np.einsum("bcjk,deil,cglk,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 3

    r2 -= 5 * np.einsum("bckl,deij,cglk,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 3

    r2 -= 2 * np.einsum("ackj,deil,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 2 * np.einsum("bcki,delj,dgkl,ceag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= 2 * np.einsum("bckl,deij,eglk,dcag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= np.einsum("acjk,deil,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= np.einsum("bcik,delj,dgkl,ceag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 -= np.einsum("bckl,deij,dglk,ecag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += np.einsum("aeij,cdkl,cglk,edbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 3

    r2 += np.einsum("bckj,deil,eglk,dcag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 3

    r2 += np.einsum("beji,cdkl,cglk,edag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 3

    r2 += np.einsum("acik,delj,cglk,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 6

    r2 += np.einsum("ackl,deij,cglk,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 6

    r2 += np.einsum("bcjk,deil,cglk,edag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 6

    r2 += np.einsum("bckl,deij,cglk,edag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 6

    r2 += np.einsum("acki,delj,egkl,dcbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("acik,delj,egkl,dcbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("acjk,deil,eglk,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("acki,delj,dgkl,ecbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("bcik,delj,dglk,ceag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 2 * np.einsum("bcjk,deil,dgkl,ecag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 4 * np.einsum("acjk,deil,cglk,edbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 4 * np.einsum("ackj,deil,cglk,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 4 * np.einsum("bcik,delj,cglk,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 4 * np.einsum("bcki,delj,cglk,edag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("acjk,deil,dgkl,ecbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("ackj,deil,egkl,dcbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("ackl,deij,dglk,cebg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("ackl,deij,egkl,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("bcik,delj,egkl,dcag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("bcki,delj,dgkl,ecag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("bckl,deij,dgkl,ceag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 5 * np.einsum("bckl,deij,eglk,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 10 * np.einsum("acjk,deil,egkl,dcbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 10 * np.einsum("ackl,deij,eglk,cdbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 10 * np.einsum("bcik,delj,dgkl,ecag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 10 * np.einsum("bckl,deij,dglk,ceag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 9

    r2 += 19 * np.einsum("acjk,deil,cglk,debg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 19 * np.einsum("ackj,deil,cglk,edbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 19 * np.einsum("bcik,delj,cglk,edag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 19 * np.einsum("bcki,delj,cglk,deag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 23 * np.einsum("ackj,deil,dgkl,ecbg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 23 * np.einsum("ackl,deij,dgkl,cebg->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 23 * np.einsum("bcki,delj,egkl,dcag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 += 23 * np.einsum("bckl,deij,egkl,cdag->abij", l2, l2, t2, u[v, v, v, v], optimize=True) / 18

    r2 -= 4 * np.einsum("abim,cdkl,cekl,djem->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("abim,cdkl,cekm,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 -= 4 * np.einsum("abmj,cdkl,cekl,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 -= 4 * np.einsum("abmj,cdkl,cekm,idel->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("acik,bdlm,cekl,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 -= 4 * np.einsum("acik,bdlm,cekm,djel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("acik,bdlm,demk,cjel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("acik,bdlm,deml,cjke->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 -= 4 * np.einsum("acik,delj,cdkm,embl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("acik,delj,cdml,embk->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("acik,delj,cekm,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("acik,delj,delm,mcbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("ackl,bdjm,celk,idem->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("ackl,bdjm,celm,idke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 -= 4 * np.einsum("ackl,bdjm,demk,icel->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("ackl,bdjm,deml,icke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 -= 4 * np.einsum("aeij,cdkl,cdkm,embl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("aeij,cdkl,cekm,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("bcjk,deil,cdkm,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("bcjk,deil,cekm,dmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("bcjk,deil,ceml,dmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("bcjk,deil,deml,mcak->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 -= 4 * np.einsum("beji,cdkl,cdkm,emal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 4 * np.einsum("beji,cdkl,cekm,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("abim,cdkl,cekl,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("abim,cdkl,cekm,djel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("abim,cdkl,cemk,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("abim,cdlk,cemk,djel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("abkl,cdim,cekl,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("abkl,cdim,cekm,djel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("abkl,cdim,demk,cjel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("abkl,cdim,deml,cjke->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("abkl,cdmj,cemk,idel->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("abkl,cdmj,ceml,idke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("abkl,cdmj,delk,icem->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("abkl,cdmj,delm,icke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("abmj,cdkl,cekl,idem->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("abmj,cdkl,cekm,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("abmj,cdkl,cemk,idel->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("abmj,cdlk,cemk,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,celk,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,celm,djek->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,cemk,djel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,ceml,djke->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,dekl,cjme->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,dekm,cjel->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,demk,cjle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("acik,bdlm,deml,cjek->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,delj,cdlm,embk->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,delj,cdmk,embl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,delj,cdml,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,delj,cemk,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,delj,delm,cmbk->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acik,delj,demk,cmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acjk,bdlm,cekl,idem->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("acjk,bdlm,cekm,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("acjk,bdlm,demk,icle->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("acjk,bdlm,deml,icek->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("acjk,deil,cdkm,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("acjk,deil,cekm,dmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acjk,deil,ceml,dmbk->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("acjk,deil,deml,mcbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackj,bdlm,celk,idem->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackj,bdlm,celm,idke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("ackj,bdlm,demk,icel->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackj,bdlm,deml,icke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("ackj,deil,cdmk,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackj,deil,dekm,cmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackj,deil,deml,cmbk->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdim,celk,djme->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,bdim,celm,djek->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdim,demk,cjle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,bdim,deml,cjek->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,celk,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,celm,idek->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,cemk,idel->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,ceml,idke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,dekl,icme->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,dekm,icel->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,delk,icem->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdjm,delm,icke->abij", l2, l2, t2, u[o, v, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,bdmi,celk,djem->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdmi,celm,djke->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,bdmi,dekl,cjem->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,bdmi,dekm,cjle->abij", l2, l2, t2, u[v, o, o, v], optimize=True)

    r2 += 2 * np.einsum("ackl,deij,cdlm,embk->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,deij,cdmk,embl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("ackl,deij,dekm,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("aeij,cdkl,cdkm,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("aeij,cdkl,cekm,dmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("aeij,cdkl,cemk,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("aeij,cdlk,cemk,dmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcik,delj,cdkm,emal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcik,delj,cdml,emak->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcik,delj,cekm,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("bcik,delj,delm,mcak->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("bcjk,deil,cdmk,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("bcjk,deil,celm,dmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcjk,deil,cemk,dmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcjk,deil,ceml,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("bcjk,deil,dekm,cmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcjk,deil,deml,cmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcki,delj,cemk,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("bcki,delj,delm,cmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bcki,delj,demk,cmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bckl,deij,celm,dmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bckl,deij,cemk,dmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("bckl,deij,demk,mcal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("beji,cdkl,cdkm,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("beji,cdkl,cekm,dmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 += 2 * np.einsum("beji,cdkl,cemk,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True)

    r2 += 2 * np.einsum("beji,cdlk,cemk,dmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True)

    r2 -= 23 * np.einsum("ackj,deil,dekm,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 23 * np.einsum("ackl,deij,cdmk,mebl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 23 * np.einsum("bcki,delj,demk,mcal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 23 * np.einsum("bckl,deij,cemk,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 19 * np.einsum("acjk,deil,cdlm,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 19 * np.einsum("ackj,deil,celm,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 19 * np.einsum("bcik,delj,celm,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 19 * np.einsum("bcki,delj,cdlm,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 18

    r2 -= 10 * np.einsum("acjk,deil,demk,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 10 * np.einsum("ackl,deij,ceml,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 10 * np.einsum("bcik,delj,dekm,mcal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 10 * np.einsum("bckl,deij,cdml,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("acjk,deil,dekm,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("ackj,deil,demk,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("ackl,deij,cdml,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("ackl,deij,cemk,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("bcik,delj,demk,mcal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("bcki,delj,dekm,mcal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("bckl,deij,cdmk,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 5 * np.einsum("bckl,deij,ceml,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 4 * np.einsum("abkl,cdim,cekl,djem->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("abkl,cdim,cekm,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("abkl,cdim,delk,cjem->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("abkl,cdim,delm,cjke->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("abkl,cdmj,cekl,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("abkl,cdmj,cekm,idel->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("abkl,cdmj,delk,icme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("abkl,cdmj,delm,icek->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("acjk,bdlm,cemk,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("acjk,bdlm,ceml,idek->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("acjk,bdlm,dekl,icem->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("acjk,bdlm,dekm,icle->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("acjk,deil,celm,dmbk->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("acjk,deil,cemk,dmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackj,bdlm,celk,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("ackj,bdlm,celm,idek->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackj,bdlm,dekl,icme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("ackj,bdlm,dekm,icel->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackj,deil,cdlm,embk->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackj,deil,cdmk,embl->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdim,cemk,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdim,ceml,djek->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdim,delk,cjme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdim,delm,cjek->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdmi,cemk,djel->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdmi,ceml,djke->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdmi,dekl,cjme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,bdmi,dekm,cjel->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,deij,dekm,cmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("ackl,deij,deml,cmbk->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("bcik,delj,cdlm,emak->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("bcik,delj,cdmk,emal->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("bcki,delj,celm,dmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("bcki,delj,cemk,dmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("bckl,deij,delm,cmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("bckl,deij,demk,cmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 4 * np.einsum("acjk,deil,celm,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 4 * np.einsum("ackj,deil,cdlm,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 4 * np.einsum("bcik,delj,cdlm,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 4 * np.einsum("bcki,delj,celm,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("abkl,cdim,celk,djem->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("abkl,cdim,celm,djke->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("abkl,cdim,dekl,cjem->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("abkl,cdim,dekm,cjle->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("abkl,cdmj,celk,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("abkl,cdmj,celm,idek->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("abkl,cdmj,dekl,icme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("abkl,cdmj,dekm,icel->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("acjk,bdlm,celk,idme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("acjk,bdlm,celm,idek->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("acjk,bdlm,dekl,icme->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("acjk,bdlm,dekm,icel->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("acjk,deil,cdlm,embk->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("acjk,deil,cdmk,embl->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackj,bdlm,cemk,idle->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("ackj,bdlm,ceml,idek->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackj,bdlm,dekl,icem->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackj,bdlm,dekm,icle->abij", l2, l2, t2, u[o, v, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("ackj,deil,celm,dmbk->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackj,deil,cemk,dmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdim,cemk,djel->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdim,ceml,djke->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdim,dekl,cjme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdim,dekm,cjel->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdmi,cemk,djle->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdmi,ceml,djek->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdmi,delk,cjme->abij", l2, l2, t2, u[v, o, o, v], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,bdmi,delm,cjek->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,deij,delm,cmbk->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("ackl,deij,demk,cmbl->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("bcik,delj,celm,dmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("bcik,delj,cemk,dmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("bcki,delj,cdlm,emak->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("bcki,delj,cdmk,emal->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("bckl,deij,dekm,cmal->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("bckl,deij,deml,cmak->abij", l2, l2, t2, u[v, o, v, o], optimize=True) / 3

    r2 -= 2 * np.einsum("acik,delj,demk,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("acjk,deil,ceml,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("acki,delj,dekm,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("bcik,delj,cdml,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("bcjk,deil,dekm,mcal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= np.einsum("aeij,cdlk,cekm,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= np.einsum("bckj,deil,deml,mcak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= np.einsum("beji,cdlk,cekm,mdal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 -= np.einsum("acik,delj,cdlm,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 6

    r2 -= np.einsum("ackl,deij,cdlm,mebk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 6

    r2 -= np.einsum("bcjk,deil,celm,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 6

    r2 -= np.einsum("bckl,deij,celm,mdak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 6

    r2 -= np.einsum("acki,delj,demk,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += np.einsum("acjk,deil,cemk,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += np.einsum("bcik,delj,cdmk,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += np.einsum("bckl,deij,delm,mcak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 2 * np.einsum("ackj,deil,cemk,mdbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 2 * np.einsum("bcki,delj,cdmk,meal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 2 * np.einsum("bckl,deij,deml,mcak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 5 * np.einsum("acik,delj,celm,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 += 5 * np.einsum("ackl,deij,celm,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 += 5 * np.einsum("bcjk,deil,cdlm,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 += 5 * np.einsum("bckl,deij,cdlm,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 3

    r2 += 14 * np.einsum("acik,delj,dekm,mcbl->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 14 * np.einsum("ackj,deil,ceml,mdbk->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 14 * np.einsum("bcjk,deil,demk,mcal->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 += 14 * np.einsum("bcki,delj,cdml,meak->abij", l2, l2, t2, u[o, v, v, o], optimize=True) / 9

    r2 -= 2 * np.einsum("abim,cdkl,cdkn,njml->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("abim,cdkl,dcmn,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("abkl,cdim,cdkn,njml->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("abkl,cdim,cdnm,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("abkl,cdmj,cdmn,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("abkl,cdmj,cdnl,inkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("abmj,cdkl,cdkn,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("abmj,cdkl,cdmn,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("acik,bdlm,cdln,njmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("acik,bdlm,cdmn,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("acik,bdlm,cdnk,njml->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("acik,bdlm,cdnm,njlk->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("acjk,bdlm,cdkn,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("acjk,bdlm,cdnm,inlk->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackj,bdlm,cdln,inkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackj,bdlm,cdnm,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdim,cdln,njmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdim,cdnm,njlk->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdjm,cdln,inmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdjm,cdmn,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdjm,cdnk,inml->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdjm,cdnl,inkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdmi,cdln,njkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 -= 2 * np.einsum("ackl,bdmi,cdnk,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += 4 * np.einsum("abim,cdkl,cdkn,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += 4 * np.einsum("abmj,cdkl,cdkn,inml->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += 4 * np.einsum("acik,bdlm,cdkn,njml->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += 4 * np.einsum("acik,bdlm,cdnm,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += 4 * np.einsum("ackl,bdjm,cdln,inkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += 4 * np.einsum("ackl,bdjm,cdnm,inkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True)

    r2 += 2 * np.einsum("abkl,cdim,cdln,njkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("abkl,cdim,cdnk,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("abkl,cdmj,cdln,inmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("abkl,cdmj,cdnk,inml->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("acjk,bdlm,cdln,inmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("acjk,bdlm,cdnk,inml->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("ackj,bdlm,cdmn,inlk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("ackj,bdlm,cdnk,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("ackl,bdim,cdmn,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("ackl,bdim,cdnk,njml->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("ackl,bdmi,cdmn,njlk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 2 * np.einsum("ackl,bdmi,cdnl,njmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("abkl,cdim,cdkn,njlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("abkl,cdim,cdnl,njkm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("abkl,cdmj,cdkn,inml->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("abkl,cdmj,cdnl,inmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("acjk,bdlm,cdmn,inlk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("acjk,bdlm,cdnk,inlm->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("ackj,bdlm,cdln,inmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("ackj,bdlm,cdnk,inml->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("ackl,bdim,cdmn,njlk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("ackl,bdim,cdnl,njmk->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("ackl,bdmi,cdmn,njkl->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 += 4 * np.einsum("ackl,bdmi,cdnk,njml->abij", l2, l2, t2, u[o, o, o, o], optimize=True) / 3

    r2 -= 8 * np.einsum(
        "abim,cdkl,cgkm,deln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "abmj,cdkl,cekm,dgln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acik,bdlm,cekn,dgml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acik,bdlm,cgkl,demn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acik,delj,cekm,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acik,delj,cgkn,delm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "ackl,bdjm,celk,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "ackl,bdjm,cgln,demk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "aeij,cdkl,cekm,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "aeim,cdkl,cdkn,egml,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "aeim,cdkl,cgkl,denm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "aeim,cdkl,cgkm,deln,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bcjk,deil,cdkm,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bcjk,deil,cgkn,deml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "beji,cdkl,cekm,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bejm,cdkl,cgkl,denm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bejm,cdkl,cgkm,deln,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdkl,cdkn,egml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdkl,cekl,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdkl,cekm,dgnl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdkl,cgkl,denm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdkl,cgkn,deml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdkl,cgmk,denl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdlk,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abim,cdlk,cemk,dgnl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cdkn,egml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cdnm,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cekl,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cekm,dgnl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cenk,dgml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cgkl,denm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cgkn,deml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdim,cgnl,demk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cdmn,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cdnl,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cemk,dgnl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cemn,dglk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cenk,dglm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cgmk,deln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cgml,denk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdmj,cgnm,delk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdkl,cdmn,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdkl,cekl,dgnm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdkl,cekn,dgml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdkl,cemk,dgnl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdkl,cgkl,demn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdkl,cgkm,denl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdlk,cdln,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmj,cdlk,cgmk,denl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmn,cdkl,cekl,dgmn,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmn,cdkl,cekm,dgnl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmn,cdkl,cgkl,denm,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abmn,cdkl,cgkn,deml,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cdln,egmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cdnk,egml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cdnm,eglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,celk,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,celm,dgnk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cemk,dgnl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,ceml,dgkn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cenl,dgmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cenm,dgkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cglk,denm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cgln,demk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cgml,denk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cgmn,dekl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cgnk,deml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,bdlm,cgnl,dekm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cdln,egmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cdml,egnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cdmn,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cdnk,egml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cdnl,egkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cemk,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cemn,dglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cenk,dglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cgmk,deln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,delj,cgml,denk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cdkn,eglm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cdnm,eglk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cekl,dgnm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cekn,dgml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cenl,dgmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cgkl,demn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cgkm,denl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,bdlm,cgnk,deml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cdkm,egnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cdkn,eglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cdnm,eglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cekn,dgml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cenl,dgmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cgkl,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cgkm,denl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,deil,cgnk,deml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,cdln,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,cdnm,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,celk,dgnm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,celn,dgmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,cenk,dgml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,cglk,demn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,cglm,denk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,bdlm,cgnl,demk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,deil,cdmk,egnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,deil,cdmn,eglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,deil,cdnk,eglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,deil,cgmk,denl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,deil,cgml,dekn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,cdln,egmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,cdnm,eglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,celk,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,celm,dgnk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,cenl,dgmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,cglk,denm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,cgln,demk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdim,cgnk,deml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cdln,egmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cdmn,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cdnk,egml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cdnl,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,celm,dgnk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,celn,dgkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cemk,dgnl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cemn,dglk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cenk,dglm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cenm,dgkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cglk,denm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cglm,dekn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cgmk,deln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cgml,denk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cgnl,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdjm,cgnm,delk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,cdln,egkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,cdnk,eglm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,celk,dgnm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,celm,dgkn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,cenl,dgkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,cglm,denk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,cgln,dekm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmi,cgnm,dekl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,celm,dgnk,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,celn,dgkm,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,cemk,dgnl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,cenk,dglm,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,cglk,denm,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,cglm,dekn,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,cgml,denk,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,bdmn,cgnm,delk,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deij,cdln,egmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deij,cdnk,egml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deij,cglk,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deij,cglm,dekn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deij,cgnl,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deim,cdln,egmk,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deim,cdnk,egml,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deim,celn,dgkm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deim,cenm,dgkl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deim,cglk,denm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,deim,cglm,dekn,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cdln,egkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cdnm,egkl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cdnm,eglk,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,celn,dgkm,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,celn,dgmk,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cemn,dglk,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cenk,dgml,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cenl,dgmk,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cglk,demn,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cglm,dekn,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cglm,denk,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,demj,cgmk,deln,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdkl,cdkm,egnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdkl,cdmn,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdkl,cekn,dgml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdkl,cemk,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdkl,cgkl,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdkl,cgkm,denl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdlk,cdln,egkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeij,cdlk,cgmk,denl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeim,cdkl,cdmn,egkl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeim,cdkl,cekn,dgml,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeim,cdkl,cgkl,demn,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeim,cdkl,cgkm,denl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeim,cdlk,cdln,egkm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aeim,cdlk,cgmk,denl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aejm,cdkl,cdkn,egml,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aejm,cdkl,cdmn,egkl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aejm,cdkl,cgkl,demn,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aejm,cdkl,cgkl,denm,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aejm,cdkl,cgkm,denl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aejm,cdlk,cgmk,denl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aemj,cdkl,cekn,dgml,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aemj,cdkl,cgkl,demn,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aemj,cdkl,cgkm,denl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "aemj,cdlk,cdln,egkm,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cdkn,egml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cdnl,egmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cekm,dgnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cekn,dglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cenm,dglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cgkl,denm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cgkm,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,delj,cgnk,delm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cdmk,egnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cdmn,eglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cdnk,eglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,celn,dgmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,ceml,dgnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cemn,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cenk,dgml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cenl,dgkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cgmk,denl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,deil,cgml,dekn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcki,delj,cemk,dgnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcki,delj,cemn,dglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcki,delj,cenk,dglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcki,delj,cgmk,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcki,delj,cgml,denk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deij,celn,dgmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deij,cenk,dgml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deij,cglk,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deij,cglm,denk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deij,cgnl,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cdln,egkm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cdln,egmk,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cdmn,eglk,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cdnk,egml,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cdnl,egmk,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,celn,dgkm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cenm,dgkl,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cenm,dglk,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cglk,denm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cglm,dekn,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cglm,denk,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,deim,cgmk,denl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,demj,cdln,egkm,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,demj,cdnm,egkl,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,demj,celn,dgmk,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,demj,cenk,dgml,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,demj,cglk,demn,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,demj,cglm,denk,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beim,cdkl,cdkn,egml,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beim,cdkl,cdmn,egkl,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beim,cdkl,cgkl,demn,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beim,cdkl,cgkl,denm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beim,cdkl,cgkm,denl,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beim,cdlk,cgmk,denl,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdkl,cdkm,egnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdkl,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdkl,cekn,dgml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdkl,cemk,dgnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdkl,cgkl,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdkl,cgkm,denl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdlk,cdln,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "beji,cdlk,cgmk,denl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bejm,cdkl,cdmn,egkl,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bejm,cdkl,cekn,dgml,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bejm,cdkl,cgkl,demn,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bejm,cdkl,cgkm,denl,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bejm,cdlk,cdln,egkm,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bejm,cdlk,cgmk,denl,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bemi,cdkl,cekn,dgml,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bemi,cdkl,cgkl,demn,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bemi,cdkl,cgkm,denl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bemi,cdlk,cdln,egkm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,cdml,egkn,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,cdnk,eglm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,celm,dgkn,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,cemk,dgln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,ceml,dgnk,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,cenl,dgkm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,cgln,dekm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,eglj,cgmk,denl,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdkl,egij,cdkm,egln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdkl,egij,cdkn,egml,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdkl,egij,cekm,dgnl,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdkl,egij,cgkn,deml,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abim,cdkl,cekl,dgnm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abim,cdkl,cekm,dgln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abim,cdkl,cekn,dgml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abim,cdkl,cgkl,demn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abim,cdkl,cgkm,denl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abim,cdlk,cdln,egkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abkl,cdim,cekn,dgml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abkl,cdim,cgkl,demn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abkl,cdmj,cemk,dgln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abkl,cdmj,cgmn,delk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abmj,cdkl,cdkn,egml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abmj,cdkl,cekl,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abmj,cdkl,cekm,dgnl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abmj,cdkl,cgkl,denm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abmj,cdkl,cgkm,deln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abmj,cdkl,cgkn,deml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abmn,cdkl,cekm,dgln,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cdkn,egml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cdnm,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cekl,dgmn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cekm,dgnl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,celn,dgmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cemn,dgkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cenk,dgml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cgkl,denm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cgkn,deml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cglk,demn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cgml,dekn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,bdlm,cgnl,demk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cdkn,egml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cdml,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cdnl,egmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cekm,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cekn,dglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cemk,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cenm,dglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cgkl,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cgkm,deln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,delj,cgnk,delm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acjk,bdlm,cekl,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acjk,bdlm,cgkn,deml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acjk,deil,cdkm,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acjk,deil,cgkn,deml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackj,bdlm,celk,dgmn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackj,bdlm,cgln,demk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackj,deil,cdmk,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdim,celn,dgmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdim,cglk,demn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cdln,egkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cdnm,egkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,celk,dgnm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,celm,dgkn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,celn,dgmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cemk,dgln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cenk,dgml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cglk,demn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cglm,denk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cgln,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cgmn,delk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdjm,cgnl,demk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdmi,celn,dgkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdmi,cglm,dekn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdmn,celk,dgnm,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,bdmn,cglm,denk,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,deij,cgln,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,deim,cdnk,egml,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,deim,celn,dgkm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,deim,cenm,dgkl,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,deim,cglm,dekn,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,demj,celn,dgmk,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,demj,cglk,demn,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeij,cdkl,cdkm,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeij,cdkl,cdkn,egml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeij,cdkl,cekm,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeij,cdkl,cgkl,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeij,cdkl,cgkm,deln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeij,cdkl,cgkn,deml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeim,cdkl,cdkn,egml,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeim,cdkl,cdmn,egkl,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeim,cdkl,cgkl,demn,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeim,cdkl,cgkl,denm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeim,cdkl,cgkm,deln,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeim,cdkl,cgkm,denl,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aeim,cdlk,cgmk,denl,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aejm,cdkl,cdkn,egml,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aejm,cdkl,cgkl,denm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aejm,cdkl,cgkm,deln,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "aemj,cdkl,cgkm,deln,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcik,delj,cekm,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcik,delj,cgkn,delm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cdkm,egnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cdkn,eglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cdmk,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cdnm,eglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cekn,dgml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,ceml,dgkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cenl,dgmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cgkl,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cgkm,denl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,deil,cgnk,deml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcki,delj,cemk,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,deij,cgln,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,deim,cdln,egmk,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,deim,cglk,denm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,demj,cdln,egkm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,demj,cdnm,egkl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,demj,cenk,dgml,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,demj,cglm,denk,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beim,cdkl,cdkn,egml,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beim,cdkl,cgkl,denm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beim,cdkl,cgkm,deln,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beji,cdkl,cdkm,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beji,cdkl,cdkn,egml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beji,cdkl,cekm,dgnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beji,cdkl,cgkl,denm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beji,cdkl,cgkm,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "beji,cdkl,cgkn,deml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bejm,cdkl,cdmn,egkl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bejm,cdkl,cgkl,demn,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bejm,cdkl,cgkl,denm,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bejm,cdkl,cgkm,deln,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bejm,cdkl,cgkm,denl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bejm,cdlk,cgmk,denl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bemi,cdkl,cgkm,deln,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdik,eglj,cdmk,egln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdik,eglj,ceml,dgkn,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdkl,egij,cekm,dgln,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= (
        25
        * np.einsum("bejm,cdkl,cdkn,egml,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        23
        * np.einsum("ackj,deil,cgln,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        23
        * np.einsum("ackl,deij,cdmk,egln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        23
        * np.einsum("ackl,deim,cemn,dgkl,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        23
        * np.einsum("ackl,deim,cenl,dgkm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        23
        * np.einsum("bcki,delj,cgln,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        23
        * np.einsum("bckl,deij,cemk,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        23
        * np.einsum("bckl,demj,cdmn,egkl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        23
        * np.einsum("bckl,demj,cdnl,egkm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("acjk,deil,cgln,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("ackl,deij,ceml,dgkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("ackl,deim,cdmn,eglk,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("ackl,deim,cdnk,eglm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("bcik,delj,cgln,dekm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("bckl,deij,cdml,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("bckl,demj,cemn,dglk,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        20
        * np.einsum("bckl,demj,cenk,dglm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("acjk,deil,cdlm,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("ackj,deil,celm,dgkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("ackl,deim,cgmk,denl,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("ackl,deim,cgml,dekn,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("bcik,delj,celm,dgkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("bcki,delj,cdlm,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("bckl,demj,cgmk,deln,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        19
        * np.einsum("bckl,demj,cgml,denk,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("acik,delj,cglk,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("acik,delj,cglm,dekn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("acik,delj,cgnl,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("ackj,deil,ceml,dgnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("ackj,deil,cemn,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("ackj,deil,cenl,dgkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("aejm,cdkl,cekn,dgml,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("aejm,cdlk,cdln,egkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("bcjk,deil,cglk,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("bcjk,deil,cglm,denk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("bcjk,deil,cgnl,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("bcki,delj,cdml,egnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("bcki,delj,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("bcki,delj,cdnl,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("beim,cdkl,cekn,dgml,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        14
        * np.einsum("beim,cdlk,cdln,egkm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("acjk,deil,cgln,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("ackj,deil,cgln,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("ackl,deij,cdml,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("ackl,deij,cemk,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("ackl,deim,cdmn,egkl,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("ackl,deim,cdnl,egkm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("ackl,deim,cemn,dglk,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("ackl,deim,cenk,dglm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bcik,delj,cgln,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bcki,delj,cgln,dekm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bckl,deij,cdmk,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bckl,deij,ceml,dgkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bckl,demj,cdmn,eglk,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bckl,demj,cdnk,eglm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bckl,demj,cemn,dgkl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        10
        * np.einsum("bckl,demj,cenl,dgkm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("abkl,cdim,cekn,dglm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("abkl,cdim,cgkm,deln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("abkl,cdmj,cekm,dgln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("abkl,cdmj,cgkn,delm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("acjk,bdlm,ceml,dgkn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("acjk,bdlm,cgmn,dekl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("ackj,bdlm,celm,dgkn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("ackj,bdlm,cgln,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("ackl,bdim,cemn,dglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("ackl,bdim,cgmk,deln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("ackl,bdmi,cemn,dgkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("ackl,bdmi,cgml,dekn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        8
        * np.einsum("acjk,deil,celm,dgkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("ackj,deil,cdlm,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("ackl,deim,cgmk,deln,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("ackl,deim,cgml,denk,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("bcik,delj,cdlm,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("bcki,delj,celm,dgkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("bckl,demj,cgmk,denl,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        8
        * np.einsum("bckl,demj,cgml,dekn,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        5
        * np.einsum("acik,delj,celm,dgnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("acik,delj,celn,dgkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("acik,delj,cenm,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("ackl,deij,celm,dgnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("ackl,deij,celn,dgkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("ackl,deij,cenm,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("bcjk,deil,cdlm,egnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("bcjk,deil,cdln,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("bcjk,deil,cdnm,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("bckl,deij,cdlm,egnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("bckl,deij,cdln,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        5
        * np.einsum("bckl,deij,cdnm,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("abkl,cdim,celn,dgkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("abkl,cdim,cglm,dekn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("abkl,cdmj,celm,dgkn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("abkl,cdmj,cgln,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("acjk,bdlm,celm,dgkn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("acjk,bdlm,cgln,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("ackj,bdlm,ceml,dgkn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("ackj,bdlm,cgmn,dekl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("ackl,bdim,cemn,dgkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("ackl,bdim,cgml,dekn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("ackl,bdmi,cemn,dglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("ackl,bdmi,cgmk,deln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        4
        * np.einsum("acik,delj,cgln,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("acjk,deil,ceml,dgkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("acki,delj,cgln,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("ackl,deim,cdkn,egml,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("aeim,cdlk,cekn,dgml,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("bcik,delj,cdml,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("bcjk,deil,cgln,dekm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("bckl,demj,cekn,dgml,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        4
        * np.einsum("bejm,cdlk,cekn,dgml,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("aeij,cdlk,cekm,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("aeim,cdlk,cgkl,denm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("aemi,cdkl,cgkm,deln,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("bckj,deil,cgkn,deml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("beji,cdlk,cekm,dgln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("bejm,cdlk,cgkl,denm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("bemj,cdkl,cgkm,deln,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        2
        * np.einsum("acki,delj,cgln,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("ackj,deil,cemk,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("ackj,deil,cemn,dglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("ackj,deil,cenk,dglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("bcki,delj,cdmk,egnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("bcki,delj,cdmn,eglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("bcki,delj,cdnk,eglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("bckl,deij,cgkl,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("bckl,deij,cgkm,denl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        2
        * np.einsum("bckl,deij,cgnk,deml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("acik,delj,cdlm,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("ackl,deij,cdlm,egkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("ackl,deim,cglk,demn,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("aeim,cdkl,cdln,egkm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("bcjk,deil,celm,dgkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("bckl,deij,celm,dgkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("bckl,demj,cglk,denm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("bejm,cdkl,cdln,egkm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 -= (
        np.einsum("acjk,deil,cemk,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("acjk,deil,cemn,dglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("acjk,deil,cenk,dglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("bcik,delj,cdmk,egnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("bcik,delj,cdmn,eglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("bcik,delj,cdnk,eglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("bckl,deij,cgkl,denm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("bckl,deij,cgkm,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 -= (
        np.einsum("bckl,deij,cgnk,delm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("aeij,cdlk,cekm,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("aeij,cdlk,cgkl,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("aeij,cdlk,cgkm,deln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("aejm,cdlk,cgkl,denm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("bckj,deil,cgkl,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("bckj,deil,cgkm,denl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("bckj,deil,cgnk,deml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("beim,cdlk,cgkl,denm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("beji,cdlk,cekm,dgnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("beji,cdlk,cgkl,denm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("beji,cdlk,cgkm,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        np.einsum("acik,delj,cdlm,egnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("acik,delj,cdln,egkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("acik,delj,cdnm,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("ackl,deij,cdlm,egnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("ackl,deij,cdln,egkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("ackl,deij,cdnm,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("bcjk,deil,celm,dgnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("bcjk,deil,celn,dgkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("bcjk,deil,cenm,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("bckl,deij,celm,dgnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("bckl,deij,celn,dgkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("bckl,deij,cenm,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        np.einsum("acki,delj,cglk,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("acki,delj,cglm,denk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("acki,delj,cgnl,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("aemj,cdkl,cdln,egkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("aemj,cdlk,cekn,dgml,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("bemi,cdkl,cdln,egkm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        np.einsum("bemi,cdlk,cekn,dgml,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,cdln,egkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,cdnk,eglm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,celk,dgnm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,celm,dgkn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,cenl,dgkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,cglm,denk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,cgln,dekm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdim,cgnm,dekl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,cdln,egmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,cdnk,egml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,celm,dgnk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,celn,dgkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,cenm,dgkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,cglk,denm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,cglm,dekn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abkl,cdmj,cgnl,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abmn,cdkl,cgmk,denl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("abmn,cdlk,cemk,dgnl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,cdln,egmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,cdnk,egml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,celm,dgnk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,celn,dgkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,cenm,dgkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,cglk,denm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,cglm,dekn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,bdlm,cgnl,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,deil,cdln,egmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acjk,deil,cdnk,egml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,cdmn,eglk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,cdnk,eglm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,ceml,dgnk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,cemn,dgkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,cenl,dgkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,cgmk,denl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,cgml,dekn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,bdlm,cgnm,dekl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,deil,celn,dgmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackj,deil,cenk,dgml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,cdnk,egml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,cemk,dgnl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,ceml,dgkn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,cenm,dgkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,cgml,denk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,cgmn,dekl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdim,cgnl,dekm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,cdmn,eglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,cdnl,egmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,cemk,dgln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,ceml,dgnk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,cenm,dglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,cgmk,denl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,cgmn,delk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmi,cgnk,delm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmn,cemn,dglk,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmn,cenm,dgkl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmn,cgmk,deln,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,bdmn,cgnl,dekm,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deij,cgmk,deln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deij,cgml,denk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cdmn,egkl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cdnl,egkm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cemn,dglk,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cenk,dglm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cgmk,deln,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cgml,denk,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,demj,cdmn,eglk,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,demj,cdnl,egkm,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,demj,cemn,dgkl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("ackl,demj,cenk,dglm,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("aejm,cdkl,cgmk,denl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("aejm,cdlk,cdmn,egkl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("aemj,cdkl,cdmn,egkl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("aemj,cdlk,cgmk,denl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bcik,delj,celn,dgmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bcik,delj,cenk,dgml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bcki,delj,cdln,egmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bcki,delj,cdnk,egml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,deij,cgmk,denl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,deij,cgml,dekn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,deim,cdmn,egkl,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,deim,cdnk,eglm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,deim,cemn,dglk,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,deim,cenl,dgkm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cdmn,eglk,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cdnk,eglm,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cemn,dgkl,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cenl,dgkm,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cgmk,denl,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cgml,dekn,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("beim,cdkl,cgmk,denl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("beim,cdlk,cdmn,egkl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bemi,cdkl,cdmn,egkl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("bemi,cdlk,cgmk,denl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdik,eglj,cdln,egkm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdik,eglj,cdnl,egmk,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdik,eglj,cglm,denk,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdik,eglj,cgnk,delm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdkl,egij,cgmk,denl,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("cdlk,egij,cemk,dgnl,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        2
        * np.einsum("acik,delj,cglk,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acik,delj,cglm,denk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acik,delj,cgnl,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acjk,deil,cemk,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acjk,deil,ceml,dgnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acjk,deil,cemn,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acjk,deil,cenl,dgkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acki,delj,cglk,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acki,delj,cglm,dekn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("acki,delj,cgnl,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cdkn,eglm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("ackl,deim,cdnm,eglk,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("aejm,cdkl,cdln,egkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("aejm,cdlk,cekn,dgml,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("aemj,cdkl,cekn,dgml,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("aemj,cdlk,cdln,egkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bcik,delj,cdmk,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bcik,delj,cdml,egnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bcik,delj,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bcik,delj,cdnl,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bcjk,deil,cglk,denm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bcjk,deil,cglm,dekn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bcjk,deil,cgnl,dekm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bckl,deij,cgkn,delm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cekn,dglm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bckl,demj,cenm,dglk,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("beim,cdkl,cdln,egkm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("beim,cdlk,cekn,dgml,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bemi,cdkl,cekn,dgml,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        2
        * np.einsum("bemi,cdlk,cdln,egkm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cdkn,eglm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cdnl,egkm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cekl,dgnm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cekm,dgln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cenk,dglm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cgkm,denl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cgkn,delm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdim,cgnm,delk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cdkn,egml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cdnl,egmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cekm,dgnl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cekn,dglm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cenm,dglk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cgkl,denm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cgkm,deln,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abkl,cdmj,cgnk,delm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abmn,cdkl,cemk,dgnl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("abmn,cdlk,cgmk,denl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,cdmn,eglk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,cdnk,eglm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,ceml,dgnk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,cemn,dgkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,cenl,dgkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,cgmk,denl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,cgml,dekn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,bdlm,cgnm,dekl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,deil,celn,dgmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,deil,cenk,dgml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,cdln,egmk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,cdnk,egml,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,celm,dgnk,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,celn,dgkm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,cenm,dgkl,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,cglk,denm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,cglm,dekn,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,bdlm,cgnl,dekm,ineg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,deil,cdln,egmk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackj,deil,cdnk,egml,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,cdmn,eglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,cdnl,egmk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,cemk,dgln,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,ceml,dgnk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,cenm,dglk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,cgmk,denl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,cgmn,delk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdim,cgnk,delm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,cdmn,egkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,cdnk,egml,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,cemk,dgnl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,ceml,dgkn,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,cenm,dgkl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,cgml,denk,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,cgmn,dekl,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmi,cgnl,dekm,njeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmn,cemn,dgkl,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmn,cenm,dglk,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmn,cgml,dekn,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,bdmn,cgnk,delm,ijeg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deij,cgmk,denl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deij,cgml,dekn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cdmn,eglk,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cdnk,eglm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cemn,dgkl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cenl,dgkm,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cgmk,denl,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cgml,dekn,njbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,demj,cdmn,egkl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,demj,cdnk,eglm,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,demj,cemn,dglk,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("ackl,demj,cenl,dgkm,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("aejm,cdkl,cdmn,egkl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("aejm,cdlk,cgmk,denl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("aemj,cdkl,cgmk,denl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("aemj,cdlk,cdmn,egkl,ingb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bcik,delj,cdln,egmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bcik,delj,cdnk,egml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bcki,delj,celn,dgmk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bcki,delj,cenk,dgml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,deij,cgmk,deln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,deij,cgml,denk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,deim,cdmn,eglk,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,deim,cdnl,egkm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,deim,cemn,dgkl,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,deim,cenk,dglm,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,demj,cdmn,egkl,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,demj,cdnl,egkm,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,demj,cemn,dglk,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,demj,cenk,dglm,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,demj,cgmk,deln,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bckl,demj,cgml,denk,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("beim,cdkl,cdmn,egkl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("beim,cdlk,cgmk,denl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bemi,cdkl,cgmk,denl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("bemi,cdlk,cdmn,egkl,njag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdik,eglj,cdln,egmk,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdik,eglj,cdnl,egkm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdik,eglj,celm,dgnk,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdik,eglj,cenk,dglm,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdkl,egij,cemk,dgnl,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("cdlk,egij,cgmk,denl,mnab->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        4
        * np.einsum("acjk,deil,celm,dgnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("acjk,deil,celn,dgkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("acjk,deil,cenm,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("ackj,deil,cdlm,egnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("ackj,deil,cdln,egkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("ackj,deil,cdnm,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("ackj,deil,cemk,dgln,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cdln,egkm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("ackl,deim,cenm,dglk,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bcik,delj,cdlm,egnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bcik,delj,cdln,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bcik,delj,cdnm,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bcki,delj,cdmk,egln,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bcki,delj,celm,dgnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bcki,delj,celn,dgkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bcki,delj,cenm,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bckl,deij,cgkn,deml,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bckl,demj,cdnm,eglk,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        4
        * np.einsum("bckl,demj,celn,dgkm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("acjk,deil,cglk,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("acjk,deil,cglm,dekn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("acjk,deil,cgnl,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackj,deil,cglk,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackj,deil,cglm,denk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackj,deil,cgnl,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,deij,cdml,egnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,deij,cdmn,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,deij,cdnl,egkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,deij,cemk,dgnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,deij,cemn,dglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,deij,cenk,dglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,demj,cdnk,eglm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,demj,cenl,dgkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,demj,cgmk,denl,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("ackl,demj,cgml,dekn,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bcik,delj,cglk,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bcik,delj,cglm,denk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bcik,delj,cgnl,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bcki,delj,cglk,denm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bcki,delj,cglm,dekn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bcki,delj,cgnl,dekm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deij,cdmk,egnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deij,cdmn,eglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deij,cdnk,eglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deij,ceml,dgnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deij,cemn,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deij,cenl,dgkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deim,cdnl,egkm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deim,cenk,dglm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deim,cgmk,deln,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        5
        * np.einsum("bckl,deim,cgml,denk,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("acik,delj,celm,dgkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("ackl,deij,celm,dgkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("ackl,deim,cglk,denm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("aeim,cdlk,cdln,egkm,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("bcjk,deil,cdlm,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("bckl,deij,cdlm,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("bckl,demj,cglk,demn,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("bejm,cdlk,cdln,egkm,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 3
    )

    r2 += (
        10
        * np.einsum("acjk,deil,cglk,demn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("acjk,deil,cglm,denk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("acjk,deil,cgnl,demk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("ackl,deij,ceml,dgnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("ackl,deij,cemn,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("ackl,deij,cenl,dgkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("ackl,demj,cenk,dglm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("ackl,demj,cgml,denk,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bcik,delj,cglk,denm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bcik,delj,cglm,dekn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bcik,delj,cgnl,dekm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bckl,deij,cdml,egnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bckl,deij,cdmn,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bckl,deij,cdnl,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bckl,deim,cdnk,eglm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        10
        * np.einsum("bckl,deim,cgml,dekn,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        19
        * np.einsum("acjk,deil,cdlm,egnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("acjk,deil,cdln,egkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("acjk,deil,cdnm,egkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("ackj,deil,celm,dgnk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("ackj,deil,celn,dgkm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("ackj,deil,cenm,dgkl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("bcik,delj,celm,dgnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("bcik,delj,celn,dgkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("bcik,delj,cenm,dgkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("bcki,delj,cdlm,egnk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("bcki,delj,cdln,egkm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        19
        * np.einsum("bcki,delj,cdnm,egkl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackj,deil,cglk,denm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackj,deil,cglm,dekn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackj,deil,cgnl,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackl,deij,cdmk,egnl,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackl,deij,cdmn,eglk,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackl,deij,cdnk,eglm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackl,demj,cdnl,egkm,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("ackl,demj,cgmk,deln,inbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bcki,delj,cglk,demn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bcki,delj,cglm,denk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bcki,delj,cgnl,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bckl,deij,cemk,dgnl,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bckl,deij,cemn,dglk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bckl,deij,cenk,dglm,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bckl,deim,cenl,dgkm,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        23
        * np.einsum("bckl,deim,cgmk,denl,njga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 18
    )

    r2 += (
        25
        * np.einsum("bejm,cdkl,cdkn,egml,inga->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 6
    )

    r2 += (
        28
        * np.einsum("acik,delj,cgln,dekm,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        28
        * np.einsum("ackj,deil,ceml,dgkn,mnbg->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        28
        * np.einsum("ackl,deim,cdln,egmk,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        28
        * np.einsum("aeim,cdkl,cekn,dgml,njgb->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        28
        * np.einsum("bcjk,deil,cgln,demk,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        28
        * np.einsum("bcki,delj,cdml,egkn,mnag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        28
        * np.einsum("bckl,demj,celn,dgmk,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    r2 += (
        28
        * np.einsum("bejm,cdkl,cekn,dgml,inag->abij", l2, l2, t2, t2, u[o, o, v, v], optimize=True)
        / 9
    )

    return r2
