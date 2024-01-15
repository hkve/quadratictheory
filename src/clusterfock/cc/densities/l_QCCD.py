import numpy as np

def one_body_density_addition(rho, t2, l2, o, v):

    oo = np.einsum("abjk,abik->ij", l2, t2)
    print(oo)
    rho[o,o] = +0.25*(oo + oo.transpose(1,0))

    vv = np.einsum("acij,bcij->ab", l2, t2)
    rho[v,v] += 0.25*(vv + vv.transpose(1,0))

    return rho
