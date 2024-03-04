import numpy as np


def reference_addition_qccsd(t1, t2, l1, l2, det):
    det += np.einsum("ai,bj,bcjk,acik->", l1, t1, l2, t2, optimize=True)

    det -= np.einsum("ai,aj,bcjk,bcik->", l1, t1, l2, t2, optimize=True) / 2

    det -= np.einsum("ai,bi,bcjk,acjk->", l1, t1, l2, t2, optimize=True) / 2

    det += np.einsum("ai,ai,bcjk,bcjk->", l1, t1, l2, t2, optimize=True) / 4

    det += np.einsum("ai,ai,bj,ck,bcjk->", l1, t1, t1, t1, l2, optimize=True) / 2

    det -= np.einsum("ai,aj,bi,ck,bcjk->", l1, t1, t1, t1, l2, optimize=True)

    det -= np.einsum("ai,bj,abij->", l1, l1, t2, optimize=True) / 2

    det += np.einsum("ai,bj,ai,bj->", l1, l1, t1, t1, optimize=True) / 2

    det -= np.einsum("ai,bj,aj,bi->", l1, l1, t1, t1, optimize=True) / 2

    det -= np.einsum("abjk,cdil,abjl,cdik->", l2, l2, t2, t2, optimize=True) / 8

    det -= np.einsum("abjk,cdil,acjk,bdil->", l2, l2, t2, t2, optimize=True) / 8

    det += np.einsum("abjk,cdil,acjl,bdik->", l2, l2, t2, t2, optimize=True) / 4

    det += np.einsum("abjk,cdil,abil,cdjk->", l2, l2, t2, t2, optimize=True) / 32

    det += np.einsum("abjk,cdil,abjk,cdil->", l2, l2, t2, t2, optimize=True) / 32

    det += np.einsum("ai,bj,ck,dl,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 2

    det += np.einsum("ai,bl,cj,dk,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 8

    det += np.einsum("aj,bk,ci,dl,abjk,cdil->", t1, t1, t1, t1, l2, l2, optimize=True) / 8

    return det

def singles_addition_qccsd(t1, t2, l1, l2, det):
    return det

def doubles_addition_qccsd(t1, t2, l1, l2, det):
    return det