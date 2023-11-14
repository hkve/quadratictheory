import numpy as np


def one_body_density_oo(rho, t1, t2, l1, l2, o, v):
    N = o.stop

    rho[o, o] += np.eye(N, dtype=rho.dtype)
    rho[o, o] -= np.einsum("aj,ai->ij", l1, t1)
    rho[o, o] += np.einsum("abkj,abik->ij", l2, t2) / 2

    return rho


def one_body_density_vv(rho, t1, t2, l1, l2, o, v):
    rho[v, v] += np.einsum("ai,bi->ab", l1, t1)
    rho[v, v] += np.einsum("acij,bcij->ab", l2, t2) / 2

    return rho


def one_body_density_ov(rho, t1, t2, l1, l2, o, v):
    rho[o, v] += t1.T
    rho[o, v] -= np.einsum("aj,bcjk,bcik->ia", t1, l2, t2, optimize=True) / 2
    rho[o, v] -= np.einsum("bi,bcjk,acjk->ia", t1, l2, t2, optimize=True) / 2
    rho[o, v] += np.einsum("bj,abij->ia", l1, t2)
    rho[o, v] -= np.einsum("bj,aj,bi->ia", l1, t1, t1)

    return rho


def one_body_density_vo(rho, t1, t2, l1, l2, o, v):
    rho[v, o] += l1

    return rho


def one_body_density(rho, t1, t2, l1, l2, o, v):
    rho = one_body_density_oo(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_vv(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_ov(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_vo(rho, t1, t2, l1, l2, o, v)

    return rho


def two_body_density_oooo(rho, t1, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, o, o, o] += np.einsum("il,ak,aj->ijkl", I, l1, t1, optimize=True)

    rho[o, o, o, o] += np.einsum("jk,al,ai->ijkl", I, l1, t1, optimize=True)

    rho[o, o, o, o] -= np.einsum("jl,ak,ai->ijkl", I, l1, t1, optimize=True)

    rho[o, o, o, o] -= np.einsum("ik,al,aj->ijkl", I, l1, t1, optimize=True)

    rho[o, o, o, o] += np.einsum("ik,jl->ijkl", I, I, optimize=True)

    rho[o, o, o, o] -= np.einsum("il,jk->ijkl", I, I, optimize=True)

    rho[o, o, o, o] += np.einsum("abkl,abij->ijkl", l2, t2, optimize=True) / 2

    rho[o, o, o, o] += np.einsum("ai,bj,abkl->ijkl", t1, t1, l2, optimize=True)

    rho[o, o, o, o] += np.einsum("jl,abmk,abim->ijkl", I, l2, t2, optimize=True) / 2

    rho[o, o, o, o] += np.einsum("il,abmk,abmj->ijkl", I, l2, t2, optimize=True) / 2

    rho[o, o, o, o] -= np.einsum("jk,abml,abim->ijkl", I, l2, t2, optimize=True) / 2

    rho[o, o, o, o] -= np.einsum("ik,abml,abmj->ijkl", I, l2, t2, optimize=True) / 2

    return rho


def two_body_density_vvvv(rho, t1, t2, l1, l2, o, v):
    rho[v, v, v, v] += np.einsum("abij,cdij->abcd", l2, t2, optimize=True) / 2

    rho[v, v, v, v] += np.einsum("ci,dj,abij->abcd", t1, t1, l2, optimize=True)

    return rho


def two_body_density_oovv(rho, t1, t2, l1, l2, o, v):
    rho[o, o, v, v] += np.einsum("ck,ak,bcij->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] += np.einsum("ck,bi,ackj->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] += np.einsum("ck,bj,acik->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("ck,ai,bckj->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("ck,aj,bcik->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("ck,bk,acij->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("ck,ci,abkj->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("ck,cj,abik->ijab", l1, t1, t2, optimize=True)

    rho[o, o, v, v] += np.einsum("ck,aj,bk,ci->ijab", l1, t1, t1, t1, optimize=True)

    rho[o, o, v, v] += np.einsum("ck,ak,bi,cj->ijab", l1, t1, t1, t1, optimize=True)

    rho[o, o, v, v] -= np.einsum("ck,ai,bk,cj->ijab", l1, t1, t1, t1, optimize=True)

    rho[o, o, v, v] -= np.einsum("ck,ak,bj,ci->ijab", l1, t1, t1, t1, optimize=True)

    rho[o, o, v, v] += np.einsum("ai,bj->ijab", t1, t1, optimize=True)

    rho[o, o, v, v] -= np.einsum("aj,bi->ijab", t1, t1, optimize=True)

    rho[o, o, v, v] += np.einsum("abij->ijab", t2, optimize=True)

    rho[o, o, v, v] += np.einsum("cdkl,ackj,bdil->ijab", l2, t2, t2, optimize=True)

    rho[o, o, v, v] += np.einsum("cdkl,ablj,cdik->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("cdkl,adij,bckl->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("cdkl,acik,bdlj->ijab", l2, t2, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("cdkl,abil,cdkj->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("cdkl,ackl,bdij->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("cdkl,abkl,cdij->ijab", l2, t2, t2, optimize=True) / 4

    rho[o, o, v, v] += np.einsum("ak,ci,cdkl,bdlj->ijab", t1, t1, l2, t2, optimize=True)

    rho[o, o, v, v] += np.einsum("ak,cj,cdkl,bdil->ijab", t1, t1, l2, t2, optimize=True)

    rho[o, o, v, v] += np.einsum("ai,bk,cdkl,cdlj->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("aj,bk,cdkl,cdil->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("aj,ci,cdkl,bdkl->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("ak,bl,cdkl,cdij->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("bi,cj,cdkl,adkl->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("ci,dj,cdkl,abkl->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("bk,ci,cdkl,adlj->ijab", t1, t1, l2, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("bk,cj,cdkl,adil->ijab", t1, t1, l2, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("ai,cj,cdkl,bdkl->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("ak,bi,cdkl,cdlj->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("ak,bj,cdkl,cdil->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("bj,ci,cdkl,adkl->ijab", t1, t1, l2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("ak,bl,ci,dj,cdkl->ijab", t1, t1, t1, t1, l2, optimize=True)

    return rho


def two_body_density_vvoo(rho, t1, t2, l1, l2, o, v):
    rho[v, v, o, o] += np.einsum("abij->abij", l2, optimize=True)

    return rho


def two_body_density_ovov(rho, t1, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, v, o, v] += np.einsum("ackj,bcik->iajb", l2, t2, optimize=True)

    rho[o, v, o, v] -= np.einsum("bk,ci,ackj->iajb", t1, t1, l2, optimize=True)

    rho[o, v, o, v] -= np.einsum("aj,bi->iajb", l1, t1, optimize=True)

    rho[o, v, o, v] += np.einsum("ij,ackl,bckl->iajb", I, l2, t2, optimize=True) / 2

    rho[o, v, o, v] += np.einsum("ij,ak,bk->iajb", I, l1, t1, optimize=True)

    rho[o, v, v, o] = -rho[o, v, o, v].transpose(0, 1, 3, 2)
    rho[v, o, o, v] = -rho[o, v, o, v].transpose(1, 0, 2, 3)
    rho[v, o, v, o] = rho[o, v, o, v].transpose(1, 0, 3, 2)

    return rho


def two_body_density_ooov(rho, t1, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, o, o, v] -= np.einsum("jk,bl,abil->ijka", I, l1, t2, optimize=True)

    rho[o, o, o, v] -= np.einsum("ik,bl,ablj->ijka", I, l1, t2, optimize=True)

    rho[o, o, o, v] += np.einsum("jk,bl,al,bi->ijka", I, l1, t1, t1, optimize=True)

    rho[o, o, o, v] -= np.einsum("ik,bl,al,bj->ijka", I, l1, t1, t1, optimize=True)

    rho[o, o, o, v] += np.einsum("jk,al,bclm,bcim->ijka", I, t1, l2, t2, optimize=True) / 2

    rho[o, o, o, v] += np.einsum("ik,al,bclm,bcmj->ijka", I, t1, l2, t2, optimize=True) / 2

    rho[o, o, o, v] += np.einsum("jk,bi,bclm,aclm->ijka", I, t1, l2, t2, optimize=True) / 2

    rho[o, o, o, v] -= np.einsum("ik,bj,bclm,aclm->ijka", I, t1, l2, t2, optimize=True) / 2

    rho[o, o, o, v] += np.einsum("ik,aj->ijka", I, t1, optimize=True)

    rho[o, o, o, v] -= np.einsum("jk,ai->ijka", I, t1, optimize=True)

    rho[o, o, o, v] += np.einsum("ai,bclk,bclj->ijka", t1, l2, t2, optimize=True) / 2

    rho[o, o, o, v] += np.einsum("aj,bclk,bcil->ijka", t1, l2, t2, optimize=True) / 2

    rho[o, o, o, v] -= np.einsum("bi,bclk,aclj->ijka", t1, l2, t2, optimize=True)

    rho[o, o, o, v] -= np.einsum("bj,bclk,acil->ijka", t1, l2, t2, optimize=True)

    rho[o, o, o, v] -= np.einsum("al,bclk,bcij->ijka", t1, l2, t2, optimize=True) / 2

    rho[o, o, o, v] -= np.einsum("al,bi,cj,bclk->ijka", t1, t1, t1, l2, optimize=True)

    rho[o, o, o, v] += np.einsum("bk,abij->ijka", l1, t2, optimize=True)

    rho[o, o, o, v] += np.einsum("bk,ai,bj->ijka", l1, t1, t1, optimize=True)

    rho[o, o, o, v] -= np.einsum("bk,aj,bi->ijka", l1, t1, t1, optimize=True)

    rho[o, o, v, o] = -rho[o, o, o, v].transpose(0, 1, 3, 2)

    return rho


def two_body_density_ovoo(rho, t1, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, v, o, o] += np.einsum("ij,ak->iajk", I, l1, optimize=True)

    rho[o, v, o, o] -= np.einsum("ik,aj->iajk", I, l1, optimize=True)

    rho[o, v, o, o] += np.einsum("bi,abjk->iajk", t1, l2, optimize=True)

    rho[v, o, o, o] = -rho[o, v, o, o].transpose(1, 0, 2, 3)

    return rho


def two_body_density_vvvo(rho, t1, t2, l1, l2, o, v):
    rho[v, v, v, o] -= np.einsum("cj,abij->abci", t1, l2, optimize=True)

    rho[v, v, o, v] = -rho[v, v, v, o].transpose(0, 1, 3, 2)

    return rho


def two_body_density_vovv(rho, t1, t2, l1, l2, o, v):
    rho[v, o, v, v] += np.einsum("bj,adjk,cdik->aibc", t1, l2, t2, optimize=True)

    rho[v, o, v, v] += np.einsum("ci,adjk,bdjk->aibc", t1, l2, t2, optimize=True) / 2

    rho[v, o, v, v] -= np.einsum("cj,adjk,bdik->aibc", t1, l2, t2, optimize=True)

    rho[v, o, v, v] -= np.einsum("bi,adjk,cdjk->aibc", t1, l2, t2, optimize=True) / 2

    rho[v, o, v, v] -= np.einsum("di,adjk,bcjk->aibc", t1, l2, t2, optimize=True) / 2

    rho[v, o, v, v] -= np.einsum("bj,ck,di,adjk->aibc", t1, t1, t1, l2, optimize=True)

    rho[v, o, v, v] -= np.einsum("aj,bcij->aibc", l1, t2, optimize=True)

    rho[v, o, v, v] += np.einsum("aj,bj,ci->aibc", l1, t1, t1, optimize=True)

    rho[v, o, v, v] -= np.einsum("aj,bi,cj->aibc", l1, t1, t1, optimize=True)

    rho[o, v, v, v] = -rho[v, o, v, v].transpose(1, 0, 2, 3)

    return rho


def two_body_density(rho, t1, t2, l1, l2, o, v):
    rho = two_body_density_oooo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vvvv(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_oovv(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vvoo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_ovov(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_ooov(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_ovoo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vvvo(rho, t1, t2, l1, l2, o, v)
    rho = two_body_density_vovv(rho, t1, t2, l1, l2, o, v)

    return rho
