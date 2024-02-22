import numpy as np


def one_body_density_oo(rho, t2, l1, l2, o, v):
    N = o.stop

    rho[o, o] += np.eye(N, dtype=rho.dtype)
    rho[o, o] += np.einsum("abkj,abik->ij", l2, t2) / 2

    return rho


def one_body_density_vv(rho, t2, l1, l2, o, v):
    rho[v, v] += np.einsum("acij,bcij->ab", l2, t2) / 2

    return rho


def one_body_density_ov(rho, t2, l1, l2, o, v):
    rho[o, v] += np.einsum("bj,abij->ia", l1, t2)

    return rho


def one_body_density_vo(rho, t2, l1, l2, o, v):
    rho[v, o] += l1

    return rho


def one_body_density(rho, t2, l1, l2, o, v):
    rho = one_body_density_oo(rho, t2, l1, l2, o, v)
    rho = one_body_density_vv(rho, t2, l1, l2, o, v)
    rho = one_body_density_ov(rho, t2, l1, l2, o, v)
    rho = one_body_density_vo(rho, t2, l1, l2, o, v)

    return rho


def two_body_density_oooo(rho, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, o, o, o] += np.einsum("ik,jl->ijkl", I, I, optimize=True)

    rho[o, o, o, o] -= np.einsum("il,jk->ijkl", I, I, optimize=True)

    rho[o, o, o, o] += np.einsum("abkl,abij->ijkl", l2, t2, optimize=True) / 2

    rho[o, o, o, o] += np.einsum("jl,abmk,abim->ijkl", I, l2, t2, optimize=True) / 2

    rho[o, o, o, o] += np.einsum("il,abmk,abmj->ijkl", I, l2, t2, optimize=True) / 2

    rho[o, o, o, o] -= np.einsum("jk,abml,abim->ijkl", I, l2, t2, optimize=True) / 2

    rho[o, o, o, o] -= np.einsum("ik,abml,abmj->ijkl", I, l2, t2, optimize=True) / 2

    return rho


def two_body_density_vvvv(rho, t2, l1, l2, o, v):
    rho[v, v, v, v] += np.einsum("abij,cdij->abcd", l2, t2, optimize=True) / 2

    return rho


def two_body_density_oovv(rho, t2, l1, l2, o, v):
    rho[o, o, v, v] += np.einsum("abij->ijab", t2, optimize=True)

    rho[o, o, v, v] += np.einsum("cdkl,ackj,bdil->ijab", l2, t2, t2, optimize=True)

    rho[o, o, v, v] += np.einsum("cdkl,ablj,cdik->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("cdkl,adij,bckl->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("cdkl,acik,bdlj->ijab", l2, t2, t2, optimize=True)

    rho[o, o, v, v] -= np.einsum("cdkl,abil,cdkj->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] -= np.einsum("cdkl,ackl,bdij->ijab", l2, t2, t2, optimize=True) / 2

    rho[o, o, v, v] += np.einsum("cdkl,abkl,cdij->ijab", l2, t2, t2, optimize=True) / 4

    return rho


def two_body_density_vvoo(rho, t2, l1, l2, o, v):
    rho[v, v, o, o] += np.einsum("abij->abij", l2, optimize=True)

    return rho


def two_body_density_ovov(rho, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, v, o, v] += np.einsum("ackj,bcik->iajb", l2, t2, optimize=True)

    rho[o, v, o, v] += np.einsum("ij,ackl,bckl->iajb", I, l2, t2, optimize=True) / 2

    rho[o, v, v, o] = -rho[o, v, o, v].transpose(0, 1, 3, 2)
    rho[v, o, o, v] = -rho[o, v, o, v].transpose(1, 0, 2, 3)
    rho[v, o, v, o] = rho[o, v, o, v].transpose(1, 0, 3, 2)

    return rho


def two_body_density_ooov(rho, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, o, o, v] -= np.einsum("jk,bl,abil->ijka", I, l1, t2, optimize=True)

    rho[o, o, o, v] -= np.einsum("ik,bl,ablj->ijka", I, l1, t2, optimize=True)

    rho[o, o, o, v] += np.einsum("bk,abij->ijka", l1, t2, optimize=True)

    rho[o, o, v, o] = -rho[o, o, o, v].transpose(0, 1, 3, 2)

    return rho


def two_body_density_ovoo(rho, t2, l1, l2, o, v):
    I = np.eye(o.stop, dtype=rho.dtype)

    rho[o, v, o, o] += np.einsum("ij,ak->iajk", I, l1, optimize=True)

    rho[o, v, o, o] -= np.einsum("ik,aj->iajk", I, l1, optimize=True)

    rho[v, o, o, o] = -rho[o, v, o, o].transpose(1, 0, 2, 3)

    return rho

def two_body_density_vovv(rho, t2, l1, l2, o, v):
    rho[v, o, v, v] -= np.einsum("aj,bcij->aibc", l1, t2, optimize=True)

    rho[o, v, v, v] = -rho[v, o, v, v].transpose(1, 0, 2, 3)

    return rho


def two_body_density(rho, t2, l1, l2, o, v):
    rho = two_body_density_oooo(rho, t2, l1, l2, o, v)
    rho = two_body_density_vvvv(rho, t2, l1, l2, o, v)
    rho = two_body_density_oovv(rho, t2, l1, l2, o, v)
    rho = two_body_density_vvoo(rho, t2, l1, l2, o, v)
    rho = two_body_density_ovov(rho, t2, l1, l2, o, v)
    rho = two_body_density_ooov(rho, t2, l1, l2, o, v)
    rho = two_body_density_ovoo(rho, t2, l1, l2, o, v)
    rho = two_body_density_vovv(rho, t2, l1, l2, o, v)

    return rho
