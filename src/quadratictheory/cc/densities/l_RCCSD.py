import numpy as np


def one_body_density_oo(rho, t1, t2, l1, l2, o, v):
    N = o.stop

    rho[o, o] += 2 * np.eye(N, dtype=rho.dtype)
    rho[o, o] -= np.einsum("aj,ai->ij", l1, t1)
    rho[o, o] -= 2 * np.einsum("abkj,baik->ij", l2, t2)

    return rho


def one_body_density_vv(rho, t1, t2, l1, l2, o, v):
    rho[v, v] += np.einsum("ai,bi->ab", l1, t1)
    rho[v, v] += 2 * np.einsum("acij,bcij->ab", l2, t2)

    return rho


def one_body_density_ov(rho, t1, t2, l1, l2, o, v):
    rho[o, v] += 2 * t1.T
    rho[o, v] -= 2 * np.einsum("aj,bcjk,bcik->ia", t1, l2, t2, optimize=True)
    rho[o, v] -= 2 * np.einsum("bi,bcjk,acjk->ia", t1, l2, t2, optimize=True)
    rho[o, v] += 2 * np.einsum("bj,abij->ia", l1, t2)
    rho[o, v] -= np.einsum("bj,abji->ia", l1, t2)
    rho[o, v] -= np.einsum("bj,aj,bi->ia", l1, t1, t1)

    return rho


def one_body_density_vo(rho, t1, t2, l1, l2, o, v):
    rho[v, o] += l1

    return rho


def one_body_density_restricted(rho, t1, t2, l1, l2, o, v):
    rho = one_body_density_oo(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_vv(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_ov(rho, t1, t2, l1, l2, o, v)
    rho = one_body_density_vo(rho, t1, t2, l1, l2, o, v)

    return rho


"""
Borrowed from
https://raw.githubusercontent.com/HyQD/coupled-cluster/refs/heads/master/coupled_cluster/rccsd/density_matrices.py
"""
def two_body_density_restricted(rho, t1, t2, l1, l2, o, v):    
    l1 = l1.transpose(1,0)
    l2 = l2.transpose(2,3,0,1)
    
    add_rho_klij(rho, t1, t2, l1, l2, o, v)

    add_rho_kaij(rho, t1, t2, l1, l2, o, v)
    add_rho_akij(rho, t1, t2, l1, l2, o, v)
    add_rho_jkai(rho, t1, t2, l1, l2, o, v)
    add_rho_jkia(rho, t1, t2, l1, l2, o, v)

    add_rho_ijab(rho, t1, t2, l1, l2, o, v)
    add_rho_abij(rho, t1, t2, l1, l2, o, v)

    add_rho_jbia(rho, t1, t2, l1, l2, o, v)
    add_rho_bjai(rho, t1, t2, l1, l2, o, v)

    add_rho_bjia(rho, t1, t2, l1, l2, o, v)
    add_rho_jbai(rho, t1, t2, l1, l2, o, v)

    add_rho_bcai(rho, t1, t2, l1, l2, o, v)
    add_rho_bcia(rho, t1, t2, l1, l2, o, v)
    add_rho_ciab(rho, t1, t2, l1, l2, o, v)
    add_rho_icab(rho, t1, t2, l1, l2, o, v)

    add_rho_cdab(rho, t1, t2, l1, l2, o, v)

    l1 = l1.transpose(1,0)
    l2 = l2.transpose(2,3,0,1)

    return rho


def add_rho_klij(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[o, o, o, o] -= 2 * np.einsum("il, jk->klij", delta, delta)

    rho[o, o, o, o] += 4 * np.einsum("ik, jl->klij", delta, delta)

    rho[o, o, o, o] += np.einsum("il, mkab,abmj->klij", delta, l2, t2)

    rho[o, o, o, o] += np.einsum("jk, mlab,baim->klij", delta, l2, t2)

    rho[o, o, o, o] -= 2 * np.einsum("jl, mkab,baim->klij", delta, l2, t2)

    rho[o, o, o, o] -= 2 * np.einsum("ik, mlab,abmj->klij", delta, l2, t2)

    rho[o, o, o, o] += np.einsum("il, ka,aj->klij", delta, l1, t1)

    rho[o, o, o, o] += np.einsum("jk, la,ai->klij", delta, l1, t1)

    rho[o, o, o, o] -= 2 * np.einsum("jl, ka,ai->klij", delta, l1, t1)

    rho[o, o, o, o] -= 2 * np.einsum("ik, la,aj->klij", delta, l1, t1)

    rho[o, o, o, o] += np.einsum("klab,abij->klij", l2, t2)

    rho[o, o, o, o] += np.einsum("ai,bj,klab->klij", t1, t1, l2)


def add_rho_kaij(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[o, v, o, o] += np.einsum("ai,lkbc,bclj->kaij", t1, l2, t2)

    rho[o, v, o, o] += np.einsum("al,lkbc,cbij->kaij", t1, l2, t2)

    rho[o, v, o, o] += np.einsum("bj,lkbc,acli->kaij", t1, l2, t2)

    rho[o, v, o, o] += np.einsum("ci,lkbc,ablj->kaij", t1, l2, t2)

    rho[o, v, o, o] += np.einsum("cj,lkbc,abil->kaij", t1, l2, t2)

    rho[o, v, o, o] -= 2 * np.einsum("aj,lkbc,cbil->kaij", t1, l2, t2)

    rho[o, v, o, o] -= 2 * np.einsum("ci,lkbc,abjl->kaij", t1, l2, t2)

    rho[o, v, o, o] += np.einsum("al,bj,ci,lkbc->kaij", t1, t1, t1, l2)

    rho[o, v, o, o] += np.einsum("kb,abij->kaij", l1, t2)

    rho[o, v, o, o] -= 2 * np.einsum("kb,abji->kaij", l1, t2)

    rho[o, v, o, o] += np.einsum("kb,ai,bj->kaij", l1, t1, t1)

    rho[o, v, o, o] -= 2 * np.einsum("kb,aj,bi->kaij", l1, t1, t1)

    rho[o, v, o, o] += np.einsum("jk, lb,abli->kaij", delta, l1, t2)

    rho[o, v, o, o] -= 2 * np.einsum("jk, lb,abil->kaij", delta, l1, t2)

    rho[o, v, o, o] -= 2 * np.einsum("ik, lb,ablj->kaij", delta, l1, t2)

    rho[o, v, o, o] += 4 * np.einsum("ik, lb,abjl->kaij", delta, l1, t2)

    rho[o, v, o, o] += np.einsum("jk, lb,al,bi->kaij", delta, l1, t1, t1)

    rho[o, v, o, o] -= 2 * np.einsum("ik, lb,al,bj->kaij", delta, l1, t1, t1)

    rho[o, v, o, o] -= 2 * np.einsum("jk, ai->kaij", delta, t1)

    rho[o, v, o, o] += 4 * np.einsum("ik, aj->kaij", delta, t1)

    rho[o, v, o, o] += np.einsum("jk, al,lmbc,bcim->kaij", delta, t1, l2, t2)

    rho[o, v, o, o] += np.einsum("jk, bi,lmbc,aclm->kaij", delta, t1, l2, t2)

    rho[o, v, o, o] -= 2 * np.einsum("ik, al,lmcb,bcmj->kaij", delta, t1, l2, t2)

    rho[o, v, o, o] -= 2 * np.einsum("ik, bj,lmbc,aclm->kaij", delta, t1, l2, t2)


def add_rho_akij(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[v, o, o, o] += np.einsum("aj,lkbc,cbil->akij", t1, l2, t2)

    rho[v, o, o, o] += np.einsum("al,lkbc,bcij->akij", t1, l2, t2)

    rho[v, o, o, o] += np.einsum("bi,lkbc,aclj->akij", t1, l2, t2)

    rho[v, o, o, o] += np.einsum("ci,lkbc,abjl->akij", t1, l2, t2)

    rho[v, o, o, o] += np.einsum("cj,lkbc,abli->akij", t1, l2, t2)

    rho[v, o, o, o] -= 2 * np.einsum("ai,lkbc,bclj->akij", t1, l2, t2)

    rho[v, o, o, o] -= 2 * np.einsum("cj,lkbc,abil->akij", t1, l2, t2)

    rho[v, o, o, o] += np.einsum("al,bi,cj,lkbc->akij", t1, t1, t1, l2)

    rho[v, o, o, o] += np.einsum("kb,abji->akij", l1, t2)

    rho[v, o, o, o] -= 2 * np.einsum("kb,abij->akij", l1, t2)

    rho[v, o, o, o] += np.einsum("kb,aj,bi->akij", l1, t1, t1)

    rho[v, o, o, o] -= 2 * np.einsum("kb,ai,bj->akij", l1, t1, t1)

    rho[v, o, o, o] += np.einsum("ik, lb,ablj->akij", delta, l1, t2)

    rho[v, o, o, o] -= 2 * np.einsum("ik, lb,abjl->akij", delta, l1, t2)

    rho[v, o, o, o] -= 2 * np.einsum("jk, lb,abli->akij", delta, l1, t2)

    rho[v, o, o, o] += 4 * np.einsum("jk, lb,abil->akij", delta, l1, t2)

    rho[v, o, o, o] += np.einsum("ik, lb,al,bj->akij", delta, l1, t1, t1)

    rho[v, o, o, o] -= 2 * np.einsum("jk, lb,al,bi->akij", delta, l1, t1, t1)

    rho[v, o, o, o] -= 2 * np.einsum("ik, aj->akij", delta, t1)

    rho[v, o, o, o] += 4 * np.einsum("jk, ai->akij", delta, t1)

    rho[v, o, o, o] += np.einsum("ik, al,lmcb,bcmj->akij", delta, t1, l2, t2)

    rho[v, o, o, o] += np.einsum("ik, bj,lmbc,aclm->akij", delta, t1, l2, t2)

    rho[v, o, o, o] -= 2 * np.einsum("jk, al,lmbc,bcim->akij", delta, t1, l2, t2)

    rho[v, o, o, o] -= 2 * np.einsum("jk, bi,lmbc,aclm->akij", delta, t1, l2, t2)


def add_rho_jkai(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[o, o, v, o] -= np.einsum("ij, ka->jkai", delta, l1)

    rho[o, o, v, o] += 2 * np.einsum("ik, ja->jkai", delta, l1)

    rho[o, o, v, o] -= np.einsum("bi,jkab->jkai", t1, l2)


def add_rho_jkia(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[o, o, o, v] -= np.einsum("ik, ja->jkia", delta, l1)

    rho[o, o, o, v] += 2 * np.einsum("ij, ka->jkia", delta, l1)

    rho[o, o, o, v] -= np.einsum("bi,jkba->jkia", t1, l2)


def add_rho_ijab(rho, t1, t2, l1, l2, o, v):
    rho[o, o, v, v] += l2


def add_rho_abij(rho, t1, t2, l1, l2, o, v):
    rho[v, v, o, o] -= 2 * np.einsum("abji->abij", t2)

    rho[v, v, o, o] += 4 * np.einsum("abij->abij", t2)

    rho[v, v, o, o] -= 2 * np.einsum("aj,bi->abij", t1, t1)

    rho[v, v, o, o] += 4 * np.einsum("ai,bj->abij", t1, t1)

    rho[v, v, o, o] += np.einsum("kc,aj,bcki->abij", l1, t1, t2)

    rho[v, v, o, o] += np.einsum("kc,ak,bcij->abij", l1, t1, t2)

    rho[v, v, o, o] += np.einsum("kc,bi,ackj->abij", l1, t1, t2)

    rho[v, v, o, o] += np.einsum("kc,bk,acji->abij", l1, t1, t2)

    rho[v, v, o, o] += np.einsum("kc,ci,abjk->abij", l1, t1, t2)

    rho[v, v, o, o] += np.einsum("kc,cj,abki->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,ai,bckj->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,aj,bcik->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,ak,bcji->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,bi,acjk->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,bj,acki->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,bk,acij->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,ci,abkj->abij", l1, t1, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kc,cj,abik->abij", l1, t1, t2)

    rho[v, v, o, o] += 4 * np.einsum("kc,ai,bcjk->abij", l1, t1, t2)

    rho[v, v, o, o] += 4 * np.einsum("kc,bj,acik->abij", l1, t1, t2)

    rho[v, v, o, o] += np.einsum("kc,aj,bk,ci->abij", l1, t1, t1, t1)

    rho[v, v, o, o] += np.einsum("kc,ak,bi,cj->abij", l1, t1, t1, t1)

    rho[v, v, o, o] -= 2 * np.einsum("kc,ai,bk,cj->abij", l1, t1, t1, t1)

    rho[v, v, o, o] -= 2 * np.einsum("kc,ak,bj,ci->abij", l1, t1, t1, t1)

    rho[v, v, o, o] += np.einsum("klcd,abjk,cdil->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("klcd,abkl,cdij->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("klcd,acji,bdkl->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("klcd,acjk,bdli->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("klcd,acki,bdlj->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("klcd,ackj,bdil->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("kldc,abki,cdlj->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("kldc,ackj,bdli->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("kldc,ackl,bdij->abij", l2, t2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("klcd,abkj,cdil->abij", l2, t2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("klcd,acij,bdkl->abij", l2, t2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("klcd,acik,bdlj->abij", l2, t2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("klcd,acjk,bdil->abij", l2, t2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("klcd,acki,bdjl->abij", l2, t2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kldc,abik,cdlj->abij", l2, t2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("kldc,ackl,bdji->abij", l2, t2, t2)

    rho[v, v, o, o] += 4 * np.einsum("klcd,acik,bdjl->abij", l2, t2, t2)

    rho[v, v, o, o] += np.einsum("aj,bk,klcd,cdil->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("aj,ci,klcd,bdkl->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("ak,bi,kldc,cdlj->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("ak,bl,klcd,cdij->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("ak,ci,klcd,bdlj->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("ak,cj,klcd,bdil->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("ak,cj,kldc,bdli->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("bi,cj,klcd,adkl->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("bk,ci,klcd,adjl->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("bk,ci,kldc,adlj->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("bk,cj,klcd,adli->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("ci,dj,klcd,abkl->abij", t1, t1, l2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("ai,bk,kldc,cdlj->abij", t1, t1, l2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("ai,cj,klcd,bdkl->abij", t1, t1, l2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("ak,bj,klcd,cdil->abij", t1, t1, l2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("ak,ci,klcd,bdjl->abij", t1, t1, l2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("bj,ci,klcd,adkl->abij", t1, t1, l2, t2)

    rho[v, v, o, o] -= 2 * np.einsum("bk,cj,klcd,adil->abij", t1, t1, l2, t2)

    rho[v, v, o, o] += np.einsum("ak,bl,ci,dj,klcd->abij", t1, t1, t1, t1, l2)


def add_rho_jbia(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[o, v, o, v] -= np.einsum("ja,bi->jbia", l1, t1)

    rho[o, v, o, v] += 2 * np.einsum("ij, klac,bckl->jbia", delta, l2, t2)

    rho[o, v, o, v] += 2 * np.einsum("ij, ka,bk->jbia", delta, l1, t1)

    rho[o, v, o, v] -= np.einsum("kjac,bcki->jbia", l2, t2)

    rho[o, v, o, v] -= np.einsum("kjca,bcik->jbia", l2, t2)

    rho[o, v, o, v] -= np.einsum("bk,ci,kjac->jbia", t1, t1, l2)


def add_rho_bjai(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[v, o, v, o] -= np.einsum("ja,bi->bjai", l1, t1)

    rho[v, o, v, o] += 2 * np.einsum("ij, klac,bckl->bjai", delta, l2, t2)

    rho[v, o, v, o] += 2 * np.einsum("ij, ka,bk->bjai", delta, l1, t1)

    rho[v, o, v, o] -= np.einsum("kjac,bcki->bjai", l2, t2)

    rho[v, o, v, o] -= np.einsum("kjca,bcik->bjai", l2, t2)

    rho[v, o, v, o] -= np.einsum("bk,ci,kjac->bjai", t1, t1, l2)


def add_rho_bjia(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[v, o, o, v] += 2 * np.einsum("ja,bi->bjia", l1, t1)

    rho[v, o, o, v] -= np.einsum("ij, klac,bckl->bjia", delta, l2, t2)

    rho[v, o, o, v] -= np.einsum("ij, ka,bk->bjia", delta, l1, t1)

    rho[v, o, o, v] -= np.einsum("kjca,bcki->bjia", l2, t2)

    rho[v, o, o, v] += 2 * np.einsum("kjca,bcik->bjia", l2, t2)

    rho[v, o, o, v] -= np.einsum("bk,ci,kjca->bjia", t1, t1, l2)


def add_rho_jbai(rho, t1, t2, l1, l2, o, v):
    delta = np.eye(o.stop)

    rho[o, v, v, o] += 2 * np.einsum("ja,bi->jbai", l1, t1)

    rho[o, v, v, o] -= np.einsum("ij, klac,bckl->jbai", delta, l2, t2)

    rho[o, v, v, o] -= np.einsum("ij, ka,bk->jbai", delta, l1, t1)

    rho[o, v, v, o] -= np.einsum("kjca,bcki->jbai", l2, t2)

    rho[o, v, v, o] += 2 * np.einsum("kjca,bcik->jbai", l2, t2)

    rho[o, v, v, o] -= np.einsum("bk,ci,kjca->jbai", t1, t1, l2)


def add_rho_bcai(rho, t1, t2, l1, l2, o, v):
    rho[v, v, v, o] -= np.einsum("bi,jkad,cdjk->bcai", t1, l2, t2)

    rho[v, v, v, o] -= np.einsum("bj,jkad,cdki->bcai", t1, l2, t2)

    rho[v, v, v, o] -= np.einsum("cj,jkad,bdik->bcai", t1, l2, t2)

    rho[v, v, v, o] -= np.einsum("ck,jkad,bdji->bcai", t1, l2, t2)

    rho[v, v, v, o] -= np.einsum("di,jkad,bcjk->bcai", t1, l2, t2)

    rho[v, v, v, o] += 2 * np.einsum("bj,jkad,cdik->bcai", t1, l2, t2)

    rho[v, v, v, o] += 2 * np.einsum("ci,jkad,bdjk->bcai", t1, l2, t2)

    rho[v, v, v, o] -= np.einsum("bj,ck,di,jkad->bcai", t1, t1, t1, l2)

    rho[v, v, v, o] -= np.einsum("ja,bcij->bcai", l1, t2)

    rho[v, v, v, o] += 2 * np.einsum("ja,bcji->bcai", l1, t2)

    rho[v, v, v, o] -= np.einsum("ja,bi,cj->bcai", l1, t1, t1)

    rho[v, v, v, o] += 2 * np.einsum("ja,bj,ci->bcai", l1, t1, t1)


def add_rho_bcia(rho, t1, t2, l1, l2, o, v):
    rho[v, v, o, v] -= np.einsum("bj,jkad,cdik->bcia", t1, l2, t2)

    rho[v, v, o, v] -= np.einsum("bk,jkad,cdji->bcia", t1, l2, t2)

    rho[v, v, o, v] -= np.einsum("ci,jkad,bdjk->bcia", t1, l2, t2)

    rho[v, v, o, v] -= np.einsum("cj,jkad,bdki->bcia", t1, l2, t2)

    rho[v, v, o, v] -= np.einsum("di,jkad,bckj->bcia", t1, l2, t2)

    rho[v, v, o, v] += 2 * np.einsum("bi,jkad,cdjk->bcia", t1, l2, t2)

    rho[v, v, o, v] += 2 * np.einsum("cj,jkad,bdik->bcia", t1, l2, t2)

    rho[v, v, o, v] -= np.einsum("bk,cj,di,jkad->bcia", t1, t1, t1, l2)

    rho[v, v, o, v] -= np.einsum("ja,bcji->bcia", l1, t2)

    rho[v, v, o, v] += 2 * np.einsum("ja,bcij->bcia", l1, t2)

    rho[v, v, o, v] -= np.einsum("ja,bj,ci->bcia", l1, t1, t1)

    rho[v, v, o, v] += 2 * np.einsum("ja,bi,cj->bcia", l1, t1, t1)


def add_rho_icab(rho, t1, t2, l1, l2, o, v):
    rho[o, v, v, v] += np.einsum("cj,ijab->icab", t1, l2)


def add_rho_ciab(rho, t1, t2, l1, l2, o, v):
    rho[v, o, v, v] += np.einsum("cj,ijba->ciab", t1, l2)


def add_rho_cdab(rho, t1, t2, l1, l2, o, v):
    rho[v, v, v, v] += np.einsum("ijab,cdij->cdab", l2, t2)

    rho[v, v, v, v] += np.einsum("ci,dj,ijab->cdab", t1, t1, l2)

