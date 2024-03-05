"""
Script to compute the electronic correlation energy using
coupled-cluster theory through single and double excitations,
from a RHF reference wavefunction.

References:
- Algorithms from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects
- DPD Formulation of CC Equations: [Stanton:1991:4334]
"""
import numpy as np

# DPD approach to CCSD equations from [Stanton:1991:4334]

# occ orbitals i, j, k, l, m, n
# virt orbitals a, b, c, d, e, f
# all oribitals p, q, r, s, t, u, v


#Bulid Eqn 9: tilde{\Tau})
def build_tilde_tau(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 9"""
    ttau = t2.copy()
    tmp = 0.5 * np.einsum('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 10: \Tau)
def build_tau(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 10"""
    ttau = t2.copy()
    tmp = np.einsum('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 3:
def build_Fae(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 3"""
    Fae = f[v, v].copy()
    Fae[np.diag_indices_from(Fae)] = 0

    Fae -= 0.5 * np.einsum('me,ma->ae', f[o, v], t1)
    Fae += np.einsum('mf,mafe->ae', t1, u[o, v, v, v])

    tmp_tau = build_tilde_tau(t1, t2, f, u, o, v)
    Fae -= 0.5 * np.einsum('mnaf,mnef->ae', tmp_tau, u[o, o, v, v])
    return Fae


#Build Eqn 4:
def build_Fmi(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 4"""
    Fmi = f[o, o].copy()
    Fmi[np.diag_indices_from(Fmi)] = 0

    Fmi += 0.5 * np.einsum('ie,me->mi', t1, f[o, v])
    Fmi += np.einsum('ne,mnie->mi', t1, u[o, o, o, v])

    tmp_tau = build_tilde_tau(t1, t2, f, u, o, v)
    Fmi += 0.5 * np.einsum('inef,mnef->mi', tmp_tau, u[o, o, v, v])
    return Fmi


#Build Eqn 5:
def build_Fme(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 5"""
    Fme = f[o, v].copy()
    Fme += np.einsum('nf,mnef->me', t1, u[o, o, v, v])
    return Fme


#Build Eqn 6:
def build_Wmnij(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 6"""
    Wmnij = u[o, o, o, o].copy()

    Pij = np.einsum('je,mnie->mnij', t1, u[o, o, o, v])
    Wmnij += Pij
    Wmnij -= Pij.swapaxes(2, 3)

    tmp_tau = build_tau(t1, t2, f, u, o, v)
    Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tmp_tau, u[o, o, v, v])
    return Wmnij


#Build Eqn 7:
def build_Wabef(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 7"""
    # Rate limiting step written using tensordot, ~10x faster
    # The commented out lines are consistent with the paper

    Wabef = u[v, v, v, v].copy()

    Pab = np.einsum('baef->abef', np.tensordot(t1, u[v, o, v, v], axes=(0, 1)))
    # Pab = np.einsum('mb,amef->abef', t1, u[v, o, v, v])

    Wabef -= Pab
    Wabef += Pab.swapaxes(0, 1)

    tmp_tau = build_tau(t1, t2, f, u, o, v)

    Wabef += 0.25 * np.tensordot(tmp_tau, u[v, v, o, o], axes=((0, 1), (2, 3)))
    # Wabef += 0.25 * np.einsum('mnab,mnef->abef', tmp_tau, u[o, o, v, v])
    return Wabef


#Build Eqn 8:
def build_Wmbej(t1, t2, f, u, o, v):
    """Builds [Stanton:1991:4334] Eqn. 8"""
    Wmbej = u[o, v, v, o].copy()
    Wmbej += np.einsum('jf,mbef->mbej', t1, u[o, v, v, v])
    Wmbej -= np.einsum('nb,mnej->mbej', t1, u[o, o, v, o])

    tmp = (0.5 * t2) + np.einsum('jf,nb->jnfb', t1, t1)

    Wmbej -= np.einsum('jbme->mbej', np.tensordot(tmp, u[o, o, v, v], axes=((1, 2), (1, 3))))
    # Wmbej -= np.einsum('jnfb,mnef->mbej', tmp, u[o, o, v, v])
    return Wmbej


def ccsd_t_Gauss_Stanton(t1, t2, u, f, o, v):
    t1 = t1.T
    t2 = t2.transpose(2,3,0,1)
    ### Build intermediates: [Stanton:1991:4334] Eqns. 3-8
    Fae = build_Fae(t1, t2, f, u, o, v)
    Fmi = build_Fmi(t1, t2, f, u, o, v)
    Fme = build_Fme(t1, t2, f, u, o, v)

    Wmnij = build_Wmnij(t1, t2, f, u, o, v)
    Wabef = build_Wabef(t1, t2, f, u, o, v)
    Wmbej = build_Wmbej(t1, t2, f, u, o, v)

    #### Build RHS side of t1 equations, [Stanton:1991:4334] Eqn. 1
    rhs_T1  = f[o, v].copy()
    rhs_T1 += np.einsum('ie,ae->ia', t1, Fae)
    rhs_T1 -= np.einsum('ma,mi->ia', t1, Fmi)
    rhs_T1 += np.einsum('imae,me->ia', t2, Fme)
    rhs_T1 -= np.einsum('nf,naif->ia', t1, u[o, v, o, v])
    rhs_T1 -= 0.5 * np.einsum('imef,maef->ia', t2, u[o, v, v, v])
    rhs_T1 -= 0.5 * np.einsum('mnae,nmei->ia', t2, u[o, o, v, o])

    ### Build RHS side of t2 equations, [Stanton:1991:4334] Eqn. 2
    rhs_T2 = u[o, o, v, v].copy()

    # P_(ab) t_ijae (F_be - 0.5 t_mb F_me)
    tmp = Fae - 0.5 * np.einsum('mb,me->be', t1, Fme)
    Pab = np.einsum('ijae,be->ijab', t2, tmp)
    rhs_T2 += Pab
    rhs_T2 -= Pab.swapaxes(2, 3)

    # P_(ij) t_imab (F_mj + 0.5 t_je F_me)
    tmp = Fmi + 0.5 * np.einsum('je,me->mj', t1, Fme)
    Pij = np.einsum('imab,mj->ijab', t2, tmp)
    rhs_T2 -= Pij
    rhs_T2 += Pij.swapaxes(0, 1)

    tmp_tau = build_tau(t1, t2, f, u, o, v)
    rhs_T2 += 0.5 * np.einsum('mnab,mnij->ijab', tmp_tau, Wmnij)
    rhs_T2 += 0.5 * np.einsum('ijef,abef->ijab', tmp_tau, Wabef)

    # P_(ij) * P_(ab)
    # (ij - ji) * (ab - ba)
    # ijab - ijba -jiab + jiba
    tmp = np.einsum('ie,ma,mbej->ijab', t1, t1, u[o, v, v, o])
    Pijab = np.einsum('imae,mbej->ijab', t2, Wmbej)
    Pijab -= tmp

    rhs_T2 += Pijab
    rhs_T2 -= Pijab.swapaxes(2, 3)
    rhs_T2 -= Pijab.swapaxes(0, 1)
    rhs_T2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)

    Pij = np.einsum('ie,abej->ijab', t1, u[v, v, v, o])
    rhs_T2 += Pij
    rhs_T2 -= Pij.swapaxes(0, 1)

    Pab = np.einsum('ma,mbij->ijab', t1, u[o, v, o, o])
    rhs_T2 -= Pab
    rhs_T2 += Pab.swapaxes(2, 3)

    ### Update t1 and t2 amplitudes
    return rhs_T1.T, rhs_T2.transpose(2,3,0,1)