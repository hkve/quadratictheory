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
def build_tilde_tau(t1, t2, u, f, o, v):
    """Builds [Stanton:1991:4334] Eqn. 9"""
    ttau = t2.copy()
    tmp = 0.5 * np.einsum('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 10: \Tau)
def build_tau(t1, t2, u, f, o, v):
    """Builds [Stanton:1991:4334] Eqn. 10"""
    ttau = t2.copy()
    tmp = np.einsum('ia,jb->ijab', t1, t1)
    ttau += tmp
    ttau -= tmp.swapaxes(2, 3)
    return ttau


#Build Eqn 3:
def build_Fae(t1, t2, u, f, o, v):
    """Builds [Stanton:1991:4334] Eqn. 3"""
    Fae = f[v, v].copy()
    # Fae[np.diag_indices_from(Fae)] = 0

    Fae -= 0.5 * np.einsum('me,ma->ae', f[o, v], t1)
    Fae += np.einsum('mf,mafe->ae', t1, u[o, v, v, v])

    tmp_tau = build_tilde_tau(t1, t2, u, f, o, v)
    Fae -= 0.5 * np.einsum('mnaf,mnef->ae', tmp_tau, u[o, o, v, v])
    return Fae


#Build Eqn 4:
def build_Fmi(t1, t2, u, f, o, v):
    """Builds [Stanton:1991:4334] Eqn. 4"""
    Fmi = f[o, o].copy()
    # Fmi[np.diag_indices_from(Fmi)] = 0

    Fmi += 0.5 * np.einsum('ie,me->mi', t1, f[o, v])
    Fmi += np.einsum('ne,mnie->mi', t1, u[o, o, o, v])

    tmp_tau = build_tilde_tau(t1, t2, u, f, o, v)
    Fmi += 0.5 * np.einsum('inef,mnef->mi', tmp_tau, u[o, o, v, v])
    return Fmi


#Build Eqn 5:
def build_Fme(t1, t2, u, f, o, v):
    """Builds [Stanton:1991:4334] Eqn. 5"""
    Fme = f[o, v].copy()
    Fme += np.einsum('nf,mnef->me', t1, u[o, o, v, v])
    return Fme


#Build Eqn 6:
def build_Wmnij(t1, t2, u, f, o, v):
    """Builds [Stanton:1991:4334] Eqn. 6"""
    Wmnij = u[o, o, o, o].copy()

    Pij = np.einsum('je,mnie->mnij', t1, u[o, o, o, v])
    Wmnij += Pij
    Wmnij -= Pij.swapaxes(2, 3)

    tmp_tau = build_tau(t1, t2, u, f, o, v)
    Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tmp_tau, u[o, o, v, v])
    return Wmnij


#Build Eqn 7:
def build_Wabef(t1, t2, u, f, o, v):
    """Builds [Stanton:1991:4334] Eqn. 7"""
    # Rate limiting step written using tensordot, ~10x faster
    # The commented out lines are consistent with the paper

    Wabef = u[v, v, v, v].copy()

    Pab = np.einsum('baef->abef', np.tensordot(t1, u[v, o, v, v], axes=(0, 1)))
    # Pab = np.einsum('mb,amef->abef', t1, u[v, o, v, v])

    Wabef -= Pab
    Wabef += Pab.swapaxes(0, 1)

    tmp_tau = build_tau(t1, t2, u, f, o, v)

    Wabef += 0.25 * np.tensordot(tmp_tau, u[v, v, o, o], axes=((0, 1), (2, 3)))
    # Wabef += 0.25 * np.einsum('mnab,mnef->abef', tmp_tau, u[o, o, v, v])
    return Wabef


#Build Eqn 8:
def build_Wmbej(t1, t2, u, f, o, v):
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
    Fae = build_Fae(t1, t2, u, f, o, v)
    Fmi = build_Fmi(t1, t2, u, f, o, v)
    Fme = build_Fme(t1, t2, u, f, o, v)

    Wmnij = build_Wmnij(t1, t2, u, f, o, v)
    Wabef = build_Wabef(t1, t2, u, f, o, v)
    Wmbej = build_Wmbej(t1, t2, u, f, o, v)

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

    tmp_tau = build_tau(t1, t2, u, f, o, v)
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

#Step 1: Build one and two-particle elements of the similarity-transformed Hamiltonian
#Equations from [Gauss:1995:3561], Table III (b) & (c)

#### Note that Eqs. 1 - 6 of Table III (b) only modifies some intermediates that were 
#### already defined during the ground-state CCSD computation!

# 3rd equation
# Fme   = ccsd.build_Fme()

# 1st equation
def update_Fae(Fme, Fae, t1):
    """Eqn 1 from [Gauss:1995:3561], Table III (b)"""
    Fae -= 0.5 * np.einsum('me,ma->ae', Fme, t1)
    return Fae

# 2nd equation
def update_Fmi(Fme, Fmi, t1):
    """Eqn 2 from [Gauss:1995:3561], Table III (b)"""
    Fmi += 0.5 * np.einsum('me,ie->mi', Fme, t1)
    return Fmi

# 4th equation
def update_Wmnij(tau, Wmnij, u, o, v):
    """Eqn 4 from [Gauss:1995:3561], Table III (b)"""
    Wmnij += 0.25 * np.einsum('ijef,mnef->mnij', tau, u[o,o,v,v])
    return Wmnij

# 5th equation
def update_Wabef(tau, Wabef, u, o, v):
    """Eqn 5 from [Gauss:1995:3561], Table III (b)"""
    Wabef += 0.25 * np.einsum('mnab,mnef->abef', tau, u[o,o,v,v])
    return Wabef

# 6th equation
def update_Wmbej(t2, Wmbej, u, o, v):
    """Eqn 6 from [Gauss:1995:3561], Table III (b)"""
    Wmbej -= 0.5 * np.einsum('jnfb,mnef->mbej', t2, u[o,o,v,v])
    return Wmbej

#### Eqs. 7 - 10 define new intermediates

# 7th equation
def build_Wmnie(t1, u, o, v):
    """Eqn 7 from [Gauss:1995:3561], Table III (b)"""
    Wmnie = u[o,o,o,v].copy()
    Wmnie += np.einsum('if,mnfe->mnie', t1, u[o,o,v,v])
    return Wmnie

# 8th equation
def build_Wamef(t1, u, o, v):
    """Eqn 8 from [Gauss:1995:3561], Table III (b)"""
    Wamef = u[v,o,v,v].copy()
    Wamef -= np.einsum('na,nmef->amef', t1, u[o,o,v,v])
    return Wamef

# 9th equation
def build_Wmbij(t1, t2, tau, Fme, Wmnij, u, o, v):
    """Eqn 9 from [Gauss:1995:3561], Table III (b)"""
    Wmbij = u[o,v,o,o].copy()
    
    Wmbij -= np.einsum('me,ijbe->mbij', Fme, t2)
    Wmbij -= np.einsum('nb,mnij->mbij', t1, Wmnij)
   
    Wmbij += 0.5 * np.einsum('mbef,ijef->mbij', u[o,v,v,v],tau)
   
    Pij = np.einsum('jnbe,mnie->mbij', t2, u[o,o,o,v])
    Wmbij += Pij
    Wmbij -= Pij.swapaxes(2, 3)

    temp_mbej = u[o,v,v,o].copy()
    temp_mbej -= np.einsum('njbf,mnef->mbej', t2, u[o,o,v,v])
    Pij = np.einsum('ie,mbej->mbij', t1, temp_mbej)
    Wmbij += Pij
    Wmbij -= Pij.swapaxes(2, 3)
    return Wmbij

# 10th equation
def build_Wabei(t1, t2, tau, Fme, Wabef, u, o, v):
    """Eqn 10 from [Gauss:1995:3561], Table III (b)"""
    Wabei = u[v,v,v,o].copy()

    Wabei -= np.einsum('me,miab->abei', Fme, t2)
    Wabei += np.einsum('if,abef->abei', t1, Wabef)
    
    Wabei += 0.5 * np.einsum('mnei,mnab->abei', u[o,o,v,o],tau)
   
    Pab = np.einsum('mbef,miaf->abei', u[o,v,v,v], t2)
    Wabei -= Pab
    Wabei += Pab.swapaxes(0, 1)

    temp_mbei = u[o,v,v,o].copy()
    temp_mbei -= np.einsum('nibf,mnef->mbei', t2, u[o,o,v,v])
    Pab = np.einsum('ma,mbei->abei', t1, temp_mbei)
    Wabei -= Pab
    Wabei += Pab.swapaxes(0, 1)
    return Wabei    
 
### Build three-body intermediates: [Gauss:1995:3561] Table III (c)

# 1st equation
def build_Gae(t2, l2):
    """Eqn 1 from [Gauss:1995:3561], Table III (c)"""
    Gae = -0.5 * np.einsum('mnef,mnaf->ae', t2, l2)
    return Gae

# 2nd equation
def build_Gmi(t2, l2):
    """Eqn 2 from [Gauss:1995:3561], Table III (c)"""
    Gmi = 0.5 * np.einsum('mnef,inef->mi', t2, l2)
    return Gmi



### begin LCCSD iterations: Lambda equations from [Gauss:1995:3561] Table II, (a) & (b). 
def ccsd_l_Gauss_Stanton(t1, t2, l1, l2, u, f, o, v):
    t1, l1 = t1.T, l1.T
    t2, l2 = t2.transpose(2,3,0,1), l2.transpose(2,3,0,1)

    tau = build_tau(t1, t2, u, f, o, v)
    Fae = build_Fae(t1, t2, u, f, o, v)
    Fmi = build_Fmi(t1, t2, u, f, o, v)
    Fme = build_Fme(t1, t2, u, f, o, v)

    Wmnij = build_Wmnij(t1, t2, u, f, o, v)
    Wabef = build_Wabef(t1, t2, u, f, o, v)
    Wmbej = build_Wmbej(t1, t2, u, f, o, v)

    Fae   = update_Fae(Fme, Fae, t1)
    Fmi   = update_Fmi(Fme, Fmi, t1)

    Wmnij = update_Wmnij(tau, Wmnij, u, o, v)
    Wabef = update_Wabef(tau, Wabef, u, o, v)
    Wmbej = update_Wmbej(t2, Wmbej, u, o, v)

    Wmnie = build_Wmnie(t1, u, o, v)
    Wamef = build_Wamef(t1, u, o, v)
    Wmbij = build_Wmbij(t1, t2, tau, Fme, Wmnij, u, o, v)
    Wabei = build_Wabei(t1, t2, tau, Fme, Wabef, u, o, v)

    # Build intermediates that depend on lambda
    Gae = build_Gae(t2, l2)
    Gmi = build_Gmi(t2, l2)

    # Build RHS of l1 equations: Table II (a)
    rhs_L1  = Fme.copy()
    rhs_L1 += np.einsum('ie,ea->ia', l1, Fae)
    rhs_L1 -= np.einsum('ma,im->ia', l1, Fmi)
    rhs_L1 += np.einsum('me,ieam->ia', l1, Wmbej)
    rhs_L1 += 0.5 * np.einsum('imef,efam->ia', l2, Wabei)
    rhs_L1 -= 0.5 * np.einsum('mnae,iemn->ia', l2, Wmbij)
    rhs_L1 -= np.einsum('ef,eifa->ia', Gae, Wamef)
    rhs_L1 -= np.einsum('mn,mina->ia', Gmi, Wmnie)

    ### Build RHS of l2 equations
    ### Table II (b)
    rhs_L2 = u[o,o,v,v].copy()

    # P_(ab) l_ijae * F_eb
    Pab = np.einsum('ijae,eb->ijab', l2, Fae)
    rhs_L2 += Pab
    rhs_L2 -= Pab.swapaxes(2, 3)

    # P_(ij) l_imab * F_jm
    Pij = np.einsum('imab,jm->ijab', l2, Fmi)
    rhs_L2 -= Pij
    rhs_L2 += Pij.swapaxes(0, 1)

    # 0.5 * l_mnab * W_ijmn
    rhs_L2 += 0.5 * np.einsum('mnab,ijmn->ijab', l2, Wmnij)

    # 0.5 * l_ijef * W_efab
    rhs_L2 += 0.5 * np.einsum('ijef,efab->ijab', l2, Wabef)

    # P_(ij) l_ie W_ejab
    Pij = np.einsum('ie,ejab->ijab', l1, Wamef)
    rhs_L2 += Pij
    rhs_L2 -= Pij.swapaxes(0, 1)

    # P_(ab) l_ma W_ijmb
    Pab = np.einsum('ma,ijmb->ijab', l1, Wmnie)
    rhs_L2 -= Pab
    rhs_L2 += Pab.swapaxes(2, 3)
   
    # P_(ij) P_(ab) l_imae W_jebm
    Pijab = np.einsum('imae,jebm->ijab', l2, Wmbej)
    rhs_L2 += Pijab
    rhs_L2 -= Pijab.swapaxes(0, 1)
    rhs_L2 -= Pijab.swapaxes(2, 3)
    rhs_L2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
     
    # P_(ij) P_(ab) l_ia F_jb
    Pijab = np.einsum('ia,jb->ijab', l1, Fme)
    rhs_L2 += Pijab
    rhs_L2 -= Pijab.swapaxes(0, 1)
    rhs_L2 -= Pijab.swapaxes(2, 3)
    rhs_L2 += Pijab.swapaxes(0, 1).swapaxes(2, 3)
    
    # P_(ab) <ij||ae> G_be
    Pab = np.einsum('be,ijae->ijab', Gae, u[o,o,v,v])
    rhs_L2 += Pab
    rhs_L2 -= Pab.swapaxes(2, 3)

    # P_(ij) <im||ab> G_mj
    Pij = np.einsum('mj,imab->ijab', Gmi, u[o,o,v,v])
    rhs_L2 -= Pij
    rhs_L2 += Pij.swapaxes(0, 1)

    return rhs_L1.T, rhs_L2.transpose(2,3,0,1)