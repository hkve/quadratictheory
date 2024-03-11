import numpy as np
from opt_einsum import contract
from pyscf import gto, scf, cc, fci, ao2mo
from pyscf.ci.cisd import tn_addrs_signs
from scipy.linalg import eigh

def fci_pyscf(geometry, basis, nroots=5):
    mol = gto.M(unit="bohr")
    mol.verbose = 3
    mol.build(atom=geometry, basis=basis)

    s = mol.intor_symmetric("int1e_ovlp")

    hf = scf.RHF(mol)
    hf.conv_tol_grad = 1e-10
    hf.max_cycle = 1000
    hf.kernel()

    n_fci_states = 3

    myfci = fci.direct_spin0.FCI(mol)
    myfci.conv_tol = 1e-10
    h1 = hf.mo_coeff.T.dot(hf.get_hcore()).dot(hf.mo_coeff)
    eri = ao2mo.kernel(mol, hf.mo_coeff)
    e_fci, c_fci = myfci.kernel(
        h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc(), nroots=nroots
    )

    nmo = mol.nao
    nocc = mol.nelectron // 2

    t1addrs, t1signs = tn_addrs_signs(nmo, nocc, 1)
    t2addrs, t2signs = tn_addrs_signs(nmo, nocc, 2)
    # t3addrs, t3signs = tn_addrs_signs(nmo, nocc, 3)
    # t4addrs, t4signs = tn_addrs_signs(nmo, nocc, 4)

    ref_weights = []
    s_weights = []
    d_weights = []
    for j in range(nroots):
        ref_weights.append(c_fci[j][0, 0] ** 2)

        cis_a = c_fci[j][t1addrs, 0] * t1signs
        cis_b = c_fci[j][0, t1addrs] * t1signs
        s_weights.append((cis_a**2, cis_b**2))

        cid_aa = c_fci[j][t2addrs, 0] * t2signs
        cid_bb = c_fci[j][0, t2addrs] * t2signs
        cid_ab = np.einsum(
            "ij,i,j->ij", c_fci[j][t1addrs[:, None], t1addrs], t1signs, t1signs
        )

        d_weights.append((cid_aa**2, cid_bb**2, cid_ab**2))

    sum_s_weights_fci = np.sum(s_weights[0][0].ravel()) + np.sum(
        s_weights[0][1].ravel()
    )
    sum_d_weights_fci = (
        np.sum(d_weights[0][0].ravel())
        + np.sum(d_weights[0][1].ravel())
        + np.sum(d_weights[0][2].ravel())
    )

    W = {"0": ref_weights[0], "S": sum_s_weights_fci, "D": sum_d_weights_fci}

    return e_fci, W

    # if mol.nelectron >= 4:
    #     cit_aab = np.einsum('ij,i,j->ij', c_fci[0][t2addrs[:,None], t1addrs], t2signs, t1signs)
    #     cit_abb = np.einsum('ij,i,j->ij', c_fci[0][t1addrs[:,None], t2addrs], t1signs, t2signs)

    # if mol.nelectron >= 4:
    #     ciq_aabb = np.einsum('ij,i,j->ij', c_fci[0][t2addrs[:,None], t2addrs], t2signs, t2signs)

    # if mol.nelectron >= 6:
    #     cit_aaa = c_fci[0][t3addrs, 0] * t3signs
    #     cit_bbb = c_fci[0][0, t3addrs] * t3signs

    # if mol.nelectron >= 6:
    #     ciq_aaab = np.einsum('ij,i,j->ij', c_fci[0][t3addrs[:,None], t1addrs], t3signs, t1signs)
    #     ciq_abbb = np.einsum('ij,i,j->ij', c_fci[0][t1addrs[:,None], t3addrs], t1signs, t3signs)

    # if mol.nelectron  >= 8:
    #     ciq_aaaa = c_fci[0][t4addrs, 0] * t4signs
    #     ciq_bbbb = c_fci[0][0, t4addrs] * t4signs

