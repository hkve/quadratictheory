import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci, ao2mo
from pyscf.ci.cisd import tn_addrs_signs
from pyscf import lib
lib.num_threads(1)


def get_geometry(theta):
    theta1 = (theta / 2) * np.pi / 180
    theta2 = -theta1
    theta3 = theta1 + np.pi
    theta4 = -theta3

    r = 3.284  # Place H atoms on circle with R=1.738Å
    x1, y1 = r * np.cos(theta1), r * np.sin(theta1)
    x2, y2 = r * np.cos(theta2), r * np.sin(theta2)
    x3, y3 = r * np.cos(theta3), r * np.sin(theta3)
    x4, y4 = r * np.cos(theta4), r * np.sin(theta4)

    geometry = f"""
	H {x1} {y1} 0.0; 
	H {x2} {y2} 0.0;
	H {x3} {y3} 0.0;
	H {x4} {y4} 0.0
	"""
 
    return geometry

def fci_pyscf(hf, mol, nroots=5):
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

    return e_fci, ref_weights, s_weights, d_weights

def run_fci(geometry, basis):
    basis = "dzp"
    mol = gto.M(unit="bohr")
    mol.verbose = 3
    mol.build(atom=geometry, basis=basis)

    rhf = scf.RHF(mol)
    rhf.conv_tol_grad = 1e-10
    rhf.max_cycle = 1000
    rhf.kernel()

    n_fci_states = 5
    fci_energies, fci_ref_weight, fci_s_weight, fci_d_weight = fci_pyscf(
        rhf, mol, nroots=n_fci_states
    )

    e_fci = fci_energies[0]
    W0 = fci_ref_weight[0]
    WS_max = np.max(fci_s_weight[0][0].ravel()) 
    WD_max = np.max(
            np.array(
                [
                    np.max(fci_d_weight[0][0].ravel()),
                    np.max(fci_d_weight[0][1].ravel()),
                    np.max(fci_d_weight[0][2].ravel()),
                ]
            )
        )

    return e_fci, W0, WS_max, WD_max


def run_ccsd(geometry, basis, quad=False):
    b = cf.PyscfBasis(geometry, basis, restricted=False).pyscf_hartree_fock()
    b.from_restricted()

    cc = cf.QCCSD(b) if quad else cf.CCSD(b)

    if quad:
        cc.run()
    else:
        cc.run(include_l=True)

    e_cc = cc.energy()

    W0 = cc.reference_weights()
    WS = cc.singles_weights().max()
    WD = cc.doubles_weights().max()

    WT, WQ = 0, 0
    if quad:
        WT = cc.triples_weight()
        WQ = cc.quadruple_weight()

    return e_cc, W0, WS, WD, WT, WQ

def run():
    thetas = np.arange(80,100+1,1)
    basis = "DZ"

    print(thetas)
    exit()
    filename = f"dat/H4_rectangle_{basis}_"
    filename_fci = filename + "fci.npz"
    filename_ccsd = filename + "ccsd.npz"
    filename_qccsd = filename + "qccsd.npz"

    fci = {
        "energy": np.zeros_like(thetas),
        "W0": np.zeros_like(thetas),
        "WS": np.zeros_like(thetas),
        "WD": np.zeros_like(thetas),
    }
    for i, theta in next(thetas):
        geometry = get_geometry(theta) 
        e_fci, W0, WS_max, WD_max = run_fci(geometry, basis)
        fci["energy"][i] = e_fci
        fci["W0"][i] = W0
        fci["WS"][i] = WS_max
        fci["WD"][i] = WD_max
        print(f"FCI {theta = } done")
    np.saves(filename_fci, **fci)

    ccsd = {
        "energy": np.zeros_like(thetas),
        "W0": np.zeros_like(thetas),
        "WS": np.zeros_like(thetas),
        "WD": np.zeros_like(thetas),
    }
    for i, theta in next(thetas):
        geometry = get_geometry(theta) 
        e_ccsd, W0, WS_max, WD_max, _, _ = run_ccsd(geometry, basis, quad=False)
        ccsd["energy"][i] = e_ccsd
        ccsd["W0"][i] = W0
        ccsd["WS"][i] = WS_max
        ccsd["WD"][i] = WD_max
        print(f"CCSD {theta = } done")

    np.saves(filename_ccsd, **ccsd)

    qccsd = {
        "energy": np.zeros_like(thetas),
        "W0": np.zeros_like(thetas),
        "WS": np.zeros_like(thetas),
        "WD": np.zeros_like(thetas),
        "WT": np.zeros_like(thetas),
        "WQ": np.zeros_like(thetas),
    }
    for i, theta in next(thetas):
        geometry = get_geometry(theta) 
        e_qccsd, W0, WS_max, WD_max, WT, WQ = run_ccsd(geometry, basis, quad=True)
        qccsd["energy"][i] = e_qccsd
        qccsd["W0"][i] = W0
        qccsd["WS"][i] = WS_max
        qccsd["WD"][i] = WD_max
        qccsd["WT"][i] = WT
        qccsd["WQ"][i] = WQ
        print(f"QCCSD {theta = } done")

    np.saves(filename_qccsd, **qccsd)

if __name__ == "__main__":
    run()