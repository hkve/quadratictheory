import numpy as np
import clusterfock as cf

from pyscf import lib
from pyscf import gto, scf, cc, fci, ao2mo

def displace_water(r1_new, r2_new, r1, r2):
    r1_unit = r1 / np.linalg.norm(r1)
    r2_unit = r2 / np.linalg.norm(r2)

    r1 = r1_new * r1_unit
    r2 = r2_new * r2_unit

    r_H1_geom = " ".join(str(r) for r in r1)
    r_H2_geom = " ".join(str(r) for r in r2)
    geometry = f"O 0 0 0; H {r_H1_geom}; H {r_H2_geom}"

    return geometry

def run_fci(atom, basis, *args):
    mol = gto.M(unit="angstrom")
    mol.verbose = 0
    mol.build(atom=atom, basis=basis)

    hf = scf.RHF(mol)
    hf.conv_tol_grad = 1e-6
    hf.max_cycle = 50
    hf.init_guess = "minao"
    hf_energy = hf.kernel(vocal=0)

    myfci = fci.direct_spin0.FCI(mol)
    myfci.conv_tol = 1e-10
    h1 = hf.mo_coeff.T.dot(hf.get_hcore()).dot(hf.mo_coeff)
    eri = ao2mo.kernel(mol, hf.mo_coeff)
    myfci.max_space = 12
    myfci.davidson_only = True

    nroots = 1
    e_fci, c_fci = myfci.kernel(
        h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc(), nroots=nroots, max_memory=2000
    )

    return e_fci


def run(distances, tols):
    basis_string = "sto-3g"
    energy = np.zeros_like(distances)
    energy_fci = np.zeros_like(distances)

    r1 = np.array([0.757, 0.586, 0.000])
    r2 = np.array([-0.757, 0.586, 0.000])

    t_prev, l_prev = 0, 0
    for i, r in enumerate(distances):
        atom = displace_water(r,r,r1,r2)
        basis = cf.PyscfBasis(atom, basis_string, restricted=True)

        hf = cf.HF(basis)
        hf.run(tol=1e-4)

        if hf.converged:
            basis.change_basis(hf.C)
            basis.from_restricted()

            cc = cf.QCCD(basis)
            cc.run(tol=tol[i], maxiters=300, vocal=False)
            energy[i] = cc.energy()
            print(f"Done QCCD {r}")
        
        energy_fci[i] = run_fci(atom, basis_string)
        print(f"Done FCI {r}")
    return energy, energy_fci

if __name__ == "__main__":
    params = {
        1.00: 1e-4,
        1.25: 1e-4,
        1.50: 1e-4,
        1.75: 1e-4,
        2.00: 0.001,
        2.25: 0.0004,
    }

    r = np.array(list(params.keys()))
    tol = np.array(list(params.values()))
    energy_qccd, energy_fci = run(r, tol)
    E0 = energy_fci[-1]

    for i in range(len(r)):
        print(f"{r[i]}\t{energy_qccd[i]-energy_fci[i]:.5f}\t{energy_fci[i]:.5f}")
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots()
    # ax.plot(r, energy_qccd-E0, label="QCCD")
    # ax.plot(r, energy_fci-E0, label="FCI")
    
    # ax.legend()
    # plt.show()