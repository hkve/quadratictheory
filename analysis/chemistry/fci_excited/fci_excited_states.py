import numpy as np
from pyscf import gto, scf, cc, fci, ao2mo
from pyscf import lib
import time

hartree_eV = 27.2114079527 #1 Hartree energy = 27.2114079527 eV

#geometry = f"Be 0.0 0.0 0.0"
geometry = f"C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
basis = f"chp.dat"

mol = gto.M(unit="bohr")
mol.verbose = 3
mol.build(atom=geometry, basis=basis, charge=1, cart=True)

hf = scf.RHF(mol)
hf.conv_tol_grad = 1e-10
hf_energy = hf.kernel()

tic = time.time()
myfci = fci.direct_spin0.FCI(mol)
myfci.conv_tol = 1e-10
h1 = hf.mo_coeff.T.dot(hf.get_hcore()).dot(hf.mo_coeff)
eri = ao2mo.kernel(mol, hf.mo_coeff)

nroots = 2
e_fci, c_fci = myfci.kernel(
    h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc(), nroots=nroots, max_memory=2000
)
toc = time.time()
print(f"Time FCI: {toc-tic}")
print(e_fci)
# excitation_energies = e_fci[1:]-e_fci[0]
# print()
# print(f"E0_FCI: {e_fci[0]}")
# print()
# print(excitation_energies)
# print()
# print(excitation_energies*hartree_eV)
# print()