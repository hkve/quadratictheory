import numpy as np
from pyscf import gto, scf, cc, fci, ao2mo
from pyscf import lib
import time

hartree_eV = 27.2114079527 #1 Hartree energy = 27.2114079527 eV

#geometry = f"Be 0.0 0.0 0.0"
geometry = f"C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
basis = f"chp.dat"

mol = gto.M(unit="bohr")
mol.verbose = 0
mol.build(atom=geometry, basis=basis, charge=1, cart=True)


hf = scf.RHF(mol)
hf.conv_tol_grad = 1e-10
hf_energy = hf.kernel()



mycc = cc.RCCSD(hf, frozen=0)
mycc.conv_tol = 1e-10
mycc.kernel()

print(f"RHF  energy: {hf.e_tot:.6f}")
print(f"CCSD energy: {mycc.e_tot:.8f}") 

e_ee_singlet, c_ee_singlet = mycc.eomee_ccsd_singlet(nroots=10)

print()
print(f"** EOM-CCSD singlet excitation energies **")
print(e_ee_singlet, "[a.u.]")
print()
print(e_ee_singlet*hartree_eV, "[eV]")
print()






