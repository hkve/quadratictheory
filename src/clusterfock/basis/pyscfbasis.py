from clusterfock.basis import Basis
from functools import cached_property
import numpy as np
import pyscf

from pyscf import lib
lib.num_threads(1)

class PyscfBasis(Basis):
    def __init__(self, atom: str, basis: str, restricted: bool = True, center=True, dtype=float):
        mol = pyscf.gto.Mole()
        mol.build(atom=atom, basis=basis)

        if center:
            charges = mol.atom_charges()
            coords = mol.atom_coords()
            nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
            mol.set_common_orig_(nuc_charge_center)

        self.mol = mol

        super().__init__(L=2 * mol.nao, N=mol.nelectron, restricted=True, dtype=dtype)

        self._atom_string = atom
        self._basis_string = basis

        L = self.L
        
        self.restricted = restricted
        self.setup()

    def setup(self):
        L = self.L
        self._energy_shift = self.mol.energy_nuc()
        self.s = self.mol.intor_symmetric("int1e_ovlp")
        # self.h = self.mol.intor_symmetric("int1e_kin") + self.mol.intor_symmetric("int1e_nuc")
        self.h = pyscf.scf.hf.get_hcore(self.mol)
        self.u = self.mol.intor("int2e").reshape(L, L, L, L).transpose(0, 2, 1, 3)
        
        if not self.restricted:
            self.from_restricted()

    def pyscf_hartree_fock(self, tol=1e-8, inplace=True):
        """
        Uses pyscf to perform hf basis and changes to this basis
        """

        # Make and run mean field object
        self.mf = pyscf.scf.RHF(self.mol)
        self.mf.run(verbose=0)

        self.C = self.mf.mo_coeff
        print("CF")
        print(np.round(self.C.real[-6:-1,-6:-1], 4))
        if not self.restricted:
            self.C = self._add_spin_one_body(self.C)

        return self.change_basis(self.C, inplace=inplace)

    @cached_property
    def r(self) -> np.ndarray:
        r = self.mol.intor("int1e_r")
        return self._new_one_body_operator(r)

    def density(self, rho, r=None):
        if r is None:
            N = 10
            x_ = np.linspace(-4, 4, N)
            x, y, z = np.meshgrid(x_, x_, x_, indexing="ij")
            r = np.c_[x.reshape(N**3), y.reshape(N**3), z.reshape(N**3)]

        phi = self.mol.eval_ao("GTOval_sph", r).T

        if not self.restricted:
            phi = np.repeat(phi, repeats=2, axis=0)

        return np.einsum("px,pq,qx->x", phi, rho, phi)
