from clusterfock.basis import Basis
import numpy as np
import pyscf


class PyscfBasis(Basis):
    def __init__(self, atom: str, basis: str, restricted: bool = True):
        mol = pyscf.gto.Mole()
        mol.build(atom=atom, basis=basis)
        self.mol = mol

        super().__init__(L=2 * mol.nao, N=mol.nelectron, restricted=True)

        self._atom_string = atom
        self._basis_string = basis

        L = self.L

        self.restricted = restricted
        self.setup()

    def setup(self):
        L = self.L
        self._energy_shift = self.mol.energy_nuc()
        self.s = self.mol.intor_symmetric("int1e_ovlp")
        self.h = self.mol.intor_symmetric("int1e_kin") + self.mol.intor_symmetric("int1e_nuc")
        self.u = self.mol.intor("int2e").reshape(L, L, L, L).transpose(0, 2, 1, 3)

        if not self.restricted:
            self.from_restricted()

    def pyscf_hartree_fock(self, tol=1e-8, inplace=True):
        """
        Uses pyscf to perform hf basis and changes to this basis
        """

        # Make and run mean field object
        self.mf = pyscf.scf.HF(self.mol)
        self.mf.run(verbose=0, tol=tol)

        return self.change_basis(self.mf.mo_coeff, inplace=inplace)

    @property
    def r(self) -> np.ndarray:
        r = self.mol.intor("int1e_r")

        if not self.restricted:
            r = self._add_spin_one_body(r)

        if not np.trace(self.C) == self.L:
            print("In a changed basis")
            r = self._change_basis_one_body(r, self.C)


        return r