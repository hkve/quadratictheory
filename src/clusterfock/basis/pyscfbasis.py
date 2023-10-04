from clusterfock.basis import Basis
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
        self._energy_shift = mol.energy_nuc()
        self.s = mol.intor_symmetric("int1e_ovlp")
        self.h = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
        self.u = mol.intor("int2e").reshape(L, L, L, L).transpose(0, 2, 1, 3)

        if not restricted:
            self.from_restricted()

    def pyscf_hartree_fock(self, tol=1e-8, inplace=True):
        """
        Uses pyscf to perform hf basis and changes to this basis
        """

        # Make and run mean field object
        self.mf = pyscf.scf.HF(self.mol)
        self.mf.run(verbose=0, tol=tol)

        return self.change_basis(self.mf.mo_coeff, inplace=inplace)

