from clusterfock.basis import Basis
from functools import cached_property
import numpy as np
import pyscf


class PyscfBasis(Basis):
    def __init__(self, atom: str, basis: str, restricted: bool = True, dtype=float, **kwargs):
        defaults = {"center": True, "charge": 0, "mol": None}

        defaults.update(kwargs)

        center = defaults["center"]
        charge = defaults["charge"]
        mol = defaults["mol"]

        if mol is None:
            mol = pyscf.gto.Mole()
            mol.unit = "bohr"
            mol.build(atom=atom, basis=basis, charge=charge)

        if center:
            charges = mol.atom_charges()
            coords = mol.atom_coords()
            nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
            mol.set_common_orig_(nuc_charge_center)

        self.mol = mol

        super().__init__(L=2 * mol.nao, N=mol.nelectron, restricted=True, dtype=dtype)

        self._args = (atom, basis, restricted, dtype)
        self._kwargs = defaults

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
        self.mf.conv_tol_grad = tol
        self.mf.run(verbose=0)
        self.C = self.mf.mo_coeff

        if not self.mf.converged:
            raise ValueError("Pyscf Hartree-Fock did not converge")

        if not self.restricted:
            self.C = self._add_spin_one_body(self.C)

        return self.change_basis(self.C, inplace=inplace)

    @cached_property
    def r(self) -> np.ndarray:
        r = self.mol.intor("int1e_r")
        return self._new_one_body_operator(r)

    @cached_property
    def Q(self) -> np.ndarray:
        L = self.L
        
        rr = self.mol.intor("int1e_rr", comp=9, hermi=0).reshape(3,3,L,L)
        
        # r = self.mol.intor("int1e_r")
        # r2 = np.einsum("xpq,xpq->pq", r, r)
        # delta_ij_r2 = np.einsum("ij,pq->ijpq", np.eye(3), r2)

        r2 = np.einsum("iipq->pq", rr)
        delta_ij_r2 = np.einsum("ij,pq->ijpq", np.eye(3), r2)

        Q = 3*rr - delta_ij_r2 

        return self._new_one_body_operator(Q)
    

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
