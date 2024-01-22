import numpy as np

from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)

import warnings

def make_pyscf_system(molecule, basis="cc-pvdz", verbose=False, charge=0):

    import pyscf

    # Build molecule in AO-basis
    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.build(atom=molecule, basis=basis)
    nuclear_repulsion_energy = mol.energy_nuc()

    n = mol.nelectron
    assert (
        n % 2 == 0
    ), "We require closed shell, with an even number of particles"

    l = mol.nao

    hf = pyscf.scf.RHF(mol)
    hf_energy = hf.kernel()

    if not hf.converged:
        warnings.warn("RHF calculation did not converge")

    if verbose:
        print(f"RHF energy: {hf.e_tot}")

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)

    C = np.asarray(hf.mo_coeff)

    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
    position = mol.intor("int1e_r").reshape(3, l, l)
    
    momentum = 1j * mol.intor("int1e_ipovlp").reshape(3, l, l)
    
    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = position
    bs.momentum = momentum
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(n, bs)
    system.change_basis(C)

    return system


def get_pyscf_geometries():
    pyscf_geometries = dict(
        he="he 0.0 0.0 0.0",
        be="be 0.0 0.0 0.0",
        ne="ne 0.0 0.0 0.0",
        h2="h 0.0 0.0 0.0; h 0.0 0.0 1.4",
        lih="li 0.0 0.0 0.0; h 0.0 0.0 3.08",
        bh="b 0.0 0.0 0.0; h 0.0 0.0 2.3289",
        chp="c 0.0 0.0 0.0; h 0.0 0.0 2.137130",
        h2o="O 0.0 0.0 -0.1239093563; H 0.0 1.4299372840 0.9832657567; H 0.0 -1.4299372840 0.9832657567",
        no2="N; O 1 3.25; O 1 3.25 2 160",
        lif="f 0.0 0.0 0.0; li 0.0 0.0 -2.9552746891",
        co="c 0.0 0.0 0.0; o 0.0 0.0 2.1316109151",
        n2="n 0.0 0.0 0.0; n 0.0 0.0 2.074",
        co2="c 0.0 0.0 0.0; o 0.0 0.0 2.1958615987; o 0.0 0.0 -2.1958615987",
        nh3="N 0.0 0.0 0.2010; H 0.0 1.7641 -0.4690; H 1.5277 -0.8820 -0.4690; H -1.5277 -0.8820 -0.4690",
    	hf="h 0.0 0.0 0.0; f 0.0 0.0 1.7328795",
    	ch4="c 0.0 0.0 0.0; h 1.2005 1.2005 1.2005; h -1.2005 -1.2005 1.2005; h -1.2005 1.2005 -1.2005; h 1.2005 -1.2005 -1.2005",
    )

    return pyscf_geometries

class cos_ramp_laser:

    def __init__(self, F_str, omega, n_c=1):

        self.F_str = F_str
        self.omega = omega
        self.n_c = n_c

    def __call__(self, t):
        if t < 2*np.pi/self.omega*self.n_c:
            return self.omega*t/(2*np.pi*self.n_c)*self.F_str*np.cos(self.omega*t)
        else:
            return self.F_str*np.cos(self.omega*t)

class sin_laser:
    def __init__(self, F_str, omega, tprime, gauge="length"):
        self.F_str = F_str
        self.omega = omega
        self.tprime = tprime
        self.gauge = gauge

    def __call__(self, t):
        if self.gauge == "velocity":
            return -self.F_str/self.omega * (1-np.cos(self.omega*t))
        elif self.gauge == "length":
            omega = self.omega
            return (
                self.F_str*np.sin(self.omega*t)
            )
            

class sin_ramp_laser:

    def __init__(self, F_str, omega):

        self.F_str = F_str
        self.omega = omega

    def __call__(self, t):
        if t < 2*np.pi/self.omega:
            return self.omega*t/(2*np.pi)*self.F_str*np.sin(self.omega*t)
        else:
            return self.F_str*np.sin(self.omega*t)

class sine_square_laser:
    def __init__(self, F_str, omega, tprime, phase=0):
        self.F_str = F_str
        self.omega = omega
        self.tprime = tprime
        self.phase = phase

    def __call__(self, t):
        pulse = (
            (np.sin(np.pi * t / self.tprime) ** 2)
            * np.heaviside(t, 1.0)
            * np.heaviside(self.tprime - t, 1.0)
            * np.sin(self.omega * t + self.phase)
            * self.F_str
        )
        return pulse


class Gaussian_cos_laser:
    def __init__(self, F_str=0.1, t_c=6, w=1, omega=0.057, phase=0):

        self.F_str = F_str
        self.t_c = t_c
        self.w = w
        self.omega = omega
        self.phase = phase

    def __call__(self, t):

        return (
            self.F_str
            * np.exp(-((t - self.t_c) ** 2) / (2 * self.w ** 2))
            * np.cos(self.omega * t + self.phase)
            / (np.sqrt(2 * np.pi) * self.w)
        )


class Discrete_delta_pulse:
    def __init__(self, F_str, dt):
        self.F_str = F_str
        self.dt = dt

    def __call__(self, t):
        if t < self.dt:
            return self.F_str/self.dt
        else:
            return 0


class Gaussian_delta_pulse:
    # https://pubs.acs.org/doi/10.1021/ct200137z
    def __init__(self, F_str=1e-3, t_c=5, gamma=5.0):

        self.F_str = F_str
        self.t_c = t_c
        self.gamma = gamma

    def __call__(self, t):
        return (
            self.F_str
            * np.sqrt(self.gamma / np.pi)
            * np.exp(-self.gamma * (t - self.t_c) ** 2)
        )
