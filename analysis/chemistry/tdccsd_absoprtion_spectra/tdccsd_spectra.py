import numpy as np
import sys
import os

import tqdm
import matplotlib.pyplot as plt

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from coupled_cluster.rccsd import RCCSD, TDRCCSD
from gauss_integrator import GaussIntegrator
from rk4_integrator import Rk4Integrator
from scipy.integrate import complex_ode
from utils import sine_square_laser, get_pyscf_geometries, Gaussian_delta_pulse, Discrete_delta_pulse

import basis_set_exchange as bse

geometries = get_pyscf_geometries()

# System and basis parameters
name = "chp"
basis = "cc-pvdz"
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 1

# Laser pulse parameters
F_str = 1e-3
polarization_direction = 0
integrator = "rk4"

molecule = geometries[name]

system = construct_pyscf_system_rhf(
    molecule=molecule,
    basis=basis_set,
    add_spin=False,
    anti_symmetrize=False,
    charge=charge,
)
print(f"Nr of orbitals: {system.l}")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

polarization = np.zeros(3)
polarization[polarization_direction] = 1

tfinal = 100

dt = 0.1
num_steps = int(tfinal / dt) + 1
time_points = np.linspace(0, tfinal, num_steps)

system.set_time_evolution_operator(
    DipoleFieldInteraction(
        Discrete_delta_pulse(F_str=F_str, dt=dt),
        polarization_vector=polarization,
    )
)
################################################################################
rccsd = RCCSD(system)
ground_state_tolerance = 1e-10
rccsd.compute_ground_state(
    t_kwargs=dict(tol=ground_state_tolerance),
    l_kwargs=dict(tol=ground_state_tolerance),
)

print(f"{rccsd.compute_energy() = }")
exit()

tdrccsd = TDRCCSD(system)
if integrator=="vode":
    r = complex_ode(tdrccsd).set_integrator("vode")
elif integrator=="rk4":
    print(f"Rk4 integrator is used with dt: {dt}")
    r = complex_ode(tdrccsd).set_integrator("Rk4Integrator", dt=dt)
elif integrator=="gauss":
    r = complex_ode(tdrccsd).set_integrator("GaussIntegrator", s=3, eps=1e-6)
r.set_initial_value(rccsd.get_amplitudes(get_t_0=True).asarray())
################################################################################

# Initialize arrays to hold different "observables".
energy = np.zeros(num_steps, dtype=np.complex128)
dipole_moment = np.zeros((num_steps, 3), dtype=np.complex128)

# Set initial values
energy[0] = tdrccsd.compute_energy(r.t, r.y)
for j in range(3):
    dipole_moment[0, j] = tdrccsd.compute_one_body_expectation_value(
        r.t,
        r.y,
        system.dipole_moment[j],
    )

for i in tqdm.tqdm(range(num_steps - 1)):

    r.integrate(r.t + dt)

    for j in range(3):
        dipole_moment[i + 1, j] = tdrccsd.compute_one_body_expectation_value(
            r.t,
            r.y,
            system.dipole_moment[j],
        )

samples = dict()
samples["time_points"] = time_points
samples["dipole_moment"] = dipole_moment

np.savez(
    f"dat/tdrccsd_{name}_{basis}_discrete_delta_pulse_E0={F_str}_dt={dt}_pol_dir={polarization_direction}.npz",
    **samples,
)