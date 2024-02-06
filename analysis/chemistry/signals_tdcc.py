import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt

from utils.misc import load_files, dat_path
from utils.runs import get_geometries, get_pyscf_mol_custom_basis, get_symmetries

from plotting.general import plot_compare

def run_cc(
    molecule_name,
    basis_name,
    time,
    pulse,
    sampler,
    CC,
    integrator,
    general_args={},
    integrator_args={},
):
    default = {
        "charge": 0,
        "hf_tol": 1e-10,
        "cc_tol": 1e-10,
        "mol": None,
    }
    default.update(general_args)

    geometries = get_geometries()
    geometry = geometries[molecule_name]

    charge = default["charge"]
    hf_tol = default["hf_tol"]
    cc_tol = default["cc_tol"]
    mol = default["mol"]

    
    if mol is not None:
        basis = cf.PyscfBasis(geometry, basis_name, charge=charge, mol=mol)
    else:
        basis = cf.PyscfBasis(geometry, basis_name, charge=charge)

    basis.pyscf_hartree_fock()
    basis.from_restricted()

    cc_run_args = {"tol": cc_tol, "vocal": False}
    if not "Q" in CC.__name__:
        cc_run_args["include_l"] = True

    cc = CC(basis).run(**cc_run_args)

    tdcc = cf.TimeDependentCoupledCluster(
        cc, time, integrator=integrator, integrator_args=integrator_args
    )
    tdcc.external_one_body = pulse
    tdcc.sampler = sampler

    tdcc.run(vocal=True)
    results = tdcc.results

    cartesian_polarisation_direction = get_cartesian_direction(pulse._u)
    filename = f"{CC.__name__}_{molecule_name}_{basis_name}_Tend={time[1]}_dt={time[2]}_{integrator}_{pulse.__class__.__name__}_{cartesian_polarisation_direction}.npz"
    print(f"DONE: {CC.__name__} {molecule_name} {pulse.__class__.__name__} {integrator} with pol = {cartesian_polarisation_direction}")

    np.savez(f"{dat_path()}/{filename}", **results)

def get_cartesian_direction(u):
    largest_idx = np.argmax(u)
    if np.abs(u[largest_idx]-1) < 1e-8:
        return largest_idx
    else:
        return -1

def delta_kick_compare(molecule_name, basis_name, **kwargs):
    u = np.array([1.0, 0.0, 0.0])
    F_str = 1e-3
    dt = 0.025
    t_end = 1000

    default = {
        "integrator": "Rk4Integrator",
        "integrator_args": {"dt": dt},
    }
    # integrator = "GaussIntegrator"
    # integrator_args = {"s": 3, "maxit": 20, "eps": 1e-6, "method": "A", "mu": 1.75}
    kwargs.update(default)

    integrator = kwargs["integrator"]
    integrator_args = kwargs["integrator_args"]

    time = (0, t_end, dt)
    pulse = cf.pulse.DeltaKick(u, F_str, dt)
    sampler = cf.sampler.DipoleSamplerExpanded()

    run_cc(
        molecule_name,
        basis_name,
        time,
        pulse,
        sampler,
        cf.CCSD,
        integrator,
        integrator_args=integrator_args,
        general_args=kwargs,
    )
    run_cc(
        molecule_name,
        basis_name,
        time,
        pulse,
        sampler,
        cf.QCCSD,
        integrator,
        integrator_args=integrator_args,
        general_args=kwargs,
    )


def sin2_pulse_compare(molecule_name, basis_name, **kwargs):
    u = np.array([1.0, 0.0, 0.0])
    F_str = 1e-2
    dt = 0.1
    omega = 0.2
    tprime = 2 * np.pi / omega
    t_end = 0.5

    default = {
        "integrator": "Rk4Integrator",
        "integrator_args": {"dt": dt},
    }
    # integrator = "GaussIntegrator"
    # integrator_args = {"s": 3, "maxit": 20, "eps": 1e-6, "method": "A", "mu": 1.75}
    kwargs.update(default)

    integrator = kwargs["integrator"]
    integrator_args = kwargs["integrator_args"]

    time = (0, t_end, dt)
    pulse = cf.pulse.Sin2(u, F_str, omega, tprime)
    sampler = cf.sampler.DipoleSamplerExpanded()

    run_cc(
        molecule_name,
        basis_name,
        time,
        pulse,
        sampler,
        cf.CCSD,
        integrator,
        integrator_args=integrator_args,
    )
    run_cc(
        molecule_name,
        basis_name,
        time,
        pulse,
        sampler,
        cf.QCCSD,
        integrator,
        integrator_args=integrator_args,
    )

def delta_kick_1990_basis(CC):
    F_str = 1e-3
    dt = 0.1
    t_end = 1
    default = {
        "integrator": "Rk4Integrator",
        "integrator_args": {"dt": dt},
    }
    geometry = f"C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
    name, basis = "chp", "custom"
    mol = get_pyscf_mol_custom_basis(geometry=geometry)

    integrator = default["integrator"]
    integrator_args = default["integrator_args"]

    time = (0, t_end, dt)
    sampler = cf.sampler.DipoleSamplerExpanded()

    polarisations = get_symmetries(geometry)
    us = [polarisations["z"], polarisations["y"]]

    for u in us:
        pulse = cf.pulse.DeltaKick(u, F_str, dt)

        run_cc(
            name,
            basis,
            time,
            pulse,
            sampler,
            CC,
            integrator,
            integrator_args=integrator_args,
            general_args={"mol": mol},
        )

def main():
    delta_kick_1990_basis(cf.CCD)
    delta_kick_1990_basis(cf.CCSD)
    # molecule_name, basis_name = "be", "cc-pVDZ"
    # delta_kick_compare("be", basis_name, charge=0)
    # delta_kick_compare("lih", basis_name, charge=0)
    # delta_kick_compare("chp", basis_name, charge=1)

if __name__ == "__main__":
    main()
