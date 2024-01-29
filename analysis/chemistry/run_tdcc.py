import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt
from tdccsd_absoprtion_spectra.utils import get_pyscf_geometries

from IPython import embed

def run_cc(molecule_name, basis_name, time, pulse, sampler, CC, integrator, general_args={}, integrator_args={}):
    default = {
        "charge": 0,
        "hf_tol": 1e-10,
        "cc_tol": 1e-10,
    }
    default.update(general_args)

    geometries = get_pyscf_geometries()
    geometry = geometries[molecule_name]

    charge = default["charge"]
    hf_tol = default["hf_tol"]
    cc_tol = default["cc_tol"]

    basis = cf.PyscfBasis(geometry, basis_name, charge=charge).pyscf_hartree_fock()
    basis.from_restricted()

    cc_run_args = {"tol": cc_tol, "vocal": False}
    if not "Q" in CC.__name__:
        cc_run_args["include_l"] = True

    cc = CC(basis).run(**cc_run_args)

    tdcc = cf.TimeDependentCoupledCluster(cc, time, integrator=integrator, integrator_args=integrator_args)
    tdcc.external_one_body = pulse
    tdcc.sampler = sampler

    tdcc.run(vocal=True)
    results = tdcc.results

    filename = f"{CC.__name__}_{molecule_name}_{basis_name}_Tend={time[1]}_dt={time[2]}_{integrator}_{pulse.__class__.__name__}.npz"
    print(f"DONE: {CC.__name__} {molecule_name} {pulse.__class__.__name__} {integrator}")

    np.savez(f"dat/{filename}", **results)


def delta_kick(molecule_name, basis_name):
    u = np.array([1.,0.,0.])
    F_str = 1e-3
    dt = 0.025
    t_end = 0.5
    integrator = "Rk4Integrator"
    integrator_args = {"dt": dt}

    time = (0, t_end, dt)
    pulse = cf.pulse.DeltaKick(u, F_str, dt)
    sampler = cf.sampler.DipoleSampler()

    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.CCSD, integrator, integrator_args=integrator_args)
    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.QCCSD, integrator, integrator_args=integrator_args)

    integrator = "GaussIntegrator"
    integrator_args = {"s": 3, "maxit": 20, "eps": 1e-6, "method": "A", "mu": 1.75}
    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.CCSD, integrator, integrator_args=integrator_args)
    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.QCCSD, integrator, integrator_args=integrator_args)

def sin2_pulse(molecule_name, basis_name):
    u = np.array([1.,0.,0.])
    F_str = 1e-2
    dt = 0.1
    omega = 0.2
    tprime = 2*np.pi/omega
    t_end = 0.5
    integrator = "Rk4Integrator"
    integrator_args = {"dt": dt}

    time = (0, t_end, dt)
    pulse = cf.pulse.Sin2(u, F_str, omega, tprime)
    sampler = cf.sampler.DipoleSampler()

    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.CCSD, integrator, integrator_args=integrator_args)
    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.QCCSD, integrator, integrator_args=integrator_args)

    integrator = "GaussIntegrator"
    integrator_args = {"s": 3, "maxit": 20, "eps": 1e-6, "method": "A", "mu": 1.75}
    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.CCSD, integrator, integrator_args=integrator_args)
    run_cc(molecule_name, basis_name, time, pulse, sampler, cf.QCCSD, integrator, integrator_args=integrator_args)

def main():
    molecule_name, basis_name = "be", "cc-pVDZ"
    delta_kick(molecule_name, basis_name)
    sin2_pulse(molecule_name, basis_name)

    molecule_name, basis_name = "lih", "cc-pVDZ"
    delta_kick(molecule_name, basis_name)
    sin2_pulse(molecule_name, basis_name)

if __name__ == "__main__":
    main()