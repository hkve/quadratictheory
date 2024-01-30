import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt
from tdccsd_absoprtion_spectra.utils import get_pyscf_geometries


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
    t_end = 100
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
    t_end = 100
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


def load_systems(molecule_name, basis_name, pulse, methods, integrators):
    import os
    files = os.listdir("dat/")

    results = {method: {} for method in methods}

    for file in files:
        if not f"_{molecule_name}_" in file: continue 
        if not f"_{basis_name}_" in file: continue
        if not f"{pulse}" in file: continue

        for method in methods:
            for integrator in integrators:
                if method in file and integrator in file:
                    results[method][integrator] = np.load(f"dat/{file}", allow_pickle=True)
    return results

def plot_compare(molecule_name, basis_name, pulse):
    # integrators = ["GaussIntegrator", "Rk4Integrator"]
    integrators = ["GaussIntegrator"]
    methods = ["CCSD", "QCCSD"]

    results = load_systems(molecule_name, basis_name, pulse, methods, integrators)

    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5,3], figsize=(10,8))
    fig.suptitle(f"Energy for {molecule_name}({basis_name}) {pulse} pulse ", fontsize=16)
    for integrator in integrators:
        t_cc, e_cc = results["CCSD"][integrator]["t"], results["CCSD"][integrator]["energy"] 
        t_qcc, e_qcc = results["QCCSD"][integrator]["t"], results["QCCSD"][integrator]["energy"] 

        ax[0].plot(t_cc, e_cc, label=f"CCSD: {integrator}")
        ax[0].plot(t_qcc, e_qcc, label=f"QCCSD: {integrator}")

        ax[1].plot(t_cc, np.abs(e_qcc - e_cc), label=f"DIFF: {integrator}")
    
    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale("log")

    plt.show()


def main():
    molecule_name, basis_name = "be", "cc-pVDZ"
    # delta_kick(molecule_name, basis_name)
    # sin2_pulse(molecule_name, basis_name)

    # molecule_name, basis_name = "lih", "cc-pVDZ"
    # delta_kick(molecule_name, basis_name)
    # sin2_pulse(molecule_name, basis_name)

    plot_compare(molecule_name, basis_name, "DeltaKick")

if __name__ == "__main__":
    main()