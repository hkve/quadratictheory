import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt
from tdccsd_absoprtion_spectra.utils import get_pyscf_geometries


def _pack_as_list(thing):
    if type(thing) not in [list, tuple]:
        thing = [thing]
    return thing


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

    tdcc = cf.TimeDependentCoupledCluster(
        cc, time, integrator=integrator, integrator_args=integrator_args
    )
    tdcc.external_one_body = pulse
    tdcc.sampler = sampler

    tdcc.run(vocal=True)
    results = tdcc.results

    filename = f"{CC.__name__}_{molecule_name}_{basis_name}_Tend={time[1]}_dt={time[2]}_{integrator}_{pulse.__class__.__name__}.npz"
    print(f"DONE: {CC.__name__} {molecule_name} {pulse.__class__.__name__} {integrator}")

    # np.savez(f"dat/{filename}", **results)


def delta_kick(molecule_name, basis_name, **kwargs):
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


def sin2_pulse(molecule_name, basis_name, **kwargs):
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


def load_systems(molecule_name, basis_name, pulse, methods, integrators, dt=None):
    import os

    files = os.listdir("dat/")

    methods = _pack_as_list(methods)
    integrators = _pack_as_list(integrators)

    results = {method: {} for method in methods}

    for file in files:
        if not f"_{molecule_name}_" in file:
            continue
        if not f"_{basis_name}_" in file:
            continue
        if not f"{pulse}" in file:
            continue

        if dt is not None:
            if f"dt={dt}" not in file:
                continue

        for method in methods:
            for integrator in integrators:
                if file.startswith(method) and integrator in file:
                    print(f"LOADING {file}", method, integrator)
                    results[method][integrator] = np.load(f"dat/{file}", allow_pickle=True)

    return results


def plot_compare(molecule_name, basis_name, pulse, method1, method2, integrator, dt=None):
    results = load_systems(molecule_name, basis_name, pulse, [method1, method2], integrator, dt=dt)

    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5, 3], figsize=(10, 8))
    fig.suptitle(f"Energy for {molecule_name}({basis_name}) {pulse} pulse ", fontsize=16)

    t_cc, e_cc = results[method1][integrator]["t"], results[method1][integrator]["energy"]
    t_qcc, e_qcc = results[method2][integrator]["t"], results[method2][integrator]["energy"]

    ax[0].plot(t_cc, e_cc, label=f"{method1}: {integrator}")
    ax[0].plot(t_qcc, e_qcc, label=f"{method2}: {integrator}")
    ax[1].plot(t_cc, np.abs(e_qcc - e_cc), label=f"DIFF: {integrator}")

    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale("log")

    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5, 3], figsize=(10, 8))
    t_cc, r_cc = results[method1][integrator]["t"], results[method1][integrator]["r"][:, 0]
    t_qcc, r_qcc = results[method2][integrator]["t"], results[method2][integrator]["r"][:, 0]

    ax[0].plot(t_cc, r_cc, label=f"{method1}: {integrator}")
    ax[0].plot(t_qcc, r_qcc, label=f"{method2}: {integrator}")

    ax[1].plot(t_cc, np.abs(r_qcc - r_cc), label=f"DIFF: {integrator}")

    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale("log")
    plt.show()


def main():
    molecule_name, basis_name = "be", "cc-pVDZ"
    delta_kick("be", basis_name, charge=0)
    delta_kick("lih", basis_name, charge=0)
    delta_kick("chp", basis_name, charge=1)

    # sin2_pulse("be", basis_name)
    # sin2_pulse("lih", basis_name)

    # plot_compare(molecule_name, basis_name, "DeltaKick", dt=0.025)


if __name__ == "__main__":
    main()
