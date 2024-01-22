import clusterfock as cf
import matplotlib.pyplot as plt
import numpy as np

from coupled_cluster.ccsd import CCSD, TDCCSD
from coupled_cluster.ccd import CCD, TDCCD
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from rk4_integrator import Rk4Integrator
from scipy.integrate import complex_ode
from tdccsd_absoprtion_spectra.utils import sine_square_laser, get_pyscf_geometries, Gaussian_delta_pulse, Discrete_delta_pulse
import basis_set_exchange as bse
import tqdm

def pulse(t, basis, dt, F_str, direction):
    if t < dt:
        return F_str/dt * -basis.r[direction]
    else:
        return 0

def sampler(basis):
    return {"r": basis.r}

def run_linear_cc(params, filename=None, methods=["CCD", "CCSD"]):
    m = {"CCD": cf.CCD, "CCSD": cf.CCSD}

    basis = cf.PyscfBasis("Be 0 0 0", "cc-pVDZ")
    hf = cf.HF(basis).run()
    basis.change_basis(hf.C)
    basis.from_restricted()

    dt = params["dt"]
    t_end = params["t_end"]
    F_str = params["F_str"]
    tol = params["tol"]
    direction = params["direction"]
    time = (0, t_end, dt)

    for method in methods:
        CC = m[method]
    
        cc = CC(basis).run(vocal=True, include_l=True, tol=tol)

        tdcc = cf.TimeDependentCoupledCluster(cc, time)
        tdcc.external_one_body = lambda t, basis: pulse(t, basis, dt=dt, F_str=F_str, direction=0)
        tdcc.one_body_sampler = sampler
        results = tdcc.run(vocal=True)

        if filename:
            np.savez(f"{filename}_{method}", **results)

def run_quadratic_cc(params, filename=None, methods=["QCCD", "QCCSD"]):
    m = {"QCCD": cf.QCCD, "QCCSD": cf.QCCSD}

    basis = cf.PyscfBasis("Be 0 0 0", "cc-pVDZ")
    hf = cf.HF(basis).run()
    basis.change_basis(hf.C)
    basis.from_restricted()

    dt = params["dt"]
    t_end = params["t_end"]
    F_str = params["F_str"]
    tol = params["tol"]
    direction = params["direction"]
    time = (0, t_end, dt)

    for method in methods:
        CC = m[method]
    
        cc = CC(basis).run(vocal=True, tol=tol)

        tdcc = cf.TimeDependentCoupledCluster(cc, time)
        tdcc.external_one_body = lambda t, basis: pulse(t, basis, dt=dt, F_str=F_str, direction=direction)
        tdcc.one_body_sampler = sampler
        results = tdcc.run(vocal=True)

        if filename:
            np.savez(f"{filename}_{method}", **results)

def run_hyqd_cc(params, filename=None, method="HYQD_CCSD"):
    m = {"HYQD_CCD": CCD, "HYQD_CCSD": CCSD}
    m_td = {"HYQD_CCD": TDCCD, "HYQD_CCSD": TDCCSD}

    geometries = get_pyscf_geometries()

    # System and basis parameters
    name = "be"
    basis = "cc-pvdz"
    basis_set = bse.get_basis(basis, fmt='nwchem')
    charge = 0

    # Laser pulse parameters
    dt = params["dt"]
    tfinal = params["t_end"]
    F_str = params["F_str"]
    ground_state_tolerance = params["tol"]
    polarization_direction = params["direction"]

    integrator = "rk4"
    molecule = geometries[name]

    system = construct_pyscf_system_rhf(
        molecule=molecule,
        basis=basis_set,
        add_spin=True,
        anti_symmetrize=True,
        charge=charge,
    )


    polarization = np.zeros(3)
    polarization[polarization_direction] = 1

    num_steps = int(tfinal / dt) + 1
    time_points = np.linspace(0, tfinal, num_steps)

    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            Discrete_delta_pulse(F_str=F_str, dt=dt),
            polarization_vector=polarization,
        )
    )

    cc = m[method](system)
    cc.compute_ground_state(
        t_kwargs=dict(tol=ground_state_tolerance),
        l_kwargs=dict(tol=ground_state_tolerance),
    )

    tdcc = m_td[method](system)

    print(f"Rk4 integrator is used with dt: {dt}")
    r = complex_ode(tdcc).set_integrator("Rk4Integrator", dt=dt)
    r.set_initial_value(cc.get_amplitudes(get_t_0=True).asarray())

    energy = np.zeros(num_steps, dtype=np.complex128)
    dipole_moment = np.zeros((num_steps, 3), dtype=np.complex128)

    # Set initial values
    energy[0] = tdcc.compute_energy(r.t, r.y)

    for j in range(3):
        dipole_moment[0, j] = tdcc.compute_one_body_expectation_value(
            r.t,
            r.y,
            system.dipole_moment[j],
        )

    for i in tqdm.tqdm(range(num_steps - 1)):

        r.integrate(r.t + dt)

        for j in range(3):
            dipole_moment[i + 1, j] = tdcc.compute_one_body_expectation_value(
                r.t,
                r.y,
                system.dipole_moment[j],
            )

        energy[i+1] = tdcc.compute_energy(r.t, r.y)

    samples = dict()
    samples["t"] = time_points
    samples["r"] = dipole_moment
    samples["energy"] = energy

    np.savez(f"{filename}_{method}",
    **samples,
    )

def combine_expvals(results1, results2, results3, expval_key):
    combined = results1.copy()
    combined.update(results2)
    combined.update(results3)

    all_methods, all_expvals = [], []
    for method, results in combined.items():
        all_methods.append(method)
        if expval_key == "r":
            all_expvals.append(results[expval_key][:,0])
        else:
            all_expvals.append(results[expval_key])

    return all_methods, all_expvals


def compare(filename):
    methods_linear = ["CCD", "CCSD", ]
    methods_quadratic = ["QCCD", "QCCSD"]
    methods_hyqd = ["HYQD_CCD", "HYQD_CCSD"]

    results_linear = {}
    for method in methods_linear:
        try:
            results_linear[method] = np.load(f"{filename}_{method}.npz", allow_pickle=True)
            print(f"Load {method} OK")
        except:
            print(f"Load {method} MISSING")
            pass

    results_quadratic = {}
    for method in methods_quadratic:
        try:
            results_quadratic[method] = np.load(f"{filename}_{method}.npz", allow_pickle=True)
            print(f"Load {method} OK")
        except:
            print(f"Load {method} MISSING")
            pass

    results_hyqd = {}
    for method in methods_hyqd:
        try:
            results_hyqd[method] = np.load(f"{filename}_{method}.npz", allow_pickle=True)
            print(f"Load {method} OK")
        except:
            print(f"Load {method} MISSING")
            pass

    t = results_linear[methods_linear[0]]["t"]
    to_show = ["energy", "r"]

    for expval_key in to_show:
        fig, ax = plt.subplots()
        methods, expvals = combine_expvals(results_linear, results_quadratic, results_hyqd, expval_key)
        
        for method, expval in zip(methods, expvals):
            ax.plot(t, expval.real, label=method)
        ax.set_title(expval_key)
        ax.legend()
        plt.show()

def main():
    params = {
        "dt" : 0.1,
        "t_end" : 10.0,
        "F_str" : 1e-3,
        "tol" : 1e-10,
        "direction" : 0,
    }
    filename = "dat/short_time"

    # run_linear_cc(params, filename=filename)
    run_quadratic_cc(params, filename=filename, methods=["QCCD", "QCCSD"])

    # run_hyqd_cc(params, filename=filename, method="HYQD_CCD")
    # run_hyqd_cc(params, filename=filename, method="HYQD_CCSD")

    compare(filename)

if __name__ == '__main__':
    main()