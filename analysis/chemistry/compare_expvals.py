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

# def pulse(t, basis, dt, F_str, direction):
#     if t < dt:
#         return F_str/dt * -basis.r[direction]
#     else:
#         return 0

def sampler(basis):
    return {"r": basis.r}

def run_linear_cc(params, filename=None, methods=["CCD", "CCSD"]):
    m = {"CCD": cf.CCD, "CCSD": cf.CCSD}

    molecule = params["molecule"]
    basis = params["basis"]
    system = cf.PyscfBasis(molecule, basis).pyscf_hartree_fock()
    # hf = cf.HF(basis).run()
    # basis.change_basis(hf.C)
    system.from_restricted()

    dt = params["dt"]
    t_end = params["t_end"]
    F_str = params["F_str"]
    tol = params["tol"]
    direction = params["direction"]
    omega = params["omega"]
    tprime = 2*np.pi / omega
    time = (0, t_end, dt)

    pulse = cf.pulse.DeltaKick(
        u = np.array([1,0,0]),
        dt = dt,
        F_str = F_str,
    )

    for method in methods:
        CC = m[method]
    
        cc = CC(system).run(vocal=False, include_l=True, tol=tol)

        tdcc = cf.TimeDependentCoupledCluster(cc, time)
        tdcc.external_one_body = pulse
        tdcc.one_body_sampler = sampler
        results = tdcc.run(vocal=True)

        if filename:
            np.savez(f"{filename}_{method}", **results)

def run_quadratic_cc(params, filename=None, methods=["QCCD", "QCCSD"]):
    m = {"QCCD": cf.QCCD, "QCCSD": cf.QCCSD}

    molecule = params["molecule"]
    basis = params["basis"]

    basis = cf.PyscfBasis(molecule, basis).pyscf_hartree_fock()
    # hf = cf.HF(basis).run()
    # basis.change_basis(hf.C)
    basis.from_restricted()

    dt = params["dt"]
    t_end = params["t_end"]
    F_str = params["F_str"]
    tol = params["tol"]
    direction = params["direction"]
    time = (0, t_end, dt)
    omega = params["omega"]
    tprime = 2*np.pi / omega

    pulse = cf.pulse.DeltaKick(
        u = np.array([1,0,0]),
        dt = dt,
        F_str = F_str,
    )

    for method in methods:
        CC = m[method]
    
        cc = CC(basis).run(vocal=False, tol=tol)
        tdcc = cf.TimeDependentCoupledCluster(cc, time)
        tdcc.external_one_body = pulse
        tdcc.one_body_sampler = sampler
        results = tdcc.run(vocal=True)

        if filename:
            np.savez(f"{filename}_{method}", **results)

def run_hyqd_cc(params, filename=None, method="HYQD_CCSD"):
    m = {"HYQD_CCD": CCD, "HYQD_CCSD": CCSD}
    m_td = {"HYQD_CCD": TDCCD, "HYQD_CCSD": TDCCSD}

    # basis_set = bse.get_basis(basis, fmt='nwchem')
    charge = 0

    # Laser pulse parameters
    dt = params["dt"]
    tfinal = params["t_end"]
    F_str = params["F_str"]
    ground_state_tolerance = params["tol"]
    polarization_direction = params["direction"]
    omega = params["omega"]
    tprime = 2*np.pi / omega
    molecule = params["molecule"]
    basis = params["basis"]

    integrator = "rk4"

    system = construct_pyscf_system_rhf(
        molecule=molecule,
        basis=basis,
        add_spin=True,
        anti_symmetrize=True,
        charge=charge,
    )


    polarization = np.zeros(3)
    polarization[polarization_direction] = 1

    num_steps = int(tfinal / dt) + 1
    time_points = np.linspace(0, tfinal, num_steps)
    # Discrete_delta_pulse(F_str=F_str, dt=dt), polarization_vector=polarization,
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
            make_hermitian=False
        )

    for i in tqdm.tqdm(range(num_steps - 1)):

        r.integrate(r.t + dt)

        for j in range(3):
            dipole_moment[i + 1, j] = tdcc.compute_one_body_expectation_value(
                r.t,
                r.y,
                system.dipole_moment[j],
                make_hermitian=False
            )

        energy[i+1] = tdcc.compute_energy(r.t, r.y)

    samples = dict()
    samples["t"] = time_points
    samples["r"] = dipole_moment
    samples["energy"] = energy + system.nuclear_repulsion_energy

    np.savez(f"{filename}_{method}",
    **samples,
    )

def combine_expvals(results1, results2, results3, expval_key):
    combined = results1.copy()
    combined.update(results2)
    combined.update(results3)

    all_methods, all_expvals, all_times = [], [], []
    for method, results in combined.items():
        all_methods.append(method)
        all_times.append(results["t"])
        if expval_key == "r":
            all_expvals.append(results[expval_key][:,0])
        else:
            all_expvals.append(results[expval_key])
    return all_methods, all_expvals, all_times

def get_ls(name):
    if name.startswith("Q"): return "-"
    if name.startswith("HYQD"): return "--"
    return "-"

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

    to_show = ["energy", "r", "delta_rho"]

    for expval_key in to_show:
        fig, ax = plt.subplots()
        methods, expvals, times = combine_expvals(results_linear, results_quadratic, results_hyqd, expval_key)
        
        for method, expval, t in zip(methods, expvals, times):
            ax.plot(t, expval.real, label=method, ls=get_ls(method))
        
        ax.legend()
        ax.set_title(expval_key)
        plt.show()

def cc_diff(filename, method, **params):
    results1 = np.load(f"{filename}_{method}.npz", allow_pickle=True)
    # results2 = np.load(f"{filename}_HYQD_{method}.npz", allow_pickle=True)
    
    t1, t2 = results1["t"], None # , results2["t"]
    r1, r2 = results1["r"][:,0], None #results2["r"][:,0]
    e1, e2 = results1["energy"], None #results2["energy"]

    # fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5,3], figsize=(10,8))
    # fig.suptitle('Energy for  LiH sin2 pulse', fontsize=16)
    # ax[0].plot(t1, e1, label="CF: CCSD", c="k")
    # ax[0].plot(t2, e2, label="HYQD: CCSD", ls=":", c="r")
    # ax[0].legend()
    # ax[0].set(ylabel="Energy [au]")

    # ax[1].plot(t1, np.abs(e1-e2), label="DIFF")
    # ax[1].legend()
    # ax[1].set(xlabel="Time  [au]", ylabel="Energy [au]")
    # ax[1].set_yscale("log")
    # fig.tight_layout()
    # fig.savefig("to_haakon/lih_energy.pdf")
    # plt.show()

    # fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5,3], figsize=(10,8))
    # fig.suptitle('<x> for  LiH sin2 pulse', fontsize=16)
    # ax[0].plot(t1, r1, label="CF: CCSD", c="k")
    # ax[0].plot(t2, r2, label="HYQD: CCSD", ls=":", c="r")
    # ax[0].legend()
    # ax[0].set(ylabel="Distance [au]")

    # ax[1].plot(t1, np.abs(r1-r2), label="CCSD DIFF TO HYQD")
    # ax[1].legend()
    # ax[1].set(xlabel="Time  [au]", ylabel="Distance [au]")
    # ax[1].set_yscale("log")
    # fig.tight_layout()
    # fig.savefig("to_haakon/lih_r.pdf")
    # plt.show()

    results3 = np.load(f"{filename}_Q{method}.npz", allow_pickle=True)
    t3, r3, e3 = results3["t"], results3["r"][:,0], results3["energy"]

    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5,3], figsize=(10,8))
    fig.suptitle('Energy for  LiH sin2 pulse', fontsize=16)
    ax[0].plot(t3, e3, label="QCCSD", c="k")
    ax[0].plot(t1, e1, label="CCSD", ls=":", c="r")
    ax[0].legend()
    ax[0].set(ylabel="Energy [au]")

    ax[1].plot(t1, np.abs(e1-e3), label="DIFF")
    ax[1].legend()
    ax[1].set(xlabel="Time  [au]", ylabel="Energy [au]")
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig("to_haakon/lih_energy_QCCSD.pdf")
    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5,3], figsize=(10,8))
    fig.suptitle('<x> for  LiH sin2 pulse', fontsize=16)
    ax[0].plot(t3, r3, label="QCCSD", c="k")
    ax[0].plot(t1, r1, label="CCSD", ls=":", c="r")
    ax[0].legend()
    ax[0].set(ylabel="Distance [au]")

    ax[1].plot(t1, np.abs(r1-r3), label="CCSD-QCCSD DIFF")
    ax[1].legend()
    ax[1].set(xlabel="Time  [au]", ylabel="Distance [au]")
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig("to_haakon/lih_r_QCCSD.pdf")
    plt.show()

def main():
    params = {
        "molecule": get_pyscf_geometries()["be"],
        "basis": "cc-pvdz",
        "dt" : 0.025,
        "t_end" : 1,
        "F_str" : 1e-3,
        "tol" : 1e-10,
        "omega": 0.2,
        "direction" : 0,
    }
    filename = "dat/test_Be_delta"

    # run_linear_cc(params, filename=filename, methods=["CCSD"])
    # run_quadratic_cc(params, filename=filename, methods=["QCCSD"])

    # run_hyqd_cc(params, filename=filename, method="HYQD_CCD")
    # run_hyqd_cc(params, filename=filename, method="HYQD_CCSD")

    compare(filename)
    # cc_diff(filename, method="CCD")

if __name__ == '__main__':
    main()