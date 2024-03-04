import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt

def run_imag_timeprop(name, geometry, basis, CC, **kwargs):
    default = {
        "time": (0,5,0.1),
        "restricted": False,
        "integrator": "Rk4Integrator",
        "integrator_args": {"dt": 0.1},
        "hf_tol": 1e-8,
        "cc_tol": 1e-8,
        "intermediates": True,
        "path": "dat"
    }

    default.update(kwargs)

    time = default["time"]
    restricted = default["restricted"]
    integrator = default["integrator"]
    integrator_args = default["integrator_args"]
    hf_tol = default["hf_tol"]
    cc_tol = default["cc_tol"]
    path = default["path"]
    intermediates = default["intermediates"]


    b = cf.PyscfBasis(geometry, basis).pyscf_hartree_fock(tol=hf_tol)

    if not restricted:
        b.from_restricted()

    print(f"Running {CC.__name__}, {name}, {time}, {restricted = }, {intermediates = }")
    cc_gs = CC(b).run(tol=cc_tol, include_l=True)
    energy_gs = cc_gs.energy()

    cc = CC(b, intermediates=intermediates)
    itdcc = cf.ImaginaryTimeCoupledCluster(cc, cc_gs, time, integrator, integrator_args)
    itdcc.run(vocal=True)

    results = itdcc.results
    results["gs_energy"] = energy_gs

    filename = f"ImagTimeProp_{CC.__name__}_{name}_{basis}_{integrator}_{time[0]}_{time[1]}_{time[2]}_{intermediates}"
    np.savez(f"{path}/{filename}", **results)

def plot_imag_timeprop(filename):
    defaults = {
        "path": "dat",
    }

    path = defaults["path"]

    results = np.load(f"{path}/{filename}", allow_pickle=True)
    
    dE = np.abs(results["gs_energy"] - results["energy"])
    time = results["t"]

    fig, ax = plt.subplots()
    ax.plot(time, dE, label=r"$\Delta E$")
    # ax.plot(time, results["delta_t1"], label=r"$|\Delta t_1|$")
    # ax.plot(time, results["delta_t2"], label=r"$|\Delta t_2|$")
    # ax.plot(time, results["delta_l1"], label=r"$|\Delta l_1|$")
    # ax.plot(time, results["delta_l2"], label=r"$|\Delta l_2|$")
    ax.legend()
    ax.set_yscale("log")
    plt.show()

def main():
    # integrator = "GaussIntegrator"
    # integrator_args = {"s": 3, "maxit": 20, "eps": 1e-10, "method": "A", "mu": 1.75}
    integrator = "Rk4Integrator"
    integrator_args = {"dt": 0.01}
    
    run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.RCCSD, hf_tol=1e-8, cc_tol=1e-8, time=(0,50,0.1), integrator=integrator, integrator_args=integrator_args, intermediates=True, restricted=True)
    
    # run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.CCD, hf_tol=1e-8, cc_tol=1e-8, time=(0,10,0.1), integrator=integrator, integrator_args=integrator_args, intermediates=False)


    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.CCSD, hf_tol=1e-8, cc_tol=1e-8, time=(0,5.5,0.01), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.QCCSD, hf_tol=1e-8, cc_tol=1e-8, time=(0,5.5,0.01), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.CCSD,  time=(0,5.5,0.01), hf_tol=1e-8, cc_tol=1e-8, restricted=True)#, integrator=integrator, integrator_args=integrator_args)

    plot_imag_timeprop("ImagTimeProp_RCCSD_he_cc-pVDZ_Rk4Integrator_0_50_0.1_True.npz")
    # plot_imag_timeprop("ImagTimeProp_CCD_he_cc-pVDZ_Rk4Integrator_0_10_0.1_False.npz")
if __name__ == "__main__":
    main()