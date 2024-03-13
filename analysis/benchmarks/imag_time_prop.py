import clusterfock as cf
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as pl

def run_imag_timeprop(name, geometry, basis, CC, **kwargs):
    default = {
        "time": (0,5,0.1),
        "restricted": False,
        "integrator": "Rk4Integrator",
        "integrator_args": {"dt": 0.1},
        "hf_tol": 1e-8,
        "cc_tol": 1e-8,
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


    b = cf.PyscfBasis(geometry, basis).pyscf_hartree_fock(tol=hf_tol)

    if not restricted:
        b.from_restricted()

    print(f"Running {CC.__name__}, {name}, {time}, {restricted = }")
    run_args = {"tol": cc_tol}
    if not "Q" in CC.__name__:
        run_args["include_l"] = True

    cc_gs = CC(b).run(**run_args)
    energy_gs = cc_gs.energy()

    cc = CC(b)
    itdcc = cf.ImaginaryTimeCoupledCluster(cc, cc_gs, time, integrator, integrator_args)
    itdcc.run(vocal=True)

    results = itdcc.results
    results["gs_energy"] = energy_gs

    filename = f"ImagTimeProp_{CC.__name__}_{name}_{basis}_{integrator}_{time[0]}_{time[1]}_{time[2]}"
    np.savez(f"{path}/{filename}", **results)

def plot_imag_timeprop(filename, **kwargs):
    defaults = {
        "path": "dat",
        "tol": None,
    }

    defautls = defaults.update(**kwargs)

    tol = defaults["tol"]
    path = defaults["path"]

    results = np.load(f"{path}/{filename}", allow_pickle=True)
    
    dE_i = np.abs(results["energy"][1:] - results["energy"][0:-1])

    dE = np.abs(results["gs_energy"] - results["energy"])
    time = results["t"]

    fig, ax = plt.subplots()
    ax.plot(time, dE, label=r"$\Delta E$")
    ax.plot(time[1:], dE_i, label=r"$|E_{i+1} - E_i|$")
    ax.plot(time, results["delta_t2"], label=r"$|\Delta t_2|$")
    ax.plot(time, results["delta_l2"], label=r"$|\Delta l_2|$")

    keys = list(results.keys())
    if "delta_t1" in keys:
        ax.plot(time, results["delta_t1"], label=r"$|\Delta t_1|$")
    if "delta_l1" in keys:
        ax.plot(time, results["delta_l1"], label=r"$|\Delta l_1|$")

    if tol:
        ax.hlines(tol, time.min(), time.max(), color="gray", alpha=0.5, ls="--")
    
    ax.set(xlabel=r"$\pi_{\pm}$")
    ax.legend()
    ax.set_yscale("log")
    plt.show()

def plot_two_imag_timeprop(filenames, **kwargs):
    defaults = {
        "path": "dat",
        "tol": None,
    }
    defautls = defaults.update(**kwargs)

    tol = defaults["tol"]
    path = defaults["path"]
    ls = ["-", "--", "-."]

    fig, ax = plt.subplots()
    for i, filename in enumerate(filenames):
        results = np.load(f"{path}/{filename}", allow_pickle=True)

        dE_i = np.abs(results["energy"][1:] - results["energy"][0:-1])
        dE = np.abs(results["gs_energy"] - results["energy"])
        time = results["t"]

        ax.plot(time, dE, label=r"$\Delta E$", color=pl.colors[i], ls=ls[0])
        ax.plot(time[1:], dE_i, label=r"$|E_{i+1} - E_i|$", color=pl.colors[i], ls=ls[1])

    ax.set(xlabel=r"$\pi_{\pm} = \pm it$ [au]", ylabel="Energy [au]")
    ax.legend()
    ax.set_yscale("log")
    plt.show()

    fig, ax = plt.subplots()
    for i, filename in enumerate(filenames):
        results = np.load(f"{path}/{filename}", allow_pickle=True)
        time = results["t"]

        ax.plot(time, results["delta_t2"], label=r"$|\Delta t_2|$", ls=ls[i])
        ax.plot(time, results["delta_l2"], label=r"$|\Delta l_2|$", ls=ls[i])

        keys = list(results.keys())
        if "delta_t1" in keys:
            ax.plot(time, results["delta_t1"], label=r"$|\Delta t_1|$", ls=ls[i])
        if "delta_l1" in keys:
            ax.plot(time, results["delta_l1"], label=r"$|\Delta l_1|$", ls=ls[i])

        if tol:
            ax.hlines(tol, time.min(), time.max(), color="gray", alpha=0.5, ls="--")
        
        ax.set(xlabel=r"$\pi_{\pm}$")
        ax.legend()
        ax.set_yscale("log")
    plt.show()

def run():
    # integrator = "GaussIntegrator"
    # integrator_args = {"s": 3, "maxit": 20, "eps": 1e-10, "method": "A", "mu": 1.75}
    integrator = "Rk4Integrator"
    integrator_args = {"dt": 0.05}
    
    # Run Helium
    run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.CCD, hf_tol=1e-12, cc_tol=1e-12, time=(0,40,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.CCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,40,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.QCCD, hf_tol=1e-12, cc_tol=1e-12, time=(0,40,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.QCCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,40,0.05), integrator=integrator, integrator_args=integrator_args)

    # Run Beryllium
    run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.CCD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.CCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.QCCD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.QCCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)

    # Run Lithium-Hydride
    lih = "Li 0 0 0; H 0 0 3.0708047314"
    run_imag_timeprop("lih", lih, "cc-pVDZ", cf.CCD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("lih", lih, "cc-pVDZ", cf.CCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("lih", lih, "cc-pVDZ", cf.QCCD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("lih", lih, "cc-pVDZ", cf.QCCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)


def plot():
    plot_imag_timeprop("ImagTimeProp_CCSD_lih_cc-pVDZ_Rk4Integrator_0_200_0.05.npz", tol=1e-10)

    # filenames = [
    #     "ImagTimeProp_CCD_lih_cc-pVDZ_Rk4Integrator_0_100_0.05.npz",
    #     "ImagTimeProp_QCCD_lih_cc-pVDZ_Rk4Integrator_0_100_0.05.npz"
    # ]
    # plot_two_imag_timeprop(filenames, tol=1e-10)
if __name__ == "__main__":
    # run()
    plot()