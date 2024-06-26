import quadratictheory as qt
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as pl

def run_imag_timeprop(name, geometry, basis, CC, **kwargs):
    default = {
        "time": (0,5,0.1),
        "restricted": False,
        "integrator": "Rk4Integrator",
        "integrator_args": {"dt": 0.1},
        "charge": 0,
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
    charge = default["charge"]


    b = qt.PyscfBasis(geometry, basis, charge=charge).pyscf_hartree_fock(tol=hf_tol)

    if not restricted:
        b.from_restricted()

    print(f"Running {CC.__name__}, {name}, {time}, {restricted = }")
    run_args = {"tol": cc_tol}
    if not "Q" in CC.__name__:
        run_args["include_l"] = True

    cc_gs = CC(b).run(**run_args)
    energy_gs = cc_gs.energy()

    cc = CC(b)
    itdcc = qt.ImaginaryTimeCoupledCluster(cc, cc_gs, time, integrator, integrator_args)
    itdcc.run_until_convergence(tol=cc_tol, vocal=False)

    results = itdcc.results
    results["gs_energy"] = energy_gs

    filename = f"ImagTimeProp_{CC.__name__}_{name}_{basis}_{integrator}_{time[2]}"
    np.savez(f"{path}/{filename}", **results)

def plot_imag_timeprop(filename, **kwargs):
    defaults = {
        "path": "dat",
        "tol": None,
        "orders": [2],
    }

    defautls = defaults.update(**kwargs)

    tol = defaults["tol"]
    path = defaults["path"]
    orders = defaults["orders"]

    results = np.load(f"{path}/{filename}", allow_pickle=True)
    
    dE_i = np.abs(results["energy"][1:] - results["energy"][0:-1])
    dE = np.abs(results["gs_energy"] - results["energy"])
    time = results["t"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    color1, color2 = pl.colors.copy(), pl.colors.copy()

    ax1 = axes[0]
    ax1_amps = ax1.twinx()
    ax1.set_yscale("log")
    ax1_amps.set_yscale("log")

    line1 = ax1.plot(time[1:], dE_i, color=color1.pop(0), label="$\delta E_i$")
    lines = []
    for i, order in enumerate(orders, start=1):
        line_t = ax1_amps.plot(time, results[f"rhs_t{order}"], color=color1.pop(0), label=r"$|\dot{\tau}_{" + str(order) + "}|$")
        line_l = ax1_amps.plot(time, results[f"rhs_l{order}"], color=color1.pop(0), label=r"$|\dot{\lambda}_{" + str(order) + "}|$")

        lines = lines + line_t + line_l
    
    lines = line1 + lines
    
    labels = [l.get_label() for l in lines]
    
    ax1.set(xlabel="$\pi_{\pm}$ [au]")
    ax1.set(ylabel="Energy [au]")
    ax1_amps.set(ylabel=r"$\delta \tau, \delta \lambda$")
    ax1.legend(lines, labels)
    ax1_amps.hlines(1e-10, time[-1]*0.8, time[-1]*1.05, color="gray", alpha=0.4, ls="--")

    ax2 = axes[1]
    ax2_amps = ax2.twinx()
    ax2.set_yscale("log")
    ax2_amps.set_yscale("log")

    line2 = ax2.plot(time, dE, color=color2.pop(0), label="$\delta E$")
    lines = []
    for i, order in enumerate(orders, start=1):
        line_t = ax2_amps.plot(time, results[f"delta_t{order}"], color=color2.pop(0), label=r"$\delta \tau_{" + str(order) + "}$")
        line_l = ax2_amps.plot(time, results[f"delta_l{order}"], color=color2.pop(0), label=r"$\delta \lambda_{" + str(order) + "}$")

        lines = lines + line_t + line_l

    lines = line2 + lines
    labels = [l.get_label() for l in lines]
   
    ax2_amps.hlines(1e-10, time[-1]*0.8, time[-1]*1.05, color="gray", alpha=0.4, ls="--")
    ax2.set(ylabel="Energy [au]")
    ax2_amps.set(ylabel=r"$\delta \tau, \delta \lambda$")
    ax2.set(xlabel="$\pi_{\pm}$ [au]")
    ax2.legend(lines, labels)
    plt.show()

    # ax.plot(time, dE, label=r"$\Delta E$")
    # ax.plot(time[1:], dE_i, label=r"$|E_{i+1} - E_i|$")
    # ax.plot(time, results["delta_t2"], label=r"$|\Delta t_2|$")
    # ax.plot(time, results["delta_l2"], label=r"$|\Delta l_2|$")

    # keys = list(results.keys())
    # if "delta_t1" in keys:
    #     ax.plot(time, results["delta_t1"], label=r"$|\Delta t_1|$")
    # if "delta_l1" in keys:
    #     ax.plot(time, results["delta_l1"], label=r"$|\Delta l_1|$")

    # if tol:
    #     ax.hlines(tol, time.min(), time.max(), color="gray", alpha=0.5, ls="--")
    
    # ax.set(xlabel=r"$\pi_{\pm}$")
    # ax.legend()
    # ax.set_yscale("log")
    # plt.show()

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
    run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", qt.CCD, hf_tol=1e-12, cc_tol=1e-12, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", qt.QCCD, hf_tol=1e-12, cc_tol=1e-12, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", qt.CCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,40,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", qt.QCCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,40,0.05), integrator=integrator, integrator_args=integrator_args)

    # Run Beryllium
    run_imag_timeprop("be", "be 0 0 0", "sto-3g", qt.CCD, hf_tol=1e-12, cc_tol=1e-12, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("be", "be 0 0 0", "sto-3g", qt.QCCD, hf_tol=1e-12, cc_tol=1e-12, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "sto-3g", qt.CCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "sto-3g", qt.QCCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)

    run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", qt.CCD, hf_tol=1e-12, cc_tol=1e-12, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", qt.QCCD, hf_tol=1e-12, cc_tol=1e-12, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", qt.CCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", qt.QCCSD, hf_tol=1e-12, cc_tol=1e-12, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)


    # Run Lithium-Hydride
    lih = "Li 0 0 0; H 0 0 3.0708047314"
    run_imag_timeprop("lih", lih, "sto-3g", qt.CCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("lih", lih, "sto-3g", qt.QCCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("lih", lih, "sto-3g", qt.CCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("lih", lih, "sto-3g", qt.QCCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)


    run_imag_timeprop("lih", lih, "cc-pVDZ", qt.CCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    run_imag_timeprop("lih", lih, "cc-pVDZ", qt.QCCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("lih", lih, "cc-pVDZ", qt.CCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("lih", lih, "cc-pVDZ", qt.QCCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)

    # Run Ch+
    chp = f"C 0.0 0.0 0.0; H 0.0 0.0 2.13713"
    run_imag_timeprop("chp", chp, "sto-3g", qt.CCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args, charge=1)
    run_imag_timeprop("chp", chp, "sto-3g", qt.QCCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args, charge=1)
    # run_imag_timeprop("chp", chp, "sto-3g", qt.CCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("chp", chp, "sto-3g", qt.QCCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)

    run_imag_timeprop("chp", chp, "cc-pVDZ", qt.CCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args, charge=1)
    run_imag_timeprop("chp", chp, "cc-pVDZ", qt.QCCD, hf_tol=1e-10, cc_tol=1e-10, time=(None,None,0.05), integrator=integrator, integrator_args=integrator_args, charge=1)
    # run_imag_timeprop("chp", chp, "cc-pVDZ", qt.CCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("chp", chp, "cc-pVDZ", qt.QCCSD, hf_tol=1e-10, cc_tol=1e-10, time=(0,100,0.05), integrator=integrator, integrator_args=integrator_args)



def plot():
    filename = "ImagTimeProp_CCSD_lih_cc-pVDZ_Rk4Integrator_0.05.npz"
    plot_imag_timeprop(filename, orders=[1,2], tol=1e-10)

def format_scientific_notation(number):
    if number == 0.0:
        return str(number)
    sigfig = 3

    mantissa = f"{number:.3e}"[:sigfig+2]
    exponent = int(np.floor(np.log10(number)))

    string = f"${mantissa} \cdot " + "10^{" + str(exponent) + "}$" 

    return string


def list_results(method):
    import os

    order = [2]
    if "S" in method:
        order = [1, 2]

    for filename in os.listdir("dat/"):
        l = filename.split("_")
        if len(l) != 6:
            continue

        met = l[1]
        sys = l[2]
        basis = l[3]


        if met == method:
            print(filename)
            results = np.load(f"dat/{filename}", allow_pickle=True)

            time_end = results["t"][-1]
            dE_end = np.abs(results["energy"][-1] - results["gs_energy"])            
            
            print(sys, basis, met)
            for o in order:
                rhs_t_last = results[f"delta_t{o}"][-1]
                rhs_l_last = results[f"delta_l{o}"][-1]

                
                rhs_t = format_scientific_notation(rhs_t_last)
                rhs_l = format_scientific_notation(rhs_l_last)

                print(f"& {rhs_t} & {rhs_l}", end="")

            print(f"\t{time_end:.2f}")

def plot_energy():
    path = "dat"
    filename = "ImagTimeProp_QCCSD_lih_cc-pVDZ_Rk4Integrator_0.05.npz"
    
    fig, ax = plt.subplots()

    results = np.load(f"{path}/{filename}", allow_pickle=True)

    t = results["t"]
    E = results["energy"]
    E_gs = results["gs_energy"]

    E_iter = np.abs(E[1:] - E[:-1])
    E_diff = np.abs(E - E_gs)

    ax.plot(t[1:], E_iter, label=r"$\delta E^{(n)}$")
    ax.plot(t, E_diff, label=r"$\delta E$")
    ax.set(xlabel=r"$\pi_{\pm}$ [a.u.]", ylabel=r"Energy [$E_h$]")
    ax.set_yscale("log")
    ax.legend()
    pl.save("Imag_time_prop_Be_QCCSD_energy")
    plt.show()

def plot_amplitudes():
    orders = [1, 2]
    path = "dat"
    filename = "ImagTimeProp_QCCSD_lih_cc-pVDZ_Rk4Integrator_0.05.npz"
    
    fig, ax = plt.subplots()

    results = np.load(f"{path}/{filename}", allow_pickle=True)

    time = results["t"]
    E = results["energy"]
    E_gs = results["gs_energy"]

    t, l = np.zeros_like(time), np.zeros_like(time)
    for order in orders:
        t += results[f"delta_t{order}"]
        l += results[f"delta_l{order}"]

    t_dot, l_dot = np.zeros_like(time), np.zeros_like(time)
    for order in orders:
        t_dot += results[f"rhs_t{order}"]
        l_dot += results[f"rhs_l{order}"]


    ax.plot(time, t, label=r"$\delta \tau_1 + \delta \tau_2$", c=pl.colors[0])
    ax.plot(time, l, label=r"$\delta \lambda_1 + \delta \lambda_2$", c=pl.colors[1])
    ax.plot(time, t_dot, label=r"$|\!|\dot{\tau}_1|\!|_F + |\!|\dot{\tau}_2|\!|_F$", c=pl.colors[2], ls="--")
    ax.plot(time, l_dot, label=r"$|\!|\dot{\lambda}_1|\!|_F + |\!|\dot{\lambda}_2|\!|_F$", c=pl.colors[3], ls="--")

    ax.legend()
    ax.set(xlabel="$\pi_{\pm}$ [a.u.]")
    ax.set_yscale("log")
    pl.save("Imag_time_prop_Be_QCCSD_amplitudes")
    plt.show()
if __name__ == "__main__":
    # run()
    # plot()
    plot_energy()
    plot_amplitudes()
    
    # list_results("CCSD")
    # list_results("QCCSD")

    # results = np.load("dat/ImagTimeProp_QCCSD_lih_cc-pVDZ_Rk4Integrator_0.05.npz")

    # print(list(results.keys()))

    # print(
    #     results["energy"][-1]
    # )
    # print(
    #     results["gs_energy"]
    # )