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

    cc = CC(b)
    itdcc = cf.ImaginaryTimeCoupledCluster(cc, time, integrator, integrator_args)
    itdcc.run(vocal=True)

    results = itdcc.results

    cc = CC(b).run(tol=cc_tol)
    results["gs_energy"] = cc.energy()

    filename = f"ImagTimeProp_{CC.__name__}_{name}_{basis}_{integrator}_{time[0]}_{time[1]}_{time[2]}"
    np.savez(f"{path}/{filename}", **results)

def plot_imag_timeprop(filename):
    defaults = {
        "path": "dat",
    }

    path = defaults["path"]

    results = np.load(f"{path}/{filename}", allow_pickle=True)

    dE = np.abs(results["gs_energy"] - results["energy"])
    time = results["t"]

    cut = None
    fig, ax = plt.subplots()
    ax.plot(time[:cut], dE[:cut])
    ax.set_yscale("log")
    plt.show()

def main():
    # integrator = "GaussIntegrator"
    # integrator_args = {"s": 3, "maxit": 20, "eps": 1e-10, "method": "A", "mu": 1.75}
    
    # run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.CCSD, hf_tol=1e-8, cc_tol=1e-8, time=(0,5.5,0.01), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("he", "he 0 0 0", "cc-pVDZ", cf.QCCSD, hf_tol=1e-8, cc_tol=1e-8, time=(0,5.5,0.01), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.CCSD, hf_tol=1e-8, cc_tol=1e-8, time=(0,5.5,0.01), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.QCCSD, hf_tol=1e-8, cc_tol=1e-8, time=(0,5.5,0.01), integrator=integrator, integrator_args=integrator_args)
    # run_imag_timeprop("be", "be 0 0 0", "cc-pVDZ", cf.CCSD,  time=(0,5.5,0.01), hf_tol=1e-8, cc_tol=1e-8, restricted=True)#, integrator=integrator, integrator_args=integrator_args)

    plot_imag_timeprop("ImagTimeProp_CCSD_be_cc-pVDZ_Rk4Integrator_0_5.5_0.01.npz")
if __name__ == "__main__":
    main()