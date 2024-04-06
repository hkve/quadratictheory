import numpy as np
from signals_tdcc import run_cc
from utils.misc import load_files, dat_path
import clusterfock as cf


def run(dts, omega=2.87, atom_name="he"):
    u = np.array([1.0, 0.0, 0.0])
    F_str = 1.0

    cycle_length = (2 * np.pi / omega)
    t_end = 3*cycle_length
    tprime = 2*cycle_length

    integrators = ["Rk4Integrator", "GaussIntegrator"]
    integrator_args = {}

    METHODS = [cf.CCSD, cf.QCCSD]

    for i, dt in enumerate(dts):
        print(f"Starting {dt = } ", end="")
        for j, integrator in enumerate(integrators):
            print(f" {integrator = } ", end="")
            
            time = (0, t_end, dt)
            pulse = cf.pulse.Sin2(u, F_str, omega, tprime)
            sampler = cf.sampler.DipoleSamplerExpanded()

            if integrator == "Rk4Integrator":
                integrator_args = {"dt": dt}
            elif integrator == "GaussIntegrator":
                integrator_args = {"s": 3, "maxit": 20, "eps": 1e-10, "method": "A", "mu": 1.75}

            for METHOD in METHODS:
                print(f"method = {METHOD.__name__}")
                run_cc(
                    atom_name,
                    "cc-pVDZ",
                    time,
                    pulse,
                    sampler,
                    METHOD,
                    integrator,
                    integrator_args=integrator_args,
                    subfolder="he_integrator_test_new"
                )

from IPython import embed
import plotting.plot_utils  as pl
import matplotlib.pyplot as plt
def plot(dts, omega=2.87, atom_name="he"):

    energy_diff_ccsd_qccsd(dts, omega)
    energy_diff_after_pulse_off_integrators(dts, "CCSD", omega=omega)
    energy_diff_after_pulse_off_integrators(dts, "QCCSD", omega=omega)

    energy_diff_after_pulse_off_methods(dts, "Rk4Integrator", omega)
    energy_diff_after_pulse_off_methods(dts, "GaussIntegrator", omega)

def energy_diff_ccsd_qccsd(dts, omega, integrators=["Rk4Integrator", "GaussIntegrator"]):
    fig, ax = plt.subplots()
    path = dat_path() / "he_integrator_test_new"

    cycle_length = (2 * np.pi / omega)

    ls = {
        "Rk4Integrator": "--",
        "GaussIntegrator": "-"
    }
    labels = [[None, rf"$\Delta t$ = {dt}"] for dt in dts]

    for i, dt in enumerate(dts):
        for j, integrator in enumerate(integrators):
            results_ccsd, = load_files(path=path, dt=dt, integrator=integrator, method="CCSD")
            results_qccsd, = load_files(path=path, dt=dt, integrator=integrator, method="QCCSD")
    
            realtive_error = (results_ccsd["energy"] - results_qccsd["energy"]).real
            realtive_error = np.abs(realtive_error)
            time = results_ccsd["t"] / cycle_length

            ax.plot(time, realtive_error, label=labels[i][j], ls=ls[integrator], c=pl.colors[i])

    ax.plot(np.nan, np.nan, label="RK 4", ls="--", color="gray")
    ax.plot(np.nan, np.nan, label="Gauss", ls="-", color="gray")

    ax.set_yscale("log")
    ax.set(ylim=(1e-14, 0.5))
    ax.set(xlabel=r"$\omega t / 2 \pi$", ylabel=r"$|E_{CCSD} - E_{QCCSD}|$ [au]")
    ax.legend(ncol=3, loc="upper left", fontsize=12.56)
    plt.show()

def energy_diff_after_pulse_off_integrators(dts, method, omega, integrators=["Rk4Integrator", "GaussIntegrator"]):
    fig, ax = plt.subplots()
    path = dat_path() / "he_integrator_test_new"

    cycle_length = 2*np.pi/omega
    ls = {
        "Rk4Integrator": "--",
        "GaussIntegrator": "-"
    }
    labels = [[None, rf"$\Delta t$ = {dt}"] for dt in dts]

    for i, dt in enumerate(dts):
        for j, integrator in enumerate(integrators):
            result, = load_files(path=path, dt=dt, integrator=integrator, method=method)

            time, energy = result["t"]/ cycle_length, result["energy"]
            pulse_off = np.argmin(np.abs(time - 2)) + 1

            energy = np.abs(energy - energy[pulse_off])
            time = time[pulse_off+1:]
            energy = energy[pulse_off+1:]

            ax.plot(time, energy, label=labels[i][j], ls=ls[integrator], color=pl.colors[i])

    ax.plot(np.nan, np.nan, label="RK 4", ls="--", color="gray")
    ax.plot(np.nan, np.nan, label="Gauss", ls="-", color="gray")

    ax.set(title=method, xlabel=r"$\omega t /2 \pi$", ylabel=r"$|E(t) - E(t')|$")
    ax.set_yscale("log")
    ax.legend(ncol=3, loc="upper left")
    plt.show()


def energy_diff_after_pulse_off_methods(dts, integrator, omega, methods=["CCSD", "QCCSD"]):
    fig, ax = plt.subplots()
    path = dat_path() / "he_integrator_test_new"

    cycle_length = 2*np.pi/omega
    ls = {
        "CCSD": "--",
        "QCCSD": "-"
    }
    labels = [[None, rf"$\Delta t$ = {dt}"] for dt in dts]

    for i, dt in enumerate(dts):
        for j, method in enumerate(methods):
            result, = load_files(path=path, dt=dt, integrator=integrator, method=method)

            time, energy = result["t"]/ cycle_length, result["energy"]
            pulse_off = np.argmin(np.abs(time - 2)) + 1

            energy = np.abs(energy - energy[pulse_off])
            time = time[pulse_off+1:]
            energy = energy[pulse_off+1:]

            ax.plot(time, energy, label=labels[i][j], ls=ls[method], color=pl.colors[i])

    ax.plot(np.nan, np.nan, label="CCSD", ls="--", color="gray")
    ax.plot(np.nan, np.nan, label="QCCSD", ls="-", color="gray")

    ax.set(title=integrator, xlabel=r"$\omega t /2 \pi$", ylabel=r"$|E(t) - E(t')|$")
    ax.set_yscale("log")
    ax.legend(ncol=3)
    plt.show()

if __name__ == '__main__':
    dts = np.array([0.1, 0.05, 0.01, 0.005])
    # run(dts)

    plot(dts)