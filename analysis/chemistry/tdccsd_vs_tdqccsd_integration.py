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

    # energy_diff_ccsd_qccsd(dts, omega)
    
    # energy_diff_after_pulse_off_integrators(dts, "CCD", omega=omega)
    # energy_diff_after_pulse_off_integrators(dts, "QCCD", omega=omega)

    energy_diff_after_pulse_off_methods(dts, "Rk4Integrator", omega)
    energy_diff_after_pulse_off_methods(dts, "GaussIntegrator", omega)

def energy_diff_ccsd_qccsd(dts, omega, integrators=["Rk4Integrator", "GaussIntegrator"]):
    fig, ax = plt.subplots()
    path = dat_path() / "he_integrator_test_ccd"

    cycle_length = (2 * np.pi / omega)

    ls = {
        "Rk4Integrator": "--",
        "GaussIntegrator": "-"
    }
    labels = [[None, rf"$\Delta t$ = {dt}"] for dt in dts]
    
    for i, dt in enumerate(dts):
        for j, integrator in enumerate(integrators):
            results_ccsd, = load_files(path=path, dt=dt, integrator=integrator, method="CCD")
            results_qccsd, = load_files(path=path, dt=dt, integrator=integrator, method="QCCD")
    
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
    path = dat_path() / "he_integrator_test_ccd"

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


def energy_diff_after_pulse_off_methods(dts, integrator, omega, methods=["CCD", "QCCD"]):
    fig, ax = plt.subplots()
    path = dat_path() / "he_integrator_test_ccd"

    cycle_length = 2*np.pi/omega
    ls = {
        "CCSD": "-",
        "CCD": "-",
        "QCCSD": "--",
        "QCCD": "--",
    }
    labels = [[rf"$\Delta t$ = {dt}", None] for dt in dts]

    for i, dt in enumerate(dts):
        for j, method in enumerate(methods):
            result, = load_files(path=path, dt=dt, integrator=integrator, method=method)

            time, energy = result["t"]/ cycle_length, result["energy"]
            pulse_off = np.argmin(np.abs(time - 2)) + 1

            energy = np.abs(energy - energy[pulse_off])
            time = time[pulse_off+1:]
            energy = energy[pulse_off+1:]

            ax.plot(time, energy, label=labels[i][j], ls=ls[method], color=pl.colors[i])

    ax.plot(np.nan, np.nan, label="CCSD", ls=ls["CCSD"], color="gray")
    ax.plot(np.nan, np.nan, label="QCCSD", ls=ls["QCCSD"], color="gray")

    ax.set(xlabel=r"$\omega t /2 \pi$", ylabel=r"$|E(t) - E(t')|$")
    ax.set_yscale("log")
    ax.legend(ncol=3)
    # pl.save(f"energy_diff_after_pulse_{integrator}")
    plt.show()

def plot_dipole(dts, integrator, methods=["CCD", "QCCD"]):
    path = dat_path() / "he_integ_test_ccd"
    omega = 2.87
    cycle_length = 2*np.pi/omega
    ls = {
        "CCSD": "-",
        "QCCSD": "--"
    }

    for i, dt in enumerate(dts):
        fig, ax = plt.subplots()
        for j, method in enumerate(methods):
            result, = load_files(path=path, dt=dt, integrator=integrator, method=method)

            time, energy = result["t"]/ cycle_length, result["energy"].imag
            pulse_off = np.argmin(np.abs(time - 2)) + 1

            ax.plot(time, energy, label=method, ls=ls[method], color=pl.colors[i])
        
        ax.legend()
        plt.show()


def plot_pulse():
    omega = 2.87

    cycle_length = (2 * np.pi / omega)
    t_end = 3*cycle_length
    tprime = 2*cycle_length

    u = np.array([1.0, 0.0, 0.0])
    F_str = 1
    pulse = cf.pulse.Sin2(u, F_str, omega, tprime)

    fig, ax = plt.subplots()

    time = np.linspace(0, t_end, 1000) 
    E = np.zeros_like(time)

    for i, t in enumerate(time): 
        E[i] = pulse.E(t)

    ax.plot(time/ cycle_length, E, label=r"$\omega = 2.87$")
    ax.set(xlabel=r"$\omega t / 2\pi$", ylabel="$E(t)\sin(\omega t)$ [au]")

    y1, y2 = ax.get_ylim()
    ax.vlines(2, y1*1.1, y2*1.1, ls="--", color="gray", label="$t' = 4\pi/\omega$")
    ax.legend()
    ax.set_ylim((y1,y2))
    plt.show()

if __name__ == '__main__':
    dts = np.array([0.1, 0.05, 0.01, 0.005])
    # run(dts)

    plot(dts)

    # plot_pulse()