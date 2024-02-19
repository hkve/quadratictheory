from plotting.expvals_imag_part import plot_imag_part, plot_rho_hermiticity
from utils.misc import load_files
import numpy as np 

def plot_sin2_diagnostic():
    method = ["CCSD", "QCCSD"]
    results = load_files(method=method, name="he", basis="cc-pVDZ", dt=0.01, integrator="Rk4Integrator", pulse="Sin2", polarisation=0)
    
    plot_imag_part(results, "energy", ylabel=r"$Im(\langle E \rangle)$ [au]")
    # plot_imag_part(results, "r", 2, ylabel=r"$Im(\langle z \rangle)$ [au]")
    # plot_rho_hermiticity(results)

    # method = ["CCSD", "QCCSD"]
    # results = load_files(method=method, name="be", basis="cc-pVDZ", Tend=100, dt=0.1, integrator="Rk4Integrator", pulse="Sin2")
    # plot_imag_part(results, "r", 2, ylabel=r"$Im(\langle z \rangle)$ [au]")
    # plot_imag_part(results, "energy", ylabel=r"$Im(\langle E \rangle)$ [au]")
    # plot_rho_hermiticity(results)



if __name__ == "__main__":
    plot_sin2_diagnostic()