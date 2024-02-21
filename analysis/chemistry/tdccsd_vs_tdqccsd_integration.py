import numpy as np
from signals_tdcc import run_cc
from utils.misc import load_files
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
                    subfolder="he_integrator_test"
                )

if __name__ == '__main__':
    dts = np.array([0.1, 0.05, 0.01, 0.005])
    run(dts)