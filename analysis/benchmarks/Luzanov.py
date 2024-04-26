# https://onlinelibrary.wiley.com/doi/epdf/10.1002/qua.24487

import clusterfock as cf
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import plot_utils as pu

from IPython import embed


class CustomSampler(cf.td.sampler.Sampler):
    def __init__(self, one_body=True, two_body=False, misc=True):
        super().__init__(one_body, two_body, misc)
        self.has_overlap = True

    def one_body(self, basis):
        return {"r": basis.r}

    def misc(self, tdcc):
        return {
            "energy": tdcc.cc.time_dependent_energy(),
            "overlap": tdcc.cc.overlap(tdcc._t0, tdcc._l0, tdcc.cc._t, tdcc.cc._l),
        }

def run_Luzanov_Li2(quadratic=False):
    r = 5.0510485949
    E_max = 0.05
    omega = 0.06
    T = 314.2
    t_end = 400
    dt = 0.01
    u = np.array([0,0,1], dtype=float)

    
    integrator = "GaussIntegrator"
    integrator_args = {"s": 3, "maxit": 20, "eps": 1e-6, "method": "A", "mu": 1.75}
    
    b = cf.PyscfBasis(f"Li 0 0 0; Li 0 0 {r}", basis="6-311G").pyscf_hartree_fock()
    b.from_restricted()

    if quadratic:
        cc = cf.QCCSD(b).run(tol=1e-8, vocal=True)
    else:
        cc = cf.CCSD(b).run(tol=1e-8, include_l=True, vocal=True)

    tdcc = cf.TimeDependentCoupledCluster(cc, time=(0, t_end, dt), integrator=integrator, integrator_args=integrator_args)
    
    tdcc.external_one_body = cf.pulse.Luzanov1(u, E_max, T, omega)
    tdcc.sampler = CustomSampler()

    results = tdcc.run(vocal=True)

    name = "QCCSD" if quadratic else "CCSD"

    path = pl.Path("dat")
    filename = pl.Path(f"Luzanov_{name}_6-311G.npz")

    np.savez(path / filename, **results)

def run_Luzanov_H10(quadratic=False):
    r = 1.401118437
    N = 10
    E_max = 0.05
    omega = 0.1699
    T = 41.3
    t_end = 400
    dt = 0.01
    u = np.array([0,0,1], dtype=float)

    geometry = ""
    for i in range(N):
        geometry += f"H 0 0 {i*r};"
    
    integrator = "GaussIntegrator"
    integrator_args = {"s": 3, "maxit": 20, "eps": 1e-6, "method": "A", "mu": 1.75}
    
    b = cf.PyscfBasis(geometry, basis="sto-3g").pyscf_hartree_fock()
    b.from_restricted()

    if quadratic:
        cc = cf.QCCSD(b).run(tol=1e-8, vocal=True)
    else:
        cc = cf.CCSD(b).run(tol=1e-8, include_l=True, vocal=True)

    tdcc = cf.TimeDependentCoupledCluster(cc, time=(0, t_end, dt), integrator=integrator, integrator_args=integrator_args)
    
    tdcc.external_one_body = cf.pulse.Luzanov2(u, E_max, T, omega)
    tdcc.sampler = CustomSampler()

    results = tdcc.run(vocal=True)

    name = "QCCSD" if quadratic else "CCSD"

    path = pl.Path("dat")
    filename = pl.Path(f"Luzanov_{name}_sto-3g.npz")

    np.savez(path / filename, **results)


if __name__ == '__main__':
    # run_Luzanov_Li2(quadratic=False)
    run_Luzanov_H10(quadratic=False)