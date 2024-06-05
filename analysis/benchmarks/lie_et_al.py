import quadratictheory as qt
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
import plot_utils as pu

def run_lie_et_al(quadratic=False):
    r = 1.389704492
    E_max = 0.07
    omega = 0.1
    u = np.array([0,0,1], dtype=float)
    t_end = 2 #225
    dt = 0.01

    
    integrator = "GaussIntegrator"
    integrator_args = {"s": 3, "maxit": 20, "eps": 1e-6, "method": "A", "mu": 1.75}
    
    b = qt.PyscfBasis(f"H 0 0 0; H 0 0 {r}", basis="6-311++Gss", restricted=False).pyscf_hartree_fock()

    if quadratic:
        cc = qt.QCCSD(b).run(tol=1e-8, vocal=True)
    else:
        cc = qt.CCSD(b).run(tol=1e-8, include_l=True, vocal=True)

    tdcc = qt.TimeDependentCoupledCluster(cc, time=(0, t_end, dt), integrator=integrator, integrator_args=integrator_args)
    
    tdcc.external_one_body = qt.pulse.LieEtAl(u, E_max, omega)
    tdcc.sampler = qt.sampler.DipoleSampler()

    results = tdcc.run(vocal=True)

    name = "QCCSD" if quadratic else "CCSD"

    path = pl.Path("dat")
    filename = pl.Path(f"LieEtAl_{name}_6-311++Gss.npz")

    np.savez(path / filename, **results)

def plot():
    path = pl.Path("dat")
    filename_ccsd = pl.Path("LieEtAl_CCSD_6-311++Gss.npz")
    filename_qccsd = pl.Path("LieEtAl_QCCSD_6-311++Gss.npz")

    results_ccsd = np.load(path / filename_ccsd)
    results_qccsd = np.load(path / filename_qccsd)

    time = results_ccsd["t"]
    dipole_ccsd = results_ccsd["r"][:,2].real
    dipole_qccsd = results_qccsd["r"][:,2].real

    omega = 0.1
    time = omega*time

    fig, ax = plt.subplots()

    ax.plot(time, dipole_ccsd, label="CCSD")
    ax.plot(time, dipole_qccsd, label="QCCSD", ls="--", dashes=(4,4))
    ax.set(xlabel=r"$\omega t$", ylabel=r"$\mu_z (t)$ [a.u.]")
    ax.legend()
    pu.save("LieEtAl_dipoles")
    plt.show()


    fig, ax = plt.subplots()

    ax.plot(time, np.abs(dipole_ccsd-dipole_qccsd))

    ax.set(xlabel=r"$\omega t$", ylabel=r"$\Delta \mu_z (t)$ [a.u.]")
    ax.set_yscale("log")

    ylims = ax.get_ylim()
    ax.set_ylim((1e-16, ylims[1]))
    pu.save("LieEtAl_dipoles_differences")
    plt.show()


if __name__ == "__main__":
    # run_lie_et_al(quadratic=False)
    # run_lie_et_al(quadratic=True)

    plot()

    # geom = ""

    # for i in range(10):
    #     geom += f"H 0 0 {i};"

    # print(geom)
    # b = qt.PyscfBasis(geom, basis="sto-3g").pyscf_hartree_fock()

    # b.from_restricted()
    # print(b.h.shape)