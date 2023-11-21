import numpy as np
import matplotlib.pyplot as plt

import clusterfock as cf

def plot_density():
    basis = cf.basis.HarmonicOscillatorOneDimension(L=20, N=2, restricted=False, omega=0.25, a=0.25, x=(-10, 10, 1000))

    hf = cf.HF(basis=basis).run()
    rho_hf = basis.density(hf.rho)
    basis.change_basis(hf.C)

    fig, ax = plt.subplots()
    x = basis.x
    ax.plot(x, rho_hf, label="HF")
    
    methods = [cf.CCD, cf.CCSD]

    for CC in methods: 
        cc = CC(basis=basis).run(include_l=True, tol=1e-6, vocal=True)
        rho_cc = cc.one_body_density()
        rho_cc = np.einsum("ap,pq,bq->ab", hf.C, rho_cc, hf.C)
        rho_cc = basis.density(rho_cc)
        ax.plot(x, rho_cc, label=CC.__name__)

    ax.legend()
    ax.set(xlim=(-6,6), ylim=(0,0.4))
    ax.set(xlabel="x [au]", ylabel=r"$\rho(x)$")
    plt.show()

def plot_time_evolution():
    omega = 0.25
    a = 0.25
    electric_field_freq = 8*omega
    eps0 = 1.0
    L = 10
    num_grid_points = 1000
    grid_length = 10

    basis = cf.basis.HarmonicOscillatorOneDimension(L=2*L, N=2, restricted=True, omega=omega, a=eps0, x=(-grid_length, grid_length, num_grid_points))

    def electric_field(t, basis, freq=electric_field_freq, eps0=eps0):
        return eps0*np.sin(freq*t)*basis.r

    def sampler(basis):
        return {"r": basis.r}
    
    hf = cf.HF(basis).run()
    basis.change_basis(hf.C)
    basis.from_restricted()

    cc = cf.CCSD(basis).run(include_l=True, tol=1e-6)
    tdcc = cf.td.TimeDependentCoupledCluster(cc, time=(0, 8*np.pi/electric_field_freq, 0.01))
    
    tdcc.external_one_body = electric_field
    tdcc.one_body_sampler = sampler

    t, e, overlap = tdcc.run()
    t *= electric_field_freq/(2*np.pi)

    fig, ax = plt.subplots()
    ax.plot(t, overlap)
    ax.set(ylim=(0,1), xlim=(0,4), xlabel=r"$\omega t / 2\pi$", ylabel="|Overlap(t)|$^2$")
    plt.show()

if __name__ == "__main__":
    # plot_density()
    plot_time_evolution()