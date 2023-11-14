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

if __name__ == "__main__":
    plot_density()