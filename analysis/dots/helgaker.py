import quadratictheory as qt
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

def setup_ho(filepaths, N, omega):
    base_omega = 0.5

    h = np.load(filepaths["h"])
    u = np.load(filepaths["u"])

    L = 2*h.shape[0]

    ho = qt.CustomBasis(
        N = N,
        L = L,
    )

    ho.h = h
    ho.u = u

    return ho


def test_ho():
    root = "../../HelgakerPaper/grid_elms/"
    filepaths = {
        "h": root + "one_body_51_grid_3_spf.npy",
        "u": root + "u_abcd_51_grid_3_spf.npy",
    }

    ho = setup_ho(filepaths, N=2, omega=0.5)

    hf = qt.RHF(ho).run(tol=1e-4, maxiters=100, vocal=True)

    print(
        hf.energy()
    )

    ho.change_basis(hf.C)


    cc = qt.CCSD(ho).run(tol=1e-4, maxiters=100, vocal=True)

    print(
        cc.energy()
    )
if __name__ == '__main__':
    test_ho()
