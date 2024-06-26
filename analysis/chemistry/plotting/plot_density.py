from quadratictheory.basis import PyscfBasis
from quadratictheory.hf import HF
from quadratictheory.cc import CCD
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab


def plot_density(basis, rho, log=True):
    mu, sigma = 0, 0.1
    x = np.linspace(-3.0, 3.0, 5000)
    y = np.linspace(-3.0, 3.0, 5000)
    z = np.linspace(-3.0, 3.0, 5000)

    xyz = np.vstack([x, y, z])

    xmin, ymin, zmin = x.min(), y.min(), z.min()
    xmax, ymax, zmax = x.max(), y.max(), z.max()
    xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = basis.density(hf.rho, coords.T).reshape(xi.shape)

    if log:
        density = np.where(density < 1e-2, 1e-2, density)
        density = np.log(density) - np.log(1e-2)

    figure = mlab.figure("DensityPlot", bgcolor=(0.0, 0.0, 0.0))

    grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
    min = density.min()
    max = density.max()
    mlab.pipeline.volume(grid, vmin=min, vmax=min + 0.5 * (max - min))
    mlab.colorbar(orientation="vertical")

    mlab.axes()
    mlab.show()


if __name__ == "__main__":
    atom = """
    O	0.000	 0.000	 0.434;
    O	0.000	 1.064	-0.217;
    O	0.000	-1.064	-0.217;
    """
    # atom = "N 0 0 -0.6; N 0 0 0.6"
    # atom = "Li 0 0 -0.5; H 0 0 0.5"
    basis = PyscfBasis(atom=atom, basis="cc-pVDZ", restricted=False)
    hf = HF(basis).run()
    print(hf.energy())
    basis.change_basis(hf.C)

    ccd = CCD(basis).run(include_l=True)
    print(ccd.energy())
    ccd.densities()

    rho_in_hf_basis = np.einsum("ap,pq,bq->ab", basis.C, ccd.rho_ob, basis.C)
    plot_density(basis, rho_in_hf_basis)
