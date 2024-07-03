import quadratictheory as qt
import matplotlib.pyplot as plt
from matplotlib import cm, ticker, colors
import numpy as np

def get_geometry(theta):
    theta1 = (theta / 2) * np.pi / 180
    theta2 = -theta1
    theta3 = theta1 + np.pi
    theta4 = -theta3

    r = 3.284  # Place H atoms on circle with R=1.738Å
    x1, y1 = r * np.cos(theta1), r * np.sin(theta1)
    x2, y2 = r * np.cos(theta2), r * np.sin(theta2)
    x3, y3 = r * np.cos(theta3), r * np.sin(theta3)
    x4, y4 = r * np.cos(theta4), r * np.sin(theta4)

    geometry = f"""
	H {x1} {y1} 0.0; 
	H {x2} {y2} 0.0;
	H {x3} {y3} 0.0;
	H {x4} {y4} 0.0
	"""
 
    return geometry

def plot_2d_density(cc, b):
    N_x, N_z = 150, 150

    x = np.linspace(-6, 6, N_x)
    z = np.linspace(-6, 6, N_z)

    X, Z = np.meshgrid(x, z, indexing="ij")
    X, Z = X.reshape(N_x*N_z), Z.reshape(N_x*N_z)
    r = np.c_[X, Z, np.zeros_like(X)]

    gamma = cc.one_body_density()
    gamma = b._change_basis_one_body(gamma, b.C.T)
    rho = b.density(gamma, r)
    
    rho = rho.reshape(N_x, N_z)
    X = X.reshape(N_x, N_z)
    Z = Z.reshape(N_x, N_z)
    
    
    fig, ax = plt.subplots()



    # rho = np.ma.masked_where(rho <= 0, rho)

    # vmin = -3
    # vmax = np.ceil(np.log10(rho.max())+1)

    # levels = np.log10(np.logspace(vmin, vmax, 100))
    # # levels = np.arange(vmin, vmax)

    # print(levels)

    # levs = np.power(10, levels)
    # con = ax.contourf(X, Z, rho, levs, norm=colors.LogNorm(), locator=ticker.LogLocator(numticks=10))
    
    # # subs = (0.5,1.0,)
    con = ax.contourf(X, Z, np.log(rho), levels=200, vmin=-7, vmax=1)
    cbar = fig.colorbar(con)
    ax.set(xlabel="X", ylabel="Z")

    plt.show()


if __name__ == '__main__':
    r = 5
    # geometry, basis = f"N 0 0 -{r/2}; N 0 0 {r/2}", "sto-3g"
    geometry = get_geometry(90)
    basis = "dzp"

    b = qt.PyscfBasis(geometry, basis, restricted=False).pyscf_hartree_fock()
    cc = qt.QCCSD(b)
    cc.mixer = qt.mix.SoftStartDIISMixer(alpha=0.90, start_DIIS_after=4, n_vectors=5)
    
    cc.run(tol=1e-6,  vocal=True)


    plot_2d_density(cc, b)