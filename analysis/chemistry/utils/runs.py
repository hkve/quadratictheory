import numpy as np
from pyscf import gto, scf, cc, fci, ao2mo
from pyscf.fci.rdm import make_rdm1
from utils.misc import custom_basis_path

def get_geometries():
    geometries = dict(
        he="he 0.0 0.0 0.0",
        be="be 0.0 0.0 0.0",
        ne="ne 0.0 0.0 0.0",
        h2="h 0.0 0.0 0.0; h 0.0 0.0 1.4",
        lih="li 0.0 0.0 0.0; h 0.0 0.0 3.08",
        bh="b 0.0 0.0 0.0; h 0.0 0.0 2.3289",
        chp="c 0.0 0.0 0.0; h 0.0 0.0 2.137130",
        h2o="O 0.0 0.0 -0.1239093563; H 0.0 1.4299372840 0.9832657567; H 0.0 -1.4299372840 0.9832657567",
        no2="N; O 1 3.25; O 1 3.25 2 160",
        lif="f 0.0 0.0 0.0; li 0.0 0.0 -2.9552746891",
        co="c 0.0 0.0 0.0; o 0.0 0.0 2.1316109151",
        n2="n 0.0 0.0 0.0; n 0.0 0.0 2.074",
        co2="c 0.0 0.0 0.0; o 0.0 0.0 2.1958615987; o 0.0 0.0 -2.1958615987",
        nh3="N 0.0 0.0 0.2010; H 0.0 1.7641 -0.4690; H 1.5277 -0.8820 -0.4690; H -1.5277 -0.8820 -0.4690",
    	hf="h 0.0 0.0 0.0; f 0.0 0.0 1.7328795",
    	ch4="c 0.0 0.0 0.0; h 1.2005 1.2005 1.2005; h -1.2005 -1.2005 1.2005; h -1.2005 1.2005 -1.2005; h 1.2005 -1.2005 -1.2005",
    )

    return geometries

def get_symmetries(geometry):
    if geometry.endswith(";"):
        geometry = geometry[:-1]

    split_by_atoms = geometry.split(";")

    # Assume that we only have a single atom
    dof = 1

    # If the split gives a list, set the dof to the lenght of the list
    if dof is not None:
        dof = len(split_by_atoms)

    # If we have more than three atoms, we still have 3 dof
    if dof > 3:
        dof = 3

    coords = "zyx"
    vecs = np.eye(3)

    # Construct the dict with directions and dof
    polarisations = {"dof": dof}
    for i in range(dof):
        polarisations[coords[i]] = vecs[-i-1,:]

    return polarisations

def disassociate_2dof(atom1, atom2, distances):
    return [
        f"{atom1} 0 0 0; {atom2} 0 0 {distance}" for distance in distances
    ]

def disassociate_h2o(distances):
    distances = distances if type(distances) in [list, tuple, np.ndarray] else [distances]
    theta = np.deg2rad(109.5)
    alpha = theta/2

    geometries = []
    for distance in distances:
        x, y = distance*np.cos(alpha), distance*np.sin(alpha)
        geometry = f"O 0 0 0; H {x} {y} 0; H {x} {-y} 0;"
        geometries.append(geometry)

    return geometries

def get_pyscf_mol_custom_basis(filename="chp.dat", geometry=f"C 0.0 0.0 0.0; H 0.0 0.0 2.13713"):
    filename = custom_basis_path(filename)
    mol = gto.M(unit="bohr")
    mol.verbose = 0
    mol.build(atom=geometry, basis=str(filename), charge=1, cart=True)

    return mol

def run_fci_single(atom, basis, *args):
    mol = gto.M(unit="bohr")
    mol.verbose = 0
    mol.build(atom=atom, basis=basis)

    hf = scf.RHF(mol)
    hf.conv_tol_grad = 1e-6
    hf.max_cycle = 50
    hf.init_guess = "minao"
    hf_energy = hf.kernel(vocal=0)

    myfci = fci.direct_spin0.FCI(mol)
    myfci.conv_tol = 1e-10
    h1 = hf.mo_coeff.T.dot(hf.get_hcore()).dot(hf.mo_coeff)
    eri = ao2mo.kernel(mol, hf.mo_coeff)
    myfci.max_space = 12
    myfci.davidson_only = True

    nroots = 1
    e_fci, c_fci = myfci.kernel(
        h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc(), nroots=nroots, max_memory=2000
    )

    return e_fci

def run_fci_density_matrix(atom, basis, *args):
    from pyscf import lib
    lib.num_threads(1)
    
    mol = gto.M(unit="bohr")
    mol.verbose = 0
    mol.build(atom=atom, basis=basis)

    myhf = scf.RHF(mol)
    myhf.kernel()

    cisolver = fci.FCI(mol, myhf.mo_coeff)
    e, fcivec = cisolver.kernel()

    #
    # Spin-traced 1-particle density matrix
    #
    norb = myhf.mo_coeff.shape[1]

    # alpha electrons and beta electrons. spin = nelec_a-nelec_b
    nelec_a, nelec_b = mol.nelec
    # dm1 = cisolver.make_rdm1(fcivec, norb, (nelec_a,nelec_b))
    

    dm1_a, dm1_b = cisolver.make_rdm1s(fcivec, norb, mol.nelec)

    dm1 = np.zeros((2*mol.nao, 2*mol.nao), dtype=dm1_a.dtype)

    dm1[::2,::2] = dm1_a
    dm1[1::2,1::2] = dm1_b

    return dm1