from pyscf import gto, scf, cc, fci, ao2mo

def run_fci_single(atom, basis, *args):
    mol = gto.M(unit="angstrom")
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