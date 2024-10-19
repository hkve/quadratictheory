import quadratictheory as qt

# Setup basis and perform HF calculation, bases default to restricted scheme
basis = qt.PyscfBasis("N 0 0 0; N 0 0 3.5", basis="sto-3g", restricted=True)
hf = qt.HF(basis).run(tol=1e-6)

# Chage to HF basis
print(f"{hf.converged = }?")
basis.change_basis(hf.C, inplace=True)

# Run RCCSD calculation (calls qt.RCCSD)
ccsd = qt.CCSD(basis).run(tol=1e-6, vocal=True).run()
energy_ccsd = ccsd.energy()

# Change basis from restricted scheme to general scheme
basis.from_restricted(inplace=True)

# Perform QCCSD calculation (general scheme only)
qccsd = qt.QCCSD(basis).run(tol=1e-6, vocal=True)
energy_qccsd = qccsd.energy()

print(f"{energy_ccsd = :.5f}\n{energy_qccsd = :.5f}")