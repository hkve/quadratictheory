# Quadratic Coupled Cluster Theory
We present functionality to perform a variety of coupled cluster (CC) calculation for atomic and molecular systems in the package ``quadratictheory``, interfaced with [``PySCF``](https://pyscf.org/index.html). In particular, the quadratic coupled cluster (QCC) theories of [Troy Van Voorhis and Martin Head-Gordon](https://www.sciencedirect.com/science/article/pii/S0009261400011374?casa_token=iXLM3MZtmbUAAAAA:ggKW7m3vTGZk3w_Y--fxKO9Prhwsgyi_J_CWwiKHsOAPIZYeTny21g-B5_ALLlmw0vqGqU4fng) are implemented, include time evolution.

## Installation
To install the functionality of ``quadratictheory``, we recommend creating a virtual environment to mitigate the potential conflicts with your current ``Python`` environment.

The code can be installed by running (from the root directory):
```bash
pip install .
```
After the installation is complete, make sure the tests pass by running:
```bash
python -m pytest tests
```

## Computer Algebra System
The algebraic equations implemented in ``quadratictheory`` are derived using [``Drudge``](https://github.com/tschijnmo/drudge) and [``Gristmill``](https://github.com/tschijnmo/gristmill). For installation instructions, go [here](https://tschijnmo.github.io/drudge/install.html), where we strongly recommend installing via. ``Docker`` for most usage. 

All code related to the derivation of the algebraic equations are present in the ``symbolic`` directory. To clean-up expressions in a more readable format, we have made use of the functionality of [``SymPy``](https://www.sympy.org/en/index.html).

## Examples
The ``examples`` directory shows some simple usage of ``quadratictheory``. 

The following code bit constructs a N2 molecule system, transforms to the HF basis and performs a CCSD and QCCSD calculation.

```python
import quadratictheory as qt

# Setup basis and perform HF calculation, bases default to restricted scheme
basis = qt.PyscfBasis("N 0 0 0; N 0 0 3.5", basis="sto-3g", restricted=True)
hf = qt.HF(basis).run(tol=1e-6)

# Chage to HF basis
basis.change_basis(hf.C, inplace=True)

# Run RCCSD calculation (calls qt.RCCSD)
ccsd = qt.CCSD(basis).run(tol=1e-6, vocal=True).run()
energy_ccsd = ccsd.energy()

# Change basis from restricted scheme to general scheme
basis.from_restricted(inplace=True)

# Perform QCCSD calculation (general scheme only)
qccsd = qt.QCCSD(basis).run(tol=1e-6, vocal=True)
energy_qccsd = qccsd.energy()
```
In this example, we compute the ground state of LiH with QCCSD. Under a sinusoidal laser pulse, the LiH system is propagated for 1 a.u., where we compute the energy and dipole moment.
```python
import numpy as np
import quadratictheory as qt

# Make LiH system, perform (pyscf) hf and qccsd calculation
basis = qt.PyscfBasis("Li 0 0 0; H 0 0 2.0", basis="6-31g", restricted=False).pyscf_hartree_fock()
qccsd = qt.QCCSD(basis).run(vocal=True)

# Setup time propagation
td_qccsd = qt.TimeDependentCoupledCluster(cc = qccsd, time = (0, 1, 0.05))

# Setup laser field, polarized along z-axis
td_qccsd.external_one_body = qt.pulse.Sin2(
    u = np.array([0.0,0.0,1.0]),
    F_str = 1e-4,
    omega = 10.0,
)

# Setup sampler (what should be computed and stored during the calculation)
td_qccsd.sampler = qt.sampler.DipoleSampler()

# Run the calculation and get results
results = td_qccsd.run(vocal=True)

# Time, energy and dipole moment
time, energy, mu_z = results["t"], results["energy"], -results["r"][:,2]
```

Further results, in particular the code used to produce results for the master's thesis, is presented in the ``analysis`` directory. For more detailed usage; Use The Source Luke.