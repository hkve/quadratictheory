import numpy as np
import quadratictheory as qt

# Make LiH system, perform (pyscf) hf and qccsd calculation
basis = qt.PyscfBasis("Li 0 0 0; H 0 0 2.0", basis="3-21g", restricted=False).pyscf_hartree_fock()
qccsd = qt.QCCSD(basis).run(vocal=True)

# Setup time propagation
td_qccsd = qt.TimeDependentCoupledCluster(cc = qccsd, time = (0, 1, 0.01))

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

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(time, energy)
axes[1].plot(time, mu_z)

axes[0].set(xlabel="Time [a.u.]", ylabel=r"Energy [$E_h$]")
axes[1].set(xlabel="Time [a.u.]", ylabel=r"$\mu_z$ [a.u.]")

plt.show()