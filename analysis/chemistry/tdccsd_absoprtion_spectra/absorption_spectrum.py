import numpy as np

import matplotlib as mpl

# Use the pgf backend (must be set before pyplot imported)
# mpl.use('pgf')
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
import sys
from scipy.integrate import simps
from scipy.signal import find_peaks
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText

def get_spectral_lines(time_points, dipole_moment):
    dt = time_points[1] - time_points[0]

    freq = (
        scipy.fftpack.fftshift(scipy.fftpack.fftfreq(len(time_points)))
        * 2
        * np.pi
        / dt
    )

    a = scipy.fftpack.fftshift(scipy.fftpack.fft(dipole_moment))

    return freq, a


def compute_absorption_spectrum(name, basis, method):
    """
    Compute absorption spectrum according to (22)-(24) in: https://pubs.acs.org/doi/pdf/10.1021/ct200137z
    """
    pol_dir_list = [0, 0, 0]
    dip_tot_list = []

    S_tot_list = []

    gamma = 0.00921

    for pol_dir in pol_dir_list:

        # Load data
        samples = np.load(
            f"dat/tdr{method}_{name}_{basis}_discrete_delta_pulse_E0=0.001_dt=0.1_pol_dir={pol_dir}.npz",
            allow_pickle=True,
        )
        

        time_points = samples["time_points"]
        dipole_moment = samples["dipole_moment"].real

        freq, alpha_jj = get_spectral_lines(
            time_points,
            (dipole_moment[:, pol_dir] - dipole_moment[0, pol_dir])
            * np.exp(-gamma * time_points),
        )

        sigma_jj = alpha_jj.imag * 4 * freq

        S_tot_list.append(1 / 3 * sigma_jj)

    S_tot = S_tot_list[0] + S_tot_list[1] + S_tot_list[2]
    S_tot = np.abs(S_tot)

    return freq, S_tot


name = "be"
basis = "cc-pvdz"


freq_ccsd, S_tot_ccsd = compute_absorption_spectrum(name, basis, method="ccsd")

S_tot_ccsd = S_tot_ccsd / np.abs(S_tot_ccsd).max()


ccsd_peaks, _ = find_peaks(S_tot_ccsd, height=0.005)


freq_peaks_ccsd = freq_ccsd[ccsd_peaks]

one_ev = 27.211386245988468

excitations_ccsd = freq_peaks_ccsd[freq_peaks_ccsd > 0] * one_ev
counter = 0
while excitations_ccsd[counter] < 30:
    print(excitations_ccsd[counter])
    counter += 1


fig, ax = plt.subplots(1, 1)
ax.plot(freq_ccsd * one_ev, S_tot_ccsd, "#003f5c", linewidth=1, label="TDCCSD")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(0.5)
ax.spines["left"].set_linewidth(0.5)
ax.set_xlim(0, 930)
ax.set_ylabel("Relative intensity")
plt.tick_params(axis="both", which="major", labelsize=8, width=0.5)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(loc="upper center", fontsize=8, frameon=False)
spacing = 0.600
fig.subplots_adjust(bottom=spacing)

plt.show()

