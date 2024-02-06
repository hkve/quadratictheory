import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
from scipy.integrate import simps
from scipy.signal import find_peaks

from utils.misc import load_files
from plotting.spectra import plot_spectrum

def absorption_spectrum_preprocess(results, dof):
    assert len(results) == dof, f"Amount of data does not match expected {len(results)} != {dof}"

    dipole_moments, pol_dirs = [], []
    for (info, samples) in results:
        r, pol_dir = samples["r"], int(info["polarisation"])
        
        assert pol_dir in [0,1,2], f"{pol_dir = } is not along a cartesian axis, which preprocess assumes"
        dipole_moments.append(r)
        pol_dirs.append(pol_dir)

    if dof == 1:
        pol_dirs = pol_dirs + pol_dirs + pol_dirs
        dipole_moments = dipole_moments + dipole_moments + dipole_moments

    if dof == 2:
        y_pol = pol_dirs.index(1)
        pol_dirs = pol_dirs + [1]
        dipole_moments = dipole_moments + [dipole_moments[y_pol]]

    time = samples["t"]

    return time, dipole_moments, pol_dirs

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

def compute_absorption_spectrum(time, dipole_moments, pol_dirs, normalize=True):
    """
    Compute absorption spectrum according to (22)-(24) in: https://pubs.acs.org/doi/pdf/10.1021/ct200137z
    """
    S_tot = []

    gamma = 0.00921
    for dipole_moment, pol_dir in zip(dipole_moments, pol_dirs):
        freq, alpha_jj = get_spectral_lines(
            time,
            (dipole_moment[:, pol_dir] - dipole_moment[0, pol_dir])
            * np.exp(-gamma * time),
        )

        sigma_jj = alpha_jj.imag * 4 * freq

        S_tot.append(1 / 3 * sigma_jj)

    S_tot = S_tot[0] + S_tot[1] + S_tot[2]
    S_tot = np.abs(S_tot)

    if normalize:
        S_tot = S_tot / S_tot.max()

    return freq, S_tot

def absorption_spectrum_peaks(S_tot, freq, height=0.005, ev=True, vocal=True):
    peaks, _ = find_peaks(S_tot, height=height)
    freq_peaks = freq[peaks]

    one_ev = 27.211386245988468

    excitations = freq_peaks[freq_peaks > 0]
    if ev:
        excitations *= one_ev
    
    if vocal:
        for excitation in excitations: print(excitation)

    return excitations


def compare_two(methods=["CCSD", "QCCSD"], name="be"):
    methods = ["CCSD", "QCCSD"]
    freqs, S_tots = [], []
    for method in methods:
        results = load_files(method=method, name=name, basis="cc-pVDZ", Tend=1000, dt=0.025, integrator="Rk4Integrator", pulse="DeltaKick")
        
        time, dipole_moments, pol_dirs = absorption_spectrum_preprocess(results, 1)
        freq, S_tot = compute_absorption_spectrum(time, dipole_moments, pol_dirs)
        print(method)
        absorption_spectrum_peaks(S_tot, freq)

        freqs.append(freq)
        S_tots.append(S_tot)

    plot_spectrum(freqs, S_tots, methods)

if __name__ == '__main__':
    compare_two()