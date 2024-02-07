import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plot_utils
from scipy.interpolate import CubicSpline

def spline(x, y):
    spl = CubicSpline(x, y, bc_type=((2, 0.0), (2, 0.0)))

    x_new = np.linspace(x[0], x[-1], 1000)
    y_new = spl(x_new)

    return x_new, y_new

def plot(filename, ax, splines=False, x_min=None, y_max=None, ylabel=True):
    assert ax.shape == (2,), "Wrong shape"
    df = pd.read_csv(filename, sep=",", header=0, index_col=0)
    df.sort_values("r", inplace=True)
    c, m = plot_utils.colors, plot_utils.markers

    r = df["r"]
    methods = df.columns[1:]
    E0 = df["FCI"].iloc[-1]

    for i, method in enumerate(methods):
        e = df[method].to_numpy() - E0
        ax[0].scatter(r, e, color=c[i], label=method, marker=m[i])

        if splines:
            if np.any(np.isnan(e)):
                first_nan = np.isnan(e).argmax()
            else:
                first_nan = len(e)

            r_spline, e_spline = spline(r[:first_nan].to_numpy(), e[:first_nan])
            ax[0].plot(r_spline, e_spline, color=c[i])

    y_lims = ax[0].get_ylim()
    x_lims = ax[0].get_xlim()
    ax[0].hlines(0, *x_lims, color="k", ls="--", alpha=0.4)
    if x_min is not None: ax[0].set_xlim(x_min, x_lims[1])
    if y_max is not None: ax[0].set_ylim(y_lims[0], y_max)
    if ylabel: ax[0].set(ylabel=r"$E_{bound} - E_{free}$  [au]")

    for i, method in enumerate(methods[1:], start=1):
        de = np.abs(df[method].to_numpy() - df["FCI"].to_numpy())
        ax[1].plot(r, de, color=c[i], marker=m[i])
        ax[1].set_yscale("log")

    # ax[0].set_xticklabels([])
    xlims = ax[0].get_xlim()
    ax[1].set(xlim=xlims)

    ax[0].legend()
    ax[1].set(xlabel="R [Å]")
    if ylabel: ax[1].set(ylabel=r"$E - E_{FCI}$ [au]")
    # plot_utils.save("N2")


def plot_N2():
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,8), height_ratios=[6,3])
    plot("N2.csv", axes, splines=True, ylabel=True, x_min=0.8, y_max=0.2)
    plt.show()

def main():
    plot_N2()

if __name__ == "__main__":
    main()