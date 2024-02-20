import matplotlib.pyplot as plt
import plotting.plot_utils as pl
import numpy as np

pack_as_list = lambda thing: [thing] if not type(thing) in [list, tuple] else thing

def plot_imag_part(results, expval, direction=None, filename=None, **kwargs):
    default = {
        "xlabel": "Time [au]",
        "ylabel": r"YLABEL"
    }
    default.update(kwargs)

    xlabel, ylabel = default["xlabel"], default["ylabel"]

    fig, ax = plt.subplots()
    r1, r2 = results

    for i, result in enumerate(results):
        x = result["t"]
        y = result[expval]

        if direction is not None:
            y = y[:,direction]


        ax.plot(x, y.imag, label=result["method"], c=pl.colors[i], marker="x")

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    # ax.set_yscale("log")
    plt.show()

def plot_rho_hermiticity(results, filename=None, **kwargs):
    fig, ax = plt.subplots()
    for i, result in enumerate(results):
        x = result["t"]

        ax.plot(x, result["delta_rho1"], label=result["method"] + r" $\gamma$", c = pl.colors[i])
        ax.plot(x, result["delta_rho2"], label=result["method"] + r" $\Gamma$", ls="--", c = pl.colors[i])

    ax.legend()
    ax.set_yscale("log")
    ax.set(ylabel=r"$|\rho - \rho^{\dagger}|$")
    plt.show()
