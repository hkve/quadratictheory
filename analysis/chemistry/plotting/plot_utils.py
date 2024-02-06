import pathlib as pl
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

def make_figs_path(filename):
    cur_path = pl.Path(__file__)
    root_path = cur_path

    while root_path.name != "analysis":
        root_path = root_path.parent

    figs_path = root_path / pl.Path("plots")

    if not figs_path.exists():
        return None
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    figs_path /= filename

    return str(figs_path)

def save(filename, addition):
    if filename:
        filename = make_figs_path(filename + addition)
        plt.savefig(filename)

fancy = True

if fancy:
    colors = ["black", "orange", "green", "red", "purple", "blue"]
    markers = ["o", "^", "s", "x", "+", "<"]

    # Set all fonts to be equal to tex
    # https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["text.usetex"] = True

    # Saving parameters
    plt.rcParams["savefig.dpi"] = 300

    # Figure options, set tight layout
    plt.rc("figure", autolayout=True)

    # Font sizes
    plt.rc("axes", titlesize=18, labelsize=16, prop_cycle=cycler('color', colors))
    plt.rc("legend", fontsize=14, shadow=True)

    # Tick parameters
    _ticks_default_parameters = {
        "labelsize": 12
    }
    plt.rc("xtick", **_ticks_default_parameters)
    plt.rc("ytick", **_ticks_default_parameters)

    # Line options
    plt.rc("lines", linewidth=2)

    # To see more paramteres, print the possible options:
    # print(plt.rcParams)

