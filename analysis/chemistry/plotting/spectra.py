import matplotlib.pyplot as plt
import plotting.plot_utils as pl

pack_as_list = lambda thing: [thing] if not type(thing) in [list, tuple] else thing

def plot_spectrum(freqs, S_tots, labels, ev=True, filename=None, **kwargs):
    default = {
        "cutoff": 0.05,
        "line_styles": ["-", ":"],
        "colors": ["blue", "black"]
    }
    default.update(kwargs)
    
    cutoff = default["cutoff"]
    line_styles = default["line_styles"]
    colors = default["colors"]

    freqs = pack_as_list(freqs)
    S_tots = pack_as_list(S_tots)
    labels = pack_as_list(labels)


    fig, ax = plt.subplots()
    for freq, S_tot, label, ls, color in zip(freqs, S_tots, labels, line_styles, colors):
        xunit = "[au]"
        one_ev = 27.211386245988468
        if ev:
            freq *= one_ev
            xunit = "[eV]"


        n_points = len(freq)
        freq = freq[n_points//2:]
        S_tot = S_tot[n_points//2:]
        n_points /= 2
        max_freq = freq.max()

        cutoff_idx = int((cutoff/max_freq)*n_points)
        print(cutoff, n_points)
        freq = freq[:cutoff_idx]
        S_tot = S_tot[:cutoff_idx]

        ax.plot(freq, S_tot, label=label, ls=ls, color=color)
    ax.legend()
    ax.set(xlabel=rf"$\omega$ {xunit}", ylabel="Intensity [arb. unit]")
    ax.set_ylim((-0.05, 1.05))
    plt.show()