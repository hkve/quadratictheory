import matplotlib.pyplot as plt
import plotting.plot_utils as pl

pack_as_list = lambda thing: [thing] if not type(thing) in [list, tuple] else thing

def plot_spectrum(freqs, S_tots, labels, ev=True, filename=None, **kwargs):
    default = {
        "freq_cutoff": 0.05,
        "line_styles": ["-", ":"]
    }
    default.update(kwargs)
    
    freq_cutoff = default["freq_cutoff"]
    line_styles = default["line_styles"]

    freqs = pack_as_list(freqs)
    S_tots = pack_as_list(S_tots)
    labels = pack_as_list(labels)


    fig, ax = plt.subplots()
    for freq, S_tot, label, ls in zip(freqs, S_tots, labels, line_styles):
        xunit = "[au]"
        one_ev = 27.211386245988468
        if ev:
            freq *= one_ev
            xunit = "[eV]"


        n_points = len(freq)
        freq = freq[n_points//2:]
        S_tot = S_tot[n_points//2:]

        cutoff_idx = int(freq_cutoff*n_points)
        freq = freq[:cutoff_idx]
        S_tot = S_tot[:cutoff_idx]

        ax.plot(freq, S_tot, label=label, ls=ls)
    ax.legend()
    ax.set(xlabel=rf"$\omega$ {xunit}", ylabel="Intensity [arb. unit]")
    ax.set_ylim((-0.05, 1.05))
    plt.show()