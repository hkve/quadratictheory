import plotting.plot_utils as pu
import matplotlib.pyplot as plt
import numpy as np

def plot_compare(results, **kwargs):
    assert len(results) == 2, f"This function only takes two series"
    
    defaults = {
        "polarisation": [0],
        "filename": None
    }
    defaults.update(kwargs)

    polarisation = defaults["polarisation"]
    filename = defaults["filename"]

    result1, result2 = results

    print("First passed: ")
    for k, v in result1.items():
        if type(v) is not np.ndarray:
            print(f"{k}: {v}")

    print("Second passed: ")
    for k, v in result2.items():
        if type(v) is not np.ndarray:
            print(f"{k}: {v}")

    name, basis, integrator, pulse = result1["name"], result1["basis"], result1["integrator"], result1["pulse"] 

    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5, 3], figsize=(10, 8))
    fig.suptitle(f"Energy for {name}({basis}) {pulse} pulse using {integrator}", fontsize=16)

    t1, e1 = result1["t"], result1["energy"]
    t2, e2 = result2["t"], result2["energy"]

    ax[0].plot(t1, e1, label=f"{result1['method']}")
    ax[0].plot(t2, e2, label=f"{result2['method']}")

    ax[1].plot(t1, np.abs(e1 - e2), label=f"DIFF: {integrator}")

    ax[0].set(ylabel="E [au]")
    ax[1].set(xlabel="time [au]", ylabel="E [au]")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale("log")
    pu.save(filename, "_energy")
    plt.show()


    t1, t2 = result1["t"], result2["t"]
    fig, ax = plt.subplots(nrows=2, ncols=1, height_ratios=[5, 3], figsize=(10, 8))
    fig.suptitle(f"r_i for {name}({basis}) {pulse} pulse using {integrator}", fontsize=16)
    for pol in polarisation:
        r1 = result1["r"][:,pol]
        r2 = result2["r"][:,pol]

        ax[0].plot(t1, r1, label=f"{result1['method']}")
        ax[0].plot(t2, r2, label=f"{result2['method']}")

        ax[1].plot(t1, np.abs(r1 - r2), label=f"DIFF: direction {pol}")

    ax[0].set(ylabel="<r_i> [au]")
    ax[1].set(xlabel="time [au]", ylabel="<r_i> [au]")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_yscale("log")
    pu.save(filename, "_r")
    plt.show()