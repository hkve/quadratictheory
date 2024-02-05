import numpy as np


pack_as_list = lambda thing: [thing] if not type(thing) in [list, tuple] else thing

def load_systems(molecule_name, basis_name, pulse, methods, integrators, dt=None):
    import os

    files = os.listdir("dat/")

    methods = pack_as_list(methods)
    integrators = pack_as_list(integrators)

    results = {method: {} for method in methods}

    for file in files:
        if not f"_{molecule_name}_" in file:
            continue
        if not f"_{basis_name}_" in file:
            continue
        if not f"{pulse}" in file:
            continue

        if dt is not None:
            if f"dt={dt}" not in file:
                continue

        for method in methods:
            for integrator in integrators:
                if file.startswith(method) and integrator in file:
                    print(f"LOADING {file}", method, integrator)
                    results[method][integrator] = np.load(f"dat/{file}", allow_pickle=True)

    return results

def get_geometries():
    geometries = dict(
        he="he 0.0 0.0 0.0",
        be="be 0.0 0.0 0.0",
        ne="ne 0.0 0.0 0.0",
        h2="h 0.0 0.0 0.0; h 0.0 0.0 1.4",
        lih="li 0.0 0.0 0.0; h 0.0 0.0 3.08",
        bh="b 0.0 0.0 0.0; h 0.0 0.0 2.3289",
        chp="c 0.0 0.0 0.0; h 0.0 0.0 2.137130",
        h2o="O 0.0 0.0 -0.1239093563; H 0.0 1.4299372840 0.9832657567; H 0.0 -1.4299372840 0.9832657567",
        no2="N; O 1 3.25; O 1 3.25 2 160",
        lif="f 0.0 0.0 0.0; li 0.0 0.0 -2.9552746891",
        co="c 0.0 0.0 0.0; o 0.0 0.0 2.1316109151",
        n2="n 0.0 0.0 0.0; n 0.0 0.0 2.074",
        co2="c 0.0 0.0 0.0; o 0.0 0.0 2.1958615987; o 0.0 0.0 -2.1958615987",
        nh3="N 0.0 0.0 0.2010; H 0.0 1.7641 -0.4690; H 1.5277 -0.8820 -0.4690; H -1.5277 -0.8820 -0.4690",
    	hf="h 0.0 0.0 0.0; f 0.0 0.0 1.7328795",
    	ch4="c 0.0 0.0 0.0; h 1.2005 1.2005 1.2005; h -1.2005 -1.2005 1.2005; h -1.2005 1.2005 -1.2005; h 1.2005 -1.2005 -1.2005",
    )

    return geometries

def get_symmetries(geometry):
    if geometry.endswith(";"):
        geometry = geometry[:-1]

    split_by_atoms = geometry.split(";")

    # Assume that we only have a single atom
    dof = 1

    # If the split gives a list, set the dof to the lenght of the list
    if dof is not None:
        dof = len(split_by_atoms)

    # If we have more than three atoms, we still have 3 dof
    if dof > 3:
        dof = 3

    coords = "zyx"
    vecs = np.eye(3)

    # Construct the dict with directions and dof
    polarisations = {"dof": dof}
    for i in range(dof):
        polarisations[coords[i]] = vecs[-i-1,:]

    return polarisations

if __name__ == "__main__":
    geometries = get_geometries()
    geometry = geometries["ch4"]

    print(
        get_symmetries(geometry)
    )