import pathlib as pl
import numpy as np
import os
import re

pack_as_list = lambda thing: [thing] if not type(thing) in [list, tuple] else thing

def root_path():
    current_dir = pl.Path(__file__)

    while str(current_dir.name) != "chemistry":
        current_dir = current_dir.parent

    return current_dir

def dat_path(root=None):
    if root is None:
        root = root_path()

    path = root / "dat"

    assert path.exists(), f"Looks like there is no directory named {path}"

    return path

def custom_basis_path(name, root=None):
    if root is None:
        root = root_path()

    path = root / f"utils/{name}"

    assert path.exists(), f"Looks like there is no file named {path}"

    return path


def check_if_file_should_be_included(prop, file_split, preamble=""):
    if prop == [None]:
        return True
    else:
        for p in prop:
            if f"{preamble}{p}" in file_split:
                return True
    return False

def make_info_from_filename(file):
    pattern = r'^([A-Za-z]+)_([A-Za-z]+)_([A-Za-z\-]+)_Tend=(\d+)_dt=([\d\.]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d)\.npz$'
    match = re.match(pattern, file)

    keys = ["method", "name", "basis", "Tend", "dt", "integrator", "pulse", "polarisation"]
    info = {key: match.group(i+1) for i, key in enumerate(keys)} 

    return info

def load_files(path=None, method=None, name=None, basis=None, Tend=None, dt=None, integrator=None, pulse=None, polarisation=None):
    if path is None:
        path = dat_path()
    
    method = pack_as_list(method)
    name = pack_as_list(name)
    basis = pack_as_list(basis)
    Tend = pack_as_list(Tend)
    dt = pack_as_list(dt)
    integrator = pack_as_list(integrator)
    pulse = pack_as_list(pulse)
    polarisation = pack_as_list(polarisation)

    files = os.listdir(path)
    
    results = []
    for file in files:
        file_root = file.rstrip(".npz")
        file_split = file_root.split("_")

        include = check_if_file_should_be_included(method, file_split)
        include *= check_if_file_should_be_included(name, file_split)
        include *= check_if_file_should_be_included(basis, file_split)
        include *= check_if_file_should_be_included(Tend, file_split, preamble="Tend=")
        include *= check_if_file_should_be_included(dt, file_split, preamble="dt=")
        include *= check_if_file_should_be_included(integrator, file_split)
        include *= check_if_file_should_be_included(pulse, file_split)
        include *= check_if_file_should_be_included(polarisation, file_split)

        if include:
            print(f"Load {file}")
            info = make_info_from_filename(file)
            result = np.load(path / file, allow_pickle=True)

            results.append([info, result])

    return results