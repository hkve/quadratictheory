import pathlib as pl

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

