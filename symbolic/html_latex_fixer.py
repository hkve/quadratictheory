import pathlib as pl
import re


def read_HTMLFixer(path):
    string = ""
    with open(path, "r") as file:
        for line in file:
            string += line

    return string

def write_HTMLFixer(path, string):
    with open(path, "w") as file:
        file.write(string)

def get_sympy_expr(expr):
    pass

def fix_sympy_expr(expr):
    pass

def get_new_tex_string(expr):
    pass

def fix_pipeline(string):
    new_string = string
    tex_strings = re.findall(r"<p>(.*?)</p>", string)

    for tex_string in tex_strings:
        sympy_expr = get_sympy_expr(tex_string)
        sympy_expr = fix_sympy_expr(sympy_expr)
        new_tex_string = get_new_tex_string(sympy_expr)

        new_string = new_string.replace(tex_string, "heihei")

    return new_string

def fix(readpath, writepath):
    readpath = pl.Path(readpath)
    writepath = pl.Path(writepath)

    if not readpath.suffix == ".html":
        readpath = readpath.parent = (readpath.name + ".html")
    if not writepath.suffix == ".html":
        writepath = writepath.parent / (writepath.name + ".html")

    assert readpath.exists(), f"The file {str(readpath)} does not exist"
    assert readpath.is_file(), f"The object {str(readpath)} is not a file"

    string = read_HTMLFixer(readpath)
    new_string = fix_pipeline(string)
    write_HTMLFixer(writepath, new_string)

if __name__ == "__main__":
    readpath = "html_raw/ccd_energy_and_t2.html"
    writepath  = "html_formatted/ccd_energy_and_t"

    fix(readpath, writepath)