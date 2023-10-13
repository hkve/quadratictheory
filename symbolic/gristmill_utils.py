import gristmill
from sympy import IndexedBase, Symbol
import drudge
import drudge_utils as utils

EINSUM_RAW_PATH = utils.MAIN_PATH / "einsum_raw"
EINSUM_FORMATTED_PATH = utils.MAIN_PATH / "einsum_formatted"


def get_working_equations(dr, equations, ranks=[0, 2]):
    equations = utils.pack_as_list(equations)

    (i, j), (a, b) = utils.get_indicies(num=max(ranks))
    working_eqs = []
    for equation, rank in zip(equations, ranks):
        working_eq = None

        if rank == 0:
            e = Symbol("e")
            working_eq = dr.define(e, equation)
        if rank == 1:
            r = utils.make_rk1(dr, "r1")
            working_eq = dr.define(r[a, i], equation)
        if rank == 2:
            r = utils.make_rk2(dr, "r2")
            working_eq = dr.define(r[a, b, i, j], equation)

        working_eqs.append(working_eq)

    return working_eqs


def einsum_raw(dr, filename, working_eqs):
    if not filename.endswith(".txt"):
        filename += ".txt"
    filename = EINSUM_RAW_PATH / filename

    printer = gristmill.EinsumPrinter()
    with open(str(filename), "w") as outfile:
        outfile.write(printer.doprint(working_eqs))
