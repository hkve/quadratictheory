import gristmill
from sympy import IndexedBase, Symbol
import drudge
import drudge_utils as utils
from einsum_print_fixer import fix

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
            # r = utils.make_rk1(dr, "r")
            working_eq = dr.define(r[a, i], equation)
        if rank == 2:
            r = utils.make_rk2(dr, "r")
            working_eq = dr.define(r[a, b, i, j], equation)

        working_eqs.append(working_eq)

    return working_eqs


def optimize_equations(dr, equations, **kwargs):
    options = {
        "contr_strat": gristmill.ContrStrat.EXHAUST,
    }
    options.update(kwargs)
    
    equations = utils.pack_as_list(equations)

    orig_cost = gristmill.get_flop_cost(equations, leading=True)
    print(f"Cost before optimazation {orig_cost}")

    eval_seq = gristmill.optimize(
        equations,
        substs={dr.names.nv: 5000, dr.names.no: 1000},
        **options
    )

    opt_cost = gristmill.get_flop_cost(eval_seq, leading=True)
    print(f"Cost after optimization {opt_cost}")

    return eval_seq


def einsum_raw(dr, filename, working_eqs):
    working_eqs = utils.pack_as_list(working_eqs)

    if not filename.endswith(".txt"):
        filename += ".txt"
    rfilename = EINSUM_RAW_PATH / filename

    printer = gristmill.EinsumPrinter()
    with open(str(rfilename), "w") as outfile:
        outfile.write(printer.doprint(working_eqs))

    ffilename = EINSUM_FORMATTED_PATH / filename
    fix(rfilename, ffilename)
