import drudge_utils as drutils
import gristmill_utils as grutils

def define_rhs(dr, term, rank):
    if rank == 0:
        equation = drutils.define_rk0_rhs(dr, term)
    elif rank == 1:
        equation = drutils.define_rk1_rhs(dr, term)
    elif rank == 2:
        equation = drutils.define_rk2_rhs(dr, term)
    else:
        print("Uknown rank here, how manu free indicies do you have?")
        exit()

    return equation

def load(dr, basename):
    equations = {}
    names = ["t1", "t2"]

    for name in names:
        equation = drutils.load_from_pickle(dr, basename + f"_{name}")
        equations[name] = equation

    return equations

@drutils.timeme
def get_t_equation(dr, ham_bar, Y, L, filename=None):
    term = (Y * L * ham_bar).eval_fermi_vev().simplify()
    rank = len(Y.terms[0].args[2]) // 2
    equation = define_rhs(dr, term, rank)

    if filename is not None: drutils.save_to_pickle(equation, filename)

    return equation

def calculate_expressions(dr, basename):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T, L = T1+T2, L1+L2

    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T)
    
    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y1 = drutils.get_Y(dr, 1, (i,), (a,))
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))

    equations = {}

    # T1 QCCSD addition
    t1_equation = get_t_equation(dr, ham_bar, Y1, L, basename + "_t1")
    equations["t1"] = t1_equation
    grutils.einsum_raw(dr, basename + "_t1", t1_equation)
    drutils.timer.tock("QCCSD t1 done")

    # T2 QCCSD addition
    t2_equation = get_t_equation(dr, ham_bar, Y2, L, basename + "_t2")
    equations["t2"] = t2_equation
    grutils.einsum_raw(dr, basename + "_t2", t2_equation)
    drutils.timer.tock("QCCSD t2 done")

    return equations

def optimize_expressions(dr, equations):
    for name, eq in equations.items():
        eval_seq = grutils.optimize_equations(dr, eq)
        drutils.timer.tock(f"QCCSD {name} optimization done")

        grutils.einsum_raw(dr, basename + "_opti_" + name, eval_seq)
        drutils.save_html(dr, basename + "_opti_" + name, eval_seq)

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()
    drutils.timer.vocal = True
    
    basename = "qccsd_testing"

    equations = calculate_expressions(dr, basename)
    # equations = load(dr, basename)
    optimize_expressions(dr, equations)