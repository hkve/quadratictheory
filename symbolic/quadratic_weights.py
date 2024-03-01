import drudge_utils as drutils
import gristmill_utils as grutils
from sympy import Rational, Symbol, IndexedBase
from IPython import embed

def save(dr, name, equation):
    drutils.save_html(dr, name, equation)
    drutils.save_to_pickle(equation, name)
    grutils.einsum_raw(dr, name, equation)

    eval_seq = grutils.optimize_equations(dr, equation)
    drutils.save_html(dr, f"{name}_optimized", eval_seq)
    drutils.save_to_pickle(eval_seq, f"{name}_optimized")
    grutils.einsum_raw(dr, f"{name}_optimized", eval_seq)


@drutils.timeme
def reference(dr):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)

    # Double excitations
    expr = -L1*L1*T2
    expr += Rational(1,2)*L1*L1*T1*T1
    
    # # Triple excitations
    expr += 2*L1*L2*T1*T2
    expr += Rational(1,3)*L1*L2*T1*T1*T1

    # # Quadruple excitations
    expr += Rational(1,2)*L2*L2*T2*T2
    expr += Rational(1,24)*L2*L2*T1*T1*T1*T1

    # # Add overall factor and calculate
    expr *= Rational(1,2)
    expr = expr.simplify()
    expr = expr.eval_fermi_vev().simplify()
    
    res = dr.define(Symbol("ref"), expr)
    save(dr, "qccsd_ref_weight", res)

@drutils.timeme
def single_excited(dr):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)

    i, a = drutils.get_indicies(dr, num=1)
    X1 = drutils.get_X(dr, 1, i, a)

    # Double excitations
    expr = -L1*L1*T1
    res = (expr*X1).eval_fermi_vev().simplify()

    # Triple excitations
    expr = -2*L1*L2*T2
    res += (expr*X1).eval_fermi_vev().simplify()

    expr = -L1*L2*T1*T1
    res += (expr*X1).eval_fermi_vev().simplify()
    
    # Quadruple excitations
    expr = L2*L2*T1*T2
    res += (expr*X1).eval_fermi_vev().simplify()
    
    expr = Rational(1,6)*L2*L2*T1*T1*T1
    res += (expr*X1).eval_fermi_vev().simplify()

    # Add overall factor and calculate
    res = (Rational(1,2)*res).simplify()
    
    res = dr.define(IndexedBase(f"det")[a,i], res)
    save(dr, "qccsd_1p1h_weight", res)

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()

    drutils.timer.vocal = True
    reference(dr)