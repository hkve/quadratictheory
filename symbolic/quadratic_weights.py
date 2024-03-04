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

def get_operators(dr):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)

    T1 = T1.simplify()
    T2 = T2.simplify()
    L1 = L1.simplify()
    L2 = L2.simplify()
    
    T1.cache()
    T2.cache()
    L1.cache()
    L2.cache()

    return T1, T2, L1, L2

@drutils.timeme
def reference(dr):
    T1, T2, L1, L2 = get_operators(dr)

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
    T1, T2, L1, L2 = get_operators(dr)

    i, a = drutils.get_indicies(dr, num=1)
    X1 = drutils.get_X(dr, 1, i, a)
    
    X1 = X1.simplify()
    X1.cache()
    
    # Double excitations
    expr = -L1*L1*T1
    res = (expr*X1).eval_fermi_vev().simplify()
    print("Dobules done")

    # Triple excitations
    expr = -2*L1*L2*T2
    res += (expr*X1).eval_fermi_vev().simplify()
    print("Tripls 1 done")

    expr = -L1*L2*T1*T1
    res += (expr*X1).eval_fermi_vev().simplify()
    print("Tripls 2 done")
    
    # Quadruple excitations
    expr = L2*L2*T1*T2
    res += (expr*X1).eval_fermi_vev().simplify()
    print("Quad 1 done")

    expr = Rational(1,6)*L2*L2*T1*T1*T1
    res += (expr*X1).eval_fermi_vev().simplify()
    print("Quad 2 done")

    # Add overall factor and calculate
    res = (Rational(1,2)*res).simplify()
    print("Overal factor")

    res = dr.define(IndexedBase(f"det")[a,i], res)
    save(dr, "qccsd_1p1h_weight", res)

@drutils.timeme
def double_excited(dr):
    T1, T2, L1, L2 = get_operators(dr)

    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    X2 = drutils.get_X(dr, 2, (i,j), (a,b))
    
    # Double excitations
    expr = L1*L1
    res = (expr*X2).eval_fermi_vev().simplify()
    drutils.timer.tock(f"Doubles done", res)

    # Triple excitations
    expr = -2*L1*L2*T1
    res += (expr*X2).eval_fermi_vev().simplify()
    drutils.timer.tock(f"Triples done", res)
    
    # Quadruple excitations
    expr = -L2*L2*T2
    res += (expr*X2).eval_fermi_vev().simplify()
    
    expr = Rational(1,2)*L2*L2*T1*T1
    res += (expr*X2).eval_fermi_vev().simplify()
    drutils.timer.tock(f"Quadruples done", res)

    # Add overall factor and calculate
    res = (Rational(1,2)*res).simplify()

    res = dr.define(IndexedBase(f"det")[a,b,i,j], res)
    save(dr, "qccsd_2p2h_weight", res)

def triple_excited(dr):
    T1, T2, L1, L2 = get_operators(dr)

    (i, j, k), (a, b, c) = drutils.get_indicies(dr, num=3)
    X3 = drutils.get_X(dr, 3, (i,j,k), (a,b,c))
    Y3 = drutils.get_Y(dr, 3, (i,j,k), (a,b,c))
   
    expr = T1*T2
    ket_res = (Y3*expr).eval_fermi_vev().simplify()

    expr = Rational(1,6)*T1*T1*T1
    ket_res += (Y3*expr).eval_fermi_vev().simplify()

    expr = 2*L1*L2
    bra_res = (expr*X3).eval_fermi_vev().simplify()

    expr = -L2*L2*T1
    bra_res += (expr*X3).eval_fermi_vev().simplify()

    bra_res *= Rational(1,2)

    bra_res = bra_res.simplify()
    ket_res = ket_res.simplify()

    save(dr, "qccsd_3p3h_weight_ket", ket_res)
    save(dr, "qccsd_3p3h_weight_bra", bra_res)

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge(dummy=True)

    drutils.timer.vocal = True
    # reference(dr)
    # single_excited(dr)
    double_excited(dr)