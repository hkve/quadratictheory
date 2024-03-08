import drudge_utils as drutils
import gristmill_utils as grutils
from sympy import Rational, Symbol, IndexedBase
from permutations import permutations
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

    expr = L1*L1 - 2*L1*L2*T1 + L2*L2*(-T2 + Rational(1,2)*T1*T1)

    # Add overall factor and calculate
    res = (Rational(1,2)*expr*X2).eval_fermi_vev().simplify()

    res = dr.define(IndexedBase(f"bra")[a,b,i,j], res)
    # save(dr, "qccsd_2p2h_weight", res)
    from IPython import embed
    embed()

def triple_excited(dr):
    T1, T2, L1, L2 = get_operators(dr)

    (i, j, k), (a, b, c) = drutils.get_indicies(dr, num=3)
    X3 = drutils.get_X(dr, 3, (i,j,k), (a,b,c))
    Y3 = drutils.get_Y(dr, 3, (i,j,k), (a,b,c))
   
    expr = T1*T2 + Rational(1,6)*T1*T1*T1
    ket_res = (Y3*expr).eval_fermi_vev().simplify()

    expr = 2*L1*L2 -L2*L2*T1
    bra_res = (expr*X3).eval_fermi_vev().simplify()

    bra_res = bra_res.simplify()
    bra_res *= Rational(1,2)
    
    ket_res = ket_res.simplify()

    bra_lhs = IndexedBase("bra")
    dr.set_dbbar_base(bra_lhs, 3)
    bra = dr.define(bra_lhs[a,b,c,i,j,k], bra_res)

    ket_lhs = IndexedBase("ket")
    dr.set_dbbar_base(ket_lhs, 3)
    ket = dr.define(ket_lhs[a,b,c,i,j,k], ket_res)

    res = dr.einst(Rational(1,36)*bra*ket).simplify()

    W_T_lhs = Symbol("WT")
    W_T = dr.define(W_T_lhs, res)

    grutils.einsum_raw(dr, "triples_weight", W_T)


def quadrouple_excited(dr):
    T1, T2, L1, L2 = get_operators(dr)

    (i, j, k, l), (a, b, c, d) = drutils.get_indicies(dr, num=4)
    X4 = drutils.get_X(dr, 4, (i,j,k,l), (a,b,c,d))
    Y4 = drutils.get_Y(dr, 4, (i,j,k,l), (a,b,c,d))
   
    expr = Rational(1,2)*T2*T2 + Rational(1,4)*T1*T1*T2
    ket_res = (Y4*expr).eval_fermi_vev().simplify()

    expr = L2*L2
    bra_res = (expr*X4).eval_fermi_vev().simplify()

    bra_res = bra_res.simplify()
    bra_res *= Rational(1,2)
    
    ket_res = ket_res.simplify()

    bra_lhs = IndexedBase("bra")
    dr.set_dbbar_base(bra_lhs, 4)
    bra = dr.define(bra_lhs[a,b,c,d,i,j,k,l], bra_res)

    ket_lhs = IndexedBase("ket")
    dr.set_dbbar_base(ket_lhs, 4)
    ket = dr.define(ket_lhs[a,b,c,d,i,j,k,l], ket_res)

    res = dr.einst(Rational(1,576)*bra*ket).simplify()

    W_Q_lhs = Symbol("WQ")
    W_Q = dr.define(W_Q_lhs, res)

    grutils.einsum_raw(dr, "quadruplets_weight", W_Q)



if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge(dummy=True)

    drutils.timer.vocal = True
    # reference(dr)
    # single_excited(dr)
    double_excited(dr)
    # triple_excited(dr)
    # quadrouple_excited(dr)