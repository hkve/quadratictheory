"""Automatic derivation of resitrict CCDF theory.

This script serves as an example of using drudge for complex symbolic
manipulations.  The derivation here is going to be based on the approach in GE
Scuseria et al, J Chem Phys 89 (1988) 7382 (10.1063/1.455269).

"""

from pyspark import SparkConf, SparkContext
from sympy import IndexedBase, Rational, symbols, Symbol

from drudge import RestrictedPartHoleDrudge, Stopwatch

import drudge_utils as drutils
import gristmill_utils as grutils

stopwatch = Stopwatch()

def run_t(dr):
    p = dr.names
    e_ = p.e_
    a, b, c, d = p.V_dumms[:4]
    i, j, k, l = p.O_dumms[:4]

    #
    # Cluster excitation operator
    # 
    # Here, we first write the cluster excitation operator in terms of the
    # unitary group generator.  Then they will be substituted by their fermion
    # operator definition.
    #

    t = IndexedBase('t')

    cluster = dr.einst(
        t[a, i] * e_[a, i] +
        Rational(1, 2) * t[a, b, i, j] * e_[a, i] * e_[b, j]
    )

    dr.set_n_body_base(t, 2)
    cluster = cluster.simplify()
    cluster.cache()

    #
    # Similarity transform of the Hamiltonian
    # 

    curr = dr.ham
    h_bar = dr.ham
    for order in range(4):
        curr = (curr | cluster).simplify() * Rational(1, order + 1)
        stopwatch.tock('Commutator order {}'.format(order + 1), curr)
        h_bar += curr
        continue

    h_bar = h_bar.simplify()
    h_bar.repartition(cache=True)
    stopwatch.tock('H-bar assembly', h_bar)

    en_eqn = h_bar.eval_fermi_vev().simplify()
    stopwatch.tock('Energy equation', en_eqn)

    dr.wick_parallel = 1

    beta, gamma, i48, i49 = symbols("beta gamma i48 i49")
    r = IndexedBase("r")

    Y1 = e_[i48, beta]
    Y2 = e_[i48, beta] * e_[i49, gamma]

    t1 = (Y1 * h_bar).eval_fermi_vev().simplify()
    t1_eq = dr.define(r[beta, i48], t1)
    t1_eq = t1_eq[a,i]

    t2 = (Y2 * h_bar).eval_fermi_vev().simplify()
    t2_eq = dr.define(r[beta, gamma, i48, i49], t2)
    t2_eq = t2_eq[a,b,i,j]

    amp_eqns = [t1_eq, t2_eq]

    drutils.save_to_pickle(amp_eqns, "rccsd_t_amplitudes")
    drutils.save_html(dr, "rccsd_t_amplitudes", amp_eqns, titles=["T_1", "T_2"])
    return amp_eqns

# Environment setting up.

conf = SparkConf().setAppName('rccsd')
ctx = SparkContext(conf=conf)
dr = RestrictedPartHoleDrudge(ctx)
dr.full_simplify = False

# amp_eqns = run_t(dr)
amp_eqns = drutils.load_from_pickle(dr, "rccsd_t_amplitudes")

from IPython import embed
embed()

stopwatch.tock_total()

t1_eval_seq = grutils.optimize_equations(dr, amp_eqns[0])
t2_eval_seq = grutils.optimize_equations(dr, amp_eqns[1])

drutils.save_html(dr, "rccsd_t1_opti", t1_eval_seq)
drutils.save_html(dr, "rccsd_t2_opti", t2_eval_seq)

grutils.einsum_raw(dr, "rccsd_t1_opti", t1_eval_seq)
grutils.einsum_raw(dr, "rccsd_t2_opti", t2_eval_seq)