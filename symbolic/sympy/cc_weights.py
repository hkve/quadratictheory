from sympy.physics.secondquant import (
    AntiSymmetricTensor,
    wicks,
    F,
    Fd,
    NO,
    evaluate_deltas,
    substitute_dummies,
    Commutator,
    simplify_index_permutations,
    PermutationOperator,
)
from sympy import symbols, expand, pprint, Rational, latex, Dummy, factorial
from utils import (
    get_T1,
    get_T2,
    get_T3,
    get_L1,
    get_L2,
    get_Hamiltonian,
    compute_projection,
    wicks_comm,
    get_permutations_list,
)
import time

pretty_dummies_dict = {"above": "efgh", "below": "mno", "general": "pqrstu"}

L1, L2 = get_L1(), get_L2()
L = 1 + L1 + L2
T1, T2 = get_T1(), get_T2()
T1_ = get_T1()

L1_T1 = wicks(
    L1 * T1,
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)

L1_T1 = substitute_dummies(
    L1_T1, new_indices=True, pretty_indices=pretty_dummies_dict
)

i, j, k, l = symbols("i,j,k,l", below_fermi=True)
a, b, c, d = symbols("a,b,c,d", above_fermi=True)

L2_T1_2 = wicks(
    Rational(1, 4)
    * AntiSymmetricTensor("l", (i, j), (a, b))
    * Fd(i)
    * F(a)
    * Fd(j)
    * F(b)
    * AntiSymmetricTensor("t", (c,), (k,))
    * Fd(c)
    * F(k)
    * AntiSymmetricTensor("t", (d,), (l,))
    * Fd(d)
    * F(l),
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)

L2_T1_2 = substitute_dummies(
    L2_T1_2, new_indices=True, pretty_indices=pretty_dummies_dict
)
L2_T1_2 = simplify_index_permutations(L2_T1_2, [PermutationOperator(i, j)])

L2_T2 = wicks(
    L2 * T2,
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)

L2_T2 = substitute_dummies(
    L2_T2, new_indices=True, pretty_indices=pretty_dummies_dict
)

print(latex(L1_T1))
print(latex(L2_T1_2))
print(latex(L2_T2))

print()
print(f"Singles weights")

L1_S = wicks(
    L1 * Fd(a) * F(i),
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)
L1_S = substitute_dummies(
    L1_S, new_indices=True, pretty_indices=pretty_dummies_dict
)

L2_T1_S = wicks(
    L2 * T1 * Fd(a) * F(i),
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)

L2_T1_S = substitute_dummies(
    L2_T1_S, new_indices=True, pretty_indices=pretty_dummies_dict
)

print(latex(L1_S))
print(latex(L2_T1_S))

print(f"Doubles weights")
L2_D = wicks(
    L2 * Fd(a) * F(i) * Fd(b) * F(j),
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)
L2_D = substitute_dummies(
    L2_D, new_indices=True, pretty_indices=pretty_dummies_dict
)
print(latex(L2_D))

D_T1_2 = wicks(
    Fd(i)
    * F(a)
    * Fd(j)
    * F(b)
    * AntiSymmetricTensor("t", (c,), (k,))
    * Fd(c)
    * F(k)
    * AntiSymmetricTensor("t", (d,), (l,))
    * Fd(d)
    * F(l),
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)
D_T1_2 = substitute_dummies(
    D_T1_2, new_indices=True, pretty_indices=pretty_dummies_dict
)
print(latex(D_T1_2))
