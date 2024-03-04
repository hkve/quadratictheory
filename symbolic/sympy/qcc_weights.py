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
i, j, k, l, I, J, K, L = symbols("i,j,k,l,I,J,K,L", below_fermi=True)
a, b, c, d, A, B, C, D = symbols("a,b,c,d,A,B,C,D", above_fermi=True)

# L1_sq_T2 = wicks(
#     Rational(1, 2)*Rational(1, 4)
#     *AntiSymmetricTensor("l", (i,), (a,))*Fd(i)*F(a)
#     *AntiSymmetricTensor("l", (j,), (b,))*Fd(j)*F(b)
#     *AntiSymmetricTensor("t", (c,d), (k,l))*Fd(c)*F(k)*Fd(d)*F(l),
#     simplify_dummies=True,
#     keep_only_fully_contracted=True,
#     simplify_kronecker_deltas=True,
# )

# print(
#     latex(L1_sq_T2)
# )
L2_sq_T1_ft = wicks(
    Rational(1, 48)*Rational(1,4)*Rational(1,4)
    *AntiSymmetricTensor("l", (i, j), (a, b))*Fd(i)*F(a)*Fd(j)*F(b)
    *AntiSymmetricTensor("l", (k, l), (c, d))*Fd(k)*F(c)*Fd(l)*F(d)
    *AntiSymmetricTensor("t", (A,), (I,))*Fd(A)*F(I)
    *AntiSymmetricTensor("t", (B,), (J,))*Fd(B)*F(J)
    *AntiSymmetricTensor("t", (C,), (K,))*Fd(C)*F(K)
    *AntiSymmetricTensor("t", (D,), (L,))*Fd(D)*F(L),
    simplify_dummies=True,
    keep_only_fully_contracted=True,
    simplify_kronecker_deltas=True,
)

L2_sq_T1_ft = substitute_dummies(
    L2_sq_T1_ft, new_indices=True, pretty_indices=pretty_dummies_dict
)
P_list = [PermutationOperator(B,C)]
L2_sq_T1_ft = simplify_index_permutations(L2_sq_T1_ft, P_list)

L2_sq_T1_ft = L2_sq_T1_ft.simplify()
common_term = AntiSymmetricTensor("t", (A,), (I,)) * AntiSymmetricTensor("t", (B,), (J,)) * AntiSymmetricTensor("t", (C,), (K,)) * AntiSymmetricTensor("t", (D,), (L,))
L2_sq_T1_ft = L2_sq_T1_ft / common_term   

P_list_hole = [PermutationOperator(I,J), PermutationOperator(I,K), PermutationOperator(I, L), PermutationOperator(J,K), PermutationOperator(J,L),PermutationOperator(K,L)]
P_list_part = [PermutationOperator(A,B), PermutationOperator(A,C), PermutationOperator(A,D), PermutationOperator(B,C), PermutationOperator(B,D), PermutationOperator(C,D)]
P_list = P_list_hole + P_list_part

L2_sq_T1_ft = simplify_index_permutations(L2_sq_T1_ft, P_list)
L2_sq_T1_ft = common_term * L2_sq_T1_ft
print(
    latex(L2_sq_T1_ft)
)

# from IPython import embed
# embed()