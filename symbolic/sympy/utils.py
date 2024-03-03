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

pretty_dummies_dict = {"above": "efgh", "below": "mno", "general": "pqrstu"}


def get_permutations_list():
    i, j, k, l, m, n = symbols("i,j,k,l, m, n", below_fermi=True)
    a, b, c, d, e, f = symbols("a,b,c,d, e, f", above_fermi=True)
    return [
        PermutationOperator(a, b),
        PermutationOperator(a, c),
        PermutationOperator(a, d),
        PermutationOperator(a, d),
        PermutationOperator(a, e),
        PermutationOperator(a, f),
        PermutationOperator(b, c),
        PermutationOperator(b, d),
        PermutationOperator(b, e),
        PermutationOperator(b, f),
        PermutationOperator(c, d),
        PermutationOperator(c, e),
        PermutationOperator(c, f),
        PermutationOperator(d, e),
        PermutationOperator(d, f),
        PermutationOperator(e, f),
        PermutationOperator(i, j),
        PermutationOperator(i, k),
        PermutationOperator(i, l),
        PermutationOperator(i, m),
        PermutationOperator(i, n),
        PermutationOperator(j, k),
        PermutationOperator(j, l),
        PermutationOperator(j, m),
        PermutationOperator(j, n),
        PermutationOperator(k, l),
        PermutationOperator(k, m),
        PermutationOperator(k, n),
        PermutationOperator(l, m),
        PermutationOperator(l, n),
        PermutationOperator(m, n),
    ]


def wicks_comm(A, B):
    A_B = wicks(Commutator(A, B))
    A_B = evaluate_deltas(A_B)
    A_B = substitute_dummies(A_B)
    return A_B


def compute_projection(left, commutators, right=1, Permutations=[]):
    for I in range(0, len(commutators)):
        rhs = wicks(
            left * commutators[I] * right,
            simplify_dummies=True,
            keep_only_fully_contracted=True,
            simplify_kronecker_deltas=True,
        )

        rhs = substitute_dummies(
            rhs, new_indices=True, pretty_indices=pretty_dummies_dict
        )
        if len(Permutations) > 0:
            rhs = simplify_index_permutations(rhs, Permutations)

        print(latex(rhs))
        print()


def get_Hamiltonian():
    """Generates normal ordered Hamiltonian. Remember to include the reference
    energy in the energy expressions. That is,
        E_ref = f^{i}_{i} - 0.5 * u^{ij}_{ij}
            = h^{i}_{i} + 0.5 * u^{ij}_{ij}.
    """
    p, q, r, s = symbols("p, q, r, s", cls=Dummy)
    f = AntiSymmetricTensor("f", (p,), (q,))
    u = AntiSymmetricTensor("u", (p, q), (r, s))

    f = f * NO(Fd(p) * F(q))
    u = u * NO(Fd(p) * Fd(q) * F(s) * F(r))

    return f, Rational(1, 4) * u


def get_T1(ast_symb="t"):
    i = symbols("i", below_fermi=True, cls=Dummy)
    a = symbols("a", above_fermi=True, cls=Dummy)

    t_ai = AntiSymmetricTensor(ast_symb, (a,), (i,))
    c_ai = NO(Fd(a) * F(i))

    T_1 = t_ai * c_ai

    return T_1


def get_T2(ast_symb="t"):
    i, j = symbols("i, j", below_fermi=True, cls=Dummy)
    a, b = symbols("a, b", above_fermi=True, cls=Dummy)

    t_abij = AntiSymmetricTensor(ast_symb, (a, b), (i, j))
    c_abij = NO(Fd(a) * Fd(b) * F(j) * F(i))

    T_2 = Rational(1, 4) * t_abij * c_abij

    return T_2


def get_T3(ast_symb="t"):
    i, j, k = symbols("i, j, k", below_fermi=True, cls=Dummy)
    a, b, c = symbols("a, b, c", above_fermi=True, cls=Dummy)

    t_abcijk = AntiSymmetricTensor(ast_symb, (a, b, c), (i, j, k))
    c_abcijk = NO(Fd(a) * Fd(b) * Fd(c) * F(k) * F(j) * F(i))

    T_3 = Rational(1, 36) * t_abcijk * c_abcijk

    return T_3


def get_L1(ast_symb="l"):
    i = symbols("i", below_fermi=True, cls=Dummy)
    a = symbols("a", above_fermi=True, cls=Dummy)

    l_ia = AntiSymmetricTensor(ast_symb, (i,), (a,))
    c_ia = NO(Fd(i) * F(a))

    L_1 = l_ia * c_ia

    return L_1


def get_L2(ast_symb="l"):
    i, j = symbols("i, j", below_fermi=True, cls=Dummy)
    a, b = symbols("a, b", above_fermi=True, cls=Dummy)

    l_ijab = AntiSymmetricTensor(ast_symb, (i, j), (a, b))
    c_ijab = NO(Fd(i) * Fd(j) * F(b) * F(a))

    L_2 = Rational(1, 4) * l_ijab * c_ijab

    return L_2


def get_L3(ast_symb="l"):
    i, j, k = symbols("i, j, k", below_fermi=True, cls=Dummy)
    a, b, c = symbols("a, b, c", above_fermi=True, cls=Dummy)

    l_ijkabc = AntiSymmetricTensor(ast_symb, (i, j, k), (a, b, c))
    c_ijkabc = NO(Fd(i) * Fd(j) * Fd(k) * F(c) * F(b) * F(a))

    L_3 = Rational(1, 36) * l_ijkabc * c_ijkabc

    return L_3
