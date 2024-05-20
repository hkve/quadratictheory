import drudge_utils as drutils
import gristmill_utils as grutils

from sympy import IndexedBase, Rational, symbols, Symbol
from sympy import Indexed
from sympy.physics.secondquant import PermutationOperator, simplify_index_permutations

# For Restricted Permutation Operator
from sympy import Expr, sympify, default_sort_key, Basic, Dummy, S
from sympy import Mul

from IPython import embed

def permutations(self, P_list=None):
    restricted = self.drudge.__class__.__name__ == "RestrictedPartHoleDrudge"

    if P_list is None:
        if restricted:
            P_list = _default_restricted_permutations(self.drudge)
        else:
            P_list = _default_general_permutations(self.drudge)

    if type(P_list) != list:
        P_list = [P_list]

    sympy_amps = _simplify_index_permutations(self, self._terms, expanded=self._expanded, P_list=P_list, restricted=restricted)

    return self.drudge.einst(sympy_amps)

def _default_general_permutations(dr):
    (i,j), (a,b) = drutils.get_indicies(dr, num=2)
    P_ij = PermutationOperator(i,j)
    P_ab = PermutationOperator(a,b)

    return [P_ij, P_ab]

def _default_restricted_permutations(dr):
    (i,j), (a,b) = drutils.get_indicies(dr, num=2)
    P = RestrictedPermutationOperator(i,j,a,b)

    return [P]

def get_permutation_until_order(dr, order):
    o_dumms, v_dumms = drutils.get_indicies(dr, num=order)

    P_holes = []
    P_parts = []
    for p in range(order):
        for q in range(p+1, order):
            i, a = o_dumms[p], v_dumms[p]
            j, b = o_dumms[q], v_dumms[q]

            Pij = PermutationOperator(i,j)
            Pab = PermutationOperator(a,b)
            
            P_holes.append(Pij)
            P_parts.append(Pab)
    
    return P_holes + P_parts

def _is_in_alphabetical_order(indicies):
    str_indicies = [str(i) for i in indicies]
    return str_indicies == sorted(str_indicies)

def _sort_alphabetically(amps_sympy, restricted):
    new_amps_sympy = 0
    for term in amps_sympy.args:
        new_amplitudes = []

        term_new = 1
        for amplitude in term.as_ordered_factors():
            if type(amplitude) == Indexed:
                indicies = amplitude.indices

                if len(indicies) != 4:
                    term_new *= amplitude
                    continue
                
                if restricted:
                    amplitude_new = _get_restricted_alphabetical_ordered(amplitude, indicies)
                else:
                    amplitude_new = _get_alphabetical_ordered(amplitude, indicies)

                term_new *= amplitude_new
            
            else:
                term_new *= amplitude
 
        new_amps_sympy += term_new
                
    return new_amps_sympy


def _get_alphabetical_ordered(amplitude, indicies):
    p, q, r, s = indicies

    permute_bra = not _is_in_alphabetical_order([p, q])
    permute_ket = not _is_in_alphabetical_order([r, s])
    
    amplitude_new = amplitude.copy()
    if permute_bra:
        P = PermutationOperator(p, q)
        amplitude_new = P.get_permuted(amplitude_new)
    if permute_ket:
        P = PermutationOperator(r, s)
        amplitude_new = P.get_permuted(amplitude_new)

    return amplitude_new

def _get_restricted_alphabetical_ordered(amplitude, indicies):
    p, q, r, s = indicies
    permute_bra = not _is_in_alphabetical_order([p, q])
    permute_ket = not _is_in_alphabetical_order([r, s])

    both_free_bra = all([str(i) in ["a", "b", "i", "j"] for i in [p,q]])
    both_free_ket = all([str(i) in ["a", "b", "i", "j"] for i in [r,s]])

    amplitude_new = amplitude.copy()
    P = RestrictedPermutationOperator(p, q, r, s)
    if permute_bra and not both_free_bra:
        amplitude_new = P.get_permuted(amplitude_new)
    elif permute_ket and not both_free_ket:
        amplitude_new = P.get_permuted(amplitude_new)
    
    return amplitude_new

def _simplify_index_permutations(self, terms, expanded, P_list, restricted):
    if not expanded:
        terms = self._expand(terms)

    resolvers = self._drudge.resolvers

    amps_sympy = 0
    for term in terms.collect():
        fixed_term = _fix_strange_u(self._drudge, term)
        amps_sympy += fixed_term.amp

    # amps = terms.map(lambda x: x.amp)
    # amps_sympy = sum(list(amps.collect()))
        
    amps_sympy = _sort_alphabetically(amps_sympy, restricted)

    amps_sympy = simplify_index_permutations(amps_sympy, P_list)

    return amps_sympy

def _swap_dummy_index(dr, term, p, q):
    pdums, hdums, = dr.DEFAULT_PART_DUMMS, dr.DEFAULT_HOLE_DUMMS

    is_part = p in pdums and q in pdums
    is_hole = p in hdums and q in hdums

    assert is_part != is_hole, f"Must swap dummy indicies with same range {p}, {q} are from different ranges"

    dums = pdums if is_part else hdums

    term = term.subst({p: dums[-1]})
    term = term.subst({q: p})
    term = term.subst({dums[-1]: q})

    return term

def _fix_strange_u(dr, term):
    (i,j,k,l), (a,b,c,d) = drutils.get_indicies(dr, num=4)
    u = IndexedBase("u")

    for part in term.exprs:
        if type(part) == Mul:
            if part.has(u[k,l,d,c]):
                term = _swap_dummy_index(dr, term, c, d) 
            if part.has(u[l,k,c,d]):
                term = _swap_dummy_index(dr, term, k, l)

    return term 

class RestrictedPermutationOperator(Expr):
    """
    Represents the index permutation operator P((ai)(bj)).

    P((ai)(bj))*f(i)*g(j)*c(a)*d(b) = f(i)*g(j)*c(a)*d(b) + f(j)*g(i)*c(b)*d(a)
    """
    is_commutative = True

    def __new__(cls, i, j, a, b):
        i, j, a, b = sorted(map(sympify, (i, j, a, b)), key=default_sort_key)
        obj = Basic.__new__(cls, i, j, a, b)
        return obj

    def get_permuted(self, expr):
        i = self.args[0]
        j = self.args[1]
        a = self.args[2]
        b = self.args[3]
        if expr.has(i) and expr.has(j) and expr.has(a) and expr.has(b):
            tmp = Dummy()
            expr = expr.subs(i, tmp)
            expr = expr.subs(j, i)
            expr = expr.subs(tmp, j)

            expr = expr.subs(a, tmp)
            expr = expr.subs(b, a)
            expr = expr.subs(tmp, b)
            return expr
        else:
            return expr

    def _latex(self, printer):
        i, j, a, b = self.args[0], self.args[1], self.args[2], self.args[3]
        return f"P(({i}{a})({j}{b}))"

if __name__ == '__main__':
    dr = drutils.get_particle_hole_drudge(dummy=True)
    drutils.timer.vocal = True

    t2, l2 = drutils.make_rk2(dr, "t"), drutils.make_rk2(dr, r"\lambda")
    u = dr.names.u

    (i,j,k,l), (a,b,c,d) = drutils.get_indicies(dr, num=4)
    
    term = dr.einst(
        l2[a,c,i,k]*t2[c,d,k,l]*u[l,j,b,d]
    )
    term = term - dr.einst(
        l2[b,c,i,k]*t2[c,d,k,l]*u[j,l,d,a]
    )

    new_term = permutations(term)

    embed()

