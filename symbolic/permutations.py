import drudge_utils as drutils
import gristmill_utils as grutils

from sympy import IndexedBase, Rational, symbols, Symbol
from sympy import Indexed
from sympy.physics.secondquant import PermutationOperator, simplify_index_permutations

# For Restricted Permutation Operator
from sympy import Expr, sympify, default_sort_key, Basic, Dummy, S
# from sympy.core.expr import Expr
# from sympy.core.sympify import sympify
# from sympy.core.sorting import default_sort_key
# from sympy.core.basic import Basic
# from sympy.core.symbol import Dummy
# from sympy.core.singleton import S

from IPython import embed

def permutations(self, P_list=None):
    if P_list is None:
        (i,j), (a,b) = drutils.get_indicies(self.drudge, num=2)
        P_ij = PermutationOperator(i,j)
        P_ab = PermutationOperator(a,b)
        P_list = [P_ij, P_ab]

    if type(P_list) != list:
        P_list = [P_list]

    sympy_amps = _simplify_index_permutations(self, self._terms, expanded=self._expanded, P_list=P_list)

    return self.drudge.einst(sympy_amps)

def _is_in_alphabetical_order(indicies):
    str_indicies = [str(i) for i in indicies]
    return str_indicies == sorted(str_indicies)

def _sort_alphabetically(amps_sympy):
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

                term_new *= amplitude_new
            
            else:
                term_new *= amplitude
 
        new_amps_sympy += term_new
                
    return new_amps_sympy

def _simplify_index_permutations(self, terms, expanded, P_list):
    if not expanded:
        terms = self._expand(terms)

    resolvers = self._drudge.resolvers

    amps = terms.map(lambda x: x.amp)

    amps_sympy = sum(list(amps.collect()))
    amps_sympy = _sort_alphabetically(amps_sympy)

    amps_sympy = simplify_index_permutations(amps_sympy, P_list)

    return amps_sympy

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