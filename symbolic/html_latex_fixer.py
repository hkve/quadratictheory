import pathlib as pl
import re

from sympy import symbols, IndexedBase
from sympy.physics.secondquant import (
    AntiSymmetricTensor, simplify_index_permutations, PermutationOperator
)
from sympy import (
    print_latex, Rational, Symbol, Dummy
)

def read_HTMLFixer(path):
    string = ""
    with open(path, "r") as file:
        for line in file:
            string += line

    return string

def write_HTMLFixer(path, string):
    with open(path, "w") as file:
        file.write(string)

def find_rational(term, sign):
    rational = 0
    frac = re.findall(r"\\frac{(.*?)}{(.*?)}", term)

    if frac != []:
        up, down = map(int, frac[0])
        rational = Rational(up, down)
    else:
        rational = 1

    if sign == "+":
        sign = 1
    else:
        sign = -1

    return sign*rational

def find_one_body_tensors(tex_term, im, tensors=["f", "t", r"\\lambda"]):
    one_body = {}
    for tensor in tensors:
        pat = rf"{tensor}" + r"_{(.*?)}"
        x = re.findall(pat, tex_term)
        
        if x != []:
            one_body[tensor] = list(x[0])

    term = 1
    for tensor, indicies in one_body.items():
        sym_tensor = IndexedBase(tensor)
        p, q = indicies
        term *= sym_tensor[im[p],im[q]]

    return term

def find_two_body_tensors(tex_term, im, tensors=["u","t",r"\\lambda"]):
    two_body = {}
    for tensor in tensors:
        pat = rf"{tensor}" + r"\^{(.*?)}_{(.*?)}"
        x = re.findall(pat, tex_term)
        
        if x != []:
            two_body[tensor] = {"upper": list(x[0][0]), "lower": list(x[0][1])}

    term = 1
    for tensor, indicies in two_body.items():
        p, q = indicies["upper"]
        r, s = indicies["lower"]
        sym_tensor = AntiSymmetricTensor(tensor, (im[p], im[q]), (im[r], im[s]))
        term *= sym_tensor

    return term

def count_index_occurrences(string):
    raw_indicies = re.findall(r"{[a-z]{2}}", string)
    indicies = []

    for pair in raw_indicies:
        pair = pair.replace("{", "").replace("}", "")
        indicies += list(pair)

    unique_indicies = list(set(indicies))
    counts = [0]*len(unique_indicies)

    for i, unique in enumerate(unique_indicies):
        counts[i] = indicies.count(unique)

    return unique_indicies,counts

def get_indicies(tex_term):
    unique_indicies, counts = count_index_occurrences(tex_term)
    upper = "abcdefg"

    index_map = {}
    for index, count in zip(unique_indicies, counts):
        cls = Dummy if count == 2 else Symbol
        above_fermi = True if index in upper else False
        
        index_map[index] = symbols(index, above_fermi=above_fermi, cls=cls)

    return index_map

def get_free_indicies(tex_string):
    unique_indices, counts = count_index_occurrences(tex_string)
    
    free = {"upper": [], "lower": []}
    upper = "abcdefg"

    for index, count in zip(unique_indices, counts):
        if count == 1:
            above_fermi = index in upper
            free[above_fermi*"upper" + (1-above_fermi)*"lower"].append(symbols(index, above_fermi=above_fermi))

    return free

def add_sympy_term(tex_term):
    if not tex_term.startswith("+") and not tex_term.startswith("-"):
        tex_term = "+" + tex_term

    index_map = get_indicies(tex_term)

    sign = tex_term[0]
    rational = find_rational(tex_term, sign)
    one_body_tensors = find_one_body_tensors(tex_term, index_map)
    two_body_tensors = find_two_body_tensors(tex_term, index_map)

    return rational*one_body_tensors*two_body_tensors

def get_sympy_expr(tex_string):
    tex_string = re.sub(r"\\sum_{.*?}", "", tex_string)
    tex_string = re.sub(r"t\^{\d}_{(.*?),(.*?),(.*?),(.*?)}", r"t^{\1\2}_{\3\4}", tex_string)
    tex_string = re.sub(r"u_{(.*?),(.*?),(.*?),(.*?)}", r"u^{\1\2}_{\3\4}", tex_string)
    tex_string = re.sub(r"f_{(.*?),(.*?)}", r"f_{\1\2}", tex_string).strip()

    matches = [match.start() for match in re.finditer(r"[+-]", tex_string)]
    matches = [0] + matches + [len(tex_string)]

    free_indicies = get_free_indicies(tex_string[matches[0] : matches[1]])

    eq = 0
    for i in range(len(matches)-1):
        tex_term = tex_string[matches[i] : matches[i+1]]
        term = add_sympy_term(tex_term)
        eq += term

    return eq, free_indicies

def fix_sympy_expr(expr, fix_sympy_expr):
    permutation_operators = []

    for index in fix_sympy_expr.values():
        if index !=  []:
            permutation_operators.append(
                PermutationOperator(index[0], index[1])
            )
    
    if permutation_operators != []:
        for arg in expr.args:
            print(arg)
        expr2 = simplify_index_permutations(expr, permutation_operators)
        
        print("--------- FIXED")
        for arg in expr2.args:
            print(arg)
    

def get_new_tex_string(expr):
    pass

def fix_pipeline(string):
    new_string = string
    tex_strings = re.findall(r"\\\[(.*?)\\\]", string)

    for tex_string in tex_strings:
        sympy_expr, free_indicies = get_sympy_expr(tex_string)
        sympy_expr = fix_sympy_expr(sympy_expr, free_indicies)
        # print(sympy_expr)
        # new_tex_string = get_new_tex_string(sympy_expr)
        
        # new_string = new_string.replace(tex_string, sympy_expr)
    exit()
    return new_string

def fix(readpath, writepath):
    readpath = pl.Path(readpath)
    writepath = pl.Path(writepath)

    if not readpath.suffix == ".html":
        readpath = readpath.parent = (readpath.name + ".html")
    if not writepath.suffix == ".html":
        writepath = writepath.parent / (writepath.name + ".html")

    assert readpath.exists(), f"The file {str(readpath)} does not exist"
    assert readpath.is_file(), f"The object {str(readpath)} is not a file"

    string = read_HTMLFixer(readpath)
    new_string = fix_pipeline(string)
    write_HTMLFixer(writepath, new_string)

if __name__ == "__main__":
    a, b = symbols("a,b", above_fermi=True)
    i, j = symbols("i,j", above_fermi=True)

    c, d = symbols("c,d", above_fermi=True, cls=Dummy)
    k, l = symbols("k,l", above_fermi=False, cls=Dummy)

    f = IndexedBase("f")
    t1 = AntiSymmetricTensor("t", (a,c), (i,j))
    t2 = AntiSymmetricTensor("t", (b,c), (i,j))

    P = PermutationOperator(a, b)

    expr = f[b,c]*t1  - f[b,c]*t1*t2 - f[a,c]*t2
    for arg in expr.args:
        print(arg)
    print("-------- FIXED")
    expr2 = simplify_index_permutations(expr, [P])
    for arg in expr2.args:
        print(arg)
    print("\n\n")
    
    readpath = "html_raw/ccd_energy_and_t2.html"
    writepath  = "html_formatted/ccd_energy_and_t"

    fix(readpath, writepath)