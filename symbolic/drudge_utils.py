from sympy import (Symbol, IndexedBase)
import drudge
from dummy_spark import SparkContext
import pathlib as pl

MAIN_PATH = pl.Path(__file__).parent
HTML_RAW_PATH = MAIN_PATH / "html_raw"
HTML_FORMATTED_PATH = MAIN_PATH / "html_formatted"

def pack_as_list(x):
    if type(x) not in [list, tuple]:
        x = [x]
    
    return x

def get_particle_hole_drudge():
    DEFAULT_PART_DUMMS = tuple(Symbol(i) for i in 'abcdefg') + tuple(
        Symbol('a{}'.format(i)) for i in range(50)
    )
    DEFAULT_HOLE_DUMMS = tuple(Symbol(i) for i in 'ijklmno') + tuple(
        Symbol('i{}'.format(i)) for i in range(50)
    )

    part_orb=(drudge.Range('V', 0, Symbol('nv')), DEFAULT_PART_DUMMS)
    hole_orb=(drudge.Range('O', 0, Symbol('no')), DEFAULT_HOLE_DUMMS)
    
    ctx = SparkContext()
    dr = drudge.PartHoleDrudge(ctx, part_orb=part_orb, hole_orb=hole_orb)
    dr.full_simplify = False

    return dr

def get_indicies(dr, num=4):
    names = dr.names
    o_dums = names.O_dumms[:num]
    v_dums = names.V_dumms[:num]

    if num == 1:
        o_dums, = o_dums
        v_dums, = v_dums

    return o_dums, v_dums

def get_secondquant_operators(dr):
    names = dr.names
    c_ = names.c_
    c_dag = names.c_dag

    return c_, c_dag

def get_X(dr, order, o_dums, v_dums):
    o_dums = pack_as_list(o_dums)
    v_dums = pack_as_list(v_dums)

    assert len(o_dums) == order and len(v_dums) == order

    c_, c_dag = get_secondquant_operators(dr)

    X = c_dag[v_dums[0]]*c_[o_dums[0]]  

    for z in range(1, order):
        X = X*c_dag[v_dums[z]]*c_[o_dums[z]]

    return X

def get_Y(dr, order, o_dums, v_dums):
    o_dums = pack_as_list(o_dums)
    v_dums = pack_as_list(v_dums)

    assert len(o_dums) == order and len(v_dums) == order

    c_, c_dag = get_secondquant_operators(dr)

    X = c_dag[o_dums[0]]*c_[v_dums[0]]  

    for z in range(1, order):
        X = X*c_dag[o_dums[z]]*c_[v_dums[z]]

    return X

def get_clusters_1(dr):
    i, a = get_indicies(dr, num=1)
    t1, l1 = make_rk1(dr, "t"), make_rk1(dr, r"\lambda")
    X1, Y1 = get_X(dr, 1, i, a), get_Y(dr, 1, i, a)
    return (
        dr.einst(t1[a,i]*X1),
        dr.einst(l1[a,i]*Y1)
    )

def get_clusters_2(dr):
    (i, j), (a, b) = get_indicies(dr, num=2)
    t2, l2 = make_rk2(dr, "t"), make_rk2(dr, r"\lambda")
    X2, Y2 = get_X(dr, 2, (i, j), (a,b)), get_Y(dr, 2, (i, j), (a,b))
    return (
        dr.einst(t2[a,b,i,j]*X2/4),
        dr.einst(l2[a,b,i,j]*Y2/4)
    )

def make_rk1(dr, symbol):
    return IndexedBase(f"{symbol}^1")

def make_rk2(dr, symbol):
    t = IndexedBase(f"{symbol}^2")
    dr.set_dbbar_base(t,2)
    return t

def similarity_transform(tensor, clusters):
    curr = tensor
    tensor_bar = tensor

    for order in range(0, 4):
        curr = (curr | clusters).simplify() / (order + 1)
        curr.cache()
        tensor_bar += curr

    tensor_bar.repartition(cache=True)

    return tensor_bar

def get_ob_density_blocks(dr, o_dums, v_dums):
    assert len(o_dums) == 2 and len(v_dums) == 2

    i, j = o_dums
    a, b = v_dums
    c_, c_dag = get_secondquant_operators(dr)

    blocks = [
        c_dag[i]*c_[j],
        c_dag[a]*c_[b],
        c_dag[i]*c_[a]
    ]
    block_names = ["oo", "vv", "ov"]

    return blocks, block_names

def get_tb_density_blocks(dr, o_dums, v_dums):
    assert len(o_dums) == 4 and len(v_dums) == 4

    i, j, k, l = o_dums
    a, b, c, d = v_dums
    c_, c_dag = get_secondquant_operators(dr)

    blocks = [
        c_dag[i]*c_dag[j]*c_[l]*c_[k], # ijkl
        c_dag[a]*c_dag[b]*c_[d]*c_[c], # abcd
        c_dag[i]*c_dag[j]*c_[b]*c_[a], # ijab
        c_dag[a]*c_dag[b]*c_[j]*c_[i], # abij
        c_dag[i]*c_dag[a]*c_[b]*c_[j], # iajb 
    ]
    block_names = ["oooo", "vvvv", "oovv", "vvoo", "ovov"]

    return blocks, block_names


def save_html(dr, filename, equations, titles = None):
    if not filename.endswith(".html"):
        filename = filename + ".html"

    filename = HTML_RAW_PATH / filename
    
    equations = pack_as_list(equations)
    titles = pack_as_list(titles)

    if titles == [None]:
        titles = [f"equation {i+1}" for i in range(len(equations))]

    with dr.report(str(filename), filename.stem) as rep:
        for title, equation in zip(titles, equations):
            rep.add(title, equation)

if __name__ == '__main__':
    dr = get_particle_hole_drudge()

    # X = get_clusters_1(dr)

    T2, L2 = get_clusters_2(dr)

    print(T2)

    print(L2)