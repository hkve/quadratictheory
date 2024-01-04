from sympy import Symbol, IndexedBase
import drudge
from dummy_spark import SparkContext

import pickle
import pathlib as pl
import functools
import time

MAIN_PATH = pl.Path(__file__).parent
HTML_RAW_PATH = MAIN_PATH / "html_raw"
HTML_FORMATTED_PATH = MAIN_PATH / "html_formatted"
PICKLE_RAW_PATH = MAIN_PATH / "pickle_results"

class Time(drudge.Stopwatch):
    def __init__(self, vocal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocal = vocal

    def tock(self, label):
        if self.vocal:
            super().tock(label)


timer = Time(vocal=False)


def timeme(func):
    @functools.wraps(func)
    def wrapper_timeme(*args, **kwargs):
        print(f"Starting {func.__name__!r} ...\n")
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs\n")
        return value

    return wrapper_timeme


def pack_as_list(x):
    if type(x) not in [list, tuple]:
        x = [x]

    return x


def get_particle_hole_drudge():
    DEFAULT_PART_DUMMS = tuple(Symbol(i) for i in "abcdefg") + tuple(
        Symbol("a{}".format(i)) for i in range(50)
    )
    DEFAULT_HOLE_DUMMS = tuple(Symbol(i) for i in "ijklmno") + tuple(
        Symbol("i{}".format(i)) for i in range(50)
    )

    part_orb = (drudge.Range("V", 0, Symbol("nv")), DEFAULT_PART_DUMMS)
    hole_orb = (drudge.Range("O", 0, Symbol("no")), DEFAULT_HOLE_DUMMS)

    ctx = SparkContext()
    dr = drudge.PartHoleDrudge(ctx, part_orb=part_orb, hole_orb=hole_orb)
    dr.full_simplify = False

    return dr


def get_indicies(dr, num=4):
    names = dr.names
    o_dums = names.O_dumms[:num]
    v_dums = names.V_dumms[:num]

    if num == 1:
        (o_dums,) = o_dums
        (v_dums,) = v_dums

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

    X = c_dag[v_dums[0]] * c_[o_dums[0]]

    for z in range(1, order):
        X = X * c_dag[v_dums[z]] * c_[o_dums[z]]

    return X


def get_Y(dr, order, o_dums, v_dums):
    o_dums = pack_as_list(o_dums)
    v_dums = pack_as_list(v_dums)

    assert len(o_dums) == order and len(v_dums) == order

    c_, c_dag = get_secondquant_operators(dr)

    X = c_dag[o_dums[0]] * c_[v_dums[0]]

    for z in range(1, order):
        X = X * c_dag[o_dums[z]] * c_[v_dums[z]]

    return X


def get_clusters_1(dr):
    i, a = get_indicies(dr, num=1)
    t1, l1 = make_rk1(dr, "t"), make_rk1(dr, r"\lambda")
    X1, Y1 = get_X(dr, 1, i, a), get_Y(dr, 1, i, a)
    return (dr.einst(t1[a, i] * X1), dr.einst(l1[a, i] * Y1))


def get_clusters_2(dr):
    (i, j), (a, b) = get_indicies(dr, num=2)
    t2, l2 = make_rk2(dr, "t"), make_rk2(dr, r"\lambda")
    X2, Y2 = get_X(dr, 2, (i, j), (a, b)), get_Y(dr, 2, (i, j), (a, b))
    return (dr.einst(t2[a, b, i, j] * X2 / 4), dr.einst(l2[a, b, i, j] * Y2 / 4))


def make_rk1(dr, symbol):
    return IndexedBase(f"{symbol}^1")


def make_rk2(dr, symbol):
    t = IndexedBase(f"{symbol}^2")
    dr.set_dbbar_base(t, 2)
    return t


def define_rk0_rhs(dr, equation):
    return dr.define(Symbol("e"), equation)


def define_rk1_rhs(dr, equation, symbol="r"):
    i, a = get_indicies(dr, num=1)
    r1 = make_rk1(dr, symbol)
    return dr.define(r1[a, i], equation)


def define_rk2_rhs(dr, equation, symbol="r"):
    (i, j), (a, b) = get_indicies(dr, num=2)
    r2 = make_rk2(dr, symbol)
    return dr.define(r2[a, b, i, j], equation)

def diff_rk2_antisym(term, var, o, v):
    i, j = o
    a, b = v

    term_derivative = (term.diff(var[a,b,i,j])
                     - term.diff(var[b,a,i,j]) 
                     - term.diff(var[a,b,j,i])
                     + term.diff(var[b,a,j,i]))/4
    term_derivative = term_derivative.simplify()

    return term_derivative

def similarity_transform(tensor, clusters):
    stopwatch = None

    curr = tensor
    tensor_bar = tensor

    for order in range(0, 4):
        curr = (curr | clusters).simplify() / (order + 1)
        curr.cache()
        tensor_bar += curr
        timer.tock(f"Commutator at order {order+1}")

    tensor_bar.repartition(cache=True)

    return tensor_bar


def get_ob_density_blocks(dr, o_dums, v_dums):
    assert len(o_dums) == 2 and len(v_dums) == 2

    i, j = o_dums
    a, b = v_dums
    c_, c_dag = get_secondquant_operators(dr)

    blocks = [c_dag[i] * c_[j], c_dag[a] * c_[b], c_dag[i] * c_[a]]
    block_names = ["oo", "vv", "ov"]

    return blocks, block_names


def get_tb_density_blocks(dr, o_dums, v_dums):
    assert len(o_dums) == 4 and len(v_dums) == 4

    i, j, k, l = o_dums
    a, b, c, d = v_dums
    c_, c_dag = get_secondquant_operators(dr)

    blocks = [
        c_dag[i] * c_dag[j] * c_[l] * c_[k],  # ijkl
        c_dag[a] * c_dag[b] * c_[d] * c_[c],  # abcd
        c_dag[i] * c_dag[j] * c_[b] * c_[a],  # ijab
        c_dag[a] * c_dag[b] * c_[j] * c_[i],  # abij
        c_dag[i] * c_dag[a] * c_[b] * c_[j],  # iajb
        c_dag[i] * c_dag[j] * c_[a] * c_[k],  # ijka
        c_dag[i] * c_dag[a] * c_[k] * c_[j],  # iajk
        c_dag[a] * c_dag[b] * c_[i] * c_[c],  # abci
        c_dag[a] * c_dag[i] * c_[c] * c_[b],  # aibc
    ]
    block_names = ["oooo", "vvvv", "oovv", "vvoo", "ovov", "ooov", "ovoo", "vvvo", "vovv"]

    return blocks, block_names


def define_tb_density_blocks(dr, rho, block_names, o_dums, v_dums):
    assert len(o_dums) == 4 and len(v_dums) == 4

    i, j, k, l = o_dums
    a, b, c, d = v_dums

    rs = [IndexedBase(f"\rho_{name}") for name in block_names]
    for r in rs:
        dr.set_dbbar_base(r, 2)

    blocks = [
        rs[0][i, j, k, l],
        rs[1][a, b, c, d],
        rs[2][i, j, a, b],
        rs[3][a, b, i, j],
        rs[4][i, a, j, b],
        rs[5][i, j, k, a],
        rs[6][i, a, j, k],
        rs[7][a, b, c, i],
        rs[8][a, i, b, c],
    ]

    return [dr.define(rhs, term) for rhs, term in zip(blocks, rho)]


def save_html(dr, filename, equations, titles=None):
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


def save_to_pickle(terms, filename):
    if not filename.endswith(".pickle"): 
        filename = filename + ".pickle"

    filename = PICKLE_RAW_PATH / filename

    with open(filename, "wb") as file:
        pickle.dump(terms, file)

def load_from_pickle(dr, filename):
    if not filename.endswith(".pickle"): 
        filename = filename + ".pickle"

    filename = PICKLE_RAW_PATH / filename

    with dr.pickle_env():
        with open(filename, "rb") as file:
            terms_loaded = pickle.load(file)

    return terms_loaded

if __name__ == "__main__":
    dr = get_particle_hole_drudge()

    # X = get_clusters_1(dr)

    T2, L2 = get_clusters_2(dr)

    print(T2)

    print(L2)
