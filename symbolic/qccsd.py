import drudge_utils as drutils
import gristmill_utils as grutils
from sympy import Rational

from IPython import embed
from permutations import permutations
from latex import pretty

def get_bits(dr, T1_trans=False):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    ham = dr.ham

    T = T2 if T1_trans else (T1+T2).simplify()
    L = (L1+L2).simplify()

    T.cache()
    L.cache()
    ham.cache()

    return T, L, ham

@drutils.timeme
def energy_addition(dr, filename, L, ham_bar):
    energy = (Rational(1,2)*L*L*ham_bar).eval_fermi_vev().simplify()
    energy_eq = drutils.define_rk0_rhs(dr, energy)
    drutils.timer.tock(f"Done QCCSD energy addition, saving to {filename}", energy_eq)

    drutils.save_html(dr, filename, energy_eq)
    drutils.save_to_pickle(energy_eq, filename)
    grutils.einsum_raw(dr, filename, energy_eq)

@drutils.timeme
def t1_additon(dr, filename, L, ham_bar):
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y1 = drutils.get_Y(dr, 1, (i,), (a,))

    t1 = (Y1*L*ham_bar).eval_fermi_vev().simplify()
    t1_eq = drutils.define_rk1_rhs(dr, t1)
    drutils.timer.tock(f"Done QCCSD T1 addition, saving to {filename}", t1_eq)

    drutils.save_html(dr, filename, t1_eq)
    drutils.save_to_pickle(t1_eq, filename)
    grutils.einsum_raw(dr, filename, t1_eq)

@drutils.timeme
def t2_additon(dr, filename, L, ham_bar):
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y2 = drutils.get_Y(dr, 2, (i,j), (a,b))

    t2 = (Y2*L*ham_bar).eval_fermi_vev().simplify()
    t2_eq = drutils.define_rk2_rhs(dr, t2)
    drutils.timer.tock(f"Done QCCSD T2 addition, saving to {filename}", t2_eq)

    drutils.save_html(dr, filename, t2_eq)
    drutils.save_to_pickle(t2_eq, filename)
    grutils.einsum_raw(dr, filename, t2_eq)

@drutils.timeme
def l1_additon(dr, filename, L, ham_bar):
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    X1 = drutils.get_X(dr, 1, (i,), (a,))

    com = ham_bar | X1

    l1 = (Rational(1,2)*L*L*com).eval_fermi_vev().simplify()
    l1_eq = drutils.define_rk1_rhs(dr, l1)
    drutils.timer.tock(f"Done QCCSD L1 addition, saving to {filename}", l1_eq)

    drutils.save_html(dr, filename, l1_eq)
    drutils.save_to_pickle(l1_eq, filename)
    grutils.einsum_raw(dr, filename, l1_eq)

@drutils.timeme
def l2_additon(dr, filename, L, ham_bar):
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    X2 = drutils.get_X(dr, 2, (i,j), (a,b))

    com = ham_bar | X2

    l2 = (Rational(1,2)*L*L*com).eval_fermi_vev().simplify()
    l2_eq = drutils.define_rk2_rhs(dr, l2)
    drutils.timer.tock(f"Done QCCSD L2 addition, saving to {filename}", l2_eq)

    drutils.save_html(dr, filename, l2_eq)
    drutils.save_to_pickle(l2_eq, filename)
    grutils.einsum_raw(dr, filename, l2_eq)

@drutils.timeme
def optimize(dr, filename):
    eq = drutils.load_from_pickle(dr, filename)
    new_filename = f"{filename}_optimized"

    eval_seq_eq = grutils.optimize_equations(dr, eq)
    drutils.timer.tock(f"Done QCCSD optimazation from {filename}, saving to {new_filename}", eq)

    drutils.save_html(dr, new_filename, eval_seq_eq)
    drutils.save_to_pickle(eval_seq_eq, new_filename)
    grutils.einsum_raw(dr, new_filename, eval_seq_eq)

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge(dummy=True)

    drutils.timer.vocal = True
    
    # Use these for full qccsd
    # filenames = {
    #     "e": "TEST_qccsd_energy_addition",
    #     "t1": "TEST_qccsd_t1_addition",
    #     "t2": "TEST_qccsd_t2_addition",
    #     "l1": "TEST_qccsd_l1_addition",
    #     "l2": "TEST_qccsd_l2_addition",
    # }

    # Filenames for T1-transformed Hamiltonian
    filenames = {
        "e": "TEST_qccsd_t1trans_energy_addition",
        "t1": "TEST_qccsd_t1trans_t1_addition",
        "t2": "TEST_qccsd_t1trans_t2_addition",
        "l1": "TEST_qccsd_t1trans_l1_addition",
        "l2": "TEST_qccsd_t1trans_l2_addition",
    }

    T, L, ham = get_bits(dr, T1_trans=True)
    ham_bar = drutils.similarity_transform(ham, T)

    energy_addition(dr, filename=filenames["e"], L=L, ham_bar=ham_bar)

    t1_additon(dr, filename=filenames["t1"], L=L, ham_bar=ham_bar)
    t2_additon(dr, filename=filenames["t2"], L=L, ham_bar=ham_bar)

    l1_additon(dr, filename=filenames["l1"], L=L, ham_bar=ham_bar)
    l2_additon(dr, filename=filenames["l2"], L=L, ham_bar=ham_bar)

    for name, filename in filenames.items():
        optimize(dr, filename)