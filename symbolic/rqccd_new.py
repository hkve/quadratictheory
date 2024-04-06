import drudge_utils as drutils
import gristmill_utils as grutils
import gristmill 
from sympy import Rational


from IPython import embed
from permutations import permutations
from latex import pretty

def get_bits(dr):
    T2, L2 = drutils.get_restricted_clusters_2(dr)
    ham = dr.ham

    T = (T2).simplify()
    L = (L2).simplify()

    T.cache()
    L.cache()
    ham.cache()

    return T, L, ham

@drutils.timeme
def energy_addition(dr, filename, L, ham_bar):
    energy = (Rational(1,2)*L*L*ham_bar).eval_fermi_vev().simplify()
    energy_eq = drutils.define_rk0_rhs(dr, energy)
    drutils.timer.tock(f"Done RQCCD energy addition, saving to {filename}", energy_eq)

    drutils.save_html(dr, filename, energy_eq)
    drutils.save_to_pickle(energy_eq, filename)
    grutils.einsum_raw(dr, filename, energy_eq)

@drutils.timeme
def t2_additon(dr, filename, L, ham_bar):
    _, Y2 = drutils.get_restricted_Y(dr)

    t2 = (Y2*L*ham_bar).eval_fermi_vev().simplify()
    t2_eq = drutils.define_rk2_rhs_restrictd(dr, t2)
    drutils.timer.tock(f"Done RQCCD T2 addition, saving to {filename}", t2_eq)

    drutils.save_html(dr, filename, t2_eq)
    drutils.save_to_pickle(t2_eq, filename)
    grutils.einsum_raw(dr, filename, t2_eq)


@drutils.timeme
def l2_additon(dr, filename, L, ham_bar):
    _, X2 = drutils.get_restricted_X(dr)

    com = ham_bar | X2

    l2 = (Rational(1,2)*L*L*com).eval_fermi_vev().simplify()
    l2_eq = drutils.define_rk2_rhs_restrictd(dr, l2)
    drutils.timer.tock(f"Done RQCCD L2 addition, saving to {filename}", l2_eq)

    drutils.save_html(dr, filename, l2_eq)
    drutils.save_to_pickle(l2_eq, filename)
    grutils.einsum_raw(dr, filename, l2_eq)

@drutils.timeme
def optimize(dr, filename):
    eq = drutils.load_from_pickle(dr, filename)
    new_filename = f"{filename}_optimized"

    # gristmill.ContrStrat.GREEDY
    # gristmill.ContrStrat.OPT
    # gristmill.ContrStrat.TRAV
    # gristmill.ContrStrat.EXHAUST
    eval_seq_eq = grutils.optimize_equations(dr, eq, contr_strat=gristmill.ContrStrat.OPT)
    drutils.timer.tock(f"Done RQCCD optimazation from {filename}, saving to {new_filename}", eq)

    drutils.save_html(dr, new_filename, eval_seq_eq)
    drutils.save_to_pickle(eval_seq_eq, new_filename)
    grutils.einsum_raw(dr, new_filename, eval_seq_eq)

if __name__ == "__main__":
    dr = drutils.get_restricted_particle_hole_drudge(dummy=False)
    drutils.timer.vocal = True
    
    filenames = {
        "e": "TEST_rqccd_energy_addition",
        # "t2": "TEST_rqccd_t2_addition",
        # "l2": "TEST_rqccd_l2_addition",
    }


    # T, L, ham = get_bits(dr)
    # ham_bar = drutils.similarity_transform(ham, T)

    # energy_addition(dr, filename=filenames["e"], L=L, ham_bar=ham_bar)
    # t2_additon(dr, filename=filenames["t2"], L=L, ham_bar=ham_bar)
    # l2_additon(dr, filename=filenames["l2"], L=L, ham_bar=ham_bar)

    for name, filename in filenames.items():
        optimize(dr, filename)
