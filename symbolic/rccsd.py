import drudge_utils as drutils
import gristmill_utils as grutils
import pathlib as pl

def load(dr, basename):
    equations = {}
    names = ["t1", "t2", "l1", "l2"]

    for name in names:
        equation = drutils.load_from_pickle(dr, basename + f"_{name}")
        # drutils.save_html(dr, basename + f"_{name}", equation)

        equations[name] = equation

    return equations

def run(dr, basename):
    ham = dr.ham  
    equations = {}

    T1, L1 = drutils.get_restricted_clusters_1(dr)
    T2, L2 = drutils.get_restricted_clusters_2(dr)

    T = (T1 + T2).simplify()
    L = (L1 + L2).simplify()

    e_ = drutils.get_restricted_secondquant_operator(dr)
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    ham_bar = drutils.similarity_transform(ham, T)

    energy = (ham_bar).eval_fermi_vev().simplify()
    energy = drutils.define_rk0_rhs(dr, energy)
    energy_td_part = (L*ham_bar).eval_fermi_vev().simplify()
    energy_td_part = drutils.define_rk0_rhs(dr, energy_td_part)
    equations["energy_td_part"] = energy_td_part
    grutils.einsum_raw(dr, basename + "_energy_td_part", energy_td_part)
    drutils.save_to_pickle(energy_td_part, basename + "_energy_td_part")
    drutils.timer.tock("Done RCCSD energy")

    t1_eq = (e_[i,a]*ham_bar).eval_fermi_vev().simplify()
    t1_eq = drutils.define_rk1_rhs(dr, t1_eq)
    equations["t1"] = t1_eq
    grutils.einsum_raw(dr, basename + "_t1", t1_eq)
    drutils.save_to_pickle(t1_eq, basename + "_t1")
    drutils.timer.tock("Done RCCSD t1")

    t2_eq = (e_[i,a]*e_[j,b]*ham_bar).eval_fermi_vev().simplify()
    t2_eq = drutils.define_rk2_rhs(dr, t2_eq)
    equations["t2"] = t2_eq
    grutils.einsum_raw(dr, basename + "_t2", t2_eq)
    drutils.save_to_pickle(t2_eq, basename + "_t2")
    drutils.timer.tock("Done RCCSD t2")

    com = ham | e_[a,i]
    com_bar = drutils.similarity_transform(com, T)
    l1_eq = ((1+L)*com_bar).eval_fermi_vev().simplify()
    l1_eq = drutils.define_rk1_rhs(dr, l1_eq)
    equations["l1"] = l1_eq
    grutils.einsum_raw(dr, basename + "_l1", l1_eq)
    drutils.save_to_pickle(l1_eq, basename + "_l1")
    drutils.timer.tock("Done RCCSD l1")

    com = ham | e_[a,i]*e_[b,j]
    com_bar = drutils.similarity_transform(com, T)
    l2_eq = ((1+L)*com_bar).eval_fermi_vev().simplify()
    l2_eq = drutils.define_rk2_rhs(dr, l2_eq)
    equations["l2"] = l2_eq
    grutils.einsum_raw(dr, basename + "_l2", l2_eq)
    drutils.save_to_pickle(l2_eq, basename + "_l2")
    drutils.timer.tock("Done RCCSD l2")

    return equations

def optimize_expressions(dr, equations, basename):
    for name, eq in equations.items():
        eval_seq = grutils.optimize_equations(dr, eq)
        drutils.timer.tock(f"RCCSD {name} optimization done")

        drutils.save_to_pickle(eval_seq, basename + "_opti_" + name)
        drutils.save_html(dr, basename + "_opti_" + name, eval_seq)
        grutils.einsum_raw(dr, basename + "_opti_" + name, eval_seq)

if __name__ == "__main__":
    drutils.timer.vocal = True
    dr = drutils.get_restricted_particle_hole_drudge()
    
    basename = "rccsd"

    # equations = run(dr, basename)
    equations = load(dr, basename)

    optimize_expressions(dr, equations, basename)