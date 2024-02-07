import drudge_utils as drutils
import gristmill_utils as grutils

def load(dr, basename):
    equations = {}
    names = ["energy_td_part", "t1", "t2", "l1", "l2"]

    from IPython import embed
    for name in names:
        equation = drutils.load_from_pickle(dr, basename + f"_{name}")
        embed()
        equations[name] = equation

    return equations

def run(dr, basename):
    ham = dr.ham  
    equations = {}

    T1, L1 = drutils.get_restricted_clusters_1(dr)
    T2, L2 = drutils.get_restricted_clusters_2(dr)

    T, L = T2, L2
    e_ = drutils.get_restricted_secondquant_operator(dr)
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    ham_bar = drutils.similarity_transform(ham, T)

    energy = (ham_bar).eval_fermi_vev().simplify()
    energy = drutils.define_rk0_rhs(dr, energy)
    energy_td_part = ( (L*L/2) * ham_bar).eval_fermi_vev().simplify()
    energy_td_part = drutils.define_rk0_rhs(dr, energy_td_part)
    equations["energy_td_part"] = energy_td_part
    grutils.einsum_raw(dr, basename + "_energy_td_part", energy_td_part)
    drutils.save_to_pickle(energy_td_part, basename + "_energy_td_part")
    drutils.timer.tock("Done RQCCD energy")

    t2_eq = (e_[i,a]*e_[j,b]*ham_bar).eval_fermi_vev().simplify()
    t2_eq = drutils.define_rk2_rhs(dr, t2_eq)
    equations["t2"] = t2_eq
    grutils.einsum_raw(dr, basename + "_t2", t2_eq)
    drutils.save_to_pickle(t2_eq, basename + "_t2")
    drutils.timer.tock("Done RQCCD t2")


    com = ham | e_[a,i]*e_[b,j]
    com_bar = drutils.similarity_transform(com, T)
    l2_eq = ((L*L/2)*com_bar).eval_fermi_vev().simplify()
    l2_eq = drutils.define_rk2_rhs(dr, l2_eq)
    equations["l2"] = l2_eq
    grutils.einsum_raw(dr, basename + "_l2", l2_eq)
    drutils.save_to_pickle(l2_eq, basename + "_l2")
    drutils.timer.tock("Done RQCCD l2")

    return equations

def optimize_expressions(dr, equations, basename):
    for name, eq in equations.items():
        eval_seq = grutils.optimize_equations(dr, eq)
        drutils.timer.tock(f"RQCCD {name} optimization done")

        drutils.save_to_pickle(eval_seq, basename + "_opti_" + name)
        drutils.save_html(dr, basename + "_opti_" + name, eval_seq)
        grutils.einsum_raw(dr, basename + "_opti_" + name, eval_seq)

if __name__ == "__main__":
    drutils.timer.vocal = True
    dr = drutils.get_restricted_particle_hole_drudge()
    
    basename = "rqccd"

    equations = run(dr, basename)
    # equations = load(dr, basename)

    optimize_expressions(dr, equations, basename)