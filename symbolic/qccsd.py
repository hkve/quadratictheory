import drudge_utils as drutils
import gristmill_utils as grutils
import gristmill

def calculate_expressions(dr):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T, L = T1+T2, L1+L2

    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T)

    t1, l1 = drutils.make_rk1(dr, "t"), drutils.make_rk1(dr, "\lambda")
    t2, l2 = drutils.make_rk2(dr, "t"), drutils.make_rk2(dr, "\lambda")
    (i,j), (a,b) = drutils.get_indicies(dr, num=2)

    # Performing algebraic derivatives
    energy_terms = ((1 + L + L*L/2)*ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("QCCSD lagrangian addition done")

    t1_terms = energy_terms.diff(t1[a,i]).simplify()
    drutils.timer.tock("QCCSD t1 done")
    t2_terms = (4*drutils.diff_rk2_antisym(energy_terms, l2, (i,j), (a,b))).simplify()
    drutils.timer.tock("QCCSD t2 done")

    l1_terms = energy_terms.diff(l1[a,i]).simplify()
    drutils.timer.tock("QCCSD l1 done")
    l2_terms = (4*drutils.diff_rk2_antisym(energy_terms, t2, (i,j), (a,b))).simplify()
    drutils.timer.tock("QCCSD l2 done")
    drutils.timer.tock("DONE DOING BASE ALGEBRA")

    # Define equations and saving einsums, save algebraic equations in raw form
    energy_eq = drutils.define_rk0_rhs(dr, energy_terms)
    grutils.einsum_raw(dr, "qccsd_energy_from_lag", [energy_eq])

    t1_equations = drutils.define_rk1_rhs(dr, t1_terms)
    l1_equations = drutils.define_rk1_rhs(dr, l1_terms)
    grutils.einsum_raw(dr, "qccsd_t1_from_lag", [t1_equations])
    grutils.einsum_raw(dr, "qccsd_l1_from_lag", [l1_equations])

    t2_equations = drutils.define_rk2_rhs(dr, t2_terms)
    l2_equations = drutils.define_rk2_rhs(dr, l2_terms)
    grutils.einsum_raw(dr, "qccsd_t2_from_lag", [t2_equations])
    grutils.einsum_raw(dr, "qccsd_l2_from_lag", [l2_equations])
    
    all_raw_eqs = [energy_eq, t1_equations, l1_equations, t2_equations, l2_equations]
    drutils.save_html(dr, "qccsd_from_lag", all_raw_eqs, ["Energy", "t1=0", "l1=0", "t2=0", "l2=0"])
    drutils.save_to_pickle(all_raw_eqs, "qccsd_from_lag")
    drutils.timer.tock("DONE SAVING RAW EQUATIONS")

    return all_raw_eqs

def optimize(dr, all_raw_eqs):
    # Creating intermediates, saving einsums and intermediate algebraic equations
    opti_strat = {
        "contr_strat": gristmill.ContrStrat.OPT,
        # "drop_cutoff": 4,
    }
    energy_eq, t1_equations, l1_equations, t2_equations, l2_equations = all_raw_eqs

    eval_seq_e = grutils.optimize_equations(dr, energy_eq, **opti_strat)
    grutils.einsum_raw(dr, "qccsd_energy_optimized_from_lag", eval_seq_e)
    drutils.timer.tock("QCCSD energy opti done")
    
    eval_seq_t = grutils.optimize_equations(dr, [t1_equations, t2_equations], **opti_strat)
    grutils.einsum_raw(dr, "qccsd_t_optimized_from_lag", eval_seq_t)
    drutils.timer.tock("QCCD t opti done")

    
    eval_seq_l = grutils.optimize_equations(dr, [l1_equations, l2_equations], **opti_strat)
    grutils.einsum_raw(dr, "qccsd_l_optimized_from_lag", eval_seq_l)
    drutils.timer.tock("QCCSD l opti done")
    
    all_raw_eqs = [eval_seq_e, eval_seq_t, eval_seq_l]
    drutils.save_to_pickle(all_raw_eqs, "qccsd_opti_from_lag")
    drutils.save_html(dr, "qccsd_opti_lagrangian", eval_seq_e)
    drutils.save_html(dr, "qccsd_opti_t", eval_seq_t)
    drutils.save_html(dr, "qccsd_opti_l", eval_seq_l)

@drutils.timeme
def run(dr):
    calculate = False

    all_raw_eqs = None
    if calculate:
        all_raw_eqs = calculate_expressions(dr)
    else:
        all_raw_eqs = drutils.load_from_pickle(dr, "qccsd_from_lag")

    optimize(dr, all_raw_eqs)

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()

    drutils.timer.vocal = True
    run(dr)