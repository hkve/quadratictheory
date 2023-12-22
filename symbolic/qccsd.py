import drudge_utils as drutils
import gristmill_utils as grutils

@drutils.timeme
def run(dr):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T, L = T1+T2, L1+L2

    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T)

    t1, l1 = drutils.make_rk1(dr, "t"), drutils.make_rk1(dr, "\lambda")
    t2, l2 = drutils.make_rk2(dr, "t"), drutils.make_rk2(dr, "\lambda")
    (i,j), (a,b) = drutils.get_indicies(dr, num=2)

    energy_eq = ((1 + L + L*L/2)*ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("QCCD lagrangian addition done")

    t1_terms = energy_eq.diff(t1[a,i]).simplify()
    drutils.timer.tock("QCCD t1 done")
    t2_terms = (4*drutils.diff_rk2_antisym(energy_eq, l2, (i,j), (a,b))).simplify()
    drutils.timer.tock("QCCD t2 done")

    l1_terms = energy_eq.diff(l1[a,i]).simplify()
    drutils.timer.tock("QCCD l1 done")
    l2_terms = (4*drutils.diff_rk2_antisym(energy_eq, t2, (i,j), (a,b))).simplify()
    drutils.timer.tock("QCCD l2 done")

    e = drutils.define_rk0_rhs(dr, energy_eq)
    grutils.einsum_raw(dr, "qccsd_energy_from_lag", [e])
    eval_seq_e = grutils.optimize_equations(dr, e)
    grutils.einsum_raw(dr, "qccsd_energy_optimized_from_lag", eval_seq_e)
    drutils.timer.tock("QCCSD energy opti done")

    t1_equations = drutils.define_rk1_rhs(dr, t1_terms)
    l1_equations = drutils.define_rk1_rhs(dr, l1_terms)

    t2_equations = drutils.define_rk2_rhs(dr, t2_terms)
    l2_equations = drutils.define_rk2_rhs(dr, l2_terms)

    grutils.einsum_raw(dr, "qccsd_t1_from_lag", [t1_equations])
    grutils.einsum_raw(dr, "qccsd_t2_from_lag", [t2_equations])

    eval_seq_t = grutils.optimize_equations(dr, [t1_equations, t2_equations])
    grutils.einsum_raw(dr, "qccsd_t_optimized_from_lag", eval_seq_t)
    drutils.timer.tock("QCCD t opti done")

    grutils.einsum_raw(dr, "qccsd_l1_from_lag", [l1_equations])
    grutils.einsum_raw(dr, "qccsd_l2_from_lag", [l2_equations])

    eval_seq_l = grutils.optimize_equations(dr, [l1_equations, l2_equations])
    grutils.einsum_raw(dr, "qccsd_l_optimized_from_lag", eval_seq_l)
    drutils.timer.tock("QCCSD l opti done")

    drutils.save_html(dr, "qccsd_from_lag", [energy_eq, t1_equations, l1_equations, t2_equations, l2_equations], ["Energy", "t1=0", "l1=0", "t2=0", "l2=0"])

    drutils.save_html(dr, "qccsd_opti_lagrangian", eval_seq_e)
    drutils.save_html(dr, "qccsd_opti_t", eval_seq_t)
    drutils.save_html(dr, "qccsd_opti_l", eval_seq_l)

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()

    drutils.timer.vocal = True
    run(dr)