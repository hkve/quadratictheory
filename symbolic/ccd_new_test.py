import drudge_utils as drutils
import gristmill_utils as grutils

def optimize_and_dump(dr, equation, name):
    drutils.timer.tock(f"Done {name} calculation", equation)
    drutils.save_html(dr, name, equation)
    drutils.save_to_pickle(equation, name)
    grutils.einsum_raw(dr, name, equation)

    eval_seq = grutils.optimize_equations(dr, equation, check_result=True)

    drutils.save_html(dr, f"{name}_optimized", eval_seq)
    drutils.save_to_pickle(eval_seq, f"{name}_optimized")
    grutils.einsum_raw(dr, f"{name}_optimized", eval_seq)
    drutils.timer.tock(f"Done {name}  optimazation", equation)

@drutils.timeme
def run(dr):
    T2, L2 = drutils.get_clusters_2(dr)
    ham = dr.ham
    # ham_bar = drutils.similarity_transform(ham, T2)

    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))
    X2 = drutils.get_X(dr, 2, (i, j), (a, b))

    # td_energy = (L2*ham_bar).eval_fermi_vev().simplify()
    # td_energy = drutils.define_rk0_rhs(dr, td_energy)
    # optimize_and_dump(dr, td_energy, "ccd_td_energy")

    # t2 = (Y2*ham_bar).eval_fermi_vev().simplify()
    # t2 = drutils.define_rk2_rhs(dr, t2)
    # optimize_and_dump(dr, t2, "ccd_t2")

    com = ham | X2
    com_bar = drutils.similarity_transform(com, T2)
    l2 = ((1+L2)*com_bar).eval_fermi_vev().simplify()
    l2 = drutils.define_rk2_rhs(dr, l2)
    optimize_and_dump(dr, l2, "ccd_l2")


if __name__ == '__main__':
    dr = drutils.get_particle_hole_drudge(dummy=True)
    drutils.timer.vocal = True

    run(dr)