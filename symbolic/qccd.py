import drudge_utils as drutils
import gristmill_utils as grutils

@drutils.timeme
def E_equations(dr):
    T2, L2 = drutils.get_clusters_2(dr)
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T2)

    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))

    energy_ccd = (ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("CCD energy done")
    energy_qccd_addition = (L2*L2*ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("QCCD energy addition done")

    energy_eq = (energy_ccd - energy_qccd_addition/2).simplify()

    drutils.save_html(dr, "qccd_energy", [energy_eq], ["Energy"])

    e = drutils.define_rk0_rhs(dr, energy_eq)
    grutils.einsum_raw(dr, "qccd_energy", [e])
    eval_seq = grutils.optimize_equations(dr, e)
    grutils.einsum_raw(dr, "qccd_energy_optimized", eval_seq)

@drutils.timeme
def T_equations(dr):
    # Get T2 operators and sim transform ham
    T2, L2 = drutils.get_clusters_2(dr)
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T2)

    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))

    # Calculate energy and t2 equations
    # energy_eq = ham_bar.eval_fermi_vev().simplify()
    # drutils.timer.tock("T2 energy equation")
    amplitude_t2_eq_linear = (Y2 * ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("T2 amplitude linear term")
    amplitude_t2_eq_quad = (Y2 * L2 * ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("T2 amplitude quad term")

    amplitude_t2_eq = (amplitude_t2_eq_linear + amplitude_t2_eq_quad).simplify()
    drutils.save_html(dr, "qccd_energy_and_t2", [amplitude_t2_eq], ["t2 = 0"])

    # e = drutils.define_rk0_rhs(dr, energy_eq)
    t2 = drutils.define_rk2_rhs(dr, amplitude_t2_eq)

    grutils.einsum_raw(dr, "qccd_energy_t2", [t2])
    eval_seq = grutils.optimize_equations(dr, t2)
    grutils.einsum_raw(dr, "qccd_t2_optimized", eval_seq)


@drutils.timeme
def L_equations(dr):
    # Get T2 and L2 operator, excitation and sim transform commutator
    T2, L2 = drutils.get_clusters_2(dr)
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    X2 = drutils.get_X(dr, 2, (i, j), (a, b))

    comm = dr.ham | X2
    comm_sim = drutils.similarity_transform(comm, T2)

    # Calculate A(t) and B(t, l) term
    A = comm_sim.eval_fermi_vev().simplify()
    drutils.timer.tock("L2, A term")
    B = (L2 * comm_sim).eval_fermi_vev().simplify()
    drutils.timer.tock("L2 B term")
    C = (L2 * L2 * comm_sim / 2).eval_fermi_vev().simplify()
    drutils.timer.tock("L2 C term")

    amplitude_l2_eq = (A + B + C).simplify()
    drutils.save_html(dr, "qccd_l2", [amplitude_l2_eq], ["l2 = 0"])

    l2 = drutils.define_rk2_rhs(dr, amplitude_l2_eq)
    grutils.einsum_raw(dr, "qccd_l2", l2)
    eval_seq = grutils.optimize_equations(dr, l2)
    grutils.einsum_raw(dr, "qccd_l2_optimized", eval_seq)

def main():
    dr = drutils.get_particle_hole_drudge()

    drutils.timer.vocal = True
    E_equations(dr)
    # T_equations(dr)
    # L_equations(dr)


if __name__ == "__main__":
    main()
