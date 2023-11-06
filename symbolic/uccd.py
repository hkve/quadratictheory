import drudge_utils as drutils
import gristmill_utils as grutils

from IPython import embed


@drutils.timeme
def T_equations(dr):
    T2, _ = drutils.get_clusters_2(dr)
    T = T2.simplify()
    T_dag = dr.dagger(T2).simplify()
    H = dr.ham

    # energy_C1 = (H*T + T_dag*H)
    # energy_C2 = (-T_dag*T*H/2 - H*T_dag*T/2 + T_dag*H*T)

    # energy_operators = (energy_C1 + energy_C2).simplify()
    # drutils.timer.tock("Constructed energy operators")

    # energy_eq = (energy_operators).eval_fermi_vev().simplify()
    # drutils.timer.tock("UCCD2 energy equation")

    amplitude_C0 = H
    amplitude_C1 = H * T - T * H
    amplitude_C2 = (
        H * T * T / 2
        - H * T_dag * T / 2
        - T_dag * T * H / 2
        - T * T_dag * H / 2
        - T * H * T
        + T_dag * H * T
        + T * H * T_dag
        - T_dag * H * T_dag
    ).simplify()

    amplitude_operators = (amplitude_C0 + amplitude_C1 + amplitude_C2).simplify()
    drutils.timer.tock("Constructed amplitude operators")

    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))
    amplitude_t2_eq = (Y2 * amplitude_operators).eval_fermi_vev().simplify()
    drutils.timer.tock("UCCD2 amplitude equation")

    # drutils.save_html(dr, "uccd_e_t2_expanded", [energy_eq, amplitude_t2_eq], ["E", "t2  = 0"])

    # e = drutils.define_rk0_rhs(dr, energy_eq)
    t2 = drutils.define_rk2_rhs(dr, amplitude_t2_eq)

    # grutils.einsum_raw(dr, "uccd2_e_t2", [e, t2])
    grutils.einsum_raw(dr, "uccd2_t2", [t2])

    eval_seq = grutils.optimize_equations(dr, [t2])
    grutils.einsum_raw(dr, "uccd2_e_t2_optimized", eval_seq)


if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()

    drutils.timer.vocal = True
    T_equations(dr)
