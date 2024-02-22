import drudge_utils as drutils
import gristmill_utils as grutils

@drutils.timeme
def energy_lambda_contribution(dr):
    # Get T2 operators and sim transform ham
    T2, L2 = drutils.get_clusters_2(dr)
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T2)

    energy_eq = (L2*ham_bar).eval_fermi_vev().simplify()
    energy_eq = drutils.define_rk0_rhs(dr, energy_eq)
    drutils.timer.tock("energy equation")

    eval_seq = grutils.optimize_equations(dr, energy_eq)
    
    drutils.save_html(dr, "ccd_energy_addition", [energy_eq], ["E addition"])
    grutils.einsum_raw(dr, "ccd_energy_addition", [energy_eq])

    drutils.save_html(dr, "ccd_energy_addition_optimized", eval_seq)
    grutils.einsum_raw(dr, "ccd_energy_addition_optimized", eval_seq)

@drutils.timeme
def T_equations(dr):
    # Get T2 operators and sim transform ham
    T2, _ = drutils.get_clusters_2(dr)
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T2)

    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y1 = drutils.get_Y(dr, 1, (i,), (a,))
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))

    # Calculate energy and t2 equations
    energy_eq = ham_bar.eval_fermi_vev().simplify()
    drutils.timer.tock("T2 energy equation")
    
    amplitude_t1_eq = (Y1 * ham_bar).eval_fermi_vev().simplify()
    t1 = drutils.define_rk1_rhs(dr, amplitude_t1_eq)
    eval_seq = grutils.optimize_equations(dr, t1)
    drutils.save_html(dr, "ccsd_t1transform", t1)
    drutils.save_html(dr, "ccsd_t1transform_optimized", eval_seq)
    grutils.einsum_raw(dr, "ccsd_t1transform", t1)
    grutils.einsum_raw(dr, "ccsd_t1transform_optimized", eval_seq)
    drutils.timer.tock("T1 amplitude equation for T1-transformed CCSD done")

    amplitude_t2_eq = (Y2 * ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("T2 amplitude equation")

    drutils.save_html(dr, "ccd_energy_and_t2", [energy_eq, amplitude_t2_eq], ["E", "t2 = 0"])

    e, t2 = drutils.define_rk0_rhs(dr, energy_eq), drutils.define_rk2_rhs(dr, amplitude_t2_eq)

    grutils.einsum_raw(dr, "ccd_energy_t2", [e, t2])
    eval_seq = grutils.optimize_equations(dr, t2)
    grutils.einsum_raw(dr, "ccd_t2_optimized", eval_seq)


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

    amplitude_l2_eq = (A + B).simplify()
    drutils.save_html(dr, "ccd_l2", [amplitude_l2_eq], ["l2 = 0"])

    l2 = drutils.define_rk2_rhs(dr, amplitude_l2_eq)
    grutils.einsum_raw(dr, "ccd_l2", l2)
    eval_seq = grutils.optimize_equations(dr, l2)
    grutils.einsum_raw(dr, "ccd_l2_optimized", eval_seq)


def _run_blocks(dr, blocks, block_names):
    # Get clusters
    T2, L2 = drutils.get_clusters_2(dr)
    rho = [None] * len(blocks)

    # Loop over each block, sim trans and resolve 1 and L term
    for i, block in enumerate(blocks):
        block_sim = drutils.similarity_transform(block, T2)

        A = block_sim.eval_fermi_vev().simplify()
        B = (L2 * block_sim).eval_fermi_vev().simplify()
        rho[i] = (A + B).simplify()
        drutils.timer.tock(f"Density {block_names[i]}")

    return rho


@drutils.timeme
def L_densities(dr):
    # One body
    # o_dums, v_dums = drutils.get_indicies(dr, num=2)
    # blocks, block_names = drutils.get_ob_density_blocks(dr, o_dums, v_dums)
    # rho = _run_blocks(dr, blocks, block_names)
    # drutils.save_html(dr, "ccd_ob_density", rho, block_names)

    # Two body
    o_dums, v_dums = drutils.get_indicies(dr, num=4)
    blocks, block_names = drutils.get_tb_density_blocks(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks, block_names)
    drutils.save_html(dr, "ccd_2b_density", rho, block_names)

    rho_eqs = drutils.define_tb_density_blocks(dr, rho, block_names, o_dums, v_dums)
    grutils.einsum_raw(dr, "ccd_l_2b_density", rho_eqs)


def main():
    dr = drutils.get_particle_hole_drudge()

    drutils.timer.vocal = True
    # energy_lambda_contribution(dr)
    T_equations(dr)
    # L_equations(dr)
    # L_densities(dr)


if __name__ == "__main__":
    main()
