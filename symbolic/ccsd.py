import drudge_utils as drutils
import gristmill_utils as grutils


@drutils.timeme
def energy_lambda_contribution(dr):
    # Get T2 operators and sim transform ham
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    
    T = (T1+T2).simplify()
    L = (L1+L2).simplify()
    
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T)

    energy_eq = (L*ham_bar).eval_fermi_vev().simplify()
    energy_eq = drutils.define_rk0_rhs(dr, energy_eq)
    drutils.timer.tock("energy equation")

    drutils.save_to_pickle(energy_eq, "ccsd_lambda_energy")
    # eval_seq = grutils.optimize_equations(dr, energy_eq)
    
    # drutils.save_html(dr, "ccsd_energy_addition", [energy_eq], ["E addition"])
    # grutils.einsum_raw(dr, "ccsd_energy_addition", [energy_eq])

    # drutils.save_html(dr, "ccsd_energy_addition_optimized", eval_seq)
    # grutils.einsum_raw(dr, "ccsd_energy_addition_optimized", eval_seq)

@drutils.timeme
def T_equations(dr):
    # Get T2 operators and sim transform ham
    T1, _ = drutils.get_clusters_1(dr)
    T2, _ = drutils.get_clusters_2(dr)
    T = (T1 + T2).simplify()
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T)

    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y1 = drutils.get_Y(dr, 1, (i,), (a,))
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))

    # Calculate energy and t2 equations
    energy_eq = ham_bar.eval_fermi_vev().simplify()
    drutils.timer.tock("CCSD energy equation")
    amplitude_t1_eq = (Y1 * ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("T1 amplitude equation")
    amplitude_t2_eq = (Y2 * ham_bar).eval_fermi_vev().simplify()
    drutils.timer.tock("T2 amplitude equation")

    # drutils.save_html(
    #     dr,
    #     "ccsd_energy_and_t",
    #     [energy_eq, amplitude_t1_eq, amplitude_t2_eq],
    #     ["E", "t1 = 0", "t2 = 0"],
    # )

    e = drutils.define_rk0_rhs(dr, energy_eq)
    t1 = drutils.define_rk1_rhs(dr, amplitude_t1_eq)
    t2 = drutils.define_rk2_rhs(dr, amplitude_t2_eq)

    drutils.save_to_pickle(e, "ccsd_energy_raw")
    drutils.save_to_pickle(t1, "ccsd_t1_raw")
    drutils.save_to_pickle(t2, "ccsd_t2_raw")

    # grutils.einsum_raw(dr, "ccsd_t", [t1, t2])
    # grutils.einsum_raw(dr, "ccsd_t1", [t1])
    # eval_seq = grutils.optimize_equations(dr, [t1, t2])
    # eval_seq_t1 = grutils.optimize_equations(dr, t1)
    # grutils.einsum_raw(dr, "ccsd_t1_optimized", eval_seq_t1)
    # grutils.einsum_raw(dr, "ccsd_t_optimized", eval_seq)


@drutils.timeme
def L_equations(dr):
    # Get T2 and L2 operator, excitation and sim transform commutator
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T = (T1 + T2).simplify()
    L = (L1 + L2).simplify()

    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    X1 = drutils.get_X(dr, 1, (i,), (a,))
    X2 = drutils.get_X(dr, 2, (i, j), (a, b))

    comm = dr.ham | X1
    comm_sim = drutils.similarity_transform(comm, T)

    # Calculate A(t) and B(t, l) term
    A = comm_sim.eval_fermi_vev().simplify()
    drutils.timer.tock("L1, A term")
    B = (L * comm_sim).eval_fermi_vev().simplify()
    drutils.timer.tock("L1 B term")

    amplitude_l1_eq = (A + B).simplify()

    comm = dr.ham | X2
    comm_sim = drutils.similarity_transform(comm, T)

    # Calculate A(t) and B(t, l) term
    A = comm_sim.eval_fermi_vev().simplify()
    drutils.timer.tock("L2, A term")
    B = (L * comm_sim).eval_fermi_vev().simplify()
    drutils.timer.tock("L2 B term")

    amplitude_l2_eq = (A + B).simplify()

    # drutils.save_html(dr, "ccsd_l", [amplitude_l1_eq, amplitude_l2_eq], ["l1 = 0", "l2 = 0"])

    l1 = drutils.define_rk1_rhs(dr, amplitude_l1_eq).simplify()
    l2 = drutils.define_rk2_rhs(dr, amplitude_l2_eq).simplify()

    drutils.save_to_pickle(l1, "ccsd_l1_raw")
    drutils.save_to_pickle(l2, "ccsd_l2_raw")

    # grutils.einsum_raw(dr, "ccsd_l", [l1, l2])
    # eval_seq = grutils.optimize_equations(dr, [l1, l2])
    # grutils.einsum_raw(dr, "ccsd_l_optimized", eval_seq)


@drutils.timeme
def L_equations_T1_trans(dr):
    # Get T2 and L2 operator, excitation and sim transform commutator
    _, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T = T2.simplify()
    L = (L1 + L2).simplify()

    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    X1 = drutils.get_X(dr, 1, (i,), (a,))
    X2 = drutils.get_X(dr, 2, (i, j), (a, b))

    comm = dr.ham | X1
    comm_sim = drutils.similarity_transform(comm, T)

    # Calculate A(t) and B(t, l) term
    A = comm_sim.eval_fermi_vev().simplify()
    drutils.timer.tock("L1, A term")
    B = (L * comm_sim).eval_fermi_vev().simplify()
    drutils.timer.tock("L1 B term")

    amplitude_l1_eq = (A + B).simplify()

    comm = dr.ham | X2
    comm_sim = drutils.similarity_transform(comm, T)

    # Calculate A(t) and B(t, l) term
    A = comm_sim.eval_fermi_vev().simplify()
    drutils.timer.tock("L2, A term")
    B = (L * comm_sim).eval_fermi_vev().simplify()
    drutils.timer.tock("L2 B term")

    amplitude_l2_eq = (A + B).simplify()

    drutils.save_html(dr, "ccsd_l_t1transformed", [amplitude_l1_eq, amplitude_l2_eq], ["l1 = 0", "l2 = 0"])

    l1 = drutils.define_rk1_rhs(dr, amplitude_l1_eq).simplify()
    l2 = drutils.define_rk2_rhs(dr, amplitude_l2_eq).simplify()

    grutils.einsum_raw(dr, "ccsd_l_t1transformed", [l1, l2])
    eval_seq = grutils.optimize_equations(dr, [l1, l2])
    grutils.einsum_raw(dr, "ccsd_l_t1transformed_optimized", eval_seq)

def _run_blocks(dr, blocks, block_names, T, L):
    # Get clusters
    rho = [None] * len(blocks)

    # Loop over each block, sim trans and resolve 1 and L term
    for i, block in enumerate(blocks):
        block_sim = drutils.similarity_transform(block, T)

        A = block_sim.eval_fermi_vev().simplify()
        B = (L * block_sim).eval_fermi_vev().simplify()
        rho[i] = (A + B).simplify()
        drutils.timer.tock(f"Density {block_names[i]}")

    return rho

@drutils.timeme
def L_densities(dr):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T = (T1 + T2).simplify()
    L = (L1 + L2).simplify()

    # One body
    # o_dums, v_dums = drutils.get_indicies(dr, num=2)
    # blocks, block_names = drutils.get_ob_density_blocks(dr, o_dums, v_dums)
    # rho = _run_blocks(dr, blocks, block_names, T, L)
    # drutils.save_html(dr, "ccsd_1b_density", rho, block_names)

    # Two body
    o_dums, v_dums = drutils.get_indicies(dr, num=4)
    blocks, block_names = drutils.get_tb_density_blocks(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks, block_names, T, L)
    drutils.save_html(dr, "ccsd_2b_density", rho, block_names)
    rho_eqs = drutils.define_tb_density_blocks(dr, rho, block_names, o_dums, v_dums)
    grutils.einsum_raw(dr, "ccsd_l_2b_density", rho_eqs)

@drutils.timeme
def energy_lambda_contribution_T1_trans(dr):
    # Get T2 operators and sim transform ham
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    
    T = (T2).simplify()
    L = (L1+L2).simplify()
    
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T)

    energy_eq = (L*ham_bar).eval_fermi_vev().simplify()
    energy_eq = drutils.define_rk0_rhs(dr, energy_eq)

    drutils.timer.tock("energy equation")

    eval_seq = grutils.optimize_equations(dr, energy_eq)
    
    drutils.save_html(dr, "ccsd_energy_addition_t1transformed", [energy_eq], ["E addition"])
    grutils.einsum_raw(dr, "ccsd_energy_addition_t1transformed", [energy_eq])

    drutils.save_html(dr, "ccsd_energy_addition_optimized_t1transformed", eval_seq)
    grutils.einsum_raw(dr, "ccsd_energy_addition_optimized_t1transformed", eval_seq)


def main():
    dr = drutils.get_particle_hole_drudge(dummy=True)

    drutils.timer.vocal = True
    # T_equations(dr)
    # L_equations(dr)
    # L_densities(dr)
    # energy_lambda_contribution(dr)

    # L_equations_T1_trans(dr)
    # energy_lambda_contribution_T1_trans(dr)
    num_terms = [0, 6, 4, 6, 4, 6]
    filenames = [
        "ccsd_energy_raw",
        "ccsd_t1_raw",
        "ccsd_t2_raw",
        "ccsd_l1_raw",
        "ccsd_l2_raw",
        "ccsd_lambda_energy",
    ]

    import latex
    for i, filename in enumerate(filenames):
        eq = latex.texify(dr, filename, num_terms=num_terms[i])
        
        print(f"This is {filename}\n\n")
        print(eq)
        print("\n\n")
    

if __name__ == "__main__":
    main()
