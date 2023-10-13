import drudge_utils as utils

def T_equations(dr):
    # Get T2 operators and sim transform ham
    T2, _ = utils.get_clusters_2(dr)
    ham = dr.ham
    ham_bar = utils.similarity_transform(ham, T2)

    # Free indicies for de-excitation operator
    (i, j), (a, b) = utils.get_indicies(dr, num=2)
    Y2 = utils.get_Y(dr, 2, (i,j), (a, b))

    # Calculate energy and t2 equations
    energy_eq = ham_bar.eval_fermi_vev().simplify()
    amplitude_t2_eq = (Y2*ham_bar).eval_fermi_vev().simplify()

    utils.save_html(dr, "ccd_energy_and_t2", [energy_eq,amplitude_t2_eq], ["E","t2 = 0"])

def L_equations(dr):
    # Get T2 and L2 operator, excitation and sim transform commutator
    T2, L2 = utils.get_clusters_2(dr)
    (i, j), (a, b) = utils.get_indicies(dr, num=2)
    X2 = utils.get_X(dr, 2, (i,j), (a, b))

    comm = dr.ham | X2
    comm_sim = utils.similarity_transform(comm, T2)

    # Calculate A(t) and B(t, l) term
    A = comm_sim.eval_fermi_vev().simplify()
    B = (L2*comm_sim).eval_fermi_vev().simplify()

    amplitude_l2_eq = (A+B).simplify()
    utils.save_html(dr, "ccd_l2", [amplitude_l2_eq], ["l2 = 0"])

def _run_blocks(dr, blocks):
    # Get clusters
    T2, L2 = utils.get_clusters_2(dr)
    rho = [None]*len(blocks)
    
    # Loop over each block, sim trans and resolve 1 and L term
    for i, block in enumerate(blocks):
        block_sim = utils.similarity_transform(block, T2)

        A = block_sim.eval_fermi_vev().simplify()
        B = (L2*block_sim).eval_fermi_vev().simplify()
        rho[i] = (A+B).simplify()
    
    return rho

def L_densities(dr):
    # Get T2 and L2 operator, excitation and sim transform commutator
    
    # One body
    o_dums, v_dums = utils.get_indicies(dr, num=2)
    blocks, block_names = utils.get_ob_density_blocks(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks)
    utils.save_html(dr, "ccd_ob_density", rho, block_names)

    # Two body
    o_dums, v_dums = utils.get_indicies(dr, num=4)
    blocks, block_names = utils.get_tb_density_blocks(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks)
    utils.save_html(dr, "ccd_tb_density", rho, block_names)

def main():
    dr = utils.get_particle_hole_drudge()

    # T_equations(dr)
    # L_equations(dr)
    L_densities(dr)

if __name__ == '__main__':
    main()