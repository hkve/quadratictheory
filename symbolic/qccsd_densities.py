import drudge_utils as drutils
import gristmill_utils as grutils

def _run_blocks(dr, blocks, block_names):
    # Get clusters
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T = (T1+T2).simplify()
    L = (L1+L2).simplify()
    rho = [None] * len(blocks)

    # Loop over each block, sim trans and resolve 1 and L term
    for i, block in enumerate(blocks):
        block_sim = drutils.similarity_transform(block, T)

        rho_block = (L*L/2 * block_sim).eval_fermi_vev().simplify()
        rho[i] = rho_block
        drutils.timer.tock(f"Density {block_names[i]}. = 0? {rho_block == 0}")
    
    return rho

def drop_zeros(eqs, names=None):
    new_eqs, new_names = [], []

    for i, eq in enumerate(eqs):
        if eq.rhs != 0:
            new_eqs.append(eq)
            if names != None:
                new_names.append(names[i])

    return new_eqs, new_names

@drutils.timeme
def one_body(dr):
    # One body
    o_dums, v_dums = drutils.get_indicies(dr, num=2)
    blocks, block_names = drutils.get_ob_density_blocks(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks, block_names)
    
    rho_eqs = drutils.define_ob_density_blocks(dr, rho, block_names, o_dums, v_dums)
    drutils.save_to_pickle(rho_eqs, "qccsd_1b_density")
    drutils.save_html(dr, f"qccsd_1b_density", rho_eqs, block_names)
    grutils.einsum_raw(dr, "qccsd_1b_density", rho_eqs)

    rho_eqs, block_names = drop_zeros(rho_eqs, block_names)

    if rho_eqs != [0]:
        for rho_eq, name in zip(rho_eqs, block_names):
            rho_eval_seq = grutils.optimize_equations(dr, rho_eq)
            drutils.save_html(dr, f"qccsd_1b_density_opti_{name}", rho_eval_seq)
            grutils.einsum_raw(dr, f"qccsd_1b_density_opti_{name}", rho_eval_seq)
            drutils.save_to_pickle(rho_eval_seq, f"qccsd_1b_density_{name}")


@drutils.timeme
def two_body(dr):
    o_dums, v_dums = drutils.get_indicies(dr, num=4)
    blocks, block_names = drutils.get_tb_density_blocks(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks, block_names)
    
    rho_eqs = drutils.define_tb_density_blocks(dr, rho, block_names, o_dums, v_dums)
    drutils.save_html(dr, "qccsd_2b_density", rho_eqs, block_names)
    grutils.einsum_raw(dr, "qccsd_2b_density", rho_eqs)
    drutils.save_to_pickle(rho_eqs, "qccsd_2b_density")
    # rho_eqs = drutils.load_from_pickle(dr, "qccd_2b_density")

    rho_eqs, block_names = drop_zeros(rho_eqs, block_names)

    if rho_eqs != [0]:
        for rho_eq, name in zip(rho_eqs, block_names):
            drutils.timer.tock(f"QCCSD starting two-body {name} block")
            rho_eval_seq = grutils.optimize_equations(dr, rho_eq)
            drutils.save_html(dr, f"qccsd_2b_density_opti_{name}", rho_eval_seq)
            grutils.einsum_raw(dr, f"qccsd_2b_density_opti_{name}", rho_eval_seq)
            drutils.save_to_pickle(rho_eval_seq, f"qccsd_2b_density_{name}")

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()
    drutils.timer.vocal = True
    
    one_body(dr)
    two_body(dr)