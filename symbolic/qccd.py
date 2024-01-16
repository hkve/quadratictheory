import drudge_utils as drutils
import gristmill_utils as grutils
from IPython import embed

def similarity_transform_order(tensor, clusters):
    stopwatch = None

    curr = tensor
    tensor_bar = tensor

    tensors_bars = {}

    for order in range(0, 4):
        curr = (curr | clusters).simplify() / (order + 1)
        curr.cache()
        tensors_bars[order+1] = curr
        drutils.timer.tock(f"Commutator at order {order+1}")

    return tensors_bars

@drutils.timeme
def run(dr):
    T2, L2 = drutils.get_clusters_2(dr)
    ham = dr.ham
    # ham_bars = similarity_transform_order(ham, T2)
    ham_bar = drutils.similarity_transform(ham, T2)

    t2, l2 = drutils.make_rk2(dr, "t"), drutils.make_rk2(dr, "\lambda")
    (i,j), (a,b) = drutils.get_indicies(dr, num=2)

    energy_eq = ((1 + L2 + L2*L2/2)*ham_bar).eval_fermi_vev().simplify()
    # E0 = (ham).eval_fermi_vev().simplify()
    # E1 = (L2*(ham + ham_bars[1]+ham_bars[2])).eval_fermi_vev().simplify()
    # E2 = (L2*L2*(ham_bars[2] + ham_bars[3])/2).eval_fermi_vev().simplify()

    # energy_eq = (E0+E1+E2).simplify()
    drutils.timer.tock("QCCD energy addition done")

    t2_terms = (4*drutils.diff_rk2_antisym(energy_eq, l2, (i,j), (a,b))).simplify()
    drutils.timer.tock("QCCD t2 done")

    l2_terms = (4*drutils.diff_rk2_antisym(energy_eq, t2, (i,j), (a,b))).simplify()
    drutils.timer.tock("QCCD l2 done")

    e = drutils.define_rk0_rhs(dr, energy_eq)
    grutils.einsum_raw(dr, "qccd_energy_from_lag", [e])
    eval_seq_e = grutils.optimize_equations(dr, e)
    grutils.einsum_raw(dr, "qccd_energy_optimized_from_lag", eval_seq_e)
    drutils.timer.tock("QCCD energy opti done")

    t2_equations = drutils.define_rk2_rhs(dr, t2_terms)
    l2_equations = drutils.define_rk2_rhs(dr, l2_terms)

    grutils.einsum_raw(dr, "qccd_t2_from_lag", [t2_equations])
    eval_seq_t2 = grutils.optimize_equations(dr, t2_equations)
    grutils.einsum_raw(dr, "qccd_t2_optimized_from_lag", eval_seq_t2)
    drutils.timer.tock("QCCD t2 opti done")

    grutils.einsum_raw(dr, "qccd_l2_from_lag", [l2_equations])
    eval_seq_l2 = grutils.optimize_equations(dr, l2_equations)
    grutils.einsum_raw(dr, "qccd_l2_optimized_from_lag", eval_seq_l2)
    drutils.timer.tock("QCCD l2 opti done")

    drutils.save_html(dr, "qccd_from_lag", [energy_eq, t2_equations, l2_equations], ["Energy", "t2=0", "l2=0"])

    drutils.save_html(dr, "qccd_opti_lagrangian", eval_seq_e)
    drutils.save_html(dr, "qccd_opti_t2", eval_seq_t2)
    drutils.save_html(dr, "qccd_opti_l2", eval_seq_l2)

def _run_blocks(dr, blocks, block_names):
    # Get clusters
    T2, L2 = drutils.get_clusters_2(dr)
    T = T2.simplify()
    L = L2.simplify()
    rho = [None] * len(blocks)

    # Loop over each block, sim trans and resolve 1 and L term
    for i, block in enumerate(blocks):
        block_sim = drutils.similarity_transform(block, T)

        rho_block = (L*L/2 * block_sim).eval_fermi_vev().simplify()
        rho[i] = rho_block
        drutils.timer.tock(f"Density {block_names[i]}")
    
    return rho

@drutils.timeme
def L_densities(dr):
    # One body
    # o_dums, v_dums = drutils.get_indicies(dr, num=2)
    # blocks, block_names = drutils.get_ob_density_blocks(dr, o_dums, v_dums)
    # rho = _run_blocks(dr, blocks, block_names)
    # drutils.save_html(dr, "qccd_1b_density", rho, block_names)
    
    # rho_eqs = drutils.define_ob_density_blocks(dr, rho, block_names, o_dums, v_dums)
    # drutils.save_to_pickle(rho_eqs, "qccd_1b_density")
    # grutils.einsum_raw(dr, "qccd_1b_density", rho_eqs)

    # rho_eval_seq = grutils.optimize_equations(dr, rho_eqs)
    # drutils.save_html(dr, "qccd_1b_density_opti", rho, block_names)
    # grutils.einsum_raw(dr, "qccd_1b_density_opti", rho_eqs)

    # Two body
    o_dums, v_dums = drutils.get_indicies(dr, num=4)
    blocks, block_names = drutils.get_tb_density_blocks(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks, block_names)
    
    rho_eqs = drutils.define_tb_density_blocks(dr, rho, block_names, o_dums, v_dums)
    drutils.save_html(dr, "qccd_2b_density", rho_eqs, block_names)
    grutils.einsum_raw(dr, "qccd_2b_density", rho_eqs)
    drutils.save_to_pickle(rho_eqs, "qccd_2b_density")
    # rho_eqs = drutils.load_from_pickle(dr, "qccd_2b_density")

    block_names = [
            block_name for block_name, rho_eq in zip(block_names, rho_eqs)
            if rho_eq.rhs != 0
        ]
    rho_eqs = [
            rho_eq for rho_eq in rho_eqs
            if rho_eq.rhs != 0
        ]

    rho_eqs_opti = []

    for rho in rho_eqs:
        rho_eqs_opti.append(
            grutils.optimize_equations(dr, rho)
        )


    for rho_eq_opti, block_name in zip(rho_eqs_opti, block_names):
        drutils.save_html(dr, f"qccd_2b_density_opti_{block_name}", rho_eq_opti)
        grutils.einsum_raw(dr, f"qccd_2b_density_opti_{block_name}", rho_eq_opti)
        drutils.save_to_pickle(rho_eq_opti, f"qccd_2b_density_opti_{block_name}")

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()

    drutils.timer.vocal = True
    # run(dr)
    L_densities(dr)