import drudge_utils as drutils
import gristmill_utils as grutils

def define_rhs(dr, term, rank):
    if rank == 0:
        equation = drutils.define_rk0_rhs(dr, term)
    elif rank == 1:
        equation = drutils.define_rk1_rhs(dr, term)
    elif rank == 2:
        equation = drutils.define_rk2_rhs(dr, term)
    else:
        print("Uknown rank here, how manu free indicies do you have?")
        exit()

    return equation

def load(dr, basename):
    equations = {}
    names = ["t1", "t2"]

    for name in names:
        equation = drutils.load_from_pickle(dr, basename + f"_{name}")
        equations[name] = equation

    return equations

@drutils.timeme
def get_t_equation(dr, ham_bar, Y, L, filename=None):
    term = (Y * L * ham_bar).eval_fermi_vev().simplify()
    rank = len(Y.terms[0].args[2]) // 2
    equation = define_rhs(dr, term, rank)

    if filename is not None: drutils.save_to_pickle(equation, filename)

    return equation

@drutils.timeme
def get_l_equation(dr, H, X, T, deex, filename=None):
    comm = H | X
    comm_bar = drutils.similarity_transform(comm, T)
    deex = deex.simplify()
    term = (deex * comm_bar).eval_fermi_vev().simplify()
    rank = len(X.terms[0].args[2]) // 2
    equation = define_rhs(dr, term, rank)

    if filename is not None: drutils.save_to_pickle(equation, filename)

    return equation

def calculate_expressions(dr, basename):
    T1, L1 = drutils.get_clusters_1(dr)
    T2, L2 = drutils.get_clusters_2(dr)
    T, L = T1+T2, L1+L2

    ham = dr.ham
    
    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y1 = drutils.get_Y(dr, 1, (i,), (a,))
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))
    X1 = drutils.get_X(dr, 1, (i,), (a,))
    X2 = drutils.get_X(dr, 2, (i, j), (a, b))


    equations = {}

    # ham_bar = drutils.similarity_transform(ham, T)
    # # T1 QCCSD addition
    # t1_equation = get_t_equation(dr, ham_bar, Y1, L, basename + "_t1")
    # equations["t1"] = t1_equation
    # grutils.einsum_raw(dr, basename + "_t1", t1_equation)
    # drutils.timer.tock("QCCSD t1 done")

    # # T2 QCCSD addition
    # t2_equation = get_t_equation(dr, ham_bar, Y2, L, basename + "_t2")
    # equations["t2"] = t2_equation
    # grutils.einsum_raw(dr, basename + "_t2", t2_equation)
    # drutils.timer.tock("QCCSD t2 done")

    # L1 QCCSD addition
    # l1_equation = get_l_equation(dr, ham, X1, T, L*L/2, basename + "_l1")
    # equations["l1"] = l1_equation
    # grutils.einsum_raw(dr, basename + "_l1", l1_equation)
    # drutils.timer.tock("QCCSD l1 done")

    # L2 QCCSD addition
    names = ["11", "12", "22"]
    deexes = [L1*L1/2, L1*L2, L2*L2/2]

    for name, deex in zip(names, deexes):
        l2_equation_additon = get_l_equation(dr, ham, X2, T, deex, basename + f"_l2_{name}")
        equations[f"l2_{name}"] = l2_equation_additon
        grutils.einsum_raw(dr, basename + f"_l2_{name}", l2_equation_additon)
        drutils.timer.tock(f"QCCSD l2 done, {name} term")

    return equations

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
def L_densities(dr):
    # One body
    # o_dums, v_dums = drutils.get_indicies(dr, num=2)
    # blocks, block_names = drutils.get_ob_density_blocks(dr, o_dums, v_dums)
    # rho = _run_blocks(dr, blocks, block_names)
    
    # rho_eqs = drutils.define_ob_density_blocks(dr, rho, block_names, o_dums, v_dums)
    # drutils.save_to_pickle(rho_eqs, "qccsd_1b_density")
    # drutils.save_html(dr, f"qccsd_1b_density", rho_eqs, block_names)
    # grutils.einsum_raw(dr, "qccsd_1b_density", rho_eqs)

    # rho_eqs, block_names = drop_zeros(rho_eqs, block_names)

    # if rho_eqs != [0]:
    #     for rho_eq, name in zip(rho_eqs, block_names):
    #         rho_eval_seq = grutils.optimize_equations(dr, rho_eq)
    #         drutils.save_html(dr, f"qccsd_1b_density_opti_{name}", rho_eval_seq)
    #         grutils.einsum_raw(dr, f"qccsd_1b_density_opti_{name}", rho_eval_seq)
    #         drutils.save_to_pickle(rho_eval_seq, f"qccsd_1b_density_{name}")

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

def optimize_expressions(dr, equations):
    for name, eq in equations.items():
        eval_seq = grutils.optimize_equations(dr, eq)
        drutils.timer.tock(f"QCCSD {name} optimization done")

        drutils.save_to_pickle(eval_seq, basename + "_opti_" + name)
        drutils.save_html(dr, basename + "_opti_" + name, eval_seq)
        grutils.einsum_raw(dr, basename + "_opti_" + name, eval_seq)

if __name__ == "__main__":
    dr = drutils.get_particle_hole_drudge()
    drutils.timer.vocal = True
    
    basename = "qccsd"

    # equations = calculate_expressions(dr, basename)
    # equations = load(dr, basename)
    # optimize_expressions(dr, equations)

    L_densities(dr)