import drudge_utils as drutils
import gristmill_utils as grutils
import pathlib as pl

from sympy import Rational
from drudge import Stopwatch

def load(dr, basename):
    equations = {}
    names = ["t2", "l2", "energy_td_part"]

    for name in names:
        equation = drutils.load_from_pickle(dr, basename + f"_{name}")

        equations[name] = equation

    return equations

def run(dr, basename):
    ham = dr.ham  
    equations = {}

    T2, L2 = drutils.get_restricted_clusters_2(dr)

    T = (T2).simplify()
    L = (L2).simplify()

    _, X2 = drutils.get_restricted_X(dr)
    _, Y2 = drutils.get_restricted_Y(dr)

    ham_bar = drutils.similarity_transform(ham, T)
    energy = (ham_bar).eval_fermi_vev().simplify()
    energy = drutils.define_rk0_rhs(dr, energy)
    energy_td_part = (L*ham_bar).eval_fermi_vev().simplify()
    energy_td_part = drutils.define_rk0_rhs(dr, energy_td_part)
    equations["energy_td_part"] = energy_td_part
    drutils.save_html(dr, basename + f"_energy_td_part", energy_td_part)
    grutils.einsum_raw(dr, basename + "_energy_td_part", energy_td_part)
    drutils.save_to_pickle(energy_td_part, basename + "_energy_td_part")
    drutils.timer.tock("Done RQCCD energy")

    t2_eq = (Y2*ham_bar).eval_fermi_vev().simplify()
    t2_eq = drutils.define_rk2_rhs_restrictd(dr, t2_eq)
    equations["t2"] = t2_eq
    drutils.save_html(dr, basename + f"_t2", t2_eq)
    grutils.einsum_raw(dr, basename + "_t2", t2_eq)
    drutils.save_to_pickle(t2_eq, basename + "_t2")
    drutils.timer.tock("Done RQCCD t2")

    com = ham | X2
    com_bar = drutils.similarity_transform(com, T)
    l2_eq = ((1+L)*com_bar).eval_fermi_vev().simplify()
    l2_eq = drutils.define_rk2_rhs_restrictd(dr, l2_eq)
    equations["l2"] = l2_eq
    drutils.save_html(dr, basename + f"_l2", l2_eq)
    grutils.einsum_raw(dr, basename + "_l2", l2_eq)
    drutils.save_to_pickle(l2_eq, basename + "_l2")
    drutils.timer.tock("Done RQCCD l2")

    return equations

def optimize_expressions(dr, equations, basename):
    for name, eq in equations.items():
        eval_seq = grutils.optimize_equations(dr, eq)
        drutils.timer.tock(f"RCCD {name} optimization done")

        drutils.save_to_pickle(eval_seq, basename + "_opti_" + name)
        drutils.save_html(dr, basename + "_opti_" + name, eval_seq)
        grutils.einsum_raw(dr, basename + "_opti_" + name, eval_seq)

def _run_blocks(dr, blocks, block_names):
    # Get clusters
    T2, L2 = drutils.get_restricted_clusters_2(dr)
    rho = [None] * len(blocks)

    # Loop over each block, sim trans and resolve 1 and L term
    for i, block in enumerate(blocks):
        block_sim = drutils.similarity_transform(block, T2)

        A = block_sim.eval_fermi_vev().simplify()
        B = (L2 * block_sim).eval_fermi_vev().simplify()
        rho[i] = (A + B).simplify()
        drutils.timer.tock(f"Density {block_names[i]}", rho[i])

    return rho

def L_densities(dr, basename):
    # One body
    o_dums, v_dums = drutils.get_indicies(dr, num=2)
    blocks, block_names = drutils.get_ob_density_blocks_restricted(dr, o_dums, v_dums)
    rho = _run_blocks(dr, blocks, block_names)
    drutils.save_html(dr, basename + "_rho_ob", rho, block_names)

    

if __name__ == "__main__":
    drutils.timer.vocal = True
    dr = drutils.get_restricted_particle_hole_drudge()
    
    basename = "rccd"

    # equations = run(dr, basename)
    # equations = load(dr, basename)

    # optimize_expressions(dr, equations, basename)

    L_densities(dr, basename)