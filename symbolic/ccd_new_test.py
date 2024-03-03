import drudge_utils as drutils
import gristmill_utils as grutils

def optimize_and_dump(dr, equation, name):
    drutils.save_html(dr, name, equation)
    drutils.save_to_pickle(equation, name)
    grutils.einsum_raw(dr, name, equation)

    eval_seq = grutils.optimize_equations(dr, equation)

    drutils.save_html(dr, f"{name}_optimized", eval_seq)
    drutils.save_to_pickle(eval_seq, f"{name}_optimized")
    grutils.einsum_raw(dr, f"{name}_optimized", eval_seq)

@drutils.timeme
def run(dr):
    T2, L2 = drutils.get_clusters_2(dr)
    ham = dr.ham
    ham_bar = drutils.similarity_transform(ham, T2)

    # Free indicies for de-excitation operator
    (i, j), (a, b) = drutils.get_indicies(dr, num=2)
    Y2 = drutils.get_Y(dr, 2, (i, j), (a, b))

    td_energy = (L2*ham_bar).eval_fermi_vev().simplif()
    optimize_and_dump(dr, td_energy, "ccd_td_energy")

if __name__ == '__main__':
    pass