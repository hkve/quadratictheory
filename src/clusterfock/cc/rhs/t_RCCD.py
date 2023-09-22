import numpy as np


def amplitudes_ccd_restricted(t2, u, f_hh_o, f_pp_o, v, o):
    res = np.zeros_like(t2)
    # Here we have a long permutation term, so we collect all sums and then perform the permutation

    # Fock terms, single sum
    res += np.einsum("bc,acij->abij", f_pp_o, t2, optimize=True)
    res -= np.einsum("kj,abik->abij", f_hh_o, t2, optimize=True)

    # virvir and occocc sums
    res += 0.5 * np.einsum("abcd,cdij->abij", u[v, v, v, v], t2, optimize=True)
    res += 0.5 * np.einsum("klij,abkl->abij", u[o, o, o, o], t2, optimize=True)

    # v o double sum
    res += 2 * np.einsum("kbcj,acik->abij", u[o, v, v, o], t2, optimize=True)
    res -= np.einsum("kbcj,acki->abij", u[o, v, v, o], t2, optimize=True)
    res -= np.einsum("kbic,ackj->abij", u[o, v, o, v], t2, optimize=True)
    res -= np.einsum("kbjc,acik->abij", u[o, v, o, v], t2, optimize=True)

    # vvoo sums
    res += 0.5 * np.einsum("klcd,cdij,abkl->abij", u[o, o, v, v], t2, t2, optimize=True)
    res += 2 * np.einsum("klcd,acik,dblj->abij", u[o, o, v, v], t2, t2, optimize=True)
    res -= 2 * np.einsum("klcd,acik,dbjl->abij", u[o, o, v, v], t2, t2, optimize=True)
    res += 0.5 * np.einsum("klcd,caik,bdlj->abij", u[o, o, v, v], t2, t2, optimize=True)
    res -= np.einsum("klcd,adik,cblj->abij", u[o, o, v, v], t2, t2, optimize=True)
    res += np.einsum("klcd,adki,cblj->abij", u[o, o, v, v], t2, t2, optimize=True)

    res += 0.5 * np.einsum("klcd,cbil,adkj->abij", u[o, o, v, v], t2, t2, optimize=True)
    res -= 2 * np.einsum("klcd,cdki,ablj->abij", u[o, o, v, v], t2, t2, optimize=True)
    res += np.einsum("klcd,cdik,ablj->abij", u[o, o, v, v], t2, t2, optimize=True)
    res -= 2 * np.einsum("klcd,cakl,dbij->abij", u[o, o, v, v], t2, t2, optimize=True)
    res += np.einsum("klcd,ackl,dbij->abij", u[o, o, v, v], t2, t2, optimize=True)

    res = res + res.transpose(1, 0, 3, 2)
    # DONE WITH PERM TERM

    res += u[v, v, o, o]  # v_abij

    return res
