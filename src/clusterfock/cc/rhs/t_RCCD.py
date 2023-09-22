import numpy as np


def amplitudes_ccd_restricted(t, v, f_pp_o, f_hh_o, vir, occ):
    res = np.zeros_like(t)
    # Here we have a long permutation term, so we collect all sums and then perform the permutation

    # Fock terms, single sum
    res += np.einsum("bc,acij->abij", f_pp_o, t, optimize=True)
    res -= np.einsum("kj,abik->abij", f_hh_o, t, optimize=True)

    # virvir and occocc sums
    res += 0.5 * np.einsum("abcd,cdij->abij", v[vir, vir, vir, vir], t, optimize=True)
    res += 0.5 * np.einsum("klij,abkl->abij", v[occ, occ, occ, occ], t, optimize=True)

    # vir occ double sum
    res += 2 * np.einsum("kbcj,acik->abij", v[occ, vir, vir, occ], t, optimize=True)
    res -= np.einsum("kbcj,acki->abij", v[occ, vir, vir, occ], t, optimize=True)
    res -= np.einsum("kbic,ackj->abij", v[occ, vir, occ, vir], t, optimize=True)
    res -= np.einsum("kbjc,acik->abij", v[occ, vir, occ, vir], t, optimize=True)

    # vvoo sums
    res += 0.5 * np.einsum("klcd,cdij,abkl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res += 2 * np.einsum("klcd,acik,dblj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res -= 2 * np.einsum("klcd,acik,dbjl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res += 0.5 * np.einsum("klcd,caik,bdlj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res -= np.einsum("klcd,adik,cblj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res += np.einsum("klcd,adki,cblj->abij", v[occ, occ, vir, vir], t, t, optimize=True)

    res += 0.5 * np.einsum("klcd,cbil,adkj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res -= 2 * np.einsum("klcd,cdki,ablj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res += np.einsum("klcd,cdik,ablj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res -= 2 * np.einsum("klcd,cakl,dbij->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res += np.einsum("klcd,ackl,dbij->abij", v[occ, occ, vir, vir], t, t, optimize=True)

    res = res + res.transpose(1, 0, 3, 2)
    # DONE WITH PERM TERM

    res += v[vir, vir, occ, occ]  # v_abij

    return res
