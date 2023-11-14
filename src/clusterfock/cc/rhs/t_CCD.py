import numpy as np


def amplitudes_ccd(t2, u, f, v, o):
    res = np.zeros_like(t2, dtype=u.dtype)

    res += u[v, v, o, o]  # v_abij

    tp = np.einsum("bc,acij->abij", f[v, v], t2, optimize=True)
    res += tp - tp.transpose(1, 0, 2, 3)

    tp = np.einsum("kj,abik->abij", f[o, o], t2, optimize=True)
    res -= tp - tp.transpose(0, 1, 3, 2)

    # Two first sums, over cd and kl
    res += 0.5 * np.einsum("abcd,cdij->abij", u[v, v, v, v], t2, optimize=True)
    res += 0.5 * np.einsum("klij,abkl->abij", u[o, o, o, o], t2, optimize=True)

    # First permutation term, P(ij|ab), over kc
    tp = np.einsum("kbcj,acik->abij", u[o, v, v, o], t2, optimize=True)
    res += tp - tp.transpose(1, 0, 2, 3) - tp.transpose(0, 1, 3, 2) + tp.transpose(1, 0, 3, 2)

    # First double t2 sum, over klcd
    res += 0.25 * np.einsum("klcd,cdij,abkl->abij", u[o, o, v, v], t2, t2, optimize=True)

    # First P(ij) permutation, double t2 sum over klcd
    tp = np.einsum("klcd,acik,bdjl->abij", u[o, o, v, v], t2, t2, optimize=True)
    res += tp - tp.transpose(0, 1, 3, 2)

    # second P(ij) permutation, double t2 sum over klcd
    tp = np.einsum("klcd,dcik,ablj->abij", u[o, o, v, v], t2, t2, optimize=True)
    res -= 0.5 * (tp - tp.transpose(0, 1, 3, 2))

    # Only P(ab) term, double t2 sum over klcd
    tp = np.einsum("klcd,aclk,dbij->abij", u[o, o, v, v], t2, t2, optimize=True)
    res -= 0.5 * (tp - tp.transpose(1, 0, 2, 3))

    return res
