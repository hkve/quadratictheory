import numpy as np

def lambda_amplitudes_ccd(t2, l2, u, f, v, o):
    M, _, N, _ = t2.shape
    r2 = np.zeros((M, M, N, N))

    r2 += np.einsum(
        "ackj,cdkl,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bcik,cdkl,ljad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "abik,cdkl,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 += np.einsum(
        "bcij,cdkl,klad->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 += np.einsum(
        "bckl,cdkl,ijad->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 += np.einsum(
        "cdik,cdkl,ljab->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 -= np.einsum(
        "acik,cdkl,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "bckj,cdkl,ilad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= np.einsum(
        "abkj,cdkl,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 -= np.einsum(
        "acij,cdkl,klbd->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 -= np.einsum(
        "ackl,cdkl,ijbd->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 -= np.einsum(
        "cdkj,cdkl,ilab->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 2

    r2 += np.einsum(
        "abkl,cdkl,ijcd->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 4

    r2 += np.einsum(
        "cdij,cdkl,klab->abij", l2, t2, u[o, o, v, v], optimize=True
    ) / 4

    r2 += np.einsum(
        "ijab->abij", u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ackj,icbk->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 += np.einsum(
        "bcik,cjak->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 -= np.einsum(
        "acik,cjbk->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 -= np.einsum(
        "bckj,icak->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 += np.einsum(
        "cdij,cdab->abij", l2, u[v, v, v, v], optimize=True
    ) / 2

    r2 += np.einsum(
        "abkl,ijkl->abij", l2, u[o, o, o, o], optimize=True
    ) / 2

    r2 -= np.einsum(
        "ik,abkj->abij", f[o, o], l2, optimize=True
    )

    r2 -= np.einsum(
        "jk,abik->abij", f[o, o], l2, optimize=True
    )

    r2 += np.einsum(
        "cb,acij->abij", f[v, v], l2, optimize=True
    )

    r2 -= np.einsum(
        "ca,bcij->abij", f[v, v], l2, optimize=True
    )

    return r2