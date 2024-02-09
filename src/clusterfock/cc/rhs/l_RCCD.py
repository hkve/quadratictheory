import numpy as np

def lambda_amplitudes_rccd(t2, l2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r2 = zeros((M, M, N, N))

    r2 += np.einsum(
        "ca,bcji->abij", f[v, v], l2, optimize=True
    )

    r2 += np.einsum(
        "cb,acij->abij", f[v, v], l2, optimize=True
    )

    r2 -= np.einsum(
        "ijba->abij", u[o, o, v, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "ijab->abij", u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "abkl,ijkl->abij", l2, u[o, o, o, o], optimize=True
    )

    r2 -= np.einsum(
        "cdij,dcab->abij", l2, u[v, v, v, v], optimize=True
    ) / 12

    r2 += 5 * np.einsum(
        "cdij,cdab->abij", l2, u[v, v, v, v], optimize=True
    ) / 6

    r2 -= np.einsum(
        "ik,abkj->abij", f[o, o], l2, optimize=True
    )

    r2 -= np.einsum(
        "jk,abik->abij", f[o, o], l2, optimize=True
    )

    r2 -= np.einsum(
        "acik,cjbk->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 -= np.einsum(
        "acjk,icbk->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 -= np.einsum(
        "bcik,cjka->abij", l2, u[v, o, o, v], optimize=True
    )

    r2 -= np.einsum(
        "bcjk,icka->abij", l2, u[o, v, o, v], optimize=True
    )

    r2 -= np.einsum(
        "bcki,cjak->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 += 2 * np.einsum(
        "acik,cjkb->abij", l2, u[v, o, o, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "bcjk,icak->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,ickb->abij", l2, u[o, v, o, v], optimize=True
    ) / 3

    r2 += np.einsum(
        "acjk,ickb->abij", l2, u[o, v, o, v], optimize=True
    ) / 6

    r2 += np.einsum(
        "abik,cdkl,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "abkj,cdlk,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "abkl,cdkl,ijcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "acij,cdlk,klbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "acik,cdlk,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "acjk,cdkl,ildb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "acjk,cdlk,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ackj,cdlk,ildb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "ackl,cdlk,ijbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bcik,cdkl,ljad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bcik,cdlk,ljda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bcji,cdlk,klad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bcjk,cdlk,ilda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bcki,cdlk,ljad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "bckl,cdlk,ijda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cdij,cdkl,klab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cdik,cdlk,ljba->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += np.einsum(
        "cdkj,cdkl,ilba->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abik,cdlk,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkj,cdkl,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acij,cdkl,klbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,cdkl,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acik,cdlk,ljdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,cdkl,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,cdlk,ijdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,cdkl,ljda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcji,cdkl,klad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,cdkl,ilda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcjk,cdlk,ilad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,cdlk,ijad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,cdlk,ljab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdkj,cdkl,ilab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,cdkl,ljdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,cdkl,ilad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    return r2