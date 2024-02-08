import numpy as np

def lambda_amplitudes_rccd(t2, l2, u, f, v, o):
    r2 = np.zeros_like(t2)
    M, _, N, _ = r2.shape
    dtype = u.dtype
    zeros = lambda shape: np.zeros(shape, dtype=dtype)

    r2 = np.zeros((M, M, N, N))

    r2 -= 2 * np.einsum(
        "ijba->abij", u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ijab->abij", u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ca,bcij->abij", f[v, v], l2, optimize=True
    )

    r2 -= 2 * np.einsum(
        "cb,acji->abij", f[v, v], l2, optimize=True
    )

    r2 += 4 * np.einsum(
        "ca,bcji->abij", f[v, v], l2, optimize=True
    )

    r2 += 4 * np.einsum(
        "cb,acij->abij", f[v, v], l2, optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,ijlk->abij", l2, u[o, o, o, o], optimize=True
    )

    r2 += 4 * np.einsum(
        "abkl,ijkl->abij", l2, u[o, o, o, o], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdij,dcab->abij", l2, u[v, v, v, v], optimize=True
    )

    r2 += 7 * np.einsum(
        "cdij,cdab->abij", l2, u[v, v, v, v], optimize=True
    ) / 2

    r2 -= 4 * np.einsum(
        "ik,abkj->abij", f[o, o], l2, optimize=True
    )

    r2 -= 4 * np.einsum(
        "jk,abik->abij", f[o, o], l2, optimize=True
    )

    r2 += 2 * np.einsum(
        "ik,abjk->abij", f[o, o], l2, optimize=True
    )

    r2 += 2 * np.einsum(
        "jk,abki->abij", f[o, o], l2, optimize=True
    )

    r2 -= 4 * np.einsum(
        "acik,cjbk->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 -= 4 * np.einsum(
        "acjk,icbk->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 -= 4 * np.einsum(
        "acki,cjkb->abij", l2, u[v, o, o, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "bcik,cjka->abij", l2, u[v, o, o, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "bcjk,icka->abij", l2, u[o, v, o, v], optimize=True
    )

    r2 -= 4 * np.einsum(
        "bcki,cjak->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 -= 4 * np.einsum(
        "bckj,icak->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 -= 3 * np.einsum(
        "ackj,ickb->abij", l2, u[o, v, o, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "acjk,ickb->abij", l2, u[o, v, o, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "acki,cjbk->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 += 2 * np.einsum(
        "ackj,icbk->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 += 2 * np.einsum(
        "bcik,cjak->abij", l2, u[v, o, v, o], optimize=True
    )

    r2 += 2 * np.einsum(
        "bcki,cjka->abij", l2, u[v, o, o, v], optimize=True
    )

    r2 += 2 * np.einsum(
        "bckj,icka->abij", l2, u[o, v, o, v], optimize=True
    )

    r2 += 8 * np.einsum(
        "acik,cjkb->abij", l2, u[v, o, o, v], optimize=True
    )

    r2 += 8 * np.einsum(
        "bcjk,icak->abij", l2, u[o, v, v, o], optimize=True
    )

    r2 -= 8 * np.einsum(
        "abik,cdlk,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "abkj,cdkl,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acij,cdkl,klbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acik,cdkl,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acik,cdlk,ljdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acjk,cdkl,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "acki,cdkl,ljdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "ackl,cdlk,ijdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bcik,cdkl,ljda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bcji,cdkl,klad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bcjk,cdkl,ilda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bcjk,cdlk,ilad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bckj,cdkl,ilad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "bckl,cdlk,ijad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "cdik,cdlk,ljab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 8 * np.einsum(
        "cdkj,cdkl,ilab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abjk,cdlk,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abki,cdkl,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "abkl,cdlk,ijcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acji,cdlk,klbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acjk,cdlk,ildb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "acki,cdlk,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,cdkl,ildb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackj,cdlk,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "ackl,cdkl,ijbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcij,cdlk,klad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcik,cdlk,ljad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcki,cdkl,ljad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bcki,cdlk,ljda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckj,cdlk,ilda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "bckl,cdkl,ijda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdij,cdlk,klab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdik,cdkl,ljba->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 -= 2 * np.einsum(
        "cdkj,cdlk,ilba->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abik,cdkl,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abjk,cdkl,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abki,cdlk,ljcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abkj,cdlk,ilcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "abkl,cdkl,ijcd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acij,cdlk,klbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acik,cdlk,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acji,cdkl,klbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acjk,cdkl,ildb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acjk,cdlk,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acki,cdkl,ljbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "acki,cdlk,ljdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackj,cdkl,ilbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackj,cdlk,ildb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,cdkl,ijdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "ackl,cdlk,ijbd->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcij,cdkl,klad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcik,cdkl,ljad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcik,cdlk,ljda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcji,cdlk,klad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcjk,cdlk,ilda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcki,cdkl,ljda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bcki,cdlk,ljad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckj,cdkl,ilda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckj,cdlk,ilad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,cdkl,ijad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "bckl,cdlk,ijda->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdij,cdkl,klab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdik,cdkl,ljab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdik,cdlk,ljba->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdkj,cdkl,ilba->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 4 * np.einsum(
        "cdkj,cdlk,ilab->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 16 * np.einsum(
        "acik,cdkl,ljdb->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    r2 += 16 * np.einsum(
        "bcjk,cdkl,ilad->abij", l2, t2, u[o, o, v, v], optimize=True
    )

    return r2