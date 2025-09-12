import numpy as np
from scipy import constants as const

def lanczos_krylov(H, v, tol=1e-14, full_reorth=True, symmetrize=True):
    """
    Lanczos tridiagonalization (Hermitian) up to natural breakdown.
    Returns the Krylov basis Q (n x m), tridiagonal T (m x m),
    diagonal alpha (m,), off-diagonal beta (m-1,), and a status dict.
    No arbitrary basis completion is performed.

    Parameters
    ----------
    H : (n,n) array_like
        Hermitian matrix. If `symmetrize` is True, H is symmetrized as (H+H^*)/2.
    v : (n,) array_like
        Nonzero starting vector; Q[:,0] is v / ||v||.
    tol : float
        Breakdown tolerance for the residual norm.
    full_reorth : bool
        If True, use double-pass DGKS reorthogonalization. If False, no reorth.
    symmetrize : bool
        If True, defensively symmetrize H numerically.

    Returns
    -------
    Q : (n,m) ndarray
        Orthonormal Krylov basis. m is the reached Krylov dimension (m >= 1).
    T : (m,m) ndarray
        Hermitian tridiagonal such that Q^* H Q = T (up to rounding).
    alpha : (m,) ndarray
        Diagonal entries of T.
    beta : (m-1,) ndarray
        Off-diagonal entries of T, nonnegative by construction.
    status : dict
        Keys:
          - 'breakdown' (bool): True if residual < tol (natural termination).
          - 'm' (int): Krylov dimension reached.
          - 'err_tridiag' (float): ||Q^* H Q - T||_F.
          - 'err_orth' (float): ||Q^* Q - I||_F.
    """
    H = np.asarray(H)
    v = np.asarray(v)
    n = H.shape[0]
    if H.shape != (n, n) or v.shape != (n,):
        raise ValueError("Shape mismatch between H and v.")
    if symmetrize:
        H = (H + H.conj().T) / 2

    dtype = np.result_type(H.dtype, v.dtype, np.float64)
    H = H.astype(dtype, copy=False)
    v = v.astype(dtype, copy=False)

    # Normalize starting vector
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv <= 0:
        raise ValueError("Starting vector must be nonzero and finite.")
    q = v / nv

    # Reorthogonalization helper (double-pass DGKS)
    def reorth(r, Qblock):
        if Qblock.size == 0:
            return r
        if full_reorth:
            for _ in range(2):
                c = Qblock.conj().T @ r
                r = r - Qblock @ c
            return r
        else:
            c = Qblock.conj().T @ r
            return r - Qblock @ c

    Q_cols = [q]
    alpha = []
    beta = []

    # First step
    w = H @ q
    alpha.append(np.vdot(q, w).real)
    r = w - alpha[-1] * q

    breakdown = False
    while True:
        # Reorthogonalize residual
        Qblock = np.column_stack(Q_cols)
        r = reorth(r, Qblock)

        b = np.linalg.norm(r)
        if b < tol:  # natural breakdown: Krylov space has closed
            breakdown = True
            break

        beta.append(b)  # beta_j >= 0 by construction
        q_next = r / b  # choose the phase of r; keeps beta real, nonnegative
        Q_cols.append(q_next)

        if len(Q_cols) == n:  # reached full dimension
            # compute last alpha and exit
            w = H @ q_next
            alpha.append(np.vdot(q_next, w).real)
            break

        # recurrence for next residual
        w = H @ q_next
        alpha.append(np.vdot(q_next, w).real)
        r = w - alpha[-1] * q_next - beta[-1] * Q_cols[-2]

    # Assemble outputs
    Q = np.column_stack(Q_cols)
    m = Q.shape[1]
    alpha = np.asarray(alpha, dtype=dtype)
    beta = np.asarray(beta, dtype=dtype)

    T = np.diag(alpha)
    if m > 1:
        T += np.diag(beta, 1) + np.diag(beta, -1)

    # Diagnostics
    err_tridiag = np.linalg.norm(Q.conj().T @ H @ Q - T)
    err_orth = np.linalg.norm(Q.conj().T @ Q - np.eye(m, dtype=dtype))

    status = {
        "breakdown": breakdown,
        "m": m,
        "err_tridiag": err_tridiag,
        "err_orth": err_orth,
    }
    return Q, T, alpha, beta, status

def L_to_El(L):
    return (const.hbar / 2 / const.e) ** 2 / (L) / const.h


def C_to_Ec(C):
    return const.e**2 / 2 / C / const.h


def El_to_L(El):
    return (const.hbar / 2 / const.e) ** 2 / (El * const.h)


def Ec_to_C(Ec):
    return const.e**2 / (2 * Ec * const.h)