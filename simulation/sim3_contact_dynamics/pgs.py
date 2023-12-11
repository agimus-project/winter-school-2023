import numpy as np
# PGS algo #####################################################################
def solve_contact(G: np.ndarray,g: np.ndarray, mus: list, tol : float = 1e-6, max_iter :int = 100) -> np.ndarray:
    """PGS algorithm solving a contact problem with frictions.

    Args:
        G (np.ndarray): Delassus matrix.
        g (np.ndarray): free velocity of contact points.
        mus (list): list of coefficients of friction for the contact points.
        tol (float, optional): solver tolerance. Defaults to 1e-6.
        max_iter (int, optional): maximum number of iterations for the solver. Defaults to 100.

    Returns:
        np.ndarray: contact impulses.
    """
    # TODO : PGS
    nc = len(mus)
    lam = np.zeros(3*nc)
    for j in range(max_iter):
        for i in range(nc):
            lam[3*i+2] = lam[3*i+2] - (1./G[3*i+2, 3*i+2])*(G[3*i+2] @ lam + g[3*i+2])
            lam[3*i+2] = np.max([0., lam[3*i+2]])
            lam[3*i:3*i+2] = lam[3*i:3*i+2] - (1./np.min([G[3*i, 3*i], G[3*i+1, 3*i+1]]))*(G[3*i:3*i+2] @ lam + g[3*i:3*i+2])
            lam[3*i:3*i+2] = np.clip(lam[3*i:3*i+2], -mus[i]*lam[3*i+2], mus[i]*lam[3*i+2])
    return lam