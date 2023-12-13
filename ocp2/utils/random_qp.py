import numpy as np


def wishart(n, p):
    """Generate a random symmetric positive (semi)definite matrix.
    
    Definite if p>=n.
    """
    root = np.random.randn(n, p)
    return root @ root.T


def gaussian_orthogonal_ensemble(n):
    A = np.random.randn(n)
    return 0.5 * (A + A.T)


def assemble_kkt_system(Q, J, q, c):
    """Assemble a KKT system (left-hand side and right-hand side)."""
    nc = c.shape[0]
    assert J.shape[0] == nc
    A = np.block([[Q, J.T], [J, np.zeros(nc, nc)]])
    r = np.concatenate([q, c])
    return A, r


def generate_convex_eqp(n, p, nc, check_strictly_convex = False):
    """Generate the parameters of a convex equality-QP.
    
    The check_strictly_convex flag will check if the problem is strictly convex.
    """
    if check_strictly_convex:
        assert n <= p
    Q = wishart(n, p)
    J = np.random.randn(nc, n)

    q = np.random.randn(n)
    c = np.random.randn(nc)

    return [Q, J, q, c]

