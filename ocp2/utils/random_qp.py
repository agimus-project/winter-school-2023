import numpy as np
import numpy.linalg as npla
from scipy.stats import ortho_group
from collections import namedtuple
import unittest

QP = namedtuple("QP", ["Q", "A", "q", "b"])



def wishart(n, p):
    """Generate a random symmetric positive (semi)definite matrix.
    
    Definite if p>=n.
    """
    root = np.random.randn(n, p)
    return root @ root.T


def gaussian_orthogonal_ensemble(n):
    A = np.random.randn(n)
    return 0.5 * (A + A.T)


def __assemble_kkt_system(qp: QP):
    """Assemble a KKT system (left-hand side and right-hand side)."""
    Q = qp.Q
    q = qp.q
    A = qp.A
    b = qp.b
    nc = b.shape[0]
    assert A.shape[0] == nc
    mat = np.block([[Q, A.T], [A, np.zeros((nc, nc))]])
    r = np.concatenate([q, b])
    return mat, r


def generate_convex_eqp(n, p, nc, check_strictly_convex = False):
    """Generate the parameters of a convex equality-QP.
    
    The check_strictly_convex flag will check if the problem is strictly convex.
    """
    if check_strictly_convex:
        assert n <= p
    Q = wishart(n, p)
    A = np.random.randn(nc, n)

    q = np.random.randn(n)
    b = np.random.randn(nc)

    return QP(Q, A, q, b)


def generate_convex_qp_nolicq(n, p, nc, nredundant=1):
    """Generate a convex QP which lacks LICQ conditions, by making some row constraints redundant.
    """

    qp = generate_convex_eqp(n, p, nc, False)

    # modify the A matrix to add redundant rows
    ridx = list(np.random.choice(nc, nredundant))

    rows = qp.A[ridx, :].copy()
    br = qp.b[ridx].copy()
    if nredundant > 1:
        random_rot = ortho_group.rvs(nredundant)
        rows = random_rot @ rows
        br = random_rot @ br
    else:
        rows = -rows
        br = -br
    rows = rows.reshape(nredundant, n)
    Anew = np.vstack([qp.A, rows])
    bnew = np.concatenate([qp.b, br])

    assert Anew.shape[0] == nc + nredundant
    assert bnew.shape[0] == Anew.shape[0]

    return QP(qp.Q, Anew, qp.q, bnew)



class Test(unittest.TestCase):
    def test_strict_convex(self):
        qp = generate_convex_eqp(4, 5, 2, True)
        kkt, _ = __assemble_kkt_system(qp)
        d = npla.det(kkt)

    def test_nolicq(self):
        qp = generate_convex_qp_nolicq(4, 4, 3, 2)
        kkt, _ = __assemble_kkt_system(qp)
        print(kkt)
        d = npla.det(kkt)
        print("det:", d)
        self.assertAlmostEqual(d, 0.)


if __name__ == "__main__":
    Test().test_nolicq()
