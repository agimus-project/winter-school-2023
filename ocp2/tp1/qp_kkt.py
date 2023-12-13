import numpy as np
import numpy.linalg as npla
import unittest

from utils.random_qp import generate_convex_eqp, generate_convex_qp_nolicq, infNorm, QP

# %jupyter_snippet get_a_qp
nx = 3
nc = 1

qp = generate_convex_eqp(nx, nx, nc)
# %end_jupyter_snippet

# %jupyter_snippet assemble_kkt
# Assemble the KKT matrix
K = np.block([[qp.Q, qp.A.T], [qp.A, np.zeros([nc, nc])]])
# Assemble the corresponding vector
k = np.concatenate([-qp.q, -qp.b])

# Solve the QP by inverting the QP
primal_dual = npla.inv(K) @ k
# Extact primal and dual optimal from the KKT inversion
x_opt = primal_dual[:nx]
mult_opt = primal_dual[nx:]
# %end_jupyter_snippet

# %jupyter_snippet pd_err
perr = infNorm(qp.A @ x_opt + qp.b)
derr = infNorm(qp.Q @ x_opt + qp.q + qp.A.T @ mult_opt)
print("Primal error:", perr)
print("Dual   error:", derr)
# %end_jupyter_snippet

# %jupyter_snippet kkt


def solve_qp_inv_kkt(qp: QP):
    """Routine to solve a QP from its KKT matrix.

    Must return: primal solution, dual solution,
    primal and dual residual.
    """
    ...


# %end_jupyter_snippet
# %jupyter_snippet kkt_solution
def solve_qp_inv_kkt(qp: QP):
    Q = qp.Q
    q = qp.q
    A = qp.A
    b = qp.b
    nx = Q.shape[0]
    nc = b.shape[0]
    mat = np.block([[Q, A.T], [A, np.zeros((nc, nc))]])
    rhs = np.concatenate([q, b])

    matinv = npla.inv(mat)
    pd_opt = -matinv @ rhs
    xopt, yopt = pd_opt[:nx], pd_opt[nx:]
    derr = infNorm(Q @ xopt + q + A.T @ yopt)
    perr = infNorm(A @ xopt + b)
    return xopt, yopt, perr, derr


# %end_jupyter_snippet


# %jupyter_snippet get_some_qps
qp1_strict = generate_convex_eqp(4, 4, 2, check_strictly_convex=True)
qp2_nostrict = generate_convex_eqp(4, 2, 2)
qp3_nolicq = generate_convex_qp_nolicq(4, 2, 2, 1)
qp4_strict_nolicq = generate_convex_qp_nolicq(4, 5, 2, 1)
qp5_overconstrained = generate_convex_eqp(4, 5, 5)
# %end_jupyter_snippet

# %jupyter_snippet solve_some_qps
for qp in [
    qp1_strict,
    qp2_nostrict,
    qp3_nolicq,
    qp4_strict_nolicq,
    qp5_overconstrained,
]:
    try:
        print("====")
        xopt, yopt, perr, derr = solve_qp_inv_kkt(qp)
        print("primal error = {}".format(perr))
        print("dual   error = {}".format(derr))
    except Exception as e:
        print("Got an exception: {}".format(e))

# %end_jupyter_snippet


from quadprog import solve_qp

x, f, _, _, mult, _ = solve_qp(qp.Q, -qp.q, qp.A.T, -qp.b, nc)
assert np.allclose(x_opt, x)
assert np.allclose(mult_opt, -mult)


### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class LocalTest(unittest.TestCase):
    """
    If you have QuadProg, check the KKT solution against it.
    """

    def test_logs(self):
        try:
            from quadprog import solve_qp

            x, f, _, _, mult, _ = solve_qp(qp.Q, -qp.q, qp.A.T, -qp.b, nc)
            assert np.allclose(x_opt, x)
            assert np.allclose(mult_opt, -mult)
        except ModuleNotFoundError:
            print("You don't have QuadProg")


if __name__ == "__main__":
    LocalTest().test_logs()
