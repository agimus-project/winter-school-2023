import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from utils.random_qp import generate_convex_eqp, generate_convex_qp_nolicq, QP, infNorm

# %jupyter_snippet prox_iteration


def solve_qp_prox_one_iter(qp: QP, prev_l, mu: float = 0.01):
    """Perform one iteration of the proximal algo.

    This should return the same outputs as `solve_qp_inv_kkt`.
    """
    ...


# %end_jupyter_snippet
# %jupyter_snippet prox_iteration_sol


def solve_qp_prox_one_iter(qp: QP, prev_l, mu: float = 0.01):
    """Perform one iteration of the proximal algo.

    This should return the same outputs as `solve_qp_inv_kkt`.
    """
    Q = qp.Q
    q = qp.q
    A = qp.A
    b = qp.b + mu * prev_l
    nx = Q.shape[0]
    nc = b.shape[0]
    mat = np.block([[Q, A.T], [A, -mu * np.eye(nc)]])
    rhs = np.concatenate([q, b])

    matinv = npla.inv(mat)
    pd_opt = -matinv @ rhs
    xopt, yopt = pd_opt[:nx], pd_opt[nx:]
    derr = infNorm(Q @ xopt + qp.q + A.T @ yopt)
    perr = infNorm(A @ xopt + qp.b)
    return xopt, yopt, perr, derr


# %end_jupyter_snippet

# %jupyter_snippet iterate_once
test_qp = generate_convex_eqp(5, 4, 4)
yinit = np.zeros(4)
solve_qp_prox_one_iter(test_qp, yinit)
# %end_jupyter_snippet

# %jupyter_snippet iterate_manual
numiters = 30
yi = yinit
errs_ = []

for t in range(numiters):
    xi, yi, perr, derr = solve_qp_prox_one_iter(test_qp, yi, mu=0.01)  # play with mu?
    errs_.append((perr, derr))
    print(errs_[-1])

# %end_jupyter_snippet

# %jupyter_snippet plot_conv
errs_ = np.asarray(errs_)
plt.subplot(121)
# primal error
plt.plot(errs_[:, 0], ls="--")
plt.yscale("log")
plt.title("Primal error $\| Ax - b \|$")

plt.subplot(122)
plt.plot(errs_[:, 1], ls="--")
plt.yscale("log")
plt.title("Dual error $\| Qx + q + A^\\top y \|$")
plt.tight_layout()

# %end_jupyter_snippet

# %jupyter_snippet auto_iteration
def solve_qp_prox(qp: QP, yinit, mu: float = 0.01, epsilon=1e-12, max_iters=200):
    """Iterate the proximal algorithm until you have converged to a desired threshold :math:`epsilon`.

    Parameters
        qp: QP instance
        yinit: initial-guess for the dual variables
        mu: proximal parameter
        epsilon: threshold
        max_iters: maximum number of iterations

    This should return the same outputs as `solve_qp_inv_kkt`."""
    ...
# %end_jupyter_snippet
# %jupyter_snippet auto_iteration_sol
def solve_qp_prox(qp: QP, yinit, mu: float = 0.01, epsilon=1e-12, max_iters=200):
    yi = yinit.copy()
    for t in range(max_iters):
        xi, yi, perr, derr = solve_qp_prox_one_iter(qp, yi, mu)
        conv = max(perr, derr) <= epsilon
        if conv:
            break
    return xi, yi, perr, derr
# %end_jupyter_snippet

# %jupyter_snippet get_some_qps
qp1_strict = generate_convex_eqp(4, 4, 2, check_strictly_convex=True)
qp2_nostrict = generate_convex_eqp(4, 2, 2)
qp3_nolicq = generate_convex_qp_nolicq(4, 2, 2, 1)
qp4_strict_nolicq = generate_convex_qp_nolicq(4, 5, 2, 1)
qp5_overconstrained = generate_convex_eqp(4, 5, 5)

mu = 1e-5

for qp in [
    qp1_strict,
    qp2_nostrict,
    qp3_nolicq,
    qp4_strict_nolicq,
    qp5_overconstrained,
]:
    print("====")    
    xopt, yopt, perr, derr = solve_qp_prox(qp, yinit=np.zeros(qp.b.size), mu=mu)
    print("primal error = {}".format(perr))
    print("dual   error = {}".format(derr))
# %end_jupyter_snippet

# %jupyter_snippet auto_iteration_sol_diag
def solve_qp_prox(qp: QP, yinit, mu: float = 0.01, epsilon=1e-12, max_iters=200):
    yi = yinit.copy()
    perrs = []
    derrs = []
    for t in range(max_iters):
        xi, yi, _perr, _derr = solve_qp_prox_one_iter(qp, yi, mu)
        conv = max(_perr, _derr) <= epsilon
        perrs.append(_perr)
        derrs.append(_derr)
        if conv:
            break
    return xi, yi, perrs, derrs
# %end_jupyter_snippet

# %jupyter_snippet solver_mu_impact
qp = qp1_strict
mu_vals = [0.1, 0.01, 1e-4, 1e-6]
epsilon = 1e-14

plt.subplot(111)
for mu in mu_vals:
    _, _, perrs, derrs = solve_qp_prox(qp, np.zeros(qp.b.size), mu, epsilon=epsilon)
    perrs = np.asarray(perrs)
    plt.plot(perrs, label="$\\mu = {:.2e}$".format(mu), ls='--')

plt.yscale("log")
plt.legend()
plt.tight_layout()

# %end_jupyter_snippet