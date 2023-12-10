import numpy as np
import proxddp
from proxddp import dynamics, manifolds
from utils.bicopter import plotBicopterSolution, ViewerBicopter

# %jupyter_snippet hyperparams
### Horizon and initial state
timeStep = 0.01
x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
T = 50
mass = 1.0
span = 0.2
grav = 10.0
costWeights = np.array([
    0.1, # x
    0.1, # z
    .10, # s
    .10, # c
    0.001, # vx
    0.001, # vz
    0.001, # w
    0.0, # fr
    0.0, # fl
    0.001,0.001,0.001, # a
])  # sin, 1-cos, x, xdot, thdot, f
nu = 2
space = manifolds.VectorSpace(6)  # state space
# %end_jupyter_snippet

# %jupyter_snippet bicopter_acc
def bicopter_acc_impl(x, u, span, mass, g):
    # Implementation of the Bicopter acceleration
    inertia = mass * span**2
    x1, x2, th, v1, v2, w = x
    fr, fl = u
    s, c = np.sin(th), np.cos(th)
    loc_f = np.array([0, fr + fl, (fl - fr) * span])
    return np.array([-loc_f[1] * s / mass, loc_f[1] * c / mass - g, loc_f[2] / inertia])
# %end_jupyter_snippet

# %jupyter_snippet state_error

class BicopterStateError(proxddp.StageFunction):
    def __init__(self):
        super().__init__(6, nu, nr=12)

    def evaluate(self, x, u, y, data: proxddp.StageFunctionData):
        x1, x2, th, v1, v2, w = x
        fr, fl = u
        s, c = np.sin(th), np.cos(th)
        # finish implementation here
# %end_jupyter_snippet
        acc = bicopter_acc_impl(x, u, span, mass, grav)
        r = np.array([x1, x2, s, 1 - c, v1, v2, w, fr, fl, *acc])
        data.value[:] = r

# %jupyter_snippet computeJacobians
    def computeJacobians(self, x, u, y, data: dynamics.StageFunctionData):
        # you can implement the derivatives of the error function here
        pass
# %end_jupyter_snippet

# %jupyter_snippet ode
class BicopterODE(dynamics.ODEAbstract):
    def __init__(self):
        super().__init__(space, nu)
        self.unone = np.zeros(nu)
        self.inertia = mass * span**2

    def forward(self, x, u, data: dynamics.ODEData):
        # data.xdot must contain the first-order time derivative
        # of the state variable x
        x1, x2, th, v1, v2, w = x
        fr, fl = u
        # finish implementation here
# %end_jupyter_snippet
        data.xdot[:3] = v1, v2, w
        data.xdot[3:] = bicopter_acc_impl(x, u, span, mass, grav)

# %jupyter_snippet dForward
    def dForward(self, x, u, data: dynamics.ODEData):
        # you can implement the derivatives of th dynamics here
        pass
# %end_jupyter_snippet

# %jupyter_snippet dynamics_and_residual
ode = BicopterODE()
state_err_ = BicopterStateError()
# %end_jupyter_snippet

# %jupyter_snippet integrator
dyn_model_ = dynamics.IntegratorSemiImplEuler(ode, timeStep)  # has no derivatives
# %end_jupyter_snippet

# %jupyter_snippet finite_difference
# Use the finite-difference helpers from proxddp
# DynamicsFiniteDifferenceHelper, FiniteDifferenceHelper 
fd_eps = 1e-4
dyn_model_nd = proxddp.DynamicsFiniteDifferenceHelper(space, dyn_model_, fd_eps)
state_err_nd = proxddp.FiniteDifferenceHelper(space, state_err_, fd_eps)
# define a quadratic cost from the bicopter state error
rcost = proxddp.QuadraticResidualCost(space, state_err_nd, np.diag(costWeights ** 2 * timeStep))
# %end_jupyter_snippet

# %jupyter_snippet termmodel
# Terminal cost: same as the runningcost, but we increase the cost weights
# in the library, they're a square matrix you can access through `cost.weights`
term_cost = proxddp.QuadraticResidualCost(space, state_err_nd, np.diag(costWeights ** 2))
_w = np.diagonal(term_cost.weights)
_w.setflags(write=True)
_w[:] = 1e4

# %end_jupyter_snippet


# %jupyter_snippet ocp
stage = proxddp.StageModel(rcost, dyn_model_nd)
problem = proxddp.TrajOptProblem(x0, [stage] * T, term_cost)

TOL = 1e-5
verbosity = proxddp.VERBOSE
solver = proxddp.SolverProxDDP(TOL, max_iters=300, verbose=verbosity)
solver.setup(problem)  # allocate data for this problem
ok = solver.run(problem, [], [])

rs: proxddp.Results = solver.results
print(rs)
xs_opt = rs.xs.tolist()
# %end_jupyter_snippet

# %jupyter_snippet plot
plotBicopterSolution(xs_opt)

print('Type plt.show() to display the result.')
# %end_jupyter_snippet

# %jupyter_snippet viz
viz = ViewerBicopter()
viz.displayTrajectory(xs_opt, timeStep)
# %end_jupyter_snippet
