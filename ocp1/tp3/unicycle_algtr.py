import numpy as np
import proxddp
from proxddp import dynamics, manifolds
import unittest
from utils.unicycle import plotUnicycleSolution

# %jupyter_snippet hyperparams
### HYPER PARAMS: horizon and initial state
T = 100
x0 = np.array([-1, -1, 1])
# %end_jupyter_snippet
### PROBLEM DEFINITION


# %jupyter_snippet dynamics
class UnicycleDynamics(dynamics.ExplicitDynamicsModel):
    def __init__(self, timestep):
        space = manifolds.R3()
        super().__init__(space, 2)
        self.dt = timestep

    def forward(self, x, u, data: dynamics.ExplicitDynamicsData):
        c, s = np.cos(x[2]), np.sin(x[2])
        data.xnext[:] = [
            x[0] + c * u[0] * self.dt,
            x[1] + s * u[0] * self.dt,
            x[2] + u[1] * self.dt,
        ]

    def dForward(self, x, u, data: dynamics.ExplicitDynamicsData):
        c, s = np.cos(x[2]), np.sin(x[2])
        data.Jx[:] = np.eye(3)
        data.Jx[0, 2] = -s * u[0] * self.dt
        data.Jx[1, 2] = c * u[0] * self.dt

        data.Ju[0, 0] = c * self.dt
        data.Ju[1, 0] = s * self.dt
        data.Ju[2, 1] = self.dt


# %end_jupyter_snippet

# %jupyter_snippet model
cost_weights = np.array([1.0, 1.0])
dt = 0.1
dyn_model = UnicycleDynamics(dt)
rcost = proxddp.QuadraticCost(
    w_x=cost_weights[0] * np.eye(3), w_u=cost_weights[1] * np.eye(2)
)
stage = proxddp.StageModel(rcost, dyn_model)
# %end_jupyter_snippet


# %jupyter_snippet problem
term_cost = rcost.copy()
term_cost.w_x[:] = 100.0 ** 2
term_cost.w_u[:] = 0.0

# Define the optimal control problem.
problem = proxddp.TrajOptProblem(x0, [stage] * T, term_cost)
# %end_jupyter_snippet

# %jupyter_snippet ddp
# Select the solver for this problem
TOL = 1e-5
ddp = proxddp.SolverProxDDP(TOL, verbose=proxddp.VERBOSE)
ddp.setup(problem)
# %end_jupyter_snippet

# %jupyter_snippet solve
done = ddp.run(problem, [], [])
assert done
# %end_jupyter_snippet

### PLOT

# %jupyter_snippet plot_sol
rs: proxddp.Results = ddp.results
plotUnicycleSolution(rs.xs)
# %end_jupyter_snippet
print("Type plt.show() to display the result.")
