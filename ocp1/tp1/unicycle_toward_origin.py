import crocoddyl
import numpy as np
import matplotlib.pylab as plt
import unittest
from utils.unicycle import plotUnicycleSolution

# %jupyter_snippet hyperparams
### HYPER PARAMS: horizon and initial state
T  = 100
x0 = np.array([-1,-1,1])
# %end_jupyter_snippet

### PROBLEM DEFINITION

model = crocoddyl.ActionModelUnicycle()
# %jupyter_snippet termmodel
model_term = crocoddyl.ActionModelUnicycle()

model_term.costWeights = np.array([
    100,   # state weight
    0  # control weight
]).T
# %end_jupyter_snippet

# Define the optimal control problem.
problem = crocoddyl.ShootingProblem(x0, [ model ] * T, model_term)
# %jupyter_snippet ddp
# Select the solver for this problem
ddp = crocoddyl.SolverDDP(problem)
# %end_jupyter_snippet

# %jupyter_snippet callback
# Add solvers for verbosity and plots
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
# %end_jupyter_snippet

### SOLVE THE PROBLEM

# %jupyter_snippet solve
done = ddp.solve()
assert(done)
# %end_jupyter_snippet

### PLOT 

# %jupyter_snippet plotlog
log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
crocoddyl.plotConvergence(
    log.costs,
    log.pregs,
    log.dregs,
    log.grads,
    log.stops,
    log.steps,
    figIndex=2,
    show=False,
)
# %end_jupyter_snippet

plotUnicycleSolution(log.xs)

print('Type plt.show() to display the result.')

### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class UnicycleTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        self.assertTrue( len(ddp.xs) == len(ddp.us)+1 )
        self.assertTrue( np.allclose(ddp.xs[0],ddp.problem.x0) )
        self.assertTrue( ddp.stop<1e-6 )
        
if __name__ == "__main__":
    UnicycleTest().test_logs()
