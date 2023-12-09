import matplotlib.pyplot as plt
import numpy as np
import crocoddyl
import unittest
from utils.bicopter import plotBicopter, plotBicopterSolution,ViewerBicopter

# %jupyter_snippet hyperparams
### HYPER PARAMS: horizon and initial state
timeStep = 0.01
x0 = np.array([1.0, 0.0, 0.0,  0.0, 0.0, 0.0])
T = 50
# %end_jupyter_snippet

### MODEL DEFINITION

# Definition of the differential action model for the bicopter
# The state x=(q,v) is the concatenation of position and velocity of the copter
# (with q=(x1,x2,theta), x1 the horizontal position, x2 the vertical position and
# theta the angle)
# (with v=(v1,v2,w) the horizontal, vertical and angle velocities)
# The control is the thrust (vertical forces) of right then leg propellers
# (u=(f_right, f_left))
# The calc function compute the system acceleration data.xout and cost residuals
# data.residuals, along with the cost in data.cost=.5*sum(data.residuals**2).
# The calcDiff function should compute the acceleration derivatives dxout/dx,dxout/du
# and cost derivatives dcost/dx,dcost/du (and corresponding hessians). Here we skip
# the calcDiff and use finite differences to compute the Jacobians (of xout and r)
# and approximate Hessian (with Gauss H=J'J).
# %jupyter_snippet dam_header
class DifferentialActionModelBicopter(crocoddyl.DifferentialActionModelAbstract):

    def __init__(self):
        '''
        Init on top of the DAM class. 
        Mostly set up the hyperparameters of this model (mass, length, cost, etc).
        '''
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, crocoddyl.StateVector(6), nu=2, nr=12
        )
        self.unone = np.zeros(self.nu)

        self.span = .2
        self.mass = 2.
        self.g = 10
        self.inertia = self.mass*self.span**2

        self.costWeights = [
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
        ]  # sin, 1-cos, x, xdot, thdot, f

    def calc(self, data, x, u=None):
        if u is None:
            u = model.unone
        # Getting the state and control variables
        x1,x2,th,v1,v2,w = x
        fr,fl = u
# %end_jupyter_snippet

        # Shortname for system parameters
        mass,span,g,inertia=self.mass,self.span,self.g,self.inertia
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        # Force (fx,fz,tauy) in local frame
        loc_f = np.array([0, fr+fl, (fl-fr)*span])
        # Acceleration (x,z,th) in world frame
        data.xout = np.array([
            -loc_f[1]*s/mass,
            loc_f[1]*c/mass - g,
            loc_f[2]/inertia
        ])

        # Computing the cost residual and value
        data.r = self.costWeights * np.array([x1, x2, s, 1 - c,
                                              v1, v2, w,
                                              fr, fl,
                                              data.xout[0], data.xout[1], data.xout[2] ])
        data.cost = 0.5 * sum(data.r ** 2)

# %jupyter_snippet dam_calcdiff_template
    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives. Here
        # we will rely on finite differences.
        pass
# %end_jupyter_snippet

# %jupyter_snippet dam
# Creating the DAM for the bicopter
dam = DifferentialActionModelBicopter()
# %end_jupyter_snippet

# %jupyter_snippet dam_test
# Create a local DAM data for testing the implementation
dad = dam.createData()
x = dam.state.rand()
u = np.array([12,8])                    
dam.calc(dad,x,u)
# %end_jupyter_snippet

# %jupyter_snippet dam_nd
# Using NumDiff for computing the derivatives. We specify the
# withGaussApprox=True to have approximation of the Hessian based on the
# Jacobian of the cost residuals.
damND = crocoddyl.DifferentialActionModelNumDiff(dam, True)
# %end_jupyter_snippet

# %jupyter_snippet iam
# Getting the IAM using the simpletic Euler rule
iam = crocoddyl.IntegratedActionModelEuler(damND, timeStep)
# %end_jupyter_snippet

# %jupyter_snippet termmodel
# Similarly creates a terminal model, but change the cost weights
terminalDam = DifferentialActionModelBicopter()
terminalDamND = crocoddyl.DifferentialActionModelNumDiff(terminalDam, True)
terminalIam = crocoddyl.IntegratedActionModelEuler(terminalDamND)

terminalDam.costWeights[0] = 100 # horizontal position
terminalDam.costWeights[1] = 100 # vertical position
terminalDam.costWeights[2] = 100.0 # angle sin (first order)
terminalDam.costWeights[3] = 100.0 # angle cos (second order)
terminalDam.costWeights[4] = 100 # horizontal velocity
terminalDam.costWeights[5] = 100 # vertical velocity
terminalDam.costWeights[6] = 100 # angular velocity
# %end_jupyter_snippet

### PROBLEM DEFINITION

# %jupyter_snippet ocp
# Define the optimal control problem.
problem = crocoddyl.ShootingProblem(x0, [iam] * T, terminalIam)

# Solving it using DDP
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

### SOLVE THE PROBLEM

done = ddp.solve([], [], 300)
assert(done)
# %end_jupyter_snippet

### PLOT 

# %jupyter_snippet plot
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

plotBicopterSolution(list(ddp.xs)[::3])
# %end_jupyter_snippet

print('Type plt.show() to display the result.')

# %jupyter_snippet viz
# Animate the solution in meshcat
viz = ViewerBicopter()
viz.displayTrajectory(ddp.xs,timeStep)
# %end_jupyter_snippet
