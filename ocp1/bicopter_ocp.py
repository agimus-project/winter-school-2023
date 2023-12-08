import numpy as np
import crocoddyl
from utils.bicopter import plotBicopter, plotBicopterSolution,ViewerBicopter
import matplotlib.pyplot as plt                                                                                                  
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

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


# Creating the DAM for the bicopter
bicopterModel = DifferentialActionModelBicopter()
bicopterData = bicopterModel.createData()
x = bicopterModel.state.rand()
u = np.array([12,8])                    
bicopterModel.calc(bicopterData,x,u)

# OCP
# Hyperparameters
timeStep = 0.01
x0 = np.array([1.0, 0.0, 0.0,  0.0, 0.0, 0.0])
T = 50

bicopterDAM = model = DifferentialActionModelBicopter()

# Using NumDiff for computing the derivatives. We specify the
# withGaussApprox=True to have approximation of the Hessian based on the
# Jacobian of the cost residuals.
bicopterND = crocoddyl.DifferentialActionModelNumDiff(bicopterDAM, True)

# Getting the IAM using the simpletic Euler rule
bicopterIAM = crocoddyl.IntegratedActionModelEuler(bicopterND, timeStep)

# Creating the shooting problem

terminalBicopter = DifferentialActionModelBicopter()
terminalBicopterDAM = crocoddyl.DifferentialActionModelNumDiff(terminalBicopter, True)
terminalBicopterIAM = crocoddyl.IntegratedActionModelEuler(terminalBicopterDAM)

terminalBicopter.costWeights[0] = 100
terminalBicopter.costWeights[1] = 100
terminalBicopter.costWeights[2] = 100.0
terminalBicopter.costWeights[3] = 100.0
terminalBicopter.costWeights[4] = 100
terminalBicopter.costWeights[5] = 100
terminalBicopter.costWeights[6] = 100
problem = crocoddyl.ShootingProblem(x0, [bicopterIAM] * T, terminalBicopterIAM)

# Solving it using DDP
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
ddp.solve([], [], 300)

# Display trajectory
fig,ax = plt.subplots(2,1, figsize=(6.4, 6.4))
xs = np.array(ddp.xs)
us = np.array(ddp.us)
ax[0].plot(xs[:,:3])
ax[1].plot(us)

plotBicopterSolution(list(ddp.xs)[::3],show=True)

viz = ViewerBicopter()
viz.displayTrajectory(xs,timeStep)
