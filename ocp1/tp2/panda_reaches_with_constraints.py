'''
In this example test, we will solve the reaching-goal task with the Panda arm.
For that, we use the forward dynamics (with its analytical derivatives) based on Pinocchio
and featured in the Pinocchio-based front-end implementation of Crocoddyl. The main code is
inside the DifferentialActionModelFullyActuated class.
We set up 3 main costs: state regularization, control regularization wrt gravity, and target reaching.
For temporal integration, we use an Euler implicit integration scheme.

This example is based on https://github.com/Gepetto/supaero2023/blob/main/tp5/arm_example.py
'''

import crocoddyl
import pinocchio as pin
import numpy as np
import example_robot_data as robex
import matplotlib.pylab as plt
import time
import mim_solvers
import unittest

# First, let's load the Pinocchio model for the Panda arm.
robot = robex.load('panda')
# The 2 last joints are for the fingers, not important in arm motion, freeze them
robot.model,[robot.visual_model,robot.collision_model] = \
    pin.buildReducedModel(robot.model,[robot.visual_model,robot.collision_model],[8,9],robot.q0)
robot.q0 = robot.q0[:7].copy()

# Hyperparameters of the movement
HORIZON_LENGTH = 100
TIME_STEP = 1e-2
FRAME_TIP = robot.model.getFrameId("panda_hand_tcp")
GOAL_POSITION = np.array([.2,0.5,.5])
GOAL_PLACEMENT = pin.SE3(pin.utils.rpyToMatrix(-np.pi,0,np.pi/4), GOAL_POSITION)
GOAL_POSITION = np.array([.2,0.6,.5])
GOAL_PLACEMENT = pin.SE3(pin.utils.rpyToMatrix(-np.pi,-1.5,1.5), GOAL_POSITION)
REACH_DIMENSION = "6d" # "3d"
# %jupyter_snippet hyperparameters
X_WALL_LOWER = .25
X_WALL_UPPER = .35
# %end_jupyter_snippet

# Configure viewer to vizualize the robot and a green box to feature the goal placement.
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
viz.addBox('world/goal',[.1,.1,.1],[0,1,0,1])
viz.applyConfiguration('world/goal',GOAL_PLACEMENT)

# Set robot model
robot_model = robot.model
robot_model.armature = np.ones(robot.model.nv)*2 # Arbitrary value representing the true armature
robot_model.q0 = robot.q0.copy()
robot_model.x0 = np.concatenate([robot_model.q0, np.zeros(robot_model.nv)])

# Define the state to be x=(q,v) position and velocity of the robot in the configuration space.
state = crocoddyl.StateMultibody(robot_model)

# Define the cost to be the sum of 3 terms: state regularisation xReg, control regularization uReg,
# and end-effector reaching. We create a special value for the terminal cost.
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# You can ask the end effector to reach a position (3d) or a placement (6d)
if REACH_DIMENSION == "3d":
    # Cost for 3d tracking || p(q) - pref ||**2
    goalTrackingRes = crocoddyl.ResidualModelFrameTranslation(state,FRAME_TIP,GOAL_POSITION)
    goalTrackingWeights = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1]))
elif REACH_DIMENSION == "6d":
    # Cost for 6d tracking  || log( M(q)^-1 Mref ) ||**2
    goalTrackingRes = crocoddyl.ResidualModelFramePlacement(state,FRAME_TIP,GOAL_PLACEMENT)
    goalTrackingWeights = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1, 1,1,1]))
else:
    assert( REACH_DIMENSION=="3d" or REACH_DIMENSION=="6d" )
goalTrackingCost = crocoddyl.CostModelResidual(state,goalTrackingWeights,goalTrackingRes)
runningCostModel.addCost("gripperPose", goalTrackingCost, .001)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4)
    
# Cost for state regularization || x - x* ||**2
# We set up different values for the integral cost and terminal cost term.

# Regularization is stronger on position than velocity (to account for typical unit scale)
xRegWeights = crocoddyl.ActivationModelWeightedQuad(np.array([1,1,1,1,1,1,1, .1,.1,.1,.1,.1,.1,.1]))
xRegRes = crocoddyl.ResidualModelState(state,robot_model.x0)
xRegCost = crocoddyl.CostModelResidual(state,xRegWeights,xRegRes)
runningCostModel.addCost("xReg", xRegCost, 1e-2)

# Terminal cost for state regularization || x - x* ||**2
# Require more strictly a small velocity at task end (but we don't car for the position)
xRegWeightsT=crocoddyl.ActivationModelWeightedQuad(np.array([.5,.5,.5,.5,.5,.5,.5,  5.,5.,5.,5.,5.,5.,5.]))
xRegResT = crocoddyl.ResidualModelState(state,robot_model.x0)
xRegCostT = crocoddyl.CostModelResidual(state,xRegWeightsT,xRegResT)
terminalCostModel.addCost("xReg", xRegCostT, .1)

# Cost for control regularization || u - g(q) ||**2
uRegRes = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state,uRegRes)
runningCostModel.addCost("uReg", uRegCost, 1e-5)


# %jupyter_snippet constraint_manager
# Define contraint
runningConstraints = crocoddyl.ConstraintModelManager(state, robot.nv)
# %end_jupyter_snippet

# %jupyter_snippet eewall
# Create contraint on end-effector
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state, FRAME_TIP, np.zeros(3)
)
eeWallContraint = crocoddyl.ConstraintModelResidual(
    state,
    frameTranslationResidual,
    np.array([X_WALL_LOWER, -np.inf, -np.inf]),
    np.array([X_WALL_UPPER, +np.inf, +np.inf]),
)
runningConstraints.addConstraint("ee_wall", eeWallContraint)
# %end_jupyter_snippet



# %jupyter_snippet iam
# Next, we need to create the running and terminal action model.
# The forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.

# The actuation model is here trivial: tau_q = u.
actuationModel = crocoddyl.ActuationModelFull(state)
# Running model composing the costs, the differential equations of motion and the integrator.
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, runningCostModel, runningConstraints),
    TIME_STEP)
runningModel.differential.armature = robot_model.armature
# Specific unconstrained initial model
runningModel_init = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, runningCostModel),
    TIME_STEP)
runningModel.differential.armature = robot_model.armature
# Terminal model following the same logic, although the integration is here trivial.
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuationModel, terminalCostModel), 0.)
terminalModel.differential.armature = robot_model.armature
# %end_jupyter_snippet

# Define the optimal control problem.
# For this optimal control problem, we define HORIZON_LENGTH knots (or running action
# models) plus a terminal knot
# %jupyter_snippet shoot
problem = crocoddyl.ShootingProblem(robot_model.x0,
                                    [runningModel_init] + [runningModel] * (HORIZON_LENGTH - 1),
                                    terminalModel)
# %end_jupyter_snippet

# %jupyter_snippet solver
solver = mim_solvers.SolverCSQP(problem)
solver.with_callbacks = True 
solver.termination_tolerance = 1e-3         # Termination criteria (KKT residual)
solver.max_qp_iters = 1000                  # Maximum number of QP iteration
solver.eps_abs = 1e-5                       # QP termination absolute criteria, 1e-9 
solver.eps_rel = 0.                         # QP termination absolute criteria
solver.use_filter_line_search = True        # True by default, False = use merit function
# %end_jupyter_snippet

# %jupyter_snippet solve_and_plot
# Solving it with the DDP algorithm
solver.solve([],[],1000)  # xs_init,us_init,maxiter

ees = [ d.differential.pinocchio.oMf[FRAME_TIP].translation for d in solver.problem.runningDatas ]
plt.plot(ees)
plt.plot([0,HORIZON_LENGTH],[X_WALL_UPPER,X_WALL_UPPER],'b--')
plt.plot([0,HORIZON_LENGTH],[X_WALL_LOWER,X_WALL_LOWER],'b--')
plt.legend(['x', 'y', 'z'])
# %end_jupyter_snippet

print('Type plt.show() to display the result.')

# %jupyter_snippet animate
# Visualizing the solution in gepetto-viewer
for x in solver.xs:
    viz.display(x[:robot.model.nq])
    time.sleep(TIME_STEP)
# %end_jupyter_snippet

# ## TEST ZONE ############################################################
# ## This last part is to automatically validate the versions of this example.
class LocalTest(unittest.TestCase):
    def test_logs(self):
        print(self.__class__.__name__)
        self.assertTrue( len(solver.xs) == len(solver.us)+1 )
        self.assertTrue( np.allclose(solver.xs[0],solver.problem.x0) )
        self.assertTrue( solver.stoppingCriteria()<1e-6 )
        
if __name__ == "__main__":
    LocalTest().test_logs()

