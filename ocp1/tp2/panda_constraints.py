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

# %jupyter_snippet robexload
# First, let's load the Pinocchio model for the Panda arm.
robot = robex.load('panda')
# The 2 last joints are for the fingers, not important in arm motion, freeze them
robot.model,[robot.visual_model,robot.collision_model] = \
    pin.buildReducedModel(robot.model,[robot.visual_model,robot.collision_model],[8,9],robot.q0)
robot.q0 = robot.q0[:7].copy()
# %end_jupyter_snippet

# Hyperparameters of the movement
# %jupyter_snippet hyperparameters
HORIZON_LENGTH = 100
TIME_STEP = 1e-2
FRAME_TIP = robot.model.getFrameId("panda_hand_tcp")
GOAL_POSITION = np.array([.2,0.5,.5])
GOAL_PLACEMENT = pin.SE3(pin.utils.rpyToMatrix(-np.pi,0,np.pi/4), GOAL_POSITION)
GOAL_POSITION = np.array([.3,0.6,.60])
GOAL_PLACEMENT = pin.SE3(pin.utils.rpyToMatrix(-np.pi,-1.5,1.5), GOAL_POSITION)
REACH_DIMENSION = "6d" # "3d"
# %end_jupyter_snippet

# Configure viewer to vizualize the robot and a green box to feature the goal placement.
# %jupyter_snippet viz
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
viz.addBox('world/goal',[.1,.1,.1],[0,1,0,1])
viz.applyConfiguration('world/goal',GOAL_PLACEMENT)
# %end_jupyter_snippet

# %jupyter_snippet robot_model
# Set robot model
robot_model = robot.model
robot_model.armature = np.ones(robot.model.nv)*2 # Arbitrary value representing the true armature
robot_model.q0 = robot.q0.copy()
robot_model.x0 = np.concatenate([robot_model.q0, np.zeros(robot_model.nv)])
# %end_jupyter_snippet

# Define the state to be x=(q,v) position and velocity of the robot in the configuration space.
# %jupyter_snippet state
state = crocoddyl.StateMultibody(robot_model)
# %end_jupyter_snippet

# Define the cost to be the sum of 3 terms: state regularisation xReg, control regularization uReg,
# and end-effector reaching. We create a special value for the terminal cost.
# %jupyter_snippet sumofcosts
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
# %end_jupyter_snippet

# You can ask the end effector to reach a position (3d) or a placement (6d)
# %jupyter_snippet cost_goal
if REACH_DIMENSION == "3d":
    # Cost for 3d tracking || p(q) - pref ||**2
    goalTrackingRes = crocoddyl.ResidualModelFrameTranslation(state,FRAME_TIP,GOAL_POSITION)
    goalTrackingCost = crocoddyl.CostModelResidual(state,goalTrackingRes)
    runningCostModel.addCost("gripperPose", goalTrackingCost, .001)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)
elif REACH_DIMENSION == "6d":
    # Cost for 6d tracking  || log( M(q)^-1 Mref ) ||**2
    goal6TrackingRes = crocoddyl.ResidualModelFramePlacement(state,FRAME_TIP,GOAL_PLACEMENT)
    goal6TrackingCost = crocoddyl.CostModelResidual(state,goal6TrackingRes)
    runningCostModel.addCost("gripperPose", goal6TrackingCost, .001)
    terminalCostModel.addCost("gripperPose", goal6TrackingCost, 4)
else:
    assert( REACH_DIMENSION=="3d" or REACH_DIMENSION=="6d" )
# %end_jupyter_snippet
    
# %jupyter_snippet cost_xreg
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
# %end_jupyter_snippet

# %jupyter_snippet cost_ureg
# Cost for control regularization || u - g(q) ||**2
uRegRes = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state,uRegRes)
runningCostModel.addCost("uReg", uRegCost, 1e-5)
# %end_jupyter_snippet





# Define contraint
# Create contraint on end-effector
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state, FRAME_TIP, np.zeros(3)
)
x_bound = 0.23
z_bound = 0.6
ee_contraint = crocoddyl.ConstraintModelResidual(
    state,
    frameTranslationResidual,
    np.array([-np.inf, -np.inf, -np.inf]),
    np.array([np.inf, np.inf, z_bound]),
)

jointLimitResiduals = crocoddyl.ResidualModelState(state,np.zeros(state.nx))
xLow = np.concatenate( [ robot_model.lowerPositionLimit , -np.ones(robot_model.nv)*np.inf ] )
xUp = np.concatenate( [ robot_model.upperPositionLimit , np.ones(robot_model.nv)*np.inf ] )
jointLimitContraints = crocoddyl.ConstraintModelResidual(
    state, jointLimitResiduals, xLow, xUp )

runningConstraints = crocoddyl.ConstraintModelManager(state, robot.nv)

runningConstraints.addConstraint("virtual_wall", ee_contraint)
runningConstraints.addConstraint("joint_limits", jointLimitContraints)




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
SQP_SOLVE = True

if SQP_SOLVE:
    solver = mim_solvers.SolverCSQP(problem)
    solver.with_callbacks = True 
    solver.termination_tolerance = 1e-4         # Termination criteria (KKT residual)
    solver.max_qp_iters = 10000                 # Maximum number of QP iteration
    solver.eps_abs = 1e-5                       # QP termination absolute criteria
    solver.eps_rel = 0.                         # QP termination absolute criteria

else:

    solver = crocoddyl.SolverDDP(problem)
    solver.setCallbacks(
        [
            crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
        ]
    )
    

# Solving it with the DDP algorithm
solver.solve([],[],1000)  # xs_init,us_init,maxiter
#assert( ddp.stop == 1.9384159634520916e-10 )


oMfs = [ d.differential.pinocchio.oMf[FRAME_TIP].translation for d in solver.problem.runningDatas ]
plt.ion()
plt.plot(oMfs)
#plt.plot([0,HORIZON_LENGTH],[z_bound,z_bound],'g--')
plt.legend(['x', 'y', 'z'])

# # Visualizing the solution in gepetto-viewer
for x in solver.xs:
    viz.display(x[:robot.model.nq])
    time.sleep(TIME_STEP)
