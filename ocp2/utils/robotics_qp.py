import numpy as np
from .random_qp import QP

def load_crocoddyl_qp():
    '''
    This function returns a QP problem corresponding to one step
    of a OCP solver.

    # This QP was generated with the following code
    # run ocp1/tp2/ with HORIZON_LENGTH=30 (line 33) and ddp.solve([],[],3) (line 146)
    kkt=crocoddyl.SolverKKT(problem)
    kkt.solve(ddp.xs,ddp.us)
    plt.figure(3)
    plt.imshow(np.log(abs(K)+1e-9))
    plt.imshow(np.log(abs(kkt.kkt)+1e-9))
    plt.show()
    np.save(open(f"/tmp/qp_croco.npy", "wb"),
            { 'K': kkt.kkt, 'k': kkt.kktref })
    '''

    qp_doc = '''
    This problem corresponds to one of the iteration of the FDDP solver of Crocoddyl
    on a problem of manipulation. 
    The decision variables are the robot states along 31 sampling points
    and corresponding control along the first 30 points. 
    Each state is composed of the robot position (nq=7) and velocity (nv=7)
    for a total dimension nx=14.
    Each control is composed of the robot joint torques for a total dimension nu=7.
    The total dimension of the decision variables is (31*14+30*7)==644.
    The constraints correspond to the integration of the robot equation of motion
    from each of the 30 first nodes to each of the 30 last nodes, plus an initial
    constraint on the initial state. It corresponds to 31*14=434 constraints.
    The problem is full rank, sparse and strongly structured.
    '''
    
    FILENAME = "data/qp_croco.npy"
    d=np.load(open(FILENAME, "rb"),allow_pickle=True).item()
    K=d['K']
    k=d['k']
    nx=14
    nu=7
    T=30
    NY=nx*(T+1)+nu*T
    NC=nx*(T+1)

    assert(NY+NC==K.shape[0])
    assert(NY+NC==K.shape[1])
    assert(NY+NC==k.shape[0])
    assert( np.all(K[NY:,NY:]==0) )
    
    return QP(K[:NY,:NY],
              K[NY:,:NY],
              k[:NY], k[NY:])



def load_digit_dyn_qp():
    '''
    This function returns a QP problem corresponding to the simulation
    of a closed-kinematic legged robot.
    
    Load a QP formed with the inverse dynamics of Pinocchio3x for the 
    model of Digit with all DOF, contact of the 2 feet on the ground and 
    close-loop constraints.

    The solution of this QP is a null primal and some forces for the dual.

    This is the code used to generate the QP.

def gravityCompensingControls(model, data, q, v, actMatrix, constraint_models, constraint_datas):
    
    nu = actMatrix.shape[1]
    nv = model.nv
    assert(nv == actMatrix.shape[0])
    nc = np.sum([cm.size() for cm in constraint_models])
    nx = nu + nc

    pin.computeAllTerms(model, data, q, v)
    Jac = pin.getConstraintsJacobian(model, data, constraint_models, constraint_datas)

    M = np.diag(np.concatenate((np.ones(nu), np.zeros(nc))))
    P = M.T @ M  
    p = np.zeros(nx)
    A = np.concatenate((actMatrix, Jac.transpose()), axis=1)
    b = data.nle
    G = np.zeros((nx, nx))
    h = np.zeros(nx)

    x = solve_qp(P, p, G, h, A, b, solver="proxqp", verbose=True, eps_abs=1e-12)
    np.save(open(f"/tmp/qp_digit.npy", "wb"),
            {
            'M': data.M, 'J': Jac,
            'grad': actMatrix@x[:nu]-data.g,  'gamma': np.zeros(Jac.shape[0])
            })
    print("Saved")
    '''

    qp_doc = '''
    This problem is obtained from the forward dynamics (simulation) of 
    the robot Digit in contact with the ground. Digit is composed
    of 60 degrees of freedom, with parallele linkages inducing a closed
    kinematics which can be written as 36 constraints. Additionally, 
    the two feet are in contact with the ground adding 12 more constraints.

    The decision variables are the robot joint acceleration of size na=60.
    The constraints are the jacobians of the relative positions or placements
    of the linkages (or contact points). They can be written as
    $$ J_c a + \gamma_c = 0$$
    where $J_c$ are the constraint jacobians, and $\gamma_c$ are the self acceleration
    (frame acceleration due to the robot motion), null in this particular example
    because the robot is static.
    The cost implements the Gauss principle, ie it is the difference between the
    acceleration and the "free" acceleration (obtained in free fall) following the
    metrics induced by the mass matrix:
    $$c(a) = \frac{1}{2} || a - a_0 ||_M^2 = \frac{1}{2} \left( a^T M a^T - a^T (b-\tau)\right) $$
    where $M$ is the mass matrix, $b$ are the nonlinear (Coriolis+centrifugal+gravity)
    effects, $\tau$ are the joint torques due to the motors and $a_0 = M^{-1} (\tau-b)$.
    
    In this particular example, we choose a state with 0 velocity (hence $\gamma_c=0$ 
    and $b$ is the gravity) and where the torque produces 0 acceleration (gravity
    compensation).
    The results of the QP is a=0 (primal is null), while the dual is nonzero
    and corresponds to the contact forces.
    '''

    
    FILENAME = "data/qp_digit.npy"
    d=np.load(open(FILENAME, "rb"),allow_pickle=True).item()
    M=d['M']
    nle=d['grad']
    J=d['J']
    gamma=d['gamma']

    nv=60
    nc=48

    assert(nv-nc==12)
    assert(J.shape[0]==nc)
    assert(gamma.shape[0]==nc)
    assert( np.all(gamma==0) )
    assert(M.shape==(nv,nv))
    assert(nle.shape[0]==nv)
        
    return QP(M,J,nle,gamma)


def load_tsid_qp():
    '''
    This function returns a QP problem corresponding to the inverse
    dynamics in task-space (TSID) of a legged robot in contact
    with the ground.
    
    '''

    qp_doc = '''
    This QP is a TSID-like problem.
    The decision variables are x=[qddot, tau, f] (size 38+32+12=82).
    The cost to minimize is:
        sum(J qddot + gamma - PD) for tasks COM, posture and gripper
    The constraints are 
        M qddot + b = S' tau + Jc' f (size 38)
        Jc qddot + gamma_c = 0 (size 12)
    '''
    
    FILENAME = "data/qp_tsid.npy"
    d=np.load(open(FILENAME, "rb"),allow_pickle=True).item()

    na=38
    nf=12
    ntau=32
    nx=na+nf+ntau
    
    Q,q,A,b = d['Q'],d['q'],d['A'],d['b']
    
    assert(na-ntau==6)
    assert(np.all(A[-nf:,na:]==0))
    assert(Q.shape==(nx,nx))
    assert(q.shape==(nx,))
    assert(A.shape==(na+nf,nx))
    assert(b.shape==(na+nf,))
    assert(np.all(Q[na:,:]==0))
    assert(np.all(Q[:,na:]==0))

    print(q.shape)
    return QP(Q,A,q,b)

