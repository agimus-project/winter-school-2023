import numpy as np
from random_qp import QP

def load_crocoddyl_qp():
    '''
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
    Load a QP formed with the inverse dynamics of Pinocchio3x for the 
    model of Digit with all DOF, contact of the 2 feet on the ground and 
    close-loop constraints.

    The solution of this QP is a null primal and some forces for the dual.

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

