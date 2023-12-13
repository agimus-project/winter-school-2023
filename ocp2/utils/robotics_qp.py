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


