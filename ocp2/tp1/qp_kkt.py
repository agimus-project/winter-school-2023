import numpy as np
import numpy.linalg as npla
import unittest

from utils.random_qp import generate_convex_eqp

nx = 3
nc = 1

qp = generate_convex_eqp(nx,nx,nc)

# Assemble the KKT matrix 
K = np.block([ [qp.Q, qp.A.T], [qp.A, np.zeros([nc,nc]) ]])
# Assemble the corresponding vector
k = np.concatenate([ -qp.q, -qp.b ])

# Solve the QP by inverting the QP
primal_dual = npla.inv(K) @ k
# Extact primal and dual optimal from the KKT inversion
x_opt = primal_dual[:nx]
mult_opt = primal_dual[nx:]


from quadprog import solve_qp
x,f,_,_,mult,_ = solve_qp(qp.Q,-qp.q,qp.A.T,-qp.b,nc)
assert( np.allclose(x_opt,x) )
assert( np.allclose(mult_opt,-mult) )

### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class LocalTest(unittest.TestCase):
    '''
    If you have QuadProg, check the KKT solution against it.
    '''
    def test_logs(self): 
        try:
            from quadprog import solve_qp
            x,f,_,_,mult,_ = solve_qp(qp.Q,-qp.q,qp.A.T,-qp.b,nc)
            assert( np.allclose(x_opt,x) )
            assert( np.allclose(mult_opt,-mult) )
        except ModuleNotFoundError:
            print("You don't have QuadProg")

if __name__ == "__main__":
    LocalTest().test_logs()

