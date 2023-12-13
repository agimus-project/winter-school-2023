import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import unittest

nx = 3
nc = 1

Q = np.eye(nx)
q = np.ones(nx)*3
b = np.ones(1)
A = np.eye(nx)[:nc]

K = np.block([ [Q, A.T], [A, np.zeros([nc,nc]) ]])
k = np.concatenate([ -q, -b ])

primal_dual = npla.inv(K) @ k
x_opt = primal_dual[:nx]
mult_opt = primal_dual[nx:]

### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class LocalTest(unittest.TestCase):
    def test_logs(self): 
        try:
            from quadprog import solve_qp
            x,f,_,_,mult,_ = solve_qp(Q,-q,A.T,-b)
            assert( np.allclose(x_opt,x) )
            assert( np.allclose(mult_opt,-mult) )
        except ModuleNotFoundError:
            print("You don't have QuadProg")

if __name__ == "__main__":
    LocalTest().test_logs()


