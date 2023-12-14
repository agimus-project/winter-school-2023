'''
This script build a TSID-like problem, shape it as a QP and save the
result in a pickle dict with numpy.

decide:
        qddot, tau, f

minimizing:
        sum(J qddot + gamma - PD) for tasks COM, posture and gripper

so that:

        M qddot + b = S' tau + Jc' f
        Jc qddot + gamma_c = 0


'''


import pinocchio as pin
import example_robot_data as robex
import numpy as np
np.set_printoptions(precision=3, linewidth=10000, suppress=True)

r=robex.load('talos')

rmodel=r.model
rdata=rmodel.createData()
q=r.q0.copy()
v=np.zeros(rmodel.nv)
v=np.random.rand(rmodel.nv)*.2-.1

pin.computeAllTerms(rmodel,rdata,q,v)

COMREF = np.array([ 0,0,1. ])
GRIPPERREF = pin.SE3(np.eye(3),np.array([ .3,-.5,1.5 ]))

M=rdata.M
b=rdata.nle
S=np.eye(rmodel.nv)[6:,:]

# contacts
feetNames = [ 'leg_left_sole_fix_joint', 'leg_right_sole_fix_joint' ]
feetId = [ rmodel.getFrameId(n) for n in feetNames ]
Jcs = [ pin.getFrameJacobian(rmodel,rdata,fid,pin.WORLD) for fid in feetId ]
Jc = np.vstack(Jcs)
acs = [ pin.getFrameAcceleration(rmodel,rdata,fid,pin.WORLD).vector for fid in feetId ]
ac = np.hstack(acs)

# tasks
# - task COM
J1 = pin.jacobianCenterOfMass(rmodel,rdata,q)
gamma1 = rdata.vcom[0]
acc1ref = 10*(COMREF-rdata.com[0])
# - task gripper
gripperId = rmodel.getFrameId('gripper_right_fingertip_3_link')
J2 = pin.getFrameJacobian(rmodel,rdata,gripperId,pin.LOCAL)
gamma2 = pin.getFrameAcceleration(rmodel,rdata,gripperId,pin.LOCAL).vector
acc2ref = 5*pin.log(rdata.oMf[gripperId].inverse()*GRIPPERREF).vector
# - posture task
J3 = np.eye(rmodel.nv)
gamma3 = np.zeros(rmodel.nv)
acc3ref = -.1*v
# - TOTAL
Jt = np.vstack([J1,J2,J3])
gammat = np.concatenate([ gamma1,gamma2,gamma3 ])
acct = np.concatenate([ acc1ref,acc2ref,acc3ref ])

# Decision variables x=[a,tau,f]
na = rmodel.nv
ntau = na-6
nf = 12
nx = na+ntau+nf

# Constraints
qp_A = np.block([ [ M,-S.T, -Jc.T ], [ Jc, np.zeros([nf,ntau]), np.zeros([nf,nf]) ] ])
qp_b = np.concatenate([ -b, -ac ])
qp_Q = np.zeros([nx,nx])
qp_q = np.zeros(nx)

for J,g,biais in [ [J1,gamma1,acc1ref],[J2,gamma2,acc2ref],[J3,gamma3,acc3ref] ]:
    qp_Q[:na,:na] += J.T@J
    qp_q[:na] += J.T@(biais-g)

np.save(open(f"/tmp/qp_tsid.npy", "wb"),
            {
                'Q': qp_Q, 'q': qp_q, 'A': qp_A, 'b': qp_b 
            })
print("Saved")

