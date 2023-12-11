import pinocchio as pin
import numpy as np
from hppfcl import Sphere, Halfspace, Box

def computeContactProblem(model, data, geom_model, geom_data, q, v, tau, dt):
    """Computes the contact Jacobian, the Delassus matrix and 
    collect the associated coefficients of friction.
    """
    pin.crba(model, data, q)
    vf = v + dt * pin.aba(model, data, q, v, tau)
    nc = 0
    J, g, Del, R, err = None, None, None, None, None
    mus = []
    els = []
    contact_models = []
    contact_datas = []
    contact_points = []
    for i, res in enumerate(geom_data.collisionResults):
        if res.isCollision():
            geom_id1, geom_id2 = (
                geom_model.collisionPairs[i].first,
                geom_model.collisionPairs[i].second,
            )
            mu_i = geom_model.frictions[i]
            el_i = geom_model.elasticities[i]
            mus += [mu_i]
            els += [el_i]
            joint_id1 = geom_model.geometryObjects[geom_id1].parentJoint
            joint_id2 = geom_model.geometryObjects[geom_id2].parentJoint
            contacts = res.getContacts()
            joint_placement_1 = data.oMi[joint_id1]
            joint_placement_2 = data.oMi[joint_id2]
            for contact in contacts:
                pos_i = contact.pos
                normal_i = contact.normal
                ex_i, ey_i = complete_orthonormal_basis(contact.normal, joint_placement_1)
                ex_i = np.expand_dims(ex_i, axis=1)
                ey_i = np.expand_dims(ey_i, axis=1)
                normal_i = np.expand_dims(contact.normal, axis=1)
                R_i = np.concatenate((ex_i, ey_i, normal_i), axis=1)
                R_i1 = np.dot(joint_placement_1.rotation.T, R_i)
                R_i2 = np.dot(joint_placement_2.rotation.T, R_i)
                pos_i1 = joint_placement_1.rotation.T @ (
                    pos_i - joint_placement_1.translation
                )
                pos_i2 = joint_placement_2.rotation.T @ (
                    pos_i - joint_placement_2.translation
                )
                placement_i1 = pin.SE3(R_i1, pos_i1)
                placement_i2 = pin.SE3(R_i2, pos_i2)
                contact_model_i = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_3D,
                    model,
                    joint_id2,
                    placement_i2,
                    joint_id1,
                    placement_i1,
                    pin.ReferenceFrame.LOCAL,
                )
                contact_models += [contact_model_i]
                contact_data_i = contact_model_i.createData()
                contact_datas += [contact_data_i]
                contact_points += [pos_i]
                err_i = np.array([0.0, 0.0, contact.penetration_depth])
                if R is None:
                    R = R_i
                    err = err_i
                else:
                    R = np.concatenate((R, R_i), axis=1)
                    err = np.concatenate((err, err_i), axis=0)
            nc += len(contacts)
    if nc > 0:
        chol = pin.ContactCholeskyDecomposition(model, contact_models)
        chol.compute(model, data, contact_models, contact_datas, 1e-9)
        Del = chol.getInverseOperationalSpaceInertiaMatrix()
        J = pin.getConstraintsJacobian(model, data, contact_models, contact_datas)
    if J is not None:
        g = J @ vf
    return J, vf, Del, g, mus

def complete_orthonormal_basis(ez, joint_placement):
    """When a normal is provided, this function 
    returns two tangent vectors to form an orthonormal 3D basis.
    """
    ex = joint_placement.rotation[:,0]
    if np.abs(np.dot(ex,ez))>0.999:
        ex = joint_placement.rotation[:,1]
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    return ex, ey


def create_cubes(length=[0.2], mass=[1.0], mu=0.9, el=0.1):
    """Creates the pinocchio model, geommodel and visualmodel 
    of a scene containing cubes and a floor.

    Args:
        length (list, optional): a list containing the lengthes of the cubes. Defaults to [0.2].
        mass (list, optional): a list containing the masses of the cubes. Defaults to [1.0].
        mu (float, optional): a llist containing the coefficient of friction used for every contact. Defaults to 0.9.
        el (float, optional): a list containing the elasticity used for every contact. Defaults to 0.1.

    Returns:
        model, geommodel, visualmodel, data, geomdata, visualdata, actuation
    """
    assert len(length) == len(mass) or len(length) == 1 or len(mass) == 1
    N = max(len(length), len(mass))
    if len(length) == 1:
        length = length * N
    if len(mass) == 1:
        mass = mass * N
    rmodel = pin.Model()
    rgeomModel = pin.GeometryModel()
    rgeomModel.frictions = []
    rgeomModel.elasticities = []

    n = np.array([0.0, 0.0, 1])
    plane_shape = Halfspace(n,0)
    T = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = rgeomModel.addGeometryObject(plane)

    ball_ids = []
    cube_joints = []
    for n_cube in range(N):
        a = length[n_cube]
        m = mass[n_cube]
        freeflyer = pin.JointModelFreeFlyer()
        jointCube = rmodel.addJoint(
            0, freeflyer, pin.SE3.Identity(), "joint1_" + str(n_cube)
        )
        cube_joints += [jointCube]
        M = pin.SE3(np.eye(3), np.matrix([0.0, 0.0, 0.0]).T)
        rmodel.appendBodyToJoint(jointCube, pin.Inertia.FromBox(m, a, a, a), M)
        r = np.array([a / 4, a / 4, a / 4])

        # add balls to cube

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, -a / 2, -a / 2]).T
        ball_shape1 = Sphere(a / 50)
        geom_ball1 = pin.GeometryObject(
            "ball1_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape1
        )
        geom_ball1.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball1_id = rgeomModel.addGeometryObject(geom_ball1)
        col_pair1 = pin.CollisionPair(plane_id, ball1_id)
        rgeomModel.addCollisionPair(col_pair1)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, a / 2, -a / 2]).T
        ball_shape2 = Sphere(a / 50)
        geom_ball2 = pin.GeometryObject(
            "ball2_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape2
        )
        geom_ball2.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball2_id = rgeomModel.addGeometryObject(geom_ball2)
        col_pair2 = pin.CollisionPair(plane_id, ball2_id)
        rgeomModel.addCollisionPair(col_pair2)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, a / 2, a / 2]).T
        ball_shape3 = Sphere(a / 50)
        geom_ball3 = pin.GeometryObject(
            "ball3_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape3
        )
        geom_ball3.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball3_id = rgeomModel.addGeometryObject(geom_ball3)
        col_pair3 = pin.CollisionPair(plane_id, ball3_id)
        rgeomModel.addCollisionPair(col_pair3)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, -a / 2, a / 2]).T
        ball_shape4 = Sphere(a / 50)
        geom_ball4 = pin.GeometryObject(
            "ball4_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape4
        )
        geom_ball4.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball4_id = rgeomModel.addGeometryObject(geom_ball4)
        col_pair4 = pin.CollisionPair(plane_id, ball4_id)
        rgeomModel.addCollisionPair(col_pair4)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, -a / 2, -a / 2]).T
        ball_shape5 = Sphere(a / 50)
        geom_ball5 = pin.GeometryObject(
            "ball5_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape5
        )
        geom_ball5.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball5_id = rgeomModel.addGeometryObject(geom_ball5)
        col_pair5 = pin.CollisionPair(plane_id, ball5_id)
        rgeomModel.addCollisionPair(col_pair5)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, a / 2, -a / 2]).T
        ball_shape6 = Sphere(a / 50)
        geom_ball6 = pin.GeometryObject(
            "ball6_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape6
        )
        geom_ball6.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball6_id = rgeomModel.addGeometryObject(geom_ball6)
        col_pair6 = pin.CollisionPair(plane_id, ball6_id)
        rgeomModel.addCollisionPair(col_pair6)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, a / 2, a / 2]).T
        ball_shape7 = Sphere(a / 50)
        geom_ball7 = pin.GeometryObject(
            "ball7_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape7
        )
        geom_ball7.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball7_id = rgeomModel.addGeometryObject(geom_ball7)
        col_pair7 = pin.CollisionPair(plane_id, ball7_id)
        rgeomModel.addCollisionPair(col_pair7)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, -a / 2, a / 2]).T
        ball_shape8 = Sphere(a / 50)
        geom_ball8 = pin.GeometryObject(
            "ball8_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape8
        )
        geom_ball8.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball8_id = rgeomModel.addGeometryObject(geom_ball8)
        col_pair8 = pin.CollisionPair(plane_id, ball8_id)
        rgeomModel.addCollisionPair(col_pair8)
        rgeomModel.frictions += [mu]
        rgeomModel.elasticities += [el]
        for id in ball_ids:
            col_pair = pin.CollisionPair(id, ball1_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball2_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball3_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball4_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball5_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball6_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball7_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball8_id)
            rgeomModel.addCollisionPair(col_pair)
            rgeomModel.frictions += [mu] * 8
            rgeomModel.elasticities += [el] * 8
        ball_ids += [
            ball1_id,
            ball2_id,
            ball3_id,
            ball4_id,
            ball5_id,
            ball6_id,
            ball7_id,
            ball8_id,
        ]

    rmodel.qref = pin.neutral(rmodel)
    rmodel.qinit = rmodel.qref.copy()
    rmodel.qinit[2] += a / 2 + a/50
    for n_cube in range(1, N):
        a = length[n_cube]
        rmodel.qinit[7 * n_cube + 1] = rmodel.qinit[7 * (n_cube - 1) + 1] + a + 0.03
        rmodel.qinit[7 * n_cube + 2] += a / 2
    data = rmodel.createData()
    rgeom_data = rgeomModel.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.eye(rmodel.nv)
    visual_model = rgeomModel.copy()
    for n_cube in range(N):
        R = pin.utils.eye(3)
        t = np.matrix([0.0, 0.0, 0.0]).T
        box_shape = Box(a, a, a)
        geom_box = pin.GeometryObject(
            "box_" + str(n_cube), cube_joints[n_cube], cube_joints[n_cube], pin.SE3(R, t), box_shape
        )
        geom_box.meshColor = np.array([0.0, 0.0, 1.0, 0.6])
        box_id = visual_model.addGeometryObject(geom_box)  # only for visualisation
    visual_data = visual_model.createData()
    return (
        rmodel,
        rgeomModel,
        visual_model,
        data,
        rgeom_data,
        visual_data,
        actuation,
    )