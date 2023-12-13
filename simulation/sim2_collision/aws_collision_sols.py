import hppfcl
import example_robot_data as erd
import numpy as np
import pinocchio as pin

def make_two_objects_touch(shape1: hppfcl.CollisionObject, shape2: hppfcl.CollisionObject):
    colreq = hppfcl.CollisionRequest()
    colreq.security_margin = 100
    colres = hppfcl.CollisionResult()
    hppfcl.collide(shape1, shape2, colreq, colres)

    t2 = shape2.getTranslation()
    contact: hppfcl.Contact = colres.getContact(0)
    cp1 = contact.getNearestPoint1()
    cp2 = contact.getNearestPoint2()
    v = cp2 - cp1
    new_t2 = t2 - v
    shape2.setTranslation(new_t2)

def find_free_collision_configuration(model, data, geom_model, geom_data):
    MAX_ITERATIONS = 1000
    for i in range(MAX_ITERATIONS):
        q = pin.randomConfiguration(model)
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
        res = pin.computeCollisions(geom_model, geom_data, stop_at_first_collision=True)
        viz.display(q)
        if not res:
            print(f"Found in {i} iterations!")
            break
    return q

def create_panda():
    robot = erd.load("panda")
    model = robot.model
    geom_model = robot.collision_model
    visual_model = robot.visual_model
    geom_data = geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    return model, geom_model, visual_model
