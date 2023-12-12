import hppfcl
import pinocchio as pin
import numpy as np
import meshcat
from typing import List
from utils_render import load_primitive, meshcat_material, load_convex, create_visualizer, AgimusScene
import time

if __name__ == "__main__":
    # Create some shapes
    scene = AgimusScene()
    pin.seed(0)
    np.random.seed(0)

    N = 50
    for i in range(N):
        shape = hppfcl.Ellipsoid(0.05, 0.15, 0.2)
        M = pin.SE3.Random()
        color = np.random.rand(4)
        color[3] = 1
        scene.register_object(shape, M, color)

        shape = hppfcl.Capsule(0.1, 0.2)
        M = pin.SE3.Random()
        color = np.random.rand(4)
        color[3] = 1
        scene.register_object(shape, M, color)

        shape = load_convex("./assets/mesh.stl")
        M = pin.SE3.Random()
        color = np.random.rand(4)
        color[3] = 1
        scene.register_object(shape, M, color)

    # Add walls
    transparent_color = np.ones(4)
    transparent_color[3] = 0.
    wall_size = 4.0
    n_walls = 6
    # - Lower wall
    M = pin.SE3.Identity()
    M.translation = np.array([0., 0., -wall_size])/2
    # shape = hppfcl.Halfspace(np.array([0.0, 0.0, 1.0]), 0.)
    shape = hppfcl.Box(wall_size, wall_size, 0.5)
    scene.register_object(shape, M, transparent_color)
    # - Upper wall
    M = pin.SE3.Identity()
    M.translation = np.array([0., 0., wall_size])/2
    # shape = hppfcl.Halfspace(np.array([0.0, 0.0, -1.0]), 0.)
    shape = hppfcl.Box(wall_size, wall_size, 0.5)
    scene.register_object(shape, M, transparent_color)
    # - Side walls
    M = pin.SE3.Identity()
    M.translation = np.array([-wall_size, 0., 0.])/2
    # shape = hppfcl.Halfspace(np.array([1.0, 0.0, 0.0]), 0.)
    shape = hppfcl.Box(0.5, wall_size, wall_size)
    scene.register_object(shape, M, transparent_color)
    M = pin.SE3.Identity()
    M.translation = np.array([wall_size, 0., 0.])/2
    # shape = hppfcl.Halfspace(np.array([-1.0, 0.0, 0.0]), 0.)
    shape = hppfcl.Box(0.5, wall_size, wall_size)
    scene.register_object(shape, M, transparent_color)
    # - Front walls
    M = pin.SE3.Identity()
    M.translation = np.array([0., -wall_size, 0.])/2
    shape = hppfcl.Halfspace(np.array([0.0, 1.0, 0.0]), 0.)
    shape = hppfcl.Box(wall_size, 0.5, wall_size)
    scene.register_object(shape, M, transparent_color)
    M = pin.SE3.Identity()
    M.translation = np.array([0., wall_size, 0.])/2
    # shape = hppfcl.Halfspace(np.array([0.0, -1.0, 0.0]), 0.)
    shape = hppfcl.Box(wall_size, 0.5, wall_size)
    scene.register_object(shape, M, transparent_color)

    # Initialize scene renderer
    scene.init_renderer()

    # render the scene
    scene.render_scene()

    # SIMPLE KINEMATIC SIMULATOR
    num_collision_objects = len(scene.collision_objects)
    print(num_collision_objects)
    num_possible_collision_pairs = (int)(len(scene.collision_objects)*(len(scene.collision_objects) - 1)/2)
    print("Number of possible collision pairs: ", num_possible_collision_pairs)
    # A callback function which will collect broad phase collisions

    # Starting velocities of each shape
    velocities = []
    for i in range(num_collision_objects - n_walls):
        v = np.random.rand(6) * 0.25
        # v = np.zeros(6)
        # v[2] = -1.0 * 0.1
        # v[3:] = np.zeros(3)
        velocities.append(v)
    # Walls velocities
    for i in range(n_walls):
        velocities.append(np.zeros(6))

    # Before calling any broad phase check, need to update the aabbs of shapes
    # that have transformed

    # Collision request and result needed for the narrow phase
    colreq = hppfcl.CollisionRequest()
    colres = hppfcl.CollisionResult()
    # Number of time steps
    input()
    T = 1000
    dt = 0.05
    callback = hppfcl.CollisionCallBackCollect(num_possible_collision_pairs)
    start = time.time()
    # Create a manager and register objects
    # manager = hppfcl.NaiveCollisionManager()
    manager = hppfcl.DynamicAABBTreeCollisionManager()
    # manager = hppfcl.SaPCollisionManager()
    for shape in scene.collision_objects:
        shape.computeAABB()
        manager.registerObject(shape)
    for t in range(T):
        # Render the current scene
        scene.render_scene()

        # Actual broad phase check
        # 1 - update the internal representation of the AABB tree
        # manager.clear()
        for shape in scene.collision_objects:
            shape.computeAABB()
            # manager.registerObject(shape)
        manager.update()

        # 2 - run the recursive aabbs collision check
        callback.init()
        # print("Iteration ", t)
        manager.collide(callback)
        # print("Broad phase done.")
        for i in range(0, num_collision_objects-1):
            for j in range(i+1, num_collision_objects):
                if i < num_collision_objects - n_walls and j < num_collision_objects - n_walls:
                    shape1 = scene.collision_objects[i]
                    shape2 = scene.collision_objects[j]
                    # print("Before callback check")
                    # if callback.exist(shape1, shape2):
                    if callback.exist(shape1, shape2):
                        # Launch the narrow phase
                        colres.clear()
                        is_colliding = hppfcl.collide(shape1, shape2, colreq, colres)
                        if (is_colliding):
                            v1 = velocities[i]
                            v2 = velocities[j]

                            contact: hppfcl.Contact = colres.getContacts()[0]

                            new_v1 = np.zeros(6)
                            new_v1[3:] = velocities[i][3:]
                            new_v1[:3] = - np.linalg.norm(v1[:3]) * contact.normal

                            new_v2 = np.zeros(6)
                            new_v2[3:] = velocities[j][3:]
                            new_v2[:3] = np.linalg.norm(v2[:3]) * contact.normal

                            if i < num_collision_objects - 6:
                                velocities[i] = new_v1
                            if j < num_collision_objects - 6:
                                velocities[j] = new_v2

        for i in range(num_collision_objects - n_walls):
            M = scene.collision_objects[i].getTransform()
            v1 = np.zeros(6)
            v2 = np.zeros(6)
            v1[:3] = velocities[i][:3]
            v2[3:] = velocities[i][3:]
            M = pin.exp6(v1*dt) * M * pin.exp6(v2*dt)
            scene.collision_objects[i].setTransform(M)
        input()

    print("Simulation done")
    print(time.time() - start)
    input()


