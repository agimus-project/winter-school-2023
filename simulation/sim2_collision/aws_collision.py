import hppfcl
import pinocchio as pin
import numpy as np
from utils_render import create_complex_scene

def reset_objects_placements(scene, transforms):
    for s in range(len(scene.collision_objects)):
        scene.collision_objects[s].setTransform(transforms[s])

# The scene is made of a box (6 walls) with a bunch of objects inside
shapes, transforms, scene = create_complex_scene()
n_walls = 6

# Initialize scene renderer
scene.init_renderer()

# render the scene
scene.render_scene()

num_collision_objects = len(scene.collision_objects)
print("Number of collision objects", num_collision_objects)
num_possible_collision_pairs = (int)(len(scene.collision_objects)*(len(scene.collision_objects) - 1)/2)
print("Number of possible collision pairs: ", num_possible_collision_pairs)

# Starting velocities of each shape
velocities = []
for i in range(num_collision_objects - n_walls):
    v = np.random.rand(6) * 0.25
    velocities.append(v)
# We don't want the walls to move
for i in range(n_walls):
    velocities.append(np.zeros(6))

# Simulation loop
# You can increase the time horizon of the simulation
T = 100
dt = 0.1 # timestep

# Collision request and result needed for the narrow phase
colreq = hppfcl.CollisionRequest()
colres = hppfcl.CollisionResult()
for t in range(T):
    # Render the current scene
    scene.render_scene()

    # Loop through all collision pairs
    for i in range(0, num_collision_objects-1):
        for j in range(i+1, num_collision_objects):
            # If both object are walls, we don't really care about checking their collision
            if i < num_collision_objects - n_walls or j < num_collision_objects - n_walls:
                colres.clear()
                is_colliding = hppfcl.collide(scene.collision_objects[i], scene.collision_objects[j], colreq, colres)
                if (is_colliding):
                    v1 = velocities[i]
                    v2 = velocities[j]

                    contact: hppfcl.Contact = colres.getContact(0)

                    new_v1 = np.zeros(6)
                    new_v1[3:] = velocities[i][3:]
                    new_v1[:3] = -np.linalg.norm(v1[:3]) * contact.normal

                    new_v2 = np.zeros(6)
                    new_v2[3:] = velocities[j][3:]
                    new_v2[:3] = np.linalg.norm(v2[:3]) * contact.normal

                    if i < num_collision_objects - n_walls:
                        velocities[i] = new_v1
                    if j < num_collision_objects - n_walls:
                        velocities[j] = new_v2

    # Update the placements of the shapes based on their velocities
    for i in range(num_collision_objects - n_walls):
        M = scene.collision_objects[i].getTransform()
        # I will be using the first 3 elements of velocities[i] to apply a linear velocity to M
        # in the world frame.
        # And I will be using the last 3 elements of velocities[i] to apply an angular velocity to M
        # in its local frame.
        v1 = np.zeros(6)
        v2 = np.zeros(6)
        v1[:3] = velocities[i][:3]
        v2[3:] = velocities[i][3:]
        M = pin.exp6(v1*dt) * M * pin.exp6(v2*dt)
        scene.collision_objects[i].setTransform(M)
