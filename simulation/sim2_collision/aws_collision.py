import hppfcl
import pinocchio as pin
import numpy as np
import meshcat as mc
from typing import List
from .utils_render import loadPrimitive

# TP summary:
# - Presentation of hppfcl: shapes, collision objects, SE3
# - Present the structure with a scene, show a simple example with 2 shapes
# - Actually start with the narrow phase because only 2 shapes
# - Create a complex scene with walls etc
# - Show the broadphase
# - Put everything together to have a kinematic simulator

# BROADPHASE COLLISION MANAGER:
# How it works:
# 1 - Create a list of CollisionObject (Q: is it ptr or owned by manager?)
# 2 - Create a manager and register the shapes
#     The manager creates a N^2 set which maps every pair of (shapei, shapej) (hash table using the address of the collision objects).
#     When collide is called, this set is populated with positive results (I guess?)
# 3 - For each timestep:
#       - call computeBV on shapes which transforms have been changed
#       - call update of the tree to recompute the BVH tree
#       - call collide
#
# Q: where are the positive/negative collision results stored?

# @TODO: create a class Scene which has a meshcat renderer and a broadphase
# manager (which will store the hppfcl objects)
# Or simply a function which renders a broadphase manager...
# This renderer will render the shapes but also their aabb.
# For positive narrow phase collision, it should draw the points and the normals.

# @TODO: create a scene which is a box (defined by 4 hyperplanes) and put shapes inside it.
# To check if a shape is inside, use a narrowphase and and translate by separating vector if needed.
# Also register these hyperplanes in the manager.

SHAPE_TYPES = [hppfcl.GEOM_ELLIPSOID,
               hppfcl.GEOM_CONVEX,
               hppfcl.GEOM_CYLINDER]

def create_visualizer(grid: bool=False, axes: bool=False) -> mc.Visualizer:
    vis = mc.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis.delete()
    if not grid:
        vis["/Grid"].set_property("visible", False)
    if not axes:
        vis["/Axes"].set_property("visible", False)
    return vis

def load_convex(path: str) -> hppfcl.ConvexBase:
    shape: hppfcl.ConvexBase
    loader = hppfcl.MeshLoader()
    mesh_: hppfcl.BVHModelBase = loader.load(path)
    mesh_.buildConvexHull(True, "Qt")
    shape = mesh_.convex
    return shape

def load_shape(shape_type) -> hppfcl.ShapeBase:
    shape: hppfcl.ShapeBase = None
    if shape_type == SHAPE_TYPES[0]:   # Ellipsoids
        shape = hppfcl.Ellipsoid(np.ones(3)) # @TODO: random
    elif shape_type == SHAPE_TYPES[1]: # Convex
        shape = load_convex("./sim2_collision/assets/mesh.stl")
    elif shape_type == SHAPE_TYPES[2]: # Cylinder
        shape = hppfcl.Cylinder(1, 2) # @TODO random
    else:
        raise Exception("Can't load this shape in this application.")
    return shape

# def draw_shape(vis: meshcat.Visualizer, shape: hppfcl.ShapeBase,
#                name: str, M: pin.SE3, color: np.ndarray, render_faces=True):
#     if isinstance(shape, hppfcl.Ellipsoid):
#         renderEllipsoid(vis, shape.radii, name, M, color)
#     if isinstance(shape, hppfcl.Sphere):
#         renderSphere(vis, shape.radius, name, M, color)
#     if isinstance(shape, hppfcl.ConvexBase):
#         renderConvex(vis, shape, name, M, color, render_faces)
#         pass

# def render_scene(vis: mc.Visualizer, shapes: List[hppfcl.CollisionGeometry]):
    # for s, shape in enumerate(shapes):

def render_scene(vis: mc.Visualizer, scene: AgimusScene):
    pass

class AgimusScene:
    collision_objects: List[hppfcl.CollisionObject]
    vis: mc.Visualizer
    mc_shapes: List[mc.geometry.ShapeBase]

    def __init__(self):
        self.vis = create_visualizer(False, False)
        pass

    def register_object(self, shape: hppfcl.ShapeBase, M: pin.SE3):
        self.shapes.append(hppfcl.CollisionObject(shape, M))
        self.mc_shapes.append(loadPrimitive(shape))

    def render_scene():
        pass


if __name__ == "__main__":
    # Create some shapes
    shapes: List[hppfcl.CollisionObject] = []
    transforms: List[pin.SE3] = []

    sphere = hppfcl.Sphere(0.1)
    transforms.append(pin.SE3.Random())
    shapes.append(hppfcl.CollisionObject(sphere, transforms[0]))

    cylinder = hppfcl.Cylinder(0.1, 0.2)
    transforms.append(pin.SE3.Random())
    shapes.append(hppfcl.CollisionObject(cylinder, transforms[1]))

    convex = load_convex("./sim2_collision/assets/mesh.stl")
    transforms.append(pin.SE3.Random())
    shapes.append(hppfcl.CollisionObject(convex, transforms[2]))

    transforms.append(pin.SE3.Random())
    shapes.append(hppfcl.CollisionObject(convex, transforms[3]))

    # @TODO Add walls
    walls: List[hppfcl.Box]

    # Create a manager and register objects
    manager = hppfcl.DynamicAABBTreeCollisionManager()
    for shape in shapes:
        manager.registerObject(shape)
    # @TODO Register the walls

    num_possible_collision_pairs = (int)(len(shapes)*(len(shapes) - 1)/2)
    print("Number of possible collision pairs: ", num_possible_collision_pairs)
    callback = hppfcl.CollisionCallBackCollect(num_possible_collision_pairs)

    # Before calling any broad phase check, need to update the aabbs of shapes
    # that have transformed
    for shape in shapes:
        shape.computeAABB()

    # Actual broad phase check
    # 1 - update the internal representation of the AABB tree
    manager.update()
    # 2 - run the recursive aabbs collision check
    manager.collide(callback)
    if callback.isPairInCollision(shapes[0], shapes[1]):
        pass
    for i, shape1 in enumerate(shapes):
        for j, shape2 in enumerate(shapes):
            print(f"Are shapes {i, j} in broad phase collision? ", callback.isPairInCollision(shapes[0], shapes[1]))

    # Create meshcat visualizer
    vis: mc.Visualizer = create_visualizer(False, False)

    # Very simple kinematic simulation
    # v: np.ndarray = np.random.rand(6)
    # dt: float = 0.01
    # for i in range(100):
    #     M1 = M1 * pin.exp6(v * dt)
    #     # print(M1)
    #     shape.setTransform(M1)

