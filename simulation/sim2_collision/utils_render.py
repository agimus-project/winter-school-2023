import hppfcl
import numpy as np
import meshcat
import meshcat.geometry as mg
import meshcat.transformations as tf
import pinocchio as pin
from distutils.version import LooseVersion
import warnings
from typing import Any, Dict, Union, List
MsgType = Dict[str, Union[str, bytes, bool, float, 'MsgType']]

def npToTTuple(M):
    L = M.tolist()
    for i in range(len(L)):
        L[i] = tuple(L[i])
    return tuple(L)

def npToTuple(M):
    if len(M.shape) == 1:
        return tuple(M.tolist())
    if M.shape[0] == 1:
        return tuple(M.tolist()[0])
    if M.shape[1] == 1:
        return tuple(M.T.tolist()[0])
    return npToTTuple(M)


def load_primitive(geom: hppfcl.ShapeBase):

    import meshcat.geometry as mg

    # Cylinders need to be rotated
    basic_three_js_transform = np.array([[1.,  0.,  0.,  0.],
                  [0.,  0., -1.,  0.],
                  [0.,  1.,  0.,  0.],
                  [0.,  0.,  0.,  1.]])
    RotatedCylinder = type("RotatedCylinder", (mg.Cylinder,), {"intrinsic_transform": lambda self: basic_three_js_transform })

    # Cones need to be rotated

    if isinstance(geom, hppfcl.Capsule):
        if hasattr(mg, 'TriangularMeshGeometry'):
            obj = createCapsule(2. * geom.halfLength, geom.radius)
        else:
            obj = RotatedCylinder(2. * geom.halfLength, geom.radius)
    elif isinstance(geom, hppfcl.Cylinder):
        obj = RotatedCylinder(2. * geom.halfLength, geom.radius)
    elif isinstance(geom, hppfcl.Cone):
        obj = RotatedCylinder(2. * geom.halfLength, 0, geom.radius, 0)
    elif isinstance(geom, hppfcl.Box):
        obj = mg.Box(npToTuple(2. * geom.halfSide))
    elif isinstance(geom, hppfcl.Sphere):
        obj = mg.Sphere(geom.radius)
    elif isinstance(geom, hppfcl.Plane):
        To = np.eye(4)
        To[:3, 3] = geom.d * geom.n
        TranslatedPlane = type("TranslatedPlane", (mg.Plane,), {"intrinsic_transform": lambda self: To})
        sx = 10
        sy = 10
        obj = TranslatedPlane(sx, sy)
    elif isinstance(geom, hppfcl.Ellipsoid):
        obj = mg.Ellipsoid(geom.radii)
    elif isinstance(geom, (hppfcl.Plane,hppfcl.Halfspace)):
        plane_transform : pin.SE3 = pin.SE3.Identity()
        # plane_transform.translation[:] = geom.d # Does not work
        plane_transform.rotation = pin.Quaternion.FromTwoVectors(pin.ZAxis,geom.n).toRotationMatrix()
        TransformedPlane = type("TransformedPlane", (Plane,), {"intrinsic_transform": lambda self: plane_transform.homogeneous })
        obj = TransformedPlane(1000,1000)
    elif isinstance(geom, hppfcl.ConvexBase):
        obj = loadMesh(geom)
    else:
        msg = "Unsupported geometry type for (%s)" % (type(geom) )
        warnings.warn(msg, category=UserWarning, stacklevel=2)
        obj = None

    return obj

def loadMesh(mesh):
    if isinstance(mesh,(hppfcl.Convex,hppfcl.BVHModelBase)):
        if isinstance(mesh,hppfcl.BVHModelBase):
            num_vertices = mesh.num_vertices
            num_tris = mesh.num_tris

            call_triangles = mesh.tri_indices
            call_vertices = mesh.vertices

        elif isinstance(mesh,hppfcl.Convex):
            num_vertices = mesh.num_points
            num_tris = mesh.num_polygons

            call_triangles = mesh.polygons
            call_vertices = mesh.points

        faces = np.empty((num_tris,3),dtype=int)
        for k in range(num_tris):
            tri = call_triangles(k)
            faces[k] = [tri[i] for i in range(3)]

        if LooseVersion(hppfcl.__version__) >= LooseVersion("1.7.7"):
            vertices = call_vertices()
        else:
            vertices = np.empty((num_vertices,3))
            for k in range(num_vertices):
                vertices[k] = call_vertices(k)

        vertices = vertices.astype(np.float32)

    if num_tris > 0:
        mesh = mg.TriangularMeshGeometry(vertices, faces)
    else:
        mesh = mg.Points(
                    mg.PointsGeometry(vertices.T, color=np.repeat(np.ones((3,1)),num_vertices,axis=1)),
                    mg.PointsMaterial(size=0.002))

    return mesh

def createCapsule(length, radius, radial_resolution = 30, cap_resolution = 10):
    nbv = np.array([max(radial_resolution, 4), max(cap_resolution, 4)])
    h = length
    r = radius
    position = 0
    vertices = np.zeros((nbv[0] * (2 * nbv[1]) + 2, 3))
    for j in range(nbv[0]):
        phi = (( 2 * np.pi * j) / nbv[0])
        for i in range(nbv[1]):
            theta = ((np.pi / 2 * i) / nbv[1])
            vertices[position + i, :] = np.array([np.cos(theta) * np.cos(phi) * r,
                                               np.cos(theta) * np.sin(phi) * r,
                                               -h / 2 - np.sin(theta) * r])
            vertices[position + i + nbv[1], :] = np.array([np.cos(theta) * np.cos(phi) * r,
                                                        np.cos(theta) * np.sin(phi) * r,
                                                        h / 2 + np.sin(theta) * r])
        position += nbv[1] * 2
    vertices[-2, :] = np.array([0, 0, -h / 2 - r])
    vertices[-1, :] = np.array([0, 0, h / 2 + r])
    indexes = np.zeros((nbv[0] * (4 * (nbv[1] - 1) + 4), 3))
    index = 0
    stride = nbv[1] * 2
    last = nbv[0] * (2 * nbv[1]) + 1
    for j in range(nbv[0]):
        j_next = (j + 1) % nbv[0]
        indexes[index + 0] = np.array([j_next * stride + nbv[1], j_next * stride, j * stride])
        indexes[index + 1] = np.array([j * stride + nbv[1], j_next * stride + nbv[1], j * stride])
        indexes[index + 2] = np.array([j * stride + nbv[1] - 1, j_next * stride + nbv[1] - 1, last - 1])
        indexes[index + 3] = np.array([j_next * stride + 2 * nbv[1] - 1, j * stride + 2 * nbv[1] - 1, last])
        for i in range(nbv[1]-1):
            indexes[index + 4 + i * 4 + 0] = np.array([j_next * stride + i, j_next * stride + i + 1, j * stride + i])
            indexes[index + 4 + i * 4 + 1] = np.array([j_next * stride + i + 1, j * stride + i + 1, j * stride + i])
            indexes[index + 4 + i * 4 + 2] = np.array([j_next * stride + nbv[1] + i + 1, j_next * stride + nbv[1] + i, j * stride + nbv[1] + i])
            indexes[index + 4 + i * 4 + 3] = np.array([j_next * stride + nbv[1] + i + 1, j * stride + nbv[1] + i, j * stride + nbv[1] + i + 1])
        index += 4 * (nbv[1] - 1) + 4
    return mg.TriangularMeshGeometry(vertices, indexes)

class Plane(mg.Geometry):
    """A plane of the given width and height. 
    """
    def __init__(self, width: float, height: float, widthSegments: float = 1, heightSegments: float = 1):
        super().__init__()
        self.width = width
        self.height = height
        self.widthSegments = widthSegments
        self.heightSegments = heightSegments

    def lower(self, object_data: Any) -> MsgType:
        return {
            u"uuid": self.uuid,
            u"type": u"PlaneGeometry",
            u"width": self.width,
            u"height": self.height,
            u"widthSegments": self.widthSegments,
            u"heightSegments": self.heightSegments,
        }

def meshcat_material(r, g, b, a):
    material = mg.MeshPhongMaterial()
    material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + \
        int(b * 255)
    material.opacity = a
    return material

def create_visualizer(grid: bool=False, axes: bool=False) -> meshcat.Visualizer:
    # vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis = meshcat.Visualizer()
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

def rgbToHex(color):
    if len(color) == 4:
        c = color[:3]
        opacity = color[3]
    else:
        c = color
        opacity = 1.
    hex_color = '0x%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
    return hex_color, opacity

def renderPoint(vis: meshcat.Visualizer, point: np.ndarray, point_name: str,
                color=np.ones(4), radius_point=0.001):
    hex_color, opacity = rgbToHex(color)
    vis[point_name].set_object(mg.Sphere(radius_point), mg.MeshLambertMaterial(color=hex_color, opacity=opacity))
    vis[point_name].set_transform(tf.translation_matrix(point))

def renderLine(vis: meshcat.Visualizer, pt1: np.ndarray, pt2: np.ndarray, name: str,
               linewidth=1, color=np.array([0., 0., 0., 1.])):
    hex_color, _ = rgbToHex(color)
    points = np.hstack([pt1.reshape(-1, 1), pt2.reshape(-1, 1)]).astype(np.float32)
    vis[name].set_object(mg.Line(mg.PointsGeometry(points), mg.MeshBasicMaterial(color=hex_color, linewidth=linewidth)))

RED_COLOR = np.array([1.0, 0., 0., 1.0])
class AgimusScene:
    collision_objects: List[hppfcl.CollisionObject]
    viewer: meshcat.Visualizer
    mc_shapes: List[meshcat.geometry.Geometry]
    shape_colors: List[np.ndarray]
    _colres_idx: int

    def __init__(self):
        self.viewer = create_visualizer(False, False)
        self.clear_scene()

    def clear_scene(self):
        self.mc_shapes = []
        self.collision_objects = []
        self.shape_colors = []
        self._colres_idx = 0
        self.viewer.delete()

    def register_object(self, shape: hppfcl.ShapeBase, M: pin.SE3, shape_color=np.ones(3), transparent=False):
        shape.computeLocalAABB()
        cobj = hppfcl.CollisionObject(shape, M)
        self.collision_objects.append(cobj)
        self.mc_shapes.append(load_primitive(shape))
        color = np.ones(4)
        color[:3] = shape_color
        color[3] = 1
        if transparent:
            color[3] = 0.2
        self.shape_colors.append(color)

    def render_scene(self):
        for s, cobj in enumerate(self.collision_objects):
            M = pin.SE3(cobj.getTransform())
            shape_name = f"shape_{s}"
            if isinstance(cobj,(hppfcl.Plane, hppfcl.Halfspace)):
                T = M
                T.translation += M.rotation @ (cobj.d * cobj.n)
                T = T.homogeneous
            else:
                T = M.homogeneous

            # Update viewer configuration.
            self.viewer[shape_name].set_transform(T)

    def clear_renderer(self):
        self.init_renderer()

    def init_renderer(self):
        self.viewer.delete()
        self._colres_idx = 0
        for s, shape in enumerate(self.mc_shapes):
            shape_name = f"shape_{s}"
            self.viewer[shape_name].set_object(shape, meshcat_material(*self.shape_colors[s]))

    def visualize_separation_vector(self, colres: hppfcl.CollisionResult):
        if colres.isCollision:
            contact: hppfcl.Contact = colres.getContacts()[0]
            p1 = contact.getNearestPoint1()
            p2 = contact.getNearestPoint2()

            name = f"sep_vec_{self._colres_idx}"
            renderPoint(self.viewer, p1, name + "/p1", RED_COLOR, 0.005)
            renderPoint(self.viewer, p2, name + "/p2", RED_COLOR, 0.005)
            renderLine(self.viewer, p1, p2, name + "/sep_vec", 1., RED_COLOR)
            self._colres_idx += 1

    def delete_separation_vectors(self):
        for i in range(self._colres_idx):
            name = f"sep_vec_{i}"
            self.viewer[name].delete()
        self._colres_idx = 0

def create_complex_scene():
    # Create some shapes
    scene = AgimusScene()
    shapes = []
    transforms = []
    pin.seed(0)
    np.random.seed(0)

    N = 25
    for _ in range(N):
        shape = hppfcl.Ellipsoid(0.05, 0.15, 0.2)
        shapes.append(shape)
        shape = hppfcl.Capsule(0.1, 0.2)
        shapes.append(shape)
        shape = load_convex("./assets/mesh.stl")
        shapes.append(shape)

    for s in range(len(shapes)):
        M = pin.SE3.Random()
        transforms.append(M)
        color = np.random.rand(3)
        scene.register_object(shapes[s], M, color)

    # Add walls
    walls_color = np.ones(3)
    wall_size = 4.0

    # X-axis
    M = pin.SE3.Identity()
    M.translation = np.array([-wall_size, 0., 0.])/2
    transforms.append(M)
    shape = hppfcl.Box(0.5, wall_size, wall_size)
    shapes.append(shape)
    scene.register_object(shapes[-1], M, walls_color, True)

    M = pin.SE3.Identity()
    M.translation = np.array([wall_size, 0., 0.])/2
    transforms.append(M)
    shape = hppfcl.Box(0.5, wall_size, wall_size)
    shapes.append(shape)
    scene.register_object(shapes[-1], M, walls_color, True)

    # Y-axis
    M = pin.SE3.Identity()
    M.translation = np.array([0., -wall_size, 0.])/2
    transforms.append(M)
    shape = hppfcl.Box(wall_size, 0.5, wall_size)
    shapes.append(shape)
    scene.register_object(shapes[-1], M, walls_color, True)

    M = pin.SE3.Identity()
    M.translation = np.array([0., wall_size, 0.])/2
    transforms.append(M)
    shape = hppfcl.Box(wall_size, 0.5, wall_size)
    shapes.append(shape)
    scene.register_object(shapes[-1], M, walls_color, True)

    # Y-axis
    M = pin.SE3.Identity()
    M.translation = np.array([0., 0., -wall_size])/2
    transforms.append(M)
    shape = hppfcl.Box(wall_size, wall_size, 0.5)
    shapes.append(shape)
    scene.register_object(shapes[-1], M, walls_color, True)

    M = pin.SE3.Identity()
    M.translation = np.array([0., 0., wall_size])/2
    transforms.append(M)
    shape = hppfcl.Box(wall_size, wall_size, 0.5)
    shapes.append(shape)
    scene.register_object(shapes[-1], M, walls_color, True)

    return shapes, transforms, scene
