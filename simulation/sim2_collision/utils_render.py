import hppfcl
import numpy as np
import meshcat.geometry as mg
import pinocchio as pin
from distutils.version import LooseVersion
import warnings
from typing import Any, Dict, Union
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


def loadPrimitive(self, geometry_object: hppfcl.ShapeBase):

    import meshcat.geometry as mg

    # Cylinders need to be rotated
    basic_three_js_transform = np.array([[1.,  0.,  0.,  0.],
                  [0.,  0., -1.,  0.],
                  [0.,  1.,  0.,  0.],
                  [0.,  0.,  0.,  1.]])
    RotatedCylinder = type("RotatedCylinder", (mg.Cylinder,), {"intrinsic_transform": lambda self: basic_three_js_transform })

    # Cones need to be rotated

    geom: hppfcl.ShapeBase = geometry_object.geometry
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
        sx = geometry_object.meshScale[0] * 10
        sy = geometry_object.meshScale[1] * 10
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
        msg = "Unsupported geometry type for %s (%s)" % (geometry_object.name, type(geom) )
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
