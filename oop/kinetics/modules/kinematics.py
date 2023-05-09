import math
import array
import operator


class Rotation(object):
    def __init__(self, axis, angle):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        self.t1 = Vector(
            cosine + axis.x**2 * (1-cosine),
            axis.x*axis.y*(1-cosine)-axis.z*sine,
            axis.x*axis.z*(1-cosine)+axis.y*sine
        )
        self.t2 = Vector(
            axis.y*axis.x*(1-cosine)+axis.z*sine,
            cosine + axis.y**2*(1-cosine),
            axis.y*axis.z*(1-cosine)-axis.x*sine
        )
        self.t3 = Vector(
            axis.z*axis.x*(1-cosine)-axis.y*sine,
            axis.z*axis.y*(1-cosine)+axis.x*sine,
            cosine + axis.z**2 * (1-cosine)
        )


    def rotate_vector(self, vector):
        rotated = Vector(
            self.t1.dot_product(vector),
            self.t2.dot_product(vector),
            self.t3.dot_product(vector)
        )
        return rotated


    def rotate_vector_inverse(self, vector):
        t1 = Vector(self.t1.x, self.t2.x, self.t3.x)
        t2 = Vector(self.t1.y, self.t2.y, self.t3.y)
        t3 = Vector(self.t1.z, self.t2.z, self.t3.z)
        rotated = Vector(
            t1.dot_product(vector),
            t2.dot_product(vector),
            t3.dot_product(vector)
        )
        return rotated


class Translation(object):
    def __init__(self, vector):
        self.v = vector


    def translate(self, vector):
        return vector + self.v


    def translate_inverse(self, vector):
        return vector - self.v



class Vector(object):
    def __init__(self, x, y, z):
        self.xyz    = array.array('f', [x, y, z])


    @property
    def x(self):
        return self.xyz[0]


    @property
    def y(self):
        return self.xyz[1]


    @property
    def z(self):
        return self.xyz[2]


    def rotate(self, axis, angle):
        rotation = Rotation(axis, angle)
        return rotation.rotate_vector(self)


    def dot_product(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z


    def cross_product(self, other):
        return Vector(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )


    def normalize(self):
        return self / abs(self)


    def operate(self, other, op):
        if isinstance(other, Vector):
            xyz = list([op(c1, c2) for c1, c2 in zip(self.xyz, other.xyz)])

        else:
            xyz = list([op(c1, other) for c1 in self.xyz])

        return Vector(*xyz)


    def __add__(self, other):
        return self.operate(other, operator.add)


    def __radd__(self, other):
        return self.operate(other, operator.add)


    def __sub__(self, other):
        return self.operate(other, operator.sub)


    def __mul__(self, other):
        return self.operate(other, operator.mul)


    def __rmul__(self, other):
        return self.operate(other, operator.mul)


    def __div__(self, other):
        return self.operate(other, operator.div)


    def __truediv__(self, other):
        return self.operate(other, operator.truediv)


    def __eq__(self, other):
        return self.compare(other, operator.eq)


    def __ne__(self, other):
        return self.compare(other, operator.ne)

    def __abs__(self):
        return math.sqrt(self.xyz[0]**2 + self.xyz[1]**2 + self.xyz[2]**2)


class Entity(object):
    X = Vector(1, 0, 0)
    Y = Vector(0, 1, 0)
    Z = Vector(0, 0 ,1)

    def __init__(self, translation = None, rotation = None, children = []):

        self.parent        = None
        self.translation   = translation
        self.rotation      = rotation
        self.children      = children


    def add_child(self, child, along = None, xyz = None):
        if along == 'x':
            rot = Rotation(Vector(0,1,0), math.pi/2)
        elif along == 'y':
            rot = Rotation(Vector(1,0,0), -1*math.pi/2)
        elif along == 'z':
            rot = Rotation(Vector(0,0,1), 0)
        elif along == '-x':
            rot = Rotation(Vector(0,1,0), -1*math.pi/2)
        elif along == '-y':
            rot = Rotation(Vector(1,0,0), math.pi/2)
        elif along == '-z':
            rot = Rotation(Vector(1,0,0), math.pi)
        else:
            rot = None

        if xyz is not None:
            trans = Translation(Vector(*xyz))
        else:
            trans = None

        if rot:
            child.rotation = rot

        if trans:
            child.translation = trans

        child.parent = self
        self.children.append(child)


    def get_world_coordinates(self, local_vector = None):

        if local_vector is None:
            local_vector = Vector(0,0,0)

        if self.parent is None:
            return local_vector

        else:
            vector = self.rotation.rotate_vector(local_vector)
            vector = self.translation.translate(vector)
            return self.parent.get_world_coordinates(vector)


    def get_local_coordinates(self, world_vector):

        if self.parent is None:
            return world_vector

        else:
            vector = self.parent.get_local_coordinates(world_vector)
            vector = self.rotation.rotate_vector_inverse(vector)
            vector = self.translation.translate_inverse(vector)
            return vector


    def get_axes(self, world_coordinates = True, scale_arrows = 50):
        if world_coordinates:
            b = self.get_world_coordinates(Vector(0,0,0))
            x = self.get_world_coordinates(self.X * scale_arrows)
            y = self.get_world_coordinates(self.Y * scale_arrows)
            z = self.get_world_coordinates(self.Z * scale_arrows)
            return b, x, y, z
        else:
            return Vector(0,0,0), self.X, self.Y, self.Z



class Motor(Entity):
    def __init__(self, range, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)

        self.lower_limit, self.upper_limit = range

        # Add motor head/slide/wheel
        translation = Translation(Vector(0,0,0))
        rotation = Rotation(self.Z, 0)
        self.slide = Entity()
        Entity.add_child(self, self.slide, along = 'z', xyz = [0,0,0])
        self._set_position(0.0)



    def add_child(self, child, *args, **kwargs):
        self.slide.add_child(child, *args, **kwargs)


    def move_to(self, motor_position):
        if motor_position < self.lower_limit:
            motor_position = self.lower_limit

        if motor_position > self.upper_limit:
            motor_position = self.upper_limit

        self._set_position(motor_position)


    def move_by(self, distance):
        self.move_to(self.position + distance)


    def _set_position(self, motor_position):
        raise NotImplementedError()


    def _get_position(self):
        return self.slide.get_world_coordinates(Vector(0,0,0))


class LinearStage(Motor):
    def _set_position(self, position):
        self.slide.translation = Translation(Vector(0, 0, position))
        self.position = position


class RotationalStage(Motor):
    def _set_position(self, position):
        self.slide.rotation = Rotation(self.Z, position)
        self.position = position


class Marker(Entity):
    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)


class SphericalAnalyzer(Entity):
    def __init__(self, radius, thickness, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self.r = radius
        self.t = thickness


    def get_contour(self, world_coordinates = True):
        def range_positve(start, stop=None, step=None):
            if stop == None:
                stop = start + 0.0
                start = 0.0
            if step == None:
                step = 1.0
            while start < stop:
                yield start
                start += step

        segments_front = []
        segments_back = []
        for t in range_positve(0, 2*math.pi, 2*math.pi / 25):
            vf = Vector(self.r * math.sin(t), self.r * math.cos(t), 0)
            vb = Vector(self.r * math.sin(t), self.r * math.cos(t), -self.t)

            if world_coordinates:
                vf = self.get_world_coordinates(vf)
                vb = self.get_world_coordinates(vb)

            segments_front.append(vf)
            segments_back.append(vb)


        segments_front.append(segments_front[0])
        segments_back.append(segments_back[0])
        return segments_front, segments_back


class Circle(Entity):
    def __init__(self, radius, *args, **kwargs):
        self.r = radius

    def get_contour(self, world_coordinates = True):
        def range_positve(start, stop=None, step=None):
            if stop == None:
                stop = start + 0.0
                start = 0.0
            if step == None:
                step = 1.0
            while start < stop:
                yield start
                start += step

        segments = []

        for t in range_positve(0, 2*math.pi, 2*math.pi / 125):
            v = Vector(self.r * math.sin(t), self.r * math.cos(t), 0)

            if world_coordinates:
                v = self.get_world_coordinates(v)

            segments.append(v)


        segments.append(segments[0])
        return segments


class CylindrcalAnalyzer(Entity):
    def __init__(self, radius, height, width, *args, **kwargs):
        self.r = radius
        self.h = height
        self.w = width


    def get_contour(self, world_coordinates = True):
        def range_positve(start, stop=None, step=None):
            if stop == None:
                stop = start + 0.0
                start = 0.0
            if step == None:
                step = 1.0
            while start < stop:
                yield start
                start += step

        segments_front = []
        segments_back = []



        for t in range_positve(0, 2*math.pi, 2*math.pi / 25):
            vf = Vector(self.r * math.sin(t), self.r * math.cos(t), self.t)
            vb = Vector(self.r * math.sin(t), self.r * math.cos(t), 0)

            if world_coordinates:
                vf = self.get_world_coordinates(vf)
                vb = self.get_world_coordinates(vb)

            segments_front.append(vf)
            segments_back.append(vb)


        segments_front.append(segments_front[0])
        segments_back.append(segments_back[0])
        return segments_front, segments_back


class Detector(Entity):
    def __init__(self, height, width, *args, **kwargs):
        self.h = height
        self.w = width


    def get_contour(self, world_coordinates = True):
        s = [
                 Vector(-self.h/2, -self.w/2, 0),
                 Vector(-self.h/2, self.w/2, 0),
                 Vector(self.h/2, self.w/2, 0),
                 Vector(self.h/2, -self.w/2, 0)
            ]

        segments = []
        for v in s:
            if world_coordinates:
                v = self.get_world_coordinates(v)

            segments.append(v)

        segments.append(segments[0])
        return segments


    def _tilt(self, position):
        self.rotation = Rotation(self.Z, position)


class MotorError(Exception):

    """Exceptions related to motors."""

    pass
