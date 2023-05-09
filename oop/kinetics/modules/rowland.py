import math
from modules.kinematics import *

class RowlandCircle(object):
    def __init__(self, sample, crystal, detector):
        self.circle = Circle(500)
        self.pos = [0, 0, 0]
        origin = Entity()
        self.diff_plane = Entity()
        origin.add_child(self.diff_plane, along = 'z', xyz = [0,0,0])
        self.diff_plane.add_child(self.circle, along = 'z', xyz = [0,0,0])

        self.points = [sample, crystal, detector]
        self.markers = [Marker(), Marker(), Marker()]

        for m in self.markers:
            self.diff_plane.add_child(m, along = 'z', xyz = [0,0,0])


    def update_rowland_circle(self):
        s, c, d = self.points

        S = s.get_world_coordinates(Vector(0,0,0))
        C = c.get_world_coordinates(Vector(0,0,0))
        D = d.get_world_coordinates(Vector(0,0,0))
        SC = C - S
        SD = D - S

        # Circle should point here
        SCxSD = SC.cross_product(SD)

        # Circle points here
        Z = self.diff_plane.Z

        # Rotation axis
        rot_axis = Z.cross_product(SCxSD).normalize()

        # Angle
        theta = math.acos(Z.dot_product(SCxSD) / (abs(Z) * abs(SCxSD)))
        self.diff_plane.rotation = Rotation(rot_axis, theta)


        P01 = self.diff_plane.get_local_coordinates(S)
        P02 = self.diff_plane.get_local_coordinates(C)
        P03 = self.diff_plane.get_local_coordinates(D)



        P01 = self.diff_plane.get_local_coordinates(S)
        P02 = self.diff_plane.get_local_coordinates(C)
        P03 = self.diff_plane.get_local_coordinates(D)

        # print(P01.xyz, P02.xyz, P03.xyz)

        self.markers[0].translation = Translation(P01)
        self.markers[1].translation = Translation(P02)
        self.markers[2].translation = Translation(P03)

        rowland_trans = c.get_world_coordinates(Vector(0,0,500))
        local_trans = self.diff_plane.get_local_coordinates(rowland_trans)

        self.circle.translation = Translation(local_trans)

        P01 = self.circle.get_local_coordinates(S)
        P02 = self.circle.get_local_coordinates(C)
        P03 = self.circle.get_local_coordinates(D)

        #print(P01.xyz, P02.xyz, P03.xyz)


        # self.diff_plane.translation = Translation(P02)



    def get_all_entities(self):
        entities = [self.circle]
        entities.extend(self.markers)

        return entities
