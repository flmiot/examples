import math
from modules.kinematics import *

class JohannDetector(object):
    def __init__(self, height, width, sample_height):
        origin = Entity()
        sample_pos = Entity()
        origin.add_child(sample_pos, along = 'z', xyz = [0,0, sample_height])
        self.detector = Detector(height, width)
        sample_pos.add_child(self.detector, along = 'z', xyz = [0,0,0])
        self.pos = [0, 0, 0]


    def set_bragg_angle(self, angle):
        R = 500
        x = 4*R*math.sin(angle)*math.cos(angle)**2
        y = 4*R*math.cos(angle)*math.sin(angle)**2
        x_loc = x - self.pos[1]
        y_loc = y - self.pos[2]

        self.detector.translation = Translation(Vector(0, -x_loc, y_loc))
        beta = (3/2)*math.pi - 2*angle
        self.detector.rotation = Rotation(self.detector.X, beta)



    def get_all_entities(self):
        entities = [self.detector]

        return entities
