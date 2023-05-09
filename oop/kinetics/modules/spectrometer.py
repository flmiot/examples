import math
from modules.kinematics import *

# All units in mm

# JOHANN SPECTROMETER
class AnalyzerAssembly(object):
    def __init__(self, origin, distance_range, yaw_range, y_range,
        theta_range, assembly_position):
        self.pos = assembly_position
        self.distance = LinearStage(distance_range)
        self.yaw = RotationalStage(yaw_range)
        self.y = LinearStage(y_range)
        self.theta = RotationalStage(theta_range)
        self.analyzer = SphericalAnalyzer(50, 15)
        origin.add_child(self.distance, along = 'y', xyz = assembly_position)
        self.distance.add_child(self.yaw, along = '-y', xyz = [19, -100, 0])
        self.yaw.add_child(self.y, along = 'z', xyz = [0, 0, 0])
        # self.y.add_child(self.theta, along = 'x', xyz = [0,0,259.92])
        self.y.add_child(self.theta, along = 'x', xyz = [0,0,230.92])
        self.theta.move_to(math.pi/180 * 30)
        self.theta.add_child(self.analyzer, along = 'y', xyz = [0,0,0])


    def set_bragg_angle(self, angle):
        R = 500
        xp = math.sqrt((2*R)**2 * math.sin(angle)**4 - self.pos[0]**2)
        x = -1*(2*R*math.sin(angle) * math.cos(angle)**2 + xp*math.sin(angle))
        y = (2*R*math.cos(angle) * math.sin(angle)**2 - xp*math.cos(angle))

        x_loc = x - self.pos[1]
        y_loc = y - self.pos[2]
        phi = math.atan(self.pos[0] / (xp * math.sin(angle)))
        beta =  math.pi/2 - math.atan((math.sqrt(self.pos[0]**2 + xp**2*math.sin(angle)**2))/(xp * math.cos(angle)))
        self.distance.move_to(x_loc)
        self.y.move_to(y)
        self.yaw.move_to(phi)
        self.theta.move_to(beta)

        #print(x_loc, y, phi * 180 / math.pi, beta * 180 / math.pi)


    def get_entities(self):
        return [self.distance, self.yaw, self.y, self.theta, self.analyzer]

class JohannSpectrometer(object):

    def __init__(self, sample_distance, sample_height):
        origin = Entity()

        self.assemblies = []

        # First analyzer (looking from sample position: far right)
        a1 = AnalyzerAssembly(
            origin = origin,
            distance_range = [-82.65, 80.65],
            yaw_range = [-math.pi / 4, math.pi / 4],
            y_range = [-55, 55],
            theta_range = [0 * math.pi/180., 60 * math.pi/180.],
            assembly_position = [-347, sample_distance, 1047.50])

        a2 = AnalyzerAssembly(
            origin = origin,
            distance_range = [-82.65, 80.65],
            yaw_range = [-math.pi / 4, math.pi / 4],
            y_range = [-55, 55],
            theta_range = [0 * math.pi/180., 60 * math.pi/180.],
            assembly_position = [-347/2, sample_distance, 1047.50])

        a3= AnalyzerAssembly(
            origin = origin,
            distance_range = [-82.65, 80.65],
            yaw_range = [-math.pi / 4, math.pi / 4],
            y_range = [-55, 55],
            theta_range = [0 * math.pi/180., 60 * math.pi/180.],
            assembly_position = [0, sample_distance, 1047.50])

        a4 = AnalyzerAssembly(
            origin = origin,
            distance_range = [-82.65, 80.65],
            yaw_range = [-math.pi / 4, math.pi / 4],
            y_range = [-55, 55],
            theta_range = [0 * math.pi/180., 60 * math.pi/180.],
            assembly_position =  [347/2, sample_distance, 1047.50])

        a5 = AnalyzerAssembly(
            origin = origin,
            distance_range = [-82.65, 80.65],
            yaw_range = [-math.pi / 4, math.pi / 4],
            y_range = [-55, 55],
            theta_range = [0 * math.pi/180., 60 * math.pi/180.],
            assembly_position = [347, sample_distance, 1047.50])

        self.assemblies.extend([a1, a2, a3, a4, a5])


    def set_bragg_angle(self, angle):
        for a in self.assemblies:
            a.set_bragg_angle(angle)


    def get_all_entities(self):
        entities = []
        for a in self.assemblies:
            entities.extend(a.get_entities())

        return entities

# VON HAMOS SPECTROMETER
