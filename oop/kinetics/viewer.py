import math
import matplotlib.pyplot as plt
from modules.kinematics import Vector, Entity, Marker, Motor, SphericalAnalyzer, Circle, LinearStage, , Detector

def plot_vector(axis_object, xyz1, xyz2, *args, **kwargs):
    axis_object.plot([xyz1.x, xyz2.x], [xyz1.y, xyz2.y], [xyz1.z, xyz2.z], *args, **kwargs)

def plot_entity(axis_object, entity):
    if isinstance(entity, SphericalAnalyzer):
        seg_front, seg_back = entity.get_contour()
        x = [v.x for v in seg_front]
        y = [v.y for v in seg_front]
        z = [v.z for v in seg_front]
        axis_object.plot(x, y, z, color = "black")
        x = [v.x for v in seg_back]
        y = [v.y for v in seg_back]
        z = [v.z for v in seg_back]
        axis_object.plot(x, y, z, color = "black")

    elif isinstance(entity, Detector):
        segments = entity.get_contour()
        x = [v.x for v in segments]
        y = [v.y for v in segments]
        z = [v.z for v in segments]
        axis_object.plot(x, y, z, color = "orange")

    elif isinstance(entity, Circle):
        segments = entity.get_contour()
        x = [v.x for v in segments]
        y = [v.y for v in segments]
        z = [v.z for v in segments]
        axis_object.plot(x, y, z, color = "green", linewidth = 0.5)

    elif isinstance(entity, Marker):
        b, x, y, z = entity.get_axes(scale_arrows = 25)
        plot_vector(axis_object, b, x, color = 'orange', linewidth = 0.01)
        plot_vector(axis_object, b, y, color = 'orange', linewidth = 0.01)
        plot_vector(axis_object, b, z, color = 'orange', linewidth = 0.01)

    else:
        return
        b, x, y, z = entity.get_axes()
        plot_vector(axis_object, b, x, color = 'r')
        plot_vector(axis_object, b, y, color = 'g')
        plot_vector(axis_object, b, z, color = 'b')

class Viewer():
    def __init__(self, struct):
        self.__angle = 60 * math.pi / 180

        plt.ion()
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self.ax.set_proj_type('ortho')
        self.ax.set_xlim([-500, 500])
        self.ax.set_ylim([10, -1000])
        self.ax.set_zlim([1000, 2000])

        plt.show()

        self.s = struct

    @property
    def angle(self):
        return self.__angle

    @angle.setter
    def angle(self, new_angle):
        self.__angle = new_angle
        self.update_plot()

    def update_plot(self):
        elev = self.ax.elev
        azim = self.ax.azim
        x_min, x_max = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        z_min, z_max = self.ax.get_zlim()


        self.fig.clf()
        self.ax  = self.fig.add_subplot(111, projection='3d')

        js = self.s['js']
        js.set_bragg_angle(self.angle / 180 * math.pi)
        for e in js.get_all_entities():
            plot_entity(self.ax, e)

        d = self.s['d']
        d.set_bragg_angle(self.angle / 180 * math.pi)
        for e in d.get_all_entities():
            plot_entity(self.ax, e)

        for rc in self.s['rowland_circles']:
            rc.update_rowland_circle()
            for e in rc.get_all_entities():
                plot_entity(self.ax, e)

        # Draw sample position
        self.ax.plot([-10,10], [-0,0], [1400,1400], color = "black")
        self.ax.plot([-0,0], [-10,10], [1400,1400], color = "black")
        self.ax.plot([-0,0], [-0,0], [1390,1410], color = "black")

        # Draw rays
        for a in self.s['analyzers']:
            P01 = self.s['sample_pos'].get_world_coordinates()
            P02 = a.get_world_coordinates()
            P03 = d.detector.get_world_coordinates()

            self.ax.plot(*zip(P01.xyz, P02.xyz), color = "blue", linewidth = 0.5)
            self.ax.plot(*zip(P02.xyz, P03.xyz), color = "blue", linewidth = 0.5)

        self.ax.elev = elev
        self.ax.azim = azim
        self.ax.set_xlim([x_min, x_max])
        self.ax.set_ylim([ymin, ymax])
        self.ax.set_zlim([z_min, z_max])
