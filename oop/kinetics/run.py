import tkinter as tk

import math
import time
import matplotlib
matplotlib.use('Qt5Agg')

from modules.kinematics import Entity, SphericalAnalyzer
from modules.spectrometer import JohannSpectrometer
from modules.detector import JohannDetector
from modules.rowland import RowlandCircle
from viewer import Viewer
from window import MainWindow

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    origin = Entity()
    sample_pos = Entity()
    sample_height =  1392.5
    origin.add_child(sample_pos, along = 'z', xyz = [0,0, sample_height])

    js = JohannSpectrometer(-930, 1392.5)
    d = JohannDetector(80, 40, 1392.5)

    rowland_circles = []
    analyzers = []
    for e in js.get_all_entities():
        if isinstance(e, SphericalAnalyzer):
            analyzers.append(e)

    rowland_circles.extend([RowlandCircle(sample_pos, a, d.detector) for a in analyzers])

    struct = {
        'sample_pos' : sample_pos,
        'sample_height' : sample_height,
        'js' : js,
        'd' : d,
        'analyzers' : analyzers,
        'rowland_circles' : rowland_circles
    }

    v = Viewer(struct)
    root = tk.Tk()
    main = MainWindow(v, root)
    main.pack(side="top", fill="both", expand=True)
    root.mainloop()
