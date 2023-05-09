import tkinter as tk

import math
import time
import matplotlib

from modules.kinematics import Vector, RotationalStage

class MainWindow(tk.Frame):
    def __init__(self, viewer, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.last_updated = time.time()
        self.viewer = viewer
        self.entries = []
        self.inputs = []
        parameters = {'Bragg angle': self.viewer.angle}
        for ind, k in enumerate(parameters):
            label = k
            l = tk.Label(self, text = k)
            e = tk.Scale(self, from_=70, to=89.99, orient=tk.HORIZONTAL, command = self.update_state, resolution=0.01)
            e.set(75)
            i = tk.Entry(self)
            i.delete(0, tk.END)
            i.insert(0, "75")
            l.grid(row=ind, column=0)
            e.grid(row=ind, column=1)
            i.grid(row=ind, column=2)
            self.entries.append(e)
            self.inputs.append(i)

        self.master.bind("<Return>", self.update_return)


    def update_state(self, event=None):
        parameters = {'Bragg angle': self.viewer.angle}
        for ind, k in enumerate(parameters):
            s = self.entries[ind].get()
            i = self.inputs[ind]
            i.delete(0, tk.END)
            i.insert(0, s)

        t_new = time.time()
        if t_new - self.last_updated > 0.05:
            self.master.after(10, self.update)
            self.last_updated = t_new


    def update_return(self, event = None):
        parameters = {'Bragg angle': self.viewer.angle}
        for ind, k in enumerate(parameters):
            s = float(self.inputs[ind].get())
            if s < 70:
                i = self.inputs[ind]
                i.delete(0, tk.END)
                i.insert(0, "70")
                self.entries[ind].set(70)
            elif s > 89.9:
                i = self.inputs[ind]
                i.delete(0, tk.END)
                i.insert(0, "89.9")
                self.entries[ind].set(89.9)
            else:
                self.entries[ind].set(s)


    def update(self, event = None):
        s = {}
        parameters = {'Bragg angle': self.viewer.angle}
        for ind, k in enumerate(parameters):
            s = float(self.inputs[ind].get())
        self.viewer.angle = s
        parameters['Bragg angle'] = s

        rotStages = []
        for e in self.viewer.s['js'].get_all_entities():
            if isinstance(e, RotationalStage):
                rotStages.append(e)
        yaws, thetas = rotStages[0::2], rotStages[1::2]

        output = {}
        for ind, a in enumerate(self.viewer.s['analyzers']):
            output['Analyzer {}'.format(ind+1)] = {}
            v = a.get_world_coordinates()
            v = Vector(v.x, v.y, v.z - self.viewer.s['sample_height'])
            p = v.xyz
            length = abs(v)
            output['Analyzer {}'.format(ind+1)]['X Y Z'] = [round(p[1],2), round(p[2],2), round(p[0],2)]
            output['Analyzer {}'.format(ind+1)]['sample-crystal distance'] = length
            t = thetas[ind]
            output['Analyzer {}'.format(ind+1)]['theta angle'] = 90 - t.position * 180 / 3.14

        self.output(output, header = {'Bragg angle': self.viewer.angle})



    def output(self, parameters, header = None):
        str = ''
        l = 0

        for analyzer_name, analyzer_parameters in parameters.items():
            for ind, parameter in enumerate(analyzer_parameters.items()):
                parameter_name, parameter_value = parameter
                if ind == 0:
                    s = '{:>11} {:<25}: {}\n'.format(analyzer_name, parameter_name, parameter_value)
                else:
                    s = '{:>11} {:<25}: {}\n'.format('', parameter_name, parameter_value)
                str += s
                l = max(l, len(s))
            str += '\n'

        border = '=' * l + '\n'
        str = '\n'+ border + str + border

        header_str = ''
        if len(header.items()) > 0:
            header_str += border
            for header_name, header_value in header.items():
                s = ' {:>10} {:<25}\n'.format(header_name, header_value)
                header_str += s
        str = header_str + str

        print(str)
