import re
import sys
import logging
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from fes.simulation import Simulation, init_state, init_results, check_sim_state, read_state_file

Log = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG)



class FES_UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.progress = 0
        voxels = int(state['General']['Voxels (^3)'][0])
        self.imv1data = np.zeros((voxels, voxels, voxels))
        self.imv3data = np.zeros((voxels, voxels, voxels))
        self.gl1data = np.zeros((voxels, voxels, voxels, 4), dtype=np.ubyte)
        self.gl2data = np.array([[[0,0,0], [1,1,1]]])

        self.setupUi()

    def setupUi(self):

        self.centralWidget = QtWidgets.QWidget()
        hbox = QtWidgets.QVBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.centralWidget.setLayout(hbox)

        # Layouts
        self.left_frame = QtWidgets.QWidget()
        self.right_frame = QtWidgets.QWidget()
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setHandleWidth(4)
        self.splitter.addWidget(self.left_frame)
        self.splitter.addWidget(self.right_frame)
        self.splitter.setSizes([100,200])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        hbox.addWidget(self.splitter)

        self.verticalLayout_left = QtWidgets.QVBoxLayout()
        self.verticalLayout_left.setStretch(0, 1)
        self.verticalLayout_left.setStretch(1, 0)
        self.verticalLayout_left.setContentsMargins(5, 5, 1, 5)

        self.gridLayout_right = QtWidgets.QGridLayout()
        self.gridLayout_right.setContentsMargins(1, 5, 5, 5)
        self.left_frame.setLayout(self.verticalLayout_left)
        self.right_frame.setLayout(self.gridLayout_right)






        # # Make TreeWidget
        # self.treeWidget = QtGui.QTreeWidget(self.centralwidget)
        # self.treeWidget.headerItem().setText(0, "Property")
        # self.treeWidget.headerItem().setText(1, "Value")
        # self.treeWidget.headerItem().setText(2, "Unit")
        # self.treeWidget.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        # self.treeWidget.header().setStretchLastSection(True)
        #
        # # Populate TreeWidget
        # self.top_items = []
        # self.sub_items = []
        # for top_ind, topic in enumerate(sim_state['topics']):
        #     top_item = QtGui.QTreeWidgetItem(self.treeWidget)
        #     self.treeWidget.topLevelItem(top_ind).setText(0, topic)
        #     self.top_items.append(top_item)
        #     for sub_ind, property in enumerate(sim_state['properties'][top_ind]):
        #         sub_item = QtGui.QTreeWidgetItem(top_item)
        #         sub_item.setText(0, property)
        #         value = str(sim_state['values'][top_ind][sub_ind])
        #         sub_item.setText(1, value)
        #         unit = str(sim_state['units'][top_ind][sub_ind])
        #         sub_item.setText(2, unit)
        #         self.sub_items.append(sub_item)
        #
        # self.verticalLayout_left.addWidget(self.treeWidget)

        tree_parameters = []
        for topic in state.keys():
            d = {'name': topic, 'type':'group'}
            children = []
            for property in state[topic]:
                name = property
                if isinstance(state[topic][property], dict):
                    values = state[topic][property]['options']
                    sel = state[topic][property]['selected']
                    p = {'name':property, 'type':'list', 'values':values, 'default':sel, 'value':sel, 'suffix': 'mm'}
                else:
                    value, ptype, unit = state[topic][property]
                    if unit is None:
                        p = {'name':property, 'type':ptype, 'value':value}
                    else:
                        p = {'name':property, 'type':ptype, 'value':value, 'siPrefix':False, 'suffix': unit}

                children.append(p)
            d['children'] = children
            tree_parameters.append(d)

        self.p = Parameter.create(name='params', type='group', children=tree_parameters, tip = ["hi!:)"])
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)


        result_parameters = []
        for topic in results.keys():
            d = {'name': topic, 'type':'group'}
            children = []
            for property in results[topic]:
                name = property
                value, ptype, unit = results[topic][property]
                if unit is None:
                    p = {'name':property, 'type':ptype, 'value':value, 'readonly':True}
                else:
                    p = {'name':property, 'type':ptype, 'value':value, 'siPrefix':False, 'suffix': unit, 'readonly':True}
                children.append(p)
            d['children'] = children
            result_parameters.append(d)


        self.rp = Parameter.create(name='params', type='group', children=result_parameters)

        self.r = ParameterTree()
        self.r.setParameters(self.rp, showTop=False)

        # # Tabbing
        # self.tab_widget = QtGui.QTabWidget()
        # self.tab_setup = QtGui.QWidget()
        # setup_layout = QtGui.QVBoxLayout()
        # setup_layout.addWidget(self.t)
        # self.tab_setup.setLayout(setup_layout)
        # self.tab_result = QtGui.QWidget()
        # result_layout = QtGui.QVBoxLayout()
        # result_layout.addWidget(self.r)
        # self.tab_result.setLayout(result_layout)
        # self.tab_widget.addTab(self.tab_setup, "Setup")
        # self.tab_widget.addTab(self.tab_result, "Results")
        # self.verticalLayout_left.addWidget(self.tab_widget)

        # Setup / Parameter splitting widget
        self.upper_w = QtWidgets.QWidget()
        self.upper_l = QtWidgets.QVBoxLayout()
        self.upper_l.addWidget(self.t)
        self.upper_l.setContentsMargins(2, 0, 0, 0)
        self.upper_w.setLayout(self.upper_l)
        self.lower_w = QtWidgets.QWidget()
        self.lower_l = QtWidgets.QVBoxLayout()
        self.lower_l.addWidget(self.r)
        self.lower_l.setContentsMargins(2, 0, 0, 0)
        self.lower_w.setLayout(self.lower_l)
        self.splitter2 = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter2.setHandleWidth(4)
        self.splitter2.addWidget(self.upper_w)
        self.splitter2.addWidget(self.lower_w)
        self.splitter2.setSizes([200,100])
        self.splitter2.setCollapsible(0, False)
        self.splitter2.setCollapsible(1, False)
        self.verticalLayout_left.addWidget(self.splitter2)

        # Button and progress bar
        self.buttonBarLayout = QtWidgets.QHBoxLayout()
        self.runButton = QtWidgets.QPushButton()
        self.runButton.setText("Run (F5)")
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setProperty("value", self.progress)
        self.buttonBarLayout.addWidget(self.runButton)
        self.buttonBarLayout.addWidget(self.progressBar)
        self.verticalLayout_left.addLayout(self.buttonBarLayout)

        # Make ImageView
        self.imv1 = pg.ImageView()
        self.imv2 = pg.ImageView()
        self.imv3 = pg.ImageView()
        self.imv4 = pg.ImageView()
        self.imv1.getView().invertY(False)
        self.imv2.getView().invertY(False)
        self.imv3.getView().invertY(False)
        self.imv4.getView().invertY(False)

        self.imv3.getView().linkView(pg.ViewBox.XAxis, self.imv1.getView())
        self.imv3.getView().linkView(pg.ViewBox.YAxis, self.imv1.getView())

        # self.imv1.setImage(self.imv1data)
        # self.imv3.setImage(self.imv3data)
        # self.imv1.setCurrentIndex(int(state['General']['Voxels (^3)'][0]/2))

        # Make ImageView labels
        l1 = QtWidgets.QLabel()
        l2 = QtWidgets.QLabel()
        l3 = QtWidgets.QLabel()
        l4 = QtWidgets.QLabel()
        l1.setText('<font size="2"><b>Weighted excited state fraction</b> (Top view)')
        l2.setText('<font size="2"><b>Weighted excited state fraction</b> (Side view)')
        l3.setText('<font size="2"><b>Rel. absorbed photons</b> (Top view) - <font color="blue">Pump</font>/<font color="red">Probe</font>');
        l4.setText('<font size="2"><b>Rel. absorbed photons</b> (Side view) - <font color="blue">Pump</font>/<font color="red">Probe</font>');
        self.gridLayout_right.addWidget(l1, 0,0)
        self.gridLayout_right.addWidget(self.imv1, 1,0)
        self.gridLayout_right.addWidget(l2, 0,1)
        self.gridLayout_right.addWidget(self.imv2, 1,1)
        self.gridLayout_right.addWidget(l3, 2,0)
        self.gridLayout_right.addWidget(self.imv3, 3,0)
        self.gridLayout_right.addWidget(l4, 2,1)
        self.gridLayout_right.addWidget(self.imv4, 3,1)

        vox = state['General']['Voxels (^3)'][0]
        radius = state['Sample enviroment']['Jet radius'][0]
        frame = state['General']['Voxel frame'][0]
        scale_factor =  vox / ((1+frame)*2*radius)

        self.roi = pg.LineSegmentROI([[-vox/2, 0], [vox/2,0]], pen='r')
        self.imv1.addItem(self.roi)

        # ImageView guides
        self.guides_top = self.draw_guides_top(self.imv1)
        self.guides_top2 = self.draw_guides_top(self.imv3)
        self.guides_side = self.draw_guides_side(self.imv2)
        self.guides_side2 = self.draw_guides_side(self.imv4)

        self.setCentralWidget(self.centralWidget)




        # Events
        self.roi.sigRegionChanged.connect(self.update_ImageView)
        self.imv1.getView().scene().sigMouseMoved.connect(self.mouse_moved_imv)
        self.imv1.sigTimeChanged.connect(self.time_slider_moved1)
        self.imv3.sigTimeChanged.connect(self.time_slider_moved2)
        self.p.sigTreeStateChanged.connect(self.parameter_changed)
        self.runButton.clicked.connect(self.run_simulation)

        # Animations
        self.lower_w.setAutoFillBackground(True)
        self.anim = QtCore.QPropertyAnimation(self, b"color")
        self.anim.setDuration(2500)
        self.anim.setLoopCount(1)
        df = self.lower_w.palette().window().color()
        self.anim.setStartValue(QtGui.QColor(0, 255, 0))
        self.anim.setEndValue(df)


    def _set_color(self, col):
        w = self.lower_w
        p = w.palette()
        p.setColor(w.backgroundRole(), col)
        w.setPalette(p)

    color = QtCore.pyqtProperty(QtGui.QColor, fset=_set_color)


    def draw_guides_top(self, widget):
        vox = state['General']['Voxels (^3)'][0]
        radius = state['Sample enviroment']['Jet radius'][0]
        frame = state['General']['Voxel frame'][0]
        scale_factor =  vox / ((1+frame)*2*radius)

        # Grid
        grid = pg.GridItem()
        grid.scale()
        widget.addItem(grid)

        # Jet
        t = np.linspace(0,2*np.pi, 100)
        x = radius * scale_factor * np.cos(t)
        y = radius * scale_factor * np.sin(t)
        circle = pg.PlotCurveItem(pen='g')
        circle.setData(x,y)
        widget.addItem(circle)

        return [grid, circle]
        #return [grid]

    def time_slider_moved1(self):
        z = self.imv1.currentIndex
        self.imv3.setCurrentIndex(z)

    def time_slider_moved2(self):
        z = self.imv3.currentIndex
        self.imv1.setCurrentIndex(z)

    def draw_guides_side(self, widget):
        vox = state['General']['Voxels (^3)'][0]
        radius = state['Sample enviroment']['Jet radius'][0]
        frame = state['General']['Voxel frame'][0]
        scale_factor =  vox / ((1+frame)*2*radius)

        # Grid
        grid = pg.GridItem()
        grid.scale()
        widget.addItem(grid)

        # Jet
        ll = pg.InfiniteLine(pos = -radius*scale_factor)
        lr = pg.InfiniteLine(pos = radius*scale_factor)
        # widget.addItem(ll)
        # widget.addItem(lr)

        # return [grid, ll, lr]
        return [grid]


    def get_composite_image(self):
        overlap = sim.r['arrays']["Weighted excited state fraction"]#[z,:,:]
        pump = sim.r['arrays']["Pump abs."]#[z,:,:]
        probe = sim.r['arrays']["Probe abs."]#[z,:,:]
        sh = overlap.shape + (3,)
        comp = np.empty(sh, dtype = np.float)
        comp[...,0] = probe / np.max(probe)
        comp[...,1] = overlap / np.max(overlap)
        comp[...,2] = pump  / np.max(pump)
        # comp[...,3] = 0.3 * comp[...,0] + 0.3*comp[...,1] + 0.3 * comp[...,2]
        # comp[...,3] = (comp[...,3].astype(float) / 255.) **2 * 255

        return comp



    def mouse_moved_imv(self, pos):
        if self.imv1.image is not None:

            vox = state['General']['Voxels (^3)'][0]
            pos = self.imv1.getView().mapSceneToView(pos)
            x,y = int(pos.x() + vox/2), int(pos.y() + vox/2)
            z = self.imv1.currentIndex

            if x >= 0 and x < vox and y >= 0 and y < vox and z >= 0 and z < vox:

                value1 = self.imv1.image[z,x,y]
                value2 = self.imv3.image[z,x,y]
                pump_abs_photons = np.max(sim.r['arrays']['Pump abs.'])*value2[2]
                probe_rel_abs_photons = value2[0]
                self.rp.child('Cursor position').child('Weighted excited state fraction').setValue(value1)
                self.rp.child('Cursor position').child('Pump absorbed photons').setValue(pump_abs_photons)
                self.rp.child('Cursor position').child('Probe rel. absorbed photons').setValue(probe_rel_abs_photons)



    def update_result_values(self):
        global sim
        # values = [
        #     "{:.4E}".format(sim.r["Pump total incident photons"]),
        #     "{:.4E}".format(sim.r["Pump mean absorbed photons"]),
        #     "{:.4E}".format(sim.r["Pump total absorbed photons"]),
        #     "{:.4f}".format(sim.r["Pump mean excited state fraction"]),
        #     "{:.4E}".format(sim.r["Volume per voxel"]),
        #     "{:.4E}".format(sim.r["Molecules per voxel"])]


        for parameter in self.rp.child('General results'):
            value = sim.r['scalars'][parameter.name()]
            parameter.setValue(value)

        # for topic in sim_state.keys():
        #     for property in sim_state[topic]:
        #         v = sim_state[topic][property]
        #         if isinstance(v, dict):
        #             continue
        #         else:
        #             value = sim_state[topic][property][0]
        #             if property in ints:
        #                 checked[topic][property][0] = int(value)
        #             elif property in arrays:
        #                 pattern = r'([-+]?\d*\.\d+|[-+]?\d+)'
        #                 numbers = re.findall(pattern, str(value))
        #                 value = np.array(list([float(n) for n in numbers]))
        #                 checked[topic][property][0] = value
        #             else:
        #                 checked[topic][property][0] = float(value)



    def update_ImageView(self):
        d2 = self.roi.getArrayRegion(self.imv1data, self.imv1.imageItem, axes=(1,2))
        vox = state['General']['Voxels (^3)'][0]
        self.imv2.setImage(d2.T, pos = [-vox/2,-vox/2])

        d2 = self.roi.getArrayRegion(self.imv3data, self.imv3.imageItem, axes=(1,2))
        self.imv4.setImage(np.transpose(d2, axes = (1,0,2)), pos = [-vox/2,-vox/2])


    def parameter_changed(self, param, changes):
        global state
        for param, change, data in changes:
            parent_key = param.parent().name()
            self_key = param.name()

            if isinstance(state[parent_key][self_key], dict):
                state[parent_key][self_key]['selected'] = data
            state[parent_key][self_key][0] = data



    # def update_3Dview(self):
    #
    #     self.glvw1.removeItem(self.v)
    #     self.glvw1.removeItem(self.gl1_c)
    #     self.gl1_c = self._create_cylinder_line_segments(sim_state['values'][0][2]/2, 100)
    #     self.glvw1.addItem(self.gl1_c)
    #     self.glvw2.removeItem(self.gl2_c)
    #     self.gl2_c = self._create_cylinder_line_segments(float(sim_state['values'][4][0]), 100)
    #     self.glvw2.addItem(self.gl2_c)
    #
    #     h1 = sim.r["Probe abs."].transpose(1, 2, 0)
    #     h2 = sim.r["Pump abs."].transpose(1, 2, 0)
    #     data = sim.r["Weighted excited state fraction"].transpose(1, 2, 0)
    #     d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
    #     d2[..., 0] = h1 * (255./h1.max())
    #     d2[..., 1] = data * (255./data.max())
    #     d2[..., 2] = h2 * (255./h2.max())
    #     d2[..., 3] = d2[..., 0]*0.3 + d2[..., 1]*0.3 + d2[..., 2]*0.3
    #     d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255
    #     self.v = gl.GLVolumeItem(d2, smooth=False, sliceDensity=5)
    #     voxels = int(sim_state['values'][0][2])
    #     self.v.translate(-voxels/2,-voxels/2,-voxels/2)
    #     self.glvw1.addItem(self.v)
    #     ax = gl.GLAxisItem()
    #     self.glvw1.addItem(ax)
    #
    #     for line in self.lines:
    #         self.glvw2.removeItem(line)
    #     self.lines.clear()
    #
    #     for ray_source,c in zip(["Probe", "Pump"], ['r', 'b']):
    #         z0,y0,x0,_ = list(zip(*sim.r[ray_source+' r0']))
    #         z1,y1,x1,_ = list(zip(*sim.r[ray_source+' r1']))
    #         z2,y2,x2,_ = list(zip(*sim.r[ray_source+' r2']))
    #         r0 = list(zip(*[x0,y0,z0]))
    #         r1 = list(zip(*[x1,y1,z1]))
    #         r2 = list(zip(*[x2,y2,z2]))
    #         self.gl2data = np.array(list(zip(r0,r1,r2)))
    #         for pt in self.gl2data:
    #             plt = gl.GLLinePlotItem(pos=pt, color = pg.glColor(*c), antialias=True)
    #             self.lines.append(plt)
    #             self.glvw2.addItem(plt)


    # def close_value_editor(self, item, col, input):
    #     global sim_state
    #     self.treeWidget.setItemWidget(item,col,None)
    #     topic_ind = sim_state['topics'].index(item.parent().text(0))
    #     prop_ind = sim_state['properties'][topic_ind].index(item.text(0))
    #     sim_state['values'][topic_ind][prop_ind] = input
    #     item.setText(1, input)
    #
    # def open_value_editor(self, item, col):
    #     if col == 1 and item not in self.top_items:
    #         index = self.treeWidget.indexFromItem(item)
    #         rowHeight = self.treeWidget.rowHeight(index)
    #         edit = QtGui.QLineEdit()
    #         edit.setText(item.text(1))
    #         edit.setMaximumHeight(rowHeight)
    #         edit.editingFinished.connect(lambda : self.close_value_editor(item, col, edit.text()))
    #         self.treeWidget.setItemWidget(item,col,edit)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F5:
            self.run_simulation()
        event.accept()


    def run_simulation(self):
        global state
        state = check_sim_state(state)
        sim.run(state, self.update_progress_bar)

        self.imv1data = sim.r['arrays']["Weighted excited state fraction"]
        vox = state['General']['Voxels (^3)'][0]

        for guide_t in self.guides_top:
            self.imv1.removeItem(guide_t)
        for guide_t in self.guides_top2:
            self.imv3.removeItem(guide_t)
        for guide_s in self.guides_side:
            self.imv2.removeItem(guide_s)
        for guide_s in self.guides_side2:
            self.imv4.removeItem(guide_s)

        self.guides_top = self.draw_guides_top(self.imv1)
        self.guides_top2 = self.draw_guides_top(self.imv3)
        self.guides_side = self.draw_guides_side(self.imv2)
        self.guides_side2 = self.draw_guides_side(self.imv4)

        self.imv1.setImage(self.imv1data, pos = [-vox/2,-vox/2])

        self.imv1.setCurrentIndex(int(vox/2))
        self.imv1.autoRange()

        self.imv3data = self.get_composite_image()
        self.imv3.setImage(self.imv3data, pos = [-vox/2,-vox/2])
        self.imv3.setCurrentIndex(int(vox/2))
        self.imv3.autoRange()
        self.update_ImageView()


        # self.update_3Dview()
        self.update_result_values()
        self.anim.start()


    def update_progress_bar(self, progress):
        self.progressBar.setProperty("value", progress)
        app.processEvents()




        # for topic in sim_state.keys():
        #     for property in sim_state[topic].keys():
        #
        #
        #         value = checked['values'][ind_topic][ind_prop]
        #
        #
        #
        # for ind_topic, topic in enumerate(sim_state['topics']):
        #     for ind_prop, property in enumerate(sim_state['properties'][ind_topic]):
        #         value = checked['values'][ind_topic][ind_prop]
        #         if property in ints:
        #             checked['values'][ind_topic][ind_prop] = int(value)
        #         elif property in arrays:
        #             pattern = r'([-+]?\d*\.\d+|[-+]?\d+)'
        #             numbers = re.findall(pattern, str(value))
        #             value = np.array(list([float(n) for n in numbers]))
        #             checked['values'][ind_topic][ind_prop] = value
        #         else:
        #             checked['values'][ind_topic][ind_prop] = float(value)


    # def _create_cylinder_line_segments(self, radius, points):
    #     z = np.array([0]*points)
    #     t = np.linspace(0,2*np.pi, points)
    #     x = radius * np.cos(t)
    #     y = radius * np.sin(t)
    #     pts = np.array(list(zip(x,y,z)))
    #     ls = gl.GLLinePlotItem(pos=pts, color = pg.glColor('g'), antialias=True)
    #     return ls
    #
    # def _create_circle_line_segments_2d(self, radius, points):
    #
    #     return ls




try:
    filename = sys.argv[1]
    state = read_state_file(filename)
except:
    state = init_state()
    Log.error("No state file was found or reading failed. Using internal default values.")

results = init_results()
sim = Simulation()
app = QtWidgets.QApplication([])
win = FES_UI()
win.setWindowTitle('Fractional excited state tool (GUI)');

win.show()
app.exec_()
