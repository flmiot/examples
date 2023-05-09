"""
HDFpreview plotting module
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

class previewPlot(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        """
        HDFpreview plotting widget. Will display 1D, 2D and 3D shaped data in
        the appropiate pyqtgraph widget.
        """
        super(self.__class__, self).__init__(*args, **kwargs)
        self.setupUi()

    def setupUi(self):
        layout = QtGui.QVBoxLayout()
        self.imageView = pg.ImageView()
        self.plotWidget = pg.PlotWidget()
        self.imageView.hide()
        self.plotWidget.hide()
        self.plotWidget.getPlotItem().addLegend()
        layout.addWidget(self.imageView)
        layout.addWidget(self.plotWidget)
        self.setLayout(layout)


    def displayData(self, data, name):
        """
        Specifiy *data* and *name* to display a dataset with legend entries.
        This widget will try to decide the appropiate plotting widget and
        display either a pyqtgraph.ImageView or pyqtgraph.PlotWidget.
        """

        shape = data.shape
        if len(shape) == 1:
            pi = self.plotWidget.getPlotItem()
            items = pi.listDataItems()
            for item in items:
                pi.legend.removeItem(item.name())
                pi.removeItem(item)
            self.imageView.hide()
            self.plotWidget.show()
            self.plotWidget.enableAutoRange()
            self.plotWidget.plot(data,
                name = name,
                symbolSize=5,
                symbolBrush=(255,0,0),
                symbolPen='w')

        else:
            self.imageView.show()
            self.plotWidget.hide()
            self.imageView.setImage(data)
