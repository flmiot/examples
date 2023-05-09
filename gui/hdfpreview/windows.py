"""
HDFpreview GUI implementation module. Implements the MainWindow and AboutDialog
widgets.
"""

import os
from pyqtgraph.Qt import QtGui, QtCore, uic
from hdfpreview.logic import Dataset
import hdfpreview

class MainWindow(QtGui.QMainWindow):

    """
    Derives from Ui_MainWindow and implements the HDFpreview MainWindow
    widget with the four main program functionalities.
    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)

        # --------- UIC method ---------
        # Use the uic module to dynamically load the *.ui (QT user interface) files
        # We do not want a hard-coded path here, so we derive the location of the *.ui files
        # relative to the location of this file
        dirname = os.path.dirname(__file__)
        uic.loadUi(os.path.join(dirname, 'ui/mainWindow.ui'), self)

        # --------- pyuic5 method ---------
        # There is another way of using the QT ui files in Python. See e.g.
        # https://stackoverflow.com/questions/52471705/why-in-pyqt5-should-i-use-pyuic5-and-not-uic-loaduimy-ui
        # I choose here to use the UIC method, because I do not have to retranslate the UI files
        # everytime I change something in the GUI. If speed and user experience is critical (and if
        # the GUI is mature enough) pyuic5 could be a better option.

        # self.setupUi(self)
        self.show()
        self.aboutDialog = AboutDialog(self)


    @QtCore.pyqtSlot()
    def on_actionExit_triggered(self, *args, **kwargs):
        """Called on exit."""
        hdfpreview.app.quit()


    @QtCore.pyqtSlot()
    def on_actionPreviewDataSource_triggered(self, *args, **kwargs):
        """Preview the currently selected data source."""
        try:
            currentItem = self.treeWidget.currentItem()
            path = currentItem.toolTip(0)
            data = hdfpreview.data.getData(path)
            self.central_plot.displayData(data, path)
        except:
            self.statusbar.showMessage("Error: No valid data source selected!")


    @QtCore.pyqtSlot()
    def on_actionLoadFiles_triggered(self, *args, **kwargs):
        """
        Spawn a file dialog to let you select a number of HDF5 files. Create a
        new hdfpreview.Dataset with the selected files and populate the filelist
        and datasource tree widgets.
        """

        # Read HDF5 filenames
        files = QtGui.QFileDialog.getOpenFileNames(self, 'Select HDF5 files')[0]
        hdfpreview.data = Dataset(sorted(files))

        # Update the file QListWidget
        self.listWidget.clear()
        self.listWidget.addItems(hdfpreview.data.get_filenames())

        # Update the data source QTreeWidget
        self.treeWidget.clear()
        data_sources = hdfpreview.data.get_data_tree()
        self.fillTreeWidget(data_sources, None)


    @QtCore.pyqtSlot()
    def on_actionAbout_Hdf5_preview_triggered(self, *args, **kwargs):
        """Spawn the About HDFpreview dialog."""
        if self.aboutDialog.isHidden():
            self.aboutDialog.show()


    def fillTreeWidget(self, dictionary, rootItem = None):
        """
        Helper function to populate the datasource tree widget. Specifiy a
        nested *dictionary* with datasources. If rootItem is *None*, this method
        will automatically create a new root item in the tree widget for you,
        which will act as root for all entries in *dictionary*. Call this method
        recursivly with rootItem = ..., to create a nested treeItem structure.
        """

        if rootItem is None:
            rootItem = QtGui.QTreeWidgetItem(self.treeWidget)
            rootItem.setText(0, "Select a data source...")
            self.fillTreeWidget(dictionary, rootItem)
        else:
            for key, value in dictionary.items():
                item = QtGui.QTreeWidgetItem(rootItem)
                item.setText(0, key)
                if isinstance(value, dict):
                    self.fillTreeWidget(value, item)
                else:
                    item.setToolTip(0, value)


class AboutDialog(QtGui.QDialog):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        dirname = os.path.dirname(__file__)
        uic.loadUi(os.path.join(dirname, 'ui/about.ui'), self)
