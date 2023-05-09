# Python application with a QT GUI (example project)

This application makes use of the PyQT5 framework via the PyQtGraph package. To run this application, you will need to have PyQtGraph installed in your Python enviroment. 
With Anaconda this can be easily done like this:

```
conda install -c anaconda pyqtgraph
```

This specific example is a very simple HDF5 file inspector. The functionality includes inspection of the group & dataset structure and visualization of 1D and 2D data.

## Quickstart
- *.ui files: Edit this files to your liking with the QT Designer program. This files provide the layout of the GUI. QT Designer is not officially available anymore, 
but a kind individual is still providing binaries for windows. A quick google search will help. 

- Implementation: Here, the *.ui files are dynamically loaded during runtime via the uic module. A lot of documentation for PyQT5 
is available on the riverbank computing sites. For uic specifically, you may find help here: https://www.riverbankcomputing.com/static/Docs/PyQt5/api/uic/uic-module.html?highlight=uic#uic
